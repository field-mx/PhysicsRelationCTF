"""Dataset adapter: unchanged native CTF and existing RelationAttention core."""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from public_physics import RELATIONS
from public_physics_v2 import equation_bank


def valid_center(x,valid):
    weight=valid.to(x.dtype)
    # Subtract one valid reference first to avoid cancellation of large levels.
    ref=x.gather(-1,valid.to(torch.int64).argmax(-1,keepdim=True))
    delta=(x-ref)*weight
    mean=delta.sum(-1,keepdim=True)/weight.sum(-1,keepdim=True).clamp_min(1.)
    return (delta-mean)*weight


def feature_lag_score(query,history,valid,temperature):
    centered=valid_center(history,valid)
    scale=(centered.square().sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1)+1e-5).sqrt()
    aligned=centered/scale
    q=query-query.mean(-1,keepdim=True)
    q=q/(q.square().mean(-1,keepdim=True)+1e-5).sqrt()
    return -(((q[:,:,None]-aligned).square()*valid).sum(-1)/valid.sum(-1).clamp_min(1))*temperature[None,:,None]

PARENT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PARENT))
from physics import OriginalCTF, config
from relation_model import RelationAttention, copy_ctf, lag_scores


class PublicPhysicsCTF(OriginalCTF):
    def __init__(self,cfg,graph,scalers,beta=.05,gamma=.75,rho=.001,
                 representation='centered',lag_mode='physical',rank=8):
        super().__init__(cfg)
        if cfg.seq_len!=96 or any(p!=24 for p in cfg.patch_lens):raise ValueError('96/24 required')
        import copy
        self.graph=copy.deepcopy(graph);self.graph['kappa']=scalers.get('kappa',1.)
        self.representation=representation;self.lag_mode=lag_mode
        self.register_buffer('raw_mean',torch.tensor(scalers['mean'],dtype=torch.float32))
        self.register_buffer('raw_scale',torch.tensor(scalers['scale'],dtype=torch.float32))
        self.register_buffer('gain',torch.tensor(scalers['gains'],dtype=torch.float32))
        e=graph['edges'];c=len(graph['channels'])
        for name,values in [('src',[v['source'] for v in e]),('dst',[v['target'] for v in e]),
                            ('rid',[v['relation'] for v in e]),
                            ('feature_idx',[v['source'] if v['feature'] is None else c+v['feature'] for v in e])]:
            self.register_buffer(name,torch.tensor(values,dtype=torch.long))
        lags=sorted({l for v in e for l in v['lags']})
        self.register_buffer('lags',torch.tensor(lags,dtype=torch.float32))
        self.register_buffer('lag_forbidden',torch.tensor([[l not in v['lags'] for l in lags] for v in e]))
        self.register_buffer('history_index',95-torch.tensor(lags)[:,None]-torch.arange(23,-1,-1)[None])
        r=len(RELATIONS);d=cfg.d_model
        self.lag_log_temperature=nn.Parameter(torch.full((r,),-2.))
        self.adapter_in=nn.Parameter(torch.randn(r,2*d,rank)*.01)
        self.adapter_out=nn.Parameter(torch.zeros(r,rank,d))
        self.bias_net=nn.Sequential(nn.Linear(6,8),nn.GELU(),nn.Linear(8,1))
        nn.init.zeros_(self.bias_net[-1].weight);nn.init.zeros_(self.bias_net[-1].bias)
        self.relation_bias=nn.Parameter(torch.zeros(r))
        self.residual_net=nn.Sequential(nn.Linear(2*d,8),nn.GELU(),nn.Linear(8,d),nn.Tanh())
        for layer in self.encoder.layers:
            layer.self_attention.inner_attention=RelationAttention(layer.self_attention.inner_attention,beta,gamma,rho)
        self.channel_wise_patch_embedding.register_forward_hook(self._prepare_messages)
        self.raw_context=None;self.last_lag=None

    def set_strengths(self,beta=None,gamma=None,rho=None):
        for layer in self.encoder.layers:
            layer.self_attention.inner_attention.override=None if beta is None else (float(beta),float(gamma),float(rho))

    def forecast_multi(self,x_enc,x_mark_enc=None,x_dec=None,x_mark_dec=None):
        if x_enc.shape[1]!=96:raise ValueError('Only 96 historical inputs allowed')
        self.raw_context=x_enc*self.raw_scale+self.raw_mean
        return super().forecast_multi(x_enc,x_mark_enc,x_dec,x_mark_dec)

    def _prepare_messages(self,embedding,args,output):
        if all(l.self_attention.inner_attention.override==(0.,0.,0.) for l in self.encoder.layers):return
        tokens,counts,_=output;raw=self.raw_context;b=raw.shape[0];d=tokens.shape[-1]
        mean=raw.mean(1);std=(raw.var(1,unbiased=False)+1e-5*self.raw_scale.square()).sqrt()
        normal=(raw-mean[:,None])/std[:,None]
        source=normal[:,:,self.src].transpose(1,2)[:,:,self.history_index]
        query=normal[:,-24:,self.dst].transpose(1,2)
        eq,valid=equation_bank(raw,self.graph)
        bank=torch.cat([raw,eq],-1)
        vb=torch.cat([torch.ones_like(raw,dtype=torch.bool),valid],-1)
        history=bank[:,:,self.feature_idx].transpose(1,2)[:,:,self.history_index]
        validity=vb[:,:,self.feature_idx].transpose(1,2)[:,:,self.history_index].float()
        score=feature_lag_score(query,history*self.gain[None,:,None,None],validity.bool(),self.lag_log_temperature.exp()[self.rid])
        if self.lag_mode=='uniform':score=score*0
        pi=score.masked_fill(self.lag_forbidden[None],-torch.inf).softmax(-1)
        self.last_lag=pi.detach();aligned=(pi[...,None]*source).sum(2)
        gate=(pi*validity.mean(-1)).sum(-1)*vb[:,-1,self.feature_idx].float()
        if self.representation=='centered':
            eqpatch=(pi[...,None]*valid_center(history,validity.bool())).sum(2)*self.gain[None,:,None]/std[:,self.dst,None]
        elif self.representation=='anchored':
            variation=(history-history[...,-1:])*validity*validity[...,-1:]*self.gain[None,:,None,None]/std[:,self.dst,None,None]
            eqpatch=(pi[...,None]*variation).sum(2)+normal[:,-1,self.dst,None]
        else:raise ValueError(self.representation)
        eqpatch=eqpatch.clamp(-8,8)
        projection=embedding.embedding_dict['pl24']
        physical_base=F.linear(eqpatch,projection.weight,None)
        sl=F.linear(aligned,projection.weight,None)
        adapter=F.gelu(torch.einsum('bed,edr->ber',torch.cat([physical_base,sl],-1),self.adapter_in[self.rid]))
        physics=(physical_base+torch.einsum('ber,erd->bed',adapter,self.adapter_out[self.rid]))*gate[...,None]
        local=tokens.reshape(b,len(self.graph['channels']),6,d)[:,:,:4].mean(2)
        residual=self.residual_net(torch.cat([local[:,self.dst],local[:,self.src]],-1))*gate[...,None]
        expected=(pi*self.lags).sum(-1)/72.
        inputs=torch.stack([aligned.mean(-1),aligned[:,:,-1]-aligned[:,:,0],query.mean(-1),
                            query[:,:,-1]-query[:,:,0],eqpatch.mean(-1),expected],-1)
        bias=(1+.5*torch.tanh(self.bias_net(inputs).squeeze(-1)+self.relation_bias[self.rid]))*gate
        g=self.num_global_tokens;starts=torch.arange(len(self.graph['channels']),device=tokens.device)*6+4
        slots=torch.arange(g,device=tokens.device)
        qs=(starts[self.dst,None]+slots).flatten();ks=(starts[self.src,None]+slots).flatten()
        ei=torch.arange(len(self.src),device=tokens.device).repeat_interleave(g)
        select=F.one_hot(qs,num_classes=tokens.shape[1]).to(tokens.dtype)
        diag={}
        if any(l.self_attention.inner_attention.record for l in self.encoder.layers):
            diag={'lag_minutes':float((expected*72).detach().mean()),'valid_edge_fraction':float(gate.detach().mean()),
                  'lag_entropy':float((-(pi.clamp_min(1e-12).log()*pi).sum(-1)).detach().mean())}
        for layer in self.encoder.layers:
            a=layer.self_attention
            a.inner_attention.context=dict(qs=qs,ks=ks,ei=ei,
                 physics=F.linear(physics,a.value_projection.weight,None),
                 residual=F.linear(residual,a.value_projection.weight,None),bias=bias,
                 target_select=select,lag_diagnostics=diag)
