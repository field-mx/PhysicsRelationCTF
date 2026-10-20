"""Length-generalized first-round TEP physics, without padding or new physics."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

PARENT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PARENT))
from tep_model import TEPPhysicsCTF, OriginalCTF, config, copy_ctf
from tep_model import ConditionalRelationAttention, valid_center, feature_lag_score
from tep_physics import equation_bank


def configuration(length,horizon,patch,lr=3e-4):
    if length<=0 or horizon<=0 or patch<=0 or length%patch:
        raise ValueError('Only complete real-data patches are allowed; no padding or truncation')
    cfg=config(52,lr=lr)
    cfg.update(seq_len=length,pred_len=horizon,patch_lens=[patch]*52,batch_size=64)
    return SimpleNamespace(**cfg)


class LengthPhysicsCTF(TEPPhysicsCTF):
    def __init__(self,cfg,graph,scalers,beta=.05,gamma=.75,rho=.001):
        # Reuse the native CTF, not the old adapter's fixed 96/24 constructor.
        OriginalCTF.__init__(self,cfg)
        self.length=cfg.seq_len;self.patch=cfg.patch_lens[0]
        if self.length%self.patch or any(p!=self.patch for p in cfg.patch_lens):
            raise ValueError('Uniform, exactly divisible patches required')
        self.graph=copy.deepcopy(graph);self.level_mode='absolute';self.use_confidence=True
        self.closure_init=0.;self.physical_enabled=True;self.lag_mode='physical'
        for name in ['mean','scale']:
            self.register_buffer('raw_'+name,torch.tensor(scalers[name],dtype=torch.float32))
        self.register_buffer('gain',torch.tensor(scalers['gains'],dtype=torch.float32))
        edges=graph['edges'];c=len(graph['channels'])
        for name,values in [('src',[e['source'] for e in edges]),('dst',[e['target'] for e in edges]),
                            ('rid',[e['relation'] for e in edges]),('feature_idx',[c+e['feature'] for e in edges])]:
            self.register_buffer(name,torch.tensor(values,dtype=torch.long))
        lags=sorted({v for e in edges for v in e['lags']})
        indices=self.length-1-torch.tensor(lags)[:,None]-torch.arange(self.patch-1,-1,-1)[None]
        if int(indices.min())<0:raise ValueError('History too short for the original physical lag set')
        self.register_buffer('lags',torch.tensor(lags,dtype=torch.float32))
        self.register_buffer('lag_forbidden',torch.tensor([[v not in e['lags'] for v in lags] for e in edges]))
        self.register_buffer('history_index',indices)
        r=len(graph['relations']);d=cfg.d_model;rank=8
        self.lag_log_temperature=nn.Parameter(torch.full((r,),-2.))
        self.adapter_in=nn.Parameter(torch.randn(r,2*d,rank)*.01)
        self.adapter_out=nn.Parameter(torch.zeros(r,rank,d))
        self.bias_net=nn.Sequential(nn.Linear(6,8),nn.GELU(),nn.Linear(8,1))
        nn.init.zeros_(self.bias_net[-1].weight);nn.init.zeros_(self.bias_net[-1].bias)
        self.relation_bias=nn.Parameter(torch.zeros(r))
        self.residual_net=nn.Sequential(nn.Linear(2*d,8),nn.GELU(),nn.Linear(8,d),nn.Tanh())
        for layer in self.encoder.layers:
            layer.self_attention.inner_attention=ConditionalRelationAttention(layer.self_attention.inner_attention,beta,gamma,rho,.5)
        self.channel_wise_patch_embedding.register_forward_hook(self._prepare_messages)
        self.register_buffer('equation_targets',torch.tensor([f['target'] for f in graph['features']]))
        self.raw_context=None;self.last_lag=None;self.latest_confidence=None

    def forecast_multi(self,x_enc,x_mark_enc=None,x_dec=None,x_mark_dec=None):
        if x_enc.shape[1]!=self.length:raise ValueError('Unexpected observed history length')
        self.raw_context=x_enc*self.raw_scale+self.raw_mean
        return OriginalCTF.forecast_multi(self,x_enc,x_mark_enc,x_dec,x_mark_dec)

    def _prepare_messages(self,embedding,args,output):
        if all(l.self_attention.inner_attention.override==(0.,0.,0.) for l in self.encoder.layers):return
        tokens,counts,_=output;raw=self.raw_context;b=raw.shape[0];d=tokens.shape[-1];p=self.patch
        mean=raw.mean(1);std=(raw.var(1,unbiased=False)+1e-5*self.raw_scale.square()).sqrt()
        normal=(raw-mean[:,None])/std[:,None]
        source=normal[:,:,self.src].transpose(1,2)[:,:,self.history_index]
        query=normal[:,-p:,self.dst].transpose(1,2)
        eq,valid=equation_bank(raw,self.graph)
        feature=self.feature_idx-len(self.graph['channels'])
        history=eq[:,:,feature].transpose(1,2)[:,:,self.history_index]
        validity=valid[:,:,feature].transpose(1,2)[:,:,self.history_index]
        score=feature_lag_score(query,history,validity,self.lag_log_temperature.exp()[self.rid])
        pi=score.masked_fill(self.lag_forbidden[None],-torch.inf).softmax(-1)
        self.last_lag=pi.detach();aligned=(pi[...,None]*source).sum(2)
        gate=(pi*validity.float().mean(-1)).sum(-1)
        eqpatch=((pi[...,None]*(history-mean[:,self.dst,None,None])*validity).sum(2)/std[:,self.dst,None]).clamp(-8,8)
        residual=(eq[:,-p:]-raw[:,-p:,self.equation_targets])/self.raw_scale[self.equation_targets]
        err=(residual.square()*valid[:,-p:]).sum(1)/valid[:,-p:].sum(1).clamp_min(1)
        confidence=1/(1+err);self.latest_confidence=confidence
        gate=gate*confidence[:,feature]
        projection=embedding.embedding_dict[f'pl{p}']
        physical_base=F.linear(eqpatch,projection.weight,None);sl=F.linear(aligned,projection.weight,None)
        hidden=F.gelu(torch.einsum('bed,edr->ber',torch.cat([physical_base,sl],-1),self.adapter_in[self.rid]))
        physics=(physical_base+torch.einsum('ber,erd->bed',hidden,self.adapter_out[self.rid]))*gate[...,None]
        g=self.num_global_tokens;nlocal=self.length//p
        if any(n!=nlocal for n in counts):raise RuntimeError('Unexpected patch dropping or count')
        local=tokens.reshape(b,len(self.graph['channels']),nlocal+g,d)[:,:,:nlocal].mean(2)
        correction=self.residual_net(torch.cat([local[:,self.dst],local[:,self.src]],-1))*gate[...,None]
        # Keep first-round feature scaling; lag values remain native 3-minute samples.
        expected=(pi*self.lags).sum(-1)/72.
        inputs=torch.stack([aligned.mean(-1),aligned[:,:,-1]-aligned[:,:,0],query.mean(-1),
            query[:,:,-1]-query[:,:,0],eqpatch.mean(-1),expected],-1)
        bias=(1+.5*torch.tanh(self.bias_net(inputs).squeeze(-1)+self.relation_bias[self.rid]))*gate
        starts=torch.arange(len(self.graph['channels']),device=tokens.device)*(nlocal+g)+nlocal
        slots=torch.arange(g,device=tokens.device)
        qs=(starts[self.dst,None]+slots).flatten();ks=(starts[self.src,None]+slots).flatten()
        ei=torch.arange(len(self.src),device=tokens.device).repeat_interleave(g)
        select=F.one_hot(qs,num_classes=tokens.shape[1]).to(tokens.dtype)
        diag={}
        if any(l.self_attention.inner_attention.record for l in self.encoder.layers):
            diag=dict(equation_confidence=float(confidence.detach().mean()),lag_samples=float((expected*72).detach().mean()),
                equation_confidence_per_feature=confidence.detach().mean(0).tolist(),physical_patch_norm=float(eqpatch.detach().norm(dim=-1).mean()))
        for layer in self.encoder.layers:
            a=layer.self_attention
            a.inner_attention.context=dict(qs=qs,ks=ks,ei=ei,physics=F.linear(physics,a.value_projection.weight,None),
                residual=F.linear(correction,a.value_projection.weight,None),bias=bias,target_select=select,lag_diagnostics=diag)


def pair(length,horizon,patch,graph,scalers,lr=3e-4):
    torch.manual_seed(42)
    cfg=configuration(length,horizon,patch,lr)
    baseline=OriginalCTF(cfg);physics=LengthPhysicsCTF(cfg,graph,scalers)
    copy_ctf(physics,baseline)
    return baseline,physics
