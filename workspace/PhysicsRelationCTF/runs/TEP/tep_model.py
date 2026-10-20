"""V7 conditional attention messages with an explicitly TEP equation bank."""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
CORE=HERE.parents[1]
PUBLIC=CORE/'runs/publicdata_exp'
for p in [CORE,PUBLIC,PUBLIC/'optimization_v2']:
    sys.path.insert(0,str(p))
from physics import OriginalCTF,config
from public_model_v2 import PublicPhysicsCTF as AdapterBase, valid_center, feature_lag_score
from model_v5 import ConditionalRelationAttention
from relation_model import copy_ctf
from tep_physics import equation_bank


class TEPPhysicsCTF(AdapterBase):
    def __init__(self,cfg,graph,scalers,beta=.05,gamma=.75,rho=.001,
                 message_cap=.5,level_mode='absolute',closure=0.,confidence=True,
                 lag_mode='physical',**kwargs):
        super().__init__(cfg,graph,scalers,beta=beta,gamma=gamma,rho=rho,lag_mode=lag_mode)
        self.level_mode=level_mode;self.use_confidence=confidence
        self.closure_init=float(closure);self.physical_enabled=True
        self.register_buffer('equation_targets',torch.tensor([f['target'] for f in graph['features']]))
        if closure>0:
            self.closure_logit=nn.Parameter(torch.full((len(graph['features']),),float(torch.logit(torch.tensor(closure)))))
        for layer in self.encoder.layers:
            layer.self_attention.inner_attention=ConditionalRelationAttention(
                layer.self_attention.inner_attention.original,beta,gamma,rho,message_cap)
        self.latest_confidence=None

    def _prepare_messages(self,embedding,args,output):
        if all(l.self_attention.inner_attention.override==(0.,0.,0.) for l in self.encoder.layers):return
        tokens,counts,_=output;raw=self.raw_context;b=raw.shape[0];d=tokens.shape[-1]
        mean=raw.mean(1);std=(raw.var(1,unbiased=False)+1e-5*self.raw_scale.square()).sqrt()
        normal=(raw-mean[:,None])/std[:,None]
        source=normal[:,:,self.src].transpose(1,2)[:,:,self.history_index]
        query=normal[:,-24:,self.dst].transpose(1,2)
        eq,valid=equation_bank(raw,self.graph)
        history=eq[:,:,self.feature_idx-len(self.graph['channels'])].transpose(1,2)[:,:,self.history_index]
        validity=valid[:,:,self.feature_idx-len(self.graph['channels'])].transpose(1,2)[:,:,self.history_index]
        score=feature_lag_score(query,history,validity,self.lag_log_temperature.exp()[self.rid])
        if self.lag_mode=='uniform':score=score*0
        pi=score.masked_fill(self.lag_forbidden[None],-torch.inf).softmax(-1)
        self.last_lag=pi.detach();aligned=(pi[...,None]*source).sum(2)
        gate=(pi*validity.float().mean(-1)).sum(-1)
        centered=(pi[...,None]*valid_center(history,validity)).sum(2)/std[:,self.dst,None]
        absolute=(pi[...,None]*(history-mean[:,self.dst,None,None])*validity).sum(2)/std[:,self.dst,None]
        eqpatch=absolute if self.level_mode=='absolute' else centered
        eqpatch=eqpatch.clamp(-8,8)
        # V7 equation credibility: evaluated on observed history in target units.
        residual=(eq[:,-24:]-raw[:,-24:,self.equation_targets])/self.raw_scale[self.equation_targets]
        err=(residual.square()*valid[:,-24:]).sum(1)/valid[:,-24:].sum(1).clamp_min(1)
        confidence=1/(1+err)
        if not self.use_confidence:confidence=torch.ones_like(confidence)
        self.latest_confidence=confidence
        edge_conf=confidence[:,self.feature_idx-len(self.graph['channels'])]
        gate=gate*edge_conf
        projection=embedding.embedding_dict['pl24']
        physical_base=F.linear(eqpatch,projection.weight,None)
        sl=F.linear(aligned,projection.weight,None)
        hidden=F.gelu(torch.einsum('bed,edr->ber',torch.cat([physical_base,sl],-1),self.adapter_in[self.rid]))
        physics=(physical_base+torch.einsum('ber,erd->bed',hidden,self.adapter_out[self.rid]))*gate[...,None]
        local=tokens.reshape(b,len(self.graph['channels']),6,d)[:,:,:4].mean(2)
        correction=self.residual_net(torch.cat([local[:,self.dst],local[:,self.src]],-1))*gate[...,None]
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
            diag=dict(equation_confidence=float(confidence.detach().mean()),
                lag_samples=float((expected*72).detach().mean()),
                equation_confidence_per_feature=confidence.detach().mean(0).tolist(),
                physical_patch_norm=float(eqpatch.detach().norm(dim=-1).mean()))
        for layer in self.encoder.layers:
            a=layer.self_attention
            a.inner_attention.context=dict(qs=qs,ks=ks,ei=ei,
                physics=F.linear(physics,a.value_projection.weight,None),
                residual=F.linear(correction,a.value_projection.weight,None),
                bias=bias,target_select=select,lag_diagnostics=diag)

    def forecast_multi(self,x_enc,x_mark_enc=None,x_dec=None,x_mark_dec=None):
        pred=super().forecast_multi(x_enc,x_mark_enc,x_dec,x_mark_dec)
        if self.closure_init<=0 or not self.physical_enabled:return pred
        if all(l.self_attention.inner_attention.override==(0.,0.,0.) for l in self.encoder.layers):return pred
        # Optional minimal physical head: evaluate f on PREDICTED future inputs.
        # The historical reliability gate bounds quasi-steady/noisy relations.
        raw=pred*self.raw_scale+self.raw_mean
        eq,valid=equation_bank(raw,self.graph)
        eqz=(eq-self.raw_mean[self.equation_targets])/self.raw_scale[self.equation_targets]
        weight=self.closure_logit.sigmoid()[None,None]*self.latest_confidence[:,None]*valid
        target=pred[:,:,self.equation_targets]
        adjusted=target+weight*(eqz-target).clamp(-4,4)
        return pred.index_copy(-1,self.equation_targets,adjusted)
