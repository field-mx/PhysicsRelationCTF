"""Same RelationAttention; preserve levels only for documented control references."""
import torch
from torch.nn import functional as F
from public_model_v2 import PublicPhysicsCTF as PreviousPhysicsCTF,valid_center,feature_lag_score
from public_physics_v4 import equation_bank,boundary_patch


class PublicPhysicsCTF(PreviousPhysicsCTF):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.register_buffer('boundary_edge',torch.tensor([e.get('absolute_reference',False) for e in self.graph['edges']]))

    def _prepare_messages(self,embedding,args,output):
        if all(l.self_attention.inner_attention.override==(0.,0.,0.) for l in self.encoder.layers):return
        tokens,counts,_=output;raw=self.raw_context;b=raw.shape[0];d=tokens.shape[-1]
        mean=raw.mean(1);std=(raw.var(1,unbiased=False)+1e-5*self.raw_scale.square()).sqrt()
        normal=(raw-mean[:,None])/std[:,None]
        source=normal[:,:,self.src].transpose(1,2)[:,:,self.history_index]
        query=normal[:,-24:,self.dst].transpose(1,2)
        eq,valid=equation_bank(raw,self.graph)
        bank=torch.cat([raw,eq],-1);vb=torch.cat([torch.ones_like(raw,dtype=torch.bool),valid],-1)
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
        reference=boundary_patch(history,validity,pi,mean[:,self.dst],std[:,self.dst])
        eqpatch=torch.where(self.boundary_edge[None,:,None],reference,eqpatch).clamp(-8,8)
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
                  'lag_entropy':float((-(pi.clamp_min(1e-12).log()*pi).sum(-1)).detach().mean()),
                  'boundary_message_patch_norm':float(eqpatch[:,self.boundary_edge].detach().norm(dim=-1).mean()) if self.boundary_edge.any() else 0.}
        for layer in self.encoder.layers:
            a=layer.self_attention
            a.inner_attention.context=dict(qs=qs,ks=ks,ei=ei,
                 physics=F.linear(physics,a.value_projection.weight,None),
                 residual=F.linear(residual,a.value_projection.weight,None),bias=bias,
                 target_select=select,lag_diagnostics=diag)
