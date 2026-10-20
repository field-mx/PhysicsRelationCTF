"""Typed lag-aligned physical messages inside otherwise unchanged CTF attention."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from physics import OriginalCTF,RELATIONS,FEATURES,equation_features

def inv_softplus(x):return math.log(math.expm1(x))
def logit(p):return math.log(p/(1-p))

def lag_scores(query,source,gain,temperature):
    """Match inverse responses as well as positive ones; train-only gain sign.

    This is an empirical observation-mapping sign, not a balance coefficient.
    Unknown/zero gain leaves positive matching as an uninformative fallback.
    """
    sign=torch.where(gain<0,-torch.ones_like(gain),torch.ones_like(gain))
    aligned=source*sign[None,:,None,None]
    return -(query[:,:,None]-aligned).square().mean(-1)*temperature[None,:,None]

class RelationAttention(nn.Module):
    def __init__(self,original,beta=.05,gamma=.5,rho=.001):
        super().__init__();self.original=original
        self.theta_beta=nn.Parameter(torch.tensor(inv_softplus(beta)))
        self.theta_gamma=nn.Parameter(torch.tensor(logit(gamma/2)))
        self.theta_rho=nn.Parameter(torch.tensor(logit(rho/.02)))
        self.override=None;self.context=None;self.record=False;self.diagnostics={}
    def strengths(self):
        if self.override is not None:return tuple(self.theta_beta.new_tensor(v) for v in self.override)
        return F.softplus(self.theta_beta),2*self.theta_gamma.sigmoid(),.02*self.theta_rho.sigmoid()
    def forward(self,q,k,v,attn_mask,tau=None,delta=None):
        if self.override==(0.,0.,0.):return self.original(q,k,v,attn_mask,tau,delta)
        c=self.context;beta,gamma,rho=self.strengths();scale=self.original.scale or 1/math.sqrt(q.shape[-1])
        scores=torch.einsum('blhe,bshe->bhls',q,k)*scale
        qs,ks,ei=c['qs'],c['ks'],c['ei'];b=beta*c['bias'][:,ei]
        active=scores[:,:,qs,ks].detach() if self.record else None
        scores[:,:,qs,ks]+=b[:,None]
        scores=scores.masked_fill(attn_mask,-torch.inf)
        attention=self.original.dropout(scores.softmax(-1))
        out=torch.einsum('bhls,bshd->blhd',attention,v)
        vp=v[:,ks]
        pn=c['physics'][:,ei].reshape_as(vp);rn=c['residual'][:,ei].reshape_as(vp)
        # Cap an unscaled message to twice the native edge value norm, but do
        # not artificially inflate near-zero physics. Residual bound is strict.
        vn=vp.detach().flatten(2).norm(dim=-1,keepdim=True).clamp_min(1e-6)
        norm=pn.flatten(2).norm(dim=-1,keepdim=True).clamp_min(1e-6)
        pn=pn*torch.minimum(torch.ones_like(norm),2*vn/norm)[...,None]
        rn=F.normalize(rn.flatten(2),dim=-1).reshape_as(vp)*vn[...,None]
        physical=gamma*pn;residual=rho*rn
        ae=attention[:,:,qs,ks]
        edge_message=ae.permute(0,2,1)[...,None]*(physical+residual)
        out=out+torch.einsum('el,behd->blhd',c['target_select'],edge_message)
        if self.record:
            pnorm=physical.flatten(2).norm(dim=-1).mean();rnorm=residual.flatten(2).norm(dim=-1).mean()
            self.diagnostics={
                'beta':float(beta.detach()),'gamma':float(gamma.detach()),'rho':float(rho.detach()),
                'physics_norm':float(pnorm.detach()),'residual_norm':float(rnorm.detach()),
                'value_norm':float(vn.mean()),'physics_residual_ratio':float((pnorm/(rnorm+1e-12)).detach()),
                'physics_value_ratio':float((pnorm/vn.mean()).detach()),
                'qk_active_edge_std':float(active.std(unbiased=False)),
                'bias_active_edge_std':float(b.detach().std(unbiased=False)),
                'bias_active_edge_mean':float(b.detach().mean()),
                'physical_attention_mass':float(ae.detach().mean()),
                'propagated_physics_norm':float((ae.permute(0,2,1)[...,None]*physical).detach().flatten(2).norm(dim=-1).mean()),
                **c['lag_diagnostics']}
        return out.contiguous(),attention if self.original.output_attention else None

class PhysicsRelationCTF(OriginalCTF):
    def __init__(self,config,graph,scalers,beta=.05,gamma=.5,rho=.001,rank=8,
                 lag_mode='physical',representation='anchored'):
        super().__init__(config)
        if config.seq_len!=96 or any(p!=24 for p in config.patch_lens):
            raise ValueError('This version preserves the declared96/24 CTF setting')
        self.graph=graph;self.channels=graph['channels'];self.lag_mode=lag_mode;self.representation=representation
        self.register_buffer('raw_mean',torch.tensor(scalers['raw_mean'],dtype=torch.float32))
        self.register_buffer('raw_scale',torch.tensor(scalers['raw_scale'],dtype=torch.float32))
        self.register_buffer('eq_mean',torch.tensor(scalers['eq_mean'],dtype=torch.float32))
        self.register_buffer('eq_scale',torch.tensor(scalers['eq_scale'],dtype=torch.float32))
        self.register_buffer('gain',torch.tensor(scalers['gains'],dtype=torch.float32))
        edges=graph['edges'];src=[e['source'] for e in edges];dst=[e['target'] for e in edges]
        self.register_buffer('src',torch.tensor(src));self.register_buffer('dst',torch.tensor(dst))
        rid=[e['relation'] for e in edges];self.register_buffer('rid',torch.tensor(rid))
        feature=[len(self.channels)+FEATURES.index(e['feature']) if e['feature'] else e['source'] for e in edges]
        self.register_buffer('feature_idx',torch.tensor(feature))
        all_lags=sorted(set(t for e in edges for t in e['lags']))
        mask=torch.tensor([[t not in e['lags'] for t in all_lags] for e in edges])
        self.register_buffer('lag_forbidden',mask)
        self.register_buffer('lags',torch.tensor(all_lags,dtype=torch.float32))
        self.register_buffer('history_index',95-torch.tensor(all_lags)[:,None]-torch.arange(23,-1,-1)[None])
        # Temperature of content lag score shared by relation; initializes to
        # a modest shape-matching term, not independent parameters per edge.
        self.lag_log_temperature=nn.Parameter(torch.full((len(RELATIONS),),-2.))
        d=config.d_model;r=len(RELATIONS)
        self.adapter_in=nn.Parameter(torch.randn(r,2*d,rank)*.01)
        self.adapter_out=nn.Parameter(torch.zeros(r,rank,d))
        self.bias_net=nn.Sequential(nn.Linear(6,8),nn.GELU(),nn.Linear(8,1))
        nn.init.zeros_(self.bias_net[-1].weight);nn.init.zeros_(self.bias_net[-1].bias)
        self.relation_bias=nn.Parameter(torch.zeros(r))
        self.residual_net=nn.Sequential(nn.Linear(2*d,8),nn.GELU(),nn.Linear(8,d),nn.Tanh())
        for layer in self.encoder.layers:
            layer.self_attention.inner_attention=RelationAttention(layer.self_attention.inner_attention,beta,gamma,rho)
        self.channel_wise_patch_embedding.register_forward_hook(self._prepare_messages)
        self.last_lag=None;self.raw_context=None
    def set_strengths(self,beta=None,gamma=None,rho=None):
        values=None if beta is None else (float(beta),float(gamma),float(rho))
        for layer in self.encoder.layers:layer.self_attention.inner_attention.override=values
    def forecast_multi(self,x_enc,x_mark_enc=None,x_dec=None,x_mark_dec=None):
        if x_enc.shape[1]!=96:raise ValueError('Only the96 historical points may enter forward')
        self.raw_context=x_enc*self.raw_scale+self.raw_mean
        return super().forecast_multi(x_enc,x_mark_enc,x_dec,x_mark_dec)
    def _prepare_messages(self,embedding,args,output):
        if all(l.self_attention.inner_attention.override==(0.,0.,0.) for l in self.encoder.layers):return
        tokens,counts,_=output;raw=self.raw_context;batch=raw.shape[0];d=tokens.shape[-1]
        # Match native instance normalization in physical units, epsilon included.
        mean=raw.mean(1);std=(raw.var(1,unbiased=False)+1e-5*self.raw_scale.square()).sqrt()
        normal=(raw-mean[:,None])/std[:,None]
        zsource=normal[:,:,self.src].transpose(1,2)[:,:,self.history_index]
        query=normal[:,-24:,self.dst].transpose(1,2)
        score=lag_scores(query,zsource,self.gain,self.lag_log_temperature.exp()[self.rid])
        if self.lag_mode=='uniform':score=score*0
        pi=score.masked_fill(self.lag_forbidden[None],-torch.inf).softmax(-1)
        self.last_lag=pi.detach();aligned=(pi[...,None]*zsource).sum(2)
        eq=equation_features(raw,self.channels)
        bank=torch.cat([raw,eq],-1)
        histories=bank[:,:,self.feature_idx].transpose(1,2)[:,:,self.history_index]
        # Only variations are calibrated to target units, so unknown offsets and
        # operating-point shifts cannot replace the actual observed target level.
        variations=(histories-histories[...,-1:])*self.gain[None,:,None,None]/std[:,self.dst,None,None]
        if self.representation=='anchored':
            eqpatch=(pi[...,None]*variations).sum(2)+normal[:,-1,self.dst,None]
        elif self.representation=='centered':
            eqpatch=(pi[...,None]*(histories-histories.mean(-1,keepdim=True))).sum(2)*self.gain[None,:,None]/std[:,self.dst,None]
        else:raise ValueError(self.representation)
        eqpatch=eqpatch.clamp(-8,8)
        projection=embedding.embedding_dict['pl24']
        physical_base=F.linear(eqpatch,projection.weight,None)
        source_latent=F.linear(aligned,projection.weight,None)
        a=torch.cat([physical_base,source_latent],-1)
        adapter=F.gelu(torch.einsum('bed,edr->ber',a,self.adapter_in[self.rid]))
        physics=physical_base+torch.einsum('ber,erd->bed',adapter,self.adapter_out[self.rid])
        # Native local tokens stay untouched; pool solely for bounded residual.
        shaped=tokens.reshape(batch,len(self.channels),6,d)
        local=shaped[:,:,:4].mean(2)
        residual=self.residual_net(torch.cat([local[:,self.dst],local[:,self.src]],-1))
        expected=(pi*self.lags).sum(-1)/72
        bias_input=torch.stack([aligned.mean(-1),aligned[:,:,-1]-aligned[:,:,0],
                query.mean(-1),query[:,:,-1]-query[:,:,0],eqpatch.mean(-1),expected],-1)
        bias=1+.5*torch.tanh(self.bias_net(bias_input).squeeze(-1)+self.relation_bias[self.rid])
        g=self.num_global_tokens;starts=torch.arange(len(self.channels),device=tokens.device)*6+4
        slots=torch.arange(g,device=tokens.device)
        qs=(starts[self.dst,None]+slots).flatten();ks=(starts[self.src,None]+slots).flatten()
        ei=torch.arange(len(self.src),device=tokens.device).repeat_interleave(g)
        select=F.one_hot(qs,num_classes=tokens.shape[1]).to(tokens.dtype)
        record=any(l.self_attention.inner_attention.record for l in self.encoder.layers)
        lagdiag={}
        if record:
            for ri,name in enumerate(RELATIONS):
                sel=self.rid==ri
                if sel.any():lagdiag['lag_seconds_'+name]=float((expected[:,sel]*72).detach().mean())
            lagdiag['lag_entropy']=float((-(pi.clamp_min(1e-12).log()*pi).sum(-1)).detach().mean())
        for layer in self.encoder.layers:
            attention=layer.self_attention
            pp=F.linear(physics,attention.value_projection.weight,None)
            rr=F.linear(residual,attention.value_projection.weight,None)
            attention.inner_attention.context=dict(qs=qs,ks=ks,ei=ei,physics=pp,residual=rr,
                                                   bias=bias,target_select=select,lag_diagnostics=lagdiag)

def copy_ctf(model,baseline):
    state=model.state_dict();base=baseline.state_dict()
    for k,v in base.items():
        if k not in state or state[k].shape!=v.shape:raise ValueError(k)
        state[k]=v
    model.load_state_dict(state)
