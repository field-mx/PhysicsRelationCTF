"""Suppress balance messages when their observed historical equation fails."""
import torch
from model_v6 import PublicPhysicsCTF as V6
from public_physics_v4 import equation_bank

class PublicPhysicsCTF(V6):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        applicable=[]
        for e in self.graph['edges']:
            fi=e['feature']
            op=self.graph['features'][fi]['op'] if fi is not None else ''
            applicable.append(op.startswith('balance_') or op=='mix')
        self.register_buffer('check_equation',torch.tensor(applicable,dtype=torch.bool))

    def _prepare_messages(self,embedding,args,output):
        if all(l.self_attention.inner_attention.override==(0.,0.,0.) for l in self.encoder.layers):return
        super()._prepare_messages(embedding,args,output)
        raw=self.raw_context[:,-24:]
        values,valid=equation_bank(raw,self.graph)
        bank=torch.cat([raw,values],-1)
        mask=torch.cat([torch.ones_like(raw,dtype=torch.bool),valid],-1)[:,:,self.feature_idx]
        # Equations produce the target's SI quantity. Standardization makes the
        # consistency residual unit invariant. All inputs are historical only.
        err=(bank[:,:,self.feature_idx]-raw[:,:,self.dst])/self.raw_scale[self.dst].clamp_min(1e-8)
        energy=(err.square()*mask).sum(1)/mask.sum(1).clamp_min(1)
        confidence=1/(1+energy)
        confidence=torch.where(self.check_equation[None],confidence,torch.ones_like(confidence))
        for layer in self.encoder.layers:
            inner=layer.self_attention.inner_attention;c=inner.context
            c['physics']=c['physics']*confidence[...,None]
            c['residual']=c['residual']*confidence[...,None]
            c['bias']=c['bias']*confidence
            if inner.record:
                c['lag_diagnostics']['equation_confidence']=float(confidence[:,self.check_equation].detach().mean())
