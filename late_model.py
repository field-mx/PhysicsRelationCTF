"""Restrict physical injection to the final native CTF attention layer.

No graph powers, mask changes, new head, or added parameters. Earlier layers
retain ordinary data attention. Finite -1000 logits yield exactly zero
float32 softplus/sigmoid strengths and survive state_dict serialization.
This deliberately gives up early-layer physical message multi-hop propagation.
"""
import torch
from relation_model import PhysicsRelationCTF

class LatePhysicsCTF(PhysicsRelationCTF):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for layer in self.encoder.layers[:-1]:
            attention=layer.self_attention.inner_attention
            for parameter in [attention.theta_beta,attention.theta_gamma,attention.theta_rho]:
                with torch.no_grad():parameter.fill_(-1000.)
                parameter.requires_grad_(False)
