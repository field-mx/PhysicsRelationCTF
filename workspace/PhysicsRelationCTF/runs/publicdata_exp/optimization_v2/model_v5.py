"""Condition known physics messages on physical neighbors, not all channels."""
import math
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from public_model_v4 import PublicPhysicsCTF as V4
from relation_model import RelationAttention


def physical_weights(edge_attention, active, target_select):
    """[B,H,E] conditional probabilities, exactly zero for inactive edges."""
    weighted = edge_attention * active[:, None].to(edge_attention.dtype)
    mass = torch.einsum('bhe,el->bhl', weighted, target_select)
    denominator = torch.einsum('bhl,el->bhe', mass, target_select)
    return weighted / denominator.clamp_min(1e-12)


class ConditionalRelationAttention(RelationAttention):
    def __init__(self, original, beta=.05, gamma=.1, rho=.001, message_cap=.5):
        super().__init__(original, beta, gamma, rho)
        self.message_cap = float(message_cap)

    def forward(self, q, k, v, attn_mask, tau=None, delta=None):
        if self.override == (0., 0., 0.):
            return self.original(q, k, v, attn_mask, tau, delta)
        c = self.context
        beta, gamma, rho = self.strengths()
        scale = self.original.scale or 1 / math.sqrt(q.shape[-1])
        scores = torch.einsum('blhe,bshe->bhls', q, k) * scale
        qs, ks, ei = c['qs'], c['ks'], c['ei']
        bias = beta * c['bias'][:, ei]
        active_scores = scores[:, :, qs, ks].detach() if self.record else None
        scores[:, :, qs, ks] += bias[:, None]
        probabilities = scores.masked_fill(attn_mask, -torch.inf).softmax(-1)
        attention = self.original.dropout(probabilities)
        out = torch.einsum('bhls,bshd->blhd', attention, v)
        vp = v[:, ks]
        pn = c['physics'][:, ei].reshape_as(vp)
        rn = c['residual'][:, ei].reshape_as(vp)
        vn = vp.detach().flatten(2).norm(dim=-1, keepdim=True).clamp_min(1e-6)
        norm = pn.flatten(2).norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pn = pn * torch.minimum(torch.ones_like(norm), 2 * vn / norm)[..., None]
        rn = F.normalize(rn.flatten(2), dim=-1).reshape_as(vp) * vn[..., None]
        # Validity is already included continuously in the physical message.
        # Zero-valid edges must not dilute the conditional-neighbor denominator.
        active = c['bias'][:, ei] > 0
        native_edges = probabilities[:, :, qs, ks]
        conditional = physical_weights(native_edges, active, c['target_select'])
        physics = torch.einsum('el,behd->blhd', c['target_select'],
                              conditional.permute(0, 2, 1)[..., None] * (gamma * pn))
        raw_norm = physics.detach().flatten(2).norm(dim=-1, keepdim=True)
        limit = self.message_cap * out.detach().flatten(2).norm(dim=-1, keepdim=True).clamp_min(1e-6)
        factor = (limit / raw_norm.clamp_min(1e-6)).clamp(max=1)
        physics = physics * factor[..., None]
        residual = torch.einsum('el,behd->blhd', c['target_select'],
                               attention[:, :, qs, ks].permute(0, 2, 1)[..., None] * (rho * rn))
        if self.record:
            target_mask = c['target_select'].sum(0) > 0
            pn_out = physics[:, target_mask].detach().flatten(2).norm(dim=-1)
            dn_out = out[:, target_mask].detach().flatten(2).norm(dim=-1)
            self.diagnostics = dict(
                beta=float(beta.detach()), gamma=float(gamma.detach()), rho=float(rho.detach()),
                qk_active_edge_std=float(active_scores.std(unbiased=False)),
                bias_active_edge_std=float(bias.detach().std(unbiased=False)),
                physical_attention_mass=float(native_edges.detach().mean()),
                conditional_edge_mass=float(conditional.detach().mean()),
                propagated_physics_norm=float(pn_out.mean()),
                native_target_output_norm=float(dn_out.mean()),
                physics_output_ratio=float((pn_out / dn_out.clamp_min(1e-6)).mean()),
                clipped_target_fraction=float((factor[:, target_mask] < .999).float().mean()),
                **c['lag_diagnostics'])
        return (out + physics + residual).contiguous(), attention if self.original.output_attention else None


class PublicPhysicsCTF(V4):
    def __init__(self, cfg, graph, scalers, beta=.05, gamma=.1, rho=.001,
                 representation='centered', lag_mode='physical', rank=8, message_cap=.5):
        super().__init__(cfg, graph, scalers, beta=beta, gamma=gamma, rho=rho,
                         representation=representation, lag_mode=lag_mode, rank=rank)
        for layer in self.encoder.layers:
            old = layer.self_attention.inner_attention
            layer.self_attention.inner_attention = ConditionalRelationAttention(
                old.original, beta, gamma, rho, message_cap)
