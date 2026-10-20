"""Channel-token CTF with optional channel-to-channel physical logit bias."""
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

CTF_ROOT = Path(__file__).resolve().parents[1] / 'CTF/ChannelTokenFormer'
if str(CTF_ROOT) not in sys.path:
    sys.path.insert(0, str(CTF_ROOT))
from models.ChannelTokenFormer import Model as RepositoryCTF


class OriginalCTF(RepositoryCTF):
    """Common baseline: upstream modules, true appended channel-token readout."""
    def __init__(self, config):
        super().__init__(config)
        # Cache the SAME original mask once, broadcasting batch/head axes.
        from types import MethodType
        for layer in self.encoder.layers:
            build = layer.build_attention_mask
            def cached(this,B,n_heads,n_vars,L,counts,num_global_tokens=1,device='cpu',_build=build):
                key=(str(device),tuple(counts),num_global_tokens)
                if getattr(this,'_cached_key',None)!=key:
                    this._cached_mask,this._cached_counts=_build(
                        1,1,n_vars,L,counts,num_global_tokens,device)
                    this._cached_key=key
                return this._cached_mask.expand(B,n_heads,-1,-1),this._cached_counts
            layer.build_attention_mask=MethodType(cached,layer)

    def forecast_multi(self,x_enc,x_mark_enc=None,x_dec=None,x_mark_dec=None):
        n=x_enc.shape[-1]
        if self.use_norm:
            means=[];stds=[]
            for c in range(n):
                factor=int(self.sampling_periods[c]/min(self.sampling_periods))
                sample=x_enc[:,::factor,c]
                means.append(sample.mean(1,keepdim=True).detach())
                stds.append((sample.var(1,keepdim=True,unbiased=False)+1e-5).sqrt())
            mean=torch.cat(means,1)[:,None];std=torch.cat(stds,1)[:,None]
            x_enc=(x_enc-mean)/std
        tokens,patches,n=self.channel_wise_patch_embedding(x_enc.transpose(1,2),self.keep_prob)
        encoded,counts=self.encoder(tokens,patches,n)
        self.latest_num_tokens_list=counts
        out=[];offset=0;g=self.num_global_tokens
        for p,count in zip(patches,counts):
            out.append(encoded[:,offset+p:offset+p+g])
            offset+=count
        channel=torch.stack(out,1)
        if g==1:channel=channel.squeeze(2)
        pred=self.head(channel,n,num_global_tokens=g)
        return pred*std+mean if self.use_norm else pred


class PhysicsAttention(nn.Module):
    def __init__(self, original, beta_init):
        super().__init__()
        self.original = original
        self.theta = nn.Parameter(torch.tensor(math.log(math.expm1(beta_init))))
        self.beta_override = None
        self.edge_weights = None
        self.query_slots = None
        self.key_slots = None
        self.edge_indices = None
        self.record = False
        self.diagnostics = {}

    @property
    def beta(self):
        if self.beta_override is not None:
            return self.theta.new_tensor(self.beta_override)
        return F.softplus(self.theta)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # Exact baseline execution, not merely asymptotic softplus(-large).
        if self.beta_override == 0:
            return self.original(queries, keys, values, attn_mask, tau, delta)
        if self.edge_weights is None:
            raise RuntimeError('Local embedding hook did not initialize edge weights')
        scale = self.original.scale or 1. / math.sqrt(queries.shape[-1])
        scores = torch.einsum('blhe,bshe->bhls', queries, keys) * scale
        weights = self.edge_weights[:, self.edge_indices]
        bias = self.beta * weights
        if self.record:
            active = scores[:, :, self.query_slots, self.key_slots]
            w = self.edge_weights.detach()
            self.diagnostics = {
                'beta': float(self.beta.detach()),
                'w_mean': float(w.mean()), 'w_std': float(w.std(unbiased=False)),
                'w_min': float(w.min()), 'w_max': float(w.max()),
                'qk_edge_std': float(active.detach().std(unbiased=False)),
                'bias_edge_std': float(bias.detach().std(unbiased=False)),
                'bias_max': float(bias.detach().max()),
                'qk_all_std': float(scores.detach().std(unbiased=False)),
                'bias_all_std': float(torch.sqrt((bias.detach().square().sum(1)/(scores.shape[-1]**2)-
                    (bias.detach().sum(1)/(scores.shape[-1]**2)).square()).clamp_min(0)).mean()),
                'saturated_fraction': float(((w<.01)|(w>.99)).float().mean()),
            } if w.numel() else {'beta': float(self.beta.detach()), 'edges': 0}
        scores[:, :, self.query_slots, self.key_slots] += bias[:, None, :]
        if self.original.mask_flag:
            scores = scores.masked_fill(attn_mask, -torch.inf)
        attn = self.original.dropout(scores.softmax(-1))
        output = torch.einsum('bhls,bshd->blhd', attn, values).contiguous()
        return output, attn if self.original.output_attention else None


class PhysicsCTF(OriginalCTF):
    def __init__(self, config, adjacency, mode='dynamic', beta_init=.05, hidden=32):
        if mode not in ('dynamic', 'fixed'):
            raise ValueError(mode)
        if beta_init <= 0:
            raise ValueError('Use set_beta(0) for exact disabling; initialization must be >0')
        super().__init__(config)
        adjacency = torch.as_tensor(adjacency, dtype=torch.bool)
        if adjacency.shape != (config.enc_in, config.enc_in):
            raise ValueError('Adjacency must have shape [channels, channels]')
        if adjacency.diagonal().any():
            raise ValueError('Physical graph must contain cross-channel direct edges only')
        self.register_buffer('A_phys', adjacency.clone())
        dst, src = adjacency.nonzero(as_tuple=True)
        self.register_buffer('edge_dst', dst)
        self.register_buffer('edge_src', src)
        self.mode = mode
        self.edge_net = nn.Sequential(
            nn.Linear(6 * config.d_model, hidden), nn.GELU(), nn.Linear(hidden, 1)
        ) if mode == 'dynamic' else None
        for layer in self.encoder.layers:
            layer.self_attention.inner_attention = PhysicsAttention(
                layer.self_attention.inner_attention, beta_init)
        self.channel_wise_patch_embedding.register_forward_hook(self._prepare_bias)
        self.latest_weights = None

    def set_beta(self, value=None):
        if value is not None and value < 0:
            raise ValueError('beta must be nonnegative')
        for layer in self.encoder.layers:
            layer.self_attention.inner_attention.beta_override = value

    def _prepare_bias(self, embedding, args, output):
        tokens, patch_counts, channels = output
        starts, local = [], []
        offset = 0
        for count in patch_counts:
            local.append(tokens[:, offset:offset + count].mean(1))
            starts.append(offset + count)
            offset += count + self.num_global_tokens
        z = torch.stack(local, 1)  # Local-only representations before any mixing.
        dst, src = self.edge_dst, self.edge_src
        if self.edge_net is None:
            weights = tokens.new_ones((tokens.shape[0], src.numel()))
        else:
            ids = torch.arange(channels, device=tokens.device)
            e = embedding.channel_embedding(ids)
            zi, zj = z[:, dst], z[:, src]
            ei = e[dst].unsqueeze(0).expand(tokens.shape[0], -1, -1)
            ej = e[src].unsqueeze(0).expand_as(ei)
            features = torch.cat((zi, zj, zi-zj, zi*zj, ei, ej), -1)
            weights = self.edge_net(features).squeeze(-1).sigmoid()
        starts = torch.tensor(starts, device=tokens.device)
        # Native mask connects only corresponding channel-token slots.
        g = torch.arange(self.num_global_tokens, device=tokens.device)
        qslots = (starts[dst, None] + g).flatten()
        kslots = (starts[src, None] + g).flatten()
        edge_ids = torch.arange(src.numel(), device=tokens.device).repeat_interleave(
            self.num_global_tokens)
        for layer in self.encoder.layers:
            inner = layer.self_attention.inner_attention
            inner.edge_weights = weights
            inner.query_slots, inner.key_slots = qslots, kslots
            inner.edge_indices = edge_ids
        self.latest_weights = weights.detach()


def load_native_weights(physics, original):
    """Copy every native CTF parameter exactly; extra physics parameters stay initialized."""
    missing, unexpected = physics.load_state_dict(original.state_dict(), strict=False)
    if unexpected or any(not (k.startswith('edge_net.') or k in (
            'A_phys', 'edge_dst', 'edge_src') or k.endswith('.theta')) for k in missing):
        raise RuntimeError((missing, unexpected))
