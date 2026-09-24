"""Original channel-token CTF and Jensen-wake-guided attention variant."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


CTF_ROOT = Path(__file__).resolve().parents[2] / "CTF" / "ChannelTokenFormer"
if str(CTF_ROOT) not in sys.path:
    sys.path.insert(0, str(CTF_ROOT))
from models.ChannelTokenFormer import Model as RepositoryCTF


class OriginalCTF(RepositoryCTF):
    """Repository CTF with the requested true channel-token readout."""

    def __init__(self, config):
        super().__init__(config)
        # Exact-value cache: avoids reconstructing the 804x804 native mask at
        # every batch, without changing any allowed attention path.
        for layer in self.encoder.layers:
            build = layer.build_attention_mask

            def cached(this, batch, heads, channels, length, counts,
                       num_global_tokens=1, device="cpu", _build=build):
                key = (str(device), tuple(counts), int(num_global_tokens))
                if getattr(this, "_sdwpf_mask_key", None) != key:
                    base, token_counts = _build(
                        1, 1, channels, length, counts, num_global_tokens, device
                    )
                    this._sdwpf_mask = base
                    this._sdwpf_token_counts = token_counts
                    this._sdwpf_mask_key = key
                return (
                    this._sdwpf_mask.expand(batch, heads, -1, -1),
                    this._sdwpf_token_counts,
                )

            layer.build_attention_mask = MethodType(cached, layer)

    def forecast_multi(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        channels = x_enc.shape[-1]
        if self.use_norm:
            means, stds = [], []
            minimum = min(self.sampling_periods)
            for channel in range(channels):
                factor = int(self.sampling_periods[channel] / minimum)
                sample = x_enc[:, ::factor, channel]
                means.append(sample.mean(1, keepdim=True).detach())
                stds.append((sample.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt())
            mean = torch.cat(means, 1)[:, None]
            std = torch.cat(stds, 1)[:, None]
            x_enc = (x_enc - mean) / std
        tokens, patches, channels = self.channel_wise_patch_embedding(
            x_enc.transpose(1, 2), self.keep_prob
        )
        encoded, counts = self.encoder(tokens, patches, channels)
        self.latest_num_tokens_list = counts
        channel_tokens, offset = [], 0
        for patch_count, token_count in zip(patches, counts):
            channel_tokens.append(
                encoded[:, offset + patch_count : offset + patch_count + self.num_global_tokens]
            )
            offset += token_count
        channel_tokens = torch.stack(channel_tokens, 1)
        if self.num_global_tokens == 1:
            channel_tokens = channel_tokens.squeeze(2)
        prediction = self.head(
            channel_tokens, channels, num_global_tokens=self.num_global_tokens
        )
        return prediction * std + mean if self.use_norm else prediction


class WakePhysicsAttention(nn.Module):
    """Adds beta*Jensen-bias only to cross-channel channel-token logits."""

    def __init__(self, original: nn.Module, beta_init: float = 0.1):
        super().__init__()
        if beta_init <= 0:
            raise ValueError("beta_init must be positive; use set_beta(0) to disable")
        self.original = original
        self.theta = nn.Parameter(torch.tensor(math.log(math.expm1(beta_init))))
        self.beta_override: float | None = None
        self._wake_bias: torch.Tensor | None = None
        self._patch_counts: list[int] | None = None
        self._num_global_tokens: int | None = None
        self.record = False
        self.diagnostics: dict[str, float] = {}

    @property
    def beta(self) -> torch.Tensor:
        if self.beta_override is not None:
            return self.theta.new_tensor(self.beta_override)
        return F.softplus(self.theta)

    def set_wake_bias(
        self,
        wake_bias: torch.Tensor,
        patch_counts: list[int],
        num_global_tokens: int,
    ) -> None:
        if wake_bias.ndim != 3 or wake_bias.shape[1] != wake_bias.shape[2]:
            raise ValueError("wake_bias must have shape [batch, destination, source]")
        if wake_bias.shape[1] != len(patch_counts):
            raise ValueError("wake_bias channels do not match patch counts")
        self._wake_bias = wake_bias
        self._patch_counts = list(map(int, patch_counts))
        self._num_global_tokens = int(num_global_tokens)

    def dense_physics_bias(self, total_tokens: int) -> torch.Tensor:
        if self._wake_bias is None or self._patch_counts is None:
            raise RuntimeError("wake bias was not initialized for this batch")
        wake = self._wake_bias
        batch, channels, _ = wake.shape
        dense = wake.new_zeros((batch, total_tokens, total_tokens))
        offsets, cursor = [], 0
        for count in self._patch_counts:
            offsets.append(cursor + count)
            cursor += count + self._num_global_tokens
        if cursor != total_tokens:
            raise ValueError("token count is inconsistent with patch layout")
        slots = torch.as_tensor(offsets, device=wake.device)
        for slot in range(self._num_global_tokens):
            q = slots + slot
            k = slots + slot
            dense[:, q[:, None], k[None, :]] = wake
        return dense

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # This direct branch is the exact Original CTF computation.
        if self.beta_override == 0:
            return self.original(queries, keys, values, attn_mask, tau, delta)
        scale = self.original.scale or 1.0 / math.sqrt(queries.shape[-1])
        qk = torch.einsum("blhe,bshe->bhls", queries, keys) * scale
        physics = self.dense_physics_bias(qk.shape[-1]) * self.beta
        scores = qk + physics[:, None]
        if self.original.mask_flag:
            scores = scores.masked_fill(attn_mask, -torch.inf)
        attention = self.original.dropout(scores.softmax(-1))
        output = torch.einsum("bhls,bshd->blhd", attention, values).contiguous()

        if self.record:
            active = self._wake_bias != 0
            active_values = self._wake_bias[active]
            dense_active = physics != 0
            qk_active = qk[dense_active[:, None].expand_as(qk)]
            physics_active = physics[dense_active]
            self.diagnostics = {
                "beta": float(self.beta.detach()),
                "active_edges_per_sample": float(active.sum().detach() / active.shape[0]),
                "wake_mean_active": float(active_values.detach().mean()) if active_values.numel() else 0.0,
                "wake_std_active": float(active_values.detach().std(unbiased=False)) if active_values.numel() else 0.0,
                "qk_std": float(qk.detach().std(unbiased=False)),
                "physics_bias_std": float(physics.detach().std(unbiased=False)),
                "qk_active_std": float(qk_active.detach().std(unbiased=False)) if qk_active.numel() else 0.0,
                "physics_bias_active_std": float(physics_active.detach().std(unbiased=False)) if physics_active.numel() else 0.0,
                "physics_bias_active_mean": float(physics_active.detach().mean()) if physics_active.numel() else 0.0,
            }
        return output, attention if self.original.output_attention else None


class WakePhysicsCTF(OriginalCTF):
    """CTF whose only architectural change is a dynamic Jensen attention bias."""

    def __init__(self, config, beta_init: float = 0.1):
        super().__init__(config)
        for layer in self.encoder.layers:
            layer.self_attention.inner_attention = WakePhysicsAttention(
                layer.self_attention.inner_attention, beta_init=beta_init
            )

    def set_beta(self, value: float | None) -> None:
        if value is not None and value < 0:
            raise ValueError("beta must be nonnegative")
        for layer in self.encoder.layers:
            layer.self_attention.inner_attention.beta_override = value

    def forecast_multi(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        wake_bias=None,
    ):
        if wake_bias is None:
            raise ValueError("Physics-CTF requires a Jensen wake_bias")
        channels = x_enc.shape[-1]
        if self.use_norm:
            means, stds = [], []
            minimum = min(self.sampling_periods)
            for channel in range(channels):
                factor = int(self.sampling_periods[channel] / minimum)
                sample = x_enc[:, ::factor, channel]
                means.append(sample.mean(1, keepdim=True).detach())
                stds.append((sample.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt())
            mean = torch.cat(means, 1)[:, None]
            std = torch.cat(stds, 1)[:, None]
            x_enc = (x_enc - mean) / std
        tokens, patches, channels = self.channel_wise_patch_embedding(
            x_enc.transpose(1, 2), self.keep_prob
        )
        for layer in self.encoder.layers:
            layer.self_attention.inner_attention.set_wake_bias(
                wake_bias, patches, self.num_global_tokens
            )
        encoded, counts = self.encoder(tokens, patches, channels)
        self.latest_num_tokens_list = counts
        channel_tokens, offset = [], 0
        for patch_count, token_count in zip(patches, counts):
            channel_tokens.append(
                encoded[:, offset + patch_count : offset + patch_count + self.num_global_tokens]
            )
            offset += token_count
        channel_tokens = torch.stack(channel_tokens, 1)
        if self.num_global_tokens == 1:
            channel_tokens = channel_tokens.squeeze(2)
        prediction = self.head(
            channel_tokens, channels, num_global_tokens=self.num_global_tokens
        )
        return prediction * std + mean if self.use_norm else prediction

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        mask=None,
        wake_bias=None,
    ):
        if self.task_name != "long_term_forecast" or self.features != "M":
            raise NotImplementedError("SDWPF experiment only supports multivariate forecasting")
        output = self.forecast_multi(
            x_enc, x_mark_enc, x_dec, x_mark_dec, wake_bias=wake_bias
        )
        return output[:, -self.pred_len :, :]


def load_native_weights(physics: WakePhysicsCTF, original: OriginalCTF) -> None:
    """Copy all native CTF weights; leave only per-layer beta at initialization."""
    missing, unexpected = physics.load_state_dict(original.state_dict(), strict=False)
    if unexpected or any(not key.endswith(".theta") for key in missing):
        raise RuntimeError((missing, unexpected))
