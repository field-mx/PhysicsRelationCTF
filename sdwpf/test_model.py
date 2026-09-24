import unittest
from types import SimpleNamespace

import numpy as np
import torch

from model import OriginalCTF, WakePhysicsAttention, WakePhysicsCTF, load_native_weights


def tiny_config(channels=4):
    return SimpleNamespace(
        task_name="long_term_forecast",
        features="M",
        seq_len=36,
        pred_len=6,
        use_norm=True,
        patch_lens=[9] * channels,
        enc_in=channels,
        num_global_tokens=2,
        sampling_periods=[1] * channels,
        keep_prob=1.0,
        factor=1,
        dropout=0.0,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=2,
        activation="gelu",
        batch_size=2,
    )


class WakePhysicsModelTests(unittest.TestCase):
    def test_beta_zero_is_exact_original_ctf(self):
        torch.manual_seed(7)
        cfg = tiny_config()
        original = OriginalCTF(cfg).eval()
        physics = WakePhysicsCTF(cfg, beta_init=0.1).eval()
        load_native_weights(physics, original)
        physics.set_beta(0.0)
        x = torch.randn(2, 36, 4)
        wake = torch.rand(2, 4, 4)
        wake[:, torch.arange(4), torch.arange(4)] = 0
        with torch.no_grad():
            expected = original(x, None, None, None)
            actual = physics(x, None, None, None, wake_bias=wake)
        self.assertTrue(torch.equal(expected, actual))

    def test_bias_matrix_only_has_channel_token_pairs(self):
        inner = WakePhysicsAttention(torch.nn.Identity(), beta_init=0.1)
        wake = torch.zeros(1, 2, 2)
        wake[0, 1, 0] = 0.3
        # Two channels, four local patches and two channel tokens each.
        inner.set_wake_bias(wake, patch_counts=[4, 4], num_global_tokens=2)
        dense = inner.dense_physics_bias(total_tokens=12)
        nonzero = set(map(tuple, torch.nonzero(dense[0]).tolist()))
        self.assertEqual(nonzero, {(10, 4), (11, 5)})

    def test_positive_beta_changes_forward_and_has_gradients(self):
        torch.manual_seed(11)
        cfg = tiny_config()
        model = WakePhysicsCTF(cfg, beta_init=0.1).train()
        x = torch.randn(2, 36, 4)
        wake = torch.zeros(2, 4, 4)
        wake[:, 1, 0] = 0.4
        out = model(x, None, None, None, wake_bias=wake)
        out.square().mean().backward()
        grads = [layer.self_attention.inner_attention.theta.grad for layer in model.encoder.layers]
        self.assertTrue(all(g is not None and torch.isfinite(g) for g in grads))


if __name__ == "__main__":
    unittest.main()
