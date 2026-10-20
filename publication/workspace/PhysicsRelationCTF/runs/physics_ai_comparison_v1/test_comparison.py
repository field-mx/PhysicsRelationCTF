"""Tests precede implementation; use unittest to avoid dependency changes."""
import importlib
from pathlib import Path
import sys
import unittest

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class ComparisonContracts(unittest.TestCase):
    def modules(self):
        try:
            return importlib.import_module('comparison_models'), importlib.import_module('comparison_runner')
        except ModuleNotFoundError as error:
            self.fail(f'Approved comparison implementation missing: {error}')

    def test_direction_keeps_source_to_target_not_its_transpose(self):
        models, _ = self.modules()
        graph = {'channels': ['source', 'target', 'isolated'],
                 'edges': [{'source': 0, 'target': 1, 'feature': None}]}
        adj = models.sensor_adjacency(graph)
        torch.testing.assert_close(adj, torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]))

    def test_augmented_graph_routes_multi_input_equation_without_inventing_edges(self):
        models, _ = self.modules()
        graph = {'channels': ['SCRDLFK', 'SCRDYFK', 'TT010401'],
                 'edges': [{'source': 0, 'target': 2, 'feature': 'power_W'}]}
        adj = models.augmented_adjacency(graph)
        # Sensors occupy 0..2; first equation feature is power, at index 3.
        self.assertEqual(float(adj[3, 0]), 1.)
        self.assertEqual(float(adj[3, 1]), 1.)
        self.assertEqual(float(adj[2, 3]), 1.)
        self.assertEqual(float(adj[0, 2]), 0.)
        self.assertEqual(float(adj[1, 2]), 0.)

    def test_model_shapes_gradients_future_decoder_independence_and_reload(self):
        models, runner = self.modules()
        data = runner.load_industrial()
        cfg = dict(data['config'], d_model=32, d_ff=64, n_heads=4, e_layers=1)
        x, y = data['datasets']['val'][0]
        x = x[None]; y = y[None]
        for kind in runner.NEW_MODELS:
            with self.subTest(kind=kind):
                model = models.make_model(kind, cfg, data, 42, runner.nominal_options(kind)).eval()
                p = model(x, None, None, None)
                self.assertEqual(tuple(p.shape), (1, 96, 56))
                self.assertTrue(bool(torch.isfinite(p).all()))
                torch.testing.assert_close(p, model(x, None, torch.randn_like(y), None), atol=0, rtol=0)
                (p-y).square().mean().backward()
                gradients = [v.grad for v in model.parameters() if v.requires_grad and v.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
                restored = models.make_model(kind, cfg, data, 42, runner.nominal_options(kind)).eval()
                restored.load_state_dict(model.state_dict())
                torch.testing.assert_close(p.detach(), restored(x, None, None, None), atol=0, rtol=0)

    def test_physics_loss_uses_prediction_gradients_and_not_future_truth(self):
        models, runner = self.modules()
        data = runner.load_industrial()
        loss = models.PartialPhysicsLoss(data)
        x, y = data['datasets']['train'][100]
        p = (y[None] + .3 * torch.randn_like(y[None])).requires_grad_(True)
        value = loss(x[None], p)
        self.assertGreaterEqual(float(value.detach()), 0.)
        value.backward()
        self.assertIsNotNone(p.grad)
        self.assertTrue(bool(torch.isfinite(p.grad).all()))
        self.assertGreater(float(p.grad.abs().sum()), 0.)

    def test_validations_screen_but_stronger_comparators_remain_in_ledger(self):
        _, runner = self.modules()
        self.assertEqual(runner.screen(0.11, 0.10), 'append_seeds_43_44')
        self.assertEqual(runner.screen(0.09, 0.10), 'pilot_only_stronger')
        self.assertEqual(runner.screen(0.10, 0.10), 'pilot_only_tie')
        with self.assertRaises(ValueError):
            runner.screen(float('nan'), 0.10)

    def test_cached_windows_match_frozen_existing_origins(self):
        _, runner = self.modules()
        data = runner.load_industrial()
        self.assertEqual(data['selection']['windows'], {'train': 14349, 'val': 50, 'test': 50})
        self.assertEqual(data['selection']['origins_sha256']['train'],
                         '091a68fbaa7ba7ff30655ff7b0a8495151f45f0150b95f3e9b81fc45e26acffd')
        self.assertEqual(data['selection']['origins_sha256']['val'],
                         '2f71b2a803e2f74340af983f54f80365170baa2a9583fcb1f34f48fe0dea7bff')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
