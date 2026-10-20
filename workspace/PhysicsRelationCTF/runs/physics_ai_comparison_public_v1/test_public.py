"""Public comparison contracts, written before the implementation."""
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class PublicContracts(unittest.TestCase):
    def modules(self):
        try:
            return importlib.import_module('public_models'), importlib.import_module('public_runner')
        except ModuleNotFoundError as error:
            self.fail(f'Public comparison implementation missing: {error}')

    def test_every_model_seed_is_scheduled_without_performance_screening(self):
        _, runner = self.modules()
        jobs = runner.jobs()
        expected = {(d, k, s) for d in ('lbnl', 'tep')
                    for k in ('PhyGNN', 'MTGNN', 'PatchTST', 'PG-Reg-CTF')
                    for s in (42, 43, 44)}
        self.assertEqual({(j['dataset'], j['model'], j['seed']) for j in jobs}, expected)
        self.assertEqual(len(jobs), 24)

    def test_direct_graph_and_equation_dependencies_not_reversed(self):
        models, _ = self.modules()
        graph = dict(channels=['a', 'b', 'c'],
                     features=[dict(inputs=[0, 1])],
                     edges=[dict(source=0, target=2, feature=0)])
        actual = models.augmented_adjacency(graph)
        want = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.],
                             [1., 0., 1., 1.], [1., 1., 0., 1.]])
        torch.testing.assert_close(actual, want)

    def test_train_tolerances_never_cross_segment_boundary(self):
        models, _ = self.modules()
        raw = np.array([[0.], [0.], [0.], [1000.], [1000.], [1000.]])
        bank = np.zeros_like(raw); valid = np.ones_like(raw, dtype=bool)
        tolerance, count = models.route_tolerance(raw, bank, valid, [(0, 3), (3, 6)],
                                                feature=0, target=0, lag=0, length=1,
                                                gain=1., scale=1.)
        self.assertEqual(count, 4)
        self.assertEqual(tolerance, .02)

    def test_evaluation_ignores_input_only_control_channels(self):
        models, _ = self.modules()
        p = torch.tensor([[[2., 1000.], [2., 1000.]]])
        y = torch.zeros_like(p)
        loss = models.forecast_loss(p, y, [0])
        self.assertEqual(float(loss), 4.)

    def test_previously_completed_industrial_comparators_are_marked_reused(self):
        _, runner = self.modules()
        result = runner.summarize()
        for kind in runner.MODELS:
            rows = result['results']['industrial'][kind]['rows']
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row['reused'] for row in rows))

    def test_real_one_epoch_training_saves_and_reloads_checkpoint(self):
        _, runner = self.modules()
        original = runner.load_data('tep')
        data = dict(original, config=dict(original['config'], d_model=32, d_ff=64, n_heads=4, e_layers=1),
                    policy=dict(epochs=1, lr=.0003, batch=2, min_epochs=1, patience=4),
                    datasets={k: torch.utils.data.Subset(v, [0, 1]) for k, v in original['datasets'].items()})
        job = dict(dataset='tep', model='PhyGNN', seed=42)
        with tempfile.TemporaryDirectory(prefix='.contract-', dir=HERE) as scratch:
            with patch.object(runner, 'HERE', Path(scratch)), patch.object(runner, 'load_data', return_value=data):
                try:
                    result = runner.train(job, 'cpu')
                except TypeError as error:
                    self.fail(f'Actual training cannot save checkpoint: {error}')
                path = runner.outdir(job)/'best.pth'
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                self.assertIsInstance(checkpoint['model'], dict)
                self.assertEqual(checkpoint['model_kind'], 'PhyGNN')
                self.assertEqual(result['epochs_completed'], 1)
                self.assertEqual(runner.train(job, 'cpu')['checkpoint_sha256'], runner.sha(path))

    def test_public_shapes_future_independence_finite_backward_and_reload(self):
        models, runner = self.modules()
        for dataset, length, channels in [('lbnl', 96, 73), ('tep', 20, 52)]:
            data = runner.load_data(dataset)
            cfg = dict(data['config'], d_model=32, d_ff=64, n_heads=4, e_layers=1)
            x, y = data['datasets']['val'][0]; x, y = x[None], y[None]
            for kind in runner.MODELS:
                with self.subTest(dataset=dataset, model=kind):
                    options = runner.options(dataset, kind)
                    model = models.make_model(kind, cfg, data, 42, options).eval()
                    p = model(x, None, None, None)
                    self.assertEqual(tuple(p.shape), (1, length, channels))
                    torch.testing.assert_close(p, model(x, None, torch.randn_like(y), None), atol=0, rtol=0)
                    models.forecast_loss(p, y, data['targets']).backward()
                    grads = [v.grad for v in model.parameters() if v.grad is not None]
                    self.assertTrue(grads)
                    self.assertTrue(all(bool(torch.isfinite(v).all()) for v in grads))
                    restored = models.make_model(kind, cfg, data, 42, options).eval()
                    restored.load_state_dict(model.state_dict())
                    torch.testing.assert_close(p.detach(), restored(x, None, None, None), atol=0, rtol=0)
                    if kind == 'MTGNN':
                        self.assertLessEqual(model.network.receptive_field, length)

    def test_physics_loss_has_finite_prediction_gradient_without_future_truth(self):
        models, runner = self.modules()
        for dataset in ('lbnl', 'tep'):
            data = runner.load_data(dataset)
            physical = models.PartialPhysicsLoss(data)
            x, y = data['datasets']['train'][10]
            prediction = (y[None] + 3. * torch.randn_like(y[None])).requires_grad_(True)
            loss = physical(x[None], prediction)
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertGreaterEqual(float(loss.detach()), 0.)
            loss.backward()
            self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
            self.assertGreater(float(prediction.grad.abs().sum()), 0.)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
