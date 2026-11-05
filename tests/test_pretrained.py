"""Run separately for each dataset: PRCTF_TEST_DATASET=... python -m unittest discover -s tests."""
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.model_factory import ROOT
from scripts.pretrained import load_pretrained


class PretrainedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.dataset = os.environ.get("PRCTF_TEST_DATASET", "industrial")
        cls.ctf_root = os.environ.get("PRCTF_TEST_CTF_ROOT")
        cls.predictor = load_pretrained(cls.dataset, ctf_root=cls.ctf_root)
        p = cls.predictor
        generator = torch.Generator().manual_seed(20261105)
        noise = torch.randn(2, p.input_length, len(p.channels), generator=generator) * .1
        cls.history = (p.mean.cpu() + noise * p.scale.cpu()).numpy()

    def test_full_state_and_normalization_round_trip(self):
        p = self.predictor
        predicted = p.predict(self.history, targets_only=False)
        with torch.inference_mode():
            normalized = (torch.from_numpy(self.history) - p.mean) / p.scale
            expected = (p.model(normalized, None, None, None) * p.scale + p.mean).numpy()
        np.testing.assert_array_equal(predicted, expected)
        self.assertTrue(np.isfinite(predicted).all())
        self.assertEqual(predicted.shape, (2, p.output_length, len(p.channels)))

    def test_target_selection(self):
        p = self.predictor
        selected = p.predict(self.history)
        all_channels = p.predict(self.history, targets_only=False)
        np.testing.assert_array_equal(selected, all_channels[..., p.target_indices])
        self.assertEqual(selected.shape[-1], 58 if self.dataset == "lbnl" else len(p.channels))

    def test_named_channels_restore_order(self):
        p = self.predictor
        reverse = list(reversed(range(len(p.channels))))
        actual = p.predict(self.history[..., reverse], channels=[p.channels[i] for i in reverse])
        np.testing.assert_array_equal(actual, p.predict(self.history))

    def test_missing_and_nonfinite_history_rejected(self):
        p = self.predictor
        with self.assertRaises(ValueError):
            p.predict(self.history[:, :-1])
        with self.assertRaises(ValueError):
            p.predict(self.history[..., :-1])
        with self.assertRaises(ValueError):
            p.predict(self.history, channels=[p.channels[0]] * len(p.channels))
        broken = self.history.copy()
        broken[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            p.predict(broken)

    def test_release_integrity_enforced(self):
        with patch("scripts.pretrained.hashlib.sha256") as checksum:
            checksum.return_value.hexdigest.return_value = "0" * 64
            with self.assertRaisesRegex(ValueError, "integrity"):
                load_pretrained(self.dataset, ctf_root=self.ctf_root)

    def test_no_training_state_or_observation_series(self):
        meta = self.predictor.metadata
        self.assertEqual(set(meta), {"format_version", "dataset", "config", "options", "graph", "scalers", "schema", "provenance"})
        self.assertFalse(meta["provenance"]["raw_observations_included"])
        self.assertFalse(meta["provenance"]["optimizer_included"])
        self.assertEqual(meta["provenance"]["seed"], 42)
        self.assertFalse(self.predictor.model.training)

    def test_csv_cli_matches_python_api(self):
        p = self.predictor
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "history.csv"
            destination = Path(folder) / "forecast.csv"
            # An extra, earlier window must not change the final input window.
            with source.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(p.channels)
                writer.writerows(self.history.reshape(-1, len(p.channels)))
            command = [sys.executable, str(ROOT / "scripts/predict.py"), "--dataset", self.dataset,
                       "--input", str(source), "--output", str(destination)]
            if self.ctf_root:
                command += ["--ctf-root", self.ctf_root]
            subprocess.run(command, check=True, cwd=ROOT, capture_output=True, text=True)
            with destination.open() as stream:
                reader = csv.reader(stream)
                header = next(reader)
                rows = np.asarray([[float(value) for value in row] for row in reader])
            self.assertEqual(header[2:], p.output_channels)
            np.testing.assert_allclose(rows[:, 2:], p.predict(self.history[1]), rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(rows[:, 1], np.arange(1, p.output_length + 1) * p.sample_seconds)
            existing = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(existing.returncode, 0)


if __name__ == "__main__":
    unittest.main()
