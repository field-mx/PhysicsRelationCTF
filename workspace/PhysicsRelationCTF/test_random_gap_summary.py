"""Pairwise metric arithmetic for the fixed six-run experiment."""

import importlib
import unittest


class RandomGapSummaryTests(unittest.TestCase):
    def test_gain_and_degradation_use_same_budget_baseline(self):
        try:
            summarize = importlib.import_module("run_random_gap_experiment").summarize
        except (ImportError, AttributeError) as exc:
            self.fail(f"summary implementation is missing: {exc}")
        metrics = {
            1.0: {"CTF": {"mse": 0.20, "mae": 0.10},
                  "Physics": {"mse": 0.19, "mae": 0.09}},
            0.5: {"CTF": {"mse": 0.30, "mae": 0.15},
                  "Physics": {"mse": 0.25, "mae": 0.13}},
            0.3: {"CTF": {"mse": 0.40, "mae": 0.20},
                  "Physics": {"mse": 0.35, "mae": 0.17}},
        }
        result = summarize(metrics)
        self.assertAlmostEqual(result[0.5]["mse_gain"], 1 / 6)
        self.assertAlmostEqual(result[0.5]["ctf_mse_degradation"], 0.5)
        self.assertAlmostEqual(result[0.5]["physics_mse_degradation"], 0.25 / 0.19 - 1)
        self.assertAlmostEqual(result[0.3]["mae_gain"], 0.15)


if __name__ == "__main__":
    unittest.main()
