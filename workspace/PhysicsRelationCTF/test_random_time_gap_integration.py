"""Real-cache checks for the train-only random-gap sampling mode."""

import unittest

import numpy as np

import physics


class RandomGapDataIntegrationTests(unittest.TestCase):
    def test_exact_observation_budgets_leave_val_and_test_windows_fixed(self):
        full, _, meta, _, _ = physics.get_data(1.0, sampling="prefix")
        self.assertEqual(meta["train_end"], 14540)
        for ratio, expected_points in ((1.0, 14540), (0.5, 7270), (0.3, 4362)):
            try:
                data, _, _, selection, _ = physics.get_data(
                    ratio, sampling="random_gaps", subset_seed=20260922
                )
            except ValueError as exc:
                self.fail(f"random-gaps data construction is unavailable: {exc}")
            self.assertEqual(selection["rows"], expected_points)
            np.testing.assert_array_equal(data["val"].origins, full["val"].origins)
            np.testing.assert_array_equal(data["test"].origins, full["test"].origins)
            mask = physics.nested_time_gaps(14540, seed=20260922)[ratio]
            expected = physics.filter_gap_origins(full["train"].origins, mask)
            np.testing.assert_array_equal(data["train"].origins, expected)


if __name__ == "__main__":
    unittest.main()
