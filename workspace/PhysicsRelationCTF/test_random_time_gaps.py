"""Regression checks for aligned, nested train-only time gaps."""

import unittest

import numpy as np

import physics


def kept_lengths(mask):
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    return np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)


class NestedTimeGapTests(unittest.TestCase):
    def test_exact_nested_point_budgets_and_minimum_kept_run(self):
        # Regression: selecting fewer overlapping windows must not count as
        # reducing the number of distinct observed training timestamps.
        self.assertTrue(hasattr(physics, "nested_time_gaps"))
        masks = physics.nested_time_gaps(14540, seed=20260922)
        self.assertEqual(int(masks[1.0].sum()), 14540)
        self.assertEqual(int(masks[0.5].sum()), 7270)
        self.assertEqual(int(masks[0.3].sum()), 4362)
        self.assertTrue(np.all(masks[0.3] <= masks[0.5]))
        self.assertTrue(np.all(masks[0.5] <= masks[1.0]))
        for ratio in (0.5, 0.3):
            self.assertTrue(np.all(kept_lengths(masks[ratio]) >= 288))
        repeated = physics.nested_time_gaps(14540, seed=20260922)
        for ratio in (1.0, 0.5, 0.3):
            np.testing.assert_array_equal(masks[ratio], repeated[ratio])

    def test_only_complete_windows_inside_kept_runs_survive(self):
        self.assertTrue(hasattr(physics, "filter_gap_origins"))
        mask = np.ones(700, dtype=bool)
        mask[300:396] = False
        origins = np.column_stack((np.zeros(509, dtype=np.int64),
                                   np.arange(96, 605, dtype=np.int64)))
        selected = physics.filter_gap_origins(origins, mask, 96, 96)
        expected_times = np.r_[np.arange(96, 205), np.arange(492, 605)]
        np.testing.assert_array_equal(selected[:, 1], expected_times)


if __name__ == "__main__":
    unittest.main()
