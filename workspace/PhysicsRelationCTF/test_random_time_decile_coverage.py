"""Every train-time decile must retain a usable continuous observation run."""

import unittest

import numpy as np

from time_gap_sampling import nested_time_gaps


class DecileCoverageTests(unittest.TestCase):
    def test_each_budget_retains_288_continuous_points_in_every_decile(self):
        masks = nested_time_gaps(14540, seed=20260922)
        for ratio in (0.5, 0.3):
            for tenth, segment in enumerate(np.array_split(masks[ratio], 10)):
                changes = np.diff(np.pad(segment.astype(np.int8), (1, 1)))
                starts = np.flatnonzero(changes == 1)
                stops = np.flatnonzero(changes == -1)
                self.assertGreaterEqual(
                    int((stops - starts).max(initial=0)), 288,
                    f"{ratio} budget lacks a full 288-point run in decile {tenth}",
                )


if __name__ == "__main__":
    unittest.main()
