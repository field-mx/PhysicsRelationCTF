import unittest

import numpy as np

from data import (
    causal_fill,
    chronological_splits,
    continuous_blocks,
    filter_eligible_origins,
    make_window_origins,
    official_power_mask,
    stratified_nested_block_order,
)


class DataProtocolTests(unittest.TestCase):
    def test_window_is_36_to_6_and_stays_inside_interval(self):
        origins = make_window_origins(100, 200, seq_len=36, pred_len=6, stride=6)
        self.assertEqual(origins[0], 136)
        self.assertLessEqual(origins[-1] + 6, 200)
        self.assertTrue(np.all(np.diff(origins) == 6))

    def test_splits_have_42_point_purge(self):
        splits = chronological_splits(10_000, purge=42)
        self.assertEqual(splits["val"][0] - splits["train"][1], 42)
        self.assertEqual(splits["test"][0] - splits["val"][1], 42)

    def test_official_invalid_power_rules(self):
        patv = np.array([10.0, 0.0, 0.0, np.nan])
        wspd = np.array([5.0, 3.0, 1.0, 5.0])
        pitch = np.array([[0, 0, 0], [0, 0, 0], [90, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(
            official_power_mask(patv, wspd, pitch),
            np.array([True, False, False, False]),
        )

    def test_direction_outliers_are_not_evaluated(self):
        patv = np.ones(3)
        wspd = np.ones(3)
        pitch = np.zeros((3, 3))
        mask = official_power_mask(
            patv, wspd, pitch,
            wdir=np.array([0.0, 181.0, 0.0]),
            ndir=np.array([0.0, 0.0, 721.0]),
        )
        np.testing.assert_array_equal(mask, np.array([True, False, False]))

    def test_few_shot_blocks_are_nested_and_reproducible(self):
        labels = np.array([0] * 8 + [1] * 6 + [2] * 4 + [3] * 2)
        a = stratified_nested_block_order(labels, seed=42)
        b = stratified_nested_block_order(labels, seed=42)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(set(a), set(range(len(labels))))
        sets = {p: set(a[: int(np.ceil(p * len(a)))]) for p in (0.1, 0.2, 0.5, 1.0)}
        self.assertTrue(sets[0.1] <= sets[0.2] <= sets[0.5] <= sets[1.0])

    def test_causal_fill_never_backfills_from_future(self):
        values = np.array([[np.nan], [2.0], [np.nan], [9.0]])
        filled = causal_fill(values, fallback=np.array([0.0]))
        np.testing.assert_array_equal(filled[:, 0], np.array([0.0, 2.0, 2.0, 9.0]))

    def test_windows_from_blocks_never_cross_block_boundary(self):
        blocks = continuous_blocks(0, 650, block_len=288, minimum_len=42)
        self.assertEqual(blocks, [(0, 288), (288, 576), (576, 650)])
        for start, end in blocks:
            origins = make_window_origins(start, end, 36, 6, 6)
            self.assertTrue(np.all(origins - 36 >= start))
            self.assertTrue(np.all(origins + 6 <= end))

    def test_unobserved_windows_are_excluded(self):
        valid = np.ones((100, 2), dtype=bool)
        valid[42:70] = False
        origins = np.array([36, 70, 90])
        kept = filter_eligible_origins(origins, valid, seq_len=36, pred_len=6)
        np.testing.assert_array_equal(kept, np.array([36, 90]))


if __name__ == "__main__":
    unittest.main()
