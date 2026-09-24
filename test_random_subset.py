import unittest

import numpy as np

from physics import get_data
from experiment import select_random_origins
from finalize import run_paths


class RandomFewShotSelectionTest(unittest.TestCase):
    def test_final_manifest_can_target_only_random_fewshot_ratios(self):
        paths = run_paths([0.5, 0.3], [42], "/baseline", "/physics")
        self.assertEqual(
            [str(p) for p in paths],
            [
                "/baseline/train050/CTF_seed42",
                "/physics/train050/Physics_seed42",
                "/baseline/train030/CTF_seed42",
                "/physics/train030/Physics_seed42",
            ],
        )

    def test_random_subsets_are_deterministic_nested_and_not_prefixes(self):
        origins = np.column_stack(
            [np.zeros(100, dtype=np.int64), np.arange(100, 200, dtype=np.int64)]
        )

        half = select_random_origins(origins, 0.5, seed=20260922)
        thirty = select_random_origins(origins, 0.3, seed=20260922)

        self.assertEqual(len(half), 50)
        self.assertEqual(len(thirty), 30)
        self.assertEqual(set(map(tuple, thirty)), set(map(tuple, half[:30])))
        self.assertFalse(np.array_equal(half[:, 1], np.arange(100, 150)))
        np.testing.assert_array_equal(
            half, select_random_origins(origins, 0.5, seed=20260922)
        )

    def test_random_fewshot_keeps_validation_and_test_fixed(self):
        full, _, _, full_selection, _ = get_data(
            1.0, sampling="random_windows", subset_seed=20260922
        )
        half, _, _, half_selection, _ = get_data(
            0.5, sampling="random_windows", subset_seed=20260922
        )
        thirty, _, _, thirty_selection, _ = get_data(
            0.3, sampling="random_windows", subset_seed=20260922
        )

        self.assertEqual(len(half["train"]), int(len(full["train"]) * 0.5))
        self.assertEqual(len(thirty["train"]), int(len(full["train"]) * 0.3))
        self.assertEqual(
            half_selection["origins_sha256"]["val"],
            thirty_selection["origins_sha256"]["val"],
        )
        self.assertEqual(
            half_selection["origins_sha256"]["test"],
            thirty_selection["origins_sha256"]["test"],
        )
        self.assertEqual(half_selection["sampling"], "random_windows")
        self.assertEqual(half_selection["subset_seed"], 20260922)
        self.assertTrue(
            set(map(tuple, thirty["train"].origins)).issubset(
                set(map(tuple, half["train"].origins))
            )
        )


if __name__ == "__main__":
    unittest.main()
