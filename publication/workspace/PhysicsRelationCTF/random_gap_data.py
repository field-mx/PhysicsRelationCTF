"""Apply aligned train-only observation gaps without touching shared caches."""

import hashlib

import numpy as np

from experiment import Windows, load_data
from time_gap_sampling import filter_gap_origins, intervals, nested_time_gaps


def load_random_gap_data(ratio, seed):
    if ratio not in (1.0, 0.5, 0.3):
        raise ValueError("Random-gap budgets are fixed at 100%, 50%, and 30%")
    datasets, graph, meta, _ = load_data("industrial", 1.0, sampling="prefix")
    end = int(meta["train_end"])
    mask = nested_time_gaps(end, seed=seed)[ratio]
    original = datasets["train"].origins
    train_origins = filter_gap_origins(original, mask)
    if not len(train_origins):
        raise RuntimeError("Random-gap sampling produced no training windows")
    # Every retained timestamp must actually participate in at least one
    # complete 96->96 window, so the reported point budget is not inflated.
    boundaries = np.zeros(end + 1, dtype=np.int64)
    np.add.at(boundaries, train_origins[:, 1] - 96, 1)
    np.add.at(boundaries, train_origins[:, 1] + 96, -1)
    covered = np.cumsum(boundaries[:-1]) > 0
    if not np.array_equal(covered, mask):
        raise RuntimeError("Retained-point budget differs from usable observations")
    values = datasets["train"].values[0]
    fit = np.asarray(values[:end][mask])
    mean = fit.mean(0, dtype=np.float64)
    scale = fit.std(0, dtype=np.float64)
    scale = np.where(scale < 1e-5, 1.0, scale)
    result = {
        split: Windows(data.values, train_origins if split == "train" else data.origins,
                       mean, scale)
        for split, data in datasets.items()
    }
    selection = {
        "sampling": "random_gaps", "subset_seed": int(seed),
        "train_end": end, "rows": int(mask.sum()),
        "full_train_windows": len(original),
        "selected_train_windows": len(train_origins),
        "retained_segments": intervals(mask),
        "deleted_segments": intervals(mask, value=False),
        "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "windows": {split: len(data) for split, data in result.items()},
        "origins_sha256": {
            split: hashlib.sha256(data.origins.tobytes()).hexdigest()
            for split, data in result.items()
        },
        "scaler_fit": "retained training timestamps only",
    }
    return result, graph, meta, selection
