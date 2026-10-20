"""Deterministic nested, time-aligned gaps for the industrial train interval."""

import numpy as np


def intervals(mask, value=True):
    """Half-open runs where a one-dimensional mask has the requested value."""
    marked = np.asarray(mask, dtype=bool) == value
    edges = np.diff(np.pad(marked.astype(np.int8), (1, 1)))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def _can_remove(kept, start, stop, minimum):
    for left, right in intervals(kept):
        if right <= start or left >= stop:
            continue
        if left < start < right and start - left < minimum:
            return False
        if left < stop < right and right - stop < minimum:
            return False
    return True


def _seventy_percent_mask(length, target_deleted, rng, minimum, protected):
    # Random rejection from all positions is uniform conditional on the
    # currently feasible positions. Restart only if the remaining geometry
    # makes the exact point budget unreachable.
    for _ in range(32):
        kept = np.ones(length, dtype=bool)
        deleted = 0
        steps = 0
        while deleted < target_deleted and steps < 1000:
            gap_length = int(rng.integers(96, min(720, length) + 1))
            accepted = False
            while True:
                for _ in range(200):
                    start = int(rng.integers(0, length - gap_length + 1))
                    stop = start + gap_length
                    added = int(kept[start:stop].sum())
                    if not 0 < added <= target_deleted - deleted:
                        continue
                    if protected[start:stop].any():
                        continue
                    if not _can_remove(kept, start, stop, minimum):
                        continue
                    kept[start:stop] = False
                    deleted += added
                    accepted = True
                    break
                if accepted or gap_length == 1:
                    break
                gap_length = max(1, gap_length // 2)
            if not accepted:
                break
            steps += 1
        if deleted == target_deleted:
            return kept
    raise RuntimeError("Could not construct the exact nested gap budget")


def nested_time_gaps(length, seed=20260922, minimum_kept=288):
    """Return exact 100/50/30%-retention masks from one fixed RNG stream."""
    if length < 2 * minimum_kept + 96:
        raise ValueError("Train interval is too short for 96->96 gaps")
    if length % 10:
        raise ValueError("Exact 50% and 30% point budgets require a multiple of 10")
    rng = np.random.default_rng(seed)
    target_30_deleted = length * 7 // 10
    target_50_deleted = length // 2
    # Reserve one continuous, usable run in every train-time decile. Gap
    # positions remain uniform conditional on this fixed feasibility rule.
    protected = np.zeros(length, dtype=bool)
    for part in np.array_split(np.arange(length), 10):
        if len(part) < minimum_kept:
            raise ValueError("A train-time decile is shorter than the required kept run")
        start = int(rng.integers(int(part[0]), int(part[-1]) - minimum_kept + 2))
        protected[start:start + minimum_kept] = True
    keep30 = _seventy_percent_mask(length, target_30_deleted, rng, minimum_kept, protected)
    gap_list = intervals(keep30, value=False)
    priority = rng.permutation(len(gap_list))
    keep50 = np.ones(length, dtype=bool)
    deleted = 0
    for index in priority:
        start, stop = gap_list[int(index)]
        take = min(target_50_deleted - deleted, stop - start)
        if take <= 0:
            break
        # A partial terminal gap must attach its newly retained portion to
        # the previous kept run, not leave a short isolated tail.
        if stop == length and take < stop - start:
            keep50[stop - take:stop] = False
        else:
            keep50[start:start + take] = False
        deleted += take
    if deleted != target_50_deleted:
        raise RuntimeError("Failed to construct the exact 50% point budget")
    if any(stop - start < minimum_kept for start, stop in intervals(keep50)):
        raise RuntimeError("50% mask contains a short retained fragment")
    return {1.0: np.ones(length, dtype=bool), 0.5: keep50, 0.3: keep30}


def filter_gap_origins(origins, kept, history=96, horizon=96):
    """Keep only forecasting windows entirely inside observed timestamps."""
    origins = np.asarray(origins)
    kept = np.asarray(kept, dtype=bool)
    if origins.ndim != 2 or origins.shape[1] != 2 or kept.ndim != 1:
        raise ValueError("Expected origins [N,2] and a one-dimensional time mask")
    starts = origins[:, 1] - history
    stops = origins[:, 1] + horizon
    valid = ((origins[:, 0] == 0) & (starts >= 0) & (stops <= len(kept)))
    missing_prefix = np.pad(np.cumsum(~kept, dtype=np.int64), (1, 0))
    valid[valid] &= (missing_prefix[stops[valid]] - missing_prefix[starts[valid]] == 0)
    return origins[valid]
