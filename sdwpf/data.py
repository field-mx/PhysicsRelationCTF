"""Leakage-safe SDWPF data protocol helpers."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def causal_fill(values: np.ndarray, fallback: np.ndarray | float = 0.0) -> np.ndarray:
    """Forward-fill each channel, using only a fixed fallback before first data."""
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("values must have shape [time, channel]")
    rows = np.arange(values.shape[0])[:, None]
    indices = np.where(np.isfinite(values), rows, -1)
    np.maximum.accumulate(indices, axis=0, out=indices)
    safe = np.maximum(indices, 0)
    filled = values[safe, np.arange(values.shape[1])[None, :]].copy()
    fallback = np.broadcast_to(np.asarray(fallback), (values.shape[1],))
    leading = indices < 0
    filled[leading] = np.broadcast_to(fallback, values.shape)[leading]
    return filled


def continuous_blocks(
    start: int, end: int, block_len: int = 288, minimum_len: int = 42
) -> list[tuple[int, int]]:
    """Partition an interval into non-overlapping continuous blocks."""
    if min(block_len, minimum_len) <= 0 or end <= start:
        raise ValueError("invalid block interval")
    blocks = []
    cursor = start
    while cursor < end:
        block_end = min(cursor + block_len, end)
        if block_end - cursor >= minimum_len:
            blocks.append((cursor, block_end))
        cursor = block_end
    return blocks


def filter_eligible_origins(
    origins: np.ndarray,
    valid: np.ndarray,
    seq_len: int = 36,
    pred_len: int = 6,
    min_history_fraction: float = 0.5,
) -> np.ndarray:
    """Drop windows dominated by outages or having no evaluable target."""
    origins = np.asarray(origins, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    row_counts = valid.sum(axis=1, dtype=np.int64)
    cumulative = np.concatenate(([0], np.cumsum(row_counts)))
    history = cumulative[origins] - cumulative[origins - seq_len]
    future = cumulative[origins + pred_len] - cumulative[origins]
    required_history = min_history_fraction * seq_len * valid.shape[1]
    return origins[(history >= required_history) & (future > 0)]


def chronological_splits(
    n_steps: int,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    purge: int = 42,
) -> dict[str, tuple[int, int]]:
    """Return half-open chronological ranges separated by purge gaps."""
    if n_steps <= 2 * purge:
        raise ValueError("series is too short for requested purge")
    train_end = int(n_steps * train_fraction)
    val_end = int(n_steps * (train_fraction + val_fraction))
    ranges = {
        "train": (0, train_end),
        "val": (train_end + purge, val_end),
        "test": (val_end + purge, n_steps),
    }
    if any(end <= start for start, end in ranges.values()):
        raise ValueError("empty split")
    return ranges


def make_window_origins(
    start: int,
    end: int,
    seq_len: int = 36,
    pred_len: int = 6,
    stride: int = 6,
) -> np.ndarray:
    """Origins t for history [t-seq_len,t) and target [t,t+pred_len)."""
    if min(seq_len, pred_len, stride) <= 0:
        raise ValueError("sequence lengths and stride must be positive")
    first = start + seq_len
    stop = end - pred_len + 1
    return np.arange(first, stop, stride, dtype=np.int64)


def official_power_mask(
    patv: np.ndarray,
    wspd: np.ndarray,
    pitch: np.ndarray,
    wdir: np.ndarray | None = None,
    ndir: np.ndarray | None = None,
) -> np.ndarray:
    """Mask targets excluded by the SDWPF paper's explicit unknown-value rules."""
    patv = np.asarray(patv)
    wspd = np.asarray(wspd)
    pitch = np.asarray(pitch)
    if pitch.shape[:-1] != patv.shape or pitch.shape[-1] != 3:
        raise ValueError("pitch must have shape patv.shape + (3,)")
    finite = np.isfinite(patv) & np.isfinite(wspd) & np.isfinite(pitch).all(-1)
    stopped_with_wind = (patv <= 0) & (wspd > 2.5)
    invalid_pitch = (pitch > 89).any(-1)
    valid = finite & ~stopped_with_wind & ~invalid_pitch
    if wdir is not None:
        wdir = np.asarray(wdir)
        valid &= np.isfinite(wdir) & (np.abs(wdir) <= 180)
    if ndir is not None:
        ndir = np.asarray(ndir)
        valid &= np.isfinite(ndir) & (np.abs(ndir) <= 720)
    return valid


def stratified_nested_block_order(labels: np.ndarray, seed: int = 42) -> np.ndarray:
    """Order continuous blocks so every prefix approximates the full strata mix.

    Blocks are shuffled only *within* their regime. A largest-deficit scheduler
    interleaves regimes, which makes 10/20/50/100% selections nested while
    preserving coverage much better than taking a chronological prefix.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty vector")
    rng = np.random.default_rng(seed)
    groups: dict[object, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        groups[label].append(index)
    for indices in groups.values():
        rng.shuffle(indices)
    keys = sorted(groups, key=str)
    total = float(labels.size)
    selected = {key: 0 for key in keys}
    output: list[int] = []
    while len(output) < labels.size:
        next_count = len(output) + 1
        available = [key for key in keys if selected[key] < len(groups[key])]
        # Desired cumulative count minus actual count. Deterministic tie break.
        key = max(
            available,
            key=lambda k: (next_count * len(groups[k]) / total - selected[k], -keys.index(k)),
        )
        output.append(groups[key][selected[key]])
        selected[key] += 1
    return np.asarray(output, dtype=np.int64)
