"""Freeze SDWPF splits, causal inputs, direction convention and few-shot blocks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data import (
    causal_fill,
    chronological_splits,
    continuous_blocks,
    filter_eligible_origins,
    make_window_origins,
    stratified_nested_block_order,
)
from wake import circular_error_deg, jensen_wake_bias


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/sdwpf/cache"
RATED_POWER_KW = 1500.0
FRACTIONS = (1.0, 0.5, 0.2, 0.1)


def circular_mean_deg(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if not values.size:
        return 0.0
    radians = np.deg2rad(values)
    return float(np.mod(np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())), 360))


def direction_candidates(
    ndir: np.ndarray, wdir: np.ndarray, external: np.ndarray | None, train_end: int
) -> tuple[int, dict]:
    candidates = {"plus": np.mod(ndir + wdir, 360), "minus": np.mod(ndir - wdir, 360)}
    report: dict[str, float | str] = {}
    if external is not None:
        valid = np.isfinite(external[:train_end])
        for name, direction in candidates.items():
            error = np.abs(circular_error_deg(direction[:train_end], external[:train_end]))
            report[f"{name}_external_circular_mae_deg"] = float(np.nanmean(error[valid]))
        best = min(candidates, key=lambda key: report[f"{key}_external_circular_mae_deg"])
        best_error = report[f"{best}_external_circular_mae_deg"]
        # Wdir_w is a coarse weather field and is empirically almost unrelated
        # to nacelle-relative SCADA direction here (~90 degree MAE). It may flag
        # a gross reversal, but cannot justify choosing between +/- conventions.
        if best_error < 45.0:
            choice = best
            report["selection_basis"] = "train-only external Wdir_w circular MAE"
        else:
            choice = "plus"
            report["selection_basis"] = (
                "external Wdir_w inconclusive; retain documented Ndir + Wdir convention"
            )
    else:
        # No independent reference: retain the documented relative-angle addition;
        # the separate coordinate/wake sanity score below can still flag reversal.
        choice = "plus"
        report["selection_basis"] = "SDWPF definition: Wdir relative to Ndir"
    report["selected"] = choice
    return (1 if choice == "plus" else -1), report


def wake_alignment_score(
    xy: np.ndarray,
    direction: np.ndarray,
    speed: np.ndarray,
    train_end: int,
    kw: float,
    rotation: float,
) -> float:
    indices = np.linspace(0, train_end - 1, min(512, train_end), dtype=int)
    scores, weights = [], []
    for chunk in np.array_split(indices, 16):
        bias = jensen_wake_bias(xy, direction[chunk], speed[chunk], kw, coordinate_rotation_deg=rotation)
        difference = speed[chunk, None, :] - speed[chunk, :, None]
        finite = np.isfinite(difference)
        scores.append(float(np.where(finite, bias * difference, 0).sum()))
        weights.append(float(np.where(finite, bias, 0).sum()))
    return sum(scores) / max(sum(weights), 1e-12)


def block_label(speed: np.ndarray, direction: np.ndarray, power: np.ndarray) -> int:
    median_speed = float(np.nanmedian(speed))
    speed_bin = int(np.digitize(median_speed, [4.0, 8.0, 12.0]))
    direction_bin = int(circular_mean_deg(direction) // 45) % 8
    median_power_fraction = float(np.nanmedian(power) / RATED_POWER_KW)
    state_bin = int(np.digitize(median_power_fraction, [0.2, 0.8]))
    return speed_bin * 24 + direction_bin * 3 + state_bin


def main() -> None:
    metadata = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    n_steps = int(metadata["time_steps"])
    splits = chronological_splits(n_steps, purge=42)
    train_end = splits["train"][1]
    raw = {name: np.load(CACHE / f"{name}.npy") for name in ("Patv", "Wspd", "Wdir", "Ndir")}
    target_valid = np.load(CACHE / "target_valid.npy")
    xy = np.load(CACHE / "xy.npy")

    power_input = causal_fill(np.where(target_valid, raw["Patv"], np.nan), fallback=0.0)
    speed_fallback = np.nanmedian(raw["Wspd"][:train_end], axis=0)
    speed_fallback = np.where(np.isfinite(speed_fallback), speed_fallback, 0.0)
    speed = causal_fill(
        np.where((raw["Wspd"] >= 0) & (raw["Wspd"] <= 40), raw["Wspd"], np.nan),
        fallback=speed_fallback,
    )
    wdir = causal_fill(
        np.where(np.abs(raw["Wdir"]) <= 180, raw["Wdir"], np.nan), fallback=0.0
    )
    ndir = causal_fill(
        np.where(np.abs(raw["Ndir"]) <= 720, raw["Ndir"], np.nan), fallback=0.0
    )
    external = np.load(CACHE / "Wdir_w.npy") if (CACHE / "Wdir_w.npy").exists() else None
    sign, direction_report = direction_candidates(ndir, wdir, external, train_end)
    direction = np.mod(ndir + sign * wdir, 360).astype(np.float32)

    rotation_scores = {
        str(rotation): wake_alignment_score(xy, direction, speed, train_end, 0.075, rotation)
        for rotation in (0.0, 90.0, 180.0, 270.0)
    }
    rotation = float(max(rotation_scores, key=rotation_scores.get))
    # A negative best score is an explicit failed sanity check, not something
    # silently repaired using validation/test data.
    sanity_passed = rotation_scores[str(rotation)] > 0

    np.save(CACHE / "power_input.npy", (power_input / RATED_POWER_KW).astype(np.float32))
    np.save(CACHE / "power_target.npy", (raw["Patv"] / RATED_POWER_KW).astype(np.float32))
    np.save(CACHE / "speed_filled.npy", speed.astype(np.float32))
    np.save(CACHE / "direction_from.npy", direction)

    all_train_blocks = continuous_blocks(*splits["train"], block_len=288, minimum_len=42)
    train_blocks = [
        (a, b) for a, b in all_train_blocks if float(target_valid[a:b].mean()) >= 0.1
    ]
    labels = np.asarray(
        [block_label(speed[a:b], direction[a:b], raw["Patv"][a:b]) for a, b in train_blocks]
    )
    order = stratified_nested_block_order(labels, seed=42)
    arrays = {}
    sample_report = {}
    for fraction in FRACTIONS:
        count = int(np.ceil(fraction * len(train_blocks)))
        chosen = sorted(order[:count].tolist())
        origins = np.concatenate(
            [make_window_origins(*train_blocks[index], 36, 6, 6) for index in chosen]
        )
        origins = filter_eligible_origins(origins, target_valid, 36, 6)
        key = f"train_{int(fraction * 100)}"
        arrays[key] = origins
        chosen_labels = labels[chosen]
        unique, counts = np.unique(chosen_labels, return_counts=True)
        speed_bins, speed_counts = np.unique(chosen_labels // 24, return_counts=True)
        direction_bins, direction_counts = np.unique((chosen_labels % 24) // 3, return_counts=True)
        sample_report[key] = {
            "blocks": count,
            "windows": int(len(origins)),
            "time_points": int(sum(train_blocks[i][1] - train_blocks[i][0] for i in chosen)),
            "regime_histogram": {str(int(k)): int(v) for k, v in zip(unique, counts)},
            "wind_speed_bin_counts": {
                str(int(k)): int(v) for k, v in zip(speed_bins, speed_counts)
            },
            "wind_direction_sector_counts": {
                str(int(k)): int(v) for k, v in zip(direction_bins, direction_counts)
            },
        }
    arrays["val"] = filter_eligible_origins(
        make_window_origins(*splits["val"], 36, 6, 6), target_valid, 36, 6
    )
    arrays["test"] = filter_eligible_origins(
        make_window_origins(*splits["test"], 36, 6, 6), target_valid, 36, 6
    )
    np.savez(CACHE / "window_origins.npz", **arrays)

    protocol = {
        "seq_len": 36,
        "pred_len": 6,
        "stride": 6,
        "block_len": 288,
        "seed": 42,
        "splits": {key: list(value) for key, value in splits.items()},
        "direction": direction_report,
        "coordinate_rotation_deg": rotation,
        "wake_alignment_scores_kw_0.075": rotation_scores,
        "wake_direction_sanity_passed": sanity_passed,
        "few_shot": sample_report,
        "val_windows": int(len(arrays["val"])),
        "test_windows": int(len(arrays["test"])),
        "power_scale_kw": RATED_POWER_KW,
    }
    (CACHE / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
