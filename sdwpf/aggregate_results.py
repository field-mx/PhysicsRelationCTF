"""Aggregate the frozen one-seed final SDWPF comparison."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/sdwpf_36x6"
MAPPING = {
    100: ("final_original_100", "final_physics_100"),
    50: ("final_original_50", "final_physics_50"),
    20: ("tune_original_20", "tune_physics_farm_rowmax_b2_20"),
    10: ("final_original_10", "final_physics_10"),
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired_block_bootstrap(original_path: Path, physics_path: Path, seed: int = 42):
    original = np.load(original_path)
    physics = np.load(physics_path)
    if not np.array_equal(original["origin"], physics["origin"]):
        raise RuntimeError("paired runs do not share test origins")
    valid = original["valid"] & physics["valid"]
    target = original["target"]
    original_sq = ((original["prediction"] - target) ** 2 * valid).sum((1, 2))
    physics_sq = ((physics["prediction"] - target) ** 2 * valid).sum((1, 2))
    counts = valid.sum((1, 2))
    origins = original["origin"]
    blocks, start = [], 0
    for index in range(1, len(origins) + 1):
        boundary = index == len(origins) or origins[index] - origins[index - 1] != 6
        if boundary:
            for left in range(start, index, 24):
                right = min(left + 24, index)
                blocks.append((original_sq[left:right].sum(), physics_sq[left:right].sum(), counts[left:right].sum()))
            start = index
    blocks = np.asarray(blocks, dtype=np.float64)
    rng = np.random.default_rng(seed)
    gains = np.empty(2000)
    for iteration in range(len(gains)):
        sample = blocks[rng.integers(0, len(blocks), size=len(blocks))].sum(0)
        original_mse, physics_mse = sample[0] / sample[2], sample[1] / sample[2]
        gains[iteration] = 100 * (original_mse - physics_mse) / original_mse
    return {
        "block_length_windows": 24,
        "blocks": int(len(blocks)),
        "replicates": int(len(gains)),
        "mse_gain_percent_ci95": np.percentile(gains, [2.5, 97.5]).tolist(),
    }


def main() -> None:
    rows = []
    detailed = {}
    for fraction, (original_name, physics_name) in MAPPING.items():
        original_dir, physics_dir = RUNS / original_name, RUNS / physics_name
        original = read_json(original_dir / "test_metrics.json")
        physics = read_json(physics_dir / "test_metrics.json")
        original_val = read_json(original_dir / "summary.json")["validation"]
        physics_summary = read_json(physics_dir / "summary.json")
        physics_val = physics_summary["validation"]
        row = {
            "train_percent": fraction,
            "ctf_mse": original["normalized_mse"],
            "physics_mse": physics["normalized_mse"],
            "mse_gain_percent": 100 * (original["normalized_mse"] - physics["normalized_mse"]) / original["normalized_mse"],
            "ctf_mae": original["normalized_mae"],
            "physics_mae": physics["normalized_mae"],
            "mae_gain_percent": 100 * (original["normalized_mae"] - physics["normalized_mae"]) / original["normalized_mae"],
            "ctf_mae_kw": original["mae_kw"],
            "physics_mae_kw": physics["mae_kw"],
            "ctf_val_mse": original_val["mse"],
            "physics_val_mse": physics_val["mse"],
        }
        rows.append(row)
        detailed[str(fraction)] = {
            "original_run": original_name,
            "physics_run": physics_name,
            "original": original,
            "physics": physics,
            "physics_diagnostics": physics_summary.get("diagnostics", []),
            "paired_block_bootstrap": paired_block_bootstrap(
                original_dir / "test_predictions.npz",
                physics_dir / "test_predictions.npz",
            ),
        }
    rows.sort(key=lambda row: row["train_percent"], reverse=True)
    original_100 = next(row for row in rows if row["train_percent"] == 100)["ctf_mse"]
    physics_100 = next(row for row in rows if row["train_percent"] == 100)["physics_mse"]
    for row in rows:
        row["ctf_degradation_vs_100_percent"] = row["ctf_mse"] / original_100 - 1
        row["physics_degradation_vs_100_percent"] = row["physics_mse"] / physics_100 - 1
    result = {
        "protocol": read_json(ROOT / "data/sdwpf/cache/protocol.json"),
        "rows": rows,
        "details": detailed,
        "core_hypothesis_supported": all(
            rows[index]["mse_gain_percent"] < rows[index + 1]["mse_gain_percent"]
            for index in range(len(rows) - 1)
        ),
    }
    (RUNS / "final_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (RUNS / "final_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result["rows"], indent=2))


if __name__ == "__main__":
    main()
