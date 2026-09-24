"""Create final comparison and dynamic Jensen-edge figures from frozen results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wake import jensen_wake_bias


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/sdwpf/cache"
RUNS = ROOT / "runs/sdwpf_36x6"


def farm_direction(direction: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(direction)
    mean = np.mod(
        np.rad2deg(np.arctan2(np.sin(radians).mean(1), np.cos(radians).mean(1))), 360
    )
    return np.repeat(mean[:, None], direction.shape[1], axis=1)


def rowmax(bias: np.ndarray) -> np.ndarray:
    denominator = bias.max(axis=2, keepdims=True)
    return np.divide(bias, denominator, out=np.zeros_like(bias), where=denominator > 0)


def comparison_figure(results: dict) -> None:
    rows = sorted(results["rows"], key=lambda row: row["train_percent"])
    x = np.asarray([row["train_percent"] for row in rows])
    ctf = np.asarray([row["ctf_mse"] for row in rows])
    physics = np.asarray([row["physics_mse"] for row in rows])
    gain = np.asarray([row["mse_gain_percent"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].plot(x, ctf, "o-", color="#354a5f", label="Original CTF")
    axes[0].plot(x, physics, "s-", color="#c77c24", label="Physics-CTF")
    axes[0].set(xlabel="Training data (%)", ylabel="Normalized test MSE")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    colors = ["#315d77" if value >= 0 else "#c77c24" for value in gain]
    axes[1].bar(x.astype(str), gain, color=colors)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set(xlabel="Training data (%)", ylabel="Physics-CTF MSE gain (%)")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("SDWPF 36-step input → 6-step forecast (seed 42)")
    fig.savefig(RUNS / "few_shot_comparison.png", dpi=220)
    fig.savefig(RUNS / "few_shot_comparison.pdf")
    plt.close(fig)


def edge_figure() -> None:
    origins = np.load(CACHE / "window_origins.npz")["test"]
    speed = np.load(CACHE / "speed_filled.npy", mmap_mode="r")[origins - 1]
    direction = np.load(CACHE / "direction_from.npy", mmap_mode="r")[origins - 1]
    xy = np.load(CACHE / "xy.npy")
    protocol = json.loads((CACHE / "protocol.json").read_text(encoding="utf-8"))
    all_raw_bias = []
    for indices in np.array_split(np.arange(len(origins)), 32):
        raw_bias = jensen_wake_bias(
            xy, farm_direction(direction[indices]), speed[indices], kw=0.075,
            coordinate_rotation_deg=protocol["coordinate_rotation_deg"],
        )
        all_raw_bias.append(raw_bias)
    raw_bias = np.concatenate(all_raw_bias)
    normalized_bias = rowmax(raw_bias)
    activity = (raw_bias > 0).mean(0)
    strength = raw_bias.mean(0)
    score = activity * strength
    np.fill_diagonal(score, 0)
    flat = np.argsort(score.ravel())[::-1]
    chosen = []
    for index in flat:
        dst, src = np.unravel_index(index, score.shape)
        if score[dst, src] <= 0:
            break
        chosen.append((dst, src))
        if len(chosen) == 3:
            break
    steps = np.load(CACHE / "steps.npy")[origins]
    time = np.datetime64("2021-01-01T00:10") + steps.astype("timedelta64[m]") * 10
    fig, axis = plt.subplots(figsize=(11, 4.3), constrained_layout=True)
    edge_rows = []
    for dst, src in chosen:
        values = raw_bias[:, dst, src]
        normalized = normalized_bias[:, dst, src]
        axis.plot(time, values, linewidth=1.2, label=f"T{src + 1} → T{dst + 1}")
        edge_rows.append({
            "source_turbine": int(src + 1),
            "destination_turbine": int(dst + 1),
            "active_fraction": float((values > 0).mean()),
            "mean_raw_jensen_deficit": float(values.mean()),
            "max_raw_jensen_deficit": float(values.max()),
            "mean_model_normalized_strength": float(normalized.mean()),
        })
    axis.set(
        title="Typical dynamic Jensen wake edges on the fixed test interval",
        xlabel="Time", ylabel="Raw Jensen normalized velocity deficit",
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    fig.savefig(RUNS / "dynamic_wake_edges.png", dpi=220)
    fig.savefig(RUNS / "dynamic_wake_edges.pdf")
    plt.close(fig)
    (RUNS / "dynamic_wake_edges.json").write_text(
        json.dumps(edge_rows, indent=2), encoding="utf-8"
    )


def main() -> None:
    results = json.loads((RUNS / "final_results.json").read_text(encoding="utf-8"))
    comparison_figure(results)
    edge_figure()


if __name__ == "__main__":
    main()
