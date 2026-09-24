"""Validate and reshape the official long-form SDWPF Parquet file."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from data import chronological_splits, official_power_mask


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/sdwpf/sdwpf_2001_2112_full.parquet"
COORDINATES = ROOT / "data/sdwpf/sdwpf_turb_location_elevation.csv"
CACHE = ROOT / "data/sdwpf/cache"
TURBINES = np.arange(1, 135)
FIELDS = ["Patv", "Wspd", "Wdir", "Ndir", "Pab1", "Pab2", "Pab3"]


def timestamp_step(time: pd.Series, day: pd.Series | None = None) -> np.ndarray:
    if day is None:
        parsed = pd.to_datetime(time, errors="coerce")
    else:
        parsed_time = pd.to_datetime(time.astype(str), format="%H:%M", errors="coerce")
        parsed = pd.to_datetime(day.astype(str), errors="coerce").dt.normalize()
        parsed = parsed + pd.to_timedelta(parsed_time.dt.hour * 60 + parsed_time.dt.minute, unit="m")
    if parsed.isna().any():
        raise ValueError("unparseable Tmstamp values")
    elapsed_minutes = (parsed - parsed.min()).dt.total_seconds().to_numpy() / 60
    if np.any(np.mod(elapsed_minutes, 10)):
        raise ValueError("timestamps are not aligned to a 10-minute grid")
    return (elapsed_minutes // 10).astype(np.int64)


def main() -> None:
    schema = pq.read_schema(RAW)
    available = set(schema.names)
    required = {"TurbID", "Tmstamp", *FIELDS}
    if "Day" in available:
        required.add("Day")
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(f"missing required columns: {missing}; schema={schema.names}")
    optional = [name for name in ("Wdir_w",) if name in available]
    frame = pd.read_parquet(RAW, columns=sorted(required) + optional)
    original_rows = len(frame)
    if "Day" in frame:
        frame["Step"] = timestamp_step(frame["Tmstamp"], frame["Day"])
        selected_period = "legacy Day/Tmstamp grid"
    else:
        # The full release contains 15-minute 2020 and 10-minute 2021 data.
        # Mixing them would make one model step physically inconsistent, so the
        # documented 10-minute 2021 segment is selected before any split.
        timestamp = pd.to_datetime(frame["Tmstamp"], errors="coerce")
        frame = frame[(timestamp >= "2021-01-01") & (timestamp < "2022-01-01")].copy()
        frame["Step"] = timestamp_step(frame["Tmstamp"])
        selected_period = "2021-01-01 through 2021-12-31, uniform 10-minute segment"
    duplicate_count = int(frame.duplicated(["Step", "TurbID"]).sum())
    turbine_ids = np.sort(frame["TurbID"].dropna().unique().astype(int))
    if duplicate_count:
        raise RuntimeError(f"found {duplicate_count} duplicate turbine-timestamp rows")
    if not np.array_equal(turbine_ids, TURBINES):
        raise RuntimeError(f"unexpected turbine ids: {turbine_ids.tolist()}")

    full_steps = np.arange(int(frame.Step.min()), int(frame.Step.max()) + 1)
    CACHE.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for field in FIELDS + optional:
        panel = frame.pivot(index="Step", columns="TurbID", values=field)
        panel = panel.reindex(index=full_steps, columns=TURBINES)
        arrays[field] = panel.to_numpy(dtype=np.float32)
        np.save(CACHE / f"{field}.npy", arrays[field])

    pitch = np.stack([arrays["Pab1"], arrays["Pab2"], arrays["Pab3"]], -1)
    valid = official_power_mask(
        arrays["Patv"], arrays["Wspd"], pitch,
        wdir=arrays["Wdir"], ndir=arrays["Ndir"],
    )
    np.save(CACHE / "target_valid.npy", valid)
    np.save(CACHE / "steps.npy", full_steps)

    coordinates = pd.read_csv(COORDINATES).sort_values("TurbID")
    if not np.array_equal(coordinates.TurbID.to_numpy(dtype=int), TURBINES):
        raise RuntimeError("coordinate TurbID mapping does not match SCADA TurbID 1..134")
    xy = coordinates[["x", "y"]].to_numpy(dtype=np.float32)
    np.save(CACHE / "xy.npy", xy)

    splits = chronological_splits(len(full_steps), purge=42)
    metadata = {
        "rows": int(len(frame)),
        "source_rows": int(original_rows),
        "selected_period": selected_period,
        "time_steps": int(len(full_steps)),
        "turbines": int(len(TURBINES)),
        "first_step": int(full_steps[0]),
        "last_step": int(full_steps[-1]),
        "missing_complete_timestamps": int(
            sum(np.all(~np.isfinite(arrays[field]), axis=1).sum() for field in ["Patv"])
        ),
        "target_valid_fraction": float(valid.mean()),
        "splits": {key: list(value) for key, value in splits.items()},
        "purge": 42,
        "seq_len": 36,
        "pred_len": 6,
        "optional_fields": optional,
        "columns": schema.names,
    }
    (CACHE / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
