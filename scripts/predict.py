"""Forecast the final complete historical window of a named-column CSV."""
from __future__ import annotations

import argparse
from collections import deque
import csv
from pathlib import Path

import numpy as np
import torch

from pretrained import load_pretrained


def read_history(path, channels, length):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None or len(header) != len(set(header)):
            raise ValueError("CSV requires a unique channel-name header")
        missing = set(channels) - set(header)
        if missing:
            raise ValueError(f"Missing CSV channels: {sorted(missing)}")
        indices = [header.index(name) for name in channels]
        rows = deque(maxlen=length)
        for row in reader:
            if len(row) != len(header):
                raise ValueError("CSV contains a row with an incorrect column count")
            rows.append(row)
    if len(rows) != length:
        raise ValueError(f"At least {length} real historical rows are required; no padding is performed")
    return np.asarray([[float(row[i]) for i in indices] for row in rows], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["industrial", "lbnl", "tep"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ctf-root")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.resolve().exists():
        parser.error("Output exists; choose a new path to preserve existing results")
    torch.set_num_threads(args.threads)
    predictor = load_pretrained(args.dataset, device=args.device, ctf_root=args.ctf_root)
    history = read_history(args.input, predictor.channels, predictor.input_length)
    forecast = predictor.predict(history, channels=predictor.channels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["forecast_step", "seconds_ahead", *predictor.output_channels])
        for step, row in enumerate(forecast, start=1):
            writer.writerow([step, step * predictor.sample_seconds, *row.tolist()])
    print(f"Saved {forecast.shape[0]} steps x {forecast.shape[1]} targets to {args.output}; units match the checkpoint input contract.")


if __name__ == "__main__":
    main()
