"""One-time fixed-test evaluation for a frozen SDWPF checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SDWPFForecastDataset
from model import OriginalCTF, WakePhysicsCTF
from train import CACHE, forward, masked_sums


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    settings = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    if (args.run_dir / "test_metrics.json").exists():
        raise RuntimeError("test was already evaluated for this run")
    cfg = SimpleNamespace(**settings["model_config"])
    model_name = settings["model"]
    model = (
        WakePhysicsCTF(cfg, beta_init=settings["beta_init"])
        if model_name == "physics"
        else OriginalCTF(cfg)
    )
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(args.device)
    model.to(device).eval()
    data = SDWPFForecastDataset(CACHE, "test")
    loader = DataLoader(
        data, batch_size=settings["batch_size"], shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    xy = np.load(CACHE / "xy.npy")
    rotation = float(settings["coordinate_rotation_deg"])
    mse_sum = mae_sum = count_sum = 0.0
    horizon_sq = np.zeros(6, dtype=np.float64)
    horizon_abs = np.zeros(6, dtype=np.float64)
    horizon_count = np.zeros(6, dtype=np.float64)
    predictions, targets, masks, origins = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            prediction, target, valid = forward(
                model, model_name, batch, device, xy, settings["kw"], rotation,
                settings.get("direction_mode", "local"),
                settings.get("bias_normalization", "none"),
            )
            squared, absolute, count = masked_sums(prediction, target, valid)
            mse_sum += float(squared)
            mae_sum += float(absolute)
            count_sum += float(count)
            error = prediction - target
            horizon_sq += (error.square() * valid).sum((0, 2)).cpu().numpy()
            horizon_abs += (error.abs() * valid).sum((0, 2)).cpu().numpy()
            horizon_count += valid.sum((0, 2)).cpu().numpy()
            if args.save_predictions:
                predictions.append(prediction.cpu().numpy())
                targets.append(target.cpu().numpy())
                masks.append(valid.cpu().numpy())
                origins.append(batch[-1].numpy())
    mse = mse_sum / count_sum
    mae = mae_sum / count_sum
    metrics = {
        "normalized_mse": mse,
        "normalized_mae": mae,
        "mse_kw2": mse * 1500.0 ** 2,
        "mae_kw": mae * 1500.0,
        "valid_values": int(count_sum),
        "horizon_normalized_mse": (horizon_sq / horizon_count).tolist(),
        "horizon_normalized_mae": (horizon_abs / horizon_count).tolist(),
    }
    (args.run_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if args.save_predictions:
        np.savez_compressed(
            args.run_dir / "test_predictions.npz",
            prediction=np.concatenate(predictions),
            target=np.concatenate(targets),
            valid=np.concatenate(masks),
            origin=np.concatenate(origins),
        )
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
