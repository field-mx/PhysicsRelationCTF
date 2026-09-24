"""Train one validation-selected Original CTF or Wake Physics-CTF run."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SDWPFForecastDataset
from model import OriginalCTF, WakePhysicsCTF
from wake import jensen_wake_bias


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/sdwpf/cache"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def config(batch_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        features="M",
        seq_len=36,
        pred_len=6,
        use_norm=True,
        patch_lens=[9] * 134,
        enc_in=134,
        num_global_tokens=2,
        sampling_periods=[1] * 134,
        keep_prob=1.0,
        factor=3,
        dropout=0.1,
        d_model=256,
        n_heads=8,
        d_ff=1024,
        e_layers=2,
        activation="gelu",
        batch_size=batch_size,
    )


def masked_sums(prediction, target, valid):
    error = prediction - target
    valid_float = valid.to(error.dtype)
    squared = (error.square() * valid_float).sum()
    absolute = (error.abs() * valid_float).sum()
    count = valid_float.sum()
    return squared, absolute, count


def make_wake(
    speed, direction, xy, kw, rotation, device,
    direction_mode="local", bias_normalization="none",
):
    if direction_mode == "farm":
        radians = torch.deg2rad(direction)
        farm = torch.rad2deg(torch.atan2(radians.sin().mean(1), radians.cos().mean(1))) % 360
        direction = farm[:, None].expand_as(direction)
    bias = jensen_wake_bias(
        xy,
        direction.numpy(),
        speed.numpy(),
        kw=kw,
        coordinate_rotation_deg=rotation,
    )
    if bias_normalization == "rowmax":
        denominator = bias.max(axis=2, keepdims=True)
        bias = np.divide(bias, denominator, out=np.zeros_like(bias), where=denominator > 0)
    return torch.from_numpy(bias).to(device, non_blocking=True)


def forward(
    model, model_name, batch, device, xy, kw, rotation,
    direction_mode="local", bias_normalization="none",
):
    x, y, valid, speed, direction, _ = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    valid = valid.to(device, non_blocking=True)
    if model_name == "physics":
        wake = make_wake(
            speed, direction, xy, kw, rotation, device,
            direction_mode, bias_normalization,
        )
        prediction = model(x, None, None, None, wake_bias=wake)
    else:
        prediction = model(x, None, None, None)
    return prediction, y, valid


@torch.no_grad()
def evaluate(
    model, model_name, loader, device, xy, kw, rotation,
    direction_mode="local", bias_normalization="none",
):
    model.eval()
    mse_sum = mae_sum = count_sum = 0.0
    for batch in loader:
        prediction, target, valid = forward(
            model, model_name, batch, device, xy, kw, rotation,
            direction_mode, bias_normalization,
        )
        squared, absolute, count = masked_sums(prediction, target, valid)
        mse_sum += float(squared)
        mae_sum += float(absolute)
        count_sum += float(count)
    return {"mse": mse_sum / count_sum, "mae": mae_sum / count_sum, "count": int(count_sum)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["original", "physics"], required=True)
    parser.add_argument("--fraction", type=int, choices=[100, 50, 20, 10], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kw", type=float, default=0.075)
    parser.add_argument("--beta-init", type=float, default=0.1)
    parser.add_argument("--direction-mode", choices=["local", "farm"], default="local")
    parser.add_argument("--bias-normalization", choices=["none", "rowmax"], default="none")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    protocol = json.loads((CACHE / "protocol.json").read_text(encoding="utf-8"))
    rotation = float(protocol["coordinate_rotation_deg"])
    xy = np.load(CACHE / "xy.npy")
    train_data = SDWPFForecastDataset(CACHE, f"train_{args.fraction}")
    val_data = SDWPFForecastDataset(CACHE, "val")
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)

    cfg = config(args.batch_size)
    model = (
        WakePhysicsCTF(cfg, beta_init=args.beta_init)
        if args.model == "physics"
        else OriginalCTF(cfg)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    output = ROOT / "runs/sdwpf_36x6" / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    settings = vars(args) | {
        "model_config": vars(cfg),
        "train_windows": len(train_data),
        "val_windows": len(val_data),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "coordinate_rotation_deg": rotation,
    }
    (output / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    best = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        mse_sum = count_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction, target, valid = forward(
                model, args.model, batch, device, xy, args.kw, rotation,
                args.direction_mode, args.bias_normalization,
            )
            squared, _, count = masked_sums(prediction, target, valid)
            loss = squared / count.clamp_min(1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            mse_sum += float(squared.detach())
            count_sum += float(count.detach())
        validation = evaluate(
            model, args.model, val_loader, device, xy, args.kw, rotation,
            args.direction_mode, args.bias_normalization,
        )
        beta = []
        if args.model == "physics":
            beta = [
                float(layer.self_attention.inner_attention.beta.detach())
                for layer in model.encoder.layers
            ]
        record = {
            "epoch": epoch,
            "train_mse": mse_sum / count_sum,
            "val_mse": validation["mse"],
            "val_mae": validation["mae"],
            "lr": optimizer.param_groups[0]["lr"],
            "beta": beta,
            "seconds": time.time() - started,
        }
        history.append(record)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
        if validation["mse"] < best:
            best = validation["mse"]
            bad_epochs = 0
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "val": validation, "settings": settings},
                output / "best.pt",
            )
        else:
            bad_epochs += 1
        scheduler.step()
        if bad_epochs >= args.patience:
            break

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    if args.model == "physics":
        for layer in model.encoder.layers:
            layer.self_attention.inner_attention.record = True
        first_batch = next(iter(val_loader))
        forward(
            model, args.model, first_batch, device, xy, args.kw, rotation,
            args.direction_mode, args.bias_normalization,
        )
        diagnostics = [
            layer.self_attention.inner_attention.diagnostics
            for layer in model.encoder.layers
        ]
    else:
        diagnostics = []
    summary = {
        "best_epoch": checkpoint["epoch"],
        "validation": checkpoint["val"],
        "diagnostics": diagnostics,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
