"""Validation-only search for reusable Physics-CTF initial conditions.

The forecasting model is unchanged. This entry point initializes the existing
physical branch deterministically and reuses run_experiment.train verbatim.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import run_experiment as experiment
from physics import LAGS, RELATIONS, ROOT, config, dump, get_data


SEARCH_BOUNDS = {"beta_init": [0.01, 0.15], "gamma_init": [0.4, 1.0]}
DEFAULT_LAG_LOG_TEMPERATURE = -2.0
RESIDUAL_GRADIENT_CLIP_NORM = 1.0


def configure_determinism():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _clip_unit_norm(gradient):
    norm = gradient.norm()
    return gradient * torch.clamp(RESIDUAL_GRADIENT_CLIP_NORM / norm.clamp_min(1e-12), max=1.0)


def install_initializer(lag_log_temperature):
    original_make_model = experiment.make_model

    def initialized_make_model(kind, cfg, graph, scalers, seed, options):
        model = original_make_model(kind, cfg, graph, scalers, seed, options)
        if kind == "Physics":
            with torch.no_grad():
                # The existing projector is physical_base + zero adapter.
                if torch.count_nonzero(model.adapter_out).item():
                    raise RuntimeError("Physical adapter is not initialized to zero")
                model.residual_net[2].weight.zero_()
                model.residual_net[2].bias.zero_()
                model.lag_log_temperature.fill_(lag_log_temperature)
            # At exactly zero the existing F.normalize residual has a very
            # large derivative. Bound only the final residual layer gradient.
            model.residual_net[2].weight.register_hook(_clip_unit_norm)
            model.residual_net[2].bias.register_hook(_clip_unit_norm)
        return model

    experiment.make_model = initialized_make_model


@torch.no_grad()
def initial_diagnostics(seed, options, device):
    data, graph, meta, _, scalers = get_data(1.0, sampling="prefix")
    model = experiment.make_model("Physics", config(len(meta["channels"])), graph, scalers, seed, options).to(device)
    experiment.record(model, True)
    model.eval()
    x, _ = next(iter(DataLoader(data["train"], batch_size=4, shuffle=False)))
    model(x.to(device), None, None, None)
    result = [layer.self_attention.inner_attention.diagnostics for layer in model.encoder.layers]
    if any(not math.isfinite(v) for d in result for v in d.values()):
        raise RuntimeError("Non-finite epoch-zero physics diagnostics")
    if any(d["residual_norm"] != 0.0 for d in result):
        raise RuntimeError("Residual branch is not zero-output at epoch zero")
    if any(d["physics_value_ratio"] >= 1.0 for d in result):
        raise RuntimeError("Physics message exceeds the native value scale at epoch zero")
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def run_candidate(args):
    configure_determinism()
    if args.init_yaml:
        selected = yaml.safe_load(Path(args.init_yaml).read_text())["initialization"]
        beta, gamma = float(selected["beta_init"]), float(selected["gamma_init"])
        rho = float(selected["rho_init"])
        lag_log_temperature = float(selected["lag_log_temperature_init"])
        seed = int(args.seed if args.seed is not None else selected["seed"])
    else:
        beta, gamma, rho = float(args.beta), float(args.gamma), float(args.rho)
        lag_log_temperature = DEFAULT_LAG_LOG_TEMPERATURE
        seed = int(args.seed if args.seed is not None else 42)
    if not SEARCH_BOUNDS["beta_init"][0] <= beta <= SEARCH_BOUNDS["beta_init"][1]:
        raise ValueError("beta_init is outside the declared initialization search range")
    if not SEARCH_BOUNDS["gamma_init"][0] <= gamma <= SEARCH_BOUNDS["gamma_init"][1]:
        raise ValueError("gamma_init is outside the declared initialization search range")
    if not 0 < rho < 0.02:
        raise ValueError("rho_init must be in the model's constrained interval (0, 0.02)")
    if not -5.0 <= lag_log_temperature <= 0.0:
        raise ValueError("lag log temperature outside the conservative numeric range")
    install_initializer(lag_log_temperature)
    options = dict(beta=beta, gamma=gamma, rho=rho, representation="centered", lag_mode="physical")
    out = Path(args.root) / "train100" / f"Physics_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(beta_init=beta, gamma_init=gamma, rho_init=rho, seed=seed,
                    lag_log_temperature_init=lag_log_temperature,
                    allowed_lags_seconds=LAGS, relation_types=RELATIONS,
                    physics_projector_init="direct physical base plus zero adapter",
                    residual_init="zero final linear weight and bias",
                    residual_gradient_clip_norm=RESIDUAL_GRADIENT_CLIP_NORM,
                    search_bounds=SEARCH_BOUNDS, training_ratio=1.0,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    manifest_path = out / "initialization.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError(f"Cannot overwrite a different initialization: {out}")
    if not manifest_path.exists():
        manifest["epoch_zero_diagnostics"] = initial_diagnostics(seed, options, args.device)
        dump(manifest_path, manifest)
    else:
        # Epoch-zero diagnostics are an observation, not part of the protocol.
        pass
    result = experiment.train("Physics", 1.0, seed, out, args.device, options,
                              epochs=args.epochs, lr=args.lr, sampling="prefix")
    print(json.dumps({"candidate": str(out), "best_val_mse": result["best_val_mse"],
                      "best_val_mae": result["best_val_mae"]}), flush=True)


def select_candidate(args):
    root = Path(args.root)
    rows = []
    for path in sorted(root.glob("*/*/train100/Physics_seed42")):
        training_file, init_file, history_file = (path / name for name in
                                                  ["training.json", "initialization.json", "history.json"])
        if not all(p.exists() for p in [training_file, init_file, history_file]):
            continue
        training = json.loads(training_file.read_text())
        init = json.loads(init_file.read_text())
        history = json.loads(history_file.read_text())
        best = min(history, key=lambda item: item["val_mse"])
        diagnostics = best["diagnostics"]
        first_near_best = next(row["epoch"] for row in history
                               if row["val_mse"] <= training["best_val_mse"] * 1.01)
        valid = all(d["residual_norm"] < d["physics_norm"] and
                    math.isfinite(d["physics_value_ratio"]) and
                    0 < d["gamma"] < 2 for d in diagnostics)
        rows.append(dict(path=str(path), beta_init=init["beta_init"],
                         gamma_init=init["gamma_init"], rho_init=init["rho_init"],
                         lag_log_temperature_init=init["lag_log_temperature_init"],
                         val_mse=training["best_val_mse"],
                         val_mae=training["best_val_mae"], best_epoch=training["best_epoch"],
                         first_within_1pct_epoch=first_near_best,
                         learned_beta=[d["beta"] for d in diagnostics],
                         learned_gamma=[d["gamma"] for d in diagnostics],
                         physics_value_ratio=[d["physics_value_ratio"] for d in diagnostics],
                         physics_residual_ratio=[d["physics_residual_ratio"] for d in diagnostics],
                         valid=valid))
    if len(rows) < 10:
        raise RuntimeError(f"Expected 10–20 complete candidates, found {len(rows)}")
    valid_rows = [row for row in rows if row["valid"]]
    if not valid_rows:
        raise RuntimeError("No candidate met the initial physics-scale checks")
    min_error = min(row["val_mse"] for row in valid_rows)
    close = [row for row in valid_rows if row["val_mse"] <= min_error * 1.001]
    # The reference is an engineering setting, not a known physical constant.
    def reference_distance(row):
        return abs(row["beta_init"] - 0.05) / 0.14 + abs(row["gamma_init"] - 0.75) / 0.6
    best = min(close, key=lambda row: (reference_distance(row),
                                       row["first_within_1pct_epoch"], row["val_mse"]))
    dump(root / "candidate_table.json", dict(rows=rows, selected=best,
                                              selection_policy="within 0.1% of minimum validation MSE, then nearest reference and faster convergence"))
    init = dict(beta_init=best["beta_init"], gamma_init=best["gamma_init"],
                rho_init=best["rho_init"], lag_log_temperature_init=best["lag_log_temperature_init"],
                seed=42, physics_projector_init="direct physical base plus zero adapter",
                residual_init="zero final linear weight and bias",
                residual_gradient_clip_norm=RESIDUAL_GRADIENT_CLIP_NORM)
    output = dict(schema="physics_ctf_initialization_v1", initialization=init,
                  beta_init_search_bounds=SEARCH_BOUNDS["beta_init"],
                  gamma_init_search_bounds=SEARCH_BOUNDS["gamma_init"],
                  allowed_lags_seconds=LAGS, lag_center_and_width="not separately parameterized in current model",
                  kappa="not present in current model", selected_by="validation only",
                  validation_mse=best["val_mse"], source_candidate=best["path"])
    target = root / "physics_init_best.yaml"
    if target.exists():
        raise RuntimeError(f"Refusing to overwrite a previously selected initialization: {target}")
    target.write_text(yaml.safe_dump(output, sort_keys=False, allow_unicode=True))
    print(json.dumps({"selected": best, "config": str(target)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest="action", required=True)
    run = actions.add_parser("run")
    run.add_argument("--root", required=True)
    run.add_argument("--beta", type=float, default=0.05)
    run.add_argument("--gamma", type=float, default=0.75)
    run.add_argument("--rho", type=float, default=0.001)
    run.add_argument("--seed", type=int)
    run.add_argument("--epochs", type=int, default=8)
    run.add_argument("--lr", type=float, default=1e-4)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--init-yaml")
    select = actions.add_parser("select")
    select.add_argument("--root", required=True)
    args = parser.parse_args()
    if args.action == "run":
        run_candidate(args)
    else:
        select_candidate(args)


if __name__ == "__main__":
    main()
