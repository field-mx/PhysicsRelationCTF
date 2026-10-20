"""One-seed, no-search CTF/Physics comparison under train-only time gaps."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from physics import ROOT, dump, get_data, nested_time_gaps
from run_experiment import test, train


RATIOS = (1.0, 0.5, 0.3)
MODEL_SEED = 42
MASK_SEED = 20260922
EPOCHS = 16
LEARNING_RATE = 1e-4
PHYSICS_OPTIONS = dict(beta=0.05, gamma=0.75, rho=0.001,
                       representation="centered", lag_mode="physical")


def summarize(metrics):
    result = {}
    for ratio in RATIOS:
        ctf, physics = metrics[ratio]["CTF"], metrics[ratio]["Physics"]
        full_ctf = metrics[1.0]["CTF"]
        full_physics = metrics[1.0]["Physics"]
        result[ratio] = dict(
            ctf_mse=ctf["mse"], physics_mse=physics["mse"],
            ctf_mae=ctf["mae"], physics_mae=physics["mae"],
            mse_gain=(ctf["mse"] - physics["mse"]) / ctf["mse"],
            mae_gain=(ctf["mae"] - physics["mae"]) / ctf["mae"],
            ctf_mse_degradation=ctf["mse"] / full_ctf["mse"] - 1,
            physics_mse_degradation=physics["mse"] / full_physics["mse"] - 1,
        )
    return result


def configure_determinism():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def job_path(root, ratio, kind):
    return root / f"train{int(ratio * 100):03d}" / f"{kind}_seed{MODEL_SEED}"


def data_protocol(root):
    full, _, meta, _, _ = get_data(1.0, sampling="random_gaps", subset_seed=MASK_SEED)
    end = int(meta["train_end"])
    masks = nested_time_gaps(end, MASK_SEED)
    expected = {1.0: end, 0.5: end // 2, 0.3: end * 3 // 10}
    if not np.all(masks[0.3] <= masks[0.5]):
        raise RuntimeError("The 30% observation set is not nested in the 50% set")
    report = {}
    values = full["train"].values[0][:end]
    for ratio in RATIOS:
        data, _, _, selection, _ = get_data(ratio, sampling="random_gaps", subset_seed=MASK_SEED)
        if selection["rows"] != expected[ratio]:
            raise RuntimeError(f"Incorrect observed-point count for {ratio}")
        for split in ("val", "test"):
            if not np.array_equal(data[split].origins, full[split].origins):
                raise RuntimeError(f"The {split} windows changed")
        mask = masks[ratio]
        report[str(ratio)] = dict(
            observed_points=int(mask.sum()), training_windows=len(data["train"]),
            retained_segments=selection["retained_segments"],
            deleted_segments=selection["deleted_segments"],
            mask_sha256=selection["mask_sha256"],
            time_decile_retention=[float(piece.mean()) for piece in np.array_split(mask, 10)],
        )
        histograms = {}
        for name in ("SCRDLFK", "TT010002", "PT010002", "FT010002"):
            if name not in meta["channels"]:
                continue
            series = values[:, meta["channels"].index(name)]
            bins = np.unique(np.quantile(series, np.linspace(0, 1, 6)))
            histograms[name] = np.histogram(series[mask], bins=bins)[0].tolist()
        report[str(ratio)]["train_only_quantile_bin_counts"] = histograms
    manifest = dict(
        aim="true-observation-scarcity, paired 96->96 forecasts",
        protocol="fixed nested train-only time gaps; no validation/test based mask selection",
        model_seed=MODEL_SEED, mask_seed=MASK_SEED, ratios=RATIOS,
        epochs=EPOCHS, learning_rate=LEARNING_RATE,
        physics_options=PHYSICS_OPTIONS,
        source_sha256={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("time_gap_sampling.py", "random_gap_data.py", "physics.py",
                         "relation_model.py", "run_experiment.py", "run_random_gap_experiment.py")
        },
        data=report,
        limitation="Random gaps do not guarantee coverage of all unlabelled operating regimes",
    )
    path = root / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError("Existing experiment has a different frozen protocol")
    else:
        dump(path, manifest)
        np.savez_compressed(root / "train_masks.npz", **{
            f"keep_{int(ratio * 100)}": masks[ratio] for ratio in RATIOS
        })
    return manifest


def run_worker(root, kind, device):
    configure_determinism()
    options = PHYSICS_OPTIONS if kind == "Physics" else {}
    for ratio in RATIOS:
        output = job_path(root, ratio, kind)
        result = train(kind, ratio, MODEL_SEED, output, device, options,
                       epochs=EPOCHS, lr=LEARNING_RATE,
                       sampling="random_gaps", subset_seed=MASK_SEED)
        print(json.dumps(dict(kind=kind, ratio=ratio,
                              best_val_mse=result["best_val_mse"])), flush=True)


def run_experiment(root):
    configure_determinism()
    root.mkdir(parents=True, exist_ok=True)
    manifest = data_protocol(root)
    if not manifest["source_sha256"]:
        raise RuntimeError("Missing source provenance")
    workers = []
    for kind, device in (("CTF", "cuda:0"), ("Physics", "cuda:1")):
        log = open(root / f"{kind.lower()}_train.log", "a", buffering=1)
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--worker-kind", kind, "--root", str(root), "--device", device]
        workers.append((kind, subprocess.Popen(command, stdout=log,
                                                stderr=subprocess.STDOUT), log))
    failed = []
    for kind, process, log in workers:
        code = process.wait()
        log.close()
        if code:
            failed.append((kind, code))
    if failed:
        raise RuntimeError(f"Training failed; no test set was opened: {failed}")
    checkpoints = {}
    for ratio in RATIOS:
        for kind in ("CTF", "Physics"):
            output = job_path(root, ratio, kind)
            if not (output / "training.json").exists():
                raise RuntimeError(f"Missing validation-selected training record: {output}")
            checkpoint = output / "best.pth"
            checkpoints[str(output.resolve())] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    lock_path = root / "selection_lock.json"
    lock = dict(checkpoints=checkpoints, policy="all six validation-selected checkpoints frozen before first test")
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != lock:
            raise RuntimeError("Frozen checkpoint lock differs; refusing test evaluation")
    else:
        dump(lock_path, lock)
    scores = {}
    for ratio in RATIOS:
        scores[ratio] = {}
        for kind in ("CTF", "Physics"):
            scores[ratio][kind] = test(job_path(root, ratio, kind), "cuda:0", lock_path)
    summary = summarize(scores)
    dump(root / "summary.json", {
        "status": "complete", "single_model_seed": MODEL_SEED,
        "mask_seed": MASK_SEED, "test_policy": "once after all six checkpoints locked",
        "results": {str(ratio): row for ratio, row in summary.items()},
    })
    print(json.dumps({str(ratio): row for ratio, row in summary.items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "runs" / "random_gaps_v1")
    parser.add_argument("--worker-kind", choices=("CTF", "Physics"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.worker_kind:
        run_worker(args.root, args.worker_kind, args.device)
    else:
        run_experiment(args.root)
