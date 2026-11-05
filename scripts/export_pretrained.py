"""Export only the three archived reference PRCTF checkpoints for inference.

This author-side tool reads existing local experiments; it never trains, selects
seeds, reads observation arrays or includes an optimizer in the release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def export(source_root):
    records = {}
    for dataset in ("industrial", "lbnl", "tep"):
        spec = read(ROOT / f"configs/{dataset}/model.json")
        source = source_root / spec["reference"] / "best.pth"
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        graph = checkpoint["graph"]
        if graph != read(ROOT / f"configs/{dataset}/graph.json"):
            raise ValueError(f"Reference graph mismatch: {dataset}")
        source_config = checkpoint.get("config")
        if source_config is None:
            source_config = read(source.parent / "config.json")["ctf_config"]
        if source_config != spec["config"]:
            raise ValueError(f"Reference configuration mismatch: {dataset}")
        if checkpoint.get("seed", 42) != 42:
            raise ValueError("Only the declared seed-42 reference is exported")
        if dataset != "industrial":
            if checkpoint["scalers"] != read(ROOT / f"configs/{dataset}/scalers.json"):
                raise ValueError(f"Reference calibration mismatch: {dataset}")
        if dataset == "industrial":
            schema = read(ROOT / "configs/industrial/data_schema.json")
            sample_seconds = schema["sample_seconds"]
            targets = list(range(len(graph["channels"])))
            units = "Original industrial workbook units; do not convert before inference."
        else:
            schema = read(ROOT / f"configs/{dataset}/data_metadata.json")
            sample_seconds = schema.get("sampling_seconds", schema.get("sample_seconds"))
            targets = schema.get("target_indices", list(range(len(graph["channels"]))))
            units = ("Prepared SI units: degC, m3/s, W, Pa and dimensionless controls."
                     if dataset == "lbnl" else "Native TEP measurement and manipulated-variable units.")
        if schema["channels"] != graph["channels"]:
            raise ValueError(f"Channel-order mismatch: {dataset}")
        state = checkpoint["model"]
        if not all(isinstance(value, torch.Tensor) for value in state.values()):
            raise TypeError("Expected tensor-only model state")
        history = read(source.parent / "history.json")
        epoch = checkpoint["epoch"]
        selected = next(row for row in history if row["epoch"] == epoch)
        provenance = {
            "seed": 42, "train_fraction": 1.0, "epoch": epoch,
            "selection": "Validation-selected epoch of the predeclared seed-42 reference; no new selection.",
            "source_checkpoint": str(source.relative_to(source_root)),
            "source_sha256": digest(source),
            "validation_mse": selected["val_mse"],
            "protocol": "Archived paper experiment; industrial uses the original full training pool, not the later post-noon retraining.",
            "raw_observations_included": False,
            "optimizer_included": False,
            "fitted_calibration_included": True,
        }
        bundle = {
            "format_version": 1, "dataset": dataset,
            "config": spec["config"], "options": spec["options"],
            "graph": graph, "scalers": checkpoint["scalers"],
            "state_dict": {key: value.detach().cpu().contiguous() for key, value in state.items()},
            "schema": {"channels": graph["channels"], "target_indices": targets,
                       "sampling_seconds": sample_seconds, "units": units,
                       "input_length": spec["config"]["seq_len"],
                       "output_length": spec["config"]["pred_len"]},
            "provenance": provenance,
        }
        path = ROOT / f"weights/{dataset}/prctf_seed42.pt"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, path)
        restored = torch.load(path, map_location="cpu", weights_only=True)
        assert set(restored["state_dict"]) == set(state)
        assert all(torch.equal(restored["state_dict"][key], value.cpu()) for key, value in state.items())
        records[dataset] = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                            "sha256": digest(path), "schema": bundle["schema"],
                            "provenance": provenance}
    manifest = ROOT / "weights/manifest.json"
    manifest.write_text(json.dumps({"format_version": 1, "models": records}, indent=2,
                                   ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({key: {"bytes": row["bytes"], "epoch": row["provenance"]["epoch"]}
                      for key, row in records.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    export(args.source_root.resolve())
