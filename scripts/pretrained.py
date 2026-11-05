"""Load released PRCTF weights and forecast raw-unit historical observations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

if __package__:
    from .model_factory import ROOT, configure_imports
else:
    from model_factory import ROOT, configure_imports


class PRCTFPredictor:
    """One dataset per process, matching the archived module import contract.

    ``predict`` accepts [history, channels] or [batch, history, channels] in the
    checkpoint's physical units. It returns an equally batched NumPy forecast
    in those physical units. No scaler is fitted to deployment observations.
    """

    def __init__(self, model, metadata, device):
        self.model = model.eval().to(device)
        self.metadata = metadata
        self.device = torch.device(device)
        self.channels = list(metadata["schema"]["channels"])
        self.target_indices = list(metadata["schema"]["target_indices"])
        self.output_channels = [self.channels[i] for i in self.target_indices]
        self.input_length = metadata["schema"]["input_length"]
        self.output_length = metadata["schema"]["output_length"]
        self.sample_seconds = metadata["schema"]["sampling_seconds"]
        self.mean = self.model.raw_mean
        self.scale = self.model.raw_scale
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all() or not (self.scale > 0).all():
            raise ValueError("Checkpoint contains invalid normalization statistics")

    @torch.inference_mode()
    def predict(self, values, channels=None, *, targets_only=True):
        x = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        single = x.ndim == 2
        if single:
            x = x.unsqueeze(0)
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.input_length, len(self.channels)) or x.shape[0] == 0:
            raise ValueError(f"Expected [batch, {self.input_length}, {len(self.channels)}] or [{self.input_length}, {len(self.channels)}]")
        if channels is not None:
            channels = list(channels)
            if len(channels) != len(set(channels)) or set(channels) != set(self.channels):
                raise ValueError("Input channel names must match the checkpoint exactly, without duplicates")
            positions = {name: i for i, name in enumerate(channels)}
            x = x[..., [positions[name] for name in self.channels]]
        if not torch.isfinite(x).all():
            raise ValueError("History contains NaN or infinity; missing observations are not filled automatically")
        normalized = (x - self.mean) / self.scale
        output = self.model(normalized, None, None, None) * self.scale + self.mean
        if targets_only:
            output = output[..., self.target_indices]
        if not torch.isfinite(output).all():
            raise RuntimeError("Nonfinite forecast; check units and input operating range")
        result = output.cpu().numpy()
        return result[0] if single else result


def load_pretrained(dataset="industrial", *, device="cpu", ctf_root=None):
    """Verify SHA-256, restore the full trained state and enter inference mode."""
    if dataset not in ("industrial", "lbnl", "tep"):
        raise ValueError("dataset must be industrial, lbnl or tep")
    manifest = json.loads((ROOT / "weights/manifest.json").read_text())
    entry = manifest["models"][dataset]
    path = (ROOT / entry["path"]).resolve()
    if not path.is_relative_to((ROOT / "weights").resolve()) or not path.is_file():
        raise FileNotFoundError("Released weight file is missing; clone the complete repository")
    if path.stat().st_size != entry["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError(f"Weight integrity check failed: {dataset}")
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle["format_version"] != 1 or bundle["dataset"] != dataset or bundle["schema"] != entry["schema"]:
        raise ValueError("Weight metadata does not match the release manifest")
    if bundle["schema"]["channels"] != bundle["graph"]["channels"]:
        raise ValueError("Checkpoint graph and input schema disagree")
    configure_imports(dataset, ctf_root)
    cfg = SimpleNamespace(**bundle["config"])
    if dataset == "industrial":
        from relation_model import PhysicsRelationCTF as Model
    elif dataset == "lbnl":
        from model_v7 import PublicPhysicsCTF as Model
    else:
        from length_model import LengthPhysicsCTF as Model
    model = Model(cfg, bundle["graph"], bundle["scalers"], **bundle["options"])
    model.load_state_dict(bundle["state_dict"], strict=True)
    metadata = {key: value for key, value in bundle.items() if key != "state_dict"}
    return PRCTFPredictor(model, metadata, device)
