"""Checkpoint-free constructors; original mathematical model source is reused."""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'workspace'
CORE = WORK / 'PhysicsRelationCTF'
ACTIVE_DATASET = None


def read(path):
    return json.loads(Path(path).read_text())


def configure_imports(dataset, ctf_root=None):
    global ACTIVE_DATASET
    if ACTIVE_DATASET not in (None, dataset):
        raise RuntimeError('Use a fresh process for each dataset to isolate historical module names.')
    ACTIVE_DATASET = dataset
    ctf = Path(ctf_root).resolve() if ctf_root else WORK/'CTF/ChannelTokenFormer'
    if not (ctf/'models/ChannelTokenFormer.py').is_file():
        raise FileNotFoundError('Obtain pinned CTF first: python scripts/fetch_upstreams.py --ctf')
    import hashlib
    for name, expected in read(ROOT/'manifests/ctf_upstream.json')['files'].items():
        actual = hashlib.sha256((ctf/name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Upstream CTF mismatch: {name}')
    paths = [str(CORE), str(WORK/'PhysicsGuidedCTF'), str(ctf)]
    if dataset in ('lbnl','tep'):
        pub = CORE/'runs/publicdata_exp'
        paths += [str(pub/'optimization_v2'), str(pub)]
    if dataset == 'tep':
        paths += [str(CORE/'runs/TEP/window_search_v1'), str(CORE/'runs/TEP')]
    sys.path[:0] = paths


def build_pair(dataset, seed=42, ctf_root=None):
    """Return matching-initialization OriginalCTF and PRCTF in a fresh process.

    Industrial statistics below are synthetic shape-test inputs only. Public
    graph/scalers are fixed train-calibrated snapshots, not trained weights.
    """
    configure_imports(dataset, ctf_root)
    import numpy as np
    import torch
    from physics import OriginalCTF
    from relation_model import copy_ctf
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    spec = read(ROOT/f'configs/{dataset}/model.json')
    cfg = SimpleNamespace(**spec['config'])
    graph = read(ROOT/f'configs/{dataset}/graph.json')
    base = OriginalCTF(cfg)
    if dataset == 'industrial':
        from physics import FEATURES
        from relation_model import PhysicsRelationCTF
        c = len(graph['channels'])
        scalers = dict(raw_mean=[0.]*c, raw_scale=[1.]*c,
                       eq_mean=[0.]*len(FEATURES), eq_scale=[1.]*len(FEATURES),
                       gains=[1.]*len(graph['edges']))
        model = PhysicsRelationCTF(cfg, graph, scalers, **spec['options'])
    elif dataset == 'lbnl':
        from model_v7 import PublicPhysicsCTF
        model = PublicPhysicsCTF(cfg, graph, read(ROOT/'configs/lbnl/scalers.json'), **spec['options'])
    elif dataset == 'tep':
        from length_model import LengthPhysicsCTF
        model = LengthPhysicsCTF(cfg, graph, read(ROOT/'configs/tep/scalers.json'), **spec['options'])
    else:
        raise ValueError(dataset)
    copy_ctf(model, base)
    return base, model, spec
