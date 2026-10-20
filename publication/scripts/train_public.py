"""Portable Train/Validation launcher; no automatic Test evaluation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from model_factory import ROOT, CORE, build_pair, configure_imports, read


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['lbnl','tep'], required=True)
    p.add_argument('--kind', choices=['CTF','Physics'], required=True)
    p.add_argument('--seed', choices=[42,43,44], type=int, default=42)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out = a.out.resolve()
    for folder in ['workspace','results','configs','figures','manifests']:
        if out.is_relative_to(ROOT/folder):
            raise ValueError('Training must not overwrite code, data, figures or archived evidence')
    configure_imports(a.dataset)
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    spec = read(ROOT/f'configs/{a.dataset}/model.json')
    graph = read(ROOT/f'configs/{a.dataset}/graph.json')
    scalers = read(ROOT/f'configs/{a.dataset}/scalers.json')
    if a.dataset == 'lbnl':
        import train_v7
        engine = train_v7.engine
        # Hash the actual exported source files, not absent historical files.
        def source_hashes():
            paths = list(CORE.rglob('*.py')) + list((ROOT/'workspace/PhysicsGuidedCTF').glob('*.py'))
            ctf = ROOT/'workspace/CTF/ChannelTokenFormer'
            paths += [ctf/name for name in read(ROOT/'manifests/ctf_upstream.json')['files']]
            return {str(q): hashlib.sha256(q.read_bytes()).hexdigest() for q in paths if 'vendor' not in q.parts}
        engine.source_hashes = source_hashes
        engine.build_graph = lambda channels: copy.deepcopy(graph)
        folder = out/f'{a.kind}_seed{a.seed}'
        folder.mkdir(parents=True, exist_ok=True)
        snapshot = folder/'scalers.json'
        if snapshot.exists() and read(snapshot) != scalers:
            raise RuntimeError('Output calibration differs; choose a fresh output directory')
        if not snapshot.exists():
            snapshot.write_text(json.dumps(scalers, indent=2))
        engine.train(a.kind,1.,a.seed,folder,a.device,
                     spec['options'] if a.kind == 'Physics' else {},
                     spec['max_epochs'],spec['learning_rate'])
    else:
        import runner as length_engine
        base = length_engine.base
        base.SEED = a.seed
        base.EPOCHS = spec['max_epochs']
        base.BATCH = 64
        base.HERE = out/f'seed{a.seed}'
        base.HERE.mkdir(parents=True, exist_ok=True)
        length_engine.SEED = a.seed
        base.TEPPhysicsCTF = length_engine.LengthPhysicsCTF
        base.record = length_engine.record
        base.evaluate = length_engine.evaluate
        base.config = lambda c,lr: spec['config']
        datasets = {}
        for split in ['train','val']:
            values = length_engine.values(split,scalers)
            origins = length_engine.origins(len(values),values.shape[1],1 if split == 'train' else 20)
            datasets[split] = length_engine.RealWindows(values,origins,20,20)
        def make_model(trial,g,s):
            c, model, _ = build_pair('tep',a.seed)
            return c if trial['kind'] == 'CTF' else model
        base.make_model = make_model
        trial = dict(name=a.kind,kind=a.kind,group='ratio_20x20',
                     lr=spec['learning_rate'],options=spec['options'] if a.kind == 'Physics' else {})
        base.train_one(trial,datasets,graph,scalers,a.device)
    print('Validation-selected training complete. Test was not evaluated by this launcher.')


if __name__ == '__main__':
    main()
