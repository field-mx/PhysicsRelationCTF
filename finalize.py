"""Freeze train/validation-selected checkpoints and guard the one-time test phase."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from physics import ROOT,OLD,dump
from run_experiment import test

def fingerprints(paths):
    return {str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}

def validate_manifest(manifest):
    for path,expected in manifest['files'].items():
        if not Path(path).exists() or fingerprints([path])[path]!=expected:
            raise RuntimeError('Frozen implementation/data changed: '+path)

def run_paths(ratios,seeds,baseline_root,physics_root):
    paths=[]
    for ratio in ratios:
        for seed in seeds:
            paths.extend([
                Path(baseline_root).resolve()/f'train{round(ratio*100):03d}'/f'CTF_seed{seed}',
                Path(physics_root).resolve()/f'train{round(ratio*100):03d}'/f'Physics_seed{seed}',
            ])
    return paths

def freeze(root,baseline_root,physics_root,seeds,ratios=(1.,.5,.3)):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'selection.json').exists():raise RuntimeError('Selection already frozen; do not overwrite')
    paths=run_paths(ratios,seeds,baseline_root,physics_root);validation=[]
    for p in paths:
        info=json.loads((p/'training.json').read_text());cfg=json.loads((p/'config.json').read_text())
        kind=info['kind']
        relevant=['physics.py','run_experiment.py']+(['relation_model.py'] if kind=='Physics' else [])
        for source in relevant:
            if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=cfg['source_hashes'][source]:
                raise RuntimeError(f'Training implementation changed: {source} in {p}')
        validation.append(dict(path=str(p),**info))
    # Lock native implementation, data source/cache/splits, graph mapping and
    # proposed implementation in addition to checkpoint bytes.
    files=list(ROOT.glob('*.py'))+[ROOT/'PLAN.md']
    files+=list((OLD/'data/industrial').glob('*.npy'))+list((OLD/'data/industrial').glob('*.json'))
    files+=[OLD/'model.py',OLD/'experiment.py',ROOT.parent/'CIST-MAE/data/SensorData.xlsx',
            ROOT.parent/'GNN_Transformer/equation_topology.py',ROOT.parent/'GNN_Transformer/equation_topology_v2.py',
            ROOT.parent/'GNN_Transformer/runs/sensor_96s_96s_stride96/data_summary.json']
    ctf=ROOT.parent/'CTF/ChannelTokenFormer'
    files+=list((ctf/'layers').glob('*.py'))+[ctf/'models/ChannelTokenFormer.py']
    files+=[p/'best.pth' for p in paths]
    result=dict(created=time.time(),seeds=seeds,ratios=list(ratios),validation=validation,
                policy='All candidates/checkpoints selected using validation only. Fixed previously exposed research holdout.',
                checkpoints={str(p):hashlib.sha256((p/'best.pth').read_bytes()).hexdigest() for p in paths},
                files=fingerprints(files))
    dump(root/'selection.json',result);return result

def evaluate_frozen(root,device):
    root=Path(root);manifest=json.loads((root/'selection.json').read_text());validate_manifest(manifest)
    rows=[]
    for path in manifest['checkpoints']:
        result=test(path,device,root/'selection.json');rows.append(dict(path=path,**result))
        print(json.dumps({k:result[k] for k in ['kind','ratio','seed','mse','mae']}),flush=True)
    summary=[]
    for ratio in manifest['ratios']:
        values={}
        for kind in ['CTF','Physics']:
            selected=[r for r in rows if r['ratio']==ratio and r['kind']==kind]
            values[kind]={metric+'_mean':float(np.mean([r[metric] for r in selected])) for metric in ['mse','mae']}
            values[kind].update({metric+'_std':float(np.std([r[metric] for r in selected],ddof=1)) if len(selected)>1 else None
                                for metric in ['mse','mae']})
        values['ratio']=ratio;values['mse_relative_improvement_pct']=100*(1-values['Physics']['mse_mean']/values['CTF']['mse_mean'])
        summary.append(values)
    result=dict(summary=summary,rows=rows,all_required_seeds=manifest['seeds'])
    dump(root/'summary.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','evaluate'])
    p.add_argument('--root',default=str(ROOT/'runs/final'))
    p.add_argument('--baseline-root',default=str(ROOT/'runs/v1'));p.add_argument('--physics-root',default=str(ROOT/'runs/v2'))
    p.add_argument('--seeds',type=int,nargs='+',default=[42]);p.add_argument('--device',default='cuda:0')
    p.add_argument('--ratios',type=float,nargs='+',default=[1.,.5,.3])
    a=p.parse_args();os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    if a.action=='freeze':freeze(a.root,a.baseline_root,a.physics_root,a.seeds,a.ratios)
    else:evaluate_frozen(a.root,a.device)
