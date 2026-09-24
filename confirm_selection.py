"""Choose one physics representation on validation only, then confirm seeds."""
import argparse
import json
import math
import os
import time
from physics import ROOT,dump
from run_experiment import train
import torch

def choose_variant(baseline,candidates):
    eligible={name:sum(math.log(scores[p]/baseline[p]) for p in [50,30])/2
              for name,scores in candidates.items()
              if scores[100]<=1.01*baseline[100] and all(scores[p]<baseline[p] for p in [50,30])}
    return min(eligible,key=eligible.get) if eligible else None

def main(device):
    roots={'anchored':ROOT/'runs/v2','centered':ROOT/'runs/centered'}
    baseline={p:json.loads((ROOT/f'runs/v1/train{p:03d}/CTF_seed42/training.json').read_text())['best_val_mse'] for p in [100,50,30]}
    candidates={name:{p:json.loads((root/f'train{p:03d}/Physics_seed42/training.json').read_text())['best_val_mse'] for p in [100,50,30]}
                for name,root in roots.items()}
    selected=choose_variant(baseline,candidates)
    if selected is None:raise RuntimeError('No validation-qualified physics candidate; diagnose before testing')
    record=dict(selected=selected,root=str(roots[selected]),baseline=baseline,candidates=candidates,
                rule='min mean log relative validation MSE on50/30, both must improve;100%<=+1%; one shared representation',
                confirm_seeds=[43,44],created=time.time(),test_consulted=False)
    lock=ROOT/'runs/representation_selection.json'
    if lock.exists():
        existing=json.loads(lock.read_text())
        if existing['selected']!=selected or existing['candidates']!=candidates:
            raise RuntimeError('Validation selection changed after confirmation began')
    else:dump(lock,record)
    print(json.dumps(record),flush=True)
    options=dict(beta=.05,gamma=.5,rho=.001,representation=selected,lag_mode='physical')
    for seed in [43,44]:
        for ratio in [.5,.3,1.]:
            train('Physics',ratio,seed,roots[selected]/f'train{round(100*ratio):03d}'/f'Physics_seed{seed}',device,options)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');args=p.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    main(args.device)
