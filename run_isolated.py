"""Train the gradient-isolation candidate without touching previous runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from physics import ROOT,dump
import run_experiment as engine
from isolated_model import IsolatedPhysicsCTF

def main(ratios,seeds,device):
    # Explicit factory injection. Inference is numerically identical to the
    # parent class; only gradient routing differs during training.
    engine.PhysicsRelationCTF=IsolatedPhysicsCTF
    manifest=dict(hypothesis='isolate extra physics gradient to shared patch projection',
                  forward_equivalent=True,base_representation='centered',
                  files={str(ROOT/f):hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                         for f in ['isolated_model.py','run_isolated.py','relation_model.py','physics.py','run_experiment.py']})
    options=dict(beta=.05,gamma=.5,rho=.001,representation='centered',lag_mode='physical')
    for seed in seeds:
        for ratio in ratios:
            out=ROOT/f'runs/isolated/train{round(100*ratio):03d}/Physics_seed{seed}'
            out.mkdir(parents=True,exist_ok=True);path=out/'implementation.json'
            if path.exists() and json.loads(path.read_text())!=manifest:raise RuntimeError('Implementation changed')
            if not path.exists():dump(path,manifest)
            engine.train('Physics',ratio,seed,out,device,options)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ratios',nargs='+',type=float,default=[.5,.3])
    p.add_argument('--seeds',nargs='+',type=int,default=[43]);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    main(args.ratios,args.seeds,args.device)
