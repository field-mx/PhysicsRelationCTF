"""One-factor formula-normalization pilot; reuse existing paired controls."""
import argparse
import hashlib
import json
import os
import torch
from physics import ROOT,dump
import run_experiment as engine
from equation_normalized_model import EquationNormalizedPhysicsCTF

def main(ratios,seeds,device):
    engine.PhysicsRelationCTF=EquationNormalizedPhysicsCTF
    manifest=dict(hypothesis='remove empirical-correlation magnitude from deterministic formula-feature scaling',
                  unchanged='empirical sign, lag support, graph, residual, native CTF; no gradient isolation',
                  files={str(ROOT/f):hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                         for f in ['equation_normalized_model.py','run_equation_normalized.py','relation_model.py','physics.py','run_experiment.py']})
    options=dict(beta=.05,gamma=.5,rho=.001,representation='centered',lag_mode='physical')
    for seed in seeds:
        for ratio in ratios:
            out=ROOT/f'runs/equation_normalized/train{round(100*ratio):03d}/Physics_seed{seed}'
            out.mkdir(parents=True,exist_ok=True);p=out/'implementation.json'
            if p.exists() and json.loads(p.read_text())!=manifest:raise RuntimeError('Implementation changed')
            if not p.exists():dump(p,manifest)
            engine.train('Physics',ratio,seed,out,device,options)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ratios',nargs='+',type=float,default=[.5,.3])
    p.add_argument('--seeds',nargs='+',type=int,default=[43]);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    main(args.ratios,args.seeds,args.device)
