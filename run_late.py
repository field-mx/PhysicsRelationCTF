"""Last-layer physics pilot with unchanged controls and no test evaluation."""
import argparse
import hashlib
import json
import os
import torch
from physics import ROOT,dump
import run_experiment as engine
from late_model import LatePhysicsCTF

def main(ratios,seeds,device):
    engine.PhysicsRelationCTF=LatePhysicsCTF
    manifest=dict(hypothesis='restrict physics injection to final CTF layer to prevent forward spread outside physical receivers',
                  unchanged='centered features and empirical gains; graph, lag, head, main architecture; no projection isolation',
                  tradeoff='no early-layer physics-specific multihop propagation',
                  files={str(ROOT/f):hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                         for f in ['late_model.py','run_late.py','relation_model.py','physics.py','run_experiment.py']})
    options=dict(beta=.05,gamma=.5,rho=.001,representation='centered',lag_mode='physical')
    for seed in seeds:
        for ratio in ratios:
            out=ROOT/f'runs/late/train{round(100*ratio):03d}/Physics_seed{seed}'
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
