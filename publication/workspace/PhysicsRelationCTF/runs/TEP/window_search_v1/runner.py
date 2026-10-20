"""Four paired jobs; unchanged data membership, only real contiguous windows."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

HERE=Path(__file__).resolve().parent;PREVIOUS=HERE.parent
sys.path.insert(0,str(PREVIOUS))
import run_experiment as base
from length_model import LengthPhysicsCTF,configuration,pair

DATA=base.DATA;SEED=42;BATCH=64
GROUPS={'ratio_20x20':dict(length=20,horizon=20,patch=5),
        'paper_50x10':dict(length=50,horizon=10,patch=10)}
REFERENCE=PREVIOUS/'development/physics_nominal/best.pth'


class RealWindows(Dataset):
    def __init__(self,values,starts,length,horizon):
        self.values=torch.from_numpy(np.ascontiguousarray(values,dtype=np.float32))
        self.origins=np.asarray(starts,dtype=np.int64);self.length=length;self.horizon=horizon
        if self.origins.ndim!=2 or self.origins.shape[1]!=2 or not len(self.origins):raise ValueError('No valid windows')
        if (self.origins[:,0]<0).any() or (self.origins[:,0]>=len(self.values)).any():raise ValueError('Invalid run')
        if (self.origins[:,1]<length).any() or (self.origins[:,1]+horizon>self.values.shape[1]).any():raise ValueError('Window crosses run boundary')
    def __len__(self):return len(self.origins)
    def __getitem__(self,i):
        r,t=self.origins[i]
        return self.values[r,t-self.length:t],self.values[r,t:t+self.horizon]


def origins(runs,points,stride):
    if runs<=0 or points<70 or stride<=0:raise ValueError('Insufficient real data or invalid stride')
    return np.array([(r,t) for r in range(runs) for t in range(50,points-20+1,stride)],dtype=np.int64)


def physics_spec():
    ck=torch.load(REFERENCE,map_location='cpu',weights_only=False)
    return ck['graph'],ck['scalers']


def canonical_sha(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


def values(split,scalers):
    meta=json.loads((DATA/'metadata.json').read_text())
    v=np.load(DATA/f'{split}_values.npy').astype('float64')
    raw=v*np.array(meta['reference_std'])+np.array(meta['reference_mean'])
    return (raw-np.array(scalers['mean']))/np.array(scalers['scale'])


def setup(group):
    settings=GROUPS[group];g,s=physics_spec();ds={}
    for split in ['train','val']:
        v=values(split,s);o=origins(len(v),v.shape[1],1 if split=='train' else 20)
        ds[split]=RealWindows(v,o,settings['length'],settings['horizon'])
    return ds,g,s


def make_model(spec,graph,scalers):
    base.seed_all(SEED)
    c,p=pair(**GROUPS[spec['group']],graph=graph,scalers=scalers,lr=spec['lr'])
    return c if spec['kind']=='CTF' else p


def record(model,enabled):
    if isinstance(model,LengthPhysicsCTF):
        for layer in model.encoder.layers:layer.self_attention.inner_attention.record=enabled


@torch.no_grad()
def evaluate(model,ds,device,collect=False):
    model.eval();record(model,False)
    se=torch.zeros(ds.horizon,52,dtype=torch.float64,device=device);ae=se.clone();pp=[];yy=[]
    for x,y in DataLoader(ds,batch_size=BATCH,shuffle=False):
        x,y=x.to(device),y.to(device);p=model(x,None,None,None)
        if p.shape!=y.shape or not torch.isfinite(p).all():raise RuntimeError('Invalid prediction')
        e=p.double()-y.double();se+=e.square().sum(0);ae+=e.abs().sum(0)
        if collect:pp.append(p.cpu().numpy());yy.append(y.cpu().numpy())
    mse=(se/len(ds)).cpu().numpy();mae=(ae/len(ds)).cpu().numpy()
    metric=dict(mse=float(mse.mean()),mae=float(mae.mean()),common10_mse=float(mse[:10].mean()),
        common10_mae=float(mae[:10].mean()),per_channel_mse=mse.mean(0).tolist(),
        per_channel_mae=mae.mean(0).tolist(),horizon_mse=mse.mean(1).tolist(),
        horizon_mae=mae.mean(1).tolist(),windows=len(ds))
    return metric,(np.concatenate(pp),np.concatenate(yy)) if collect else None


def train_group(group,device):
    root=HERE/group;root.mkdir(exist_ok=True)
    settings=GROUPS[group];ds,g,s=setup(group)
    meta=json.loads((DATA/'metadata.json').read_text())
    manifest=dict(group=group,**settings,seed=SEED,sample_seconds=180,
        ids=meta['ids'],data_hashes={str(p):base.sha(p) for p in DATA.glob('*.npy')},
        physics_reference=str(REFERENCE),physics_reference_sha256=base.sha(REFERENCE),
        graph_sha256=canonical_sha(g),scalers_sha256=canonical_sha(s),
        graph_features=len(g['features']),graph_edges=len(g['edges']),
        windows={k:len(v) for k,v in ds.items()},
        origin_sha256={k:hashlib.sha256(v.origins.tobytes()).hexdigest() for k,v in ds.items()},
        patching='exact real observations only; no padding, interpolation, duplication, or truncation',
        train_stride=1,evaluation_stride=20,common_origin_start=50,common_max_horizon=20,
        excluded_target_prefix_points=50,validation_tail_points=500-(470+settings['horizon']),
        test_tail_points=960-(930+settings['horizon']),
        note='Source data unchanged; derived origins synchronized across both tasks. Original test already exposed.')
    base.dump(root/'data_manifest.json',manifest);base.dump(root/'physics_graph.json',g);base.dump(root/'scalers.json',s)
    base.HERE=root;base.EPOCHS=16;base.BATCH=BATCH;base.make_model=make_model
    base.TEPPhysicsCTF=LengthPhysicsCTF;base.record=record;base.evaluate=evaluate
    base.config=lambda c,lr:vars(configuration(**settings,lr=lr))
    for kind in ['CTF','Physics']:
        spec=dict(name=kind,group=group,kind=kind,lr=3e-4,
            options={} if kind=='CTF' else dict(beta=.05,gamma=.75,rho=.001,level_mode='absolute',closure=0.))
        base.train_one(spec,ds,g,s,device)
    base.dump(root/'status.json',dict(state='training_complete',test_evaluated=False))


def selection():
    result={}
    for group in GROUPS:
        result[group]={}
        for kind in ['CTF','Physics']:
            path=HERE/group/'development'/kind/'training.json'
            if not path.exists():raise RuntimeError('All four jobs must finish before test evaluation')
            t=json.loads(path.read_text());result[group][kind]=dict(best_epoch=t['best_epoch'],validation=t['validation'],parameters=t['parameters'])
        c,p=[result[group][k]['validation'] for k in ['CTF','Physics']]
        result[group]['val_gain_percent']=100*(c['mse']-p['mse'])/c['mse']
    result['preferred_on_validation_relative_gain']=max(GROUPS,key=lambda k:result[k]['val_gain_percent'])
    base.dump(HERE/'validation_selection.json',result)
    return result


def evaluate_final(device):
    dest=HERE/'final';dest.mkdir(exist_ok=True)
    if (dest/'summary.json').exists():return json.loads((dest/'summary.json').read_text())
    selected=selection();oldlock=json.loads((PREVIOUS/'final/frozen_test_manifest.json').read_text())
    paths=[Path(p) for p in oldlock['code']]+list(HERE.glob('*.py'))
    lock=dict(selection=selected,code={str(p):base.sha(p) for p in paths},
        checkpoints={str(p):base.sha(p) for p in HERE.glob('*/development/*/best.pth')},
        reference_sha256=base.sha(REFERENCE),data={str(p):base.sha(p) for p in DATA.glob('*.npy')},
        groups=GROUPS,test='original fixed runs401–500, previously exposed; no test-based tuning')
    lockpath=dest/'frozen_manifest.json'
    if lockpath.exists():
        if json.loads(lockpath.read_text())!=lock:raise RuntimeError('Frozen protocol changed')
    else:base.dump(lockpath,lock)
    graph,scalers=physics_spec();v=values('test',scalers);o=origins(len(v),v.shape[1],20)
    results={};comparisons={}
    for group,settings in GROUPS.items():
        ds=RealWindows(v,o,settings['length'],settings['horizon']);results[group]={}
        for kind in ['CTF','Physics']:
            ck=torch.load(HERE/group/'development'/kind/'best.pth',map_location='cpu',weights_only=False)
            model=make_model(ck['spec'],ck['graph'],ck['scalers']).to(device);model.load_state_dict(ck['model'])
            metric,arr=evaluate(model,ds,device,True);out=dest/group/kind;out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out/'predictions.npz',prediction=arr[0],truth=arr[1],origins=o,
                mean=np.array(scalers['mean']),scale=np.array(scalers['scale']),channels=np.array(graph['channels']))
            base.dump(out/'metrics.json',metric);results[group][kind]=metric
            del model;torch.cuda.empty_cache()
        c,p=[results[group][k] for k in ['CTF','Physics']]
        comparisons[group]=dict(ctf_mse=c['mse'],physics_mse=p['mse'],ctf_mae=c['mae'],physics_mae=p['mae'],
            mse_gain_percent=100*(c['mse']-p['mse'])/c['mse'],mae_gain_percent=100*(c['mae']-p['mae'])/c['mae'],
            common10_ctf_mse=c['common10_mse'],common10_physics_mse=p['common10_mse'],
            common10_gain_percent=100*(c['common10_mse']-p['common10_mse'])/c['common10_mse'],
            improved_channels=int((np.array(c['per_channel_mse'])>np.array(p['per_channel_mse'])).sum()))
    summary=dict(groups=GROUPS,selection=selected,comparisons=comparisons,results=results,
        limitations=['Single seed, fault-free simulations only.','Test was exposed in previous research; not a new blind generalization test.',
            'Different full horizons are different tasks; compare paired gains and common10 at identical origins.'])
    base.dump(dest/'summary.json',summary)
    print(json.dumps(comparisons),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group',choices=list(GROUPS));parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--stage',choices=['train','test','select'],default='train');a=parser.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    if a.stage=='train':train_group(a.group,a.device)
    elif a.stage=='test':evaluate_final(a.device)
    else:print(json.dumps(selection()))


if __name__=='__main__':main()
