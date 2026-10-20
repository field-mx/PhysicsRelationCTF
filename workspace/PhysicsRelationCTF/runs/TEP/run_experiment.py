"""Auditable validation-only development and frozen paired TEP testing."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from tep_model import OriginalCTF,TEPPhysicsCTF,config,copy_ctf
from tep_physics import build_graph,calibrate

HERE=Path(__file__).resolve().parent
DATA=HERE.parents[2]/'PhysicsGuidedCTF/data/tep'
SEED=42
EPOCHS=16
BATCH=64
TRIALS=[
    dict(name='ctf_lr3e4',kind='CTF',lr=3e-4,options={}),
    dict(name='physics_nominal',kind='Physics',lr=3e-4,options=dict(beta=.05,gamma=.75,level_mode='absolute')),
    dict(name='physics_centered',kind='Physics',lr=3e-4,options=dict(beta=.05,gamma=.15,level_mode='centered')),
    dict(name='physics_closure',kind='Physics',lr=3e-4,options=dict(beta=.05,gamma=.15,level_mode='centered',closure=.25)),
    dict(name='ctf_lr1e4',kind='CTF',lr=1e-4,options={}),
    dict(name='physics_closure_lr1e4',kind='Physics',lr=1e-4,options=dict(beta=.05,gamma=.15,level_mode='centered',closure=.25)),
]


def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))
    tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


class Windows(Dataset):
    def __init__(self,values,origins):
        self.values=torch.from_numpy(np.ascontiguousarray(values,dtype=np.float32))
        self.origins=np.asarray(origins,dtype=np.int64)
    def __len__(self):return len(self.origins)
    def __getitem__(self,i):
        r,t=self.origins[i]
        return self.values[r,t-96:t],self.values[r,t:t+96]


def load_development():
    meta=json.loads((DATA/'metadata.json').read_text())
    refmean=np.asarray(meta['reference_mean']);refscale=np.asarray(meta['reference_std'])
    train=np.load(DATA/'train_values.npy').astype('float64')*refscale+refmean
    val=np.load(DATA/'val_values.npy').astype('float64')*refscale+refmean
    mean=train.reshape(-1,52).mean(0);scale=train.reshape(-1,52).std(0)
    if (scale<1e-8).any():raise ValueError('Unexpected constant TEP channel')
    graph=build_graph(meta['channels'])
    scalers,quality=calibrate(train.reshape(-1,52),graph,mean,scale)
    ds={s:Windows((v-mean)/scale,np.load(DATA/f'{s}_origins.npy')) for s,v in [('train',train),('val',val)]}
    manifest=dict(dataset='TEP fault-free; 100% of fixed cached training pool',
        train_run_ids=meta['ids']['train'],val_run_ids=meta['ids']['val'],test_run_ids=meta['ids']['test'],
        train_runs=len(train),train_points_per_run=int(train.shape[1]),
        input_points=96,output_points=96,sample_seconds=180,input_hours=4.8,output_hours=4.8,
        train_windows=len(ds['train']),val_windows=len(ds['val']),train_stride=1,evaluation_stride=96,
        seed=SEED,batch_size=BATCH,max_epochs=EPOCHS,
        target_channels=meta['channels'],metric='all 52 channels/all 96 horizons, full-train standardized MSE and MAE',
        data_hashes={str(p):sha(p) for p in DATA.glob('*.npy')},
        note='Test arrays are hashed but not loaded during development. Existing topology-only TEP results are not used for selection.')
    dump(HERE/'data_manifest.json',manifest);dump(HERE/'physics_graph.json',graph)
    dump(HERE/'equation_train_diagnostics.json',quality);dump(HERE/'scalers.json',scalers)
    return ds,graph,scalers,meta


def make_model(spec,graph,scalers):
    seed_all(SEED)
    cfg=config(len(graph['channels']),lr=spec['lr'])
    base=OriginalCTF(SimpleNamespace(**cfg))
    if spec['kind']=='CTF':return base
    model=TEPPhysicsCTF(SimpleNamespace(**cfg),graph,scalers,**spec['options'])
    copy_ctf(model,base)
    return model


def record(model,enabled):
    if isinstance(model,TEPPhysicsCTF):
        for layer in model.encoder.layers:layer.self_attention.inner_attention.record=enabled


@torch.no_grad()
def evaluate(model,ds,device,collect=False):
    model.eval();record(model,False)
    se=torch.zeros(96,52,dtype=torch.float64,device=device);ae=se.clone();pp=[];yy=[]
    for x,y in DataLoader(ds,batch_size=BATCH,shuffle=False):
        x,y=x.to(device),y.to(device);p=model(x,None,None,None)
        e=p.double()-y.double();se+=e.square().sum(0);ae+=e.abs().sum(0)
        if collect:pp.append(p.cpu().numpy());yy.append(y.cpu().numpy())
    mse=(se/len(ds)).cpu().numpy();mae=(ae/len(ds)).cpu().numpy()
    metrics=dict(mse=float(mse.mean()),mae=float(mae.mean()),
        per_channel_mse=mse.mean(0).tolist(),per_channel_mae=mae.mean(0).tolist(),
        horizon_mse=mse.mean(1).tolist(),windows=len(ds))
    return metrics,(np.concatenate(pp),np.concatenate(yy)) if collect else None


def train_one(spec,ds,graph,scalers,device):
    out=HERE/'development'/spec['name'];out.mkdir(parents=True,exist_ok=True)
    if (out/'training.json').exists():return json.loads((out/'training.json').read_text())
    dump(out/'config.json',dict(spec,seed=SEED,epochs=EPOCHS,batch_size=BATCH,ctf_config=config(52,lr=spec['lr'])))
    model=make_model(spec,graph,scalers).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=spec['lr'])
    best=float('inf');stale=0;history=[];start=1;tic=time.monotonic()
    if (out/'resume.pth').exists():
        ck=torch.load(out/'resume.pth',map_location=device,weights_only=False)
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        best,stale,history,start=ck['best'],ck['stale'],ck['history'],ck['epoch']+1
    for epoch in range(start,EPOCHS+1):
        if epoch>10 and stale>=4:break
        seed_all(SEED*1000+epoch)
        order=np.random.default_rng(SEED*1000+epoch).permutation(len(ds['train'])).tolist()
        loader=DataLoader(ds['train'],batch_size=BATCH,sampler=order,num_workers=0)
        lr=spec['lr'] if epoch<=6 else spec['lr']*(.1+.9*.5*(1+math.cos(math.pi*(epoch-6)/(EPOCHS-6))))
        for g in optimizer.param_groups:g['lr']=lr
        model.train();record(model,False);total=0.;n=0;epoch_tic=time.monotonic()
        for step,(x,y) in enumerate(loader):
            x,y=x.to(device),y.to(device);optimizer.zero_grad(set_to_none=True)
            p=model(x,None,None,None);loss=(p-y).square().mean()
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss: {spec["name"]}, epoch {epoch}')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(x);n+=len(x)
            if step%100==0:
                dump(HERE/'status.json',dict(state='training',trial=spec['name'],epoch=epoch,max_epochs=EPOCHS,
                    step=step,steps=len(loader),train_loss=total/n,best_val_mse=None if best==float('inf') else best,updated=time.time()))
        val,_=evaluate(model,ds['val'],device)
        diag=[]
        if isinstance(model,TEPPhysicsCTF):
            record(model,True)
            with torch.no_grad():model(torch.stack([ds['val'][i][0] for i in range(4)]).to(device),None,None,None)
            diag=[l.self_attention.inner_attention.diagnostics for l in model.encoder.layers]
            record(model,False)
        row=dict(epoch=epoch,train_mse=total/n,val_mse=val['mse'],val_mae=val['mae'],lr=lr,
            seconds=time.monotonic()-epoch_tic,diagnostics=diag)
        history.append(row)
        if val['mse']<best:
            best=val['mse'];stale=0
            torch.save(dict(model=model.state_dict(),spec=spec,graph=graph,scalers=scalers,epoch=epoch),out/'best.pth')
        else:stale+=1
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),best=best,stale=stale,
            history=history,epoch=epoch),out/'resume.pth')
        dump(out/'history.json',history)
        print(json.dumps(dict(trial=spec['name'],**{k:v for k,v in row.items() if k!='diagnostics'})),flush=True)
    ck=torch.load(out/'best.pth',map_location=device,weights_only=False);model.load_state_dict(ck['model'])
    metrics,arr=evaluate(model,ds['val'],device,True)
    np.savez_compressed(out/'validation_predictions.npz',prediction=arr[0],truth=arr[1],origins=ds['val'].origins)
    result=dict(spec=spec,best_epoch=ck['epoch'],epochs_completed=len(history),validation=metrics,
        total_seconds=time.monotonic()-tic,parameters=sum(p.numel() for p in model.parameters()),test_evaluated=False)
    dump(out/'training.json',result)
    del model,optimizer;torch.cuda.empty_cache()
    return result


def selection():
    all_results=[json.loads(p.read_text()) for p in (HERE/'development').glob('*/training.json')]
    ctf=[r for r in all_results if r['spec']['kind']=='CTF']
    phy=[r for r in all_results if r['spec']['kind']=='Physics']
    if not ctf or not phy:raise RuntimeError('Need both model families')
    best_ctf=min(ctf,key=lambda r:r['validation']['mse'])
    best_phy=min(phy,key=lambda r:r['validation']['mse'])
    # A matched-optimizer pair also accompanies the strongest-baseline comparison.
    matched=[r for r in ctf if r['spec']['lr']==best_phy['spec']['lr']]
    if not matched:raise RuntimeError('Missing matched baseline')
    sel=dict(ctf=best_ctf['spec']['name'],physics=best_phy['spec']['name'],
        matched_ctf=min(matched,key=lambda r:r['validation']['mse'])['spec']['name'],
        val_gain_percent=100*(best_ctf['validation']['mse']-best_phy['validation']['mse'])/best_ctf['validation']['mse'],
        candidates=[dict(name=r['spec']['name'],**{k:r['validation'][k] for k in ['mse','mae']},best_epoch=r['best_epoch']) for r in all_results])
    dump(HERE/'validation_selection.json',sel);return sel


def freeze_and_test(device):
    sel=selection()
    final=HERE/'final';final.mkdir(exist_ok=True)
    if (final/'summary.json').exists():return json.loads((final/'summary.json').read_text())
    names=list(dict.fromkeys([sel['ctf'],sel['matched_ctf'],sel['physics']]))
    sources=list(HERE.glob('*.py'))
    # Include all reused core primitives, not just the new adapter.
    core=HERE.parents[1];pub=core/'runs/publicdata_exp';old=core.parent/'PhysicsGuidedCTF'
    sources += [core/'physics.py',core/'relation_model.py',pub/'public_model_v2.py',
        pub/'optimization_v2/model_v5.py',old/'model.py',old/'experiment.py']
    ctf=core.parent/'CTF/ChannelTokenFormer'
    sources += list((ctf/'models').glob('*.py'))+list((ctf/'layers').glob('*.py'))
    lock=dict(selection=sel,created=time.time(),code={str(p):sha(p) for p in sources},
        checkpoints={str(HERE/'development'/n/'best.pth'):sha(HERE/'development'/n/'best.pth') for n in names},
        data=json.loads((HERE/'data_manifest.json').read_text())['data_hashes'])
    lockpath=final/'frozen_test_manifest.json'
    if lockpath.exists():
        oldlock=json.loads(lockpath.read_text())
        for key in ['code','checkpoints','data']:
            if oldlock[key]!=lock[key]:raise RuntimeError('Frozen artifact changed; do not retune against test')
    else:dump(lockpath,lock)
    # First materialized test targets of this experiment occur after freezing.
    meta=json.loads((DATA/'metadata.json').read_text());sc=json.loads((HERE/'scalers.json').read_text())
    raw=np.load(DATA/'test_values.npy').astype('float64')*np.asarray(meta['reference_std'])+np.asarray(meta['reference_mean'])
    ds=Windows((raw-np.asarray(sc['mean']))/np.asarray(sc['scale']),np.load(DATA/'test_origins.npy'))
    result={}
    for name in names:
        ck=torch.load(HERE/'development'/name/'best.pth',map_location='cpu',weights_only=False)
        model=make_model(ck['spec'],ck['graph'],ck['scalers']).to(device);model.load_state_dict(ck['model'])
        metrics,arr=evaluate(model,ds,device,True)
        dest=final/name;dest.mkdir(exist_ok=True)
        dump(dest/'metrics.json',metrics)
        np.savez_compressed(dest/'predictions.npz',prediction=arr[0],truth=arr[1],origins=ds.origins,
            mean=np.asarray(sc['mean']),scale=np.asarray(sc['scale']),channels=np.asarray(meta['channels']))
        result[name]=metrics;del model;torch.cuda.empty_cache()
    base=result[sel['ctf']];phys=result[sel['physics']]
    summary=dict(selection=sel,results=result,mse_gain_percent=100*(base['mse']-phys['mse'])/base['mse'],
        mae_gain_percent=100*(base['mae']-phys['mae'])/base['mae'],
        improved_channels=int(np.sum(np.array(phys['per_channel_mse'])<np.array(base['per_channel_mse']))),
        improved_horizons=int(np.sum(np.array(phys['horizon_mse'])<np.array(base['horizon_mse']))),
        limits=['One seed; fault-free normal operation; 100% means fixed 80-run Train pool.',
                'TEP is simulated; these results do not establish real-plant transfer or fault robustness.',
                'All trials selected using validation; test not used to select a winning configuration.'])
    dump(final/'summary.json',summary);dump(HERE/'status.json',dict(state='completed',summary=str(final/'summary.json'),updated=time.time()))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['results','selection']}),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','train','select','test','all'],default='train')
    p.add_argument('--trial',default='all');p.add_argument('--device',default='cuda:1');args=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    if args.stage in ['prepare','train','all']:
        ds,g,s,meta=load_development()
        if args.stage!='prepare':
            for spec in TRIALS:
                if args.trial not in ['all',spec['name']]:continue
                train_one(spec,ds,g,s,args.device)
            if args.trial=='all':print(json.dumps(selection()),flush=True)
    if args.stage=='select':print(json.dumps(selection()),flush=True)
    if args.stage in ['test','all']:freeze_and_test(args.device)


if __name__=='__main__':main()
