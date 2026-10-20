"""Matched validation training, resumable checkpoints, explicitly locked test."""
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
from torch.utils.data import DataLoader
from data import ROOT, dump
from prepare import Windows
from public_physics import build_graph,fit_gains
from public_model import OriginalCTF,PublicPhysicsCTF,config,copy_ctf


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def make_model(kind,cfg,graph,scalers,seed,options):
    seed_all(seed);base=OriginalCTF(SimpleNamespace(**cfg))
    if kind=='CTF':return base
    net=PublicPhysicsCTF(SimpleNamespace(**cfg),graph,scalers,**options)
    copy_ctf(net,base);return net


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    parent=ROOT.parents[1]
    paths=[ROOT/f for f in ['data.py','prepare.py','public_physics.py','public_model.py','train_public.py']]
    paths += [parent/'relation_model.py',parent/'physics.py',parent.parent/'PhysicsGuidedCTF/model.py',
              parent.parent/'CTF/ChannelTokenFormer/models/ChannelTokenFormer.py',
              parent.parent/'CTF/ChannelTokenFormer/layers/CTF_Embed.py']
    return {str(p):sha(p) for p in paths}


def verify_test_lock(checkpoint,lock):
    lock=Path(lock)
    if not lock.exists():raise RuntimeError('Test is locked until validation configuration is frozen')
    obj=json.loads(lock.read_text())
    if obj.get('checkpoints',{}).get(str(Path(checkpoint).resolve()))!=sha(checkpoint):
        raise RuntimeError('Checkpoint absent from frozen test manifest')
    for p,digest in obj.get('sources',{}).items():
        if sha(p)!=digest:raise RuntimeError(f'Frozen source changed: {p}')
    for p,digest in obj.get('data',{}).items():
        if sha(p)!=digest:raise RuntimeError(f'Frozen data changed: {p}')


def checkpoint(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)


@torch.no_grad()
def evaluate(model,ds,device,save=False):
    model.eval();c=len(ds.meta['channels']);se=np.zeros(c);ae=np.zeros(c);rawse=np.zeros(c)
    hs=np.zeros(96);ha=np.zeros(96);n=0;pred=[];truth=[]
    ref=torch.tensor(ds.meta['reference_scale'],device=device,dtype=torch.float32)
    scale=torch.tensor(ds.scale,device=device);mean=torch.tensor(ds.mean,device=device)
    targets=ds.meta['target_indices']
    for x,y in DataLoader(ds,batch_size=32,shuffle=False,num_workers=0):
        x=x.to(device);y=y.to(device);p=model(x,None,None,None)
        error=(p-y)*scale/ref
        se+=error.square().sum((0,1)).double().cpu().numpy()
        ae+=error.abs().sum((0,1)).double().cpu().numpy()
        rawse+=((p-y)*scale).square().sum((0,1)).double().cpu().numpy()
        hs+=error[:,:,targets].square().sum((0,2)).double().cpu().numpy()
        ha+=error[:,:,targets].abs().sum((0,2)).double().cpu().numpy()
        n+=len(x)*96
        if save:pred.append((p*scale+mean).cpu().numpy());truth.append((y*scale+mean).cpu().numpy())
    se/=n;ae/=n;rawse/=n
    result=dict(mse=float(se[targets].mean()),mae=float(ae[targets].mean()),
                per_channel_mse=se.tolist(),per_channel_mae=ae.tolist(),per_channel_raw_mse=rawse.tolist(),
                horizon_mse=(hs/(n/96*len(targets))).tolist(),horizon_mae=(ha/(n/96*len(targets))).tolist(),
                windows=len(ds),channels=ds.meta['channels'],target_indices=targets)
    return result,(np.concatenate(pred),np.concatenate(truth)) if save else None


def record(model,on):
    if isinstance(model,PublicPhysicsCTF):
        for l in model.encoder.layers:l.self_attention.inner_attention.record=on


def train(kind,ratio,seed,out,device,options,epochs=16,lr=1e-4):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ds=Windows('train',ratio);val=Windows('val',ratio)
    cfg=config(len(ds.meta['channels']),lr)
    spec=dict(kind=kind,ratio=ratio,seed=seed,options=options,epochs=epochs,lr=lr,config=cfg,
              sources=source_hashes(),data_metadata_sha256=sha(ROOT/'data/metadata.json'),
              schedule='6 constant then cosine to .1lr; min8 epochs; patience6',
              metric='macro physical sensor MSE in full-train reference std; val only selection')
    if (out/'config.json').exists():
        if json.loads((out/'config.json').read_text())!=spec:raise RuntimeError('Refusing changed protocol overwrite')
    else:dump(out/'config.json',spec)
    if (out/'training.json').exists():return json.loads((out/'training.json').read_text())
    print(json.dumps(dict(state='setup',kind=kind,ratio=ratio,windows=len(ds),output=str(out))),flush=True)
    graph=build_graph(ds.meta['channels'])
    if (out/'scalers.json').exists():scalers=json.loads((out/'scalers.json').read_text())
    else:
        intervals=ds.meta['selections'][str(float(ratio))]['intervals']
        raw=np.concatenate([ds.values[a:b] for a,b in intervals]).astype('float64')
        gains,diagnostics=fit_gains(raw,graph)
        scalers=dict(mean=ds.mean.tolist(),scale=ds.scale.tolist(),gains=gains)
        dump(out/'scalers.json',scalers);dump(out/'physics_validation.json',diagnostics)
        del raw
    dump(out/'graph.json',graph)
    model=make_model(kind,cfg,graph,scalers,seed,options).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    best=float('inf');stale=0;start=1;history=[];elapsed=0.
    if (out/'resume.pth').exists():
        r=torch.load(out/'resume.pth',map_location=device,weights_only=False)
        model.load_state_dict(r['model']);optimizer.load_state_dict(r['optimizer'])
        best=r['best'];stale=r['stale'];start=r['epoch']+1;history=r['history'];elapsed=r['elapsed']
    targets=ds.meta['target_indices'];tic=time.monotonic()
    for epoch in range(start,epochs+1):
        if epoch>8 and stale>=6:break
        seed_all(seed*1000+epoch)
        order=np.random.default_rng(seed*1000+epoch).permutation(len(ds)).tolist()
        loader=DataLoader(ds,batch_size=32,sampler=order,num_workers=0)
        rate=lr if epoch<=6 else lr*(.1+.9*.5*(1+math.cos(math.pi*(epoch-6)/max(1,epochs-6))))
        for group in optimizer.param_groups:group['lr']=rate
        model.train();record(model,False);total=0.;count=0;et=time.monotonic()
        for step,(x,y) in enumerate(loader):
            x=x.to(device);y=y.to(device);optimizer.zero_grad(set_to_none=True)
            p=model(x,None,None,None);loss=(p[:,:,targets]-y[:,:,targets]).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(x);count+=len(x)
            if step%20==0:
                dump(out/'status.json',dict(state='training',epoch=epoch,step=step,steps=len(loader),
                     train_mse=total/count,best_val_mse=best if math.isfinite(best) else None,
                     updated=time.time(),pid=os.getpid(),device=device))
        result,_=evaluate(model,val,device)
        diag=[]
        if kind=='Physics':
            record(model,True)
            with torch.no_grad():
                x,_=next(iter(DataLoader(val,batch_size=16)));model(x.to(device),None,None,None)
            diag=[l.self_attention.inner_attention.diagnostics for l in model.encoder.layers];record(model,False)
        row=dict(epoch=epoch,train_mse=total/count,val_mse=result['mse'],val_mae=result['mae'],lr=rate,
                 seconds=time.monotonic()-et,diagnostics=diag)
        history.append(row)
        if result['mse']<best:
            best=result['mse'];stale=0
            checkpoint(out/'best.pth',dict(model=model.state_dict(),config=cfg,graph=graph,scalers=scalers,
                       kind=kind,options=options,seed=seed,ratio=ratio,epoch=epoch))
        else:stale+=1
        checkpoint(out/'resume.pth',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),best=best,
                   stale=stale,history=history,epoch=epoch,elapsed=elapsed+time.monotonic()-tic))
        dump(out/'history.json',history)
        print(json.dumps(dict(kind=kind,ratio=ratio,seed=seed,**{k:v for k,v in row.items() if k!='diagnostics'})),flush=True)
    r=torch.load(out/'best.pth',map_location=device,weights_only=False);model.load_state_dict(r['model'])
    result,arrays=evaluate(model,val,device,save=True)
    np.savez_compressed(out/'validation_predictions.npz',prediction=arrays[0],truth=arrays[1],
                        origins=val.origins,channels=np.asarray(ds.meta['channels']))
    dump(out/'validation_metrics.json',result)
    summary=dict(kind=kind,ratio=ratio,seed=seed,best_epoch=r['epoch'],best_val_mse=result['mse'],
                 best_val_mae=result['mae'],epochs_completed=len(history),test_evaluated=False,
                 seconds=elapsed+time.monotonic()-tic,parameters=sum(p.numel() for p in model.parameters()))
    dump(out/'training.json',summary);dump(out/'status.json',dict(state='validation_complete',**summary))
    print(json.dumps(summary),flush=True);return summary


def test(out,device,lock):
    out=Path(out);verify_test_lock(out/'best.pth',lock)
    if (out/'test_metrics.json').exists():return json.loads((out/'test_metrics.json').read_text())
    r=torch.load(out/'best.pth',map_location='cpu',weights_only=False)
    ds=Windows('test',r['ratio'])
    net=make_model(r['kind'],r['config'],r['graph'],r['scalers'],r['seed'],r['options']).to(device)
    net.load_state_dict(r['model']);result,arrays=evaluate(net,ds,device,save=True)
    result.update(kind=r['kind'],ratio=r['ratio'],seed=r['seed'],best_epoch=r['epoch'])
    np.savez_compressed(out/'test_predictions.npz',prediction=arrays[0],truth=arrays[1],
                        origins=ds.origins,channels=np.asarray(ds.meta['channels']))
    dump(out/'test_metrics.json',result);print(json.dumps({k:result[k] for k in ['kind','ratio','seed','mse','mae']}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['CTF','Physics'],default='CTF')
    p.add_argument('--ratios',nargs='+',type=float,default=[1.]);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda:0');p.add_argument('--out',required=True)
    p.add_argument('--epochs',type=int,default=16);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--beta',type=float,default=.05);p.add_argument('--gamma',type=float,default=.75)
    p.add_argument('--rho',type=float,default=.001);p.add_argument('--representation',default='centered')
    p.add_argument('--lag-mode',default='physical');p.add_argument('--test-lock')
    a=p.parse_args();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    if a.test_lock:test(a.out,a.device,a.test_lock);return
    options=dict(beta=a.beta,gamma=a.gamma,rho=a.rho,representation=a.representation,lag_mode=a.lag_mode)
    for ratio in a.ratios:
        train(a.kind,ratio,a.seed,Path(a.out)/f'train{round(100*ratio):03d}'/f'{a.kind}_seed{a.seed}',
              a.device,options if a.kind=='Physics' else {},a.epochs,a.lr)


if __name__=='__main__':main()
