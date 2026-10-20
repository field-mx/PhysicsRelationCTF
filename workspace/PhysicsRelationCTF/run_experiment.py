"""Paired training/validation; test is a separate, explicitly locked phase."""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from physics import ROOT,get_data,config,seed_all,dump,OriginalCTF
from relation_model import PhysicsRelationCTF,copy_ctf

def make_model(kind,cfg,graph,scalers,seed,options):
    seed_all(seed)
    base=OriginalCTF(SimpleNamespace(**cfg))
    if kind=='CTF':return base
    if kind!='Physics':raise ValueError(kind)
    model=PhysicsRelationCTF(SimpleNamespace(**cfg),graph,scalers,**options)
    copy_ctf(model,base)
    return model

def record(model,enabled):
    if isinstance(model,PhysicsRelationCTF):
        for layer in model.encoder.layers:layer.self_attention.inner_attention.record=enabled

@torch.no_grad()
def evaluate(model,data,device,save=False):
    model.eval();pred=[];truth=[];se=ae=0.;n=0
    scale=torch.tensor(data.scale,device=device)[None,None]
    mean=torch.tensor(data.mean,device=device)[None,None]
    for x,y in DataLoader(data,batch_size=32,shuffle=False):
        x,y=x.to(device),y.to(device);p=model(x,None,None,None)
        err=(p-y).double()*scale.double()
        se+=float(err.square().sum());ae+=float(err.abs().sum());n+=err.numel()
        if save:
            pred.append((p*scale+mean).cpu().numpy());truth.append((y*scale+mean).cpu().numpy())
    return dict(mse=se/n,mae=ae/n),(np.concatenate(pred),np.concatenate(truth)) if save else None

def checkpoint(path,payload):
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)

def train(kind,ratio,seed,out,device,options,epochs=16,lr=1e-4,
          sampling='prefix',subset_seed=20260922):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data,graph,meta,selection,scalers=get_data(ratio,sampling=sampling,subset_seed=subset_seed)
    cfg=config(len(meta['channels']),lr)
    specification=dict(kind=kind,ratio=ratio,seed=seed,config=cfg,options=options,epochs=epochs,
                       lr=lr,sampling=sampling,subset_seed=subset_seed,selection=selection,
                       schedule='6 constant epochs then cosine to .1lr; patience5 after epoch8',
                       metric_scale='fixed original full-train reference standard deviation',
                       source_hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                                      for f in ['physics.py','relation_model.py','run_experiment.py']})
    configpath=out/'config.json'
    if configpath.exists():
        previous=json.loads(configpath.read_text())
        if previous!=specification:raise RuntimeError(f'Cannot overwrite a different protocol in {out}')
    else:dump(configpath,specification)
    if (out/'training.json').exists():return json.loads((out/'training.json').read_text())
    dump(out/'graph.json',graph);dump(out/'scalers.json',scalers)
    model=make_model(kind,cfg,graph,scalers,seed,options).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    best=float('inf');stale=0;history=[];start=1;elapsed=0.
    latest=out/'resume.pth'
    if latest.exists():
        r=torch.load(latest,map_location=device,weights_only=False)
        model.load_state_dict(r['model']);optimizer.load_state_dict(r['optimizer'])
        best=r['best'];stale=r['stale'];history=r['history'];start=r['epoch']+1;elapsed=r['seconds']
    tic=time.monotonic()
    for epoch in range(start,epochs+1):
        if epoch>8 and stale>=5:break
        seed_all(seed*1000+epoch)
        order=np.random.default_rng(seed*1000+epoch).permutation(len(data['train'])).tolist()
        loader=DataLoader(data['train'],batch_size=32,sampler=order,num_workers=0)
        rate=lr*(1. if epoch<=6 else .1+.9*.5*(1+math.cos(math.pi*(epoch-6)/max(1,epochs-6))))
        for group in optimizer.param_groups:group['lr']=rate
        model.train();record(model,False);total=0.;count=0;et=time.monotonic()
        for step,(x,y) in enumerate(loader):
            x,y=x.to(device),y.to(device);optimizer.zero_grad(set_to_none=True)
            p=model(x,None,None,None);loss=(p-y).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*y.numel();count+=y.numel()
            if step%100==0:
                dump(out/'status.json',dict(state='training',epoch=epoch,step=step,steps=len(loader),
                     train_mse=total/count,best_val_mse=None if not math.isfinite(best) else best,updated=time.time()))
        val,_=evaluate(model,data['val'],device)
        diag=[]
        if kind=='Physics':
            record(model,True)
            with torch.no_grad():
                x,_=next(iter(DataLoader(data['val'],batch_size=32)))
                model(x.to(device),None,None,None)
            diag=[l.self_attention.inner_attention.diagnostics for l in model.encoder.layers];record(model,False)
        row=dict(epoch=epoch,train_mse=total/count,val_mse=val['mse'],val_mae=val['mae'],
                 lr=rate,epoch_seconds=time.monotonic()-et,diagnostics=diag)
        history.append(row)
        if val['mse']<best:
            best=val['mse'];stale=0
            checkpoint(out/'best.pth',dict(model=model.state_dict(),config=cfg,graph=graph,scalers=scalers,
                       kind=kind,options=options,seed=seed,ratio=ratio,epoch=epoch,
                       sampling=sampling,subset_seed=subset_seed))
        else:stale+=1
        checkpoint(latest,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),best=best,
                   stale=stale,history=history,epoch=epoch,seconds=elapsed+time.monotonic()-tic))
        dump(out/'history.json',history)
        print(json.dumps(dict(model=kind,ratio=ratio,seed=seed,**{k:v for k,v in row.items() if k!='diagnostics'})),flush=True)
    saved=torch.load(out/'best.pth',map_location=device,weights_only=False);model.load_state_dict(saved['model'])
    validation,arrays=evaluate(model,data['val'],device,save=True)
    np.savez_compressed(out/'validation_predictions.npz',prediction=arrays[0],truth=arrays[1],
                       origins=data['val'].origins,channels=np.asarray(meta['channels']))
    result=dict(kind=kind,ratio=ratio,seed=seed,best_val_mse=best,best_epoch=saved['epoch'],
                best_val_mae=validation['mae'],epochs_completed=len(history),
                parameter_count=sum(p.numel() for p in model.parameters()),
                seconds=elapsed+time.monotonic()-tic,selection=selection,test_evaluated=False)
    dump(out/'training.json',result);dump(out/'status.json',dict(state='validation_complete',**result))
    print(json.dumps(result),flush=True)
    return result

def test(out,device,lockpath):
    out=Path(out).resolve();lock=json.loads(Path(lockpath).read_text())
    digest=hashlib.sha256((out/'best.pth').read_bytes()).hexdigest()
    if lock.get('checkpoints',{}).get(str(out))!=digest:raise RuntimeError('Checkpoint not in frozen selection')
    if (out/'metrics.json').exists():return json.loads((out/'metrics.json').read_text())
    r=torch.load(out/'best.pth',map_location='cpu',weights_only=False)
    data,_,meta,_,_=get_data(r['ratio'],sampling=r.get('sampling','prefix'),
                             subset_seed=r.get('subset_seed',20260922))
    model=make_model(r['kind'],r['config'],r['graph'],r['scalers'],r['seed'],r['options']).to(device)
    model.load_state_dict(r['model']);record(model,True)
    result,arrays=evaluate(model,data['test'],device,save=True)
    prediction,truth=arrays
    error=prediction.astype('float64')-truth.astype('float64')
    result.update(kind=r['kind'],ratio=r['ratio'],seed=r['seed'],best_epoch=r['epoch'],
                  per_channel_mse=np.mean(error**2,axis=(0,1)).tolist(),
                  per_channel_mae=np.mean(abs(error),axis=(0,1)).tolist(),
                  horizon_mse=np.mean(error**2,axis=(0,2)).tolist(),channels=meta['channels'],
                  diagnostics=[l.self_attention.inner_attention.diagnostics for l in model.encoder.layers] if r['kind']=='Physics' else [])
    np.savez_compressed(out/'predictions.npz',prediction=prediction,truth=truth,origins=data['test'].origins,
                       channels=np.asarray(meta['channels']))
    dump(out/'metrics.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['CTF','Physics'],required=True)
    p.add_argument('--ratios',nargs='+',type=float,default=[.5,.3,1.]);p.add_argument('--seeds',nargs='+',type=int,default=[42])
    p.add_argument('--device',default='cuda:0');p.add_argument('--root',default=str(ROOT/'runs/v1'))
    p.add_argument('--gamma',type=float,default=.5);p.add_argument('--beta',type=float,default=.05)
    p.add_argument('--rho',type=float,default=.001);p.add_argument('--representation',default='anchored',choices=['anchored','centered'])
    p.add_argument('--lag-mode',default='physical',choices=['physical','uniform'])
    p.add_argument('--sampling',default='prefix',choices=['prefix','random_windows'])
    p.add_argument('--subset-seed',type=int,default=20260922)
    p.add_argument('--epochs',type=int,default=16);p.add_argument('--lr',type=float,default=1e-4)
    a=p.parse_args();os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    options=dict(beta=a.beta,gamma=a.gamma,rho=a.rho,representation=a.representation,lag_mode=a.lag_mode)
    for seed in a.seeds:
        for ratio in a.ratios:
            train(a.kind,ratio,seed,Path(a.root)/f'train{round(ratio*100):03d}'/f'{a.kind}_seed{seed}',
                  a.device,options if a.kind=='Physics' else {},a.epochs,a.lr,a.sampling,a.subset_seed)

if __name__=='__main__':main()
