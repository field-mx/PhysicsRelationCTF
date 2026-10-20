"""Reproducible channel-readout CTF comparison; tuning never reads test targets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

from model import OriginalCTF,PhysicsCTF,load_native_weights
from graphs import adjacency

ROOT=Path(__file__).resolve().parent
RATIOS=[1.,.5,.2,.1,.05]
SEEDS=[42,43,44]

def dump(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(path)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

class Windows(Dataset):
    def __init__(self,values,origins,mean,scale):
        self.values=values if values.ndim==3 else values[None]
        self.origins=origins;self.mean=mean.astype('float32');self.scale=scale.astype('float32')
    def __len__(self):return len(self.origins)
    def __getitem__(self,index):
        run,t=self.origins[index];v=self.values[run]
        x=(v[t-96:t]-self.mean)/self.scale;y=(v[t:t+96]-self.mean)/self.scale
        return torch.from_numpy(x.copy()),torch.from_numpy(y.copy())

def select_random_origins(origins,ratio,seed=20260922):
    """Return a deterministic nested random subset of complete windows."""
    if not 0 < ratio <= 1:
        raise ValueError('ratio must be in (0, 1]')
    order=np.random.default_rng(seed).permutation(len(origins))
    return origins[order[:max(1,int(len(origins)*ratio))]]

def _covered_rows(origins,length):
    """Rows actually observed by at least one selected 96->96 window."""
    covered=np.zeros(length,dtype=bool)
    for run,t in origins:
        if int(run)!=0:raise ValueError('industrial data must contain one run')
        covered[max(0,int(t)-96):min(length,int(t)+96)]=True
    return np.flatnonzero(covered)

def load_data(name,ratio,sampling='prefix',subset_seed=20260922):
    path=ROOT/'data'/name
    meta=json.loads((path/'metadata.json').read_text())
    graph=json.loads((path/'graph.json').read_text())
    origins={s:np.load(path/f'{s}_origins.npy') for s in ['train','val','test']}
    if name=='industrial':
        a=np.load(path/'values.npy',mmap_mode='r');values={s:a for s in origins}
        if sampling=='prefix':
            end=int(meta['train_end']*ratio)
            origins['train']=origins['train'][origins['train'][:,1]+96<=end]
            fit=a[:end];selection=dict(train_end=end,rows=end,sampling='prefix')
        elif sampling=='random_windows':
            full_train=origins['train']
            origins['train']=select_random_origins(full_train,ratio,subset_seed)
            rows=_covered_rows(origins['train'],meta['train_end'])
            fit=a[rows]
            selection=dict(train_end=meta['train_end'],rows=len(rows),sampling=sampling,
                           subset_seed=subset_seed,full_train_windows=len(full_train),
                           selected_train_windows=len(origins['train']),
                           covered_rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest())
        else:
            raise ValueError(f'Unknown industrial sampling mode: {sampling}')
    else:
        values={s:np.load(path/f'{s}_values.npy',mmap_mode='r') for s in origins}
        permutation=np.random.default_rng(20260921).permutation(len(values['train']))
        selected=permutation[:max(1,int(len(permutation)*ratio))]
        origins['train']=origins['train'][np.isin(origins['train'][:,0],selected)]
        fit=values['train'][selected].reshape(-1,values['train'].shape[-1])
        selection=dict(train_run_ids=[meta['ids']['train'][i] for i in selected],rows=len(fit))
    mean=fit.mean(0,dtype=np.float64);scale=fit.std(0,dtype=np.float64)
    scale=np.where(scale<1e-5,1.,scale)
    if not len(origins['train']):raise ValueError('Insufficient history+forecast points')
    datasets={s:Windows(values[s],origins[s],mean,scale) for s in origins}
    selection.update(windows={s:len(d) for s,d in datasets.items()},
                     origins_sha256={s:hashlib.sha256(v.tobytes()).hexdigest() for s,v in origins.items()})
    return datasets,graph,meta,selection

def config(c,lr=1e-4):
    return dict(task_name='long_term_forecast',features='M',seq_len=96,pred_len=96,
                enc_in=c,dec_in=c,c_out=c,use_norm=1,patch_lens=[24]*c,sampling_periods=[1.]*c,
                num_global_tokens=2,d_model=256,d_ff=1024,n_heads=8,e_layers=2,
                dropout=.1,factor=3,activation='gelu',batch_size=32,keep_prob=1.,learning_rate=lr)

def make_model(cfg,graph,kind,seed,beta):
    seed_all(seed)
    baseline=OriginalCTF(SimpleNamespace(**cfg))
    if kind=='CTF':return baseline
    m=PhysicsCTF(SimpleNamespace(**cfg),adjacency(graph),mode=kind.lower(),beta_init=beta)
    load_native_weights(m,baseline)
    return m

def diagnostics(m,enabled):
    if isinstance(m,PhysicsCTF):
        for l in m.encoder.layers:l.self_attention.inner_attention.record=enabled

@torch.no_grad()
def evaluate(m,data,device,batch_size=32,noise=0.,noise_rep=0,save=False):
    m.eval();total=mae=0.;count=0;preds=[];truths=[];weights=[]
    rng=np.random.default_rng(80131+noise_rep)
    scale=torch.tensor(data.scale,device=device)[None,None]
    mean=torch.tensor(data.mean,device=device)[None,None]
    for x,y in DataLoader(data,batch_size=batch_size,shuffle=False):
        if noise:
            # Noise uses fixed full-training reference std, identical across models/ratios.
            eps=rng.normal(0,noise,x.shape).astype('float32')/data.scale
            x=x+torch.from_numpy(eps)
        x=x.to(device);y=y.to(device)
        p=m(x,None,None,None)
        error=(p-y).double()*scale.double()
        total+=error.square().sum().item();mae+=error.abs().sum().item();count+=error.numel()
        if save:
            preds.append((p*scale+mean).cpu().numpy());truths.append((y*scale+mean).cpu().numpy())
            if isinstance(m,PhysicsCTF):weights.append(m.latest_weights.cpu().numpy())
    r=dict(mse=total/count,mae=mae/count)
    return r,(np.concatenate(preds),np.concatenate(truths),np.concatenate(weights) if weights else None) if save else None

def train_job(name,ratio,kind,seed,beta,lr,epochs,out,device):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'training.json').exists():return json.loads((out/'training.json').read_text())
    datasets,graph,meta,selection=load_data(name,ratio)
    cfg=config(len(meta['channels']),lr)
    m=make_model(cfg,graph,kind,seed,beta).to(device)
    optimizer=torch.optim.Adam(m.parameters(),lr=lr)
    best=float('inf');stale=0;history=[];start=1;t0=time.monotonic()
    latest=out/'resume.pth'
    if latest.exists():
        r=torch.load(latest,map_location=device,weights_only=False)
        m.load_state_dict(r['model']);optimizer.load_state_dict(r['optimizer'])
        best=r['best'];stale=r['stale'];history=r['history'];start=r['epoch']+1
    for epoch in range(start,epochs+1):
        if stale>=3:break
        seed_all(seed*1000+epoch)
        order=np.random.default_rng(seed*1000+epoch).permutation(len(datasets['train'])).tolist()
        loader=DataLoader(datasets['train'],batch_size=32,sampler=order,num_workers=0)
        m.train();diagnostics(m,False);loss_sum=0.;n=0
        rate=lr*(.5**max(0,epoch-2))
        for g in optimizer.param_groups:g['lr']=rate
        for x,y in loader:
            x=x.to(device);y=y.to(device);optimizer.zero_grad(set_to_none=True)
            p=m(x,None,None,None);loss=(p-y).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();optimizer.step();loss_sum+=loss.item()*y.numel();n+=y.numel()
        val,_=evaluate(m,datasets['val'],device)
        diagnostics(m,True)
        with torch.no_grad():
            x,_=next(iter(DataLoader(datasets['val'],batch_size=32)))
            m(x.to(device),None,None,None)
        diag=[l.self_attention.inner_attention.diagnostics for l in m.encoder.layers] if kind!='CTF' else []
        diagnostics(m,False)
        row=dict(epoch=epoch,train_mse=loss_sum/n,val_mse=val['mse'],val_mae=val['mae'],lr=rate,diagnostics=diag)
        history.append(row)
        if val['mse']<best:
            best=val['mse'];stale=0
            torch.save(dict(model=m.state_dict(),config=cfg,graph=graph,kind=kind,beta_init=beta,
                            seed=seed,ratio=ratio,mean=datasets['train'].mean,scale=datasets['train'].scale),out/'best.pth')
        else:stale+=1
        tmp=out/'resume.tmp'
        torch.save(dict(model=m.state_dict(),optimizer=optimizer.state_dict(),best=best,stale=stale,
                        history=history,epoch=epoch),tmp);tmp.replace(latest)
        dump(out/'history.json',history)
        print(json.dumps(dict(dataset=name,ratio=ratio,model=kind,seed=seed,beta=beta,epoch=epoch,val=val['mse'])),flush=True)
    r=dict(dataset=name,ratio=ratio,model=kind,seed=seed,beta_init=beta,lr=lr,best_val_mse=best,
           parameter_count=sum(p.numel() for p in m.parameters()),selection=selection,seconds=time.monotonic()-t0,
           epochs_completed=len(history),history=history)
    dump(out/'training.json',r)
    return r

def test_job(out,device):
    out=Path(out)
    if (out/'metrics.json').exists():return json.loads((out/'metrics.json').read_text())
    info=json.loads((out/'training.json').read_text());r=torch.load(out/'best.pth',map_location='cpu',weights_only=False)
    d,g,meta,_=load_data(info['dataset'],info['ratio'])
    m=make_model(r['config'],g,r['kind'],r['seed'],r['beta_init']).to(device)
    m.load_state_dict(r['model']);diagnostics(m,True)
    clean,arrays=evaluate(m,d['test'],device,save=True)
    p,y,w=arrays
    np.savez_compressed(out/'predictions.npz',prediction=p,truth=y,origins=d['test'].origins,
                        channels=np.asarray(meta['channels']),edge_weights=w if w is not None else np.empty((0,)),
                        edge_src=m.edge_src.cpu().numpy() if isinstance(m,PhysicsCTF) else np.empty((0,),dtype=int),
                        edge_dst=m.edge_dst.cpu().numpy() if isinstance(m,PhysicsCTF) else np.empty((0,),dtype=int))
    diag=[l.self_attention.inner_attention.diagnostics for l in m.encoder.layers] if info['model']!='CTF' else []
    diagnostics(m,False)
    noise=[]
    if info['ratio']==1.:
        for sigma in [.01,.03,.05,.10]:
            for rep in range(3):
                score,_=evaluate(m,d['test'],device,noise=sigma,noise_rep=rep)
                noise.append(dict(sigma=sigma,rep=rep,**score))
    result={k:info[k] for k in ['dataset','ratio','model','seed','parameter_count','selection','beta_init','lr']}
    result.update(clean=clean,noise=noise,diagnostics=diag,
                  metric_scale='fixed full-training reference std; model scaler uses current subset only',
                  w_summary=dict(mean=float(w.mean()),std=float(w.std()),min=float(w.min()),max=float(w.max()),
                                 saturation=float(((w<.01)|(w>.99)).mean())) if w is not None else None)
    dump(out/'metrics.json',result);print(json.dumps(dict(completed=str(out),clean=clean)),flush=True)
    return result

def run(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    root=ROOT/'runs/channel_readout_v1'/args.dataset;root.mkdir(parents=True,exist_ok=True)
    lock=root/'selection.json'
    if not lock.exists():
        # Equal common hyperparameter opportunity: baseline selects shared LR;
        # physics beta is the only model-specific tuning dimension.
        lr_scores={}
        for lr in [1e-4,3e-4]:
            r=train_job(args.dataset,1.,'CTF',42,.05,lr,3,root/f'tuning/ctf_lr{lr}',args.device)
            lr_scores[str(lr)]=r['best_val_mse']
        lr=float(min(lr_scores,key=lr_scores.get));beta_scores={}
        for beta in [.01,.05,.1,.2]:
            r=train_job(args.dataset,1.,'Dynamic',42,beta,lr,3,root/f'tuning/dynamic_beta{beta}',args.device)
            beta_scores[str(beta)]=r['best_val_mse']
        beta=float(min(beta_scores,key=beta_scores.get))
        dump(lock,dict(lr=lr,beta=beta,lr_validation=lr_scores,beta_validation=beta_scores,
                       policy='3-epoch full-data seed42 validation-only pilot; shared LR and chosen beta frozen for all final ratios/seeds; no test evaluation in tuning'))
    chosen=json.loads(lock.read_text())
    # Finish ALL training and validation selection before evaluating any test model.
    jobs=[]
    for ratio in RATIOS:
        for seed in SEEDS:
            for kind in ['CTF','Fixed','Dynamic']:
                out=root/f'f{int(ratio*100):03d}_s{seed}_{kind}'
                train_job(args.dataset,ratio,kind,seed,chosen['beta'],chosen['lr'],10,out,args.device)
                jobs.append(out)
    for out in jobs:test_job(out,args.device)
    dump(root/'complete.json',dict(status='complete',jobs=len(jobs),selection=chosen))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['industrial','tep'],required=True)
    p.add_argument('--device',default='cuda:0');run(p.parse_args())
