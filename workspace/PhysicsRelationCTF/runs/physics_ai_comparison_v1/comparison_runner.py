"""Isolated, resumable validation screening and locked final evaluation."""
import argparse
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from physics import config, get_data, seed_all
from comparison_models import PartialPhysicsLoss, make_model

NEW_MODELS = ('PhyGNN', 'MTGNN', 'PatchTST', 'PG-Reg-CTF')
SEEDS = (42, 43, 44)
EPOCHS = 16
LR = 1e-4
BATCH = 32
EXPLORATORY_WARNING = ('Comparators selected for extra seeds only when seed42 validation MSE is higher '
    'than our same-seed reference. All pilots retained. Outcome-selected exploration, '
    'not an unbiased superiority benchmark.')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def dump(path, value):
    path = Path(path).resolve()
    path.relative_to(HERE)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def immutable(path, value):
    value = json.loads(json.dumps(value))
    if Path(path).exists():
        if read(path) != value:
            raise RuntimeError(f'Refusing to overwrite changed protocol: {path}')
    else:
        dump(path, value)


def checkpoint(path, value):
    path = Path(path).resolve(); path.relative_to(HERE)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp); tmp.replace(path)


def nominal_options(kind):
    return {
        'PhyGNN': {'graph_heads': [4, 4], 'temporal_hidden': 128},
        'MTGNN': {'neighbors': 20, 'dropout': .1},
        'PatchTST': {'stride': 12},
        'PG-Reg-CTF': {'physics_weight': .01},
    }[kind].copy()


def reference_paths(kind, seed):
    if kind == 'PhysicsRelationCTF':
        if seed == 42:
            return ROOT/'runs/random_gaps_v1/train100/Physics_seed42'
        return ROOT/f'runs/multiseed_100_v1/industrial/train100/Physics_seed{seed}'
    if kind == 'EquationInput-CTF':
        return ROOT/f'runs/paper_minimal_v1/industrial/seed{seed}/development/equation_input'
    raise ValueError(kind)


@lru_cache(maxsize=1)
def load_industrial():
    datasets, graph, meta, selection, scalers = get_data(1., sampling='random_gaps', subset_seed=20260922)
    # Runtime intervals are tuples; the frozen JSON represents them as lists.
    selection = json.loads(json.dumps(selection))
    reference = read(reference_paths('PhysicsRelationCTF', 42)/'config.json')
    if selection != reference['selection']:
        raise RuntimeError('Existing full-data split/window manifest mismatch')
    if graph != read(reference_paths('PhysicsRelationCTF', 42)/'graph.json'):
        raise RuntimeError('Existing graph mismatch')
    train = datasets['train']
    standardized = (np.asarray(train.values[0, :14540], dtype='float64')-train.mean)/train.scale
    raw = standardized*np.asarray(scalers['raw_scale'])+np.asarray(scalers['raw_mean'])
    return dict(datasets=datasets, graph=graph, meta=meta, selection=selection, scalers=scalers,
                config=reference['config'], raw_train=raw,
                feature_scaler=read(ROOT/'runs/paper_minimal_v1/industrial/feature_scaler.json'))


def screen(candidate, reference):
    if not math.isfinite(candidate) or not math.isfinite(reference):
        raise ValueError('Only finite validated runs may be screened')
    if candidate > reference:
        return 'append_seeds_43_44'
    if candidate < reference:
        return 'pilot_only_stronger'
    return 'pilot_only_tie'


def deterministic():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def source_hashes():
    files = [HERE/'comparison_models.py', HERE/'comparison_runner.py']
    files += [HERE/'vendor/MTGNN'/name for name in ('net.py', 'layer.py')]
    files += [HERE/'vendor/PhyGNN/PhyGNN/spatial_block.py']
    files += [HERE/'vendor/PatchTST/PatchTST_supervised/layers'/name
              for name in ('PatchTST_backbone.py', 'PatchTST_layers.py', 'RevIN.py')]
    return {str(path): sha(path) for path in files}


def protect_existing():
    manifest = HERE/'protected_existing.json'
    if manifest.exists():
        existing = read(manifest)
        for path, digest in existing.items():
            if sha(path) != digest:
                raise RuntimeError(f'Protected previous artifact changed: {path}')
        return existing
    files = list(ROOT.glob('*.py'))
    files += list((ROOT.parent/'CTF/ChannelTokenFormer/models').glob('*.py'))
    files += list((ROOT.parent/'CTF/ChannelTokenFormer/layers').glob('*.py'))
    files += [ROOT.parent/'PhysicsGuidedCTF'/name for name in ('experiment.py', 'model.py')]
    files += [ROOT.parent/'GNN_Transformer/equation_topology_v2.py',
              ROOT.parent/'GNN_Transformer/runs/sensor_96s_96s_stride96/data_summary.json']
    files += list((ROOT.parent/'PhysicsGuidedCTF/data/industrial').glob('*.npy'))
    files += list((ROOT.parent/'PhysicsGuidedCTF/data/industrial').glob('*.json'))
    files += [ROOT/'runs/paper_minimal_v1/controls.py',
              ROOT/'runs/paper_minimal_v1/summary.json',
              ROOT/'runs/paper_minimal_v1/industrial/feature_scaler.json']
    for kind in ('PhysicsRelationCTF', 'EquationInput-CTF'):
        for seed in SEEDS:
            folder = reference_paths(kind, seed)
            files += [folder/name for name in ('best.pth', 'config.json', 'training.json',
                       'history.json', 'metrics.json', 'paper_metrics.json', 'scalers.json',
                       'graph.json', 'predictions.npz') if (folder/name).exists()]
    result = {str(p): sha(p) for p in files}
    immutable(manifest, result)
    return result


@torch.no_grad()
def evaluate(model, dataset, device, save=False):
    model.eval(); se=ae=0.; count=0; prediction=[]; truth=[]
    scale = torch.tensor(dataset.scale, device=device)[None, None]
    mean = torch.tensor(dataset.mean, device=device)[None, None]
    for x, y in DataLoader(dataset, batch_size=BATCH, shuffle=False):
        x, y = x.to(device), y.to(device)
        p = model(x, None, None, None)
        if p.shape != y.shape or not torch.isfinite(p).all():
            raise RuntimeError('Invalid forecast')
        err = (p-y).double()*scale.double()
        se += float(err.square().sum()); ae += float(err.abs().sum()); count += err.numel()
        if save:
            prediction.append((p*scale+mean).cpu().numpy())
            truth.append((y*scale+mean).cpu().numpy())
    arrays = (np.concatenate(prediction), np.concatenate(truth)) if save else None
    return dict(mse=se/count, mae=ae/count), arrays


@contextmanager
def exclusive(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(mutex, fcntl.LOCK_UN)


def outdir(kind, seed):
    return HERE/'industrial'/f'seed{seed}'/kind


def preflight(device):
    data = load_industrial(); checks=[]
    x = torch.stack([data['datasets']['train'][i][0] for i in range(BATCH)]).to(device)
    y = torch.stack([data['datasets']['train'][i][1] for i in range(BATCH)]).to(device)
    for kind in NEW_MODELS:
        model = make_model(kind, data['config'], data, 42, nominal_options(kind)).to(device)
        torch.cuda.reset_peak_memory_stats(device)
        model.train(); p=model(x, None, None, None)
        objective=(p-y).square().mean()
        if kind=='PG-Reg-CTF':
            physical=PartialPhysicsLoss(data).to(device)
            objective=objective+.01*physical(x,p)
            dump(HERE/'physics_constraints.json', dict(routes=physical.routes,
                 meaning='Train-calibrated equation-response proxies with tolerance, plus avg<=max; not a PDE PINN'))
        objective.backward()
        if p.shape != y.shape or not torch.isfinite(p).all():
            raise RuntimeError(f'Forward failed: {kind}')
        gradients=[v.grad for v in model.parameters() if v.grad is not None]
        if not gradients or not all(torch.isfinite(v).all() for v in gradients):
            raise RuntimeError(f'Backward failed: {kind}')
        model.eval()
        with torch.no_grad():
            one=model(x[:1], None, None, None)
            torch.testing.assert_close(one, model(x[:1],None,torch.randn_like(y[:1]),None), rtol=0,atol=0)
        checks.append(dict(model=kind, output=list(p.shape), parameters=sum(v.numel() for v in model.parameters()),
                      peak_allocated_mb=torch.cuda.max_memory_allocated(device)/1024**2,
                      finite_backward=True, future_decoder_ignored=True))
        del model, p, objective
        torch.cuda.empty_cache()
    dump(HERE/'preflight.json', dict(checks=checks, selection=data['selection'],
                                    graph_edges=len(data['graph']['edges']), no_observation_padding=True))
    print(json.dumps({'preflight':checks}),flush=True)


def train(kind, seed, device):
    out=outdir(kind,seed); out.mkdir(parents=True, exist_ok=True)
    with exclusive(out/'worker.lock'):
        if (out/'training.json').exists():
            return read(out/'training.json')
        data=load_industrial(); cfg=data['config']; options=nominal_options(kind)
        specification=dict(model=kind, seed=seed, config=cfg, options=options, epochs=EPOCHS,
            batch_size=BATCH, optimizer='Adam', lr=LR, selection=data['selection'],
            source_hashes=source_hashes(), selection_metric='validation normalized MSE',
            schedule='6 constant epochs then cosine to .1lr; patience5 after epoch8')
        immutable(out/'config.json',specification)
        model=make_model(kind,cfg,data,seed,options).to(device)
        physical=PartialPhysicsLoss(data).to(device) if kind=='PG-Reg-CTF' else None
        optimizer=torch.optim.Adam(model.parameters(),lr=LR)
        best=float('inf');stale=0;history=[];start=1;elapsed=0.
        if (out/'resume.pth').exists():
            ck=torch.load(out/'resume.pth',map_location=device,weights_only=False)
            model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
            best=ck['best'];stale=ck['stale'];history=ck['history'];start=ck['epoch']+1;elapsed=ck['seconds']
        tic=time.monotonic()
        for epoch in range(start,EPOCHS+1):
            if epoch>8 and stale>=5:
                break
            seed_all(seed*1000+epoch)
            order=np.random.default_rng(seed*1000+epoch).permutation(len(data['datasets']['train'])).tolist()
            loader=DataLoader(data['datasets']['train'],batch_size=BATCH,sampler=order,num_workers=0)
            rate=LR*(1. if epoch<=6 else .1+.9*.5*(1+math.cos(math.pi*(epoch-6)/(EPOCHS-6))))
            for group in optimizer.param_groups:
                group['lr']=rate
            model.train();se=penalty=total=count=0.;et=time.monotonic()
            for step,(x,y) in enumerate(loader):
                x,y=x.to(device),y.to(device);optimizer.zero_grad(set_to_none=True)
                p=model(x,None,None,None);mse=(p-y).square().mean()
                reg=physical(x,p) if physical is not None else mse.detach()*0
                loss=mse+options.get('physics_weight',0.)*reg
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite training loss: {kind}, epoch{epoch}, step{step}')
                loss.backward();optimizer.step()
                n=y.numel();se+=float(mse.detach())*n;penalty+=float(reg.detach())*n
                total+=float(loss.detach())*n;count+=n
                if step%50==0:
                    dump(out/'status.json',dict(state='training',model=kind,seed=seed,epoch=epoch,
                        step=step,steps=len(loader),train_mse=se/count,physics_loss=penalty/count,
                        best_val_mse=None if not math.isfinite(best) else best,updated=time.time()))
            val,_=evaluate(model,data['datasets']['val'],device)
            row=dict(epoch=epoch,train_mse=se/count,physics_loss=penalty/count,objective=total/count,
                     val_mse=val['mse'],val_mae=val['mae'],lr=rate,seconds=time.monotonic()-et)
            history.append(row)
            if val['mse']<best:
                best=val['mse'];stale=0
                checkpoint(out/'best.pth',dict(model=model.state_dict(),kind=kind,seed=seed,
                    config=cfg,options=options,epoch=epoch))
            else:
                stale+=1
            checkpoint(out/'resume.pth',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                best=best,stale=stale,history=history,epoch=epoch,seconds=elapsed+time.monotonic()-tic))
            dump(out/'history.json',history)
            print(json.dumps(dict(model=kind,seed=seed,**row)),flush=True)
        ck=torch.load(out/'best.pth',map_location=device,weights_only=False)
        model.load_state_dict(ck['model'])
        val,arrays=evaluate(model,data['datasets']['val'],device,True)
        np.savez_compressed(out/'validation_predictions.npz',prediction=arrays[0],truth=arrays[1],
            origins=data['datasets']['val'].origins,channels=np.array(data['meta']['channels']))
        result=dict(model=kind,seed=seed,best_val_mse=val['mse'],best_val_mae=val['mae'],
            best_epoch=ck['epoch'],epochs_completed=len(history),
            parameter_count=sum(v.numel() for v in model.parameters()),seconds=elapsed+time.monotonic()-tic,
            selection=data['selection'],test_evaluated=False,checkpoint_sha256=sha(out/'best.pth'))
        dump(out/'training.json',result);dump(out/'status.json',dict(state='validation_complete',**result))
        del model,optimizer,physical;torch.cuda.empty_cache()
        return result


def final_test(kind,seed,device):
    out=outdir(kind,seed);lock=read(HERE/'selection_lock.json')
    if lock['checkpoints'].get(str(out))!=sha(out/'best.pth'):
        raise RuntimeError('Checkpoint is not frozen')
    if (out/'metrics.json').exists():
        return read(out/'metrics.json')
    data=load_industrial();ck=torch.load(out/'best.pth',map_location='cpu',weights_only=False)
    model=make_model(kind,ck['config'],data,seed,ck['options']).to(device)
    model.load_state_dict(ck['model'])
    score,arrays=evaluate(model,data['datasets']['test'],device,True)
    p,y=arrays;error=p.astype('float64')-y.astype('float64')
    score.update(model=kind,seed=seed,channels=data['meta']['channels'],
        per_channel_mse=(error**2).mean((0,1)).tolist(),per_channel_mae=np.abs(error).mean((0,1)).tolist(),
        horizon_mse=(error**2).mean((0,2)).tolist(),checkpoint_sha256=sha(out/'best.pth'))
    np.savez_compressed(out/'predictions.npz',prediction=p,truth=y,
        origins=data['datasets']['test'].origins,channels=np.array(data['meta']['channels']))
    dump(out/'metrics.json',score)
    del model;torch.cuda.empty_cache()
    return score


def summary(ledger):
    results={};all_rows=[]
    for kind in ('PhysicsRelationCTF','EquationInput-CTF'):
        rows=[]
        for seed in SEEDS:
            folder=reference_paths(kind,seed)
            metrics=read(folder/'metrics.json');training=read(folder/'training.json')
            rows.append(dict(model=kind,seed=seed,mse=metrics['mse'],mae=metrics['mae'],
                val_mse=training['best_val_mse'],path=str(folder),reused=True))
        results[kind]=rows;all_rows.extend(rows)
    for kind in NEW_MODELS:
        rows=[]
        for seed in SEEDS:
            out=outdir(kind,seed)
            if not (out/'metrics.json').exists():
                continue
            metrics=read(out/'metrics.json');training=read(out/'training.json')
            rows.append(dict(model=kind,seed=seed,mse=metrics['mse'],mae=metrics['mae'],
                val_mse=training['best_val_mse'],path=str(out),reused=False))
        results[kind]=rows;all_rows.extend(rows)
    aggregate={}
    for kind,rows in results.items():
        mse=[r['mse'] for r in rows];mae=[r['mae'] for r in rows]
        if not rows:
            continue
        aggregate[kind]=dict(n=len(rows),mse_mean=statistics.mean(mse),
            mse_std=statistics.stdev(mse) if len(mse)>1 else None,
            mae_mean=statistics.mean(mae),mae_std=statistics.stdev(mae) if len(mae)>1 else None)
        ours={r['seed']:r for r in results['PhysicsRelationCTF']}
        gains=[100*(r['mse']-ours[r['seed']]['mse'])/r['mse'] for r in rows]
        aggregate[kind]['our_paired_mse_gain_percent']=gains
    report=dict(status='complete',dataset='industrial',scope='100% 96->96',warning=EXPLORATORY_WARNING,
                ledger=ledger,results=results,aggregate=aggregate)
    dump(HERE/'summary.json',report)
    import csv
    with (HERE/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model','seed','mse','mae','val_mse','path','reused'])
        writer.writeheader();writer.writerows(all_rows)
    lines=['# 电解槽物理AI方法比较（探索性筛选）','',EXPLORATORY_WARNING,'',
        '主任务：100%训练池，96点历史预测96点未来。与消融实验分开。','',
        '| 方法 | 种子数 | MSE平均 | MSE标准差 | MAE平均 |','|---|---:|---:|---:|---:|']
    for kind,s in aggregate.items():
        std='—' if s['mse_std'] is None else f"{s['mse_std']:.8f}"
        lines.append(f"| {kind} | {s['n']} | {s['mse_mean']:.8f} | {std} | {s['mae_mean']:.8f} |")
    lines+=['','## 预实验完整记录','','| 方法 | seed42验证MSE | 后续处理 |','|---|---:|---|']
    for row in ledger:
        lines.append(f"| {row['model']} | {row['val_mse']:.8f} | {row['decision']} |")
    lines+=['','只做单种子的对照不能与三种子均值混作稳健排名。',
            '方程增益为训练集经验观测映射，不是已辨识设备参数；PG-Reg-CTF不是标准PINN。',
            'PhyGNN为保留官方G3CN与物理计算/适配/补偿机制的预测适配版，不是SPS原任务完全复现。',
            '本实验准确率结果不单独证明可解释性或因果机制。']
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return report


def run(device):
    deterministic()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required. Do not silently fall back to CPU.')
    protect_existing();data=load_industrial()
    protocol=dict(dataset='industrial',models=list(NEW_MODELS),seeds=list(SEEDS),epochs=EPOCHS,
        optimizer='Adam',lr=LR,batch_size=BATCH,options={k:nominal_options(k) for k in NEW_MODELS},
        selection=data['selection'],source_hashes=source_hashes(),warning=EXPLORATORY_WARNING,
        reference_validation_mse=read(reference_paths('PhysicsRelationCTF',42)/'training.json')['best_val_mse'])
    immutable(HERE/'protocol.json',protocol)
    if not (HERE/'preflight.json').exists():
        preflight(device)
    with exclusive(HERE/'orchestrator.lock'):
        if (HERE/'summary.json').exists():
            protect_existing();return
        ledger=[];jobs=[];reference=protocol['reference_validation_mse']
        for kind in NEW_MODELS:
            dump(HERE/'status.json',dict(state='pilot_training',model=kind,seed=42,
                completed_pilots=len(ledger),updated=time.time(),warning=EXPLORATORY_WARNING))
            result=train(kind,42,device);decision=screen(result['best_val_mse'],reference)
            ledger.append(dict(model=kind,seed=42,val_mse=result['best_val_mse'],
                val_mae=result['best_val_mae'],reference_val_mse=reference,decision=decision,
                epochs=result['epochs_completed'],path=str(outdir(kind,42))))
            dump(HERE/'pilot_ledger.json',ledger);jobs.append((kind,42))
        immutable(HERE/'screening_lock.json',dict(ledger=ledger,metric='validation MSE only',
            threshold=reference,warning=EXPLORATORY_WARNING,source_hashes=source_hashes()))
        for row in ledger:
            if row['decision']!='append_seeds_43_44':
                continue
            for seed in (43,44):
                dump(HERE/'status.json',dict(state='additional_seed_training',model=row['model'],seed=seed,
                    completed_jobs=len(jobs),updated=time.time(),warning=EXPLORATORY_WARNING))
                train(row['model'],seed,device);jobs.append((row['model'],seed))
        immutable(HERE/'selection_lock.json',dict(checkpoints={str(outdir(k,s)):sha(outdir(k,s)/'best.pth')
            for k,s in jobs},sources=source_hashes(),warning=EXPLORATORY_WARNING))
        for i,(kind,seed) in enumerate(jobs):
            dump(HERE/'status.json',dict(state='locked_final_test',model=kind,seed=seed,
                tested_jobs=i,total_jobs=len(jobs),updated=time.time()))
            final_test(kind,seed,device)
        report=summary(ledger);protect_existing()
        dump(HERE/'status.json',dict(state='complete',jobs=len(jobs),summary=str(HERE/'summary.json'),
            protected_artifacts_unchanged=True,updated=time.time()))
        print(json.dumps(dict(complete=True,aggregate=report['aggregate'])),flush=True)


def launch(device):
    # A detached session survives the requesting shell/remote disconnection.
    with (HERE/'orchestrator.log').open('ab',buffering=0) as log:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run','--device',device],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1'))
    dump(HERE/'launch.json',dict(pid=process.pid,device=device,launched=time.time(),
        log=str(HERE/'orchestrator.log'),detached_session=True))
    print(json.dumps(dict(pid=process.pid,log=str(HERE/'orchestrator.log'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['run','launch','preflight'])
    parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args()
    if args.action=='launch':
        launch(args.device)
    elif args.action=='preflight':
        deterministic();preflight(args.device)
    else:
        try:
            run(args.device)
        except Exception as error:
            dump(HERE/'status.json',dict(state='failed',error=str(error),traceback=traceback.format_exc(),
                updated=time.time()))
            raise
