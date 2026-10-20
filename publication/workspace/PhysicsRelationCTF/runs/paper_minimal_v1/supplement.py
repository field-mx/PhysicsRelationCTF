"""Frozen, resumable publication controls; old sources/results remain read-only."""
import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REPLICATION=ROOT/'runs/multiseed_100_v1'
sys.path.insert(0,str(REPLICATION))
import replicate as previous

DATASETS=('industrial','lbnl','tep')
VARIANTS=('no_bias','no_equation','equation_input')
SEEDS=(42,43,44)
read=previous.read
sha=previous.sha
dump=previous.dump


def immutable_json(path,value):
    path=Path(path)
    value=json.loads(json.dumps(value))
    if path.exists():
        if read(path)!=value:raise RuntimeError(f'Refusing changed protocol: {path}')
    else:dump(path,value)


def jobs():
    return [dict(dataset=d,seed=s,variant=v) for s in SEEDS for d in DATASETS for v in VARIANTS]


def output(job):
    return HERE/job['dataset']/f"seed{job['seed']}"/'development'/job['variant']


def lock_checkpoints(paths):
    locked={}
    for path in paths:
        path=Path(path)
        if not (path/'paper_training.json').exists() or not (path/'best.pth').exists():
            raise RuntimeError(f'Incomplete training: {path}')
        locked[str(path)]=sha(path/'best.pth')
    return locked


@contextmanager
def exclusive_run(out):
    import fcntl
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.lock').open('a') as mutex:
        # An orphan worker can outlive its orchestrator. A replacement waits
        # here, then sees the durable completion marker instead of duplicating it.
        fcntl.flock(mutex,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(mutex,fcntl.LOCK_UN)


def finalize_artifacts(out,job):
    import torch
    variant=job['variant'];backend='CTF' if variant=='equation_input' else 'Physics'
    out=Path(out)
    for name in ('config.json','training.json'):
        path=out/name
        obj=read(path)
        obj.update(kind=variant,backend_kind=backend,variant=variant)
        if 'spec' in obj:obj['spec'].update(kind=variant,backend_kind=backend)
        dump(path,obj)
    ck=torch.load(out/'best.pth',map_location='cpu',weights_only=False)
    code={'no_bias':1,'no_equation':2,'equation_input':3}[variant]
    if int(ck['model']['_paper_control_id'])!=code:raise RuntimeError('Control checkpoint identity mismatch')
    ck.update(kind=variant,backend_kind=backend,control_variant=variant)
    if 'spec' in ck:ck['spec'].update(kind=variant,backend_kind=backend)
    temp=out/'best.identity.tmp';torch.save(ck,temp);temp.replace(out/'best.pth')
    dump(out/'paper_training.json',dict(read(out/'training.json'),**job))


def bank_function(dataset,graph):
    if dataset=='industrial':
        from physics import equation_features
        import torch
        def bank(raw):
            values=equation_features(raw,graph['channels'])
            return values,torch.isfinite(values)
        return bank
    module=importlib.import_module('public_physics_v4' if dataset=='lbnl' else 'tep_physics')
    return lambda raw:module.equation_bank(raw,graph)


def setup(dataset):
    """Development only. Does not open test observations."""
    import numpy as np
    import torch
    e=previous.engine(dataset)
    previous.prepare_engine(dataset,e)
    reference=previous.reference(dataset,'Physics')
    if dataset=='industrial':
        ref=read(reference/'config.json')
        ds,g,meta,selection,s=e.get_data(1.,sampling=ref['sampling'],subset_seed=ref['subset_seed'])
        cfg=ref['config'];options=ref['options']
    elif dataset=='lbnl':
        ref=read(reference/'config.json')
        ds={name:e.Windows(name,1.) for name in ('train','val')}
        g=read(reference/'graph.json');s=read(reference/'scalers.json')
        cfg=ref['config'];options=ref['options']
    else:
        ds,g,s=e.setup('ratio_20x20')
        ref=read(reference/'training.json')['spec']
        cfg=vars(e.configuration(20,20,5,ref['lr']))
        options=ref['options']
    return e,ds,g,s,cfg,options


def feature_scaler(dataset,ds,graph,scalers):
    """Use unique retained training observations, never validation or test."""
    import numpy as np
    import torch
    from controls import feature_statistics
    out=HERE/dataset/'feature_scaler.json'
    if out.exists():return read(out)
    train=ds['train']
    if dataset=='industrial':
        train_end=14540
        values=np.asarray(train.values[0,:train_end],dtype='float64')
        standardized=(values-np.asarray(train.mean))/np.asarray(train.scale)
        raw=standardized*np.asarray(scalers['raw_scale'])+np.asarray(scalers['raw_mean'])
    elif dataset=='lbnl':
        intervals=train.meta['selections']['1.0']['intervals']
        raw=np.concatenate([np.asarray(train.values[a:b]) for a,b in intervals]).astype('float64')
    else:
        standard=train.values.numpy().astype('float64')
        raw=(standard*np.asarray(scalers['scale'])+np.asarray(scalers['mean'])).reshape(-1,len(graph['channels']))
    bank=bank_function(dataset,graph)
    with torch.no_grad():
        q,v=bank(torch.from_numpy(raw))
        mean,scale=feature_statistics(q,v)
    result=dict(mean=mean.tolist(),scale=scale.tolist(),rows=len(raw),
        feature_count=len(mean),valid_fraction=v.double().mean(0).tolist(),
        policy='unique retained Train observations; valid-only zscore; constants: mean0, scale=max(abs(level),1)',
        graph_sha256=hashlib.sha256(json.dumps(graph,sort_keys=True).encode()).hexdigest())
    immutable_json(out,result)
    return result


def raw_scalers(dataset,s):
    return (s['raw_mean'],s['raw_scale']) if dataset=='industrial' else (s['mean'],s['scale'])


def construct(dataset,e,variant,seed,cfg,g,s,options):
    import torch
    from types import SimpleNamespace
    from controls import ablate,EquationInputCTF
    if dataset=='tep':
        e.base.seed_all(seed)
        base,physical=previous.seeded_tep_pair(e,seed,length=cfg['seq_len'],horizon=cfg['pred_len'],
            patch=cfg['patch_lens'][0],graph=g,scalers=s,lr=3e-4)
    else:
        e.seed_all(seed)
        base=e.OriginalCTF(SimpleNamespace(**cfg))
        physical=None
        if variant!='equation_input' and variant!='CTF':
            cls=e.PhysicsRelationCTF if dataset=='industrial' else e.PublicPhysicsCTF
            physical=cls(SimpleNamespace(**cfg),g,s,**options)
            e.copy_ctf(physical,base)
    if variant=='CTF':return base
    if variant=='Physics':return physical
    if variant in ('no_bias','no_equation'):return ablate(physical,variant)
    fs=read(HERE/dataset/'feature_scaler.json')
    mean,scale=raw_scalers(dataset,s)
    return EquationInputCTF(cfg,base,bank_function(dataset,g),mean,scale,fs['mean'],fs['scale'])


def install(dataset,e,variant,seed,cfg,g,s,options):
    if dataset=='tep':
        e.SEED=e.base.SEED=seed
        e.make_model=lambda spec,graph,scalers:construct(dataset,e,variant,seed,cfg,graph,scalers,options)
    else:
        e.make_model=lambda kind,config,graph,scalers,rng,opts:construct(
            dataset,e,variant,rng,config,graph,scalers,options)


def preflight(dataset,device):
    import numpy as np
    import torch
    previous.deterministic()
    e,ds,g,s,cfg,options=setup(dataset)
    fs=feature_scaler(dataset,ds,g,s)
    reference=previous.reference(dataset,'Physics')
    ck=torch.load(reference/'best.pth',map_location='cpu',weights_only=False)
    if ck['graph']!=g or ck['scalers']!=s:raise RuntimeError('Reference graph/scaler mismatch')
    x,y=ds['val'][0];x=x[None].to(device)
    full=construct(dataset,e,'Physics',42,cfg,g,s,options).eval().to(device)
    original_state={k:v.detach().cpu().clone() for k,v in full.state_dict().items()}
    baseline=construct(dataset,e,'CTF',42,cfg,g,s,options).eval().to(device)
    for name,value in baseline.state_dict().items():
        torch.testing.assert_close(original_state[name],value.cpu(),atol=0,rtol=0)
    full.set_strengths(0,0,0)
    with torch.no_grad():torch.testing.assert_close(full(x,None,None,None),baseline(x,None,None,None),atol=1e-6,rtol=1e-6)
    del full,baseline
    checks=[]
    for variant in VARIANTS:
        model=construct(dataset,e,variant,42,cfg,g,s,options).eval().to(device)
        if variant!='equation_input':
            for name,value in model.state_dict().items():
                if name=='_paper_control_id':continue
                torch.testing.assert_close(value.cpu(),original_state[name],atol=0,rtol=0)
        # Largest declared training batch: resource test and real backward.
        batch=64 if dataset=='tep' else 32
        xx=torch.stack([ds['train'][i][0] for i in range(batch)]).to(device)
        yy=torch.stack([ds['train'][i][1] for i in range(batch)]).to(device)
        torch.cuda.reset_peak_memory_stats(device)
        model.train();p=model(xx,None,None,None)
        if p.shape!=yy.shape or not torch.isfinite(p).all():raise RuntimeError('Shape/nonfinite preflight')
        loss=(p-yy).square().mean();loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError('Nonfinite gradient')
        if variant!='equation_input':
            index=0 if variant=='no_bias' else 1
            for layer in model.encoder.layers:
                a=layer.self_attention.inner_attention
                assert a.strengths()[index].item()==0
                assert a.context['bias'].max()>0
        model.eval()
        with torch.no_grad():
            prediction=model(x,None,None,None)
            torch.testing.assert_close(prediction,model(x,None,torch.randn_like(prediction),None),atol=0,rtol=0)
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        restored=construct(dataset,e,variant,42,cfg,g,s,options).eval().to(device)
        restored.load_state_dict(state)
        with torch.no_grad():torch.testing.assert_close(restored(x,None,None,None),prediction,atol=0,rtol=0)
        checks.append(dict(variant=variant,parameters=sum(p.numel() for p in model.parameters()),
            tokens=model.n_vars*(cfg['seq_len']//cfg['patch_lens'][0]+cfg['num_global_tokens']),
            max_allocated_mb=torch.cuda.max_memory_allocated(device)/1024**2,shape=list(prediction.shape)))
        del model,restored,p,loss,xx,yy;torch.cuda.empty_cache()
    immutable_json(HERE/dataset/'preflight.json',dict(dataset=dataset,checks=checks,
        feature_scaler=fs,assertions=['zero strengths native equivalence','shared initial backbone',
        'real batch backward','finite gradients','future decoder ignored','exact checkpoint reload']))
    print(dataset,'preflight passed',checks,flush=True)


def train(job,device):
    with exclusive_run(output(job)):
        if (output(job)/'paper_training.json').exists():return
        if not (output(job)/'training.json').exists():_train_locked(job,device)
        finalize_artifacts(output(job),job)


def _train_locked(job,device):
    previous.deterministic()
    dataset,variant,seed=job['dataset'],job['variant'],job['seed']
    e,ds,g,s,cfg,options=setup(dataset)
    out=output(job);out.mkdir(parents=True,exist_ok=True)
    immutable_json(out/'control.json',dict(job,reference=str(previous.reference(dataset,'Physics')),
        protocol_sha256=sha(HERE/'protocol.json'),train_budget=1.,frozen_options=options,
        interpretation='zero beta only' if variant=='no_bias' else
        'zero gamma only; rho residual and graph/bias retained' if variant=='no_equation' else
        'same equations as extra Train-standardized channels plus validity; no physical graph or branch'))
    install(dataset,e,variant,seed,cfg,g,s,options)
    kind='CTF' if variant=='equation_input' else 'Physics'
    if dataset=='tep':
        e.base.HERE=out.parent.parent
        e.base.dump=previous.isolated_dump(e.base.dump,e.base.HERE,out)
        e.base.EPOCHS=16;e.base.BATCH=64
        e.base.make_model=e.make_model;e.base.TEPPhysicsCTF=e.LengthPhysicsCTF
        e.base.record=e.record;e.base.evaluate=e.evaluate
        e.base.config=lambda c,lr:vars(e.configuration(20,20,5,lr))
        spec=copy.deepcopy(read(previous.reference(dataset,kind)/'training.json')['spec'])
        spec['name']=variant
        e.base.train_one(spec,ds,g,s,device)
    else:
        ref=read(previous.reference(dataset,kind)/'config.json')
        kwargs=dict(epochs=ref['epochs'],lr=ref['lr'])
        if dataset=='industrial':kwargs.update(sampling=ref['sampling'],subset_seed=ref['subset_seed'])
        e.train(kind,1.,seed,out,device,options if kind=='Physics' else {},**kwargs)


def evaluate(job,device):
    import numpy as np
    import torch
    previous.deterministic()
    out=output(job);lock=read(HERE/'selection_lock.json')
    if lock['checkpoints'].get(str(out))!=sha(out/'best.pth'):raise RuntimeError('Unfrozen checkpoint')
    if (out/'paper_metrics.json').exists():return
    dataset,variant,seed=job['dataset'],job['variant'],job['seed']
    e,ds,g,s,cfg,options=setup(dataset)
    install(dataset,e,variant,seed,cfg,g,s,options)
    if dataset=='industrial':result=e.test(out,device,HERE/'selection_lock.json')
    elif dataset=='lbnl':
        local=dict(checkpoints={str(out/'best.pth'):sha(out/'best.pth')},sources=e.source_hashes(),
            data={str(previous.PUB/'data'/name):sha(previous.PUB/'data'/name)
                  for name in ('metadata.json','values.npy','test_origins.npy')})
        immutable_json(out/'test_lock.json',local)
        result=e.test(out,device,out/'test_lock.json')
    else:
        ck=torch.load(out/'best.pth',map_location='cpu',weights_only=False)
        v=e.values('test',s);origins=e.origins(len(v),v.shape[1],20)
        ds=e.RealWindows(v,origins,20,20)
        model=e.make_model(ck['spec'],g,s).to(device);model.load_state_dict(ck['model'])
        result,arrays=e.evaluate(model,ds,device,True)
        np.savez_compressed(out/'test_predictions.npz',prediction=arrays[0],truth=arrays[1],
            origins=origins,channels=np.array(g['channels']))
        dump(out/'test_metrics.json',result)
    dump(out/'paper_metrics.json',dict(result,**job,kind=variant))


def protocol():
    old=read(REPLICATION/'protocol.json')['protected']
    for path,digest in old.items():
        if sha(path)!=digest:raise RuntimeError(f'Previous protected file changed: {path}')
    protected=dict(old)
    summary=read(REPLICATION/'summary.json')
    protected[str(REPLICATION/'summary.json')]=sha(REPLICATION/'summary.json')
    for item in summary['results'].values():
        for row in item['runs']:
            folder=Path(row['path'])
            for name in ('best.pth','config.json','training.json','replication_metrics.json','metrics.json','test_metrics.json'):
                path=folder/name
                if path.exists():protected[str(path)]=sha(path)
    protected[str(previous.PUB/'data/control_reference.json')]=sha(previous.PUB/'data/control_reference.json')
    new_sources={str(p):sha(p) for p in HERE.glob('*.py')}
    for dataset in DATASETS:
        for name in ('preflight.json','feature_scaler.json'):
            p=HERE/dataset/name
            if not p.exists():raise RuntimeError(f'Missing preflight: {p}')
            new_sources[str(p)]=sha(p)
    immutable_json(HERE/'protocol.json',dict(jobs=jobs(),protected=protected,new_sources=new_sources,
        policy='27 controls, all seeds and tests reported; no search; no full-model retraining',
        prior_summary=str(REPLICATION/'summary.json')))


def audit():
    p=read(HERE/'protocol.json')
    for path,digest in {**p['protected'],**p['new_sources']}.items():
        if sha(path)!=digest:raise RuntimeError(f'Frozen file changed: {path}')


def summarize():
    old=read(REPLICATION/'summary.json')['results'];results={}
    lines=['# 最小机制补实验','',
        '固定100%数据、三种子42/43/44，无调参；CTF及完整模型复用已有结果。',
        'no_bias只关beta；no_equation只关gamma（rho与图仍保留）；equation_input是等方程信息输入对照，不是参数量匹配消融。','',
        '| 数据集 | 方法 | MSE mean ± std | MAE mean ± std | 相对CTF MSE改善 |',
        '|---|---|---:|---:|---:|']
    for dataset in DATASETS:
        results[dataset]={}
        for variant in ('CTF','Physics',*VARIANTS):
            if variant in ('CTF','Physics'):
                rows=[r for r in old[dataset]['runs'] if r['kind']==variant]
            else:
                rows=[dict(read(output(dict(dataset=dataset,seed=s,variant=variant))/'paper_metrics.json'),
                           path=str(output(dict(dataset=dataset,seed=s,variant=variant)))) for s in SEEDS]
            stats={f'{m}_{k}':fn([r[m] for r in rows]) for m in ('mse','mae')
                   for k,fn in [('mean',statistics.mean),('std',statistics.stdev)]}
            c=old[dataset]['CTF']['mse_mean']
            stats['gain_percent']=100*(c-stats['mse_mean'])/c
            results[dataset][variant]=dict(stats,rows=rows)
            lines.append(f"|{dataset}|{variant}|{stats['mse_mean']:.8f} ± {stats['mse_std']:.8f}|"
                f"{stats['mae_mean']:.8f} ± {stats['mae_std']:.8f}|{stats['gain_percent']:+.3f}%|")
    lines+=['','各随机种子及逐通道/逐时距结果见summary.json与各run中的paper_metrics.json。',
        '弱化或失败的消融结果也须报告；不能由全模型胜出直接推断每个模块有效。',
        '缺少伪图、非物理等参数对照和其他预测器，本批不是全部投稿实验。','']
    dump(HERE/'summary.json',dict(results=results,seeds=SEEDS,status='complete'))
    (HERE/'RESULTS.md').write_text('\n'.join(lines))


def child(args,log):
    with (HERE/log).open('ab',buffering=0) as f:
        return subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),*args],
            cwd=HERE,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)


def orchestrate():
    import fcntl
    with (HERE/'orchestrator.lock').open('w') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol();audit()
        pending=[j for j in jobs() if not (output(j)/'paper_training.json').exists()]
        completed=27-len(pending);running=[]
        while pending or running:
            for device in ('cuda:0','cuda:1'):
                if any(d==device for _,_,d in running):continue
                allowed=[j for j in pending if device=='cuda:1' or j['variant']!='equation_input']
                if not allowed:continue
                # Bigger extra-input models use the GPU without the existing ComfyUI workload.
                j=next((j for j in allowed if device=='cuda:1' and j['variant']=='equation_input'),allowed[0])
                pending.remove(j)
                p=child(['train','--dataset',j['dataset'],'--variant',j['variant'],
                    '--seed',str(j['seed']),'--device',device],f"{j['dataset']}_{j['variant']}_{j['seed']}.log")
                running.append((j,p,device))
            dump(HERE/'status.json',dict(state='training',completed_trainings=completed,total_trainings=27,
                running=[dict(j,pid=p.pid,device=d) for j,p,d in running],updated=time.time(),pid=os.getpid()))
            time.sleep(10)
            for item in running[:]:
                j,p,d=item
                if p.poll() is None:continue
                running.remove(item)
                if p.returncode:
                    dump(HERE/'status.json',dict(state='draining_after_failure',failed=j,
                        running_pids=[q.pid for _,q,_ in running],updated=time.time()))
                    previous.drain_workers(running)
                    raise RuntimeError(f'Training execution failed: {j}')
                completed+=1
        audit()
        immutable_json(HERE/'selection_lock.json',dict(checkpoints=lock_checkpoints([output(j) for j in jobs()]),
            protocol_sha256=sha(HERE/'protocol.json'),policy='All 27 validation-selected checkpoints locked before test'))
        for i,j in enumerate(jobs()):
            dump(HERE/'status.json',dict(state='testing',completed_trainings=27,completed_tests=i,total_tests=27,updated=time.time()))
            p=child(['test','--dataset',j['dataset'],'--variant',j['variant'],
                '--seed',str(j['seed']),'--device','cuda:1'],f"test_{j['dataset']}_{j['variant']}_{j['seed']}.log")
            if p.wait():raise RuntimeError(f'Test execution failed: {j}')
        audit();summarize()
        dump(HERE/'status.json',dict(state='complete',completed_trainings=27,completed_tests=27,
            reused_runs=18,old_artifacts_unchanged=True,updated=time.time()))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','train','test','orchestrate','summary'])
    p.add_argument('--dataset',choices=DATASETS);p.add_argument('--variant',choices=VARIANTS)
    p.add_argument('--seed',type=int,choices=SEEDS,default=42);p.add_argument('--device',default='cuda:1')
    args=p.parse_args()
    if args.action=='preflight':preflight(args.dataset,args.device)
    elif args.action=='orchestrate':orchestrate()
    elif args.action=='summary':audit();summarize()
    else:
        job=dict(dataset=args.dataset,variant=args.variant,seed=args.seed)
        (train if args.action=='train' else evaluate)(job,args.device)


if __name__=='__main__':main()
