"""Fixed paired seed replication. Old code/results are read-only inputs."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PUB = ROOT / 'runs/publicdata_exp'
TEP = ROOT / 'runs/TEP/window_search_v1'
DATASETS = ('industrial', 'lbnl', 'tep')


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def jobs():
    return [dict(dataset=d, seed=s, kind=k, ratio=1.0)
            for s in (43, 44) for d in DATASETS for k in ('CTF', 'Physics')]


def reference(dataset, kind):
    if dataset == 'industrial':
        return ROOT / f'runs/random_gaps_v1/train100/{kind}_seed42'
    if dataset == 'lbnl':
        name = 'common_lr3e4' if kind == 'CTF' else 'v7_lr3e4'
        return PUB / f'optimization_v2/{name}/train100/{kind}_seed42'
    return TEP / f'ratio_20x20/development/{kind}'


def output(job):
    if job['dataset'] == 'tep':
        return HERE / f"tep/seed{job['seed']}/ratio_20x20/development/{job['kind']}"
    return HERE / f"{job['dataset']}/train100/{job['kind']}_seed{job['seed']}"


def seed42_metric_path(dataset,kind):
    if dataset=='tep':return TEP/f'final/ratio_20x20/{kind}/metrics.json'
    return reference(dataset,kind)/('metrics.json' if dataset=='industrial' else 'test_metrics.json')


def aggregate(rows):
    if len(rows) != 6 or {(r['seed'], r['kind']) for r in rows} != {
            (s,k) for s in (42,43,44) for k in ('CTF','Physics')}:
        raise ValueError('Require every paired seed42/43/44, without selection')
    result = {}
    for kind in ('CTF', 'Physics'):
        chosen = [r for r in rows if r['kind'] == kind]
        result[kind] = {f'{metric}_{stat}': fn([r[metric] for r in chosen])
                        for metric in ('mse','mae')
                        for stat,fn in [('mean',statistics.mean),('std',statistics.stdev)]}
    result['paired'] = []
    for seed in (42,43,44):
        c,p = [next(r for r in rows if r['kind'] == k and r['seed'] == seed)
               for k in ('CTF','Physics')]
        result['paired'].append(dict(seed=seed, ctf_mse=c['mse'], physics_mse=p['mse'],
            ctf_mae=c['mae'], physics_mae=p['mae'],
            mse_gain_percent=100*(c['mse']-p['mse'])/c['mse'],
            mae_gain_percent=100*(c['mae']-p['mae'])/c['mae']))
    result['mse_wins'] = sum(r['mse_gain_percent'] > 0 for r in result['paired'])
    result['mae_wins'] = sum(r['mae_gain_percent'] > 0 for r in result['paired'])
    for m in ('mse','mae'):
        c,p = result['CTF'][m+'_mean'],result['Physics'][m+'_mean']
        result[m+'_gain_percent'] = 100*(c-p)/c
    return result


def deterministic():
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def engine(dataset):
    if dataset == 'industrial':
        sys.path.insert(0, str(ROOT))
        return importlib.import_module('run_experiment')
    if dataset == 'lbnl':
        sys.path[:0] = [str(PUB/'optimization_v2'), str(PUB)]
        return importlib.import_module('train_v7').engine
    sys.path.insert(0, str(TEP))
    return importlib.import_module('runner')


def seeded_tep_pair(module, seed, **kwargs):
    # Identical to old pair(), except replacing its literal manual_seed(42).
    import torch
    lm = importlib.import_module('length_model')
    torch.manual_seed(seed)
    cfg = lm.configuration(kwargs['length'], kwargs['horizon'], kwargs['patch'], kwargs['lr'])
    c = lm.OriginalCTF(cfg)
    p = lm.LengthPhysicsCTF(cfg, kwargs['graph'], kwargs['scalers'])
    lm.copy_ctf(p, c)
    return c,p


def configure_tep(e, seed):
    e.SEED = seed
    e.base.SEED = seed
    e.HERE = HERE / f'tep/seed{seed}'
    e.HERE.mkdir(parents=True, exist_ok=True)
    e.GROUPS = {'ratio_20x20': dict(length=20,horizon=20,patch=5)}
    e.pair = lambda **kw: seeded_tep_pair(e, seed, **kw)


def prepare_engine(dataset,e):
    if dataset=='industrial':
        original=e.get_data
        def get_data(*args,**kwargs):
            d,g,m,s,sc=original(*args,**kwargs)
            return d,g,m,json.loads(json.dumps(s)),sc
        e.get_data=get_data
    elif dataset=='lbnl':
        original=e.build_graph
        intervals=read(PUB/'data/metadata.json')['selections']['1.0']['intervals']
        def build_graph(channels):
            g=original(channels)
            g['retained_segment_lengths']=[b-a for a,b in intervals]
            return g
        e.build_graph=build_graph


def isolated_dump(original,root,out):
    def save(path,obj):
        path=Path(path)
        return original(out/'status.json' if path==root/'status.json' else path,obj)
    return save


def drain_workers(running):
    # Let the other authorized training finish and save; do not release the
    # orchestrator lock while an untracked writer could race a restart.
    for _,process,_ in running:process.wait()


def preflight(dataset):
    import torch
    import numpy as np
    deterministic()
    e = engine(dataset)
    old = reference(dataset, 'Physics')
    ck = torch.load(old/'best.pth', map_location='cpu', weights_only=False)
    if dataset != 'tep':
        for kind in ('CTF','Physics'):
            cfg = read(reference(dataset,kind)/'config.json')
            hashes = cfg.get('source_hashes', cfg.get('sources', {}))
            for name,digest in hashes.items():
                path = Path(name) if Path(name).is_absolute() else ROOT/name
                if sha(path) != digest:
                    raise RuntimeError(f'Seed42 source has changed: {path}')
        cfg = read(old/'config.json')
        def pair(seed):
            return tuple(e.make_model(k, ck['config'], ck['graph'], ck['scalers'], seed,
                                     {} if k=='CTF' else ck['options']) for k in ('CTF','Physics'))
        if dataset == 'industrial':
            ds,g,meta,sel,sc = e.get_data(1.,sampling='random_gaps',subset_seed=20260922)
            assert json.loads(json.dumps(sel)) == cfg['selection'] and g == ck['graph'] and sc == ck['scalers']
            x,y = ds['val'][0]
            windows = {k:len(v) for k,v in ds.items()}
        else:
            for name in ['data/metadata.json','data/values.npy']:
                expected = cfg['data_metadata_sha256' if name.endswith('.json') else 'values_sha256']
                assert sha(PUB/name) == expected
            ds = {s:e.Windows(s,1.) for s in ('train','val','test')}
            x,y = ds['val'][0]
            windows = {k:len(v) for k,v in ds.items()}
    else:
        frozen = read(TEP/'final/frozen_manifest.json')
        for path,digest in {**frozen['code'], **frozen['data']}.items():
            assert sha(path) == digest, f'TEP reference changed: {path}'
        ds,g,sc = e.setup('ratio_20x20')
        pair = lambda seed: seeded_tep_pair(e, seed, **e.GROUPS['ratio_20x20'],graph=g,scalers=sc,lr=3e-4)
        original = e.pair(**e.GROUPS['ratio_20x20'],graph=g,scalers=sc,lr=3e-4)
        patched = pair(42)
        for a,b in zip(original,patched):
            assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()), 'seed42 changed'
        del original,patched
        x,y = ds['val'][0]
        windows = {k:len(v) for k,v in ds.items()}
        frozen_ds = read(TEP/'ratio_20x20/data_manifest.json')
        for split in ('train','val'):
            assert hashlib.sha256(ds[split].origins.tobytes()).hexdigest() == frozen_ds['origin_sha256'][split]
    signatures = {}
    for seed in (42,43,44):
        c,p = pair(seed)
        cs,ps = c.state_dict(),p.state_dict()
        assert all(k in ps and torch.equal(v,ps[k]) for k,v in cs.items()), 'Unpaired CTF weights'
        h = hashlib.sha256()
        for k,v in cs.items():
            h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
        signatures[str(seed)] = h.hexdigest()
        c.eval();p.eval()
        with torch.no_grad():
            for m in (c,p):
                result = m(x[None],None,None,None)
                assert result.shape == y[None].shape and torch.isfinite(result).all()
        del c,p
    assert len(set(signatures.values())) == 3, 'Seed was silently fixed'
    dump(HERE/dataset/'preflight.json',dict(status='passed',windows=windows,
        shape=[list(x.shape),list(y.shape)],initialization_hashes=signatures,
        checks=['source/data provenance','paired initial CTF weights','distinct seeds',
                'finite forward and exact target shape','no change to source model'],
        seed42_initialization_exact=(dataset=='tep')))
    print(dataset, 'preflight passed', signatures, flush=True)


def train_job(job, device):
    deterministic()
    e = engine(job['dataset'])
    prepare_engine(job['dataset'],e)
    seed,kind = job['seed'],job['kind']
    if job['dataset'] == 'tep':
        configure_tep(e,seed)
        # train_group runs both methods, so configure its original engine directly
        # to run just this job without introducing a different training loop.
        settings = e.GROUPS['ratio_20x20']
        ds,g,s = e.setup('ratio_20x20')
        e.base.HERE = e.HERE/'ratio_20x20'
        e.base.dump = isolated_dump(e.base.dump,e.base.HERE,output(job))
        e.base.EPOCHS = 16;e.base.BATCH = e.BATCH
        e.base.make_model = e.make_model;e.base.TEPPhysicsCTF = e.LengthPhysicsCTF
        e.base.record = e.record;e.base.evaluate = e.evaluate
        e.base.config = lambda c,lr: vars(e.configuration(**settings,lr=lr))
        spec = read(reference('tep',kind)/'training.json')['spec']
        e.base.train_one(spec,ds,g,s,device)
    else:
        cfg = read(reference(job['dataset'],kind)/'config.json')
        kwargs = dict(epochs=cfg['epochs'],lr=cfg['lr'])
        if job['dataset']=='industrial':
            kwargs.update(sampling=cfg['sampling'],subset_seed=cfg['subset_seed'])
        e.train(kind,1.,seed,output(job),device,cfg['options'],**kwargs)


def freeze():
    manifest = read(HERE/'protocol.json')
    for p,h in manifest['protected'].items():
        if sha(p)!=h:raise RuntimeError(f'Old artifact/source changed: {p}')
    checks = {}
    for j in jobs():
        path = output(j)
        if not (path/'training.json').exists():raise RuntimeError(f'Incomplete training: {path}')
        checks[str(path)] = sha(path/'best.pth')
        if j['dataset']!='tep':
            for name in ('graph.json','scalers.json'):
                assert read(path/name)==read(reference(j['dataset'],j['kind'])/name), f'{name} changed'
        else:
            import torch
            ck=torch.load(path/'best.pth',map_location='cpu',weights_only=False)
            old=torch.load(reference('tep',j['kind'])/'best.pth',map_location='cpu',weights_only=False)
            assert ck['graph']==old['graph'] and ck['scalers']==old['scalers'] and ck['spec']==old['spec']
    locked = dict(checkpoints=checks,protocol_sha256=sha(HERE/'protocol.json'),
                  policy='All 12 validation-selected checkpoints locked before test. No search or seed removal.')
    path=HERE/'selection_lock.json'
    if path.exists():assert read(path)==locked
    else:dump(path,locked)


def evaluate_dataset(dataset,device):
    import torch
    import numpy as np
    deterministic()
    e=engine(dataset);lock=read(HERE/'selection_lock.json')
    for seed in (43,44):
        for kind in ('CTF','Physics'):
            job=dict(dataset=dataset,seed=seed,kind=kind)
            out=output(job)
            assert sha(out/'best.pth')==lock['checkpoints'][str(out)]
            target=out/'replication_metrics.json'
            if target.exists():continue
            if dataset=='industrial':
                result=e.test(out,device,HERE/'selection_lock.json')
            elif dataset=='lbnl':
                local=dict(checkpoints={str(out/'best.pth'):sha(out/'best.pth')},sources=e.source_hashes(),
                           data={str(PUB/'data'/n):sha(PUB/'data'/n) for n in ['metadata.json','values.npy','test_origins.npy']})
                lp=out/'test_lock.json';dump(lp,local)
                result=e.test(out,device,lp)
            else:
                configure_tep(e,seed)
                ck=torch.load(out/'best.pth',map_location='cpu',weights_only=False)
                v=e.values('test',ck['scalers']);o=e.origins(len(v),v.shape[1],20)
                ds=e.RealWindows(v,o,20,20)
                model=e.make_model(ck['spec'],ck['graph'],ck['scalers']).to(device)
                model.load_state_dict(ck['model'])
                result,arr=e.evaluate(model,ds,device,True)
                np.savez_compressed(out/'test_predictions.npz',prediction=arr[0],truth=arr[1],origins=o,
                                    channels=np.array(ck['graph']['channels']))
                dump(out/'test_metrics.json',result)
                del model;torch.cuda.empty_cache()
            dump(target,dict(result,seed=seed,kind=kind,dataset=dataset))


def summarize():
    results={}
    for dataset in DATASETS:
        rows=[]
        for seed in (42,43,44):
            for kind in ('CTF','Physics'):
                if seed==42:
                    path=reference(dataset,kind)
                    metric=read(seed42_metric_path(dataset,kind))
                else:
                    path=output(dict(dataset=dataset,seed=seed,kind=kind))
                    metric=read(path/'replication_metrics.json')
                rows.append(dict(seed=seed,kind=kind,mse=metric['mse'],mae=metric['mae'],path=str(path)))
        results[dataset]=dict(aggregate(rows),runs=rows)
    dump(HERE/'summary.json',dict(status='complete',seeds=[42,43,44],ratio=1.,results=results))
    lines=['# 三数据集固定配置多种子复验','',
           '100%训练池；保留seed42；新增43、44；没有选种子、调参或更改测试集。',
           '标准差为三个训练种子的样本标准差(ddof=1)，不是测试窗口bootstrap。','',
           '| 数据集 | CTF MSE mean ± std | Physics MSE mean ± std | MSE改善 | MSE胜出种子 |',
           '|---|---:|---:|---:|---:|']
    for d,r in results.items():
        c,p=r['CTF'],r['Physics']
        lines.append(f"| {d} | {c['mse_mean']:.8f} ± {c['mse_std']:.8f} | {p['mse_mean']:.8f} ± {p['mse_std']:.8f} | {r['mse_gain_percent']:+.3f}% | {r['mse_wins']}/3 |")
    for d,r in results.items():
        lines += ['',f'## {d}','', '| seed | CTF MSE | Physics MSE | MSE改善 | CTF MAE | Physics MAE | MAE改善 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for p in r['paired']:
            lines.append(f"| {p['seed']} | {p['ctf_mse']:.8f} | {p['physics_mse']:.8f} | {p['mse_gain_percent']:+.3f}% | {p['ctf_mae']:.8f} | {p['physics_mae']:.8f} | {p['mae_gain_percent']:+.3f}% |")
        lines += ['',f"CTF MAE: {r['CTF']['mae_mean']:.8f} ± {r['CTF']['mae_std']:.8f}; Physics MAE: {r['Physics']['mae_mean']:.8f} ± {r['Physics']['mae_std']:.8f}."]
    lines += ['', '限制：seed42曾用于开发，新增种子复验不等于新的盲测或工况泛化；三数据集指标不可直接合并。',
              '源代码、数据和旧checkpoint指纹见protocol.json；新checkpoint冻结清单见selection_lock.json。','']
    (HERE/'RESULTS.md').write_text('\n'.join(lines))
    return results


def protect_protocol():
    protected={}
    for folder in [ROOT,ROOT.parent/'PhysicsGuidedCTF',ROOT.parent/'CTF/ChannelTokenFormer/models',
                   ROOT.parent/'CTF/ChannelTokenFormer/layers',ROOT.parent/'CTF/ChannelTokenFormer/utils']:
        for p in folder.glob('*.py'):protected[str(p)]=sha(p)
    for folder in [PUB,PUB/'optimization_v2',TEP,TEP.parent]:
        for p in folder.glob('*.py'):protected[str(p)]=sha(p)
    for p in (ROOT.parent/'GNN_Transformer').glob('equation_topology*.py'):protected[str(p)]=sha(p)
    for d in DATASETS:
        for k in ('CTF','Physics'):
            p=seed42_metric_path(d,k);protected[str(p)]=sha(p)
            for name in ['best.pth','config.json','training.json']:
                p=reference(d,k)/name;protected[str(p)]=sha(p)
    for folder in [PUB/'data',ROOT.parent/'PhysicsGuidedCTF/data/tep',ROOT.parent/'PhysicsGuidedCTF/data/industrial']:
        for p in folder.glob('*'):
            if p.suffix in ('.npy','.json','.npz'):protected[str(p)]=sha(p)
    protected[str(HERE/'replicate.py')]=sha(HERE/'replicate.py')
    manifest=dict(jobs=jobs(),reuse_seed42={d:{k:str(reference(d,k)) for k in ('CTF','Physics')} for d in DATASETS},
                  protected=protected,policy='100% only; no hyperparameter search; every seed reported',
                  tasks=dict(industrial=[96,96],lbnl=[96,96],tep=[20,20]))
    p=HERE/'protocol.json'
    if p.exists():assert read(p)==manifest, 'Frozen protocol changed'
    else:dump(p,manifest)


def child(args,logname):
    with (HERE/logname).open('ab',buffering=0) as log:
        return subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),*args],
            cwd=HERE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)


def orchestrate():
    import fcntl
    with (HERE/'orchestrator.lock').open('w') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protect_protocol()
        dump(HERE/'status.json',dict(state='preflight',pid=os.getpid(),total_trainings=12))
        for d in DATASETS:
            proc=child(['preflight','--dataset',d],f'preflight_{d}.log')
            if proc.wait():raise RuntimeError(f'{d} preflight failed')
        pending=jobs();running=[];completed=[]
        while pending or running:
            for device in ('cuda:0','cuda:1'):
                if pending and not any(item[2]==device for item in running):
                    j=pending.pop(0)
                    proc=child(['train','--dataset',j['dataset'],'--seed',str(j['seed']),
                                '--kind',j['kind'],'--device',device],f"{j['dataset']}_{j['kind']}_seed{j['seed']}.log")
                    running.append((j,proc,device))
            dump(HERE/'status.json',dict(state='training',completed=len(completed),total=12,
                running=[dict(j,pid=p.pid,device=d) for j,p,d in running],updated=time.time()))
            time.sleep(10)
            for item in running[:]:
                j,p,d=item
                if p.poll() is not None:
                    running.remove(item)
                    if p.returncode:
                        dump(HERE/'status.json',dict(state='draining_after_failure',failed=j,
                            remaining_pids=[q.pid for _,q,_ in running],updated=time.time()))
                        drain_workers(running)
                        raise RuntimeError(f'Training failed (not a scientific loss): {j}')
                    completed.append(j)
        freeze()
        dump(HERE/'status.json',dict(state='testing',completed_trainings=12,updated=time.time()))
        for d in DATASETS:
            proc=child(['test','--dataset',d,'--device','cuda:1'],f'test_{d}.log')
            if proc.wait():raise RuntimeError(f'Test failed: {d}')
        summarize()
        for p,h in read(HERE/'protocol.json')['protected'].items():
            if sha(p)!=h:raise RuntimeError(f'Protected artifact changed: {p}')
        dump(HERE/'status.json',dict(state='complete',completed_trainings=12,evaluated_new_models=12,
                                    retained_seed42_models=6,updated=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['preflight','train','test','all','summary'])
    p.add_argument('--dataset',choices=DATASETS)
    p.add_argument('--seed',type=int,choices=[43,44])
    p.add_argument('--kind',choices=['CTF','Physics'])
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args()
    try:
        if a.stage=='preflight':preflight(a.dataset)
        elif a.stage=='train':train_job(dict(dataset=a.dataset,seed=a.seed,kind=a.kind),a.device)
        elif a.stage=='test':evaluate_dataset(a.dataset,a.device)
        elif a.stage=='summary':summarize()
        else:orchestrate()
    except Exception as exc:
        if a.stage=='all':dump(HERE/'status.json',dict(state='failed',error=repr(exc),updated=time.time()))
        raise
