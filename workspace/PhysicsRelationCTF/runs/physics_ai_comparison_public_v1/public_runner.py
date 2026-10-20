"""Detached public comparison queue with frozen evaluation and combined report."""
import argparse
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib
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
sys.path[:0] = [str(HERE), str(ROOT/'runs/paper_minimal_v1')]
import supplement
from public_models import PartialPhysicsLoss, forecast_loss, make_model

MODELS = ('PhyGNN', 'MTGNN', 'PatchTST', 'PG-Reg-CTF')
DATASETS = ('lbnl', 'tep')
SEEDS = (42, 43, 44)
read = supplement.read
sha = supplement.sha


def dump(path, value):
    path = Path(path).resolve(); path.relative_to(HERE)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def immutable(path, value):
    value = json.loads(json.dumps(value))
    if Path(path).exists():
        if read(path) != value: raise RuntimeError(f'Changed frozen protocol: {path}')
    else: dump(path, value)


def checkpoint(path, value):
    path = Path(path).resolve(); path.relative_to(HERE)
    tmp = path.with_suffix('.tmp'); torch.save(value, tmp); tmp.replace(path)


@contextmanager
def exclusive(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def jobs():
    return [dict(dataset=d, model=k, seed=s) for s in SEEDS for d in DATASETS for k in MODELS]


def options(dataset, kind):
    return {'PhyGNN': dict(graph_heads=[4, 4], temporal_hidden=128),
            'MTGNN': dict(neighbors=20, dropout=.1),
            'PatchTST': dict(stride=12 if dataset == 'lbnl' else 5),
            'PG-Reg-CTF': dict(physics_weight=.01)}[kind].copy()


def outdir(job):
    return HERE/job['dataset']/f"seed{job['seed']}"/job['model']


@lru_cache(maxsize=2)
def load_data(dataset):
    if dataset not in DATASETS: raise ValueError(dataset)
    engine, ds, graph, scalers, cfg, _ = supplement.setup(dataset)
    train = ds['train']
    mean, scale = supplement.raw_scalers(dataset, scalers)
    if dataset == 'lbnl':
        intervals = train.meta['selections']['1.0']['intervals']
        raw = np.concatenate([np.asarray(train.values[a:b]) for a, b in intervals]).astype('float64')
        lengths = [b-a for a, b in intervals]
        targets = train.meta['target_indices']
        reference = read(supplement.previous.reference(dataset, 'Physics')/'config.json')
        policy = dict(epochs=reference['epochs'], lr=reference['lr'], batch=32, min_epochs=8, patience=6)
        for key, filename in [('data_metadata_sha256', 'metadata.json'), ('values_sha256', 'values.npy')]:
            if key in reference and sha(supplement.previous.PUB/'data'/filename) != reference[key]:
                raise RuntimeError('LBNL frozen data mismatch')
        expected_origins = np.load(supplement.previous.PUB/'data/train_1.0_origins.npy')
        if not np.array_equal(train.origins, expected_origins): raise RuntimeError('LBNL origin mismatch')
        reference_scale = np.asarray(train.meta['reference_scale'], dtype=np.float32)
        evaluation_scale = np.asarray(train.scale, dtype=np.float32)/reference_scale
    else:
        standard = train.values.numpy().astype('float64')
        raw = (standard*np.asarray(scale)+np.asarray(mean)).reshape(-1, len(graph['channels']))
        lengths = [standard.shape[1]]*standard.shape[0]
        targets = list(range(len(graph['channels'])))
        reference = read(supplement.previous.reference(dataset, 'Physics')/'training.json')['spec']
        policy = dict(epochs=16, lr=reference['lr'], batch=64, min_epochs=10, patience=4)
        frozen = read(supplement.previous.TEP/'ratio_20x20/data_manifest.json')
        for name in ('train', 'val'):
            if hashlib.sha256(ds[name].origins.tobytes()).hexdigest() != frozen['origin_sha256'][name]:
                raise RuntimeError('TEP origin mismatch')
        evaluation_scale = np.ones(len(graph['channels']), dtype=np.float32)
    boundaries = np.cumsum([0]+lengths)
    segments = [(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:])]
    fs = read(ROOT/f'runs/paper_minimal_v1/{dataset}/feature_scaler.json')
    bank = supplement.bank_function(dataset, graph)
    selection = dict(dataset=dataset, train_ratio=1., windows={k: len(v) for k, v in ds.items()},
                     input=cfg['seq_len'], output=cfg['pred_len'], targets=targets,
                     origin_sha256={k: hashlib.sha256(v.origins.tobytes()).hexdigest() for k, v in ds.items()},
                     graph_sha256=hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest())
    return dict(dataset=dataset, engine=engine, datasets=ds, graph=graph, scalers=scalers,
                config=cfg, raw_mean=mean, raw_scale=scale, raw_train=raw, segments=segments,
                bank=bank, feature_scaler=fs, targets=targets, policy=policy,
                evaluation_scale=evaluation_scale, selection=selection)


def test_dataset(dataset, data):
    engine = data['engine']
    if dataset == 'lbnl': return engine.Windows('test', 1.)
    values = engine.values('test', data['scalers'])
    origins = engine.origins(len(values), values.shape[1], 20)
    return engine.RealWindows(values, origins, 20, 20)


def deterministic():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def source_hashes():
    files = list(HERE.glob('*.py'))
    files += list((ROOT/'runs/physics_ai_comparison_v1').glob('*.py'))
    files += list(ROOT.glob('*.py'))
    files += list((ROOT/'runs/publicdata_exp').glob('*.py'))
    files += list((ROOT/'runs/publicdata_exp/optimization_v2').glob('*.py'))
    files += list((ROOT/'runs/TEP').glob('*.py'))
    files += list((ROOT/'runs/TEP/window_search_v1').glob('*.py'))
    files += list((ROOT/'runs/paper_minimal_v1').glob('*.py'))
    return {str(p): sha(p) for p in files}


def protect_existing():
    path = HERE/'protected_existing.json'
    if path.exists():
        protected = read(path)
        for file, digest in protected.items():
            if sha(file) != digest: raise RuntimeError(f'Existing artifact changed: {file}')
        return protected
    protected = {}
    for manifest, key in [(ROOT/'runs/physics_ai_comparison_v1/protected_existing.json', None),
                          (ROOT/'runs/paper_minimal_v1/protocol.json', 'protected')]:
        value = read(manifest); protected.update(value if key is None else value[key])
    for p, digest in protected.items():
        if sha(p) != digest: raise RuntimeError(f'Previous protected file changed: {p}')
    roots = [ROOT/'runs/physics_ai_comparison_v1', ROOT/'runs/paper_minimal_v1', ROOT/'runs/multiseed_100_v1']
    for directory in roots:
        for name in ('summary.json', 'RESULTS.md', 'results.csv'):
            p = directory/name
            if p.exists(): protected[str(p)] = sha(p)
    for dataset in DATASETS:
        for name in ('feature_scaler.json',):
            p = ROOT/'runs/paper_minimal_v1'/dataset/name; protected[str(p)] = sha(p)
        for seed in SEEDS:
            p = ROOT/f'runs/paper_minimal_v1/{dataset}/seed{seed}/development/equation_input'
            for name in ('best.pth', 'config.json', 'training.json', 'paper_metrics.json'):
                protected[str(p/name)] = sha(p/name)
    # Public prepared data and official baseline source are also immutable.
    for folder in [supplement.previous.PUB/'data', ROOT.parent/'PhysicsGuidedCTF/data/tep',
                   ROOT/'runs/physics_ai_comparison_v1/vendor']:
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix in ('.py', '.json', '.npy') and '.git' not in p.parts:
                protected[str(p)] = sha(p)
    for p, digest in source_hashes().items():
        if not Path(p).is_relative_to(HERE): protected[p] = digest
    immutable(path, protected)
    return protected


@torch.no_grad()
def evaluate(model, dataset, data, device, save=False):
    model.eval(); channels = len(data['raw_mean']); horizon = data['config']['pred_len']
    se = torch.zeros(horizon, channels, dtype=torch.float64, device=device); ae = se.clone()
    factor = torch.tensor(data['evaluation_scale'], device=device)[None, None]
    predictions = []; truths = []
    for x, y in DataLoader(dataset, batch_size=data['policy']['batch'], shuffle=False):
        x, y = x.to(device), y.to(device); p = model(x, None, None, None)
        if p.shape != y.shape or not torch.isfinite(p).all(): raise RuntimeError('Invalid forecast')
        error = (p.double()-y.double())*factor.double()
        se += error.square().sum(0); ae += error.abs().sum(0)
        if save: predictions.append(p.cpu().numpy()); truths.append(y.cpu().numpy())
    mse = (se/len(dataset)).cpu().numpy(); mae = (ae/len(dataset)).cpu().numpy(); targets = data['targets']
    score = dict(mse=float(mse[:, targets].mean()), mae=float(mae[:, targets].mean()), windows=len(dataset),
        per_channel_mse=mse.mean(0).tolist(), per_channel_mae=mae.mean(0).tolist(),
        horizon_mse=mse[:, targets].mean(1).tolist(), horizon_mae=mae[:, targets].mean(1).tolist(),
        channels=data['graph']['channels'], target_indices=targets)
    return score, (np.concatenate(predictions), np.concatenate(truths)) if save else None


def preflight(dataset, device):
    data = load_data(dataset); batch = data['policy']['batch']; checks = []
    x = torch.stack([data['datasets']['train'][i][0] for i in range(batch)]).to(device)
    y = torch.stack([data['datasets']['train'][i][1] for i in range(batch)]).to(device)
    for kind in MODELS:
        model = make_model(kind, data['config'], data, 42, options(dataset, kind)).to(device)
        physical = PartialPhysicsLoss(data).to(device) if kind == 'PG-Reg-CTF' else None
        torch.cuda.reset_peak_memory_stats(device); tic = time.monotonic()
        model.train(); p = model(x, None, None, None); loss = forecast_loss(p, y, data['targets'])
        if physical is not None:
            loss = loss + .01*physical(x, p)
            dump(HERE/dataset/'physics_constraints.json', dict(routes=physical.routes,
                 note='Train-calibrated tolerance-aware incomplete equation proxies; not a closed PINN'))
        loss.backward()
        gradients = [v.grad for v in model.parameters() if v.grad is not None]
        if p.shape != y.shape or not gradients or not all(torch.isfinite(v).all() for v in gradients):
            raise RuntimeError('Invalid batch/backward')
        model.eval()
        with torch.no_grad():
            one = model(x[:1], None, None, None)
            torch.testing.assert_close(one, model(x[:1], None, torch.randn_like(y[:1]), None), atol=0, rtol=0)
        checks.append(dict(model=kind, shape=list(p.shape), parameters=sum(v.numel() for v in model.parameters()),
             peak_allocated_mb=torch.cuda.max_memory_allocated(device)/1024**2,
             batch_seconds=time.monotonic()-tic, future_decoder_ignored=True, finite_backward=True))
        del model, physical, loss, p; torch.cuda.empty_cache()
    dump(HERE/dataset/'preflight.json', dict(dataset=dataset, checks=checks, selection=data['selection']))
    print(json.dumps(dict(preflight=dataset, checks=checks)), flush=True)


def train(job, device):
    out = outdir(job)
    with exclusive(out/'worker.lock'):
        if (out/'training.json').exists(): return read(out/'training.json')
        data = load_data(job['dataset']); cfg = data['config']; policy = data['policy']
        opts = options(job['dataset'], job['model']); seed = job['seed']; sources = source_hashes()
        specification = dict(job, config=cfg, options=opts, policy=policy,
            selection=data['selection'], sources=sources, optimizer='Adam',
            selection_metric='validation normalized target MSE only',
            schedule='6 constant epochs, cosine to .1lr; dataset-matched early stopping')
        immutable(out/'config.json', specification)
        model = make_model(job['model'], cfg, data, seed, opts).to(device)
        physical = PartialPhysicsLoss(data).to(device) if job['model'] == 'PG-Reg-CTF' else None
        optimizer = torch.optim.Adam(model.parameters(), lr=policy['lr'])
        best = float('inf'); stale = 0; history = []; start = 1; elapsed = 0.
        if (out/'resume.pth').exists():
            ck = torch.load(out/'resume.pth', map_location=device, weights_only=False)
            model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
            best, stale, history, start, elapsed = ck['best'], ck['stale'], ck['history'], ck['epoch']+1, ck['seconds']
        tic = time.monotonic()
        for epoch in range(start, policy['epochs']+1):
            if epoch > policy['min_epochs'] and stale >= policy['patience']: break
            supplement.previous.engine  # Engine sources remain unchanged; seed explicitly below.
            from physics import seed_all
            seed_all(seed*1000+epoch)
            order = np.random.default_rng(seed*1000+epoch).permutation(len(data['datasets']['train'])).tolist()
            loader = DataLoader(data['datasets']['train'], batch_size=policy['batch'], sampler=order, num_workers=0)
            rate = policy['lr']*(1. if epoch <= 6 else .1+.9*.5*(1+math.cos(math.pi*(epoch-6)/(policy['epochs']-6))))
            for group in optimizer.param_groups: group['lr'] = rate
            model.train(); se = reg_sum = count = 0.; et = time.monotonic()
            for step, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device); optimizer.zero_grad(set_to_none=True)
                p = model(x, None, None, None); mse = forecast_loss(p, y, data['targets'])
                reg = physical(x, p) if physical is not None else mse.detach()*0
                objective = mse + opts.get('physics_weight', 0.)*reg
                if not torch.isfinite(objective): raise RuntimeError('Nonfinite training loss')
                objective.backward(); optimizer.step()
                n = len(x); se += float(mse.detach())*n; reg_sum += float(reg.detach())*n; count += n
                if step % 20 == 0:
                    dump(out/'status.json', dict(state='training', **job, epoch=epoch, max_epochs=policy['epochs'],
                        step=step, steps=len(loader), train_mse=se/count, physics_loss=reg_sum/count,
                        best_val_mse=best if math.isfinite(best) else None, device=device,
                        pid=os.getpid(), updated=time.time()))
            val, _ = evaluate(model, data['datasets']['val'], data, device)
            row = dict(epoch=epoch, train_mse=se/count, physics_loss=reg_sum/count,
                       val_mse=val['mse'], val_mae=val['mae'], lr=rate, seconds=time.monotonic()-et)
            history.append(row)
            if val['mse'] < best:
                best = val['mse']; stale = 0
                checkpoint(out/'best.pth', dict(job, model_kind=job['model'], model=model.state_dict(),
                                               config=cfg, options=opts, epoch=epoch))
            else: stale += 1
            checkpoint(out/'resume.pth', dict(model=model.state_dict(), optimizer=optimizer.state_dict(), best=best,
                stale=stale, history=history, epoch=epoch, seconds=elapsed+time.monotonic()-tic))
            dump(out/'history.json', history); print(json.dumps(dict(**job, **row)), flush=True)
        ck = torch.load(out/'best.pth', map_location=device, weights_only=False); model.load_state_dict(ck['model'])
        val, _ = evaluate(model, data['datasets']['val'], data, device)
        result = dict(**job, best_val_mse=val['mse'], best_val_mae=val['mae'], best_epoch=ck['epoch'],
            epochs_completed=len(history), parameter_count=sum(v.numel() for v in model.parameters()),
            seconds=elapsed+time.monotonic()-tic, test_evaluated=False, checkpoint_sha256=sha(out/'best.pth'))
        dump(out/'training.json', result); dump(out/'status.json', dict(state='validation_complete', **result))
        del model, physical, optimizer; torch.cuda.empty_cache()
        return result


def final_test(job, device):
    out = outdir(job); lock = read(HERE/'selection_lock.json')
    if lock['checkpoints'].get(str(out)) != sha(out/'best.pth'): raise RuntimeError('Unfrozen checkpoint')
    if (out/'metrics.json').exists(): return read(out/'metrics.json')
    data = load_data(job['dataset']); ck = torch.load(out/'best.pth', map_location='cpu', weights_only=False)
    model = make_model(job['model'], ck['config'], data, job['seed'], ck['options']).to(device)
    model.load_state_dict(ck['model']); ds = test_dataset(job['dataset'], data)
    score, arrays = evaluate(model, ds, data, device, True)
    score.update(job, checkpoint_sha256=sha(out/'best.pth'))
    np.savez_compressed(out/'predictions.npz', prediction=arrays[0], truth=arrays[1], origins=ds.origins,
        channels=np.array(data['graph']['channels']), raw_mean=np.array(data['raw_mean']),
        raw_scale=np.array(data['raw_scale']), target_indices=np.array(data['targets']))
    dump(out/'metrics.json', score); dump(out/'status.json', dict(state='complete', **job, updated=time.time()))
    del model, ds; torch.cuda.empty_cache(); return score


def summarize():
    prior = read(ROOT/'runs/paper_minimal_v1/summary.json')['results']
    industrial = read(ROOT/'runs/physics_ai_comparison_v1/summary.json')
    datasets = ('industrial', 'lbnl', 'tep')
    order = ('PhysicsRelationCTF', 'Original CTF', 'EquationInput-CTF', *MODELS)
    results = {}; rows = []
    for dataset in datasets:
        results[dataset] = {}
        for kind in order:
            if kind in ('PhysicsRelationCTF', 'Original CTF', 'EquationInput-CTF'):
                old = {'PhysicsRelationCTF': 'Physics', 'Original CTF': 'CTF', 'EquationInput-CTF': 'equation_input'}[kind]
                source_rows = prior[dataset][old]['rows']
                chosen = [dict(dataset=dataset, model=kind, seed=r['seed'], mse=r['mse'], mae=r['mae'],
                               path=r['path'], reused=True) for r in source_rows]
            elif dataset == 'industrial':
                chosen = [dict(r, dataset=dataset, reused=True) for r in industrial['results'][kind]]
            else:
                chosen = []
                for seed in SEEDS:
                    path = outdir(dict(dataset=dataset, model=kind, seed=seed))
                    if not (path/'metrics.json').exists(): continue
                    metric = read(path/'metrics.json')
                    chosen.append(dict(dataset=dataset, model=kind, seed=seed, mse=metric['mse'], mae=metric['mae'],
                                       path=str(path), reused=False))
            if not chosen: continue
            stats = {f'{metric}_{label}': fn([r[metric] for r in chosen])
                     for metric in ('mse', 'mae')
                     for label, fn in [('mean', statistics.mean), ('std', statistics.stdev)]} if len(chosen) > 1 else {
                         f'{m}_{label}': (chosen[0][m] if label == 'mean' else None)
                         for m in ('mse', 'mae') for label in ('mean', 'std')}
            results[dataset][kind] = dict(n=len(chosen), **stats, rows=chosen)
            rows.extend({k: r[k] for k in ('dataset', 'model', 'seed', 'mse', 'mae', 'path', 'reused')} for r in chosen)
    complete = all(len(results[d].get(k, {}).get('rows', [])) == 3 for d in datasets for k in order)
    report = dict(status='complete' if complete else 'running', results=results,
                  note='Three dataset-specific tasks; normalized error scales differ. Industrial comparison had validation screening; public models not screened.')
    dump(HERE/'summary.json', report)
    import csv
    with (HERE/'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['dataset', 'model', 'seed', 'mse', 'mae', 'path', 'reused'])
        writer.writeheader(); writer.writerows(rows)
    lines = ['# 三数据集完整方法对比', '', '100%训练数据；固定种子42/43/44。数值为MSE / MAE，越小越好。', '',
             '| 方法 | 电解槽 | LBNL冷水机组 | TEP |', '|---|---:|---:|---:|']
    for kind in order:
        values = []
        for d in datasets:
            s = results[d].get(kind)
            values.append('待完成' if s is None else f"{s['mse_mean']:.8f} / {s['mae_mean']:.8f} (n={s['n']})")
        lines.append('| '+kind+' | '+' | '.join(values)+' |')
    lines += ['', '## 三种子标准差', '', '| 数据集 | 方法 | MSE mean ± std | MAE mean ± std |', '|---|---|---:|---:|']
    for d in datasets:
        for kind in order:
            s = results[d].get(kind)
            if s and s['n'] == 3:
                lines.append(f"| {d} | {kind} | {s['mse_mean']:.8f} ± {s['mse_std']:.8f} | {s['mae_mean']:.8f} ± {s['mae_std']:.8f} |")
    lines += ['', '## 解释边界', '',
              '- 所有公开数据对照结果均保留，不按是否优于完整模型筛选。',
              '- 原生CTF/完整模型/等方程输入对照复用既有结果；电解槽四个新增对照也复用。',
              '- PhyGNN为公开G3CN及物理计算/适配/补偿机制的预测适配版；不是原论文任务的完整复现。',
              '- PG-Reg-CTF为局部方程软损失对照，不是完整动力学PINN。',
              '- TEP输入20点，MTGNN膨胀率1/RF19；所有模型仅使用真实观测，不补造历史。',
              '- 既有测试集已在此前研究中暴露，本次配置选择不使用测试，但不是新盲测。',
              '- 不同数据集归一化MSE不可直接平均；三种子不代表三个独立装置。']
    (HERE/'COMPLETE_RESULTS.md').write_text('\n'.join(lines)+'\n')
    return report


def run(device):
    deterministic()
    if not torch.cuda.is_available(): raise RuntimeError('GPU required; no CPU fallback')
    with exclusive(HERE/'orchestrator.lock'):
        protected = protect_existing()
        data = {d: load_data(d) for d in DATASETS}
        immutable(HERE/'protocol.json', dict(jobs=jobs(), seeds=SEEDS,
            policies={d: x['policy'] for d, x in data.items()}, selections={d: x['selection'] for d, x in data.items()},
            options={d: {k: options(d, k) for k in MODELS} for d in DATASETS},
            sources=source_hashes(), policy='all 24 jobs retained, no outcome screening, validation checkpoint selection only'))
        for d in DATASETS:
            if not (HERE/d/'preflight.json').exists(): preflight(d, device)
        for i, job in enumerate(jobs()):
            completed = sum((outdir(j)/'training.json').exists() for j in jobs())
            dump(HERE/'status.json', dict(state='training', completed_trainings=completed, total_trainings=24,
                current=job, device=device, pid=os.getpid(), updated=time.time()))
            train(job, device)
        immutable(HERE/'selection_lock.json', dict(checkpoints={str(outdir(j)): sha(outdir(j)/'best.pth') for j in jobs()},
            sources=source_hashes(), policy='All 24 validation-selected checkpoints frozen before new tests'))
        for i, job in enumerate(jobs()):
            dump(HERE/'status.json', dict(state='locked_final_test', tested_jobs=i, total_jobs=24, current=job, updated=time.time()))
            final_test(job, device)
        protect_existing(); report = summarize()
        dump(HERE/'status.json', dict(state='complete', completed_trainings=24, completed_tests=24,
            protected_artifacts_unchanged=True, protected_count=len(protected),
            summary=str(HERE/'summary.json'), report=str(HERE/'COMPLETE_RESULTS.md'), updated=time.time()))
        print(json.dumps(dict(complete=True, report=str(HERE/'COMPLETE_RESULTS.md'))), flush=True)


def launch(device):
    if (HERE/'launch.json').exists():
        old = read(HERE/'launch.json')
        try: os.kill(old['pid'], 0)
        except ProcessLookupError: pass
        else: raise RuntimeError('Detached worker already alive; refusing duplicate launch')
    with (HERE/'orchestrator.log').open('ab', buffering=0) as log:
        worker = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()), 'run', '--device', device],
            cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
    dump(HERE/'launch.json', dict(pid=worker.pid, device=device, detached_session=True,
                                launched=time.time(), log=str(HERE/'orchestrator.log')))
    print(json.dumps(read(HERE/'launch.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preflight', 'run', 'launch', 'summary'])
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--dataset', choices=DATASETS)
    args = parser.parse_args()
    try:
        if args.action == 'launch': launch(args.device)
        elif args.action == 'summary': summarize()
        elif args.action == 'preflight':
            deterministic()
            for d in ([args.dataset] if args.dataset else DATASETS): preflight(d, args.device)
        else: run(args.device)
    except Exception as error:
        dump(HERE/'status.json', dict(state='failed', error=str(error), traceback=traceback.format_exc(), updated=time.time()))
        raise
