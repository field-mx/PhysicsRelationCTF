"""Public experiment data utilities, independent of old industrial loaders."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))
    tmp.replace(path)


def to_si(value, unit):
    x = np.asarray(value,dtype=np.float64)
    if unit == 'degF': return (x-32.)*5./9.
    if unit == 'GPM': return x*0.0000630901964
    if unit == 'kW': return x*1000.
    if unit == 'inH2O': return x*249.08891
    if unit in ('W','1','degC','m3/s','Pa'): return x.copy()
    raise ValueError(f'Unknown unit: {unit}')


def fit_scaler(train):
    mean = np.mean(train,axis=0,dtype=np.float64)
    std = np.std(train,axis=0,dtype=np.float64)
    active = std > np.maximum(1e-9,1e-7*np.abs(mean))
    return mean,np.where(active,std,1.),active


def make_origins(intervals, seq=96, pred=96, stride=96):
    origins = [np.arange(a+seq,b-pred+1,stride,dtype=np.int64)
               for a,b in intervals if b-a >= seq+pred]
    return np.concatenate(origins) if origins else np.empty(0,dtype=np.int64)


def nested_blocks(blocks, strata, ratios, seed):
    """Proportional nested rank, without deleting points inside a block."""
    if len(blocks) != len(strata): raise ValueError('strata length')
    rng = np.random.default_rng(seed)
    buckets = {}
    for i,s in enumerate(strata): buckets.setdefault(str(s),[]).append(i)
    ranked = []
    for key in sorted(buckets):
        ids = rng.permutation(buckets[key])
        # Equal fractional progress in each stratum gives proportional coverage.
        ranked.extend(((j+.5)/len(ids),float(rng.random()),int(i)) for j,i in enumerate(ids))
    order = [r[2] for r in sorted(ranked)]
    return {str(r):[list(blocks[i]) for i in sorted(order[:max(1,int(len(blocks)*r))])]
            for r in ratios}
