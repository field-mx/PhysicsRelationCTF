"""Preserve industrial split; TEP has disjoint runs and no boundary-crossing windows."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from graphs import industrial_graph,tep_graph

ROOT=Path(__file__).resolve().parent

def save_meta(path,meta):
    path.write_text(json.dumps(meta,indent=2,ensure_ascii=False))

def industrial():
    src=ROOT.parent/'GNN_Transformer/runs/sensor_96s_96s_stride96'
    out=ROOT/'data/industrial';out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((src/'data_summary.json').read_text())
    select=np.flatnonzero(meta['train_varying_mask'])
    channels=[meta['channels'][i] for i in select]
    x=np.load(src/'values.npy')[:,select].astype('float32')
    np.save(out/'values.npy',x)
    for split in ['train','val','test']:
        origins=np.load(src/f'{split}_origins.npy')
        np.save(out/f'{split}_origins.npy',np.column_stack([np.zeros(len(origins),dtype=int),origins]))
    save_meta(out/'graph.json',industrial_graph(channels))
    save_meta(out/'metadata.json',dict(dataset='industrial',channels=channels,
        sample_seconds=1,reference_scaling='existing full-training mean/std',
        source=meta['source'],source_sha256=meta['source_sha256'],train_end=meta['splits']['train']['end_row_exclusive'],
        exclusions=meta['constant_channels'],fewshot='nested chronological training prefix; normalization refit within available prefix',
        split=meta['splits']))

def tep():
    sys.path.insert(0,str(ROOT/'vendor'))
    import pyreadr
    raw=ROOT/'data/tep_raw';out=ROOT/'data/tep';out.mkdir(parents=True,exist_ok=True)
    a=next(iter(pyreadr.read_r(str(raw/'TEP_FaultFree_Training.RData')).values()))
    b=next(iter(pyreadr.read_r(str(raw/'TEP_FaultFree_Testing.RData')).values()))
    channels=[c for c in a if c.startswith(('xmeas_','xmv_'))]
    # Deterministic computational subset of the published normal-operation runs.
    ids=dict(train=list(range(1,81)),val=list(range(81,101)),test=list(range(401,501)))
    train=np.stack([a[a.simulationRun==i].sort_values('sample')[channels].to_numpy() for i in ids['train']])
    mean=train.mean((0,1));std=train.std((0,1));vary=std>1e-8
    channels=[c for c,k in zip(channels,vary) if k];mean=mean[vary];std=std[vary]
    for split in ids:
        df=b if split=='test' else a
        arr=np.stack([df[df.simulationRun==i].sort_values('sample')[channels].to_numpy() for i in ids[split]])
        if not np.isfinite(arr).all():raise ValueError('Nonfinite TEP values')
        arr=((arr-mean)/std).astype('float32');np.save(out/f'{split}_values.npy',arr)
        step=1 if split=='train' else 96
        origins=np.array([(i,t) for i in range(len(arr)) for t in range(96,arr.shape[1]-95,step)])
        np.save(out/f'{split}_origins.npy',origins)
    save_meta(out/'graph.json',tep_graph(channels))
    save_meta(out/'metadata.json',dict(dataset='tep',channels=channels,sample_seconds=180,
        ids=ids,source_doi='10.7910/DVN/6C3JR1',source='Rieth fault-free data via Kaggle mirror',
        reference_mean=mean.tolist(),reference_std=std.tolist(),
        subset='80 training runs / 20 validation runs / 100 official testing runs, predetermined by run ID',
        fewshot='nested whole training runs in fixed seeded permutation; no runs concatenated',
        exclusions=[c for c,k in zip([c for c in a if c.startswith(('xmeas_','xmv_'))],vary) if not k]))

if __name__=='__main__':
    industrial();tep()
    for name in ['industrial','tep']:
        d=ROOT/'data'/name;m=json.loads((d/'metadata.json').read_text());g=json.loads((d/'graph.json').read_text())
        print(name,'channels',len(m['channels']),'edges',len(g['edges']),
              {s:len(np.load(d/f'{s}_origins.npy')) for s in ['train','val','test']})
