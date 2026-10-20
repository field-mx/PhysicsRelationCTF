"""Freeze validation-selected pairs before one held-out evaluation per checkpoint."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from data import ROOT,dump
import train_v2 as engine


def validate_pairs(rows,seeds):
    keys=[(r['kind'],float(r['ratio']),int(r['seed'])) for r in rows]
    expected={(k,r,s) for k in ['CTF','Physics'] for r in [1.,.5,.3] for s in seeds}
    if len(keys)!=len(set(keys)) or set(keys)!=expected:
        raise ValueError('Missing, duplicate or unregistered model/ratio/seed pairs')


def summarize(rows):
    result={}
    for ratio in [1.,.5,.3]:
        item={}
        for kind in ['CTF','Physics']:
            selected=[r for r in rows if r['kind']==kind and r['ratio']==ratio]
            item[kind]={'seeds':[r['seed'] for r in selected]}
            for metric in ['mse','mae']:
                values=np.asarray([r[metric] for r in selected])
                item[kind][metric+'_mean']=float(values.mean())
                item[kind][metric+'_std']=float(values.std(ddof=1)) if len(values)>1 else None
            full=[r['mse'] for r in rows if r['kind']==kind and r['ratio']==1.]
            item[kind]['degradation_mse_percent']=float((item[kind]['mse_mean']/np.mean(full)-1)*100)
        for metric in ['mse','mae']:
            item['gain_'+metric+'_percent']=(1-item['Physics'][metric+'_mean']/item['CTF'][metric+'_mean'])*100
        result[str(ratio)]=item
    return result


def freeze(selection):
    path=ROOT/'final/frozen_test_manifest.json'
    validate_pairs(selection['runs'],selection['seeds'])
    sources=engine.source_hashes()
    sources.update({str(p):engine.sha(p) for p in ROOT.glob('*.py')})
    data={str(p):engine.sha(p) for p in (ROOT/'data').glob('*') if p.suffix in ['.npy','.json']}
    checkpoints={}
    paired={}
    for row in selection['runs']:
        folder=Path(row['path']);cfg=json.loads((folder/'config.json').read_text())
        state=json.loads((folder/'training.json').read_text())
        if not path.exists() and any((folder/name).exists() for name in ['test_metrics.json','test_predictions.npz']):
            raise ValueError('Candidate already accessed test before first freeze')
        for key in ['kind','ratio','seed']:
            if cfg[key]!=row[key]:raise ValueError(f'Run mismatch: {key}')
        if state['test_evaluated']:raise ValueError('Development run already tested')
        if cfg['values_sha256']!=engine.sha(ROOT/'data/values.npy'):
            raise ValueError('Training values changed')
        if cfg['data_metadata_sha256']!=engine.sha(ROOT/'data/metadata.json'):
            raise ValueError('Training data metadata changed')
        for p,digest in cfg['sources'].items():
            if engine.sha(p)!=digest:raise ValueError(f'Training source changed: {p}')
        pair=(row['ratio'],row['seed']);matched={k:cfg[k] for k in ['epochs','lr','config','schedule']}
        if pair in paired and matched!=paired[pair]:raise ValueError('Unfair paired training protocol')
        paired[pair]=matched
        ck=engine.torch.load(folder/'best.pth',map_location='cpu',weights_only=False)
        for key in ['kind','ratio','seed','config','options']:
            if ck[key]!=cfg[key]:raise ValueError(f'Checkpoint identity mismatch: {key}')
        if ck['epoch']!=state['best_epoch']:raise ValueError('Checkpoint is not the declared validation-best epoch')
        for key,name in [('graph','graph.json'),('scalers','scalers.json')]:
            if ck[key]!=json.loads((folder/name).read_text()):raise ValueError(f'Checkpoint {key} mismatch')
        del ck
        checkpoints[str((folder/'best.pth').resolve())]=engine.sha(folder/'best.pth')
        for name in ['config.json','scalers.json','graph.json','training.json']:
            p=folder/name;data[str(p)]=engine.sha(p)
    manifest=dict(selection=selection,sources=sources,data=data,checkpoints=checkpoints)
    if path.exists():
        if json.loads(path.read_text())!=manifest:raise ValueError('Refusing to replace frozen protocol')
    else:dump(path,manifest)
    return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--selection',required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    selection=json.loads(Path(a.selection).read_text());lock=freeze(selection)
    engine.torch.set_num_threads(2)
    engine.torch.backends.cuda.matmul.allow_tf32=False
    engine.torch.backends.cudnn.allow_tf32=False
    engine.torch.backends.cudnn.benchmark=False
    engine.torch.backends.cudnn.deterministic=True
    engine.torch.use_deterministic_algorithms(True)
    rows=[]
    for i,row in enumerate(selection['runs']):
        graph=json.loads((Path(row['path'])/'graph.json').read_text())
        if graph.get('version')=='v4_known_control_boundary_reference':
            from public_model_v4 import PublicPhysicsCTF
        else:
            from public_model_v2 import PublicPhysicsCTF
        engine.PublicPhysicsCTF=PublicPhysicsCTF
        dump(ROOT/'final/status.json',dict(state='testing',completed=i,total=len(selection['runs']),current=row,updated=time.time()))
        result=engine.test(row['path'],a.device,lock)
        rows.append(dict(path=row['path'],**result))
    result=dict(protocol=selection,summary=summarize(rows),runs=rows)
    dump(ROOT/'final/summary.json',result)
    dump(ROOT/'final/status.json',dict(state='complete',completed=len(rows),updated=time.time()))
    print(json.dumps(result['summary'],indent=2),flush=True)


if __name__=='__main__':main()
