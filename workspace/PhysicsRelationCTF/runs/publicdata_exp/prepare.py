"""Freeze the public dataset protocol before any model evaluation."""
import hashlib
import json
import numpy as np
import pandas as pd
from data import ROOT, dump, to_si, fit_scaler, make_origins, nested_blocks


def prepare_data():
    out = ROOT/'data'
    if (out/'metadata.json').exists() and (out/'values.npy').exists():
        return json.loads((out/'metadata.json').read_text())
    frame = pd.read_csv(out/'ChillerPlant.csv')
    times = pd.to_datetime(frame.pop('Datetime')).to_numpy(dtype='datetime64[s]')
    if not np.all(np.diff(times).astype('timedelta64[s]').astype(int) == 60):
        raise ValueError('Non-continuous source: explicit gap handling required')
    n = len(frame); train_end = int(.7*n); val_end = int(.85*n)
    units = {}
    for c in frame:
        unit = ('degF' if 'TEMP' in c else 'GPM' if 'FLOW' in c else
                'kW' if 'POW' in c else 'inH2O' if c in ('CWL_SEC_DP','CWL_SEC_DPSPT')
                else 'W' if c == 'CWL_SEC_LOAD' else '1')
        frame[c] = to_si(frame[c].to_numpy(),unit); units[c] = unit
    allvalues = frame.to_numpy(dtype=np.float32)
    if not np.isfinite(allvalues).all(): raise ValueError('Missing/nonfinite source values')
    mean,scale,active = fit_scaler(allvalues[:train_end])
    channels = [c for c,a in zip(frame.columns,active) if a]
    values = allvalues[:,active]
    targets = [i for i,c in enumerate(channels)
               if '_STA' not in c and '_CTRL' not in c and 'SPT' not in c
               and c not in ('CWL_SEC_LOAD','TWV_CTRL')]
    blocks = [(a,min(a+1440,train_end)) for a in range(0,train_end,1440)
              if min(a+1440,train_end)-a >= 288]
    loads = np.array([frame['CWL_SEC_LOAD'].iloc[a:b].mean() for a,b in blocks])
    boundaries = np.quantile(loads,[1/3,2/3])
    months = pd.DatetimeIndex(times).month.to_numpy()
    strata = [f'{months[a]}_{np.searchsorted(boundaries,q)}' for (a,b),q in zip(blocks,loads)]
    subsets = nested_blocks(blocks,strata,[1.,.5,.3],20260929)
    subsets['1.0'] = [[0,train_end]]
    selections = {}
    for ratio, intervals in subsets.items():
        origins = make_origins(intervals)
        np.save(out/f'train_{ratio}_origins.npy',origins)
        rows = np.concatenate([np.arange(a,b) for a,b in intervals])
        sm,ss,_ = fit_scaler(values[rows])
        selections[ratio] = {'intervals':intervals,'rows':len(rows),
                            'actual_fraction':len(rows)/train_end,'windows':len(origins),
                            'mean':sm.tolist(),'scale':ss.tolist(),
                            'origins_sha256':hashlib.sha256(origins.tobytes()).hexdigest()}
    for name,a,b in [('val',train_end,val_end),('test',val_end,n)]:
        np.save(out/f'{name}_origins.npy',make_origins([(a,b)]))
    np.save(out/'values.npy',values);np.save(out/'timestamps.npy',times)
    meta = {'dataset':'LBNL ChillerPlant normal','rows':n,'train_end':train_end,
            'val_end':val_end,'sampling_seconds':60,'seq_len':96,'pred_len':96,'stride':96,
            'time_start':str(times[0]),'time_end':str(times[-1]),'channels':channels,
            'source_units':units,'excluded_constant':[c for c,a in zip(frame.columns,active) if not a],
            'target_indices':targets,'target_channels':[channels[i] for i in targets],
            'reference_mean':mean[active].tolist(),'reference_scale':scale[active].tolist(),
            'selections':selections,'stratification':'month x train-only load tertile, nested day blocks',
            'target_policy':'varying physical sensors incl weather; excludes status, command, setpoint, calculated load',
            'metric':'macro MSE/MAE in fixed full-training sensor standard deviations'}
    dump(out/'metadata.json',meta)
    print(json.dumps({k:meta[k] for k in ('rows','train_end','val_end','excluded_constant','target_channels')}),flush=True)
    return meta


class Windows:
    def __init__(self, split, ratio=1.):
        self.meta = json.loads((ROOT/'data/metadata.json').read_text())
        self.values = np.load(ROOT/'data/values.npy',mmap_mode='r')
        selection = self.meta['selections'][str(float(ratio))]
        self.mean = np.asarray(selection['mean'],dtype=np.float32)
        self.scale = np.asarray(selection['scale'],dtype=np.float32)
        fname = f'train_{float(ratio)}_origins.npy' if split=='train' else f'{split}_origins.npy'
        self.origins = np.load(ROOT/'data'/fname)
    def __len__(self): return len(self.origins)
    def __getitem__(self,index):
        import torch
        t = self.origins[index]
        return (torch.from_numpy((self.values[t-96:t]-self.mean)/self.scale),
                torch.from_numpy((self.values[t:t+96]-self.mean)/self.scale))


if __name__ == '__main__': prepare_data()
