"""Train-only observation conversion; NOT an identified chiller efficiency."""
import numpy as np
import torch
from public_physics import RHO_CP,build_graph,equation_bank as nominal_bank,fit_gains as nominal_gains


def calibrate_kappa(raw,graph):
    ix={c:i for i,c in enumerate(graph['channels'])};ratios=[]
    segment_starts=np.cumsum([0]+graph.get('retained_segment_lengths',[]))
    for k in (1,2,3):
        names=[f'{s}_{k}' for s in ['CHL_CW_FLOW','CHL_CD_FLOW','CHL_RW_TEMP','CHL_SW_TEMP',
                                   'CHL_RWCD_TEMP','CHL_SWCD_TEMP','CHL_POW']]
        if not all(n in ix for n in names):continue
        qe,qc,tei,teo,tci,tco,p=[raw[:,ix[n]] for n in names]
        de=tei-teo;dc=tco-tci
        steady=(qe>1e-5)&(qc>1e-5)&(p>10000.)
        steady &= (abs(np.diff(p,prepend=p[0]))<100.)&(abs(np.diff(de,prepend=de[0]))<.01)&(abs(np.diff(dc,prepend=dc[0]))<.01)
        for start in segment_starts:
            if start<len(raw):steady[start]=False
        r=RHO_CP*(qc[steady]*dc[steady]-qe[steady]*de[steady])/p[steady]
        ratios.extend(r[np.isfinite(r)&(r>0)].tolist())
    kappa=float(np.clip(np.median(ratios),.05,2.)) if ratios else 1.
    return kappa,dict(kappa=kappa,usable_rows=len(ratios),bounds=[.05,2.],
                      interpretation='effective observation/boundary scale, not physical efficiency',
                      equation='Qcond-Qevap approximately kappa * reported power',
                      selection='training-only flowing loaded locally steady rows; reject segment boundaries')


def equation_bank(raw,graph):
    bank,valid=nominal_bank(raw,graph);kappa=graph.get('kappa',1.)
    # Recompute only balance features; all other physical laws unchanged.
    cols=[]
    for i,f in enumerate(graph['features']):
        op=f['op'];value=bank[...,i]
        if op.startswith('balance_'):
            qe,qc,tei,teo,tci,tco,p=[raw[...,j] for j in f['inputs']]
            ev=RHO_CP*qe*(tei-teo);cd=RHO_CP*qc*(tco-tci)
            if op=='balance_power':value=(cd-ev)/kappa
            elif op=='balance_cond_temp':value=tci+(ev+kappa*p)/(RHO_CP*qc.clamp_min(1e-5))
            else:value=tei-(cd-kappa*p)/(RHO_CP*qe.clamp_min(1e-5))
            value=torch.where(valid[...,i],value,torch.zeros_like(value))
        cols.append(value)
    return torch.stack(cols,-1),valid


def fit_gains(raw,graph):
    gains,diag=nominal_gains(raw,graph)
    kappa,calibration=calibrate_kappa(raw,graph);graph['kappa']=kappa
    diag['observation_calibration']=calibration
    diag['kappa']=kappa
    return gains,diag
