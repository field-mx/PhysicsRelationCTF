"""Measured control references are boundary conditions, not centered trends."""
import json
import numpy as np
import torch
from data import ROOT
from public_physics_v3 import build_graph as previous_graph
from public_physics_v2 import equation_bank as previous_bank,fit_gains as previous_gains


def boundary_patch(history,valid,pi,target_mean,target_std):
    patch=(history-target_mean[:,:,None,None])/target_std[:,:,None,None]
    return (pi[...,None]*patch*valid).sum(2)


def build_graph(channels):
    g=previous_graph(channels);ix={c:i for i,c in enumerate(channels)}
    g['base_feature_count']=len(g['features'])
    reference=ROOT/'data/control_reference.json'
    g['pressure_reference_pa']=json.loads(reference.read_text())['pressure_reference_pa'] if reference.exists() else None
    for e in g['edges']:
        s,t=e['source_name'],e['target_name'];flow=None;ref=None;name=None
        if t=='CWL_SEC_DP' and s.startswith('CWL_SEC_PM_SPD_'):
            name='regulated_pressure';flow='CWL_SEC_CW_FLOW'
        elif s=='CWL_PRI_SW_TEMPSPT' and t.startswith('CHL_SW_TEMP_'):
            name='regulated_chiller_'+t.rsplit('_',1)[-1];flow='CHL_CW_FLOW_'+t.rsplit('_',1)[-1];ref=s
        elif s=='CT_SW_TEMPSPT' and t.startswith('CT_SW_TEMP_'):
            name='regulated_tower_'+t.rsplit('_',1)[-1];flow='CT_FLOW_'+t.rsplit('_',1)[-1];ref=s
        if name is None or flow not in ix:continue
        names=[f['name'] for f in g['features']]
        if name not in names:
            g['features'].append(dict(name=name,op='regulated_reference',flow_index=ix[flow],
                                     reference_index=None if ref is None else ix[ref]))
        e['feature']=[f['name'] for f in g['features']].index(name)
        e['known_gain']=1.;e['absolute_reference']=True;e['lags']=[0]
        e['equation']='feedback regulation toward observed historical reference while flow is active'
        e['limitation']='soft reference message only; no future setpoints, no direct output projection, no assumed settling time'
    g['version']='v4_known_control_boundary_reference'
    return g


def equation_bank(raw,graph):
    n=graph['base_feature_count'];base=dict(graph,features=graph['features'][:n])
    if n:bank,valid=previous_bank(raw,base)
    else:
        bank=raw.new_empty(*raw.shape[:-1],0);valid=torch.empty_like(bank,dtype=torch.bool)
    values=[];masks=[]
    for f in graph['features'][n:]:
        if f['reference_index'] is None:
            if graph['pressure_reference_pa'] is None:raise ValueError('Missing train-verified pressure reference')
            value=torch.full_like(raw[...,0],graph['pressure_reference_pa'])
        else:value=raw[...,f['reference_index']]
        ok=raw[...,f['flow_index']]>1e-5
        values.append(torch.where(ok,value,torch.zeros_like(value)));masks.append(ok)
    if values:
        bank=torch.cat([bank,torch.stack(values,-1)],-1);valid=torch.cat([valid,torch.stack(masks,-1)],-1)
    return bank,valid


def fit_gains(raw,graph):
    base=previous_graph(graph['channels'])
    base['retained_segment_lengths']=graph.get('retained_segment_lengths',[])
    gains,diag=previous_gains(raw,base)
    old={(e['source'],e['target']):x for e,x in zip(base['edges'],gains)}
    gains=[1. if e.get('absolute_reference') else old[(e['source'],e['target'])] for e in graph['edges']]
    graph['kappa']=diag['kappa']
    diag['boundary_reference']='Measured historical control reference preserved in target local-normalization coordinates; no output overwrite'
    diag['pressure_reference_pa']=graph['pressure_reference_pa']
    return gains,diag
