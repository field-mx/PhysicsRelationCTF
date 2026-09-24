"""History-only raw-unit features and typed physical candidate relationships."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'PhysicsGuidedCTF'
sys.path.insert(0,str(OLD))
from experiment import load_data,config,seed_all,dump
from model import OriginalCTF

FARADAY=96485.3321233100184
RELATIONS=['electrical','electrochemical','gas_generation','gas_transport',
           'thermal','hydraulic','gas_inventory','impurity','state_consistency']
LAGS={
    'electrical':[0], 'electrochemical':[0,2,4,8],
    'gas_generation':[8,16,24,32,48,60], 'gas_transport':[0,4,8,16,24,32,48],
    'thermal':[24,32,48,60,72], 'hydraulic':[0,4,8,16,24,32,48],
    'gas_inventory':[4,8,16,24,32,48,60], 'impurity':[24,32,48,60,72],
    'state_consistency':[0]}
FEATURES=['power_W','H2_per_cell_mol_s','O2_per_cell_mol_s','cooling_delta_K',
          'O2_lye_delta_K','H2_lye_delta_K','O2_heat_proxy','H2_heat_proxy',
          'O2_valve_dp_MPa','separator_dp_MPa']

def equation_features(raw,channels):
    """raw[...,C] in Excel units. Tensor-only operations preserve gradients."""
    ix={c:i for i,c in enumerate(channels)}
    def x(c):return raw[...,ix[c]]
    current=x('SCRDLFK');power=current*x('SCRDYFK')
    cold=x('TT010301');o=x('TT010401')-cold;h=x('TT010403')-cold
    return torch.stack([power,current/(2*FARADAY),current/(4*FARADAY),
        x('TT010002')-x('TT010001'),o,h,x('FT010401')*o,x('FT010401')*h,
        x('PT010303')-x('PT010501'),x('PT010302')-x('PT010301')],-1)

def build_relations(channels):
    sys.path.insert(0,str(ROOT.parent/'GNN_Transformer'))
    from equation_topology_v2 import build_equation_graph_v2
    old=build_equation_graph_v2(channels,sparse=True)
    edges=[];excluded=[]
    map_type={'electrochemical_polarization':'electrochemical','faradaic_generation':'gas_generation',
      'gas_generation_to_inventory':'gas_inventory','gas_outflow_to_inventory':'gas_inventory',
      'gas_thermal_volume_to_pressure':'gas_inventory','pressure_transport':'gas_transport',
      'liquid_inflow_to_inventory':'hydraulic','liquid_measurement':'hydraulic',
      'thermal_generation':'thermal','thermal_transport':'thermal','thermal_cooling':'thermal',
      'impurity_accumulation':'impurity','same_boundary_measurement':'state_consistency'}
    for e in old['edges']:
        s,t=e['source_name'],e['target_name'];r=e['relation_name'];reason=None
        if r=='faradaic_generation' and s.startswith('FT'):reason='paired product flows share cause, not direct transport'
        if s=='max' and t=='SCRDYFK':reason='maximum cell voltage cannot be summed as stack voltage'
        if s=='avg' and t=='SCRDYFK':reason='unknown cell count/average coverage; do not equate with whole-stack sum'
        if reason:excluded.append(dict(e,reason=reason));continue
        typ=map_type[r]
        if s=='SCRDLFK' and t=='SCRDYFK':typ='electrical'
        if typ=='state_consistency' and t=='PDT305306_Average_Level':typ='hydraulic'
        feat=None
        if s=='SCRDLFK' and t=='FT010304':feat='H2_per_cell_mol_s'
        elif s=='SCRDLFK' and t=='FT010303':feat='O2_per_cell_mol_s'
        elif r=='thermal_generation':feat='power_W'
        elif typ=='thermal' and s=='FT010401':feat='O2_heat_proxy' if t=='TT010401' else 'H2_heat_proxy'
        elif typ=='thermal' and s=='TT010001':feat='cooling_delta_K'
        elif s=='PT010303' and t=='PT010501':feat='O2_valve_dp_MPa'
        elif typ=='impurity' and s.startswith('PT'):feat='separator_dp_MPa'
        elif r=='gas_generation_to_inventory':feat='O2_per_cell_mol_s' if t=='PT010301' else 'H2_per_cell_mol_s'
        edges.append(dict(source=e['source'],target=e['target'],source_name=s,target_name=t,
            relation_type=typ,relation=RELATIONS.index(typ),lags=LAGS[typ],feature=feat,
            mechanism=e['mechanism'],physical_boundary=e['physical_boundary'],
            equation=e['equation'],evidence=e['evidence'],limitation=e['limitation']))
    return dict(channels=channels,relations=RELATIONS,edges=edges,excluded=excluded,
       virtual_features=FEATURES,lag_units='seconds at 1Hz; last24-point patch limits lag to72',
       policy='state_consistency is bidirectional redundancy, not causal; no graph powers; fixed topology')

def get_data(ratio,sampling='prefix',subset_seed=20260922):
    d,_,meta,selection=load_data('industrial',ratio,sampling=sampling,subset_seed=subset_seed)
    # Recover physical units from existing verified reference cache, then the
    # available-prefix scaler is used identically by CTF and Physics.
    src=ROOT.parent/'GNN_Transformer/runs/sensor_96s_96s_stride96/data_summary.json'
    reference=json.loads(src.read_text());mask=np.array(reference['train_varying_mask'],bool)
    refmean=np.asarray(reference['mean'])[mask];refscale=np.asarray(reference['scale'])[mask]
    input_scale=d['train'].scale.astype('float64')*refscale
    input_mean=d['train'].mean.astype('float64')*refscale+refmean
    if sampling=='prefix':
        end=selection['train_end']
        raw=np.asarray(d['train'].values[0,:end],dtype='float64')*refscale+refmean
    else:
        covered=np.zeros(selection['train_end'],dtype=bool)
        for _,t in d['train'].origins:
            covered[max(0,int(t)-96):min(selection['train_end'],int(t)+96)]=True
        raw=np.asarray(d['train'].values[0,np.flatnonzero(covered)],dtype='float64')*refscale+refmean
    eq=equation_features(torch.tensor(raw),meta['channels']).numpy()
    eqmean=eq.mean(0);eqstd=np.maximum(eq.std(0),1e-12)
    graph=build_relations(meta['channels'])
    # Scalar data-identification of an unknown gain; not claimed as eta_F, Nc,
    # heat capacity or a physically calibrated output-flow equation.
    gains=[]
    for e in graph['edges']:
        v=raw[:,e['source']] if e['feature'] is None else eq[:,FEATURES.index(e['feature'])]
        y=raw[:,e['target']]
        cov=np.mean((v-v.mean())*(y-y.mean()));gain=cov/max(float(v.var()),1e-15)
        if e['relation_type']=='state_consistency':gain=1.
        gains.append(float(gain))
    scalers=dict(raw_mean=input_mean.tolist(),raw_scale=input_scale.tolist(),
                 eq_mean=eqmean.tolist(),eq_scale=eqstd.tolist(),gains=gains)
    return d,graph,meta,selection,scalers
