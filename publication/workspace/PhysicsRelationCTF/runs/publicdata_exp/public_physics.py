"""Equipment-boundary physical features, SI units, direct sparse relations."""
import numpy as np
import torch

RHO_CP = 4_180_000.  # water 1000 kg/m3 x 4180 J/(kg K), nominal constant
RELATIONS = ['energy_balance','heat_transport','hydraulic','actuator','fan_affinity','mixing']


def build_graph(channels):
    ix={c:i for i,c in enumerate(channels)};features=[];edges=[];seen=set()
    def feature(name,op,names,**kw):
        if not all(n in ix for n in names):return None
        idx=len(features)
        features.append(dict(name=name,op=op,inputs=[ix[n] for n in names],input_names=names,**kw))
        return idx
    def edge(source,target,relation,fi=None,gain=None,boundary='',equation='',lags=None):
        if source not in ix or target not in ix or source==target:return
        pair=(ix[source],ix[target])
        if pair in seen:return
        seen.add(pair)
        edges.append(dict(source=pair[0],target=pair[1],source_name=source,target_name=target,
            relation=RELATIONS.index(relation),relation_type=relation,feature=fi,known_gain=gain,
            boundary=boundary,equation=equation,lags=lags or [0,2,4,8,16,24],
            evidence='LBNL Chiller Plant inventory: equipment schematic and Table2',
            limitation='history-only relation guidance; no causal identification or exact transient conservation'))
    for k in range(1,4):
        n=lambda stem:f'{stem}_{k}'
        qe,qc,tei,teo,tci,tco,p=[n(s) for s in ['CHL_CW_FLOW','CHL_CD_FLOW','CHL_RW_TEMP',
                            'CHL_SW_TEMP','CHL_RWCD_TEMP','CHL_SWCD_TEMP','CHL_POW']]
        feature(f'evap_heat_{k}','heat',[qe,tei,teo])
        feature(f'cond_heat_{k}','heat',[qc,tco,tci])
        inputs=[qe,qc,tei,teo,tci,tco,p]
        for name,op,target,sources,equation in [
            ('power','balance_power',p,[qe,qc,tei,teo,tci,tco],'P = rho cp [qc(Tco-Tci)-qe(Tei-Teo)]'),
            ('cond_temp','balance_cond_temp',tco,[qe,qc,tei,teo,tci,p],'Tco = Tci + [rho cp qe(Tei-Teo)+P]/(rho cp qc)'),
            ('evap_temp','balance_evap_temp',teo,[qe,qc,tei,tci,tco,p],'Teo = Tei - [rho cp qc(Tco-Tci)-P]/(rho cp qe)')]:
            fi=feature(f'balance_{name}_{k}',op,inputs)
            if fi is not None:
                for s in sources:edge(s,target,'energy_balance',fi,1.,f'chiller{k}',equation,lags=[0,1,2,4,8])
        for s,t in [('CWL_PRI_RW_TEMP',tei),('CDWL_SW_TEMP',tci),(tco,n('CT_RW_TEMP'))]:
            edge(s,t,'heat_transport',gain=1.,boundary=f'chiller{k} direct water path',
                 equation='same-stream temperature transport with unmodeled losses',lags=[0,1,2,4,8])
        # Current actuator state is a historical input, not a known future command.
        for s,t in [(n('CHL_COMP_SPD_CTRL'),p),(n('CHL_STA'),qe),(n('CHL_STA'),qc),
                    (n('CT_FAN_SPD_CTRL'),n('CT_FAN_SPD'))]:
            edge(s,t,'actuator',boundary=f'unit{k}',equation='actuator command -> equipment response')
        speed=n('CT_FAN_SPD');fp=feature(f'fan_cube_{k}','cube',[speed])
        if fp is not None:edge(speed,n('CT_POW'),'fan_affinity',fp,boundary=f'cooling tower{k}',
                               equation='P proportional to speed^3 near similar operating conditions',lags=[0])
        for s in [n('CT_RW_TEMP'),n('CT_FLOW'),speed,'OA_TEMP_WB']:
            edge(s,n('CT_SW_TEMP'),'heat_transport',gain=1. if 'TEMP' in s else None,
                 boundary=f'cooling tower{k}',equation='heat rejection depends on inlet, water flow, fan, wet bulb')
    for k in (1,2):
        edge(f'CWL_SEC_PM_SPD_{k}','CWL_SEC_CW_FLOW','hydraulic',boundary='secondary water loop',
             equation='pump speed -> loop flow; unknown pump/system curve')
        edge(f'CWL_SEC_PM_SPD_{k}','CWL_SEC_DP','hydraulic',boundary='secondary water loop',
             equation='pump speed -> pressure rise; unknown pump/system curve')
    for s,t in [('CWL_SEC_RW_TEMP','CWL_PRI_RW_TEMP'),('CWL_PRI_SW_TEMP','CWL_SEC_SW_TEMP')]:
        edge(s,t,'heat_transport',gain=1.,boundary='primary-secondary loop with bypass',
             equation='connected flow path, imperfect due to bypass mixing')
    for kind,flow,temp,target in [('primary','CHL_CW_FLOW','CHL_SW_TEMP','CWL_PRI_SW_TEMP'),
                                  ('condenser','CHL_CD_FLOW','CHL_SWCD_TEMP','CDWL_RW_TEMP')]:
        names=[f'{flow}_{k}' for k in range(1,4)]+[f'{temp}_{k}' for k in range(1,4)]
        fi=feature(f'mixed_{kind}','mix',names)
        if fi is not None:
            for s in names:edge(s,target,'mixing',fi,1.,kind+' mixing header',
                                'Tmix=sum(q*T)/sum(q)',lags=[0,1,2,4])
    return dict(channels=channels,relations=RELATIONS,features=features,edges=edges,
                adjacency_convention='A[target,source]; query target receives key source',
                lag_units='minutes at native 1-minute sampling',rho_cp=RHO_CP,
                policy='fixed direct equipment relations; no graph powers; binary input controls retained')


def equation_bank(raw, graph):
    result=[];valid=[]
    for f in graph['features']:
        z=[raw[...,i] for i in f['inputs']];op=f['op']
        ok=torch.ones_like(z[0],dtype=torch.bool)
        if op=='heat':
            value=RHO_CP*z[0]*(z[1]-z[2]);ok=z[0]>1e-5
        elif op.startswith('balance_'):
            qe,qc,tei,teo,tci,tco,p=z
            ev=RHO_CP*qe*(tei-teo);cd=RHO_CP*qc*(tco-tci)
            ok=(qe>1e-5)&(qc>1e-5)
            if op=='balance_power':value=cd-ev
            elif op=='balance_cond_temp':value=tci+(ev+p)/(RHO_CP*qc.clamp_min(1e-5))
            else:value=tei-(cd-p)/(RHO_CP*qe.clamp_min(1e-5))
        elif op=='cube':value=z[0].clamp_min(0).pow(3)
        elif op=='mix':
            q=torch.stack(z[:3],-1).clamp_min(0);t=torch.stack(z[3:],-1)
            total=q.sum(-1);ok=total>1e-5
            value=(q*t).sum(-1)/total.clamp_min(1e-5)
        else:raise ValueError(op)
        result.append(torch.where(ok,value,torch.zeros_like(value)));valid.append(ok)
    return torch.stack(result,-1),torch.stack(valid,-1)


def fit_gains(raw, graph):
    with torch.no_grad():bank,valid=equation_bank(torch.as_tensor(raw,dtype=torch.float64),graph)
    bank=bank.numpy();valid=valid.numpy();gains=[];diagnostics=[]
    for e in graph['edges']:
        fi=e['feature'];mask=np.ones(len(raw),bool) if fi is None else valid[:,fi]
        x=raw[mask,e['source']] if fi is None else bank[mask,fi];y=raw[mask,e['target']]
        if e['known_gain'] is not None:gain=e['known_gain']
        elif len(x)>1:gain=np.mean((x-x.mean())*(y-y.mean()))/max(float(x.var()),1e-15)
        else:gain=0.
        gains.append(float(gain))
        diagnostics.append(dict(source=e['source_name'],target=e['target_name'],gain=float(gain),
                                valid_fraction=float(mask.mean()),known=e['known_gain'] is not None))
    checks=[]
    for k in (1,2,3):
        fname=f'balance_power_{k}';names=[f['name'] for f in graph['features']]
        if fname in names:
            fi=names.index(fname);m=valid[:,fi];p=raw[:,graph['channels'].index(f'CHL_POW_{k}')]
            err=bank[m,fi]-p[m]
            checks.append(dict(unit=k,active_rows=int(m.sum()),mae_W=float(np.mean(abs(err))),
                relative_mae=float(np.mean(abs(err))/max(np.mean(abs(p[m])),1.))))
    return gains,dict(edges=diagnostics,energy_checks_train_only=checks)
