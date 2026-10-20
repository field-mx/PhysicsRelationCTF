"""Official common-header boundary and documented feedback-control relations.

Reference: LBNL Chiller Plant inventory, Figure 2 and Table 1.
These are history-only guidance edges, not exact transient constraints.
"""
from public_physics_v2 import build_graph as previous_graph
from public_physics_v2 import fit_gains, equation_bank
from public_physics import RELATIONS


def build_graph(channels):
    graph=previous_graph(channels);ix={c:i for i,c in enumerate(channels)}
    graph['edges']=[e for e in graph['edges'] if not
        (e['source_name'].startswith('CHL_SWCD_TEMP_') and e['target_name'].startswith('CT_RW_TEMP_'))]
    seen={(e['source_name'],e['target_name']) for e in graph['edges']}
    for k in (1,2,3):
        for s,t,relation,mechanism in [
            ('CDWL_RW_TEMP',f'CT_RW_TEMP_{k}','heat_transport','common condenser return header feeds cooling towers'),
            ('CWL_PRI_SW_TEMPSPT',f'CHL_SW_TEMP_{k}','actuator','documented local chiller controller regulates leaving-water temperature'),
            ('CT_SW_TEMPSPT',f'CT_SW_TEMP_{k}','actuator','common cooling-tower setpoint regulates leaving-water temperature through fan control')]:
            if s not in ix or t not in ix or (s,t) in seen:continue
            seen.add((s,t))
            graph['edges'].append(dict(source=ix[s],target=ix[t],source_name=s,target_name=t,
                relation=RELATIONS.index(relation),relation_type=relation,feature=None,known_gain=1.,
                boundary='common header' if relation=='heat_transport' else f'controlled unit{k}',
                equation=mechanism,lags=[0,1,2,4,8] if relation=='heat_transport' else [0,2,4,8,16,24],
                evidence='LBNL Chiller Plant inventory Figure2/Table1; historical setpoints only',
                limitation='finite feedback dynamics and transport losses remain data-learned'))
    graph['version']='v3_common_header_and_documented_setpoint_relations'
    for edge in graph['edges']:
        if edge['feature'] is None:continue
        op=graph['features'][edge['feature']]['op']
        if op.startswith('balance_'):
            edge['nominal_equation']=edge['equation']
            edge['observation_mapping']='Qcond-Qevap = kappa * reported power; kappa fitted on retained steady training rows only'
            edge['limitation']+='; effective observation scale is not identified physical efficiency'
    return graph
