"""Auditable direct physical candidate edges, never estimated from test correlations."""
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def industrial_graph(channels):
    sys.path.insert(0,str(ROOT.parent/'GNN_Transformer'))
    from equation_topology_v2 import build_equation_graph_v2
    graph=build_equation_graph_v2(channels,sparse=True)
    kept=[];removed=[]
    for edge in graph['edges']:
        # Paired sensors/common production do not establish a direct propagation edge.
        reason=None
        if edge['relation_name']=='same_boundary_measurement':
            reason='Common measured state; not direct sensor-to-sensor influence'
        if edge['relation_name']=='faradaic_generation' and edge['source_name'].startswith('FT'):
            reason='Common current causes gas stoichiometry; not gas-flow-to-gas-flow propagation'
        (removed if reason else kept).append(dict(edge,exclusion_reason=reason) if reason else edge)
    graph.update(edges=kept,excluded_directness=removed,
                 policy='V2 sparse direct balance dependencies; no common-cause paired sensors or inferred control; no powers of A')
    return graph

def tep_graph(channels):
    edges={};idx={c:i for i,c in enumerate(channels)}
    source='https://github.com/camaramm/tennessee-eastman-profBraatz/blob/master/teprob.f'
    def add(s,t,mechanism,equation,lines):
        if s in idx and t in idx and s!=t:
            edges[s,t]=dict(source=idx[s],target=idx[t],source_name=s,target_name=t,
                            mechanism=mechanism,equation=equation,evidence=source,
                            source_lines=lines,limitation='Reduced measured-variable dependency; latent composition/valve states omitted')
    def m(i):return f'xmeas_{i}'
    def v(i):return f'xmv_{i}'
    # Direct valve-to-corresponding-stream and utility dependencies in simulator.
    for a,b in [(1,2),(2,3),(3,1),(4,4),(5,5),(6,10),(7,14),(8,17),(9,19),(10,21),(11,22)]:
        add(v(a),m(b),'actuator to stream/utility','FTM=f(VPOS) or cooling-water energy balance','565-600,679-701,793-796')
    for s,t in [(16,6),(7,6),(13,10),(16,5),(13,5),(16,20),(13,20),(11,20)]:
        add(m(s),m(t),'pressure-driven flow / compressor work','flow=f(delta_p); CPDH=f(PTV,PTS,TCS)','575-600')
    for s,t in [(9,21),(21,9),(8,21),(8,9),(9,22),(22,11),(7,22),(13,22),
                (18,19),(19,18),(14,12),(14,15),(17,15),(14,18),(11,18)]:
        add(m(s),m(t),'local inventory or heat-transfer balance',
            'd inventory/dt=in-out; d energy/dt=enthalpy_in-enthalpy_out+Q','664-678,762-796')
    # Feed flow/composition enters reactor inventory and enthalpy directly.
    for s in [6,*range(23,29)]:
        for t in [7,8,9]:
            add(m(s),m(t),'reactor feed mass/component/energy input',
                'YP(I)=FCM(I,7)-FCM(I,8)+CRXR(I); YP(9)=Hin-Hout+RH+QUR','762-775')
    for s in [5,10,14]:
        for t in [11,12,13]:
            add(m(s),m(t),'separator discharge mass and enthalpy',
                'YP(I+9)=FCM(I,8)-FCM(I,9)-FCM(I,10)-FCM(I,11)','762-780')
    return dict(channels=channels,edges=list(edges.values()),policy='Direct simulator equation dependencies, no correlation edges or graph powers',sources=[source])

def adjacency(graph):
    a=np.zeros((len(graph['channels']),len(graph['channels'])),dtype=bool)
    for e in graph['edges']:a[e['target'],e['source']]=True
    if np.diag(a).any():raise ValueError('Self-edge in physical graph')
    return a
