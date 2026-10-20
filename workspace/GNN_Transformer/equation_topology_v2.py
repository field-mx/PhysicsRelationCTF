"""Mechanism-separated equation graph for alkaline-electrolyzer sensors.

V2 keeps the observable boundary of :mod:`equation_topology`, but prevents
physically opposite terms of one balance equation from sharing one R-GCN
relation.  Confidence is an explicit prior strength, not a learned causal
claim.
"""
from copy import deepcopy

from equation_topology import build_equation_graph


UNIFIED_RELATIONS = [
    'electrochemical_polarization',
    'faradaic_generation',
    'gas_inventory_coarse',
    'liquid_inventory_coarse',
    'thermal_balance_coarse',
    'impurity_accumulation',
    'gas_generation_to_inventory',
    'gas_outflow_to_inventory',
    'gas_thermal_volume_to_pressure',
    'pressure_transport',
    'control_feedback',
    'same_boundary_measurement',
    'liquid_inflow_to_inventory',
    'liquid_measurement',
    'thermal_generation',
    'thermal_transport',
    'thermal_cooling',
]

COARSE_RELATION = {
    'electrochemical_polarization': 'electrochemical_polarization',
    'faradaic_generation': 'faradaic_generation',
    'gas_inventory_pressure': 'gas_inventory_coarse',
    'liquid_inventory': 'liquid_inventory_coarse',
    'thermal_balance': 'thermal_balance_coarse',
    'impurity_accumulation': 'impurity_accumulation',
}


def _set_relation(edge, relation, confidence):
    edge['relation_name'] = relation
    edge['relation'] = UNIFIED_RELATIONS.index(relation)
    edge['confidence'] = float(confidence)
    edge['prior_level'] = 'P1' if confidence >= .8 else (
        'P2' if confidence >= .5 else 'P3')
    return edge


def adapt_v1_to_unified(channels, constants=()):
    """Represent V1 in V2's parameter space without changing its support."""
    graph = deepcopy(build_equation_graph(channels, constants))
    graph['relations'] = list(UNIFIED_RELATIONS)
    for edge in graph['edges']:
        _set_relation(edge, COARSE_RELATION[edge['relation_name']],
                      edge['confidence'])
    graph['version'] = 'equation_v1_unified_parameter_space'
    return graph


def _gas_role(edge):
    source = edge['source_name']
    target = edge['target_name']
    if edge['prior_level'] == 'P1':
        return 'same_boundary_measurement', .8
    if source == 'SCRDLFK' and target in {'PT010301', 'PT010302'}:
        return 'gas_generation_to_inventory', 1.0
    if ((source, target) in {
            ('FT010303', 'PT010301'), ('FT010304', 'PT010302')}):
        return 'gas_outflow_to_inventory', 1.0
    if target in {'PT010301', 'PT010302'} and source.startswith(
            ('TT', 'LGT', 'PDT')):
        return 'gas_thermal_volume_to_pressure', .6
    if ((source, target) in {
            ('PT010301', 'FT010303'), ('PT010302', 'FT010304')}):
        return 'control_feedback', .35
    if source.startswith('PT') and target.startswith('PT'):
        stack_to_separator = target in {'PT010301', 'PT010302'}
        return 'pressure_transport', .6 if stack_to_separator else .35
    raise ValueError(f'Unclassified gas edge: {source}->{target}')


def _liquid_role(edge):
    source = edge['source_name']
    if edge['prior_level'] == 'P1':
        return 'same_boundary_measurement', .8
    if source in {'FT010301', 'FT010302'}:
        return 'liquid_inflow_to_inventory', 1.0
    if source == 'FT010306':
        return 'liquid_inflow_to_inventory', .35
    return 'liquid_measurement', .8


def _thermal_role(edge):
    source = edge['source_name']
    target = edge['target_name']
    if edge['prior_level'] == 'P1':
        return 'same_boundary_measurement', .8
    if source in {'SCRDLFK', 'SCRDYFK'}:
        return 'thermal_generation', .8
    if source == 'TT010001':
        return 'thermal_cooling', .6
    inferred_mixing = (source in {'TT010302', 'TT010303'} and
                       target in {'TT010301', 'TT010002'})
    return 'thermal_transport', .35 if inferred_mixing else .6


def build_equation_graph_v2(channels, constants=(), sparse=False):
    """Build a full or confidence-pruned mechanism-separated graph."""
    graph = deepcopy(build_equation_graph(channels, constants))
    graph['relations'] = list(UNIFIED_RELATIONS)
    converted = []
    for edge in graph['edges']:
        old_relation = edge['relation_name']
        if old_relation == 'gas_inventory_pressure':
            relation, confidence = _gas_role(edge)
        elif old_relation == 'liquid_inventory':
            relation, confidence = _liquid_role(edge)
        elif old_relation == 'thermal_balance':
            relation, confidence = _thermal_role(edge)
        elif old_relation == 'faradaic_generation':
            relation = old_relation
            confidence = 1.0 if edge['source_name'] == 'SCRDLFK' else .8
        elif old_relation == 'electrochemical_polarization':
            relation, confidence = old_relation, .8
        elif old_relation == 'impurity_accumulation':
            relation, confidence = old_relation, .6
        else:
            raise ValueError(f'Unclassified relation: {old_relation}')
        _set_relation(edge, relation, confidence)
        if not sparse or confidence >= .5:
            converted.append(edge)
    graph['edges'] = converted
    graph['version'] = 'equation_v2_sparse' if sparse else 'equation_v2_full'
    graph['sparsification_policy'] = (
        'remove only P3 edges (confidence 0.35)' if sparse else 'none')
    graph['confidence_policy'] = {
        'direct_balance': 1.0,
        'same_boundary_or_strong_mechanism': .8,
        'supported_indirect_mechanism': .6,
        'inferred_control_routing_or_mixing': .35,
    }
    return graph
