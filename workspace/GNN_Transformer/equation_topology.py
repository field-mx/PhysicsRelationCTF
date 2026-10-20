"""Equation-derived candidate graph for the measured alkaline-electrolyzer boundary.

The graph encodes admissible predictive message paths supported by published
balance equations.  It is not a causal graph and does not invent unobserved
valves, controller commands, geometry, composition, or plant routing.
"""

RELATIONS = [
    'electrochemical_polarization',
    'faradaic_generation',
    'gas_inventory_pressure',
    'liquid_inventory',
    'thermal_balance',
    'impurity_accumulation',
]

SOURCES = {
    'thermal': 'https://arxiv.org/html/2202.13422',
    'pressure': 'https://arxiv.org/html/2505.16935',
    'dae': 'https://arxiv.org/html/2311.09882',
}

FULL_GRAPH_CHANNELS = frozenset({
    'AT010301', 'AT010302', 'FT010301', 'FT010302', 'FT010303', 'FT010304',
    'FT010306', 'FT010401', 'LGT010301', 'LGT010302', 'PDT010305',
    'PDT010306', 'PDT305306_Average_Level', 'PIC010306_PV', 'PT010301',
    'PT010302', 'PT010303', 'PT010304', 'PT010307', 'PT010308', 'PT010402',
    'PT010403', 'PT010404', 'PT010405', 'PT010501', 'SCRDLFK', 'SCRDYFK',
    'TT010001', 'TT010002', 'TT010301', 'TT010302', 'TT010303', 'TT010401',
    'TT010402', 'TT010403', 'TT010404', 'avg', 'max',
})


def build_equation_graph(channels, constants=()):
    """Return a sparse directed R-GCN graph restricted to observed channels."""
    channels = list(channels)
    index = {name: position for position, name in enumerate(channels)}
    constants = set(constants)
    edges = {}

    def link(source_names, target_names, relation, confidence, mechanism,
             equation, evidence, physical_boundary, limitation):
        for source_name in source_names.split():
            for target_name in target_names.split():
                if (source_name == target_name or source_name not in index or
                        target_name not in index or source_name in constants or
                        target_name in constants):
                    continue
                pair = (source_name, target_name)
                if pair in edges:
                    previous = edges[pair]
                    raise ValueError(
                        f'Duplicate physical edge {pair}: '
                        f'{previous["relation_name"]} and {relation}')
                edges[pair] = {
                    'source': index[source_name],
                    'target': index[target_name],
                    'source_name': source_name,
                    'target_name': target_name,
                    'relation': RELATIONS.index(relation),
                    'relation_name': relation,
                    'prior_level': 'P1' if confidence == .8 else 'P2',
                    'confidence': confidence,
                    'mechanism': mechanism,
                    'equation': equation,
                    'evidence': SOURCES[evidence],
                    'physical_boundary': physical_boundary,
                    'limitation': limitation,
                }

    p2 = .5
    p1 = .8

    # Ucell = f(I, T, P, electrolyte state). Separator pressure is deliberately
    # not treated as calibrated electrode partial pressure.
    electro_limitation = ('KOH state and electrode partial pressures are unobserved; '
                          'the edges encode polarization dependence, not a calibrated equation')
    link('SCRDLFK', 'SCRDYFK avg max', 'electrochemical_polarization', p2,
         'Stack current conditions stack and cell voltage',
         'U_cell = E_rev + eta_a + eta_c + I*R_ohmic', 'dae',
         'rectifier and electrolyzer stack', electro_limitation)
    link('TT010401 TT010402 TT010403 TT010404', 'avg max',
         'electrochemical_polarization', p2,
         'Stack outlet temperature conditions cell polarization',
         'U_cell = f(I,T,P,electrolyte)', 'dae',
         'electrolyzer stack outlets and cell-voltage aggregation', electro_limitation)
    link('avg max', 'SCRDYFK', 'electrochemical_polarization', p2,
         'Cell-voltage statistics contribute to measured stack voltage',
         'U_stack = sum(U_cell)', 'dae',
         'cell-voltage measurements inside the stack electrical boundary',
         'Cell count and individual cell voltages are unavailable')

    # Faraday production. The flow-to-flow links express stoichiometric
    # consistency between measured product outlets, not direct causation.
    link('SCRDLFK', 'FT010303 FT010304', 'faradaic_generation', p2,
         'Current drives oxygen and hydrogen generation',
         'n_H2 = eta_F*N_c*I/(2F); n_O2 = eta_F*N_c*I/(4F)', 'pressure',
         'stack current to product-gas train',
         'Measured flows are downstream outlet flows and include storage/control dynamics')
    link('FT010303 FT010304', 'FT010303 FT010304', 'faradaic_generation', p2,
         'Hydrogen and oxygen outlet flows share Faradaic stoichiometry',
         'n_H2 = 2*n_O2 under ideal Faradaic production', 'pressure',
         'paired hydrogen and oxygen product-gas trains',
         'Outlet flows need not preserve the instantaneous ideal ratio because of inventory and control')

    gas_equation = 'dp_g/dt = R_g*T/V_g*(n_generation - n_out)'
    gas_limitation = ('Separator geometry, valve position, controller setpoint and true gas '
                      'generation flow are unobserved')
    link('SCRDLFK', 'PT010301 PT010302', 'gas_inventory_pressure', p2,
         'Current-driven gas generation enters separator inventory', gas_equation,
         'pressure', 'stack production to the two separator gas inventories', gas_limitation)
    link('FT010303 TT010302 LGT010301 PDT010305 PT010402 PT010403', 'PT010301',
         'gas_inventory_pressure', p2,
         'Oxygen pressure depends on outlet flow, temperature, gas volume and stack outlet pressure',
         gas_equation, 'pressure', 'oxygen stack outlet and oxygen separator', gas_limitation)
    link('FT010304 TT010303 LGT010302 PDT010306 PT010404 PT010405', 'PT010302',
         'gas_inventory_pressure', p2,
         'Hydrogen pressure depends on outlet flow, temperature, gas volume and stack outlet pressure',
         gas_equation, 'pressure', 'hydrogen stack outlet and hydrogen separator', gas_limitation)
    link('PT010301', 'PT010303', 'gas_inventory_pressure', p2,
         'Oxygen separator pressure propagates to cooler valve inlet',
         'pressure/flow continuity along connected gas path', 'dae',
         'oxygen separator to oxygen cooler inlet', gas_limitation)
    link('PT010303', 'PT010501', 'gas_inventory_pressure', p2,
         'Oxygen pressure propagates across the regulated cooler outlet path',
         'pressure/flow continuity along connected gas path', 'pressure',
         'oxygen cooler before-valve to after-valve measurement', gas_limitation)
    link('PT010501', 'PT010307', 'gas_inventory_pressure', p2,
         'Oxygen cooler outlet feeds the oxygen washer inlet',
         'pressure/flow continuity along connected gas path', 'dae',
         'oxygen cooler to oxygen washer', 'Exact intervening equipment and pipe losses are unavailable')
    link('PT010302', 'PT010304', 'gas_inventory_pressure', p2,
         'Hydrogen separator pressure propagates to cooler valve inlet',
         'pressure/flow continuity along connected gas path', 'dae',
         'hydrogen separator to hydrogen cooler inlet', gas_limitation)
    link('PT010304', 'PT010308', 'gas_inventory_pressure', p2,
         'Hydrogen cooler outlet feeds the hydrogen washer inlet',
         'pressure/flow continuity along connected gas path', 'dae',
         'hydrogen cooler to hydrogen washer', 'Exact intervening valve position and pipe losses are unavailable')
    link('PT010301', 'FT010303', 'gas_inventory_pressure', p2,
         'Oxygen pressure and controlled outlet flow form a closed loop',
         'outlet flow = controller/valve response to pressure state', 'pressure',
         'oxygen separator pressure-control boundary', gas_limitation)
    link('PT010302', 'FT010304', 'gas_inventory_pressure', p2,
         'Hydrogen pressure and controlled outlet flow form a closed loop',
         'outlet flow = controller/compressor response to pressure state', 'pressure',
         'hydrogen separator pressure-control boundary', gas_limitation)
    link('PT010402 PT010403', 'PT010402 PT010403', 'gas_inventory_pressure', p1,
         'Paired oxygen-side stack outlet pressure measurements',
         'same-boundary pressure consistency', 'dae',
         'oxygen stack outlet', 'Sensor positions are nearby but not precisely documented')
    link('PT010404 PT010405', 'PT010404 PT010405', 'gas_inventory_pressure', p1,
         'Paired hydrogen-side stack outlet pressure measurements',
         'same-boundary pressure consistency', 'dae',
         'hydrogen stack outlet', 'Sensor positions are nearby but not precisely documented')

    inventory_equation = 'A_separator*dL/dt = q_in + q_makeup - q_out'
    inventory_limitation = ('Outlet FT010305 is constant and valve/control signals are unavailable; '
                            'the balance is structurally incomplete')
    link('FT010302', 'LGT010301 PDT010305', 'liquid_inventory', p2,
         'Oxygen-side makeup changes oxygen separator liquid inventory', inventory_equation,
         'dae', 'oxygen-side makeup and separator', inventory_limitation)
    link('FT010301', 'LGT010302 PDT010306', 'liquid_inventory', p2,
         'Hydrogen-side makeup changes hydrogen separator liquid inventory', inventory_equation,
         'dae', 'hydrogen-side makeup and separator', inventory_limitation)
    link('FT010306', 'LGT010301 LGT010302 PIC010306_PV', 'liquid_inventory', p2,
         'Balance-electrolyte flow redistributes two separator inventories', inventory_equation,
         'dae', 'inter-separator balance-electrolyte path',
         'The exact direction and control action of the balance line are unavailable')
    link('LGT010301 LGT010302', 'PDT305306_Average_Level', 'liquid_inventory', p2,
         'Two separator levels determine their reported average',
         'L_average = (L_O2 + L_H2)/2', 'pressure',
         'two separator liquid inventories', 'Offsets between level technologies may exist')
    link('LGT010301 LGT010302 PDT010305 PDT010306', 'PIC010306_PV',
         'liquid_inventory', p2,
         'Two separator level states determine the reported level difference',
         'Delta_L = L_H2 - L_O2', 'pressure',
         'two separator liquid inventories', 'Tag name suggests pressure control but supplied description is level difference')
    link('LGT010301', 'PDT010305', 'liquid_inventory', p1,
         'Two sensors observe oxygen separator liquid inventory',
         'hydrostatic differential pressure is proportional to level', 'dae',
         'oxygen separator liquid inventory', 'Calibration offsets and density changes are unobserved')
    link('PDT010305', 'LGT010301 PDT305306_Average_Level', 'liquid_inventory', p1,
         'Oxygen differential level measures inventory and contributes to average level',
         'hydrostatic differential pressure is proportional to level', 'dae',
         'oxygen separator liquid inventory', 'Calibration offsets and density changes are unobserved')
    link('LGT010302', 'PDT010306', 'liquid_inventory', p1,
         'Two sensors observe hydrogen separator liquid inventory',
         'hydrostatic differential pressure is proportional to level', 'dae',
         'hydrogen separator liquid inventory', 'Calibration offsets and density changes are unobserved')
    link('PDT010306', 'LGT010302 PDT305306_Average_Level', 'liquid_inventory', p1,
         'Hydrogen differential level measures inventory and contributes to average level',
         'hydrostatic differential pressure is proportional to level', 'dae',
         'hydrogen separator liquid inventory', 'Calibration offsets and density changes are unobserved')

    thermal_equation = 'C_th*dT/dt = Q_generation - Q_loss - Q_cooling + Q_transport'
    thermal_limitation = ('Cooling-water flow, ambient temperature, thermal capacities and exact '
                          'transport delays are unobserved')
    link('SCRDLFK SCRDYFK TT010301 FT010401', 'TT010401 TT010403',
         'thermal_balance', p2,
         'Electrical loading, inlet electrolyte temperature and circulation drive stack outlet temperature',
         thermal_equation, 'thermal', 'stack and circulating-electrolyte boundary', thermal_limitation)
    link('TT010401', 'TT010302', 'thermal_balance', p2,
         'Oxygen-side stack outlet transports heat to oxygen separator', thermal_equation,
         'thermal', 'oxygen stack outlet to oxygen separator', thermal_limitation)
    link('TT010403', 'TT010303', 'thermal_balance', p2,
         'Hydrogen-side stack outlet transports heat to hydrogen separator', thermal_equation,
         'thermal', 'hydrogen stack outlet to hydrogen separator', thermal_limitation)
    link('TT010302 TT010303', 'TT010301', 'thermal_balance', p2,
         'Separator electrolyte returns through the common lye cooler', thermal_equation,
         'thermal', 'separator return to circulating-lye cooler outlet',
         'A common mixed return is inferred from the single cooler/flow tags, not verified by P&ID')
    link('TT010001', 'TT010301 TT010002', 'thermal_balance', p2,
         'Cooling-water inlet temperature conditions heat removal and return temperature', thermal_equation,
         'thermal', 'cooling-water supply and lye cooler', thermal_limitation)
    link('TT010302 TT010303', 'TT010002', 'thermal_balance', p2,
         'Separator heat load appears in cooling-water return temperature', thermal_equation,
         'thermal', 'separator/cooler process side to cooling-water return', thermal_limitation)
    link('TT010401 TT010402', 'TT010401 TT010402', 'thermal_balance', p1,
         'Paired oxygen-side stack outlet temperature measurements',
         'same-boundary temperature consistency', 'thermal',
         'oxygen stack outlet', 'Exact sensor spacing is unavailable')
    link('TT010403 TT010404', 'TT010403 TT010404', 'thermal_balance', p1,
         'Paired hydrogen-side stack outlet temperature measurements',
         'same-boundary temperature consistency', 'thermal',
         'hydrogen stack outlet', 'Exact sensor spacing is unavailable')

    impurity_equation = 'd(n_g*c_imp)/dt = n_impurity_in - c_imp*n_out'
    impurity_limitation = ('Membrane state, exact analyzer sampling path and crossover flux are '
                           'unobserved; pressure difference is represented by the two pressures')
    link('SCRDLFK FT010303 PT010301 PT010302 TT010302', 'AT010301',
         'impurity_accumulation', p2,
         'Hydrogen-in-oxygen concentration depends on production, crossover, dilution and discharge',
         impurity_equation, 'dae', 'oxygen separator gas inventory and cross-diaphragm boundary',
         impurity_limitation)
    link('SCRDLFK FT010304 PT010301 PT010302 TT010303', 'AT010302',
         'impurity_accumulation', p2,
         'Oxygen-in-hydrogen concentration depends on production, crossover, dilution and discharge',
         impurity_equation, 'dae', 'hydrogen separator gas inventory and cross-diaphragm boundary',
         impurity_limitation)

    edge_list = list(edges.values())
    if (FULL_GRAPH_CHANNELS.issubset(index)
            and not constants.intersection(FULL_GRAPH_CHANNELS)):
        p1_count = sum(edge['confidence'] == p1 for edge in edge_list)
        p2_count = sum(edge['confidence'] == p2 for edge in edge_list)
        if (len(edge_list), p1_count, p2_count) != (91, 14, 77):
            raise ValueError(
                f'Equation graph budget mismatch: edges={len(edge_list)}, '
                f'P1={p1_count}, P2={p2_count}')

    return {
        'channels': channels,
        'relations': RELATIONS,
        'edges': edge_list,
        'sources': SOURCES,
        'policy': ('Observable equation boundary only; no invented actuator, geometry, '
                   'composition, or routing variables'),
        'exclusions': [
            'FT010305 constant on training split',
            'PT010601 ambiguous duplicate description',
            'R50 has no physical description',
            'seal-water, recovery-tank, nitrogen, instrument-air and pure-water routing unverified',
        ],
        'constants': sorted(constants),
    }
