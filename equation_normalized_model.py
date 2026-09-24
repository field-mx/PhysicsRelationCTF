"""Formula-feature scale should not disappear with low empirical correlation.

Only formula edges change. Preserve the empirical response orientation and
thus lag-matching sign; remove its magnitude from the feature's unit scaling.
This is a dimensionless representation prior, NOT a physical state estimator.
"""
import copy
from relation_model import PhysicsRelationCTF
from physics import FEATURES

class EquationNormalizedPhysicsCTF(PhysicsRelationCTF):
    def __init__(self,config,graph,scalers,**options):
        normalized=copy.deepcopy(scalers)
        for k,e in enumerate(graph['edges']):
            if e['feature']:
                sign=-1. if scalers['gains'][k]<0 else 1.
                std=scalers['eq_scale'][FEATURES.index(e['feature'])]
                normalized['gains'][k]=sign*scalers['raw_scale'][e['target']]/max(std,1e-12)
        super().__init__(config,graph,normalized,**options)
