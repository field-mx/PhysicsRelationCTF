"""Retain the level of temperature-valued balance/mixing equations."""
import torch
from model_v5 import PublicPhysicsCTF as V5

class PublicPhysicsCTF(V5):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # These equations already output target temperature units, so converting
        # to target instance coordinates preserves useful physical offsets.
        # Power balances and empirical actuator maps keep centered variations:
        # train-only residual checks show their level calibration is unreliable.
        absolute_ops = {'mix', 'balance_cond_temp', 'balance_evap_temp'}
        direct = [e['feature'] is not None and
                  self.graph['features'][e['feature']]['op'] in absolute_ops
                  for e in self.graph['edges']]
        self.boundary_edge |= torch.tensor(direct, dtype=torch.bool,
                                           device=self.boundary_edge.device)
