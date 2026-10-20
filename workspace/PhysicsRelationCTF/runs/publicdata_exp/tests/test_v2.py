import importlib.util
import tempfile
import unittest
import torch
import numpy as np
from test_physical import CHANNELS


class V2Tests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('public_model_v2'),'v2 adapter missing')
        import public_model_v2
        return public_model_v2

    def test_startup_constant_equation_does_not_create_trend(self):
        m=self.api();x=torch.tensor([[[0.,0.,42.5,42.5]]]);valid=torch.tensor([[[0,0,1,1]]],dtype=torch.bool)
        torch.testing.assert_close(m.valid_center(x,valid),torch.zeros_like(x),rtol=0,atol=0)

    def test_equation_lag_match_ignores_inverse_raw_source_sign(self):
        m=self.api();history=torch.tensor([[[[1.,2.,3.],[3.,2.,1.]]]])
        query=torch.tensor([[[1.,2.,3.]]]);v=torch.ones_like(history,dtype=torch.bool)
        score=m.feature_lag_score(query,history,v,torch.ones(1))
        self.assertGreater(float(score[0,0,0]),float(score[0,0,1]))

    def test_observation_calibration_recovers_known_scaling(self):
        self.api()
        from public_physics_v2 import calibrate_kappa,equation_bank
        from public_physics import build_graph
        g=build_graph(CHANNELS)
        raw=np.tile(np.array([.01,.01,20.,10.,30.,42.5,418000.,1.]),(200,1))
        k,diag=calibrate_kappa(raw,g)
        self.assertAlmostEqual(k,.25,places=5)
        g['kappa']=k;bank,_=equation_bank(torch.tensor(raw[:1]),g)
        ids={f['name']:i for i,f in enumerate(g['features'])}
        self.assertAlmostEqual(float(bank[0,ids['balance_power_1']]),418000.,places=4)
        self.assertAlmostEqual(float(bank[0,ids['balance_cond_temp_1']]),42.5,places=4)

    def test_kappa_is_positive_without_usable_rows(self):
        self.api()
        from public_physics_v2 import calibrate_kappa
        from public_physics import build_graph
        k,diag=calibrate_kappa(np.zeros((10,8)),build_graph(CHANNELS))
        self.assertEqual(k,1.)
        self.assertEqual(diag['usable_rows'],0)

if __name__=='__main__':unittest.main()
