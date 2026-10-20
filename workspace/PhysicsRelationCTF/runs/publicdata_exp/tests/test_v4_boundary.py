import importlib.util
import unittest
import torch

class BoundaryReferenceTests(unittest.TestCase):
    def test_known_reference_level_is_not_centered_away(self):
        self.assertIsNotNone(importlib.util.find_spec('public_physics_v4'))
        from public_physics_v4 import boundary_patch
        history=torch.full((1,1,2,24),20.)
        pi=torch.tensor([[[.5,.5]]]);valid=torch.ones_like(history,dtype=torch.bool)
        result=boundary_patch(history,valid,pi,torch.tensor([[15.]]),torch.tensor([[2.]]))
        torch.testing.assert_close(result,torch.full((1,1,24),2.5))
        torch.testing.assert_close(boundary_patch(history,valid&False,pi,torch.tensor([[15.]]),torch.tensor([[2.]])),torch.zeros(1,1,24))

    def test_reference_disabled_when_circuit_has_no_flow(self):
        from public_physics_v4 import build_graph,equation_bank
        channels=['CWL_SEC_DP','CWL_SEC_CW_FLOW','CWL_SEC_PM_SPD_1','CWL_SEC_PM_SPD_2']
        g=build_graph(channels);g['pressure_reference_pa']=240000.
        x=torch.tensor([[[230000.,.1,.5,.5],[230000.,0.,0.,0.]]])
        f,v=equation_bank(x,g);j=[f['name'] for f in g['features']].index('regulated_pressure')
        self.assertEqual(float(f[0,0,j]),240000.);self.assertTrue(v[0,0,j])
        self.assertEqual(float(f[0,1,j]),0.);self.assertFalse(v[0,1,j])

if __name__=='__main__':unittest.main()
