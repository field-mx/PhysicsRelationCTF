import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from model_v5 import physical_weights, PublicPhysicsCTF
from public_model_v2 import OriginalCTF, config, copy_ctf
from public_physics_v4 import build_graph
from test_physical import CHANNELS


class Checks(unittest.TestCase):
    def test_active_incoming_mass_is_one_and_inactive_zero(self):
        a = torch.tensor([[[.01, .02, .03, .04]]])
        select = torch.nn.functional.one_hot(torch.tensor([0,0,1,1]), 2).float()
        out = physical_weights(a, torch.tensor([[True,True,False,False]]), select)
        torch.testing.assert_close(out, torch.tensor([[[1/3,2/3,0.,0.]]]))
        torch.testing.assert_close(physical_weights(a*.1, torch.tensor([[True,True,False,False]]), select), out)

    def test_disabled_equals_original_and_enabled_has_gradients(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        channels = CHANNELS + ['CWL_SEC_DP','CWL_SEC_CW_FLOW','CWL_SEC_PM_SPD_1','CWL_SEC_PM_SPD_2','CWL_PRI_SW_TEMPSPT']
        graph = build_graph(channels);graph['pressure_reference_pa'] = 240000.
        sc = dict(mean=[0.]*len(channels), scale=[1.]*len(channels), gains=[1.]*len(graph['edges']), kappa=.285)
        cfg = config(len(channels));cfg.update(d_model=32,d_ff=64,n_heads=4,dropout=0.)
        base = OriginalCTF(SimpleNamespace(**cfg)).eval()
        net = PublicPhysicsCTF(SimpleNamespace(**cfg),graph,sc).eval();copy_ctf(net,base)
        x=torch.randn(2,96,len(channels));x[...,:2]=.01;x[...,2:6]+=25.;x[...,6]=1000.
        x[...,8]=239990.;x[...,9]=.1;x[...,10:12]=.5;x[...,12]=20.
        net.set_strengths(0,0,0)
        torch.testing.assert_close(base(x,None,None,None),net(x,None,None,None),atol=1e-6,rtol=1e-6)
        net.set_strengths(None)
        for layer in net.encoder.layers: layer.self_attention.inner_attention.record=True
        y=net(x,None,None,None);self.assertTrue(torch.isfinite(y).all());y.square().mean().backward()
        for layer in net.encoder.layers:
            a=layer.self_attention.inner_attention
            self.assertIsNotNone(a.theta_gamma.grad)
            self.assertTrue(torch.isfinite(a.theta_gamma.grad))
            self.assertLessEqual(a.diagnostics['physics_output_ratio'], .50001)

if __name__ == '__main__':unittest.main()
