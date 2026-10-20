import unittest
from types import SimpleNamespace
import torch
import test_model
from test_physical import CHANNELS


class V4IntegrationTests(test_model.ModelTests):
    def models(self):
        from public_model_v2 import OriginalCTF,config,copy_ctf
        from public_model_v4 import PublicPhysicsCTF
        from public_physics_v4 import build_graph
        channels=CHANNELS+['CWL_SEC_DP','CWL_SEC_CW_FLOW','CWL_SEC_PM_SPD_1','CWL_SEC_PM_SPD_2','CWL_PRI_SW_TEMPSPT']
        g=build_graph(channels);g['pressure_reference_pa']=240000.
        sc=dict(mean=[0.]*len(channels),scale=[1.]*len(channels),gains=[1.]*len(g['edges']),kappa=.285)
        cfg=config(len(channels));cfg.update(d_model=32,d_ff=64,n_heads=4,dropout=0.)
        base=OriginalCTF(SimpleNamespace(**cfg)).eval()
        net=PublicPhysicsCTF(SimpleNamespace(**cfg),g,sc).eval();copy_ctf(net,base)
        x=torch.randn(2,96,len(channels));x[...,:2]=.01;x[...,2:6]+=25.;x[...,6]=1000.
        x[...,8]=239990.;x[...,9]=.1;x[...,10:12]=.5;x[...,12]=20.
        return base,net,x

    def test_physical_reference_affects_prediction_without_future_inputs(self):
        _,net,x=self.models()
        with torch.no_grad():
            a=net(x,None,None,None)
            net.graph['pressure_reference_pa']=239980.
            b=net(x,None,None,None)
        self.assertTrue(torch.isfinite(a).all())
        self.assertGreater(float((a-b).abs().max()),1e-7)

if __name__=='__main__':unittest.main()
