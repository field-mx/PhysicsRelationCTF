import importlib.util
import unittest
from types import SimpleNamespace
import torch
from test_physical import CHANNELS

class ModelTests(unittest.TestCase):
    def models(self):
        self.assertIsNotNone(importlib.util.find_spec('public_model'),'model adapter not implemented')
        import public_model as m
        from public_physics import build_graph
        g=build_graph(CHANNELS);sc={'mean':[0.]*len(CHANNELS),'scale':[1.]*len(CHANNELS),
                                 'gains':[1.]*len(g['edges'])}
        cfg=m.config(len(CHANNELS));cfg.update(d_model=32,d_ff=64,n_heads=4,dropout=0.)
        base=m.OriginalCTF(SimpleNamespace(**cfg)).eval()
        net=m.PublicPhysicsCTF(SimpleNamespace(**cfg),g,sc).eval();m.copy_ctf(net,base)
        x=torch.randn(2,96,len(CHANNELS));x[...,:2]=.01;x[...,2:6]+=25.;x[...,6]=1000.
        return base,net,x

    def test_zero_strength_exact_baseline(self):
        b,n,x=self.models();n.set_strengths(0.,0.,0.)
        with torch.no_grad():
            torch.testing.assert_close(n(x,None,None,None),b(x,None,None,None),rtol=0,atol=0)

    def test_physics_changes_outputs_and_backpropagates(self):
        b,n,x=self.models();y=n(x,None,None,None)
        self.assertTrue(torch.isfinite(y).all());self.assertEqual(tuple(y.shape),(2,96,x.shape[-1]))
        self.assertGreater(float((y-b(x,None,None,None)).abs().max().detach()),1e-7)
        y.square().mean().backward()
        for l in n.encoder.layers:
            grad=l.self_attention.inner_attention.theta_gamma.grad
            self.assertTrue(torch.isfinite(grad));self.assertGreater(float(grad.abs()),0.)

    def test_future_arguments_do_not_affect_result(self):
        _,n,x=self.models()
        with torch.no_grad():
            a=n(x,None,torch.zeros_like(x),None);b=n(x,None,torch.full_like(x,1e6),None)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertGreaterEqual(int(n.history_index.min()),0)
        self.assertLessEqual(int(n.history_index.max()),95)

if __name__=='__main__':unittest.main()
