"""Protocol tests use real model code and immutable first-round artifacts."""
import importlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from tep_model import TEPPhysicsCTF, OriginalCTF


class LengthProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.ck=torch.load(HERE.parent/'development/physics_nominal/best.pth',map_location='cpu',weights_only=False)

    def api(self):
        self.assertTrue((HERE/'length_model.py').exists(),'Variable-length model adapter is not implemented')
        return importlib.import_module('length_model')

    def test_96_adapter_reproduces_saved_best_model(self):
        a=self.api();cfg=a.configuration(96,96,24)
        original=TEPPhysicsCTF(cfg,self.ck['graph'],self.ck['scalers']).eval()
        adapted=a.LengthPhysicsCTF(cfg,self.ck['graph'],self.ck['scalers']).eval()
        original.load_state_dict(self.ck['model']);adapted.load_state_dict(self.ck['model'])
        torch.manual_seed(7);x=torch.randn(2,96,52)*.3
        with torch.no_grad():p=original(x,None,None,None);q=adapted(x,None,None,None)
        torch.testing.assert_close(p,q,rtol=1e-6,atol=1e-6)

    def test_short_lengths_zero_physics_equals_ctf(self):
        a=self.api()
        for l,h,p in [(20,20,5),(50,10,10)]:
            base,phys=a.pair(l,h,p,self.ck['graph'],self.ck['scalers'])
            base.eval();phys.eval();phys.set_strengths(0,0,0)
            x=torch.randn(2,l,52)
            with torch.no_grad():b=base(x,None,None,None);q=phys(x,None,None,None)
            self.assertEqual(q.shape,(2,h,52));torch.testing.assert_close(b,q,rtol=1e-6,atol=1e-6)

    def test_patch_projection_sees_every_observation_exactly_once(self):
        a=self.api()
        for l,h,p in [(20,20,5),(50,10,10)]:
            base,_=a.pair(l,h,p,self.ck['graph'],self.ck['scalers']);base.eval()
            x=torch.arange(l*52,dtype=torch.float32).reshape(1,l,52)/100
            captured=[]
            hook=base.channel_wise_patch_embedding.embedding_dict[f'pl{p}'].register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().clone()))
            with torch.no_grad():base(x,None,None,None)
            hook.remove();self.assertEqual(len(captured),52)
            expected=(x-x.mean(1,keepdim=True))/(x.var(1,unbiased=False,keepdim=True)+1e-5).sqrt()
            for c in range(52):
                self.assertEqual(captured[c].shape,(1,l//p,p))
                torch.testing.assert_close(captured[c].reshape(1,l),expected[:,:,c])

    def test_message_slots_lags_and_gradients(self):
        a=self.api()
        for l,h,p in [(20,20,5),(50,10,10)]:
            _,m=a.pair(l,h,p,self.ck['graph'],self.ck['scalers']);m.eval()
            self.assertGreaterEqual(int(m.history_index.min()),0);self.assertLess(int(m.history_index.max()),l)
            self.assertEqual(m.graph,self.ck['graph'])
            y=m(torch.randn(2,l,52)*.2,None,None,None);y.square().mean().backward()
            for layer in m.encoder.layers:
                a=layer.self_attention.inner_attention
                self.assertTrue(torch.isfinite(a.theta_gamma.grad));self.assertGreater(abs(float(a.theta_gamma.grad)),0)
                for slot in a.context['qs'].tolist():self.assertIn(slot%(l//p+2),[l//p,l//p+1])
            a=self.api()

    def test_no_padding_configuration_rejects_nondivisible_window(self):
        a=self.api()
        with self.assertRaises(ValueError):a.configuration(19,19,5)


if __name__=='__main__':unittest.main()
