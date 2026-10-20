"""Small meaningful checks: graph direction, physics units, baseline identity."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from run_experiment import load_development,make_model,TRIALS,seed_all
from tep_physics import equation_bank,CONV


class Protocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.ds,cls.graph,cls.scalers,_=load_development()

    def test_no_cross_run_windows(self):
        for ds in self.ds.values():
            self.assertTrue((ds.origins[:,1]>=96).all())
            self.assertTrue((ds.origins[:,1]+96<=ds.values.shape[1]).all())
        self.assertEqual(len(self.ds['train']),24720)

    def test_edges_are_equation_inputs(self):
        pairs=[]
        for e in self.graph['edges']:
            f=self.graph['features'][e['feature']]
            self.assertIn(e['source'],f['inputs']);self.assertEqual(e['target'],f['target'])
            self.assertNotEqual(e['source'],e['target']);pairs.append((e['target'],e['source']))
        self.assertEqual(len(set(pairs)),len(pairs))

    def test_units_and_invariants(self):
        raw=torch.tensor(self.scalers['mean'],dtype=torch.float64).repeat(2,96,1)
        eq,v=equation_bank(raw,self.graph)
        self.assertTrue(torch.isfinite(eq).all());self.assertTrue(v.all())
        names=[f['name'] for f in self.graph['features']];c=self.graph['channels']
        self.assertTrue(torch.allclose(eq[...,names.index('feed_A')],raw[...,c.index('xmv_3')]*CONV))
        changed=raw.clone();changed[...,c.index('xmv_3')]+=1
        eq2,_=equation_bank(changed,self.graph)
        self.assertTrue(torch.allclose(eq2[...,0]-eq[...,0],torch.full_like(eq[...,0],CONV)))

    def test_zero_branch_equals_channel_token_ctf(self):
        base=make_model(TRIALS[0],self.graph,self.scalers).eval()
        phys=make_model(TRIALS[1],self.graph,self.scalers).eval();phys.set_strengths(0,0,0)
        x=torch.stack([self.ds['train'][i][0] for i in [0,800]])
        with torch.no_grad():a=base(x,None,None,None);b=phys(x,None,None,None)
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)

    def test_equations_and_head_backpropagate(self):
        model=make_model(TRIALS[3],self.graph,self.scalers).train()
        x,y=self.ds['train'][500];pred=model(x[None],None,None,None)
        self.assertEqual(pred.shape,(1,96,52));loss=(pred-y[None]).square().mean();loss.backward()
        self.assertTrue(torch.isfinite(model.closure_logit.grad).all())
        self.assertGreater(model.closure_logit.grad.abs().sum().item(),0)
        for layer in model.encoder.layers:
            self.assertIsNotNone(layer.self_attention.inner_attention.theta_gamma.grad)


if __name__=='__main__':unittest.main()
