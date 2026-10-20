"""Executable regression checks; no training or test-set model selection."""
import os
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from physics import get_data,config,equation_features,FARADAY,OriginalCTF,FEATURES
from relation_model import PhysicsRelationCTF,copy_ctf

torch.set_num_threads(2)

class RelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,cls.graph,cls.meta,cls.selection,cls.scalers=get_data(.3)
        cfg=config(56);cfg.update(d_model=32,n_heads=4,d_ff=64,dropout=0.)
        torch.manual_seed(91)
        cls.base=OriginalCTF(SimpleNamespace(**cfg)).eval()
        cls.net=PhysicsRelationCTF(SimpleNamespace(**cfg),cls.graph,cls.scalers).eval()
        copy_ctf(cls.net,cls.base)
        cls.x=torch.stack([cls.data['train'][500][0],cls.data['train'][900][0]])

    def setUp(self):
        self.net.set_strengths();self.net.zero_grad(set_to_none=True)

    def test_zero_strength_exact_native_forecast(self):
        self.net.set_strengths(0,0,0)
        with torch.no_grad():
            a=self.net(self.x,None,None,None);b=self.base(self.x,None,None,None)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_physics_affects_forecast_and_receives_gradient(self):
        p=self.net(self.x,None,None,None)
        with torch.no_grad():q=self.base(self.x,None,None,None)
        self.assertGreater(float((p-q).abs().max().detach()),1e-6)
        p.square().mean().backward()
        for layer in self.net.encoder.layers:
            grad=layer.self_attention.inner_attention.theta_gamma.grad
            self.assertTrue(torch.isfinite(grad));self.assertGreater(float(grad.abs()),0)

    def test_lag_reads_history_only_and_forbidden_mass_zero(self):
        with torch.no_grad():self.net(self.x,None,None,None)
        self.assertGreaterEqual(int(self.net.history_index.min()),0)
        self.assertLessEqual(int(self.net.history_index.max()),95)
        pi=self.net.last_lag
        self.assertEqual(float(pi.masked_select(self.net.lag_forbidden[None]).sum()),0)
        torch.testing.assert_close(pi.sum(-1),torch.ones_like(pi.sum(-1)))
        for n,e in enumerate(self.graph['edges']):
            if e['relation_type']=='state_consistency':
                self.assertEqual(float((pi[:,n]*self.net.lags).sum()),0)

    def test_future_decoder_inputs_cannot_change_prediction(self):
        with torch.no_grad():
            a=self.net(self.x,None,torch.zeros_like(self.x),None)
            b=self.net(self.x,None,torch.full_like(self.x,1e6),None)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        with self.assertRaises(ValueError):self.net(torch.cat([self.x,self.x],1),None,None,None)

    def test_native_embedding_unchanged(self):
        recorded=[]
        h=self.net.channel_wise_patch_embedding.register_forward_hook(lambda m,a,o:recorded.append(o[0].detach().clone()))
        with torch.no_grad():
            self.net(self.x,None,None,None);self.net.set_strengths(0,0,0);self.net(self.x,None,None,None)
        h.remove();torch.testing.assert_close(recorded[0],recorded[1],rtol=0,atol=0)

    def test_equations_in_raw_units(self):
        raw=torch.zeros(1,1,56,dtype=torch.float64);ix={c:i for i,c in enumerate(self.meta['channels'])}
        for c,v in {'SCRDLFK':10000.,'SCRDYFK':200.,'TT010001':20.,'TT010002':30.,
                    'TT010301':50.,'TT010401':70.,'TT010403':65.,'FT010401':4.,
                    'PT010303':2.,'PT010501':1.5,'PT010302':2.2,'PT010301':2.}.items():raw[...,ix[c]]=v
        expected=torch.tensor([2000000.,10000/(2*96485.3321233100184),10000/(4*96485.3321233100184),
                               10.,20.,15.,80.,60.,.5,.2],dtype=torch.float64)
        torch.testing.assert_close(equation_features(raw,self.meta['channels'])[0,0],expected)

    def test_split_invariance_and_raw_recovery(self):
        full,_,_,_,_=get_data(1.)
        for s in ['val','test']:np.testing.assert_array_equal(full[s].origins,self.data[s].origins)
        ds=self.data['train'];_,t=ds.origins[500]
        raw=self.x[0].numpy()*np.array(self.scalers['raw_scale'])+np.array(self.scalers['raw_mean'])
        import json
        from physics import ROOT
        r=json.loads((ROOT.parent/'GNN_Transformer/runs/sensor_96s_96s_stride96/data_summary.json').read_text())
        mask=np.array(r['train_varying_mask'],bool)
        expected=ds.values[0,t-96:t]*np.array(r['scale'])[mask]+np.array(r['mean'])[mask]
        np.testing.assert_allclose(raw,expected,rtol=1e-5,atol=1e-3)
        self.assertLessEqual(int(ds.origins[:,1].max())+96,self.selection['train_end'])

    def test_runner_native_initialization_is_paired(self):
        from run_experiment import make_model
        cfg=config(56);cfg.update(d_model=32,n_heads=4,d_ff=64,dropout=0.)
        base=make_model('CTF',cfg,self.graph,self.scalers,42,{})
        phys=make_model('Physics',cfg,self.graph,self.scalers,42,{})
        for k,v in base.state_dict().items():torch.testing.assert_close(v,phys.state_dict()[k],rtol=0,atol=0)

    def test_inverse_relation_prefers_correct_lag(self):
        # Outflow rises while inventory falls: the reversed source is the
        # correct history, not a different lag that merely looks positive.
        from relation_model import lag_scores
        query=torch.tensor([[[0.,1.,2.]]])
        source=torch.tensor([[[[0.,-1.,-2.],[0.,1.,2.]]]])
        scores=lag_scores(query,source,torch.tensor([-1.]),torch.tensor([1.]))
        self.assertGreater(float(scores[0,0,0]),float(scores[0,0,1]))
        positive=lag_scores(query,source,torch.tensor([1.]),torch.tensor([1.]))
        self.assertGreater(float(positive[0,0,1]),float(positive[0,0,0]))

    def test_frozen_manifest_rejects_changed_source_or_data(self):
        from finalize import fingerprints,validate_manifest
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'input';p.write_bytes(b'original')
            manifest={'files':fingerprints([p])}
            validate_manifest(manifest)
            p.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):validate_manifest(manifest)

    def test_one_representation_selected_with_full_data_guard(self):
        from confirm_selection import choose_variant
        base={100:1.,50:1.,30:1.}
        choices={'anchored':{100:1.,50:.9,30:.99},'centered':{100:1.,50:.94,30:.92}}
        self.assertEqual(choose_variant(base,choices),'centered')
        choices['centered'][100]=1.02
        self.assertEqual(choose_variant(base,choices),'anchored')
        choices['anchored'][30]=1.01
        self.assertIsNone(choose_variant(base,choices))

    def test_isolated_projection_keeps_forward_but_blocks_extra_encoder_gradient(self):
        from isolated_model import IsolatedPhysicsCTF
        cfg=config(56);cfg.update(d_model=32,n_heads=4,d_ff=64,dropout=0.)
        model=IsolatedPhysicsCTF(SimpleNamespace(**cfg),self.graph,self.scalers).eval()
        model.load_state_dict(self.net.state_dict())
        a=model(self.x,None,None,None);b=self.net(self.x,None,None,None)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        # Only the side message is differentiated, not the native CTF path.
        model.encoder.layers[0].self_attention.inner_attention.context['physics'].sum().backward()
        self.assertIsNone(model.channel_wise_patch_embedding.embedding_dict['pl24'].weight.grad)
        self.assertGreater(float(model.adapter_out.grad.abs().sum()),0)
        self.net.encoder.layers[0].self_attention.inner_attention.context['physics'].sum().backward()
        self.assertGreater(float(self.net.channel_wise_patch_embedding.embedding_dict['pl24'].weight.grad.abs().sum()),0)

    def test_formula_scale_no_longer_depends_on_correlation_magnitude(self):
        from equation_normalized_model import EquationNormalizedPhysicsCTF
        import copy
        cfg=config(56);cfg.update(d_model=32,n_heads=4,d_ff=64,dropout=0.)
        a=copy.deepcopy(self.scalers);b=copy.deepcopy(a)
        for k,e in enumerate(self.graph['edges']):
            if e['feature']:b['gains'][k]*=.01
        m1=EquationNormalizedPhysicsCTF(SimpleNamespace(**cfg),self.graph,a)
        m2=EquationNormalizedPhysicsCTF(SimpleNamespace(**cfg),self.graph,b)
        torch.testing.assert_close(m1.gain,m2.gain,rtol=0,atol=0)
        for k,e in enumerate(self.graph['edges']):
            if not e['feature']:self.assertAlmostEqual(float(m1.gain[k]),a['gains'][k],delta=max(1e-6,abs(a['gains'][k])*1e-6))

    def test_late_injection_respects_no_edge_targets_and_serializes(self):
        from late_model import LatePhysicsCTF
        cfg=config(56);cfg.update(d_model=32,n_heads=4,d_ff=64,dropout=0.)
        model=LatePhysicsCTF(SimpleNamespace(**cfg),self.graph,self.scalers).eval()
        copy_ctf(model,self.base)
        native=self.base(self.x,None,None,None);late=model(self.x,None,None,None)
        incoming={e['target'] for e in self.graph['edges']}
        no_edge=[i for i in range(56) if i not in incoming]
        torch.testing.assert_close(late[:,:,no_edge],native[:,:,no_edge],atol=1e-6,rtol=1e-6)
        self.assertGreater(float((late-native).detach().abs().max()),1e-6)
        reloaded=PhysicsRelationCTF(SimpleNamespace(**cfg),self.graph,self.scalers).eval()
        reloaded.load_state_dict(model.state_dict())
        torch.testing.assert_close(reloaded(self.x,None,None,None),late,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
