import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from test_physical import CHANNELS
import test_model


class V2IntegrationTests(test_model.ModelTests):
    def models(self):
        import public_model_v2 as m
        from public_physics_v2 import build_graph
        g=build_graph(CHANNELS)
        sc=dict(mean=[0.]*len(CHANNELS),scale=[1.]*len(CHANNELS),
                gains=[1.]*len(g['edges']),kappa=.285)
        cfg=m.config(len(CHANNELS));cfg.update(d_model=32,d_ff=64,n_heads=4,dropout=0.)
        base=m.OriginalCTF(SimpleNamespace(**cfg)).eval()
        net=m.PublicPhysicsCTF(SimpleNamespace(**cfg),g,sc).eval();m.copy_ctf(net,base)
        x=torch.randn(2,96,len(CHANNELS));x[...,:2]=.01;x[...,2:6]+=25.;x[...,6]=1000.
        return base,net,x

    def test_complete_model_startup_is_finite(self):
        _,net,x=self.models();x[:,:48,:2]=0.
        y=net(x,None,None,None)
        self.assertTrue(torch.isfinite(y).all())
        y.square().mean().backward()
        for p in net.parameters():
            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())

    def test_empty_freeze_manifest_is_rejected(self):
        import train_v2
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);ck=p/'best.pth';ck.write_bytes(b'fake')
            lock=p/'lock.json';lock.write_text(json.dumps({'checkpoints':{str(ck):train_v2.sha(ck)}}))
            with self.assertRaisesRegex(RuntimeError,'Incomplete'):
                train_v2.verify_test_lock(ck,lock)


if __name__=='__main__':unittest.main()
