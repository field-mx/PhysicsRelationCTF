import importlib.util
import tempfile
import unittest
from pathlib import Path
import torch
from test_physical import CHANNELS

class TrainingTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('train_public'),'training pipeline not implemented')
        import train_public
        return train_public

    def test_unfrozen_test_is_rejected(self):
        t=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ck=root/'best.pth';ck.write_bytes(b'checkpoint')
            with self.assertRaises(RuntimeError):t.verify_test_lock(ck,root/'missing.json')

    def test_paired_common_weights_are_identical(self):
        t=self.api()
        from public_physics import build_graph
        from public_model import config
        g=build_graph(CHANNELS);sc={'mean':[0.]*8,'scale':[1.]*8,'gains':[1.]*len(g['edges'])}
        cfg=config(8);cfg.update(d_model=32,n_heads=4,d_ff=64,dropout=0.)
        a=t.make_model('CTF',cfg,g,sc,42,{})
        b=t.make_model('Physics',cfg,g,sc,42,{})
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
