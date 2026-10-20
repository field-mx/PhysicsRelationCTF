import importlib
from pathlib import Path
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent


class WindowProtocol(unittest.TestCase):
    def api(self):
        self.assertTrue((HERE/'runner.py').exists(),'Paired real-observation runner is not implemented')
        return importlib.import_module('runner')

    def test_common_origins_and_real_slices(self):
        a=self.api();raw=np.arange(2*500*52,dtype=np.float32).reshape(2,500,52)
        origin=a.origins(2,500,1)
        self.assertEqual(origin.shape,(862,2));self.assertEqual(origin[0].tolist(),[0,50]);self.assertEqual(origin[-1].tolist(),[1,480])
        short=a.RealWindows(raw,origin,20,20);long=a.RealWindows(raw,origin,50,10)
        for i in [0,430,431,861]:
            r,t=origin[i];x,y=short[i];u,v=long[i]
            np.testing.assert_array_equal(x,raw[r,t-20:t]);np.testing.assert_array_equal(u,raw[r,t-50:t])
            np.testing.assert_array_equal(y,raw[r,t:t+20]);np.testing.assert_array_equal(v,y[:10])

    def test_evaluation_boundaries(self):
        a=self.api();o=a.origins(100,960,20)
        self.assertEqual(o.shape,(4500,2));self.assertEqual(o[-1].tolist(),[99,930])
        self.assertTrue((o[:,1]>=50).all());self.assertTrue((o[:,1]+20<=960).all())

    def test_frozen_graph_and_scaling_come_from_first_round(self):
        a=self.api();g,s=a.physics_spec()
        self.assertEqual(len(g['features']),12);self.assertEqual(len(g['edges']),52)
        self.assertEqual(len(s['mean']),52);self.assertNotIn('adapter_version',g)

    def test_short_run_cannot_create_synthetic_windows(self):
        a=self.api()
        with self.assertRaises(ValueError):a.origins(1,60,1)


if __name__=='__main__':unittest.main()
