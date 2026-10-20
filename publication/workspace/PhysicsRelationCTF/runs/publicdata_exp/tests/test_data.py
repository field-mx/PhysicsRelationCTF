import importlib.util
import unittest
import numpy as np


class DataTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('data'), 'isolated data adapter not implemented')
        import data
        return data

    def test_windows_never_cross_gap_or_split(self):
        d = self.api()
        o = d.make_origins([(0, 288), (500, 788)], stride=96)
        np.testing.assert_array_equal(o, [96, 192, 596, 692])

    def test_nested_blocks_are_exact_and_deterministic(self):
        d = self.api()
        blocks = [(i*1440, (i+1)*1440) for i in range(20)]
        strata = [i % 4 for i in range(20)]
        a = d.nested_blocks(blocks, strata, [1., .5, .3], 42)
        b = d.nested_blocks(blocks, strata, [1., .5, .3], 42)
        self.assertEqual(a, b)
        self.assertEqual(len(a['0.5']), 10)
        self.assertEqual(len(a['0.3']), 6)
        self.assertTrue(set(map(tuple,a['0.3'])) <= set(map(tuple,a['0.5'])))

    def test_units_numeric(self):
        d = self.api()
        self.assertAlmostEqual(float(d.to_si(np.array([32.]), 'degF')[0]), 0.)
        self.assertAlmostEqual(float(d.to_si(np.array([1.]), 'GPM')[0]), 0.0000630901964)
        self.assertAlmostEqual(float(d.to_si(np.array([1.]), 'kW')[0]), 1000.)

    def test_standardizer_cannot_read_future(self):
        d = self.api()
        a = np.array([[1.,5.],[3.,5.],[1000.,-9.]])
        mean, scale, active = d.fit_scaler(a[:2])
        np.testing.assert_array_equal(active, [True,False])
        np.testing.assert_allclose(mean, [2.,5.])
        np.testing.assert_allclose(scale, [1.,1.])


if __name__ == '__main__': unittest.main()
