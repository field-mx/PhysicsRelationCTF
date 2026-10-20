import importlib.util
import unittest
import torch

CHANNELS = ['CHL_CW_FLOW_1','CHL_CD_FLOW_1','CHL_RW_TEMP_1','CHL_SW_TEMP_1',
            'CHL_RWCD_TEMP_1','CHL_SWCD_TEMP_1','CHL_POW_1','CHL_STA_1']

class PhysicsTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('public_physics'),'physical adapter not implemented')
        import public_physics
        return public_physics

    def test_heat_and_energy_balance_in_si(self):
        p = self.api(); graph = p.build_graph(CHANNELS)
        x = torch.tensor([[[.01,.01,20.,10.,30.,42.5,104500.,1.]]],dtype=torch.float64)
        bank,valid = p.equation_bank(x,graph)
        ix = {f['name']:i for i,f in enumerate(graph['features'])}
        for name,want in [('evap_heat_1',418000.),('cond_heat_1',522500.),
                          ('balance_power_1',104500.),('balance_cond_temp_1',42.5),('balance_evap_temp_1',10.)]:
            self.assertAlmostEqual(float(bank[0,0,ix[name]]),want)

    def test_zero_flow_equation_is_disabled_and_finite(self):
        p=self.api();g=p.build_graph(CHANNELS)
        x=torch.zeros(2,96,len(CHANNELS));x[...,2:6]=30.
        b,v=p.equation_bank(x,g)
        self.assertTrue(torch.isfinite(b).all())
        for i,f in enumerate(g['features']):
            if f['op'].startswith('balance_'):self.assertFalse(v[...,i].any())

    def test_edges_are_unique_direct_query_target_key_source(self):
        g=self.api().build_graph(CHANNELS)
        pairs=[(e['source'],e['target']) for e in g['edges']]
        self.assertEqual(len(pairs),len(set(pairs)))
        self.assertTrue(all(s!=t for s,t in pairs))
        self.assertIn((CHANNELS.index('CHL_POW_1'),CHANNELS.index('CHL_SWCD_TEMP_1')),pairs)
        self.assertTrue(all('boundary' in e and 'equation' in e for e in g['edges']))

if __name__=='__main__':unittest.main()
