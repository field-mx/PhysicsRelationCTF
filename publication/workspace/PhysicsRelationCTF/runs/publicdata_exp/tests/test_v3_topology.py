import unittest
import importlib.util

class BoundaryTests(unittest.TestCase):
    def test_common_header_not_numbered_chiller_drives_tower_inlet(self):
        self.assertIsNotNone(importlib.util.find_spec('public_physics_v3'))
        from public_physics_v3 import build_graph
        channels=['CHL_SWCD_TEMP_1','CT_RW_TEMP_1','CDWL_RW_TEMP',
                  'CWL_PRI_SW_TEMPSPT','CHL_SW_TEMP_1','CT_SW_TEMPSPT','CT_SW_TEMP_1']
        g=build_graph(channels);pairs={(e['source_name'],e['target_name']) for e in g['edges']}
        self.assertNotIn(('CHL_SWCD_TEMP_1','CT_RW_TEMP_1'),pairs)
        self.assertIn(('CDWL_RW_TEMP','CT_RW_TEMP_1'),pairs)
        self.assertIn(('CWL_PRI_SW_TEMPSPT','CHL_SW_TEMP_1'),pairs)
        self.assertIn(('CT_SW_TEMPSPT','CT_SW_TEMP_1'),pairs)
        self.assertEqual(len(pairs),len(g['edges']))

if __name__=='__main__':unittest.main()
