import importlib.util
import unittest


class FinalProtocolTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('final_evaluate'))
        import final_evaluate
        return final_evaluate

    def test_requires_all_pairs_without_seed_selection(self):
        m=self.api()
        rows=[dict(kind=k,ratio=r,seed=s) for k in ['CTF','Physics'] for r in [1.,.5,.3] for s in [42,43,44]]
        m.validate_pairs(rows,[42,43,44])
        with self.assertRaises(ValueError):m.validate_pairs(rows[:-1],[42,43,44])
        with self.assertRaises(ValueError):m.validate_pairs(rows+[rows[0]],[42,43,44])

    def test_gain_and_degradation_are_not_confused(self):
        m=self.api()
        rows=[]
        for r,c,p in [(1.,1.,.9),(.5,2.,1.6),(.3,4.,3.6)]:
            for k,x in [('CTF',c),('Physics',p)]:
                rows.append(dict(kind=k,ratio=r,seed=42,mse=x,mae=x))
        results=m.summarize(rows)
        self.assertAlmostEqual(results['0.5']['gain_mse_percent'],20.)
        self.assertAlmostEqual(results['0.5']['CTF']['degradation_mse_percent'],100.)
        self.assertAlmostEqual(results['0.5']['Physics']['degradation_mse_percent'],(1.6/.9-1)*100)

if __name__=='__main__':unittest.main()
