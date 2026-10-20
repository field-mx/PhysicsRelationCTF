import importlib.util
from pathlib import Path
import unittest
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent


class ReplicationTests(unittest.TestCase):
    def api(self):
        self.assertTrue((HERE / 'replicate.py').exists(), 'Missing isolated replication entry')
        spec = importlib.util.spec_from_file_location('replication', HERE / 'replicate.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_six_pairs_no_seed42_retraining_or_fewshot(self):
        api = self.api()
        jobs = api.jobs()
        self.assertEqual(len(jobs), 12)
        self.assertEqual({j['seed'] for j in jobs}, {43, 44})
        self.assertEqual({j['dataset'] for j in jobs}, {'industrial', 'lbnl', 'tep'})
        self.assertTrue(all(j['ratio'] == 1.0 for j in jobs))
        for dataset in ['industrial', 'lbnl', 'tep']:
            for seed in [43, 44]:
                self.assertEqual({j['kind'] for j in jobs if j['dataset'] == dataset and j['seed'] == seed}, {'CTF', 'Physics'})
        for job in jobs:
            self.assertTrue(api.output(job).is_relative_to(HERE))

    def test_aggregation_keeps_losing_seeds_and_uses_sample_std(self):
        api = self.api()
        rows = [dict(seed=s, kind=k, mse=m, mae=m/2) for s,c,p in
                [(42,1.,.8),(43,2.,2.2),(44,3.,2.4)] for k,m in [('CTF',c),('Physics',p)]]
        result = api.aggregate(rows)
        self.assertAlmostEqual(result['CTF']['mse_mean'], 2.)
        self.assertAlmostEqual(result['CTF']['mse_std'], 1.)
        self.assertEqual(result['mse_wins'], 2)
        self.assertEqual(len(result['paired']), 3)
        self.assertLess(result['paired'][1]['mse_gain_percent'], 0.)
        with self.assertRaises(ValueError):
            api.aggregate(rows[:-1])

    def test_resume_adapters_preserve_serializable_protocol(self):
        api = self.api()
        self.assertTrue(hasattr(api, 'prepare_engine'), 'Missing resume adapters')
        for dataset in ['industrial', 'lbnl']:
            code = f'''import sys,json
sys.path.insert(0,{str(HERE)!r})
import replicate as r
e=r.engine({dataset!r});r.prepare_engine({dataset!r},e)
if {dataset!r}=='industrial':
 s=e.get_data(1.,sampling='random_gaps',subset_seed=20260922)[3]
 assert s==json.loads(json.dumps(s)), 'Resume config comparison rejects tuple/list'
else:
 ds=e.Windows('train',1.);g=e.build_graph(ds.meta['channels'])
 assert g['retained_segment_lengths']==[b-a for a,b in ds.meta['selections']['1.0']['intervals']]
'''
            subprocess.run([sys.executable,'-B','-c',code],check=True)

    def test_tep_status_writes_are_separate_from_other_worker(self):
        api=self.api()
        self.assertTrue(hasattr(api,'isolated_dump'), 'Missing per-worker status isolation')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for kind in ['CTF','Physics']:
                save=api.isolated_dump(api.dump,root,root/'development'/kind)
                save(root/'status.json',dict(kind=kind))
                save(root/'development'/kind/'config.json',dict(value=1))
            self.assertFalse((root/'status.json').exists())
            for kind in ['CTF','Physics']:
                self.assertEqual(api.read(root/'development'/kind/'status.json')['kind'],kind)
                self.assertEqual(api.read(root/'development'/kind/'config.json')['value'],1)

    def test_failure_drains_other_running_child(self):
        api=self.api()
        self.assertTrue(hasattr(api,'drain_workers'), 'Missing failure cleanup')
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.1)'])
        api.drain_workers([(dict(dataset='fixture'),child,'cpu')])
        self.assertEqual(child.poll(),0)

    def test_seed42_metrics_are_in_protected_reference_files(self):
        api=self.api()
        self.assertTrue(hasattr(api,'seed42_metric_path'), 'Missing old-metric protection')
        for dataset in ['industrial','lbnl','tep']:
            for kind in ['CTF','Physics']:
                path=api.seed42_metric_path(dataset,kind)
                self.assertTrue(path.exists())
                self.assertIn('mse',api.read(path))


if __name__ == '__main__':
    unittest.main()
