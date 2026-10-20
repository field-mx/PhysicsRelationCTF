import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import summarize_final
from final_evaluate import summarize


class ArtifactTests(unittest.TestCase):
    def test_report_is_built_from_metrics_without_claiming_universal_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'data').mkdir();(root/'final').mkdir()
            (root/'data/metadata.json').write_text('{}')
            rows=[]
            for ratio in [1.,.5,.3]:
                for kind,error in [('CTF',1.),('Physics',1.1)]:
                    p=root/f'{kind}_{ratio}';p.mkdir()
                    for name,obj in [('training.json',{'parameters':100}),('history.json',[{'epoch':1,'diagnostics':[]}]),('scalers.json',{}),('graph.json',{})]:
                        (p/name).write_text(json.dumps(obj))
                    rows.append(dict(kind=kind,ratio=ratio,seed=42,mse=error,mae=error,path=str(p),best_epoch=1,
                                     target_indices=[0,1],channels=['CWL_SEC_DP','temperature'],
                                     per_channel_mse=[error,error],horizon_mse=[error]*96))
            obj=dict(runs=rows,summary=summarize(rows),protocol={'chosen_group':'fixture','selected_options':{}})
            (root/'final/summary.json').write_text(json.dumps(obj))
            with patch.object(summarize_final,'ROOT',root):summarize_final.main()
            report=(root/'RESULTS.md').read_text()
            self.assertIn('lower test MSE in 0/3',report)
            self.assertIn('-10.000%',report)
            self.assertTrue((root/'final/per_channel_030.csv').exists())

if __name__=='__main__':unittest.main()
