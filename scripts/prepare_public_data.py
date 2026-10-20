"""Explicit public downloads/preparation, not automatic training or Test use."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['lbnl','tep'])
    a = p.parse_args()
    if a.dataset == 'lbnl':
        pub = ROOT/'workspace/PhysicsRelationCTF/runs/publicdata_exp'
        sys.path.insert(0, str(pub))
        import acquire
        acquire.main()
        expected = json.loads((ROOT/'configs/lbnl/data_metadata.json').read_text())
        provenance = json.loads((pub/'data/provenance.json').read_text())
        # Refuse a silently changed public source version.
        original = json.loads((ROOT/'manifests/public_sources.json').read_text())
        if provenance['csv_sha256'] != original['lbnl_csv_sha256']:
            raise RuntimeError('Public raw file differs from the study version')
        from prepare import prepare_data
        meta = prepare_data()
        for key in ['channels','target_indices','rows','train_end','val_end']:
            if meta[key] != expected[key]:
                raise RuntimeError(f'Prepared task differs: {key}')
    else:
        helpers = ROOT/'workspace/PhysicsGuidedCTF'
        sys.path.insert(0, str(helpers))
        import download_tep
        download_tep.ROOT.mkdir(parents=True, exist_ok=True)
        rows = [download_tep.fetch(name) for name in
                ['TEP_FaultFree_Training.RData','TEP_FaultFree_Testing.RData']]
        expected_hashes = {Path(r['file']).name:r['sha256'] for r in
                           json.loads((ROOT/'manifests/public_sources.json').read_text())['tep_raw_files']}
        for row in rows:
            if row['sha256'] != expected_hashes[Path(row['file']).name]:
                raise RuntimeError('TEP source file differs from the study version')
        (download_tep.ROOT/'download_manifest.json').write_text(json.dumps(rows,indent=2))
        import prepare_data
        prepare_data.tep()  # Do not invoke the unavailable industrial preparation.
        actual = json.loads((helpers/'data/tep/metadata.json').read_text())
        expected = json.loads((ROOT/'configs/tep/data_metadata.json').read_text())
        for key in ['channels','ids','sample_seconds']:
            if actual[key] != expected[key]:
                raise RuntimeError(f'Prepared TEP task differs: {key}')
    print('Prepared the fixed public-data task; no model was trained or evaluated.')


if __name__ == '__main__':
    main()
