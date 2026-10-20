"""Recompute all method mean/std/best values without training or selection."""
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT/'results/method_comparison_per_seed.csv').open() as f:
        rows = list(csv.DictReader(f))
    summary = []
    for ds, model in sorted({(r['dataset'],r['model']) for r in rows}):
        selected = [r for r in rows if r['dataset'] == ds and r['model'] == model]
        if len(selected) != 3 or {int(r['seed']) for r in selected} != {42,43,44}:
            raise ValueError('All three fixed seeds are required, without selection')
        result = dict(dataset=ds,model=model)
        for key in ['mse','mae']:
            values = [float(r[key]) for r in selected]
            result.update({key+'_mean':statistics.mean(values),key+'_std':statistics.stdev(values),key+'_best':min(values)})
        summary.append(result)
    with (ROOT/'results/method_comparison_summary.csv').open() as f:
        archived = {(r['dataset'],r['model']):r for r in csv.DictReader(f)}
    for row in summary:
        old = archived[(row['dataset'],row['model'])]
        for key,value in row.items():
            if key not in ['dataset','model'] and abs(float(old[key])-value) > 1e-14:
                raise RuntimeError('Exported summary does not match complete per-seed results')
    print(json.dumps(dict(rows=len(rows),method_dataset_pairs=len(summary),summary_verified=True),indent=2))


if __name__ == '__main__':
    main()
