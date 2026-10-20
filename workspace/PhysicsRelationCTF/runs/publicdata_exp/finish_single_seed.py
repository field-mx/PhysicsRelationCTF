"""Finish validation selection and one frozen test unattended; no test-based tuning."""
import json
import subprocess
import sys
import time
from pathlib import Path
from data import ROOT,dump

CANDIDATES=['r5_boundary32','r6_boundary_nominal32','r7_gamma32','r8_anchored32','r9_control_reference32']
RATIOS=[1.,.5,.3]


def folder(group,ratio,kind):
    return ROOT/'development'/group/f'train{round(ratio*100):03d}'/f'{kind}_seed42'


def select():
    base={r:json.loads((folder('r4_matched32',r,'CTF')/'training.json').read_text()) for r in RATIOS}
    table={}
    for group in CANDIDATES:
        records={str(r):json.loads((folder(group,r,'Physics')/'training.json').read_text()) for r in RATIOS}
        score=sum(records[str(r)]['best_val_mse']/base[r]['best_val_mse'] for r in RATIOS)/len(RATIOS)
        table[group]=dict(mean_relative_validation_mse=score,validation=records)
    winner=min(CANDIDATES,key=lambda group:table[group]['mean_relative_validation_mse'])
    runs=[dict(kind=k,ratio=r,seed=42,path=str(folder('r4_matched32' if k=='CTF' else winner,r,k)))
          for r in RATIOS for k in ['CTF','Physics']]
    return dict(dataset='LBNL EnergyPlus-Modelica normal chiller plant; simulation, not field measurements',
        seeds=[42],ratios=RATIOS,chosen_group=winner,runs=runs,validation_table=table,
        selection_rule='Minimum mean validation MSE relative to matched32-epoch CTF across all three budgets; one global configuration',
        development_policy='All valid-boundary32-epoch configurations retained; no test-based tuning',
        limitation='Single fixed seed feasibility study; not statistical proof or pure independent-observation scarcity',
        forecast='96 native one-minute historical points -> full96-minute future trajectory; stride96',
        selected_options=json.loads((folder(winner,1.,'Physics')/'config.json').read_text())['options'])


def main():
    required=[folder(g,r,'Physics')/'training.json' for g in CANDIDATES for r in RATIOS]
    required += [folder('r4_matched32',r,'CTF')/'training.json' for r in RATIOS]
    deadline=time.monotonic()+7200
    while not all(p.exists() for p in required):
        dump(ROOT/'final/status.json',dict(state='waiting_validation',missing=[str(p) for p in required if not p.exists()],updated=time.time()))
        if time.monotonic()>deadline:raise TimeoutError('Validation did not finish; test remains locked')
        time.sleep(10)
    subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-q'],cwd=ROOT,check=True)
    selection=select();path=ROOT/'final/validation_selection.json'
    if path.exists():
        if json.loads(path.read_text())!=selection:raise RuntimeError('Refusing changed selection')
    else:dump(path,selection)
    print(json.dumps({'selected':selection['chosen_group'],'validation_table':selection['validation_table']}),flush=True)
    subprocess.run([sys.executable,'-B','-u','final_evaluate.py','--selection',str(path),'--device','cuda:0'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-B','summarize_final.py'],cwd=ROOT,check=True)


if __name__=='__main__':main()
