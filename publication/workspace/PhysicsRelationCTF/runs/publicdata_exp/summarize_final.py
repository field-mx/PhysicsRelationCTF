"""Generate research result tables from frozen predictions; no training/inference."""
import csv
import json
from pathlib import Path
import numpy as np
from data import ROOT,dump


def main():
    obj=json.loads((ROOT/'final/summary.json').read_text())
    meta=json.loads((ROOT/'data/metadata.json').read_text())
    rows=obj['runs'];summary=obj['summary'];protocol=obj['protocol']
    lines=['# Public-data PhysicsRelationCTF feasibility experiment','',
        'Dataset: official LBNL ChillerPlant normal-operation EnergyPlus/Modelica simulation (not field measurements).',
        '[Official dataset](https://faultdetection.lbl.gov/dataset/simulated-chiller-plant/) · DOI 10.25984/1881324','',
        '## Frozen protocol','',
        '- Native one-minute sampling; 96 historical minutes → full96-minute trajectory; stride96.',
        '- Chronological70/15/15 split; no window crosses a split or missing training block.',
        '- 73 varying input channels; 58 continuous physical targets. Control commands/setpoints/calculated load are inputs only, not scored targets.',
        '- Primary MSE/MAE: macro average over58 targets and all96 horizons, normalized by fixed full-training standard deviations.',
        '- Paired native channel-token CTF weights, data, optimizer, schedule, batch size and maximum32 epochs; best-validation checkpoint for each run.',
        '- Fixed seed42 only. No statistical-significance or seed-robustness claim.',
        '- Nested50%/30% day blocks selected with full-training month/load stratification. Full-pool metadata/reference scales are known; this is a reduced-training-budget experiment, not total ignorance of unselected observations.',
        '- Test accessed only after global configuration and checkpoint/source/data hashes were frozen. No test-based iteration.','',
        '## Held-out results','',
        '| Train budget | CTF MSE | Physics MSE | MSE gain | CTF MAE | Physics MAE | MAE gain |',
        '|---|---:|---:|---:|---:|---:|---:|']
    comparisons={}
    for r in [1.,.5,.3]:
        s=summary[str(r)];a=s['CTF'];b=s['Physics']
        lines.append(f"| {r:.0%} | {a['mse_mean']:.8f} | {b['mse_mean']:.8f} | {s['gain_mse_percent']:+.3f}% | {a['mae_mean']:.8f} | {b['mae_mean']:.8f} | {s['gain_mae_percent']:+.3f}% |")
        base=next(x for x in rows if x['kind']=='CTF' and x['ratio']==r)
        phys=next(x for x in rows if x['kind']=='Physics' and x['ratio']==r)
        ix=base['target_indices']
        bm=np.array(base['per_channel_mse'])[ix];pm=np.array(phys['per_channel_mse'])[ix]
        comparisons[str(r)]=dict(improved_target_channels=int((pm<bm).sum()),targets=len(ix),
            improved_horizons=int((np.array(phys['horizon_mse'])<np.array(base['horizon_mse'])).sum()))
        names=[base['channels'][j] for j in ix]
        if 'CWL_SEC_DP' in names:
            j=names.index('CWL_SEC_DP');mask=np.arange(len(ix))!=j
            net=float((bm-pm).sum())
            comparisons[str(r)]['pressure_share_of_net_MSE_change'] = float((bm[j]-pm[j])/net) if abs(net)>1e-12 else None
            comparisons[str(r)]['gain_excluding_pressure_percent'] = float((1-pm[mask].mean()/bm[mask].mean())*100)
        comparisons[str(r)]['selected_horizons']={str(h):dict(CTF=base['horizon_mse'][h-1],Physics=phys['horizon_mse'][h-1]) for h in [1,10,48,96]}
        with (ROOT/f'final/per_channel_{round(r*100):03d}.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['channel','CTF_MSE','Physics_MSE','gain_percent'])
            for j,c,p in zip(ix,bm,pm):w.writerow([base['channels'][j],c,p,(1-p/c)*100 if c else 'undefined'])
    gains=[summary[str(r)]['gain_mse_percent'] for r in [1.,.5,.3]]
    won=sum(g>0 for g in gains)
    lines+=['','Positive gain means Physics is better; negative gain means worse.','',
        f'Observed outcome: Physics has lower test MSE in {won}/3 training budgets.',
        f'Strict increasing relative advantage with reduced data (100%→50%→30%): {gains[0]<gains[1]<gains[2]}.',
        'Neither a small single-seed gain nor a favorable ordering establishes a general small-sample law.','',
        '## Selected model and physics','',f"Global validation-selected candidate: `{protocol['chosen_group']}`.",
        f"Options: `{json.dumps(protocol['selected_options'],sort_keys=True)}`.",
        '- OriginalCTF channel/local tokens, mask and channel-token decoder remain unchanged.',
        '- Known heat transport, nominal water rho*cp, equipment-boundary energy balance, mixing and fan-affinity features enter existing typed RelationAttention messages.',
        '- Condenser return uses the documented common header, not assumed numbered chiller-to-tower connections.',
        '- Effective kappa bridges reported electrical power and thermal balance; fitted only on retained steady training rows. Its physical cause is not identified; it is not a measured efficiency.',
        '- The V4 model receives an explicit constant pressure-control prior that was removed from baseline varying input channels. Inputs/windows match, but prior information intentionally differs. Physics configurations received validation-only development; this is not equal-budget hyperparameter-search evidence.',
        '- V4 candidate additionally preserves historical pressure/temperature regulation reference levels, in target local-normalization coordinates, only while the corresponding flow is active. It does not replace predicted outputs or use future setpoints.','',
        '## Per-budget diagnostics','']
    diagnostics={}
    for row in rows:
        p=Path(row['path']);training=json.loads((p/'training.json').read_text())
        lines.append(f"- {row['ratio']:.0%} {row['kind']}: best epoch{row['best_epoch']}, parameters{training['parameters']:,}; checkpoint `{p/'best.pth'}`.")
        if row['kind']=='Physics':
            h=json.loads((p/'history.json').read_text());best=next(t for t in h if t['epoch']==row['best_epoch'])
            diagnostics[str(row['ratio'])]=dict(layers=best['diagnostics'],scalers=json.loads((p/'scalers.json').read_text()),
                graph_version=json.loads((p/'graph.json').read_text()).get('version'))
    lines+=['','Per-channel/horizon improvement counts: `final/comparisons.json`. Learned branch strengths and calibration: `final/physics_diagnostics.json`.','',
        '## Limitations and next evidence','',
        'This first public simulation study does not prove real-plant generalization, a causal benefit from the physical graph, or robustness across seeds. Those require preregistered additional seeds and matched false-graph/physics-component controls, not selecting favorable test windows.',
        'Earlier16-epoch pilots are retained but are not used as weaker final baselines. Boundary-invalid early adapters were excluded before final testing.',
        'If gains are absent or tiny, report that outcome; do not interpret it as confirmation of a universal physics-prior advantage.','',
        'All new code/data/checkpoints stay under `runs/publicdata_exp`; original project model files are imported read-only.']
    dump(ROOT/'final/comparisons.json',comparisons);dump(ROOT/'final/physics_diagnostics.json',diagnostics)
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
