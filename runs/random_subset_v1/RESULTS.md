# Random-window few-shot experiment

## Protocol

- Existing prefix-few-shot results under `runs/final` were retained.
- The complete training-window pool contains 14,349 overlapping 96-to-96 windows.
- A fixed permutation (`subset_seed=20260922`) selected nested 50% (7,174) and 30% (4,304) subsets.
- CTF and Physics-CTF used identical windows, validation/test splits, optimizer, schedule and seeds 42/43/44.
- Validation-only tuning compared the existing setting with stronger `beta` or `gamma` initialization.
- Selected before test evaluation: `beta=0.05`, `gamma=0.75`, `rho=0.001`.
- Twelve checkpoints were frozen before the test set was evaluated once.

## Validation-only parameter selection (seed 42)

| Setting | 50% validation MSE | 30% validation MSE |
|---|---:|---:|
| CTF | 0.094644 | 0.093975 |
| Physics, beta=.05 gamma=.50 | 0.094294 | 0.093817 |
| Physics, beta=.10 gamma=.50 | 0.094271 | 0.093827 |
| **Physics, beta=.05 gamma=.75** | **0.094075** | **0.093742** |

## Frozen test results (mean ± sample std, 3 paired seeds)

| Training windows | Model | MSE | MAE |
|---|---|---:|---:|
| 50% | CTF | 0.110676 ± 0.000567 | 0.110120 ± 0.000930 |
| 50% | Physics-CTF | **0.109824 ± 0.000600** | **0.109504 ± 0.000631** |
| 30% | CTF | 0.110610 ± 0.000440 | **0.110059 ± 0.000020** |
| 30% | Physics-CTF | **0.110392 ± 0.000473** | 0.110094 ± 0.000110 |

- 50% MSE improvement: **0.770%**; Physics-CTF wins all 3 paired seeds.
- 30% MSE improvement: **0.197%**; Physics-CTF wins all 3 paired seeds.
- 50% MAE improvement: about **0.559%**.
- 30% MAE changes by about **-0.031%** (slightly worse); only seed 42 improves MAE.
- Mean per-channel MSE improves on 33/56 channels at 50%, and 24/56 at 30%.
- Mean horizon MSE improves on 92/96 horizons at 50%, and 82/96 at 30%.

## Interpretation boundary

Randomly retaining overlapping stride-1 windows improves operating-regime coverage and removes the
chronological-prefix distribution-shift confound. It does **not** create strong raw-observation scarcity:
the selected 50% windows cover 14,535/14,540 training timestamps (99.966%), and the 30% windows cover
14,526/14,540 timestamps (99.904%). Therefore this experiment supports a more stable Physics-CTF MSE
advantage under fewer optimization windows, but it does not establish that the advantage grows as the
amount of distinct physical observations decreases. The 30% gain is also smaller than the 50% gain.

Independent audit: all 12 prediction files use the same test origins; truth differences between ratios are
only float reconstruction error (maximum 4.77e-7); independently recomputed MSE differs from stored MSE
by at most 1.98e-9. All 17 repository tests passed after the experiment.
