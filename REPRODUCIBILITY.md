# Reproducibility scope

## Frozen evidence versus portable use

`workspace/` retains selected historical code/evidence with source-to-export
checksums. The mathematical PRCTF model files are unchanged. Two I/O-only
publication adaptations are recorded: the TEP cache path becomes relative to
the workspace, and LBNL preparation checks that `values.npy` exists before
trusting a metadata-only prepared directory. Local absolute paths in exported
JSON/Markdown/CSV are replaced with workspace-relative references.

Old orchestrators can require original checkpoint locks, full historical source
trees, old caches, or private industrial statistics. They are preserved for
traceability, **not falsely advertised as zero-setup portable launchers**.

The supported portable model constructor is `scripts/model_factory.py`.
It restores the selected public-data graph, calibration and initialization
without loading a trained checkpoint. Industrial construction through this
checkpoint-free path uses synthetic unit statistics solely for shape/equivalence
tests. It must not be used as a trained deployment model.

## Pretrained inference

The new `weights/` release contains the archived seed-42 PRCTF reference for
each dataset, including fitted normalization and physical calibration.
`scripts/pretrained.py` reconstructs the original model from that bundle and
loads the complete state strictly. `scripts/predict.py` accepts named-channel
CSV histories and returns physical-unit forecasts without retraining or access
to the original experiment directories. See [deployment instructions](weights/README.md)
and [validation records](weights/VALIDATION.md). These weights correspond to the
paper's archived protocol; the industrial post-noon retraining is not included.
Raw industrial observation sequences remain private.

## Task and metric contracts

| Dataset | Inputs / targets | Observations→forecast | Patch | Fixed seeds |
|---|---:|---:|---:|---|
| Industrial | 56 / 56 | 96→96 | 24 | 42,43,44 |
| LBNL | 73 / 58 | 96→96 | 24 | 42,43,44 |
| TEP | 52 / 52 | 20→20 | 5 | 42,43,44 |

Each paired comparison uses identical task, membership, target indices,
normalization, seed, common backbone initialization and training policy.
Train-only calibration and validation-only checkpoint selection are retained.
The public-dataset instances differ in conditional-neighbor weighting, validity
and confidence handling; do not present all three as byte-identical models.

Error reduction is `100 * (control_error - PRCTF_error) / control_error`.
Positive means improvement. Across-seed standard deviation is the sample
standard deviation (`ddof=1`). `Best` is the minimum for that metric; best MSE
and best MAE can come from different seeds. Archived training times are not a
controlled hardware-speed benchmark. Normalized errors from different tasks
are not pooled into one global error.

## Public data preparation

After reviewing upstream access conditions and installing data requirements:

```bash
pip install -r requirements-data.txt
python scripts/prepare_public_data.py --dataset lbnl
python scripts/prepare_public_data.py --dataset tep
```

These commands download public files and generate local caches; neither
starts training nor overwrites published evidence. Preparation preserves the
original public-data preprocessing routines and predetermined trajectory IDs.
The scripts are provided, but a fresh bulk download was not tested in this
packaging task. Public calibration snapshots are supplied in `configs/` and
their provenance points to the selected existing run, not a new fitting result.

## Public-data training

The separate portable launcher trains and selects using **Train/Validation
only**, writing outside the evidence directories:

```bash
python scripts/train_public.py --dataset lbnl --kind CTF --seed 42 --device cuda:0 --out outputs/lbnl_seed42
python scripts/train_public.py --dataset lbnl --kind Physics --seed 42 --device cuda:0 --out outputs/lbnl_seed42
python scripts/train_public.py --dataset tep --kind CTF --seed 42 --device cuda:0 --out outputs/tep_seed42
python scripts/train_public.py --dataset tep --kind Physics --seed 42 --device cuda:0 --out outputs/tep_seed42
```

Use 43 and 44 for the other fixed seeds, not a search for winning seeds. The
launcher reuses the frozen training engines and selected physics. It does not
automatically expose Test or claim to regenerate identical errors across
different software/hardware. A new paired result requires locking both selected
checkpoints before a separately authorized evaluation with the archived
evaluation functions. Original orchestrators document that selection/evaluation
contract. Full training was not run during packaging.

## Controls and limits

- `no_bias` disables beta only; it is not deletion of the entire physical graph.
- `no_equation` disables gamma only; other paths can still use equation context.
- The equation-input control has extra inputs and is not capacity/token matched.
- PhyGNN is a forecasting adaptation, not a full spacecraft reconstruction.
- PG-Reg-CTF is a custom soft-regularized control, not a standard full-system PINN.
- Some industrial comparison expansion was outcome-selected at seed42; that
  comparison is exploratory. Public comparator results retain all methods.
- Seed42 and the fixed Test sets were used during development. Additional seeds
  do not turn those datasets into new blind tests or independent plants.
- Few-shot and the earlier SDWPF bias-only study did not establish the proposed
  monotonic scarcity advantage. Their original local records are not deleted.

## Figures and industrial accuracy

The current low-pass test illustration is the author-selected pressure panel C
from a separate seed-42 run: causal second-order Butterworth, 0.05 Hz, filtered
Train/Validation/Test inputs and targets. Existing checkpoints use 96-point
histories and 96-step outputs. The illustration is updated to 12-second rolling
origins and equal-weight overlapping-forecast combination. It shows 4847 points;
the first 96 seconds are warm-up, and both models' errors are computed over the
same remaining 4751 points. It combines leads of 1–96 seconds and is not a
fixed-96-second-ahead or single 4847-step prediction. It is distinct from the
raw three-seed accuracy table and released reference weights. Industrial plots
are approved finished images; no numerical source series is redistributed.

## Validation actually performed

Consult `VALIDATION.md` for packaging checks completed in this task. Passing
content checks and synthetic forwards is not evidence that a new machine has
completed full training or that every dependency/download path has been tested.
