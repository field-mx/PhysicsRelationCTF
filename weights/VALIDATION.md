# Pretrained deployment validation

Date: 2026-11-05. No training, seed search or new accuracy evaluation was run.

## Checkpoint export

All three bundles use the seed-42 reference paths already recorded in
`configs/*/model.json`. The selected epoch is 12 for industrial, 31 for LBNL and
16 for TEP. The graph agrees exactly with the published graph configuration;
public calibration agrees with the existing configuration snapshot. Each
exported state tensor was compared with its original checkpoint and is exactly
equal. Configuration, graph, calibration and model state are restored together.

The release contains only model state and required inference metadata. It does
not include optimizer state, raw observations, saved prediction source arrays
or manuscript content. SHA-256 and source-checkpoint provenance are recorded in
[manifest.json](manifest.json).

## Executed CPU tests

Tests ran in separate processes with the pinned upstream CTF code, two CPU
threads per process and the existing project Python environment.
The same 21 checks also passed from an isolated copy containing only release
files, using the pinned CTF source explicitly. Inference required no original
experiment directories, observation files or private data caches.

| Dataset | Tests passed | Batched historical input | Default forecast shape |
|---|---:|---|---|
| Industrial | 7 / 7 | 2 × 96 × 56 | 2 × 96 × 56 |
| LBNL | 7 / 7 | 2 × 96 × 73 | 2 × 96 × 58 |
| TEP | 7 / 7 | 2 × 20 × 52 | 2 × 20 × 52 |

The 21 tests cover manual versus wrapped normalization/inference equivalence,
finite outputs, supervised-target selection, channel-name reordering, rejection
of incorrect shapes and nonfinite inputs, weight-integrity enforcement, absence
of observation/training-state payloads, and command-line CSV versus Python API
agreement. Normalization round-trip and reordered-channel tests require exact
array equality. CLI output must also preserve the forecast timing and refuse to
overwrite an existing result.

These are deployment behavior checks using synthetic inputs. They do not
establish a new test MSE, performance on a different electrolyzer, alarm accuracy
or warning lead time. The industrial bundle uses the original paper protocol,
not the later post-noon subset experiment.
