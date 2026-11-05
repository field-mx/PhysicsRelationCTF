# Pretrained PRCTF models

[English](README.md) | [简体中文](README_zh-CN.md)

These are the **validation-selected, seed-42 reference checkpoints** underlying
the archived paper results. They are supplied for inference without retraining.
They are not an ensemble, a new seed search, or a new accuracy experiment.

| Dataset | Weight file | History → forecast | Input / returned target channels | Sampling |
|---|---|---|---|---|
| Industrial alkaline electrolyzer | [Download](industrial/prctf_seed42.pt) | 96 → 96 | 56 / 56 | 1 s |
| LBNL simulated chiller plant | [Download](lbnl/prctf_seed42.pt) | 96 → 96 | 73 / 58 | 60 s |
| Tennessee Eastman Process | [Download](tep/prctf_seed42.pt) | 20 → 20 | 52 / 52 | 180 s |

The industrial checkpoint uses the original full-training-pool experiment
reported in this repository, **not** the later experiment restricted to records
after 12:00. Graphs and equations are specific to each dataset.

## What each file contains

Each `.pt` file contains the full trained `state_dict`, model configuration,
physical graph, channel order, target indices, sampling interval and fitted
normalization/calibration. The industrial aggregate statistics are newly
released because inference needs them. There are **no raw observation series,
forecast-source series, optimizer states or manuscript files** in the bundles.

[manifest.json](manifest.json) records byte sizes, SHA-256, source checkpoint
hashes, selected epochs and validation errors. The loader verifies the hash,
uses `torch.load(..., weights_only=True)` and restores with `strict=True`.
Model mathematics and archived results are unchanged. No new licence is assigned
by this weight release; see [licence status](../LICENSE_STATUS.md) and
[upstream notices](../THIRD_PARTY_NOTICES.md).

## Install and forecast

Run from the repository root, using Python 3.10+ and an appropriate PyTorch
build (the archived APIs require PyTorch ≥ 2.7):

```bash
pip install -r requirements-model.txt
# Review THIRD_PARTY_NOTICES.md first. Skip if the pinned upstream already exists.
python scripts/fetch_upstreams.py --ctf

python scripts/predict.py --dataset industrial \
  --input /path/to/history.csv --output outputs/forecast.csv --device cpu
```

Use `--device cuda:0` for CUDA. An existing pinned upstream can be supplied with
`--ctf-root /path/to/ChannelTokenFormer`. No original experiment directory or
private dataset cache is required for loading and inference.

### Input contract

- CSV has a header naming every required input channel exactly as listed in
  `manifest.json`. Column order is restored by name; extra columns are ignored.
- The final 96 rows (industrial/LBNL), or 20 rows (TEP), form one historical
  window. They must be chronological, continuous and from one operating record.
  The caller must enforce the sampling interval. Missing points are not filled;
  short histories and nonfinite values are rejected.
- **Industrial:** supply the original workbook's measurement units. Do not
  standardize or low-pass-filter values before passing them to this raw-data
  checkpoint. Use [the schema](../configs/industrial/data_schema.json) for IDs
  and [the graph](../configs/industrial/graph.json) for relationships. New sites
  require compatible measurement definitions and deployment validation.
- **LBNL:** supply prepared SI values: temperatures in °C, flows in m³/s,
  powers in W, pressure in Pa and unchanged dimensionless controls. The
  source `ChillerPlant.csv` uses different units. Apply the conversions in
  [data.py](../workspace/PhysicsRelationCTF/runs/publicdata_exp/data.py)
  according to `source_units` in
  [metadata](../configs/lbnl/data_metadata.json) **before** inference.
- **TEP:** supply native dataset units and the exact
  [52-channel order/schema](../configs/tep/data_metadata.json), without
  concatenating different simulation runs into a history.

The loader applies the checkpoint's training-fitted standardization, runs PRCTF
in evaluation mode, and reverses standardization. Output CSV contains
`forecast_step`, `seconds_ahead` and the predicted target channels in the input
physical units. It refuses to overwrite an existing output file.

### Python API

```python
import numpy as np
from scripts.pretrained import load_pretrained

predictor = load_pretrained("industrial", device="cpu")
history = np.load("/path/to/history.npy")  # real [96, 56] values, ordered below
prediction = predictor.predict(history, channels=predictor.channels)  # [96, 56]
print(predictor.output_channels)
```

Only pass `channels=predictor.channels` when the array really uses that order;
otherwise pass its actual column-name list and the loader will reorder it.
`[batch, history, channels]` arrays are also supported. LBNL returns only its
58 scored physical targets by default. `targets_only=False` returns all model
outputs; its other outputs were not all supervised prediction targets.
Use a fresh process for each dataset, as required by the archived module layout.

## Verify deployment

```bash
python scripts/verify_package.py
PRCTF_TEST_DATASET=industrial python -m unittest discover -s tests -p test_pretrained.py
PRCTF_TEST_DATASET=lbnl python -m unittest discover -s tests -p test_pretrained.py
PRCTF_TEST_DATASET=tep python -m unittest discover -s tests -p test_pretrained.py
```

These checks validate full-state loading, normalization, channel order, output
selection, CSV/API agreement and invalid-input handling using synthetic inputs.
They do not add a new forecasting benchmark or fault-warning evaluation.
Forecasting is an input to downstream warning logic; thresholds and alarm
decisions are outside this inference interface.
