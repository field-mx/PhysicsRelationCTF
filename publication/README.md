# PhysicsRelationCTF (PRCTF)

Physics-guided multivariate forecasting with incomplete process knowledge.

This publication companion preserves ChannelTokenFormer (CTF) channel-token
decoding and introduces two complementary physical interfaces:

1. **Topology guidance:** sparse directed process relations bias channel-token
   attention logits without removing the remaining data-driven interactions.
2. **Equation messages:** history-only equation features are mapped to target
   representation scales and fused inside attention aggregation, with small
   learned adapters and residual compensation for unmodeled effects.

This is not a complete plant simulator, strict conservation solver, or classic
GCN rewritten as an identical Transformer operator. The three physical systems
use related interfaces with explicitly documented instance-specific processing.

> Research code, experiment artifacts and approved finished figures. The
> manuscript and review materials are not included. No article DOI, acceptance,
> archived release DOI or unconfirmed software licence is asserted here.

## Contents

- `workspace/PhysicsRelationCTF/`: selected original model, training, comparison,
  test and evidence files, retaining the original relative module layout.
- `workspace/PhysicsGuidedCTF/`: CTF readout wrapper and supporting data tools.
- `workspace/GNN_Transformer/`: only the physical topology definitions needed by
  the industrial instance; not the old GCN training project.
- `configs/`: selected initialization, topology and public-data calibration.
- `results/`: all three seeds, method comparisons and component controls.
- `figures/`: authorized finished figures, not the manuscript or reference papers.
- `scripts/`: pinned-source setup, checkpoint-free model construction, dataset
  preparation and validation-only public-data training entrypoints.
- `manifests/`: source-to-export mapping and file checksums.

## Installation and quick check

Use Python 3.10 or a compatible environment. Install PyTorch for your platform
separately, then the relevant requirements. The exact installed packaging-time
environment is recorded separately; it is not claimed to be the original
training environment.

```bash
pip install -r requirements-model.txt
# Obtain CTF from its official fixed revision; it is not redistributed here.
python scripts/fetch_upstreams.py --ctf
python scripts/smoke_test.py --dataset industrial
python scripts/smoke_test.py --dataset lbnl
python scripts/smoke_test.py --dataset tep
python scripts/verify_package.py
python scripts/summarize_results.py
```

The industrial smoke test uses synthetic unit-scale values, **not industrial
observations or fitted industrial gains**. It checks execution and baseline
equivalence, not the industrial accuracy numbers.

See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for LBNL and TEP links, citations,
and the precise subsets. The original industrial numerical records, numerical
figure-source series and model weights are not distributed. Approved example
plots illustrate that dataset.

Public-data preparation/training commands are described in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). Full data acquisition and training are
not automatically run by installation or smoke tests.

## Archived full-data results

Fixed seeds: 42, 43, 44. Errors use each task's frozen normalization and target
selection; numbers across different datasets are not directly comparable.

| Dataset | Original CTF mean MSE | PRCTF mean MSE | Mean-error reduction | MSE wins |
|---|---:|---:|---:|---:|
| Industrial alkaline electrolyzer | 0.10901009 | 0.10858198 | 0.393% | 2/3 |
| LBNL simulated chiller plant | 0.00257001 | 0.00255002 | 0.778% | 2/3 |
| Tennessee Eastman process | 0.82707916 | 0.82118871 | 0.712% | 3/3 |

All seeds and stronger comparators are retained in the result CSVs. In the
tested configurations, MTGNN/PatchTST perform better on LBNL; direct equation
input and the custom equation-regularized control perform better on TEP.
Topology-bias removal changes TEP negligibly. These are small mean improvements
over CTF, not universal superiority, significance guarantees, causal discovery,
zero-shot transfer, or verified benefits that grow as data decrease.

## Figures

- [Architecture](figures/architecture/model_architecture.png)
- [Topology-to-attention mapping](figures/architecture/topology_attention.png)
- [Physical message construction](figures/architecture/physics_message.png)
- [English industrial topology](figures/topology/electrolyzer_topology_en.png)
- [Industrial raw-signal overview](figures/industrial/raw_signal_overview.png)
- [Filtered test-curve illustration](figures/industrial/lowpass_test_predictions.png)

The filtered illustration is a separate single-seed experiment, not the raw-data
three-seed table. The diagram assets are finished illustrations rather than
learned causal graphs. Consult [figures/README.md](figures/README.md).

## Attribution and limits

CTF and comparison implementations remain subject to their upstream terms.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[LICENSE_STATUS.md](LICENSE_STATUS.md). See [CITATION.md](CITATION.md) for the
repository citation route; publication metadata must be updated only when real.

Internal writing/progress reports and manuscript files are excluded. Test data
were observed during earlier development; the reported replications are not
new blind tests.
