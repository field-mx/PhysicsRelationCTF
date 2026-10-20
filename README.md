# PhysicsRelationCTF (PRCTF)

**English** | [简体中文](README_zh-CN.md)

## Physics-guided multivariate forecasting with incomplete process knowledge

PhysicsRelationCTF integrates **sparse physical relationships** and **equation-derived messages** into ChannelTokenFormer (CTF). It retains the native local tokens, channel tokens, masked attention and **channel-token decoder**. Known relationships guide information exchange; computable equation features supply its numerical content; data-driven attention and small residual modules learn what the available physics does not describe.

**Research question: How can a forecaster use partial physical knowledge when measurements, equipment parameters and dynamic equations are insufficient for a complete mechanistic model?**

<p align="center">
  <img src="figures/architecture/model_architecture.png" alt="PRCTF overview: native CTF enhanced by physical topology and equation messages" width="100%">
</p>

*Overview. Two complementary interfaces inject physical knowledge into channel-token attention. This is a conceptual diagram: equation messages are fused inside attention, not added as a separate correction to the final forecast.*

> Research code, archived experiment evidence and approved finished figures. The manuscript, industrial numerical observations and trained weights are not distributed. Three-seed mean MSE reductions versus Original CTF are **0.393% / 0.778% / 0.712%** on the industrial electrolyzer / LBNL / TEP tasks. These are not claims of universal superiority.

### Navigation

- [1. Problem discovery and research question](#problem)
- [2. Architecture and physical-information injection](#architecture)
- [3. Physical modeling in three systems](#physics)
- [4. Experiments and results](#results)
- [5. Installation and quick verification](#quick-start)
- [6. Public-data preparation and training](#reproduce)
- [7. Code and experiment map](#code-map)
- [8. Figures and industrial examples](#figures)
- [9. Scope and limitations](#limits)
- [10. Attribution, licence and citation](#citation)

<a id="problem"></a>
## 1. Problem discovery and research question

### 1.1 The application: forecasting is not only curve fitting

Industrial monitoring systems record electrical, thermal, pressure, flow, liquid-level and composition measurements. Their future trajectories can support process monitoring and operating decisions. These quantities interact through material transport, energy exchange and feedback control, but have different units and response times.

Our starting application is an **industrial alkaline water electrolysis system**. A useful predictor must extract information from coupled measurements without assuming that every statistical association corresponds to a known physical dependency.

<p align="center">
  <img src="figures/industrial/raw_signal_overview.png" alt="Author-approved overview of original industrial sensor signals" width="100%">
</p>

*Illustrative industrial measurements. This approved image is a qualitative overview, not a downloadable numerical dataset.*

### 1.2 The gap: physical knowledge is available, but incomplete

Complete plant modeling requires equipment parameters, measurement boundaries, operating conditions and dynamic relationships that are often unavailable in historical monitoring studies. Nevertheless, useful local knowledge remains: power depends on current and voltage, gas-generation reference rates follow Faraday's law, and thermal transport involves flow and temperature differences.

Using this partial knowledge raises two distinct questions:

| Question | Difficulty | PRCTF interface |
|---|---|---|
| **Which channels have a supported relationship?** | A sparse known graph should guide interaction without declaring unknown connections invalid. | A soft physical bias in channel-token attention. |
| **What information should pass along the relationship?** | A connection alone does not execute an equation or reconcile different measurement scales. | Equation-derived messages, target-scale mapping and small relation adapters. |

The principle is **calculate what the available physics describes, learn unknown dynamics, and compensate for unmodeled effects**. A few local equations are not treated as a closed model of the entire plant.

### 1.3 Why retain CTF?

CTF organizes within-channel history and cross-channel exchange through local and channel tokens. This creates explicit insertion points for interaction guidance and numerical messages. PRCTF therefore enhances the existing backbone rather than replacing channel tokens with the output of another GCN encoder.

The contribution concerns the **representation and use of incomplete physical information**, not a new forecasting backbone or a claim that attention and classical GCN are mathematically identical.

<a id="architecture"></a>
## 2. Architecture and physical-information injection

### 2.1 Unchanged CTF backbone

1. Normalize each historical channel and divide it into complete patches.
2. Embed patches as **local tokens** and introduce native learnable **channel tokens**.
3. Preserve the CTF mask: local tokens read local tokens from the same channel; channel tokens read their own local history and cross-channel tokens in the same slot.
4. Stack attention layers and decode forecasts **only from the final channel tokens**.

Token construction, masking and readout remain unchanged. The selected tasks use complete observed patches, without copied observations to make lengths divisible. Neither local-token decoding nor the old `GCN → replacement channel latent` path is used.

### 2.2 Interface A: physical topology guides attention

A directed relationship from source `j` to target `i` uses the convention:

$$
A^{\mathrm{phys}}_{ij}=1 \quad\Longleftrightarrow\quad j\rightarrow i.
$$

**Rows receive information as Queries; columns provide information as Keys/Values.** This fixed graph contains supported relationships, not correlations or a complete causal model. A lightweight network adjusts existing-edge strengths from historical context; it does not invent new physical edges.

At the relevant channel-token positions, attention becomes:

$$
\alpha^{(l)}=\operatorname{Softmax}_{\mathrm{key}}\!\left(
\frac{Q^{(l)}K^{(l)\top}}{\sqrt{d_h}}+\beta_l B(X_{\mathrm{hist}})+M
\right).
$$

`B` is the physical bias lifted into the token layout, `M` is the original mask, and `beta_l` controls layer-wise guidance. The bias applies **only to same-slot Channel–Channel positions**. Local–Local, Local–Channel and Channel–Local masking is unchanged. Other originally allowed cross-channel connections remain data-driven.

<p align="center">
  <img src="figures/architecture/topology_attention.png" alt="Directed physical relationships mapped to same-slot channel-token attention positions" width="100%">
</p>

*Topology-to-attention mapping. Layers highlight different paths for explanation but share the same physical prior. Colored cells indicate supported positions, not measured attention strength. Multiple layers propagate already-mixed channel representations without manually adding physical graph powers.*

### 2.3 Interface B: equations supply numerical messages

The content interface uses **observed history only**:

1. **Calculate equation features**, such as power `UI` and the per-cell hydrogen-generation reference `I/(2F)`.
2. **Map them to the target representation**, using train-calibrated observation gains and alignment restricted to each relationship's admissible historical offsets.
3. **Embed and adapt**, reusing a patch projection and a small relation-specific adapter to address equation–measurement mismatch.
4. **Fuse inside attention**, supplementing native values with equation messages and bounded residual messages on physical edges.

In the industrial implementation, message construction is summarized by:

$$
p^{\mathrm{phys}}_{ij}
=E_P(\phi^{\mathrm{eq}}_{ij})
+\operatorname{Adapt}_{r_{ij}}\!\left(
E_P(\phi^{\mathrm{eq}}_{ij})\Vert E_P(z^{\mathrm{lag}}_{j\to i})
\right).
$$

`phi_eq` denotes the equation feature after scale conversion and alignment; `z_lag` is aligned source history; `r_ij` is the known relationship type. The selected industrial configuration uses centered equation variations, not an assumed exact absolute sensor-output equation.

<p align="center">
  <img src="figures/architecture/physics_message.png" alt="Historical observations converted into equation messages and fused with CTF values and residuals" width="100%">
</p>

*Equation-message pipeline. Known functions supply relationship content; small adapters and history-based residuals address observation mismatch and unmodeled effects.*

Aggregation has the schematic form:

$$
O_i^{(l)}=O_{\mathrm{data},i}^{(l)}
+\sum_{j:\,j\to i\in\mathcal E}\omega_{ij}^{(l)}
\left(\gamma_l\overline p_{ij}^{(l)}+\rho_l\overline r_{ij}^{(l)}\right).
$$

Native data aggregation is retained. In the industrial instance, `omega` is the corresponding native attention weight, including training-time dropout. Public-data instances additionally use their documented conditional-neighbor weighting, validity and historical-consistency handling. They share the two-interface idea, **not identical processing in every detail**.

### 2.4 Known physics, unknown dynamics and strength control

| Component | Responsibility | Not assumed |
|---|---|---|
| Fixed graph and equation functions | Record relationships and calculate available features. | A complete topology or closed dynamic simulator. |
| Data attention and relation adapters | Learn remaining dependencies and observation mismatch. | Identification of all device parameters or true propagation delays. |
| Small residual messages | Compensate for effects absent from selected equations. | A separately predicted future correction curve. |
| Learnable `beta`, `gamma`, `rho` | Control topology guidance, equation messages and residuals. | Larger initial strengths are always better. |

Industrial strengths use `beta = softplus(theta_beta)`, `gamma = 2 sigmoid(theta_gamma)`, and `rho = 0.02 sigmoid(theta_rho)`; message norms are controlled relative to native values. Explicitly disabling **all three** interfaces restores the paired Original CTF forward path. Setting only `beta = 0` does not disable equation or residual messages.

<a id="physics"></a>
## 3. Physical modeling in three systems

Applying the framework to another system requires its corresponding graph, equations and observation mapping. Electrolyzer physics is not copied unchanged to the public datasets.

| System | Measurements | Encoded knowledge examples | Important boundary |
|---|---|---|---|
| Industrial alkaline electrolyzer | Current, voltage, temperatures, pressures, flows, levels and gas composition. | `P = UI`; Faraday references `I/(2F)`, `I/(4F)`; flow–temperature-difference heat proxies; cooling temperature difference; separator pressure difference. | Unknown cell count, efficiencies, inventories and measurement boundaries prevent exact closure. |
| LBNL simulated chiller plant | Water temperatures/flows, power, pressures, pump/fan states and control references. | Water-side heat transport, chiller energy-balance features, mixing, affinity relationships and historical control references. | Effective calibration and consistency handling address nominal equations that do not fit every operating condition. |
| Tennessee Eastman process | Process flows, pressures, temperatures, compositions and manipulated variables. | Simulator-supported valve–flow, pressure–flow, compressor, heat-transfer, cooling-balance and liquid-density relations. | Noisy measurements, delayed analyzers and imperfect command–valve correspondence; unavailable hidden-state kinetics are not invented. |

Selected graphs contain **73 directed edges / 9 relation types** for 56 industrial channels, **112 / 6** for 73 LBNL channels, and **52 / 6** for 52 TEP channels. Entries retain equation sources, relation types, candidate historical offsets and limitations.

<details>
<summary>Industrial topology and its channel-attention representation</summary>

<p align="center">
  <img src="figures/topology/electrolyzer_topology_en.png" alt="Industrial directed topology labeled with physical measurement names" width="100%">
</p>

<p align="center">
  <img src="figures/topology/electrolyzer_channel_attention_en.png" alt="Channel-level matrix representation of the same industrial topology" width="100%">
</p>

The node graph and matrix represent the same prior, not learned attention heatmaps. Token-level lifting repeats each relationship at matching channel-token slots. Chinese figures and the full masked-attention mapping are in [figures/topology](figures/topology).

</details>

Exact definitions: [industrial graph](configs/industrial/graph.json), [LBNL graph](configs/lbnl/graph.json), [TEP graph](configs/tep/graph.json). These snapshots, rather than schematic figures, define the selected experiments.

<a id="results"></a>
## 4. Experiments and results

### 4.1 Evaluation contract

The full-data comparisons use **three fixed seeds: 42, 43 and 44**. Each CTF–PRCTF pair shares split membership, observations, forecast horizon, targets, preprocessing, common backbone initialization and training policy.

| Dataset | Sampling interval | Inputs / scored targets | History → forecast | Patch |
|---|---|---:|---:|---:|
| Industrial electrolyzer | 1 second | 56 / 56 | 96 → 96 points | 24 |
| LBNL simulated chiller plant | 1 minute | 73 / 58 | 96 → 96 points | 24 |
| TEP | 3 minutes | 52 / 52 | 20 → 20 points | 5 |

Not all LBNL control/setpoint inputs are scored targets. TEP windows never cross simulation-run boundaries. Normalization and empirical calibration use training data; checkpoints are selected on validation data. Different tasks' normalized errors are **not** directly comparable or pooled.

### 4.2 Original CTF versus PRCTF

Values are **mean ± sample standard deviation** across seeds (`ddof = 1`); lower is better.

| Dataset | Model | Normalized MSE | Normalized MAE |
|---|---|---:|---:|
| Industrial | Original CTF | 0.10901009 ± 0.00038041 | 0.10809476 ± 0.00016571 |
| Industrial | **PRCTF** | **0.10858198 ± 0.00075427** | **0.10775839 ± 0.00079343** |
| LBNL | Original CTF | 0.00257001 ± 0.00003264 | 0.00462957 ± 0.00018936 |
| LBNL | **PRCTF** | **0.00255002 ± 0.00003837** | **0.00456146 ± 0.00007519** |
| TEP | Original CTF | 0.82707916 ± 0.00123123 | 0.70286754 ± 0.00060949 |
| TEP | **PRCTF** | **0.82118871 ± 0.00135276** | **0.70023528 ± 0.00075871** |

Reduction is `100 × (CTF error − PRCTF error) / CTF error`: **positive means lower PRCTF error**. Mean-error reductions use the two across-seed means, not independently selected best runs.

| Dataset | Mean MSE reduction | Mean MAE reduction | Largest paired-seed MSE reduction | MSE wins |
|---|---:|---:|---:|---:|
| Industrial | **0.393%** | 0.311% | 1.275% | 2 / 3 |
| LBNL | **0.778%** | 1.471% | 2.588% | 2 / 3 |
| TEP | **0.712%** | 0.375% | 0.760% | 3 / 3 |

These results support **small mean gains over CTF in three physical settings**, with 7 of 9 paired MSE wins. Maximum single-seed reductions are not average performance. Three seeds do not establish statistical significance or independent-plant replication.

<details>
<summary>All nine paired-seed MSE comparisons, including regressions</summary>

| Dataset | Seed | CTF MSE | PRCTF MSE | Reduction |
|---|---:|---:|---:|---:|
| Industrial | 42 | 0.10912231 | 0.10773149 | +1.275% |
| Industrial | 43 | 0.10858620 | 0.10884467 | −0.238% |
| Industrial | 44 | 0.10932176 | 0.10916977 | +0.139% |
| LBNL | 42 | 0.00258835 | 0.00253391 | +2.103% |
| LBNL | 43 | 0.00253232 | 0.00259382 | −2.428% |
| LBNL | 44 | 0.00258936 | 0.00252234 | +2.588% |
| TEP | 42 | 0.82692956 | 0.82170499 | +0.632% |
| TEP | 43 | 0.82592958 | 0.81965383 | +0.760% |
| TEP | 44 | 0.82837836 | 0.82220730 | +0.745% |

Source: [ctf_prctf_paired.csv](results/ctf_prctf_paired.csv), also containing per-pair MAE.

</details>

### 4.3 Component controls: what contributes?

These controls are trained separately, not merely switches applied to one trained checkpoint. Compare the **full model directly with the model disabling an interface**:

| Dataset | Full PRCTF mean MSE | `no_bias` mean MSE | `no_equation` mean MSE |
|---|---:|---:|---:|
| Industrial | 0.10858198 | 0.10874063 | 0.10924427 |
| LBNL | 0.00255002 | 0.00255095 | 0.00257068 |
| TEP | 0.82118871 | 0.82118710 | 0.82708094 |

- `no_bias` disables **beta only**, not the graph or equation/residual messages. Full PRCTF has slightly lower mean MSE on Industrial and LBNL; TEP is effectively unchanged and marginally favors `no_bias`.
- `no_equation` disables **gamma only**; other branches can still use equation-related context. Full PRCTF lowers mean MSE by **0.606% / 0.804% / 0.712%** relative to this control on Industrial / LBNL / TEP.
- Explicit equation messages contribute more consistently; the evidence does not show every interface is independently useful everywhere.

Source: [ablation_and_input_per_seed.csv](results/ablation_and_input_per_seed.csv). The direct-input control tests whether simply supplying equation features matches the proposed injection mechanism.

### 4.4 Other methods and physics-informed controls

The archive retains **7 methods × 3 datasets × 3 seeds = 63 records**: PRCTF, Original CTF, MTGNN, PatchTST, a forecasting adaptation of PhyGNN, EquationInput-CTF and PG-Reg-CTF. The last two are custom controls, not independent published architectures or a standard full-system PINN.

**PRCTF is not best among all retained methods everywhere.** MTGNN and PatchTST obtain lower mean MSE on LBNL; direct equation input and the regularized control obtain lower mean MSE on TEP. Industrial comparator expansion followed a seed-42 pilot selection and is exploratory, not an unbiased architecture leaderboard.

<details>
<summary>Complete method-level mean errors and archived training time</summary>

Times are archived mean seconds, not a controlled hardware-efficiency benchmark. All rows use seeds 42/43/44. Standard deviations and metric-specific best values are in the CSV.

| Dataset | Method | Mean MSE | Mean MAE | Archived training time (s) |
|---|---|---:|---:|---:|
| Industrial | Original CTF | 0.10901009 | 0.10809476 | 514.08 |
| Industrial | **PRCTF** | **0.10858198** | **0.10775839** | 615.17 |
| Industrial | MTGNN | 0.11012124 | 0.10799903 | 182.18 |
| Industrial | PatchTST | 0.11144723 | 0.11123995 | 89.18 |
| Industrial | PhyGNN adapted | 0.14648583 | 0.18029671 | 69.83 |
| Industrial | EquationInput-CTF | 0.11826381 | 0.11990936 | 472.50 |
| Industrial | PG-Reg-CTF | 0.12150680 | 0.12295444 | 511.56 |
| LBNL | Original CTF | 0.00257001 | 0.00462957 | 449.79 |
| LBNL | **PRCTF** | 0.00255002 | 0.00456146 | 521.15 |
| LBNL | MTGNN | **0.00240191** | **0.00422672** | 162.06 |
| LBNL | PatchTST | 0.00253362 | 0.00438390 | 71.18 |
| LBNL | PhyGNN adapted | 0.00607557 | 0.03631560 | 30.75 |
| LBNL | EquationInput-CTF | 0.00255610 | 0.00463364 | 850.74 |
| LBNL | PG-Reg-CTF | 0.00442930 | 0.00710169 | 543.65 |
| TEP | Original CTF | 0.82707916 | 0.70286754 | 634.95 |
| TEP | **PRCTF** | 0.82118871 | 0.70023528 | 737.27 |
| TEP | MTGNN | 0.86585570 | 0.72509488 | 272.31 |
| TEP | PatchTST | 1.01060121 | 0.78922623 | 125.17 |
| TEP | PhyGNN adapted | 0.86653941 | 0.73170794 | 114.27 |
| TEP | EquationInput-CTF | **0.79712365** | **0.68775879** | 992.96 |
| TEP | PG-Reg-CTF | 0.79830351 | 0.68861369 | 1059.45 |

</details>

Sources: [per-seed records](results/method_comparison_per_seed.csv), [mean/std/best/time summary](results/method_comparison_summary.csv), [adapter disclosures](workspace/PhysicsRelationCTF/runs/physics_ai_comparison_v1/UPSTREAM.md).

<a id="quick-start"></a>
## 5. Installation and quick verification

### 5.1 Environment and pinned CTF source

Use Python 3.10 or a compatible environment. Install an appropriate CPU/CUDA PyTorch build separately; the study APIs require **PyTorch ≥ 2.7**. The environment snapshot records packaging-time software, not a guarantee of the original training environment.

```bash
git clone https://github.com/field-mx/PhysicsRelationCTF.git
cd PhysicsRelationCTF

python -m venv .venv
source .venv/bin/activate

# Install a suitable PyTorch build first, then:
pip install -r requirements-model.txt

# Review THIRD_PARTY_NOTICES.md before retrieving upstream code.
python scripts/fetch_upstreams.py --ctf
```

CTF is fetched at `b1c100effb9c27cd25de07e91de7f9005d18d5e7`, not bundled. The fetcher refuses to overwrite an existing upstream directory. Optional comparator retrieval: `python scripts/fetch_upstreams.py --comparators`, after reviewing its terms.

### 5.2 Check files, tables and synthetic forwards

```bash
# Standard-library checks; no data downloads or training.
python scripts/verify_package.py
python scripts/summarize_results.py

# Separate processes, after dependencies and CTF are available.
python scripts/smoke_test.py --dataset industrial
python scripts/smoke_test.py --dataset lbnl
python scripts/smoke_test.py --dataset tep
```

Smoke tests use paired common backbone weights, check finite outputs, explicitly disable all physical interfaces and compare against CTF. Recorded CPU packaging checks had maximum absolute differences of **0.0** for all three tasks. These are synthetic execution/equivalence checks, not accuracy tests. Industrial synthetic unit statistics are labeled and never substituted for industrial training data.

Append `--device cuda:0` for a GPU, or supply an existing pinned checkout with `--ctf-root /path/to/ChannelTokenFormer`.

<a id="reproduce"></a>
## 6. Public-data preparation and training

### 6.1 Obtain the correct public subsets

| Dataset | Source | Subset |
|---|---|---|
| LBNL | [Official chiller-plant page](https://faultdetection.lbl.gov/dataset/simulated-chiller-plant/) · [collection DOI](https://doi.org/10.25984/1881324) | Fault-free `ChillerPlant.csv`, not all fault scenarios. |
| TEP | [Harvard Dataverse](https://doi.org/10.7910/DVN/6C3JR1) · [retrieval mirror](https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset) | Fault-free training/testing RData files with fixed run membership. |

LBNL uses **367878 / 78831 / 78831 points** chronologically. TEP uses training-file runs **1–80 / 81–100** for Train/Validation, and testing-file runs **401–500** for Test. Raw public data are linked rather than mirrored. Citations, access conditions and provenance: [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

```bash
pip install -r requirements-data.txt
python scripts/prepare_public_data.py --dataset lbnl
python scripts/prepare_public_data.py --dataset tep
```

Preparation verifies source hashes and task membership; it does not train. Download/authentication conditions may change. A fresh bulk acquisition was not rerun during packaging, so these entry points are not advertised as a fully retested download service.

### 6.2 Train a controlled pair

```bash
# LBNL: 96 observed points -> 96 future points.
python scripts/train_public.py --dataset lbnl --kind CTF --seed 42 --device cuda:0 --out outputs/lbnl_seed42
python scripts/train_public.py --dataset lbnl --kind Physics --seed 42 --device cuda:0 --out outputs/lbnl_seed42

# TEP: 20 observed points -> 20 future points.
python scripts/train_public.py --dataset tep --kind CTF --seed 42 --device cuda:0 --out outputs/tep_seed42
python scripts/train_public.py --dataset tep --kind Physics --seed 42 --device cuda:0 --out outputs/tep_seed42
```

`Physics` selects PRCTF. Repeat with **43 and 44**, not a search for favorable seeds. This launcher trains/selects on **Train/Validation only**, without automatic Test evaluation. Use fresh output locations; published evidence directories cannot be training outputs.

| Instance | Initial beta / gamma / rho | Learning rate | Maximum epochs | Batch size |
|---|---|---:|---:|---:|
| Industrial | 0.05 / 0.75 / 0.001 | 0.0001 | 16 | 32 |
| LBNL | 0.05 / 0.10 / 0.001 | 0.0003 | 32 | 32 |
| TEP | 0.05 / 0.75 / 0.001 | 0.0003 | 16 | 64 |

All three selected CTF backbones use hidden dimension 256, FFN dimension 1024, 2 encoder layers, 8 heads, 2 channel tokens per channel and dropout 0.1. Configuration snapshots are **not trained weights**. Injection strengths remain learnable during training.

**Final evaluation is separate:** lock both validation-selected checkpoints before using archived evaluation functions. The portable launcher does not promise one-command, bit-identical reproduction of published errors. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the contract and historical dependencies.

### 6.3 Industrial reproduction boundary

Industrial observations, fitted scaling/gain statistics, weights and numerical figure-source series are not distributed. Graphs, schema, source, metrics and approved images support inspection, but industrial accuracy cannot be regenerated without authorized access to the original records. No unsupported promise of request-based access is made.

<a id="code-map"></a>
## 7. Code and experiment map

```text
.
├── README.md / README_zh-CN.md     # English / Chinese documentation
├── DATA_AVAILABILITY.md           # Sources, subsets and access boundaries
├── REPRODUCIBILITY.md             # Training/evaluation contract
├── configs/{industrial,lbnl,tep}/  # Selected model/physical-graph snapshots
├── results/                      # Complete comparison/control tables
├── figures/                      # Approved finished PNG/PDF figures
├── scripts/                      # Portable construction, preparation and checks
├── manifests/                    # File hashes and provenance
└── workspace/
    ├── PhysicsRelationCTF/        # Selected model/experiment source and evidence
    ├── PhysicsGuidedCTF/          # CTF channel-token wrapper and data helpers
    └── GNN_Transformer/           # Required historical topology definitions
```

The sibling `workspace/` layout preserves relative imports; it is not the full lab workspace. `GNN_Transformer/` does not mean the released model uses the old GCN encoder.

| What to inspect | Entry point |
|---|---|
| Portable paired construction | [scripts/model_factory.py](scripts/model_factory.py) |
| Industrial bias, messages, aggregation and residual | [relation_model.py](workspace/PhysicsRelationCTF/relation_model.py) |
| Industrial equation functions and relation/offset definitions | [physics.py](workspace/PhysicsRelationCTF/physics.py) |
| Original channel-token readout wrapper | [PhysicsGuidedCTF/model.py](workspace/PhysicsGuidedCTF/model.py) |
| Selected LBNL message implementation | [model_v7.py](workspace/PhysicsRelationCTF/runs/publicdata_exp/optimization_v2/model_v7.py) and preserved imports |
| LBNL equations and control references | [public_physics_v4.py](workspace/PhysicsRelationCTF/runs/publicdata_exp/public_physics_v4.py) and preserved imports |
| TEP equation bank and calibration | [tep_physics.py](workspace/PhysicsRelationCTF/runs/TEP/tep_physics.py) |
| Selected TEP 20→20 adapter | [length_model.py](workspace/PhysicsRelationCTF/runs/TEP/window_search_v1/length_model.py) |
| Component/direct-input controls | [paper_minimal_v1](workspace/PhysicsRelationCTF/runs/paper_minimal_v1) |
| Industrial comparator adapters and records | [physics_ai_comparison_v1](workspace/PhysicsRelationCTF/runs/physics_ai_comparison_v1) |
| Public comparator records | [physics_ai_comparison_public_v1](workspace/PhysicsRelationCTF/runs/physics_ai_comparison_public_v1) |
| Source-to-export mapping | [manifests/source_map.json](manifests/source_map.json) |

`paper_minimal_v1` is a component-experiment folder name, not manuscript content. Some historical orchestrators require original caches, weight locks or private calibration. Use `scripts/` for supported checkpoint-free construction and public Train/Validation runs rather than assuming every archived orchestrator is standalone.

<a id="figures"></a>
## 8. Figures and industrial examples

These are approved finished assets, not manuscript files or reference-paper PDFs.

| Figure | PNG | PDF |
|---|---|---|
| Architecture | [View](figures/architecture/model_architecture.png) | [Download](figures/architecture/model_architecture.pdf) |
| Topology-to-attention mapping | [View](figures/architecture/topology_attention.png) | [Download](figures/architecture/topology_attention.pdf) |
| Equation-message pipeline | [View](figures/architecture/physics_message.png) | [Download](figures/architecture/physics_message.pdf) |
| Industrial topology, English | [View](figures/topology/electrolyzer_topology_en.png) | [Download](figures/topology/electrolyzer_topology_en.pdf) |
| Industrial topology, Chinese | [View](figures/topology/electrolyzer_topology_zh.png) | [Download](figures/topology/electrolyzer_topology_zh.pdf) |
| Filtered forecast example | [View](figures/industrial/lowpass_test_predictions.png) | [Download](figures/industrial/lowpass_test_predictions.pdf) |

<p align="center">
  <img src="figures/industrial/lowpass_test_predictions.png" alt="Selected industrial pressure and hydrogen-cooler outlet-temperature predictions from a separate filtered-data experiment" width="100%">
</p>

*Separate filtered-data illustration: instrument-air header pressure and hydrogen-cooler outlet temperature. Train/Validation/Test inputs and targets were causally filtered with a second-order Butterworth low-pass filter at 0.05 Hz. Curves stitch successive forecast windows, not one 4847-step open-loop forecast, and do not generate the raw-data three-seed table above.*

See [figure notes](figures/README.md) and [figure provenance](manifests/figure_sources.json). Diagrams use simplified layouts; graph snapshots and source define the exact computation.

<a id="limits"></a>
## 9. Scope and limitations

- **Supported:** partial knowledge can enter attention as guidance and equation messages; the three selected full-data tasks show small mean gains over the same CTF backbone after system-specific adaptation.
- **Not established:** universal superiority, zero-shot plant transfer, statistical significance from three seeds, strict conservation, causal discovery, or an advantage that must grow with data scarcity.
- **Interpretability:** relationship sources, equation calculations and insertion locations are inspectable. Attention weights and fitted observation gains are not identified causal effects or physical constants.
- **Uncertain relationships:** public instances include documented validity and historical-consistency checks, not a guarantee that every incorrect prior is detected and suppressed.
- **Development exposure:** seed 42 and fixed Test sets were observed during development. Additional seeds do not turn previously observed tests into blind evaluations.
- **Comparison:** applicable physics methods share available physical information, but architectures are not parameter-count or runtime matched. Adaptations and pilot-selection limits remain disclosed.
- **Validation:** content/hash checks and synthetic forwards were completed; fresh bulk acquisition and full retraining were not rerun during packaging. Cross-hardware bit-identical accuracy is not promised.
- **Scarcity:** earlier few-shot and SDWPF bias-only studies did not establish a monotonic scarcity advantage; those outcomes are not presented as successful scarcity validation here.

Details: [VALIDATION.md](VALIDATION.md), [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [results/README.md](results/README.md).

<a id="citation"></a>
## 10. Attribution, licence and citation

PRCTF builds on **ChannelTokenFormer**, not an original replacement backbone. Please acknowledge upstream methods and datasets when using their components:

- [ChannelTokenFormer](https://github.com/jinkwan1115/ChannelTokenFormer): CTF backbone and masked attention.
- [MTGNN](https://github.com/nnzhan/MTGNN), [PatchTST](https://github.com/yuqinie98/PatchTST), [PhyGNN](https://github.com/DiYi1999/PhyGNN): comparison implementations, subject to their terms; see [adaptation disclosures](workspace/PhysicsRelationCTF/runs/physics_ai_comparison_v1/UPSTREAM.md).
- [LBNL FDD collection](https://doi.org/10.25984/1881324): cite the collection and specific simulated-chiller subset.
- [Rieth et al., TEP simulation data, Harvard Dataverse V1](https://doi.org/10.7910/DVN/6C3JR1): public process data.

No original-code open-source licence has yet been selected; public visibility is not a blanket reuse grant. Third-party source/data retain their terms. See [LICENSE_STATUS.md](LICENSE_STATUS.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

For now, cite this repository with a **fixed commit** following [CITATION.md](CITATION.md). No acceptance, article DOI or archived software DOI is asserted. Old code is preserved on [legacy-v1](https://github.com/field-mx/PhysicsRelationCTF/tree/legacy-v1); `main` uses the publication-oriented layout.
