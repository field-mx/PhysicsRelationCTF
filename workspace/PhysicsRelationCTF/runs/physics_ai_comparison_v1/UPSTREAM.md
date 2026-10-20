# Upstream implementations and adaptation disclosures

Fetched 2026-10-14 into this new experiment directory only. Upstream repositories
are left unchanged; imports/adapters live in comparison_models.py.

| Method | Official repository | Frozen commit |
|---|---|---|
| PhyGNN | https://github.com/DiYi1999/PhyGNN | 75ab808f4a12da9eab650f17bb38f91d9a571154 |
| MTGNN | https://github.com/nnzhan/MTGNN | f811746fa7022ebf336f9ecd2434af5f365ecbf6 |
| PatchTST | https://github.com/yuqinie98/PatchTST | 204c21efe0b39603ad6e2ca640ef5896646ab1a9 |

## PhyGNN adapted

Reference: Di et al., *PhyGNN: Physics guided graph neural network for complex
industrial power system modeling*, Mechanical Systems and Signal Processing
240 (2025), 113380, DOI 10.1016/j.ymssp.2025.113380.

Preserve the official CMTS_GCN/G3CN spatial aggregation without editing it.
Replace the spacecraft simulator by the shared historical equation feature bank.
Use measured nodes plus equation-feature nodes, known direct sensor edges and
explicit equation-input/target routes; self links retain each sensor's information.
No correlation graph is called a physical graph. Unknown scale gains reuse the
train-calibrated observation mapping, not assumed device constants. Add shared
temporal heads for unknown future evolution and learned correction, retaining a
physical-reference-plus-correction decomposition. This is an adaptation to partial
knowledge and forecasting, not exact reconstruction of the SPS benchmark.
PhyGNN's repository carries CC BY-NC 4.0; retain its license and attribution.

## MTGNN

Reference: Wu et al., *Connecting the Dots: Multivariate Time Series Forecasting
with Graph Neural Networks*, KDD 2020, arXiv:2005.11650.

Use official gtnet/mixprop/graph_constructor/dilated_inception. Add history-only
instance normalization and inverse normalization, matching the CTF task's
local scaling convention. Receptive field 43 fits inside 96 real observations.
Restore the requested GPU device for unregistered upstream index tensors.
Fix evaluation tie-breaking RNG, because graph_constructor otherwise adds random
noise even during eval; upstream training tie-breaking remains unchanged.
This is data-driven, not physics-guided. MIT license retained.

## PatchTST

Reference: Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting
with Transformers*, ICLR 2023, arXiv:2211.14730.

Use official supervised PatchTST_backbone with RevIN, residual attention, shared
channel-independent encoder/head. d_model=256, d_ff=1024, layers=2, heads=8 and
dropout=.1 follow the paired CTF capacity settings. Patch24/stride12 gives seven
complete real patches; padding_patch=None, no replicated fake observations.
Apache 2.0 license retained.

## PG-Reg-CTF

Inspired by Daw et al., *Physics-guided Neural Networks (PGNN): An Application in
Lake Temperature Modeling*, arXiv:1710.11431: physical input plus soft consistency
supervision. Reuse the existing EquationInputCTF implementation unchanged, adding
history/prediction-only partial-relationship regularization in the training objective.
Nominal finite lags are chosen from existing physical relation priors, not Test.
Train-calibrated incremental tolerances avoid forcing uncertain proxies to be exact.
An observed voltage ordering constraint avg<=max is also included. No complete
mass/energy balance or PDE residual is invented. Not a faithful standard PINN
replication, and empirical observation consistency is not proof of conservation.

## Interpretation limits

Physical topology/feature budget is shared by the applicable physical methods;
different architectures do not have equal parameter counts or training speed.
Forecast accuracy alone does not establish causal interpretability. All pilot
results are retained; extra seeds are outcome-selected and must be disclosed.
