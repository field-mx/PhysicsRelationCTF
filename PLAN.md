# Physics-Relation-Guided CTF: execution plan

User spec: attachment 8b898bf0-b758-4b65-b40b-0ee2b8e1d713/pasted-text.txt.
Only SensorData.xlsx, 96 input / 96 output, 100/50/30% training prefixes,
unchanged validation/test, true Channel Token readout. No new GNN, physics
loss, TEP, noise or missing-data experiment. Old projects remain intact.

1. Verify units/mappings and available physical quantities from Excel header.
2. Define typed direct relations with explicit history-only lag support.
3. Compute raw-unit equation features; fit normalization only on each prefix.
4. Fuse typed lag-aligned physics messages inside CTF attention value sum;
   keep native head, mask, patching, tokens, attention output projection/FFN.
5. Verify exact zero-branch equivalence, no future access, physical feature
   units, relation masks, nonzero forward influence and forecast gradients.
6. Train matched CTF/Physics at seed42; diagnose and refine representation,
   scale/lag/residual on TRAIN/VALIDATION only. Do not consult new test results
   until the candidate and protocol are frozen.
7. Confirm candidate with seeds43/44 if validation supports advancing; then
   evaluate locked models once on fixed test windows and report all outcomes.

Acceptance target (not a promised outcome): both 50% and30% lower MSE than CTF;
100% no material degradation (predeclare 1% relative tolerance). Positive
validation differences are screening evidence only. Test results may fail.
If they fail, no test-guided retuning; continue train/validation diagnosis and
label that the old holdout is already exposed in subsequent exploratory work.

Scientific limits: no cell count or gas standard-state metadata. Faraday
features are explicitly per-cell mol/s (eta=1 theoretical base); do not claim
stack gas flow. FT010401 header says m3 without time denominator, so q*deltaT
is only a flow-tag heat transport proxy, never watts. No gas inventory volume,
heat capacity, membrane state or valve commands invented. Thermal lag support
within96s cannot identify thermal constants longer than the history.

Observed training correlations: I/avg=.764294, I/Ustack=.514600,
I/H2out=.642970; pressure/temperature same-side pairs >.9996. User's .95/.91
numbers not confirmed for this train interval; no full/test-data selection.

Prior explanation correction: full-matrix sparse-bias SD is diluted by zeros.
It is not a valid estimate of relative strength on active attention edges.
This version records active-edge QK/bias and actual propagated message norms.

Sources: NIST CODATA2022 F=96485.3321233100184 C/mol;
https://www.physics.nist.gov/cuu/pdf/wall_2022.pdf
https://arxiv.org/html/2202.13422 (thermal balance and transport)
https://arxiv.org/html/2311.09882 (electrolyzer DAE)
https://arxiv.org/html/2505.16935 (pressure inventory)

Status: 8 regression checks PASS on CPU; full256-dimensional GPU zero-branch
max difference0.0, finite deterministic backward. Parameters CTF1,650,528;
Physics1,712,321 (+3.74%).73directed edges. Seed42 paired validation queue
50/30/100% launched on two4090 GPUs as independent user-systemd services.

Ruling: use a separate non-git project directory and keep this ledger; no
commit/push or edits to existing experiments. Prior work remains intact.
Ruling: both models use16maximum epochs,1e-4initial LR (previous baseline
validation choice),6constant epochs then cosine, patience5 after epoch8.
This gives new physics parameters time to learn without preferentially
extending only the experimental model. Costs more compute than old10epochs.
Ruling: virtual physics quantities are feature tokens in the side branch,
not extra native CTF tokens; the native mask/head remain exactly unchanged.
Do not claim explicit virtual-node multihop updates: stacked native channel
attention propagates channel state; formulas compute virtual features first.
Tests: python -m unittest test_relation -v ->8/8. First runner factory test
failed before run_experiment.py existed, then passed after implementation.
Equation/features and initial forward smoke checks were written after the
first model prototype, so those are regression tests, not a TDD claim.

Review: independent read-only reviewer found an Important sign defect in lag
matching: outflow/inventory inverse relations were forced to match positive
shapes. Physics v1 pilot stopped after50% completed and partial30%; CTF queue
continues unchanged (it does not use relation_model forward). v1 is validation
diagnostic only, never a final candidate. v2 uses train-only calibration sign
for lag matching. Synthetic inverse-response test failed before helper existed.
Physical balance sign and empirical observation-map sign are NOT equivalent:
closed-loop covariance cannot identify isolated balance coefficients.
Review minors: unselected valve-dp feature documented, not claimed active;
test lock will additionally validate relevant implementation/data hashes.

Seed42 v2 validation:50% .09798944 vs CTF .09902242 (+1.043% relative);
30% .09806601 vs .09813904 (+0.0744%). This does not establish robustness or
increasing low-data advantage.50% validation gains are uneven (32/56channels,
28/50windows improved), with much net gain from LGT010302.
Validation-only refinement: compare ONE centered equation-variation candidate
against anchored v2 on both50/30%. Motivation: target-last-level injection
could dominate mechanistic variations and produce last-value-like gains.
All other hyperparameters stay fixed; same seed42 CTF baseline is reused.
Select one shared representation using mean log validation MSE ratio over
50/30%, requiring both below matched CTF, then check100% tolerance. If the
candidate fails this rule keep v2. No test scores consulted in this decision.

Validation diagnostics at seed42 selected checkpoints: unclamped physical
patch fraction exceeding abs8 is0.78%(50%) /1.44%(30%); lag KL divergence
from uniform support is only0.00363/0.00369nats. Thus the prior's support is
active, but accurate sample-specific transport delay identification is NOT
established.14/73empirical gains at30% are negative; those signs describe
observed response maps, not isolated physical balance terms.
Three-seed confirmation is scheduled because GPUs are available and the
30% seed42 advantage is too small to trust. Baseline seeds43/44 queued after
the single centered-representation validation candidate onGPU0.
Centered candidate30% validation reached .097484989 vs anchored .098066010,
so queue its100% check onGPU1 when v2 finishes, in parallel with its50% run.
This is the same single representation candidate, not a new search dimension.

2026-09-22: resumed from completed background jobs; all9CTF and9centered
train/validation jobs finished. No test metrics exist. Three-seed validation
means:100% CTF .093758582 vs centered .093013494 (+.7947%);
50% .096885382 vs .096894905 (-.00983%);30% .097817538 vs .097531082 (+.29285%).
The low-data objective is NOT yet satisfied robustly; no test evaluation yet.
50% degradation concentrates in PT010003(+.0118383 channelMSE), PT010002
(+.0054468), PT_ADD(+.0035787), none of which has any physical graph edge.
Centered messages are .134/.171 timesV norm at50/30%, residuals132/168times
smaller than physics. This does not suggest global residual dominance.

Next minimal hypothesis: the extra physics gradient into the shared native
patch projection interferes with periodic self-channel encoding. Isolate
only that gradient; retain current weights as a read-only projection in the
physics branch. Native CTF path still trains, value projection still shared.
Prediction: no-edge periodic-pressure regressions shrink, physical-level
benefits persist. Forward values and parameter count MUST stay identical.
Pilot seed43 was chosen to diagnose the observed50% failure, not to select
a favorable seed for reporting. Run50/30% only, reuse completed controls.
Screen: both validation MSEs below pairedCTF, inspect periodic-pressure change;
if supported, confirm all42/43/44 and100%. Otherwise reject/refine hypothesis.
No test-guided changes, no new graph edges or decoder modifications.

Known-physics audit: average of PDT010305/PDT010306 agrees with the average
level target in training (RMSE .094/.099mm at50/30%). In contrast, hydrogen
minusoxygen measured levels do NOT reproduce PIC010306_PV (even sign differs
in training correlations). Do not hard-code that apparent level-difference
identity without accounting for unknown instrument/control definitions.

Second independent minimal hypothesis (not combined with isolation): formula
calibration equals correlation*target_std/feature_std. At30%, current→H2flow
has factor .4345; current→O2inventory -.1756; pressure-difference→impurity
.1845. Therefore deterministic features are attenuated by sample correlation.
Change only magnitude: gain=sign(old_gain)*target_std/feature_std for the16
formula-feature edges. Keep observed orientation so lag matching is unchanged;
all57other edges unchanged. Still NOT an identified physical balance gain.
Prediction: formula-message norms no longer scale down with weak sample
correlation; low-data validation improves without new periodic-pressure harm.
Pilot43 on50/30%, same controls/hyperparameters. No new train data or test use.
Compare each hypothesis independently with centered; do not assume combining
individually unverified changes is beneficial. Retain failed attempts.

Isolation pilot completed:43/50% .096525312 vs centered .096516817 vs CTF
.096052120;43/30% .098032137 vs centered .098107246 vs CTF .098348198.
No material periodic-pressure repair at50%; reject isolation as the main fix.

Conditional next hypothesis if formula normalization fails50%: early-layer
physics propagates via subsequent unconstrained native attention into channels
outside the physical receiving set. Same-checkpoint validation intervention
finds max forward change .10363 on29no-incoming channels; PT010002/3 RMS
changes .00738/.00690. Final-layer-only injection yields exactly0 change on
those channels with weights held fixed. This proves forward spread, NOT that
all retraining degradation is caused by it.
Minimal candidate: freeze first-layer beta/gamma/rho to0; keep final layer
learnable at original initial strengths. Native2-layerCTF remains unchanged.
Tradeoff: no early-layer physics-specific multihop propagation. Same scales,
graph, features and residual as centered, no combined isolation/eqnorm change.
Screen43/50% and30%, then confirm allseeds only if the shared candidate is
promising. Include periodic-pressure recovery and total MSE in evaluation.

Eq-normalization pilot completed:43/50% .096652737 (worse than centered
.096516817 and CTF .096052120);43/30% .098071503 (small improvement over
centered .098107246). Reject it as a solution to50% degradation. Stronger
deterministic-feature amplitude alone did not repair the failure.
Started conditional late-injection43/50% and30% pilots, separate directories.
Regression suite14/14 passes, including no-edge forward invariance and
round-trip serialization into the common inference implementation.

Late pilot completed:43/50% .096179520 (closer to CTF .096052120 but still
worse);43/30% .098666463 (worse than CTF .098348198 and centered .098107246).
Removing forward spread did not simultaneously improve the forecasting task;
do NOT claim it was the sole cause of regression. Reject the late candidate.

Decision before any test access: retain the common centered model with all3
seeds. The three small independent hypotheses did not meet the paired
50/30% pilot screen. No new100% jobs or extra seeds were spent on rejected
candidates; all9existing CTF training jobs were reused. Freeze source/data
and the18 selected checkpoints, then evaluate the test set once. Report
failure if test50/30% do not improve; do not retune against these test scores.
This constitutes reasonable bounded attempts, not a guarantee of the target.
