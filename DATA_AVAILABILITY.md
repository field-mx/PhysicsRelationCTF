# Data availability and public sources

## 1. Industrial alkaline electrolyzer monitoring

The study uses real multi-sensor monitoring data from an alkaline water
electrolysis system, including electrical, thermal, pressure, flow, liquid-level
and gas-composition measurements. Of 61 original channels, 56 nonconstant
channels are evaluated; exclusions are determined from training observations.
The 1 Hz task uses 96 observed points to predict 96 future points.

The numerical industrial observations, filtered observations and numerical
source series for figures are **not distributed**. No promise of public
release, request-based access or unrestricted reuse of those observations is
made. These conditions concern distribution here, not an invented legal or
ethical justification.

The author has now authorized publication of the archived seed-42 PRCTF weights
and the empirical gain/normalization statistics required for their inference.
They are included in [weights/industrial/prctf_seed42.pt](weights/industrial/prctf_seed42.pt),
with provenance and hashes in [weights/manifest.json](weights/manifest.json).
These are fitted model and aggregate calibration artifacts, not raw time-series
observations. The equivalent LBNL and TEP reference bundles are also released.

The author has authorized the finished illustrative images under
`figures/industrial/`, including `raw_signal_overview.png`. They show selected
signals or forecasting examples but do not provide the full machine-readable
dataset. The code and relationship schema permit inspection of the method;
industrial accuracy cannot be independently regenerated without authorized
access to the original records, even though the trained inference model is now
available.

## 2. LBNL simulated chiller plant

- Dataset collection: **LBNL Fault Detection and Diagnostics Datasets**.
- Specific system: **Simulated chiller plant**, EnergyPlus–Modelica co-simulation.
- Official system page: https://faultdetection.lbl.gov/dataset/simulated-chiller-plant/
- Dataset DOI: https://doi.org/10.25984/1881324
- Inventory document: https://fdddata.lbl.gov/data/Simulated_LBNL_FDD_Data_Sets_Chiller_Plant/LBNL_FDD_Data_Sets_Chiller_Plant.pdf
- Archive used by the original acquisition script:
  https://fdddata.lbl.gov/data/Simulated_LBNL_FDD_Data_Sets_Chiller_Plant/LBNL_FDD_Data_Sets_Chiller_Plant.zip

Only the fault-free member `ChillerPlant.csv` is used, not all fault scenarios.
Source provenance and SHA-256 are retained in
`workspace/PhysicsRelationCTF/runs/publicdata_exp/data/provenance.json`.
Sampling is one minute. The fixed experiment has 73 varying inputs, 58 scored
physical targets, and a 96→96 task with 24-point patches. Train/validation/test
contain 367878/78831/78831 points; controls and setpoints may be inputs but are
not all scored prediction targets. The original raw archive is not mirrored.

The official collection metadata identifies CC BY 4.0; access and reuse remain
subject to the source's current terms. The DOI identifies the broader FDD
collection, while the system page and archive specify the actual subset.

Suggested dataset citation: Lawrence Berkeley National Laboratory, *LBNL Fault
Detection and Diagnostics Datasets*, DOI: 10.25984/1881324. Cite the corresponding
inventory and system subset when describing this experiment.

## 3. Tennessee Eastman process (TEP)

- Dataset: **Additional Tennessee Eastman Process Simulation Data for Anomaly
  Detection Evaluation**, Rieth, Amsel, Tran and Cook (2017), Harvard Dataverse, V1.
- Canonical DOI: https://doi.org/10.7910/DVN/6C3JR1
- Dataset page:
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1
- Retrieval mirror used by the original project:
  https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset
- Simulator equation reference:
  https://github.com/camaramm/tennessee-eastman-profBraatz/blob/master/teprob.f

Only `TEP_FaultFree_Training.RData` and `TEP_FaultFree_Testing.RData` are used.
Sampling is three minutes. Train uses training-file runs 1–80, validation uses
training-file runs 81–100, and test uses testing-file runs 401–500. The selected
task is 20→20 with five-point patches, 52 inputs/targets, and no cross-run
windows or invented observations. Fixed origins start at point50; validation
and test use stride20. The data manifest preserves the chosen membership.

The data record specifies its own dataset terms; the simulator code and Kaggle
mirror have separate terms. Those terms are not replaced by this repository's
future code licence. Original data files are linked, not redistributed.

Suggested citation: Rieth, C. A.; Amsel, B. D.; Tran, R.; Cook, M. B. (2017).
*Additional Tennessee Eastman Process Simulation Data for Anomaly Detection
Evaluation*. Harvard Dataverse, V1. https://doi.org/10.7910/DVN/6C3JR1.

## Access verification

Source records were checked on 2026-10-20. The LBNL system page and dataset
metadata were retrievable; some DOI redirects encountered automated-access
limits. Download and authentication behaviour can change. No new bulk download
was performed while preparing this companion; scripts must obey source terms
and clearly report failures rather than substituting another dataset.
