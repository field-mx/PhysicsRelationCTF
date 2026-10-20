# Publication-package validation

Date: 2026-10-20. Packaging and publication do not change model mathematics or
source observations and do not retrain a model. No original files are deleted.

## Completed

- Seven content-contract tests on the candidate: required documents, all 63 method/seed rows,
  exact-copy approved industrial overview, no numerical industrial calibration,
  source links, forbidden-file exclusion and per-file manifest integrity.
- Separate-process CPU synthetic forward tests using the exact locally pinned
  upstream CTF source. Upstream source was read locally for validation, not
  bundled in the export.

| Instance | Input shape | Output shape | Enhanced output finite | Physics-off vs CTF maximum absolute difference |
|---|---|---|---|---:|
| Industrial | 2×96×56 | 2×96×56 | Yes | 0.0 |
| LBNL | 2×96×73 | 2×96×73 | Yes | 0.0 |
| TEP | 2×20×52 | 2×20×52 | Yes | 0.0 |

Construction uses initialization parameters, not the trained weights. The
industrial test uses invented synthetic unit statistics labelled as such, not
empirical industrial statistics or added training observations.

The selected TEP graph/scalers JSON were compared locally to the nominal
checkpoint used by the final experiment; the structures were equal. No weight
checkpoint was exported.

Mathematical model source files are preserved byte-for-byte. The exceptions in
the wider source export are explicit I/O/path adaptations and sanitized evidence
paths, recorded in `manifests/source_map.json` and REPRODUCIBILITY.md.

## Not established by these checks

- New full training, test evaluation or improved prediction results.
- Successful fresh bulk download/preparation in an empty environment.
- Bit-identical accuracy across different PyTorch/CUDA or hardware versions.
- Publication readiness of the manuscript, institutional licence approval, or
  permission to redistribute upstream CTF source.
- Safety of old remote Git history. Only this new publication directory is
  screened as the proposed upload; the existing repository history is retained.

The recorded packaging environment is in
`manifests/packaging_environment.json`; it is not automatically the original
experiment environment. Author and licence details are not invented. See
LICENSE_STATUS.md for the unselected original-code licence; third-party terms
remain independent.
