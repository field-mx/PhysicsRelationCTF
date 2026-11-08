# Finished figure copies

Only author-approved finished figures are exported here. The original paper/
directory, manuscripts, references, reviewer material, draft images, QA files,
and industrial source CSVs are not included.

- `architecture/`: overall model, topology-to-attention mapping, equation-message
  construction. PNG and PDF are byte-identical copies of their selected source.
- `topology/`: Chinese/English directed industrial topology and its attention
  mappings. They encode prior relationships, not inferred causal certainty.
- `industrial/raw_signal_overview.png`: the specifically authorized 原始信号.PNG.
  It is a qualitative overview; units and labels are preserved rather than
  silently corrected or reconstructed from unavailable numerical data.
- `industrial/lowpass_test_predictions.*`: the author-selected panel C, showing
  instrument-air header pressure (PT010003). Frozen seed-42 low-pass checkpoints
  use 96-point histories and 96-step forecasts, updated every 12 seconds, with
  equal-weight combination of overlapping forecasts. All 4847 test points are
  displayed; both models' metrics use the same 4751 points after the shaded
  96-second warm-up. MSE reduction is 2.12% for this channel/policy, not the
  raw-data three-seed benchmark. Forecast leads are mixed from 1 to 96 seconds.
  This replaces the previous two-channel block-forecast illustration; its old
  version remains in Git history and the original local experiment is unchanged.

Exact file provenance and hashes are in `manifests/figure_sources.json`.
These assets should be matched to the final typeset manuscript before reuse;
the author-supplied latest manuscript's renamed figure files were not present
in its folder, so filename similarity alone is not claimed as typeset identity.
