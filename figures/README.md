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
- `industrial/lowpass_test_predictions.*`: the selected pressure and hydrogen
  cooler outlet-temperature example; separate single-seed low-pass experiment.

Exact file provenance and hashes are in `manifests/figure_sources.json`.
These assets should be matched to the final typeset manuscript before reuse;
the author-supplied latest manuscript's renamed figure files were not present
in its folder, so filename similarity alone is not claimed as typeset identity.
