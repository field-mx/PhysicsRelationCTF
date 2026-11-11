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
- `industrial/lowpass_test_predictions.*`: two author-approved channel/policy
  illustrations from the separate seed42 causal-low-pass experiment. Panel (a)
  shows hydrogen outlet flow FT010304 over elapsed0–960 s, directly stitching
  complete96-second forecasts at stride96. Panel (b) is exactly the weighted24
  curves of panel C in the matched-minimum-lead pressure comparison: PT010003,
  stride12, leads24–96 s, each model's inverse validation-MSE weights. It is
  not the former equal-weight mean12 pressure panel. All4847 pressure
  observations are displayed; the first96 s are shaded fusion warm-up.
  Both full-region MSE comparisons use the same4751 targets (elapsed96–4846 s):
  flow23.78% lower and pressure0.62% lower for PRCTF. Flow's score is not just
  the plotted960-second interval. No curve shifts or post-fusion smoothing.
- `industrial/flow_block96_detail_and_full.pdf`: the flow policy's selected
  first960 seconds and full4847-point test timeline, providing context for
  the showcase. These figures are distinct from the raw-data three-seed
  paper benchmark and the released reference checkpoints. Old illustrations
  remain in Git history; source experiments are unchanged.
- `industrial/all_56_channels_truth_PRCTF_full_test.pdf`: the approved56-page
  attachment, one channel per page, all4847 filtered test observations and
  unchanged PRCTF curves. The separate seed42 policy updates every24 seconds
  and fuses lead1–96-second forecasts with per-channel shared monotone inverse
  validation-error weights. CTF is omitted only from the drawing, not replaced
  or rerun; the original plot limits and full timeline are retained. This is
  neither the direct96-second flow policy nor the lead24 matched-pressure
  policy of the two-channel showcase.

Exact file provenance and hashes are in `manifests/figure_sources.json`.
These assets should be matched to the final typeset manuscript before reuse;
the author-supplied latest manuscript's renamed figure files were not present
in its folder, so filename similarity alone is not claimed as typeset identity.
