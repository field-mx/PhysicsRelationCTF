# Result tables and evidence

- `method_comparison_per_seed.csv`: 63 existing rows, three datasets × seven
  methods × three fixed seeds; includes stronger comparators and losing seeds.
- `method_comparison_summary.csv`: per-method mean, sample standard deviation,
  best MSE/MAE, and archived training time when three records were available.
- `ctf_prctf_paired.csv`: nine paired CTF/PRCTF comparisons, not a seed selection.
- `ablation_and_input_per_seed.csv`: complete selected component/input controls.

These are mechanical exports of existing numerical results, not new training.
The fuller configuration, histories, metrics and selection records reside in
the selected workspace experiment folders. Source-to-export checksums preserve
their identity despite replacing local absolute paths in the exported copy.

MSE/MAE are normalized according to each fixed task. Positive gain means lower
PRCTF error. Lower errors from different datasets are not compared directly.
Best MSE and Best MAE need not refer to the same model checkpoint/seed. Empty
time fields indicate missing/incompatible archived measurements, not zero cost.

The LBNL baseline with the lowest reported MSE is MTGNN, not PRCTF. On TEP,
EquationInput-CTF and PG-Reg-CTF outperform PRCTF in these configurations.
The original industrial expansion followed a seed42 pilot selection and should
be described as exploratory; all exported numeric outcomes are nevertheless
retained. See REPRODUCIBILITY.md and the retained UPSTREAM.md for adaptation and
selection disclosures; internal writing/progress reports are not distributed.
