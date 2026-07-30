# End-to-End Sparse-Input Audit

- Seed-level rows: 150/150.
- Protocol: MainSeed-RawCount-v2.
- Methods: adaptive completion and no completion.
- Missing rates: 0%, 10%, 30%, 50%, and 70%.
- Seeds: 2024, 2025, and 2026, matched for model and mask.
- Completion is causal and never uses future test observations.
- Adaptive completion has a lower mean in 40/40 nonzero-rate dataset-metric comparisons.

Paired t-tests, Wilcoxon tests, Cohen's d_z, and seed-direction counts are reported as descriptive evidence because n=3 has low inferential power.
