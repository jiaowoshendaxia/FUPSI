# Result Evidence Index

The `results/` directory contains anonymous, machine-readable summaries used by the revised manuscript. These files are included for direct table auditing; the accompanying scripts regenerate the same summaries from seed-level outputs.

| Evidence | Directory | Contents |
|---|---|---|
| Regenerated primary results | `results/main/` | Three-seed FUPSI summaries, aligned UrbanFM/FODE comparisons, paired tests, and sanitized seed-level metrics |
| Capacity analysis | `results/capacity/` | Small/Medium/Full accuracy, paired tests, parameter counts, MACs, latency, and memory |
| Efficiency analysis | `results/complexity/` | Same-hardware FUPSI/UrbanFM/FODE pipeline measurements and relative summaries |
| Third-city study | `results/chicago/` | Chicago Taxi mean and standard deviation, paired tests, and sanitized seed-level metrics |
| Missing completion | `results/missing/` | Validation-selected adaptive completion versus the best test baseline |
| Baseline fairness | `results/fairness/` | Unified-task adaptation and validation protocol |
| GAN stability | `results/gan/` | Fifteen GAN and fifteen noGAN seed metrics, paired tests, 300-epoch histories, stability curves, and the implementation audit |

MAPE remains available in seed-level evidence but is not used for primary conclusions because zero and near-zero targets make percentage errors unstable. With only three paired seeds, statistical tests are treated as descriptive evidence.
