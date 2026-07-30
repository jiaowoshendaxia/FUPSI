# Result Evidence Index

The `results/` directory contains anonymous, machine-readable summaries used
by the revised manuscript. Superseded first-round main, completion-only,
fairness, and complexity outputs are intentionally excluded so that every
principal comparison follows the single `MainSeed-RawCount-v2` protocol.

| Evidence | Directory | Contents |
|---|---|---|
| Capacity analysis | `results/capacity/` | Corrected Small/Medium/Full accuracy and complexity evidence |
| Third-city study | `results/chicago/` | Corrected Chicago Taxi summaries and paired comparisons |
| GAN stability | `results/gan/` | Corrected GAN/noGAN seed metrics, paired tests, histories, and curves |
| Protocol audit | `results/round2/protocol_audit/` | All 45 processed dataset/split/file shape, dtype, and SHA-256 checks |
| Corrected FUPSI and HA | `results/round2/` | Fifteen corrected FUPSI rows, deterministic HA rows, and source hashes |
| End-to-end sparse-input study | `results/round2/sparse_pipeline/` | Exactly 150 causal pipeline evaluations, mean/std summaries, paired tests, and the paper table |
| HRSTT comparison | `results/round2/hrstt/` | Fifteen seed-level results for the documented HRSTT reimplementation |
| Order study | `results/round2/inverse_order/` | Corrected three-seed TaxiBJ P4 order statistics |
| SR fairness audit | `results/round2/sr_baselines/` | Thirty fresh UrbanFM/FODE rows, 90 shared-array hashes, and 90 shared coarse-metric checks |
| Unified statistics | `results/round2/main_statistics/` | Seventy-five seed rows, 50 mean/std rows, 40 paired tests, and paper tables |
| Same-hardware complexity | `results/round2/complexity/` | Fifteen RTX 4090 method-dataset profiles |
| Corrected visualization | `results/round2/visualization/` | P4 order figure and source-hash metadata |

MAPE remains available in seed-level evidence but is not used for primary
conclusions because zero and near-zero targets make percentage errors unstable.
With only three paired seeds, statistical tests are treated as descriptive
evidence.
