# Reproduction Commands

## Default residual-enabled FUPSI

Run all five MainSeed datasets with three seeds:

```bash
python revision_scripts/prepare_main_seed_datasets.py \
  --code-root ./fupsi \
  --prefix MainSeed

python revision_scripts/run_main_seed_task_queue.py \
  --code-root ./fupsi \
  --datasets MainSeed_TaxiBJ_P1,MainSeed_TaxiBJ_P2,MainSeed_TaxiBJ_P3,MainSeed_TaxiBJ_P4,MainSeed_BikeNYC \
  --seeds 2024,2025,2026 \
  --stages pretrain,train,test \
  --epochs 300 \
  --batch-size 32 \
  --train-script train.py \
  --test-script test.py \
  --lambda-adv 0
```

`fupsi/train.py` defaults to `--lambda_adv 0`, which disables discriminator updates and reproduces the supervised FUPSI variant used for the primary regenerated results.
The queue supplies the dataset-specific spatial arguments explicitly: TaxiBJ
uses coarse `8x8`, fine `32x32`, and factor 4; BikeNYC uses coarse `8x4`,
fine `16x8`, and factor 2. Both `train.py` and `test.py` reject a formal
dataset alias when these spatial arguments are inconsistent.

For parallel execution, launch one queue per dataset with a distinct status directory.

Verify that alternate experiment prefixes resolve to byte-identical processed
arrays before comparing methods:

```bash
python revision_scripts/round2/audit_protocol_data_equivalence.py \
  --code-root ./fupsi \
  --reference-prefix MainSeed \
  --candidate-prefix ResidualMainE300P5 \
  --output-dir ./results/round2/protocol_audit
```

After all 15 tests finish, collect and hash the formal metric files:

```bash
python revision_scripts/round2/collect_fupsi_seed_metrics.py \
  --code-root ./fupsi \
  --namespace MainSeed \
  --output ./results/round2/fupsi_seed_metrics.csv
```

## GAN stability

To reproduce the optional GAN variant in an isolated namespace, first rerun `prepare_main_seed_datasets.py --prefix GANStableMainE300`, then rerun the queue with `--namespace GANStableMainE300 --lambda-adv 0.01`. The queue forwards the latter option to the training entry point as `--lambda_adv 0.01`. The corrected implementation then applies BCE adversarial training with discriminator learning rate `5e-6`, one discriminator update per 20 generator steps, and real-label target 0.9. These settings were fixed using validation behavior before formal testing. The supervised and GAN variants must use the same residual architecture, split, pretraining checkpoint, epoch budget, and seeds.

After collecting the generated `test_metrics.csv` and `training_history.csv` files in the layout expected by `analyze_gan_stability.py`, run:

```bash
python revision_scripts/analyze_gan_stability.py
```

The script creates seed-level CSVs, mean and standard deviation, paired tests, a LaTeX table, and the loss-curve figure.

## Reproducible baselines

Clone the authors' official PLGF repository separately when auditing the recent-baseline adapter; third-party source code is not redistributed in this package.

```bash
python revision_scripts/run_hamean_baseline.py --help
python revision_scripts/run_sr_baseline_adapter.py --help
python revision_scripts/run_recent_sr_baseline.py --help
python revision_scripts/generate_baseline_fairness_audit.py
```

UrbanFM and FODE adapters receive the same predicted coarse maps used by FUPSI. Alignment must be checked before comparing fine-grained outputs.

## Round-two end-to-end sparse-input evaluation

After the three-seed supervised FUPSI checkpoints have been generated, run:

```bash
python revision_scripts/round2/evaluate_sparse_pipeline.py \
  --code-root ./fupsi \
  --data-prefix MainSeed \
  --model-prefix MainSeed \
  --datasets TaxiBJ_P1,TaxiBJ_P2,TaxiBJ_P3,TaxiBJ_P4,BikeNYC \
  --seeds 2024,2025,2026 \
  --rates 0,0.1,0.3,0.5,0.7 \
  --methods adaptive,no_completion \
  --output-dir ./results/round2/sparse_pipeline
```

The expected seed-level artifact has 150 rows: five datasets, five missing
rates, three seeds, and two completion conditions. It reports completion MSE
as a diagnostic and final fine-grained RMSE/MAE from the full pipeline.
Completion is causal: temporal operators use only current or earlier
observations, while spatial KNN/SVD operators are confined to the current
frame. The output records `model_seed`, `mask_seed`, `fine_RMSE`, and
`fine_MAE` explicitly.

Verify the no-future-leakage invariant independently:

```bash
python revision_scripts/round2/test_sparse_completion_causality.py
```

After evaluation, audit the row count and generate the paper table and
descriptive paired statistics:

```bash
python revision_scripts/round2/summarize_sparse_pipeline.py \
  --input ./results/round2/sparse_pipeline/sparse_pipeline_seed_metrics.csv \
  --output-dir ./results/round2/sparse_pipeline/analysis
```

## HRSTT reimplementation

No verified official HRSTT implementation was identified. The included
implementation is therefore always reported as `HRSTT (reimplementation)`.
It follows the same processed arrays, temporal inputs, split, validation
selection, raw-count inverse scaling, and metrics:

```bash
PYTHON_BIN=python bash revision_scripts/round2/run_hrstt_parallel.sh \
  ./fupsi ./results/round2/hrstt

python revision_scripts/round2/summarize_hrstt.py \
  --input-root ./results/round2/hrstt \
  --output-dir ./results/round2/hrstt/summary
```

The expected output contains 15 `test_metrics.csv` files.

## SR-only baseline fairness rerun

Download the referenced CUFAR implementation into
`external_baselines/CUFAR`, then run:

```bash
PYTHON_BIN=python bash revision_scripts/round2/run_sr_baselines_parallel.sh \
  ./fupsi ./results/round2/sr_baselines

python revision_scripts/round2/summarize_sr_baselines.py \
  --input-root ./results/round2/sr_baselines \
  --fupsi ./results/round2/fupsi_seed_metrics.csv \
  --output-dir ./results/round2/sr_baselines/summary
```

This stage produces 30 seed-level runs: UrbanFM and FODE on all five datasets,
each with three seeds. The baselines are trained on the same
aligned coarse/fine pairs and evaluated using the fresh seed-matched FUPSI
coarse predictions. The summary must pass 90 exact SHA-256 comparisons for the
shared predicted coarse map, true coarse target, and true fine target.
It also verifies 90 shared coarse-metric values against the seed-matched FUPSI
rows within CSV rounding tolerance.
BikeNYC uses the same 1500/100 coarse/fine divisors as the unified FUPSI
protocol.

## Unified statistics

After FUPSI, HRSTT, SR-only, and deterministic HA-Mean results are present,
run:

```bash
python revision_scripts/round2/build_unified_main_statistics.py
```

The script uses only round-two seed-level files by default. It writes the
paper-ready main table, mean-plus-standard-deviation summaries, absolute and
relative gains against the best available baseline, and paired t-test,
Wilcoxon, Cohen's `d_z`, and seed-direction results. An older result file is
read only when explicitly supplied with `--existing`.

## Prediction-before-SR order study

The inverse-order control first reconstructs historical fine maps and then
trains the same temporal prediction family at fine resolution:

```bash
PYTHON_BIN=python bash revision_scripts/round2/run_inverse_order_p4.sh \
  ./fupsi ./results/round2/inverse_order
```

The expected output contains three TaxiBJ P4 seed-level metric files. Seed
2024 also stores predictions for the corrected visualization. Generate the
auditable order-comparison figure after the inverse-order runs:

```bash
python revision_scripts/round2/summarize_inverse_order.py \
  --fupsi ./results/round2/fupsi_seed_metrics.csv \
  --inverse-root ./results/round2/inverse_order \
  --output-dir ./results/round2/inverse_order/summary

python revision_scripts/round2/generate_p4_order_visualization.py \
  --code-root ./fupsi \
  --namespace MainSeed \
  --inverse-root ./results/round2/inverse_order \
  --seed 2024 \
  --output-dir ./results/round2/visualization
```

The script verifies that the two methods use the same fine-grained targets and
selects the displayed sample deterministically: its corrected FUPSI
per-sample RMSE is nearest the median corrected FUPSI per-sample RMSE.

## Complexity

```bash
python revision_scripts/measure_fupsi_complexity.py --help
python revision_scripts/measure_sr_pipeline_complexity.py --help
```

Report parameter count, MACs/FLOPs, batch-1 inference latency, and peak memory on the same device and input shapes. Do not assign complexity values to methods without executable implementations.

## Paper evidence

Every reported mean and standard deviation is generated from seed-level CSV files. Paired tests use matched seeds. With only three seeds, paired t-tests and Wilcoxon tests are reported as descriptive evidence rather than definitive significance claims.
