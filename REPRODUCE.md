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

For parallel execution, launch one queue per dataset with a distinct status directory.

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

## Complexity

```bash
python revision_scripts/measure_fupsi_complexity.py --help
python revision_scripts/measure_sr_pipeline_complexity.py --help
```

Report parameter count, MACs/FLOPs, batch-1 inference latency, and peak memory on the same device and input shapes. Do not assign complexity values to methods without executable implementations.

## Paper evidence

Every reported mean and standard deviation is generated from seed-level CSV files. Paired tests use matched seeds. With only three seeds, paired t-tests and Wilcoxon tests are reported as descriptive evidence rather than definitive significance claims.
