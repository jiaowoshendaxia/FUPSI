# FUPSI Reproducibility Package

This repository contains the implementation and experiment utilities used in the revised manuscript **"FUPSI: A Unified Framework for Urban Flow Prediction and Super-Resolution with Sparse Coarse-Grained Inputs."** It covers residual-enabled FUPSI training, BCE adversarial training, evaluation, dataset preparation, missing-rate evaluation, reproducible baseline adapters, complexity profiling, and paper-table generation.

The repository is provided for peer-review reproducibility. Raw third-party datasets and trained checkpoints are intentionally excluded; public data sources and preprocessing instructions are documented in `DATA.md`.

## Repository layout

```text
fupsi/                model, data loaders, training, and test entry points
revision_scripts/     experiment queues, baselines, statistics, and table scripts
revision/             generated outputs (created by the scripts)
results/              anonymous CSV evidence used by revised manuscript tables
configs/              environment metadata
```

Model checkpoints and raw datasets are intentionally excluded. See `DATA.md` for expected files and split rules.

## Environment

The round-two formal runs use Python 3.10.8, PyTorch 2.1.2 with CUDA
12.1, and an NVIDIA RTX 4090. Create the environment with:

```bash
conda env create -f environment.yml
conda activate fupsi-revision
```

## Quick verification

From the repository root:

```bash
python -m py_compile fupsi/train.py fupsi/test.py
python revision_scripts/prepare_main_seed_datasets.py \
  --code-root ./fupsi \
  --prefix Smoke
python revision_scripts/run_main_seed_task_queue.py \
  --code-root ./fupsi \
  --datasets MainSeed_TaxiBJ_P4 \
  --seeds 2024 \
  --stages pretrain,train,test \
  --epochs 1 \
  --batch-size 32 \
  --train-script train.py \
  --test-script test.py \
  --namespace Smoke
```

The smoke command uses an isolated namespace and does not overwrite formal outputs.

## Formal protocol

- Seeds: 2024, 2025, and 2026.
- Pretraining: 300 epochs for coarse-grained prediction.
- Joint training: 300 epochs for prediction and fine-grained reconstruction.
- Batch size: 32.
- Default supervised FUPSI: `lambda_adv=0`, so discriminator updates and adversarial gradients are disabled.
- Optional GAN variant: binary cross-entropy discriminator and generator adversarial loss with `lambda_adv=0.01`; the discriminator uses learning rate `5e-6`, one-sided label smoothing, and one update per 20 generator steps.
- Primary metrics: RMSE and MAE. MAPE is auxiliary because zero and near-zero targets make it unstable.
- Main protocol: `MainSeed-RawCount-v2`, with a chronological 72/8/20 split,
  fixed coarse/fine divisors of 1500/100, flow-only inputs, and metrics after
  conversion back to raw-count units.
- TaxiBJ uses 8x8 coarse and 32x32 fine maps; BikeNYC uses 8x4 coarse and
  16x8 fine maps from the 2014 source file.
- Sparse-input evaluation uses causal completion only; no future test
  observation is used to fill a forecasting input.
- `audit_protocol_data_equivalence.py` checks the shapes, dtypes, and SHA-256
  identity of every processed split used by alternate experiment prefixes.
- End-to-end sparse evaluation uses missing rates 0%, 10%, 30%, 50%, and 70%
  and reports final fine-grained RMSE and MAE.
- The formal SR-only comparison retrains UrbanFM and the dynamically adapted
  FODE implementation on all five datasets with three seeds. BikeNYC uses the
  same coarse/fine divisors of 1500/100 as FUPSI.
- Every SR result records SHA-256 hashes for the shared predicted coarse map,
  true coarse target, and true fine target. The final audit requires all 90
  hashes (30 runs times three arrays) to match the corresponding FUPSI files.
- The same audit independently checks 90 reported coarse RMSE/MAE/MAPE values
  against the seed-matched FUPSI values within CSV rounding tolerance.

Detailed commands and output mappings are in `REPRODUCE.md`; `RESULTS.md` indexes the included table evidence.
