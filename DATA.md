# Data Protocol

## Public sources

- TaxiBJ: use the public TaxiBJ release associated with the ST-ResNet benchmark.
- BikeNYC: use Citi Bike system trip data for 2024-01-01 through 2024-03-31.
- Chicago Taxi: use the City of Chicago `Taxi Trips (2024-)` public dataset.

The repository does not redistribute raw trip records or third-party benchmark files.

## Split policy

All datasets use chronological splits. The Chicago Taxi experiment uses 70% training, 10% validation, and 20% test data. Missing masks are generated only after the split, so masked test entries cannot influence training or validation-based rule selection.

## Expected model directory

Each processed alias is placed under `fupsi/data/<dataset_alias>/` with `train`, `valid`, and `test` subdirectories. The loaders expect the following NumPy files where applicable:

```text
X.npy          historical coarse-grained flow
Y.npy          future fine-grained flow
X_next.npy     future coarse-grained target
XC_*.npy       recent temporal branch
XP_*.npy       distant/periodic temporal branch
XT_*.npy       trend branch
ext.npy        aligned external factors
date.npy       aligned timestamps
```

Run `revision_scripts/prepare_main_seed_datasets.py --code-root ./fupsi --prefix MainSeed` for the TaxiBJ/BikeNYC MainSeed layout. When an isolated experiment namespace is used, create matching data aliases with the same prefix before launching the queue. Run `revision_scripts/prepare_chicago_taxi_2024.py` followed by `prepare_chicago_mainseed.py` for Chicago Taxi.

## Missing-rate protocol

Synthetic missing rates are 10%, 30%, 50%, and 70%. Each setting uses seeds 2024, 2025, and 2026. The adaptive selection rule is chosen on validation completion MSE and fixed before test evaluation. Completion MSE is computed only on synthetically masked entries and is not a trainable loss in FUPSI.
