# Data Protocol

## Public sources

- TaxiBJ: use the public TaxiBJ release associated with the ST-ResNet benchmark.
  The exact files are `BJ13_M32x32_T30_InOut.h5`,
  `BJ14_M32x32_T30_InOut.h5`, `BJ15_M32x32_T30_InOut.h5`, and
  `BJ16_M32x32_T30_InOut.h5`.
- BikeNYC: use `NYC14_M16x8_T60_NewEnd.h5`, covering 2014-04-01 through
  2014-09-30, from the Citi Bike system-data archive.
- Chicago Taxi: use the City of Chicago `Taxi Trips (2024-)` public dataset.

The repository does not redistribute raw trip records or third-party benchmark files.

## Split policy

TaxiBJ and BikeNYC use the audited `MainSeed-RawCount-v2` chronological
protocol. The first 80% of each series is the train-validation block, the
final 10% of that block is validation, and the final 20% of the full series is
test. This is an effective 72/8/20 split:

| Dataset | Source slots | Train | Validation | Test | Coarse grid | Fine grid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| TaxiBJ P1 | 4888 | 3519 | 391 | 978 | 8x8 | 32x32 |
| TaxiBJ P2 | 4780 | 3442 | 382 | 956 | 8x8 | 32x32 |
| TaxiBJ P3 | 5596 | 4029 | 447 | 1120 | 8x8 | 32x32 |
| TaxiBJ P4 | 7220 | 5199 | 577 | 1444 | 8x8 | 32x32 |
| BikeNYC | 4392 | 3162 | 351 | 879 | 8x4 | 16x8 |

The exact zero-based slices are `[0, train)`, `[train, train + validation)`,
and `[train + validation, total)`. Chicago Taxi remains a separate
third-city generalization experiment and uses a chronological 70/10/20 split.
Missing masks are generated only after the split, so masked test entries
cannot influence training or validation-based rule selection.

## Aggregation and normalization

TaxiBJ coarse maps are produced by non-overlapping 4x4 sum aggregation of the
32x32 fine maps. BikeNYC coarse maps are produced by non-overlapping 2x2 sum
aggregation of the 16x8 fine maps. Processed NumPy arrays store raw counts.
During training, coarse and fine values are divided by fixed constants 1500
and 100. Predictions are converted back to raw-count units before RMSE and MAE
are computed. External factors are disabled in the controlled MainSeed runs.

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

Synthetic missing rates are 0%, 10%, 30%, 50%, and 70%. Each setting uses
seeds 2024, 2025, and 2026. The adaptive selection rule is chosen on
validation completion MSE and fixed before test evaluation. Completion MSE is
computed only on synthetically masked entries and is not a trainable loss in
FUPSI. The round-two evaluation additionally passes the completed inputs
through the entire prediction and super-resolution pipeline and reports final
fine-grained RMSE and MAE.
