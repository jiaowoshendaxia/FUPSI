# GAN versus noGAN stability analysis

Both variants use the same residual-enabled architecture, pretraining checkpoints, splits, 300-epoch joint-training budget, and seeds.

| Dataset | Metric | noGAN | GAN | Lower variant |
|---|---|---:|---:|---|
| TaxiBJ P1 | RMSE | 79.7831 +/- 0.0991 | 79.9082 +/- 0.0552 | noGAN |
| TaxiBJ P1 | MAE | 42.1963 +/- 0.1583 | 42.6554 +/- 0.0357 | noGAN |
| TaxiBJ P2 | RMSE | 100.4263 +/- 0.0086 | 100.6009 +/- 0.0570 | noGAN |
| TaxiBJ P2 | MAE | 54.5965 +/- 0.0115 | 55.0401 +/- 0.1324 | noGAN |
| TaxiBJ P3 | RMSE | 91.2193 +/- 0.0077 | 91.4175 +/- 0.0360 | noGAN |
| TaxiBJ P3 | MAE | 50.2299 +/- 0.0293 | 50.7677 +/- 0.1036 | noGAN |
| TaxiBJ P4 | RMSE | 68.7604 +/- 0.0526 | 68.9480 +/- 0.0548 | noGAN |
| TaxiBJ P4 | MAE | 36.2869 +/- 0.0396 | 36.7210 +/- 0.0436 | noGAN |
| BikeNYC | RMSE | 5.3887 +/- 0.0589 | 6.0537 +/- 0.0542 | noGAN |
| BikeNYC | MAE | 2.6406 +/- 0.0098 | 3.0343 +/- 0.0264 | noGAN |

MAPE is excluded from the primary conclusion because of zero and near-zero denominators.
All paired tests use n=3 and are treated as descriptive rather than definitive significance evidence.
