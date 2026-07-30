# Refreshed GAN Stability Audit

| Dataset | Metric | noGAN | GAN | Lower |
|---|---|---:|---:|---|
| TaxiBJ P1 | RMSE | 79.7437 +/- 0.0187 | 79.9082 +/- 0.0552 | noGAN |
| TaxiBJ P1 | MAE | 42.1333 +/- 0.0638 | 42.6554 +/- 0.0357 | noGAN |
| TaxiBJ P2 | RMSE | 100.4167 +/- 0.0163 | 100.6009 +/- 0.0570 | noGAN |
| TaxiBJ P2 | MAE | 54.5649 +/- 0.0143 | 55.0401 +/- 0.1324 | noGAN |
| TaxiBJ P3 | RMSE | 91.2122 +/- 0.0216 | 91.4175 +/- 0.0360 | noGAN |
| TaxiBJ P3 | MAE | 50.2293 +/- 0.0351 | 50.7677 +/- 0.1036 | noGAN |
| TaxiBJ P4 | RMSE | 68.7643 +/- 0.0590 | 68.9480 +/- 0.0548 | noGAN |
| TaxiBJ P4 | MAE | 36.2842 +/- 0.0551 | 36.7210 +/- 0.0436 | noGAN |
| BikeNYC | RMSE | 5.3890 +/- 0.0223 | 6.0537 +/- 0.0542 | noGAN |
| BikeNYC | MAE | 2.6189 +/- 0.0399 | 3.0343 +/- 0.0264 | noGAN |

The noGAN rows come from the fresh round-two rerun. The GAN rows come from the corrected residual/BCE-GAN experiment. Statistical tests are descriptive because n=3 has low inferential power.
