# MainSeed SR pipeline complexity

| Dataset | Method | Pipeline params (M) | Pipeline MACs (G) | Inference (ms) | Infer peak (MB) | Train step (ms) | Train peak (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| TaxiBJ P1 | FUPSI | 1.073 | 0.167 | 6.167 | 143.2 | 62.646 | 209.0 |
| TaxiBJ P1 | UrbanFM | 1.652 | 0.330 | 7.642 | 153.6 | 65.464 | 217.9 |
| TaxiBJ P1 | FODE | 0.689 | 0.217 | 6.719 | 149.8 | 62.006 | 203.3 |
| TaxiBJ P2 | FUPSI | 1.014 | 0.106 | 3.533 | 150.8 | 40.812 | 169.3 |
| TaxiBJ P2 | UrbanFM | 1.592 | 0.269 | 4.745 | 153.0 | 50.403 | 178.3 |
| TaxiBJ P2 | FODE | 0.630 | 0.156 | 3.837 | 149.3 | 39.362 | 161.8 |
| TaxiBJ P3 | FUPSI | 1.023 | 0.110 | 3.545 | 150.8 | 41.568 | 73.9 |
| TaxiBJ P3 | UrbanFM | 1.601 | 0.272 | 4.972 | 153.1 | 50.313 | 84.4 |
| TaxiBJ P3 | FODE | 0.639 | 0.159 | 4.038 | 149.3 | 36.824 | 60.7 |
| TaxiBJ P4 | FUPSI | 1.015 | 0.110 | 3.688 | 150.8 | 41.142 | 74.8 |
| TaxiBJ P4 | UrbanFM | 1.593 | 0.273 | 4.899 | 153.1 | 43.156 | 85.3 |
| TaxiBJ P4 | FODE | 0.631 | 0.160 | 3.961 | 149.3 | 36.580 | 61.6 |
| BikeNYC | FUPSI | 0.875 | 0.035 | 3.555 | 150.3 | 40.544 | 164.5 |
| BikeNYC | UrbanFM | 1.453 | 0.098 | 4.825 | 152.5 | 40.041 | 171.2 |
| BikeNYC | FODE | 0.487 | 0.041 | 3.868 | 148.7 | 33.479 | 159.0 |

Notes:

- All methods use identical MainSeed input shapes and the same RTX 4090.
- Baseline SR MACs/latency include both flow channels through the released single-channel interface.
- Pipeline values include the shared FUPSI coarse predictor followed by the named SR method.
- Training-step measurements cover predictor plus SR forward/backward/Adam update and exclude the FUPSI discriminator.
