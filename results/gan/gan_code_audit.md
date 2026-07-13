# GAN Training Audit

Date: 2026-07-13

## Finding

The residual-enabled MainSeed training entry point instantiated and optimized a discriminator, but the generator objective contained only the supervised fine-resolution reconstruction term and the coarse-prediction term. The discriminator loss was backpropagated through the generated image and the generator gradients were then cleared before the generator update. Consequently, the discriminator did not affect the saved generator or predictor.

The `ResidualMainE300P5` results are therefore valid residual-enabled supervised results, but they must be classified as **FUPSI-noGAN**, not as an adversarially trained variant. They remain useful as the noGAN side of the requested stability study.

## Correction

The corrected entry point is stored at:

- `revision/remote_code_snapshot/residual_main_seed/main_seed_train_residual.py`

For the formal server run it is uploaded as `main_seed_train_gan_stable2_residual.py`. The correction uses:

- discriminator BCE: `0.5 * [BCE(D(real), 1) + BCE(D(fake.detach()), 0)]`;
- generator adversarial BCE: `BCE(D(fake), 1)`;
- total generator objective: supervised reconstruction and prediction losses plus `lambda_adv * adversarial_loss`;
- `lambda_adv = 0.01`, matching the manuscript hyperparameter table;
- discriminator learning rate `5e-6`, one update per 20 generator steps, and real-label target 0.9, fixed from the TaxiBJ P4 validation pilot before formal testing;
- per-epoch CSV and NumPy histories for generator loss, discriminator loss, adversarial loss, reconstruction loss, scores, validation RMSE, PSNR, SSIM, and prediction loss.

## Validation-fixed stability setting

An initial BCE implementation with discriminator learning rate `1e-4` saturated on TaxiBJ: discriminator BCE approached zero and generator adversarial BCE increased to approximately 12-14. This queue was stopped before any formal test result was produced and is excluded from all evidence.

Two TaxiBJ P4 validation pilots were then run without inspecting formal test metrics:

- `lr_d=1e-5`, update interval 5: at epoch 60, discriminator BCE was 0.2204 and generated-score mean was 0.0839, indicating continued discriminator dominance.
- `lr_d=5e-6`, update interval 20, real-label target 0.9: at epoch 60, discriminator BCE was 0.5867, real/generated scores were 0.5392/0.4153, adversarial BCE was 0.8822, and validation RMSE was 0.8536.

The second setting was fixed for all formal datasets and seeds. The pilot namespaces are retained for audit but are not included in test statistics.

## Formal comparison

- noGAN evidence: `revision/statistics/ResidualMainE300P5/`.
- corrected GAN namespace: `GANStableMainE300`.
- datasets: TaxiBJ P1-P4 and BikeNYC.
- seeds: 2024, 2025, and 2026.
- the validated 300-epoch pretraining checkpoints are reused without modification.
- only 300-epoch joint training and testing are rerun.
- five dataset workers run in parallel through `revision_scripts/run_gan_main_parallel.sh`.

The manuscript and response letter must not describe the formal GAN stability experiment as complete until all 30 GAN stages (15 joint-training and 15 test stages), 15 test metric files, and training histories pass audit.
