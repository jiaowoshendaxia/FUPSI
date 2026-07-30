# Same-Hardware Complexity Audit

- Device: NVIDIA GeForce RTX 4090.
- MainSeed shapes: TaxiBJ 8x8 to 32x32; BikeNYC 8x4 to 16x8.
- Inference: batch size 1, 20 warmup runs, 100 measured runs.
- Training proxy: batch size 8, 3 warmup steps, 10 measured forward/backward/Adam steps.
- UrbanFM/FODE cost includes both flow channels through the released single-channel model interface.
- FUPSI is not the smallest-parameter model: FODE has fewer parameters. FUPSI has lower MACs and lower batch-1 inference latency than both reproduced SR baselines on TaxiBJ.
- The standardized training step excludes the FUPSI discriminator; do not present it as total adversarial-training wall time.
