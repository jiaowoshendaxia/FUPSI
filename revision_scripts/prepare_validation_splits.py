# -- coding:utf-8 --
"""Create validation split directories for adaptive-rule verification."""

from __future__ import annotations

from pathlib import Path

import numpy as np


DATASETS = ["TaxiBJ_P1", "TaxiBJ_P2", "TaxiBJ_P3", "TaxiBJ_P4", "BikeNYC"]


def main() -> None:
    output_root = Path("data_validation")
    for dataset in DATASETS:
        train_dir = Path("data") / dataset / "train"
        x = np.load(train_dir / "X.npy").astype(np.float32)
        y = np.load(train_dir / "Y.npy").astype(np.float32)
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]
        val_size = max(1, int(n * 0.1))
        x_val = x[-val_size:]
        y_val = y[-val_size:]

        split_dir = output_root / dataset / "test"
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / "X.npy", x_val)
        np.save(split_dir / "Y.npy", y_val)
        print(f"{dataset}: validation {x_val.shape} -> {y_val.shape}")


if __name__ == "__main__":
    main()
