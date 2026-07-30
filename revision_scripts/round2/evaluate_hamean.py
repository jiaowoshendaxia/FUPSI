#!/usr/bin/env python3
"""Evaluate deterministic HA-Mean under MainSeed-RawCount-v2."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


DATASETS = {
    "TaxiBJ_P1": ("TaxiBJ P1", 5, 48),
    "TaxiBJ_P2": ("TaxiBJ P2", 1, 48),
    "TaxiBJ_P3": ("TaxiBJ P3", 2, 48),
    "TaxiBJ_P4": ("TaxiBJ P4", 3, 48),
    "BikeNYC": ("BikeNYC", 5, 24),
}
SEEDS = (2024, 2025, 2026)


def slot_means(series: np.ndarray, day_len: int) -> np.ndarray:
    global_mean = np.mean(series, axis=0)
    means = np.empty((day_len, *series.shape[1:]), dtype=np.float64)
    indices = np.arange(len(series))
    for slot in range(day_len):
        values = series[indices % day_len == slot]
        means[slot] = np.mean(values, axis=0) if len(values) else global_mean
    return means


def metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    error = prediction.astype(np.float64) - target.astype(np.float64)
    return math.sqrt(float(np.mean(error * error))), float(
        np.mean(np.abs(error))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-prefix", default="MainSeed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("revision/round2/hamean_seed_metrics.csv"),
    )
    args = parser.parse_args()

    rows = []
    for dataset_key, (paper_name, len_period, day_len) in DATASETS.items():
        root = (
            args.code_root
            / "data"
            / f"{args.data_prefix}_{dataset_key}"
        )
        train_y = np.load(root / "train" / "Y.npy").astype(np.float32)
        test_y = np.load(root / "test" / "Y.npy").astype(np.float32)
        offset = max(3, len_period * day_len)
        target_indices = np.arange(offset, len(test_y))
        prediction = slot_means(train_y, day_len)[target_indices % day_len]
        rmse, mae = metrics(prediction, test_y[target_indices])
        for seed in SEEDS:
            rows.append(
                {
                    "dataset": paper_name,
                    "method": "HA-Mean",
                    "seed": seed,
                    "RMSE": f"{rmse:.6f}",
                    "MAE": f"{mae:.6f}",
                    "protocol": "MainSeed-RawCount-v2",
                    "source": "deterministic within-day training-slot mean",
                    "test_samples": len(target_indices),
                    "history_offset": offset,
                }
            )
    if len(rows) != 15:
        raise RuntimeError(f"Expected 15 HA-Mean rows, received {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} deterministic HA-Mean rows to {args.output}")


if __name__ == "__main__":
    main()
