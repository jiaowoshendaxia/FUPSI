#!/usr/bin/env python3
"""Prepare Chicago Taxi coarse-to-fine time series for formal FUPSI runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def block_sum(fine: np.ndarray, factor: int) -> np.ndarray:
    t, h, w = fine.shape
    return fine.reshape(t, h // factor, factor, w // factor, factor).sum(axis=(2, 4))


def rounded_scaler(value: float, step: int = 50) -> int:
    return max(step, int(math.ceil(value / step) * step))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--external-dim", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output} already contains files; use --overwrite")

    fine = np.load(source / "raw_fine_series.npy").astype(np.float32)
    coarse = np.load(source / "raw_coarse_series.npy").astype(np.float32)
    if fine.ndim != 3 or coarse.ndim != 3:
        raise ValueError(f"Expected [T,H,W], got fine={fine.shape}, coarse={coarse.shape}")
    if len(fine) != len(coarse):
        raise ValueError("Fine and coarse series have different lengths")
    inferred = block_sum(fine, 2)
    max_alignment_error = float(np.max(np.abs(inferred - coarse)))
    if max_alignment_error > 1e-5:
        raise ValueError(f"Coarse/fine structural constraint failed: {max_alignment_error}")

    # Keep raw counts and add one explicit pickup-flow channel.
    fine = fine[:, None, :, :]
    coarse = coarse[:, None, :, :]
    n = len(fine)
    train_end = int(n * args.train_ratio)
    valid_end = int(n * (args.train_ratio + args.valid_ratio))
    splits = {
        "train": slice(0, train_end),
        "valid": slice(train_end, valid_end),
        "test": slice(valid_end, n),
    }
    output.mkdir(parents=True, exist_ok=True)
    split_shapes: dict[str, object] = {}
    for name, selection in splits.items():
        split_dir = output / name
        split_dir.mkdir(parents=True, exist_ok=True)
        x = coarse[selection]
        y = fine[selection]
        np.save(split_dir / "X.npy", x)
        np.save(split_dir / "Y.npy", y)
        np.save(split_dir / "ext.npy", np.zeros((len(x), args.external_dim), dtype=np.float32))
        split_shapes[f"{name}_X"] = list(x.shape)
        split_shapes[f"{name}_Y"] = list(y.shape)

    train_max = max(float(coarse[:train_end].max()), float(fine[:train_end].max()))
    shared_scaler = rounded_scaler(train_max)
    metadata = {
        "alias": output.name,
        "source": str(source),
        "task": "historical coarse pickup maps to next fine pickup map",
        "slots": n,
        "channels": 1,
        "coarse_grid": [int(coarse.shape[-2]), int(coarse.shape[-1])],
        "fine_grid": [int(fine.shape[-2]), int(fine.shape[-1])],
        "upscale_factor": 2,
        "day_len": 24,
        "split_policy": {
            "chronological": True,
            "train_ratio": args.train_ratio,
            "valid_ratio": args.valid_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.valid_ratio,
        },
        "shared_scaler_X_Y": shared_scaler,
        "scaler_source": "training-split maximum rounded up to the next 50",
        "training_max": train_max,
        "max_structural_alignment_error": max_alignment_error,
        **split_shapes,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
