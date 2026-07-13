#!/usr/bin/env python3
"""Prepare coarse/fine datasets for main-table seed reruns.

This script rebuilds non-destructive data aliases under the experiment repo:

    data/MainSeed_TaxiBJ_P1
    ...
    data/MainSeed_BikeNYC

It uses raw fine-grid H5 files, creates coarse inputs by block-sum aggregation,
and keeps the fine-grid maps as targets. The original data directories are not
modified.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


MANUSCRIPT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = MANUSCRIPT_ROOT / "revision" / "statistics"

DATASETS: dict[str, dict[str, Any]] = {
    "TaxiBJ_P1": {
        "source": "TaxiBJ and BikeNYC/TaxiBJ/BJ13_M32x32_T30_InOut.h5",
        "upscale_factor": 4,
        "day_len": 48,
    },
    "TaxiBJ_P2": {
        "source": "TaxiBJ and BikeNYC/TaxiBJ/BJ14_M32x32_T30_InOut.h5",
        "upscale_factor": 4,
        "day_len": 48,
    },
    "TaxiBJ_P3": {
        "source": "TaxiBJ and BikeNYC/TaxiBJ/BJ15_M32x32_T30_InOut.h5",
        "upscale_factor": 4,
        "day_len": 48,
    },
    "TaxiBJ_P4": {
        "source": "TaxiBJ and BikeNYC/TaxiBJ/BJ16_M32x32_T30_InOut.h5",
        "upscale_factor": 4,
        "day_len": 48,
    },
    "BikeNYC": {
        "source": "TaxiBJ and BikeNYC/BikeNYC/NYC14_M16x8_T60_NewEnd.h5",
        "upscale_factor": 2,
        "day_len": 24,
    },
}


def block_sum(fine: np.ndarray, factor: int) -> np.ndarray:
    if fine.ndim != 4:
        raise ValueError(f"Expected fine data [T,C,H,W], got shape {fine.shape}")
    t, c, h, w = fine.shape
    if h % factor != 0 or w % factor != 0:
        raise ValueError(f"Fine shape {(h, w)} is not divisible by factor {factor}")
    return fine.reshape(t, c, h // factor, factor, w // factor, factor).sum(axis=(3, 5))


def split_indices(n: int, train_ratio: float, valid_ratio_within_train: float) -> dict[str, slice]:
    train_end = int(n * train_ratio)
    valid_size = max(1, int(train_end * valid_ratio_within_train))
    train_only_end = train_end - valid_size
    return {
        "train": slice(0, train_only_end),
        "valid": slice(train_only_end, train_end),
        "test": slice(train_end, n),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_one(
    code_root: Path,
    dataset: str,
    cfg: dict[str, Any],
    prefix: str,
    overwrite: bool,
    train_ratio: float,
    valid_ratio_within_train: float,
    external_dim: int,
) -> dict[str, Any]:
    src = code_root / cfg["source"]
    if not src.exists():
        raise FileNotFoundError(src)
    alias = f"{prefix}_{dataset}"
    out_root = code_root / "data" / alias
    if out_root.exists() and any(out_root.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_root} exists. Use --overwrite to rebuild it.")

    with h5py.File(src, "r") as h:
        fine = h["data"][:].astype(np.float32)
        dates = h["date"][:] if "date" in h else None

    factor = int(cfg["upscale_factor"])
    coarse = block_sum(fine, factor).astype(np.float32)
    splits = split_indices(len(fine), train_ratio, valid_ratio_within_train)

    out_root.mkdir(parents=True, exist_ok=True)
    split_summary: dict[str, Any] = {}
    for split, sl in splits.items():
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / "X.npy", coarse[sl])
        np.save(split_dir / "Y.npy", fine[sl])
        np.save(split_dir / "ext.npy", np.zeros((len(fine[sl]), external_dim), dtype=np.float32))
        if dates is not None:
            np.save(split_dir / "date.npy", dates[sl])
        split_summary[f"{split}_X_shape"] = tuple(int(x) for x in coarse[sl].shape)
        split_summary[f"{split}_Y_shape"] = tuple(int(x) for x in fine[sl].shape)

    metadata = {
        "alias": alias,
        "source": str(cfg["source"]),
        "upscale_factor": factor,
        "day_len": cfg["day_len"],
        "fine_shape": tuple(int(x) for x in fine.shape),
        "coarse_shape": tuple(int(x) for x in coarse.shape),
        "split_policy": {
            "train_ratio": train_ratio,
            "valid_ratio_within_train": valid_ratio_within_train,
            "chronological": True,
        },
        "external_features": {
            "dim": external_dim,
            "source": "zero placeholder used to keep the original external-feature interface stable",
        },
        **split_summary,
    }
    (out_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "dataset": dataset,
        "alias": alias,
        "source": cfg["source"],
        "upscale_factor": factor,
        "day_len": cfg["day_len"],
        "fine_shape": str(tuple(int(x) for x in fine.shape)),
        "coarse_shape": str(tuple(int(x) for x in coarse.shape)),
        **{key: str(value) for key, value in split_summary.items()},
    }


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Prepared Main-Seed Datasets",
        "",
        "These aliases are generated from raw H5 fine-grid data for main-table reruns. The original missing-study data directories are not modified.",
        "",
        "| Dataset | Alias | M | Fine series | Coarse series | Train X | Valid X | Test X |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['alias']} | {row['upscale_factor']} | {row['fine_shape']} | {row['coarse_shape']} | {row['train_X_shape']} | {row['valid_X_shape']} | {row['test_X_shape']} |"
        )
    lines += [
        "",
        "Next required code-side fixes before training:",
        "",
        "1. Use these aliases in train/test commands, e.g. `--dataset MainSeed_TaxiBJ_P1`.",
        "2. Allow `train.py` and `test.py` to accept smaller coarse maps such as 8x8 and 8x4.",
        "3. Fix the pretrain checkpoint path mismatch or copy `cpt_noext` to `cpt` in the runner.",
        "4. Isolate model/test outputs by seed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument("--prefix", default="MainSeed")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio-within-train", type=float, default=0.1)
    parser.add_argument("--external-dim", type=int, default=7)
    args = parser.parse_args()

    code_root = args.code_root.resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        prepare_one(
            code_root,
            dataset,
            cfg,
            args.prefix,
            args.overwrite,
            args.train_ratio,
            args.valid_ratio_within_train,
            args.external_dim,
        )
        for dataset, cfg in DATASETS.items()
    ]

    write_csv(
        OUT_DIR / "prepared_main_seed_datasets.csv",
        rows,
        [
            "dataset",
            "alias",
            "source",
            "upscale_factor",
            "day_len",
            "fine_shape",
            "coarse_shape",
            "train_X_shape",
            "train_Y_shape",
            "valid_X_shape",
            "valid_Y_shape",
            "test_X_shape",
            "test_Y_shape",
        ],
    )
    write_markdown(OUT_DIR / "prepared_main_seed_datasets.md", rows)
    print(f"Prepared {len(rows)} main-seed dataset aliases.")
    print(f"Wrote summary to {OUT_DIR}")


if __name__ == "__main__":
    main()
