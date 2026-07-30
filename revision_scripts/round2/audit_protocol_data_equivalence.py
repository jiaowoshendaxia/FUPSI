#!/usr/bin/env python3
"""Verify that all formal methods use byte-identical processed splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ExpectedDataset:
    split_lengths: dict[str, int]
    coarse_shape: tuple[int, int, int]
    fine_shape: tuple[int, int, int]


DATASETS = {
    "TaxiBJ_P1": ExpectedDataset(
        {"train": 3519, "valid": 391, "test": 978},
        (2, 8, 8),
        (2, 32, 32),
    ),
    "TaxiBJ_P2": ExpectedDataset(
        {"train": 3442, "valid": 382, "test": 956},
        (2, 8, 8),
        (2, 32, 32),
    ),
    "TaxiBJ_P3": ExpectedDataset(
        {"train": 4029, "valid": 447, "test": 1120},
        (2, 8, 8),
        (2, 32, 32),
    ),
    "TaxiBJ_P4": ExpectedDataset(
        {"train": 5199, "valid": 577, "test": 1444},
        (2, 8, 8),
        (2, 32, 32),
    ),
    "BikeNYC": ExpectedDataset(
        {"train": 3162, "valid": 351, "test": 879},
        (2, 8, 4),
        (2, 16, 8),
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument("--reference-prefix", default="MainSeed")
    parser.add_argument("--candidate-prefix", default="ResidualMainE300P5")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision/round2/protocol_audit"),
    )
    args = parser.parse_args()

    code_root = args.code_root.resolve()
    rows = []
    for dataset, expected in DATASETS.items():
        for split, expected_length in expected.split_lengths.items():
            for filename, expected_tail in (
                ("X.npy", expected.coarse_shape),
                ("Y.npy", expected.fine_shape),
                ("ext.npy", (7,)),
            ):
                reference = (
                    code_root
                    / "data"
                    / f"{args.reference_prefix}_{dataset}"
                    / split
                    / filename
                )
                candidate = (
                    code_root
                    / "data"
                    / f"{args.candidate_prefix}_{dataset}"
                    / split
                    / filename
                )
                if not reference.exists() or not candidate.exists():
                    raise FileNotFoundError(f"{reference} / {candidate}")
                reference_array = np.load(reference, mmap_mode="r")
                candidate_array = np.load(candidate, mmap_mode="r")
                expected_shape = (expected_length, *expected_tail)
                reference_hash = sha256(reference)
                candidate_hash = sha256(candidate)
                shape_match = (
                    tuple(reference_array.shape) == expected_shape
                    and tuple(candidate_array.shape) == expected_shape
                )
                dtype_match = reference_array.dtype == candidate_array.dtype
                hash_match = reference_hash == candidate_hash
                row = {
                    "dataset": dataset,
                    "split": split,
                    "file": filename,
                    "expected_shape": str(expected_shape),
                    "reference_shape": str(tuple(reference_array.shape)),
                    "candidate_shape": str(tuple(candidate_array.shape)),
                    "reference_dtype": str(reference_array.dtype),
                    "candidate_dtype": str(candidate_array.dtype),
                    "reference_sha256": reference_hash,
                    "candidate_sha256": candidate_hash,
                    "shape_match": str(shape_match).lower(),
                    "dtype_match": str(dtype_match).lower(),
                    "sha256_match": str(hash_match).lower(),
                }
                rows.append(row)
                if not (shape_match and dtype_match and hash_match):
                    raise ValueError(f"Protocol data mismatch: {row}")

    expected_rows = len(DATASETS) * 3 * 3
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} audit rows, found {len(rows)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "processed_split_equivalence.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "MainSeed-RawCount-v2",
        "reference_prefix": args.reference_prefix,
        "candidate_prefix": args.candidate_prefix,
        "audit_rows": len(rows),
        "all_shapes_match": True,
        "all_dtypes_match": True,
        "all_sha256_match": True,
        "split_policy": "chronological 72/8/20",
        "scalers": {"coarse": 1500, "fine": 100},
    }
    (args.output_dir / "processed_split_equivalence.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"PASS: {len(rows)} files have the expected shapes and byte-identical "
        "content across both prefixes."
    )


if __name__ == "__main__":
    main()
