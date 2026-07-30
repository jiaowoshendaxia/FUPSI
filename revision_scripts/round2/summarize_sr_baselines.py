#!/usr/bin/env python3
"""Audit and summarize the 27 formal SR-only baseline runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


DATASETS = {
    "MainSeed_TaxiBJ_P1": ("TaxiBJ_P1", "TaxiBJ P1"),
    "MainSeed_TaxiBJ_P2": ("TaxiBJ_P2", "TaxiBJ P2"),
    "MainSeed_TaxiBJ_P3": ("TaxiBJ_P3", "TaxiBJ P3"),
    "MainSeed_TaxiBJ_P4": ("TaxiBJ_P4", "TaxiBJ P4"),
    "MainSeed_BikeNYC": ("BikeNYC", "BikeNYC"),
}
SEEDS = (2024, 2025, 2026)
HASH_FIELDS = (
    "test_coarse_sha256",
    "true_coarse_sha256",
    "true_fine_sha256",
)
COARSE_METRIC_FIELDS = {
    "RMSE_c": "coarse_RMSE",
    "MAE_c": "coarse_MAE",
    "MAPE_c": "coarse_MAPE",
}


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def expected_keys() -> set[tuple[str, str, int]]:
    keys = set()
    for alias in DATASETS:
        methods = ("UrbanFM", "FODE")
        for method in methods:
            for seed in SEEDS:
                keys.add((alias, method, seed))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("revision/round2/sr_baselines"),
    )
    parser.add_argument(
        "--fupsi",
        type=Path,
        default=Path("revision/round2/fupsi_seed_metrics.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision/round2/sr_baselines/summary"),
    )
    args = parser.parse_args()

    fupsi_rows = read_rows(args.fupsi)
    fupsi_index = {
        (row["dataset_key"], int(row["seed"])): row for row in fupsi_rows
    }
    expected_fupsi = {
        (dataset_key, seed)
        for dataset_key, _paper_name in DATASETS.values()
        for seed in SEEDS
    }
    if set(fupsi_index) != expected_fupsi:
        raise ValueError(
            "FUPSI seed metrics do not contain exactly five datasets x "
            "three seeds."
        )

    baseline_rows = []
    discovered_keys = set()
    for path in sorted(args.input_root.rglob("test_metrics.csv")):
        rows = read_rows(path)
        if len(rows) != 1:
            raise ValueError(f"{path}: expected one row, found {len(rows)}")
        row = rows[0]
        alias = row["alias"]
        method = row["method"]
        seed = int(row["seed"])
        if alias not in DATASETS:
            raise ValueError(f"{path}: unexpected dataset alias {alias!r}")
        if method not in {"UrbanFM", "FODE"}:
            raise ValueError(f"{path}: unexpected SR method {method!r}")
        if seed not in SEEDS:
            raise ValueError(f"{path}: unexpected seed {seed}")
        if row.get("mode") != "test":
            raise ValueError(f"{path}: expected mode='test'")
        for metric in ("RMSE", "MAE", *COARSE_METRIC_FIELDS):
            value = float(row[metric])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{path}: invalid {metric}={value}")
        key = (alias, method, seed)
        if key in discovered_keys:
            raise ValueError(f"Duplicate SR baseline result: {key}")
        discovered_keys.add(key)
        row["_source"] = path.relative_to(args.input_root).as_posix()
        baseline_rows.append(row)

    expected = expected_keys()
    if discovered_keys != expected:
        raise ValueError(
            f"SR baseline result mismatch: missing={sorted(expected - discovered_keys)}, "
            f"extra={sorted(discovered_keys - expected)}"
        )

    normalized = []
    alignment = []
    coarse_metric_alignment = []
    for row in baseline_rows:
        alias = row["alias"]
        dataset_key, paper_name = DATASETS[alias]
        seed = int(row["seed"])
        fupsi = fupsi_index[(dataset_key, seed)]
        for hash_field in HASH_FIELDS:
            matched = row.get(hash_field) == fupsi.get(hash_field)
            alignment.append(
                {
                    "dataset": paper_name,
                    "method": row["method"],
                    "seed": seed,
                    "artifact": hash_field.removesuffix("_sha256"),
                    "fupsi_sha256": fupsi.get(hash_field, ""),
                    "baseline_sha256": row.get(hash_field, ""),
                    "exact_match": str(matched).lower(),
                }
            )
            if not matched:
                raise RuntimeError(
                    f"Shared-input hash mismatch for {paper_name}, "
                    f"{row['method']}, seed {seed}, {hash_field}"
                )
        for baseline_field, fupsi_field in COARSE_METRIC_FIELDS.items():
            baseline_value = float(row[baseline_field])
            fupsi_value = float(fupsi[fupsi_field])
            matched = math.isclose(
                baseline_value, fupsi_value, rel_tol=1e-6, abs_tol=5e-4
            )
            coarse_metric_alignment.append(
                {
                    "dataset": paper_name,
                    "method": row["method"],
                    "seed": seed,
                    "metric": baseline_field,
                    "fupsi_value": f"{fupsi_value:.6f}",
                    "baseline_value": f"{baseline_value:.6f}",
                    "exact_within_csv_rounding": str(matched).lower(),
                }
            )
            if not matched:
                raise RuntimeError(
                    f"Shared coarse metric mismatch for {paper_name}, "
                    f"{row['method']}, seed {seed}, {baseline_field}: "
                    f"{baseline_value} vs {fupsi_value}"
                )
        normalized.append(
            {
                "dataset": paper_name,
                "method": row["method"],
                "seed": seed,
                "RMSE": f"{float(row['RMSE']):.6f}",
                "MAE": f"{float(row['MAE']):.6f}",
                "protocol": "MainSeed-RawCount-v2",
                "source": row["_source"],
            }
        )

    normalized.sort(key=lambda row: (row["dataset"], row["method"], row["seed"]))
    write_rows(
        args.output_dir / "sr_baseline_seed_metrics.csv",
        normalized,
        list(normalized[0]),
    )
    write_rows(
        args.output_dir / "shared_input_sha256_audit.csv",
        alignment,
        list(alignment[0]),
    )
    write_rows(
        args.output_dir / "shared_coarse_metric_audit.csv",
        coarse_metric_alignment,
        list(coarse_metric_alignment[0]),
    )

    summaries = []
    groups = {}
    for row in normalized:
        for metric in ("RMSE", "MAE"):
            groups.setdefault(
                (row["dataset"], row["method"], metric), []
            ).append(float(row[metric]))
    for (dataset, method, metric), values in sorted(groups.items()):
        array = np.asarray(values, dtype=np.float64)
        summaries.append(
            {
                "dataset": dataset,
                "method": method,
                "metric": metric,
                "runs": len(array),
                "mean": f"{array.mean():.6f}",
                "std": f"{array.std(ddof=1):.6f}",
                "mean_std": f"{array.mean():.4f} +/- {array.std(ddof=1):.4f}",
            }
        )
    write_rows(
        args.output_dir / "sr_baseline_mean_std.csv",
        summaries,
        list(summaries[0]),
    )
    metadata = {
        "protocol": "MainSeed-RawCount-v2",
        "expected_seed_results": 30,
        "seed_results": len(normalized),
        "expected_hash_checks": 90,
        "hash_checks": len(alignment),
        "all_hash_checks_exact": all(
            row["exact_match"] == "true" for row in alignment
        ),
        "expected_coarse_metric_checks": 90,
        "coarse_metric_checks": len(coarse_metric_alignment),
        "all_coarse_metric_checks_match": all(
            row["exact_within_csv_rounding"] == "true"
            for row in coarse_metric_alignment
        ),
        "bike_coarse_scaler": 1500,
        "fine_scaler": 100,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sr_baseline_audit.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"Validated {len(normalized)}/30 SR runs, "
        f"{len(alignment)}/90 exact shared-input hashes, and "
        f"{len(coarse_metric_alignment)}/90 shared coarse metrics."
    )


if __name__ == "__main__":
    main()
