#!/usr/bin/env python3
"""Aggregate HRSTT reimplementation seed results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


DATASETS = {
    "TaxiBJ_P1": "TaxiBJ P1",
    "TaxiBJ_P2": "TaxiBJ P2",
    "TaxiBJ_P3": "TaxiBJ P3",
    "TaxiBJ_P4": "TaxiBJ P4",
    "BikeNYC": "BikeNYC",
}
SEEDS = (2024, 2025, 2026)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", type=Path, default=Path("revision/round2/hrstt")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("revision/round2/hrstt/analysis")
    )
    args = parser.parse_args()

    seed_rows: list[dict] = []
    discovered: set[tuple[str, int]] = set()
    for path in sorted(args.input_root.glob("*/seed*/test_metrics.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ValueError(f"Expected one result row in {path}, found {len(rows)}")
        row = rows[0]
        dataset_key = row["dataset_key"]
        seed = int(row["seed"])
        key = (dataset_key, seed)
        if dataset_key not in DATASETS:
            raise ValueError(f"Unexpected HRSTT dataset key: {dataset_key}")
        if row["dataset"] != DATASETS[dataset_key]:
            raise ValueError(
                f"{path}: dataset label {row['dataset']!r} does not match "
                f"{DATASETS[dataset_key]!r}"
            )
        if row["method"] != "HRSTT_reimplementation":
            raise ValueError(f"{path}: unexpected method {row['method']!r}")
        if key in discovered:
            raise ValueError(f"Duplicate HRSTT result: {key}")
        discovered.add(key)
        for metric in ("RMSE", "MAE"):
            value = float(row[metric])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{path}: invalid {metric}={value}")
        metadata_path = path.with_name("run_metadata.json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing HRSTT metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("protocol") != "MainSeed-RawCount-v2":
            raise ValueError(f"{metadata_path}: protocol mismatch")
        normalization = metadata.get("normalization", {})
        if (
            float(normalization.get("coarse_divisor", -1)) != 1500.0
            or float(normalization.get("fine_divisor", -1)) != 100.0
            or normalization.get("metrics") != "raw counts after inverse scaling"
        ):
            raise ValueError(f"{metadata_path}: normalization mismatch")
        row["checkpoint"] = (
            path.parent / "best_model.pt"
        ).relative_to(args.input_root).as_posix()
        row["source_file"] = path.relative_to(args.input_root).as_posix()
        seed_rows.append(row)
    expected = {
        (dataset_key, seed) for dataset_key in DATASETS for seed in SEEDS
    }
    if discovered != expected or len(seed_rows) != len(expected):
        raise ValueError(
            "HRSTT result matrix mismatch: "
            f"missing={sorted(expected - discovered)}, "
            f"extra={sorted(discovered - expected)}"
        )

    seed_fields = list(seed_rows[0])
    write_csv(args.output_dir / "hrstt_seed_metrics.csv", seed_rows, seed_fields)

    summary_rows: list[dict] = []
    for dataset in sorted({row["dataset"] for row in seed_rows}):
        dataset_rows = [row for row in seed_rows if row["dataset"] == dataset]
        for metric in ("RMSE", "MAE"):
            values = np.array(
                [float(row[metric]) for row in dataset_rows], dtype=np.float64
            )
            summary_rows.append(
                {
                    "dataset": dataset,
                    "method": "HRSTT_reimplementation",
                    "metric": metric,
                    "runs": len(values),
                    "mean": f"{values.mean():.6f}",
                    "std": f"{values.std(ddof=1) if len(values) > 1 else 0.0:.6f}",
                    "min": f"{values.min():.6f}",
                    "max": f"{values.max():.6f}",
                }
            )
    write_csv(
        args.output_dir / "hrstt_mean_std.csv",
        summary_rows,
        ["dataset", "method", "metric", "runs", "mean", "std", "min", "max"],
    )
    audit = {
        "protocol": "MainSeed-RawCount-v2",
        "method": "HRSTT (reimplementation)",
        "official_code_used": False,
        "expected_seed_results": len(expected),
        "seed_results": len(seed_rows),
        "datasets": list(DATASETS.values()),
        "seeds": list(SEEDS),
        "normalization": {
            "coarse_divisor": 1500,
            "fine_divisor": 100,
            "metrics": "raw counts after inverse scaling",
        },
    }
    (args.output_dir / "hrstt_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(
        f"Collected {len(seed_rows)} seed runs across "
        f"{len({row['dataset'] for row in seed_rows})} datasets."
    )


if __name__ == "__main__":
    main()
