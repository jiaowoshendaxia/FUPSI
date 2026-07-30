#!/usr/bin/env python3
"""Collect and validate the 15 formal FUPSI seed-level result files."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
SPATIAL_SHAPES = {
    "TaxiBJ_P1": ((2, 8, 8), (2, 32, 32)),
    "TaxiBJ_P2": ((2, 8, 8), (2, 32, 32)),
    "TaxiBJ_P3": ((2, 8, 8), (2, 32, 32)),
    "TaxiBJ_P4": ((2, 8, 8), (2, 32, 32)),
    "BikeNYC": ((2, 8, 4), (2, 16, 8)),
}


def read_single_row(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one metric row, found {len(rows)}")
    return rows[0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_evaluation_array(path: Path, expected_spatial: tuple[int, ...]) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    if array.ndim != 4 or tuple(array.shape[1:]) != expected_spatial:
        raise ValueError(
            f"{path}: expected [N,{','.join(map(str, expected_spatial))}], "
            f"received {array.shape}"
        )
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError(f"{path}: empty array or non-finite values")
    return array


def recompute_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[float, float]:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}"
        )
    error = prediction.astype(np.float64) - target.astype(np.float64)
    return (
        math.sqrt(float(np.mean(error * error))),
        float(np.mean(np.abs(error))),
    )


def assert_close(recomputed: float, reported: float, label: str, path: Path) -> None:
    if not math.isclose(recomputed, reported, rel_tol=1e-6, abs_tol=5e-4):
        raise ValueError(
            f"{path}: {label} mismatch, reported={reported:.6f}, "
            f"recomputed={recomputed:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument("--namespace", default="ResidualMainE300P5")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("revision/round2/fupsi_seed_metrics.csv"),
    )
    args = parser.parse_args()

    code_root = args.code_root.resolve()
    model_root = code_root / "saved_model" / "to_stage" / "no_ext(r)"
    rows = []
    evidence = {}
    for dataset_key, paper_name in DATASETS.items():
        alias = f"{args.namespace}_{dataset_key}"
        evidence[paper_name] = {}
        for seed in SEEDS:
            seed_root = model_root / alias / f"seed{seed}"
            matches = sorted(seed_root.rglob("test_metrics.csv"))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"{paper_name} seed {seed}: expected one test_metrics.csv "
                    f"under {seed_root}, found {len(matches)}"
                )
            path = matches[0]
            raw = read_single_row(path)
            if int(raw["seed"]) != seed:
                raise ValueError(
                    f"{path}: row seed={raw['seed']} does not match {seed}"
                )
            if raw.get("dataset") != alias or raw.get("mode") != "test":
                raise ValueError(
                    f"{path}: expected dataset={alias!r}, mode='test'; "
                    f"received dataset={raw.get('dataset')!r}, "
                    f"mode={raw.get('mode')!r}"
                )
            generator_dir = path.parent
            array_paths = {
                name: generator_dir / name
                for name in (
                    "test_coarse.npy",
                    "true_coarse.npy",
                    "test_fine.npy",
                    "true_fine.npy",
                )
            }
            missing_arrays = [
                str(array_path)
                for array_path in array_paths.values()
                if not array_path.exists()
            ]
            if missing_arrays:
                raise FileNotFoundError(
                    f"{paper_name} seed {seed}: missing saved evaluation "
                    f"arrays: {missing_arrays}"
                )
            coarse_shape, fine_shape = SPATIAL_SHAPES[dataset_key]
            arrays = {
                "test_coarse": load_evaluation_array(
                    array_paths["test_coarse.npy"], coarse_shape
                ),
                "true_coarse": load_evaluation_array(
                    array_paths["true_coarse.npy"], coarse_shape
                ),
                "test_fine": load_evaluation_array(
                    array_paths["test_fine.npy"], fine_shape
                ),
                "true_fine": load_evaluation_array(
                    array_paths["true_fine.npy"], fine_shape
                ),
            }
            sample_counts = {len(array) for array in arrays.values()}
            if len(sample_counts) != 1:
                raise ValueError(
                    f"{paper_name} seed {seed}: unaligned evaluation arrays "
                    f"{ {name: array.shape for name, array in arrays.items()} }"
                )
            fine_rmse, fine_mae = recompute_metrics(
                arrays["test_fine"], arrays["true_fine"]
            )
            coarse_rmse, coarse_mae = recompute_metrics(
                arrays["test_coarse"], arrays["true_coarse"]
            )
            reported_metrics = {
                "RMSE": float(raw["RMSE"]),
                "MAE": float(raw["MAE"]),
                "MAPE": float(raw["MAPE"]),
                "RMSE_c": float(raw["RMSE_c"]),
                "MAE_c": float(raw["MAE_c"]),
                "MAPE_c": float(raw["MAPE_c"]),
            }
            if not all(
                math.isfinite(value) and value >= 0
                for value in reported_metrics.values()
            ):
                raise ValueError(f"{path}: invalid reported metrics")
            assert_close(fine_rmse, reported_metrics["RMSE"], "fine RMSE", path)
            assert_close(fine_mae, reported_metrics["MAE"], "fine MAE", path)
            assert_close(
                coarse_rmse, reported_metrics["RMSE_c"], "coarse RMSE", path
            )
            assert_close(
                coarse_mae, reported_metrics["MAE_c"], "coarse MAE", path
            )
            relative_source = path.relative_to(code_root).as_posix()
            row = {
                "dataset": paper_name,
                "dataset_key": dataset_key,
                "method": "FUPSI",
                "seed": seed,
                "RMSE": f"{float(raw['RMSE']):.6f}",
                "MAE": f"{float(raw['MAE']):.6f}",
                "MAPE": f"{float(raw['MAPE']):.6f}",
                "coarse_RMSE": f"{float(raw['RMSE_c']):.6f}",
                "coarse_MAE": f"{float(raw['MAE_c']):.6f}",
                "coarse_MAPE": f"{float(raw['MAPE_c']):.6f}",
                "protocol": "MainSeed-RawCount-v2",
                "metrics_recomputed_from_saved_arrays": "true",
                "test_samples": next(iter(sample_counts)),
                "source": relative_source,
                "source_sha256": file_sha256(path),
                "test_coarse_sha256": file_sha256(
                    array_paths["test_coarse.npy"]
                ),
                "true_coarse_sha256": file_sha256(
                    array_paths["true_coarse.npy"]
                ),
                "test_fine_sha256": file_sha256(
                    array_paths["test_fine.npy"]
                ),
                "true_fine_sha256": file_sha256(
                    array_paths["true_fine.npy"]
                ),
            }
            rows.append(row)
            evidence[paper_name][str(seed)] = {
                "source": relative_source,
                "sha256": row["source_sha256"],
            }

    if len(rows) != 15:
        raise RuntimeError(f"Expected 15 rows, found {len(rows)}")
    keys = {(row["dataset"], row["seed"]) for row in rows}
    if len(keys) != 15:
        raise RuntimeError("Duplicate dataset/seed result rows detected")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "protocol": "MainSeed-RawCount-v2",
                "namespace": args.namespace,
                "expected_rows": 15,
                "rows": len(rows),
                "metric_audit": (
                    "RMSE and MAE were recomputed from saved raw-count "
                    "prediction/target arrays and matched the metric CSVs."
                ),
                "evidence": evidence,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Validated and wrote {len(rows)} FUPSI rows to {args.output}")


if __name__ == "__main__":
    main()
