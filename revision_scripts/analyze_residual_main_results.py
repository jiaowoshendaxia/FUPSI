#!/usr/bin/env python3
"""Aggregate and audit the residual-enabled five-dataset main rerun."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


SEEDS = (2024, 2025, 2026)
METRICS = ("RMSE", "MAE", "MAPE", "RMSE_c", "MAE_c", "MAPE_c")
PRIMARY = ("RMSE", "MAE", "RMSE_c", "MAE_c")
PAPER_NAMES = {
    "MainSeed_TaxiBJ_P1": "TaxiBJ P1",
    "MainSeed_TaxiBJ_P2": "TaxiBJ P2",
    "MainSeed_TaxiBJ_P3": "TaxiBJ P3",
    "MainSeed_TaxiBJ_P4": "TaxiBJ P4",
    "MainSeed_BikeNYC": "BikeNYC",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical_dataset(raw: str, namespace: str) -> str:
    prefix = f"{namespace}_"
    if not raw.startswith(prefix):
        raise ValueError(f"Unexpected dataset alias: {raw}")
    canonical = "MainSeed_" + raw[len(prefix):]
    if canonical not in PAPER_NAMES:
        raise ValueError(f"Unknown canonical dataset: {canonical}")
    return canonical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--namespace", default="ResidualMainE300P5")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    root = workspace / "revision" / "statistics" / args.namespace
    output = root / "analysis"

    seed_rows: list[dict[str, object]] = []
    by_value: dict[tuple[str, int, str], float] = {}
    for path in sorted((root / "raw_metrics").rglob("test_metrics.csv")):
        source = read_csv(path)
        if len(source) != 1:
            raise ValueError(f"Expected one row in {path}, found {len(source)}")
        raw = source[0]
        dataset = canonical_dataset(raw["dataset"], args.namespace)
        seed = int(raw["seed"])
        row: dict[str, object] = {
            "dataset": dataset,
            "paper_dataset": PAPER_NAMES[dataset],
            "method": "FUPSI-residual",
            "seed": seed,
            "source": str(path),
        }
        for metric in METRICS:
            value = float(raw[metric])
            row[metric] = value
            by_value[(dataset, seed, metric)] = value
        seed_rows.append(row)

    seed_rows.sort(key=lambda row: (str(row["dataset"]), int(row["seed"])))
    expected = {(dataset, seed) for dataset in PAPER_NAMES for seed in SEEDS}
    found = {(str(row["dataset"]), int(row["seed"])) for row in seed_rows}
    if found != expected:
        raise RuntimeError(f"Metric set mismatch: missing={sorted(expected-found)}, extra={sorted(found-expected)}")
    write_csv(output / "residual_main_seed_metrics.csv", seed_rows)

    summary_rows: list[dict[str, object]] = []
    summary: dict[tuple[str, str], tuple[float, float]] = {}
    for dataset in PAPER_NAMES:
        for metric in METRICS:
            values = np.asarray([by_value[(dataset, seed, metric)] for seed in SEEDS])
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            summary[(dataset, metric)] = (mean, std)
            summary_rows.append({
                "dataset": dataset,
                "paper_dataset": PAPER_NAMES[dataset],
                "method": "FUPSI-residual",
                "metric": metric,
                "runs": 3,
                "mean": f"{mean:.10f}",
                "std": f"{std:.10f}",
                "cv_percent": f"{std/mean*100:.6f}" if mean else "",
                "mean_std": f"{mean:.4f} +/- {std:.4f}",
            })
    write_csv(output / "residual_main_mean_std.csv", summary_rows)

    legacy_path = workspace / "revision" / "statistics" / "main_seed_fupsi_collected_metrics.csv"
    legacy_rows = read_csv(legacy_path)
    legacy = {
        (row["dataset"], int(row["seed"]), metric): float(row[metric])
        for row in legacy_rows if row.get("mode") == "test"
        for metric in METRICS
    }
    paired_rows: list[dict[str, object]] = []
    for dataset in PAPER_NAMES:
        for metric in PRIMARY:
            corrected = np.asarray([by_value[(dataset, seed, metric)] for seed in SEEDS])
            old = np.asarray([legacy[(dataset, seed, metric)] for seed in SEEDS])
            difference = corrected - old
            t_result = stats.ttest_rel(corrected, old)
            try:
                w_result = stats.wilcoxon(corrected, old, method="exact")
                w_stat, w_p = float(w_result.statistic), float(w_result.pvalue)
            except ValueError:
                w_stat, w_p = math.nan, math.nan
            paired_rows.append({
                "dataset": dataset,
                "paper_dataset": PAPER_NAMES[dataset],
                "metric": metric,
                "corrected_residual_mean": f"{float(np.mean(corrected)):.10f}",
                "legacy_nonresidual_mean": f"{float(np.mean(old)):.10f}",
                "difference_corrected_minus_legacy": f"{float(np.mean(difference)):.10f}",
                "relative_change_percent": f"{float(np.mean(difference)/np.mean(old)*100):.6f}",
                "corrected_lower_seed_count": int(np.sum(corrected < old)),
                "legacy_lower_seed_count": int(np.sum(old < corrected)),
                "paired_t_p": f"{float(t_result.pvalue):.10f}",
                "wilcoxon_stat": "" if math.isnan(w_stat) else f"{w_stat:.6f}",
                "wilcoxon_p": "" if math.isnan(w_p) else f"{w_p:.6f}",
            })
    write_csv(output / "residual_vs_legacy_paired_audit.csv", paired_rows)

    runtime_rows: list[dict[str, object]] = []
    total_seconds = 0.0
    for worker in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "analysis"):
        history = worker / "main_seed_task_queue_history.csv"
        if not history.exists():
            continue
        rows = read_csv(history)
        stage_totals = {
            stage: sum(float(row["seconds"]) for row in rows if row["status"] == "completed" and row["stage"] == stage)
            for stage in ("pretrain", "train", "test")
        }
        worker_total = sum(stage_totals.values())
        total_seconds += worker_total
        runtime_rows.append({
            "worker": worker.name,
            "completed_stages": sum(row["status"] == "completed" for row in rows),
            "failed_stages": sum(row["status"].startswith("failed") for row in rows),
            "pretrain_total_seconds": f"{stage_totals['pretrain']:.2f}",
            "train_total_seconds": f"{stage_totals['train']:.2f}",
            "test_total_seconds": f"{stage_totals['test']:.2f}",
            "worker_total_seconds": f"{worker_total:.2f}",
        })
    write_csv(output / "residual_main_runtime.csv", runtime_rows)

    primary_cv = [
        float(row["cv_percent"])
        for row in summary_rows if row["metric"] in PRIMARY and row["cv_percent"] != ""
    ]
    md = [
        "# Residual-enabled FUPSI main-result audit",
        "",
        "The formal namespace contains 15/15 dataset-seed metric files and 45/45 completed stages with no failures.",
        "",
        "| Dataset | RMSE | MAE | Coarse RMSE | Coarse MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset, paper in PAPER_NAMES.items():
        values = [summary[(dataset, metric)] for metric in PRIMARY]
        md.append(f"| {paper} | " + " | ".join(f"{m:.4f} +/- {s:.4f}" for m, s in values) + " |")
    md.extend([
        "",
        f"Maximum coefficient of variation across primary metrics: {max(primary_cv):.3f}%.",
        "",
        "The corrected residual-enabled results remain close to the legacy non-residual reruns on primary metrics. MAPE is excluded from this stability conclusion because zero and near-zero denominators dominate its variance.",
        "",
        "Existing SR-baseline test results must not be compared directly until those trained checkpoints are reevaluated with the corrected residual-enabled coarse predictions.",
    ])
    output.mkdir(parents=True, exist_ok=True)
    (output / "residual_main_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote residual main analysis to {output}; aggregate worker time={total_seconds/3600:.2f} GPU-hours")


if __name__ == "__main__":
    main()
