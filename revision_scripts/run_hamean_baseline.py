#!/usr/bin/env python3
"""Run deterministic HA-Mean on MainSeed datasets.

The baseline predicts each target map using the average training map from the
same within-day slot. It follows the same target-index alignment as
data_process.get_dataloader: the first max(history offsets) samples in each
split are skipped because they do not have enough history inside that split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "revision" / "baseline"
STAT_DIR = ROOT / "revision" / "statistics"

SEEDS = [2024, 2025, 2026]
METRICS = ["RMSE", "MAE", "MAPE", "RMSE_c", "MAE_c", "MAPE_c"]

DATASETS: dict[str, dict[str, Any]] = {
    "MainSeed_TaxiBJ_P1": {
        "paper_name": "TaxiBJ P1",
        "len_closeness": 3,
        "len_period": 5,
        "len_trend": 0,
        "day_len": 48,
    },
    "MainSeed_TaxiBJ_P2": {
        "paper_name": "TaxiBJ P2",
        "len_closeness": 3,
        "len_period": 1,
        "len_trend": 0,
        "day_len": 48,
    },
    "MainSeed_TaxiBJ_P3": {
        "paper_name": "TaxiBJ P3",
        "len_closeness": 3,
        "len_period": 2,
        "len_trend": 0,
        "day_len": 48,
    },
    "MainSeed_TaxiBJ_P4": {
        "paper_name": "TaxiBJ P4",
        "len_closeness": 3,
        "len_period": 3,
        "len_trend": 0,
        "day_len": 48,
    },
    "MainSeed_BikeNYC": {
        "paper_name": "BikeNYC",
        "len_closeness": 3,
        "len_period": 5,
        "len_trend": 0,
        "day_len": 24,
    },
    "MainSeed_ChicagoTaxi2024": {
        "paper_name": "Chicago Taxi 2024",
        "len_closeness": 3,
        "len_period": 3,
        "len_trend": 0,
        "day_len": 24,
    },
}


def mse(pred: np.ndarray, real: np.ndarray) -> float:
    return float(np.mean((pred.astype(np.float64) - real.astype(np.float64)) ** 2))


def mae(pred: np.ndarray, real: np.ndarray) -> float:
    return float(np.mean(np.abs(pred.astype(np.float64) - real.astype(np.float64))))


def mape(pred: np.ndarray, real: np.ndarray, eps: float = 1e-6) -> float:
    pred_arr = pred.astype(np.float64)
    real_arr = real.astype(np.float64)
    return float(np.mean(np.abs((real_arr - pred_arr) / np.maximum(np.abs(real_arr), eps))))


def history_offset(cfg: dict[str, Any]) -> int:
    return max(
        int(cfg["len_closeness"]),
        int(cfg["len_period"]) * int(cfg["day_len"]),
        int(cfg["len_trend"]) * int(cfg["day_len"]) * 7,
    )


def slot_means(series: np.ndarray, day_len: int) -> tuple[np.ndarray, np.ndarray]:
    global_mean = np.mean(series, axis=0)
    means = np.empty((day_len, *series.shape[1:]), dtype=np.float64)
    for slot in range(day_len):
        values = series[np.arange(len(series)) % day_len == slot]
        means[slot] = np.mean(values, axis=0) if len(values) else global_mean
    return means, global_mean


def predict_by_slot(train_series: np.ndarray, target_indices: np.ndarray, day_len: int) -> np.ndarray:
    means, _ = slot_means(train_series, day_len)
    return means[target_indices % day_len].astype(np.float32)


def evaluate_dataset(code_root: Path, alias: str, cfg: dict[str, Any]) -> dict[str, float]:
    dataset_dir = code_root / "data" / alias
    train_x = np.load(dataset_dir / "train" / "X.npy")
    train_y = np.load(dataset_dir / "train" / "Y.npy")
    test_x = np.load(dataset_dir / "test" / "X.npy")
    test_y = np.load(dataset_dir / "test" / "Y.npy")
    offset = history_offset(cfg)
    target_indices = np.arange(offset, len(test_y))
    if len(target_indices) <= 0:
        raise ValueError(f"{alias} has no evaluable test samples after offset={offset}")

    pred_fine = predict_by_slot(train_y, target_indices, int(cfg["day_len"]))
    true_fine = test_y[target_indices]
    pred_coarse = predict_by_slot(train_x, target_indices, int(cfg["day_len"]))
    true_coarse = test_x[target_indices]

    return {
        "RMSE": math.sqrt(mse(pred_fine, true_fine)),
        "MAE": mae(pred_fine, true_fine),
        "MAPE": mape(pred_fine, true_fine),
        "RMSE_c": math.sqrt(mse(pred_coarse, true_coarse)),
        "MAE_c": mae(pred_coarse, true_coarse),
        "MAPE_c": mape(pred_coarse, true_coarse),
        "test_targets": float(len(target_indices)),
        "offset": float(offset),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def load_fupsi_long() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = STAT_DIR / "main_seed_fupsi_collected_metrics.csv"
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("mode") != "test":
                continue
            dataset = DATASETS.get(row["dataset"], {}).get("paper_name", row["dataset"])
            for metric in METRICS:
                rows.append(
                    {
                        "dataset": dataset,
                        "method": "FUPSI",
                        "seed": row["seed"],
                        "metric": metric,
                        "value": row[metric],
                        "source": row.get("source", ""),
                    }
                )
    return rows


def summarize(long_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in long_rows:
        grouped[(row["dataset"], row["method"], row["metric"])].append(float(row["value"]))
    out: list[dict[str, Any]] = []
    for (dataset, method, metric), values in sorted(grouped.items()):
        std = stdev(values) if len(values) > 1 else 0.0
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "metric": metric,
                "runs": len(values),
                "mean": fmt(mean(values)),
                "std": fmt(std),
                "mean_pm_std": f"{mean(values):.4f} +/- {std:.4f}",
            }
        )
    return out


def write_markdown_tables(summary_rows: list[dict[str, Any]]) -> None:
    primary = ["RMSE", "MAE", "RMSE_c", "MAE_c"]
    lookup = {
        (row["dataset"], row["method"], row["metric"]): row["mean_pm_std"]
        for row in summary_rows
        if row["metric"] in primary
    }
    methods = ["HA-Mean", "FUPSI"]
    lines = [
        "# MainSeed HA-Mean Baseline and FUPSI Comparison",
        "",
        "HA-Mean is deterministic, so the three seed rows are repeated to keep the same schema as stochastic methods. It is comparable with FUPSI only under the MainSeed raw-count protocol.",
        "",
        "| Dataset | Method | RMSE | MAE | RMSE_c | MAE_c |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in ["TaxiBJ P1", "TaxiBJ P2", "TaxiBJ P3", "TaxiBJ P4", "BikeNYC", "Chicago Taxi 2024"]:
        for method in methods:
            values = [lookup.get((dataset, method, metric), "") for metric in primary]
            lines.append(f"| {dataset} | {method} | " + " | ".join(values) + " |")
    lines.append("")
    (OUT_DIR / "hamean_fupsi_mainseed_table.md").write_text("\n".join(lines), encoding="utf-8")


def write_latex_table(summary_rows: list[dict[str, Any]]) -> None:
    primary = ["RMSE", "MAE", "RMSE_c", "MAE_c"]
    lookup = {
        (row["dataset"], row["method"], row["metric"]): row["mean_pm_std"].replace("+/-", r"$\pm$")
        for row in summary_rows
        if row["metric"] in primary
    }
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\caption{MainSeed raw-count comparison between HA-Mean and FUPSI. HA-Mean is deterministic and repeated over three seeds for schema consistency.}",
        r"\label{tab:mainseed-hamean-fupsi}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Method & RMSE & MAE & RMSE$_c$ & MAE$_c$ \\",
        r"\midrule",
    ]
    for dataset in ["TaxiBJ P1", "TaxiBJ P2", "TaxiBJ P3", "TaxiBJ P4", "BikeNYC", "Chicago Taxi 2024"]:
        for method in ["HA-Mean", "FUPSI"]:
            values = [lookup.get((dataset, method, metric), "") for metric in primary]
            lines.append(f"{dataset} & {method} & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (OUT_DIR / "hamean_fupsi_mainseed_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=ROOT / "fupsi")
    parser.add_argument("--seeds", default="2024,2025,2026")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    args = parser.parse_args()

    seed_values = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    selected_datasets = {part.strip() for part in args.datasets.split(",") if part.strip()}
    unknown = selected_datasets.difference(DATASETS)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for alias, cfg in DATASETS.items():
        if alias not in selected_datasets:
            continue
        metrics = evaluate_dataset(args.code_root, alias, cfg)
        audit_rows.append(
            {
                "dataset": cfg["paper_name"],
                "alias": alias,
                "offset": int(metrics["offset"]),
                "test_targets": int(metrics["test_targets"]),
                "day_len": cfg["day_len"],
            }
        )
        wide_metrics = {metric: fmt(metrics[metric]) for metric in METRICS}
        for seed in seed_values:
            seed_rows.append(
                {
                    "dataset": cfg["paper_name"],
                    "method": "HA-Mean",
                    "seed": seed,
                    **wide_metrics,
                    "source": "deterministic_within_day_training_mean",
                }
            )
            for metric in METRICS:
                long_rows.append(
                    {
                        "dataset": cfg["paper_name"],
                        "method": "HA-Mean",
                        "seed": seed,
                        "metric": metric,
                        "value": wide_metrics[metric],
                        "source": "deterministic_within_day_training_mean",
                    }
                )

    fupsi_rows = load_fupsi_long()
    combined_rows = [*fupsi_rows, *long_rows]
    summary_rows = summarize(combined_rows)

    write_csv(
        OUT_DIR / "hamean_mainseed_seed_metrics_wide.csv",
        seed_rows,
        ["dataset", "method", "seed", *METRICS, "source"],
    )
    write_csv(
        OUT_DIR / "hamean_mainseed_seed_metrics_long.csv",
        long_rows,
        ["dataset", "method", "seed", "metric", "value", "source"],
    )
    write_csv(
        OUT_DIR / "mainseed_fupsi_hamean_seed_metrics_long.csv",
        combined_rows,
        ["dataset", "method", "seed", "metric", "value", "source"],
    )
    write_csv(
        OUT_DIR / "mainseed_fupsi_hamean_summary.csv",
        summary_rows,
        ["dataset", "method", "metric", "runs", "mean", "std", "mean_pm_std"],
    )
    write_csv(
        OUT_DIR / "hamean_alignment_audit.csv",
        audit_rows,
        ["dataset", "alias", "offset", "test_targets", "day_len"],
    )
    write_markdown_tables(summary_rows)
    write_latex_table(summary_rows)
    print(f"Wrote HA-Mean baseline outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
