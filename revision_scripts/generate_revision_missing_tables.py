#!/usr/bin/env python
"""Generate revision-ready missing-completion tables.

This script combines multiple expanded_missing_rate_report.json files and
produces compact manuscript tables for the validation-fixed adaptive selector.
All output uses ASCII "+/-" to avoid encoding problems in LaTeX or Markdown.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DATASET_ORDER = [
    "TaxiBJ_P1",
    "TaxiBJ_P2",
    "TaxiBJ_P3",
    "TaxiBJ_P4",
    "BikeNYC",
    "ChicagoTaxi2024",
]
RATES = [0.1, 0.3, 0.5, 0.7]
SINGLE_BASELINES = [
    "no_completion",
    "zero_fill",
    "mean_fill",
    "linear_interpolation",
    "knn_fill",
    "svd_completion",
]
ALL_METHODS = SINGLE_BASELINES + ["adaptive_completion"]


def method_label(method: str) -> str:
    return {
        "no_completion": "No completion",
        "zero_fill": "Zero fill",
        "mean_fill": "Mean fill",
        "linear_interpolation": "Linear interpolation",
        "knn_fill": "KNN fill",
        "svd_completion": "SVD completion",
        "adaptive_completion": "Adaptive",
    }.get(method, method)


def load_summary(paths: Iterable[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(report["summary"])
    return rows


def load_plans(paths: Iterable[Path]) -> Dict[Tuple[str, float], str]:
    plan: Dict[Tuple[str, float], str] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("adaptive_plan", payload)
        for dataset, by_rate in raw.items():
            for rate, method in by_rate.items():
                plan[(str(dataset), float(rate))] = str(method)
    return plan


def available_datasets(summary: List[Dict[str, object]]) -> List[str]:
    names = {str(row["dataset"]) for row in summary}
    ordered = [name for name in DATASET_ORDER if name in names]
    ordered.extend(sorted(names - set(ordered)))
    return ordered


def find_row(summary: List[Dict[str, object]], dataset: str, rate: float, method: str) -> Dict[str, object]:
    for row in summary:
        if (
            str(row["dataset"]) == dataset
            and abs(float(row["missing_rate"]) - rate) < 1e-9
            and str(row["method"]) == method
        ):
            return row
    raise KeyError((dataset, rate, method))


def pm(row: Dict[str, object], metric: str = "completion_MSE", digits: int = 6) -> str:
    return f"{float(row[f'{metric}_mean']):.{digits}f} +/- {float(row[f'{metric}_std']):.{digits}f}"


def metric_values(row: Dict[str, object], metric: str = "completion_MSE") -> Tuple[float, float]:
    return float(row[f"{metric}_mean"]), float(row[f"{metric}_std"])


def write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def generate_table10(
    test_summary: List[Dict[str, object]],
    plan: Dict[Tuple[str, float], str],
    output_dir: Path,
) -> None:
    lines = [
        "# Table 10. Validation-fixed adaptive completion versus test best baseline",
        "",
        "Metric: completion MSE on synthetic missing entries, reported as mean +/- std over three random seeds.",
        "The adaptive method is selected on the validation split and then fixed for test evaluation.",
        "",
        "| Dataset | Missing rate | Validation-selected method | Test best single baseline | Best MSE | Adaptive MSE | Adaptive / Best |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    csv_rows: List[List[object]] = []
    for dataset in available_datasets(test_summary):
        for rate in RATES:
            baseline_rows = [find_row(test_summary, dataset, rate, method) for method in SINGLE_BASELINES]
            best = min(baseline_rows, key=lambda row: float(row["completion_MSE_mean"]))
            adaptive = find_row(test_summary, dataset, rate, "adaptive_completion")
            ratio = float(adaptive["completion_MSE_mean"]) / float(best["completion_MSE_mean"])
            selected = plan[(dataset, rate)]
            lines.append(
                f"| {dataset} | {int(rate * 100)}% | {method_label(selected)} | "
                f"{method_label(str(best['method']))} | {pm(best)} | {pm(adaptive)} | {ratio:.3f} |"
            )
            best_mean, best_std = metric_values(best)
            adaptive_mean, adaptive_std = metric_values(adaptive)
            csv_rows.append(
                [
                    dataset,
                    f"{int(rate * 100)}%",
                    method_label(selected),
                    method_label(str(best["method"])),
                    f"{best_mean:.10f}",
                    f"{best_std:.10f}",
                    f"{adaptive_mean:.10f}",
                    f"{adaptive_std:.10f}",
                    f"{ratio:.6f}",
                ]
            )
    (output_dir / "table10_validation_fixed_adaptive_vs_test_best.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    write_csv(
        output_dir / "table10_validation_fixed_adaptive_vs_test_best.csv",
        [
            "Dataset",
            "Missing Rate",
            "Validation-selected Method",
            "Test Best Single Baseline",
            "Best MSE Mean",
            "Best MSE Std",
            "Adaptive MSE Mean",
            "Adaptive MSE Std",
            "Adaptive / Best",
        ],
        csv_rows,
    )


def generate_validation_table(
    validation_summary: List[Dict[str, object]],
    plan: Dict[Tuple[str, float], str],
    output_dir: Path,
) -> None:
    lines = [
        "# Validation selection check",
        "",
        "Metric: validation completion MSE on synthetic missing entries, reported as mean +/- std over three random seeds.",
        "",
        "| Dataset | Missing rate | Selected method | Validation best baseline | Selected MSE | Best MSE | Selected / Best |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    csv_rows: List[List[object]] = []
    for dataset in available_datasets(validation_summary):
        for rate in RATES:
            selected = plan[(dataset, rate)]
            selected_row = find_row(validation_summary, dataset, rate, selected)
            baseline_rows = [find_row(validation_summary, dataset, rate, method) for method in SINGLE_BASELINES]
            best = min(baseline_rows, key=lambda row: float(row["completion_MSE_mean"]))
            ratio = float(selected_row["completion_MSE_mean"]) / float(best["completion_MSE_mean"])
            lines.append(
                f"| {dataset} | {int(rate * 100)}% | {method_label(selected)} | "
                f"{method_label(str(best['method']))} | {pm(selected_row)} | {pm(best)} | {ratio:.3f} |"
            )
            selected_mean, selected_std = metric_values(selected_row)
            best_mean, best_std = metric_values(best)
            csv_rows.append(
                [
                    dataset,
                    f"{int(rate * 100)}%",
                    method_label(selected),
                    method_label(str(best["method"])),
                    f"{selected_mean:.10f}",
                    f"{selected_std:.10f}",
                    f"{best_mean:.10f}",
                    f"{best_std:.10f}",
                    f"{ratio:.6f}",
                ]
            )
    (output_dir / "validation_selection_check.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    write_csv(
        output_dir / "validation_selection_check.csv",
        [
            "Dataset",
            "Missing Rate",
            "Selected Method",
            "Validation Best Baseline",
            "Selected MSE Mean",
            "Selected MSE Std",
            "Best MSE Mean",
            "Best MSE Std",
            "Selected / Best",
        ],
        csv_rows,
    )


def generate_full_tables(test_summary: List[Dict[str, object]], output_dir: Path) -> None:
    groups = [
        ("appendix_table_a1_completion_no_zero_mean_linear", ALL_METHODS[:4]),
        ("appendix_table_a2_completion_knn_svd_adaptive", ALL_METHODS[4:]),
    ]
    for filename, methods in groups:
        lines = [
            f"# {filename}",
            "",
            "Metric: completion MSE on synthetic missing entries, reported as mean +/- std over three random seeds.",
            "",
            "| Dataset | Missing rate | " + " | ".join(method_label(method) for method in methods) + " |",
            "|---|---:" + "|---:" * len(methods) + "|",
        ]
        csv_rows: List[List[object]] = []
        for dataset in available_datasets(test_summary):
            for rate in RATES:
                row_values = [find_row(test_summary, dataset, rate, method) for method in methods]
                lines.append(
                    f"| {dataset} | {int(rate * 100)}% | "
                    + " | ".join(pm(row) for row in row_values)
                    + " |"
                )
                csv_rows.append([dataset, f"{int(rate * 100)}%"] + [pm(row, digits=10) for row in row_values])
        lines.append("")
        lines.append(
            "Note: no-completion and zero-fill can coincide when missing entries are represented by zeros. "
            "SVD may coincide with mean fill when the low-rank reconstruction is applied to a mean-initialized short sequence."
        )
        (output_dir / f"{filename}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_csv(
            output_dir / f"{filename}.csv",
            ["Dataset", "Missing Rate"] + [method_label(method) for method in methods],
            csv_rows,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-reports", nargs="+", type=Path, required=True)
    parser.add_argument("--validation-reports", nargs="+", type=Path, required=True)
    parser.add_argument("--adaptive-plans", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("revision_missing_tables"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_summary = load_summary(args.test_reports)
    validation_summary = load_summary(args.validation_reports)
    plan = load_plans(args.adaptive_plans)
    generate_table10(test_summary, plan, args.output_dir)
    generate_validation_table(validation_summary, plan, args.output_dir)
    generate_full_tables(test_summary, args.output_dir)
    print(f"Generated revision missing-completion tables in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
