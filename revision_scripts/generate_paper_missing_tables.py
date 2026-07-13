# -- coding:utf-8 --
"""Generate paper-ready missing-completion tables from experiment JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


DATASETS = ["TaxiBJ_P1", "TaxiBJ_P2", "TaxiBJ_P3", "TaxiBJ_P4", "BikeNYC"]
RATES = [0.1, 0.3, 0.5, 0.7]
METHODS = [
    "no_completion",
    "zero_fill",
    "mean_fill",
    "linear_interpolation",
    "knn_fill",
    "svd_completion",
    "adaptive_completion",
]
SINGLE_BASELINES = [
    "no_completion",
    "zero_fill",
    "mean_fill",
    "linear_interpolation",
    "knn_fill",
    "svd_completion",
]


def load_summary(path: Path) -> List[Dict[str, object]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return report["summary"]


def find_row(summary: List[Dict[str, object]], dataset: str, rate: float, method: str) -> Dict[str, object]:
    for row in summary:
        if (
            row["dataset"] == dataset
            and abs(float(row["missing_rate"]) - rate) < 1e-9
            and row["method"] == method
        ):
            return row
    raise KeyError((dataset, rate, method))


def pm(row: Dict[str, object], metric: str = "completion_MSE", digits: int = 6) -> str:
    mean = row[f"{metric}_mean"]
    std = row[f"{metric}_std"]
    if mean is None:
        return "N/A"
    return f"{float(mean):.{digits}f} ± {float(std):.{digits}f}"


def method_label(method: str) -> str:
    labels = {
        "no_completion": "No completion",
        "zero_fill": "Zero fill",
        "mean_fill": "Mean fill",
        "linear_interpolation": "Linear interpolation",
        "knn_fill": "KNN fill",
        "svd_completion": "SVD completion",
        "adaptive_completion": "Adaptive completion",
    }
    return labels.get(method, method)


def generate_table10(test_summary: List[Dict[str, object]], output_dir: Path) -> None:
    lines = [
        "# Table 10. Adaptive Completion vs. Best Single Baseline",
        "",
        "Metric: Completion MSE on missing entries, reported as mean ± std over three random seeds.",
        "",
        "| Dataset | Missing Rate | Best Single Baseline | Best MSE | Adaptive MSE | Adaptive / Best |",
        "|---|---:|---|---:|---:|---:|",
    ]
    csv_lines = [
        "Dataset,Missing Rate,Best Single Baseline,Best MSE Mean,Best MSE Std,Adaptive MSE Mean,Adaptive MSE Std,Adaptive / Best"
    ]
    for dataset in DATASETS:
        for rate in RATES:
            baseline_rows = [
                find_row(test_summary, dataset, rate, method) for method in SINGLE_BASELINES
            ]
            best = min(baseline_rows, key=lambda row: float(row["completion_MSE_mean"]))
            adaptive = find_row(test_summary, dataset, rate, "adaptive_completion")
            ratio = float(adaptive["completion_MSE_mean"]) / float(best["completion_MSE_mean"])
            lines.append(
                f"| {dataset} | {int(rate * 100)}% | {method_label(str(best['method']))} | "
                f"{pm(best)} | {pm(adaptive)} | {ratio:.3f} |"
            )
            csv_lines.append(
                ",".join(
                    [
                        dataset,
                        f"{int(rate * 100)}%",
                        method_label(str(best["method"])),
                        f"{float(best['completion_MSE_mean']):.10f}",
                        f"{float(best['completion_MSE_std']):.10f}",
                        f"{float(adaptive['completion_MSE_mean']):.10f}",
                        f"{float(adaptive['completion_MSE_std']):.10f}",
                        f"{ratio:.6f}",
                    ]
                )
            )

    (output_dir / "table10_adaptive_vs_best.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output_dir / "table10_adaptive_vs_best.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )


def generate_appendix_a1(test_summary: List[Dict[str, object]], output_dir: Path) -> None:
    lines = [
        "# Appendix Table A1. Full Missing-Completion Results",
        "",
        "Metric: Completion MSE on missing entries, reported as mean ± std over three random seeds.",
        "",
        "| Dataset | Missing Rate | No completion | Zero fill | Mean fill | Linear interpolation | KNN fill | SVD completion | Adaptive completion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    csv_lines = [
        "Dataset,Missing Rate,"
        + ",".join(method_label(method) for method in METHODS)
    ]
    for dataset in DATASETS:
        for rate in RATES:
            rows = [find_row(test_summary, dataset, rate, method) for method in METHODS]
            lines.append(
                f"| {dataset} | {int(rate * 100)}% | "
                + " | ".join(pm(row) for row in rows)
                + " |"
            )
            csv_lines.append(
                ",".join(
                    [dataset, f"{int(rate * 100)}%"]
                    + [
                        f"{float(row['completion_MSE_mean']):.10f} ± {float(row['completion_MSE_std']):.10f}"
                        for row in rows
                    ]
                )
            )
    (output_dir / "appendix_table_a1_full_completion_mse.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output_dir / "appendix_table_a1_full_completion_mse.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )


def generate_validation_rule_table(validation_summary: List[Dict[str, object]], output_dir: Path) -> None:
    lines = [
        "# Validation Rule Check for Adaptive Completion",
        "",
        "The selection rule and thresholds are determined on the validation split and then fixed for all test evaluations.",
        "",
        "Fixed rule: if missing rate <= 60%, use linear interpolation; otherwise use KNN when grid size <= 800 and mean fill when grid size > 800.",
        "",
        "| Dataset | Grid Size | Missing Rate | Validation Best Baseline | Validation Best MSE | Adaptive MSE | Adaptive / Best |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    grid_sizes = {
        "TaxiBJ_P1": 32 * 32,
        "TaxiBJ_P2": 32 * 32,
        "TaxiBJ_P3": 32 * 32,
        "TaxiBJ_P4": 32 * 32,
        "BikeNYC": 16 * 8,
    }
    for dataset in DATASETS:
        for rate in RATES:
            baseline_rows = [
                find_row(validation_summary, dataset, rate, method)
                for method in SINGLE_BASELINES
            ]
            best = min(baseline_rows, key=lambda row: float(row["completion_MSE_mean"]))
            adaptive = find_row(validation_summary, dataset, rate, "adaptive_completion")
            ratio = float(adaptive["completion_MSE_mean"]) / float(best["completion_MSE_mean"])
            lines.append(
                f"| {dataset} | {grid_sizes[dataset]} | {int(rate * 100)}% | "
                f"{method_label(str(best['method']))} | {pm(best)} | {pm(adaptive)} | {ratio:.3f} |"
            )
    (output_dir / "validation_adaptive_rule_check.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    output_dir = Path("paper_missing_tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    test_summary = load_summary(
        Path("adaptive_missing_rate_results_full/expanded_missing_rate_report.json")
    )
    validation_summary = load_summary(
        Path("adaptive_validation_results_full/expanded_missing_rate_report.json")
    )
    generate_table10(test_summary, output_dir)
    generate_appendix_a1(test_summary, output_dir)
    generate_validation_rule_table(validation_summary, output_dir)
    print(f"Generated tables in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
