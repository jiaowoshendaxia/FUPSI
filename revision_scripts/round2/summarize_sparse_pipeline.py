#!/usr/bin/env python3
"""Audit and summarize end-to-end sparse-input pipeline results."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


DATASET_ORDER = {
    "TaxiBJ P1": 0,
    "TaxiBJ P2": 1,
    "TaxiBJ P3": 2,
    "TaxiBJ P4": 3,
    "BikeNYC": 4,
}
EXPECTED_DATASETS = tuple(DATASET_ORDER)
EXPECTED_RATES = (0.0, 0.1, 0.3, 0.5, 0.7)
EXPECTED_METHODS = ("adaptive", "no_completion")
EXPECTED_SEEDS = (2024, 2025, 2026)
VALID_COMPLETION_METHODS = {
    "no_completion",
    "zero_fill",
    "mean_fill",
    "linear_interpolation",
    "knn_fill",
    "svd_completion",
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


def normalized_rows(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        method = row.get("completion") or row.get("method")
        model_seed = int(row.get("model_seed") or row["seed"])
        mask_seed = int(row.get("mask_seed") or row["seed"])
        normalized = {
            "dataset": row["dataset"],
            "missing_rate": float(row["missing_rate"]),
            "completion": method,
            "completion_method": row.get("completion_method", method),
            "model_seed": model_seed,
            "mask_seed": mask_seed,
            "fine_RMSE": float(row.get("fine_RMSE") or row["RMSE"]),
            "fine_MAE": float(row.get("fine_MAE") or row["MAE"]),
            "completion_MSE": float(row["completion_MSE"]),
        }
        if not all(
            math.isfinite(normalized[metric]) and normalized[metric] >= 0
            for metric in ("fine_RMSE", "fine_MAE", "completion_MSE")
        ):
            raise ValueError(f"Invalid sparse metric row: {normalized}")
        output.append(normalized)
    return output


def audit(rows: list[dict], expected_rows: int) -> None:
    canonical_expected_rows = (
        len(EXPECTED_DATASETS)
        * len(EXPECTED_RATES)
        * len(EXPECTED_METHODS)
        * len(EXPECTED_SEEDS)
    )
    if expected_rows != canonical_expected_rows:
        raise ValueError(
            f"Expected-row override must equal the formal protocol size "
            f"{canonical_expected_rows}, received {expected_rows}."
        )
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(rows)}")
    keys = {
        (
            row["dataset"],
            round(row["missing_rate"], 10),
            row["completion"],
            row["model_seed"],
            row["mask_seed"],
        )
        for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("Duplicate dataset/rate/completion/seed rows detected")
    expected_keys = {
        (dataset, rate, method, seed, seed)
        for dataset in EXPECTED_DATASETS
        for rate in EXPECTED_RATES
        for method in EXPECTED_METHODS
        for seed in EXPECTED_SEEDS
    }
    if keys != expected_keys:
        raise ValueError(
            "Sparse result matrix mismatch: "
            f"missing={sorted(expected_keys - keys)}, "
            f"extra={sorted(keys - expected_keys)}"
        )
    if any(row["model_seed"] != row["mask_seed"] for row in rows):
        raise ValueError("Formal run expects matched model and mask seeds")
    for row in rows:
        if row["completion_method"] not in VALID_COMPLETION_METHODS:
            raise ValueError(
                f"Unknown selected completion method: {row['completion_method']}"
            )
        if (
            row["completion"] == "no_completion"
            and row["completion_method"] != "no_completion"
        ):
            raise ValueError("No-completion row used a different operator")
        if (
            math.isclose(row["missing_rate"], 0.0, abs_tol=1e-12)
            and row["completion_method"] != "no_completion"
        ):
            raise ValueError("Zero-missing rows must use no completion")


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        for metric in ("fine_RMSE", "fine_MAE", "completion_MSE"):
            groups[
                (
                    row["dataset"],
                    row["missing_rate"],
                    row["completion"],
                    metric,
                )
            ].append(row[metric])
    output = []
    for (dataset, rate, method, metric), values in sorted(
        groups.items(),
        key=lambda item: (
            DATASET_ORDER.get(item[0][0], 999),
            item[0][1],
            item[0][2],
            item[0][3],
        ),
    ):
        vector = np.asarray(values, dtype=np.float64)
        output.append(
            {
                "dataset": dataset,
                "missing_rate": rate,
                "completion": method,
                "metric": metric,
                "runs": len(vector),
                "mean": f"{vector.mean():.6f}",
                "std": f"{vector.std(ddof=1):.6f}",
            }
        )
    return output


def paired_tests(rows: list[dict]) -> list[dict]:
    output = []
    datasets = sorted(
        {row["dataset"] for row in rows},
        key=lambda value: DATASET_ORDER.get(value, 999),
    )
    for dataset in datasets:
        rates = sorted(
            {
                row["missing_rate"]
                for row in rows
                if row["dataset"] == dataset
            }
        )
        for rate in rates:
            by_method = {}
            for method in ("adaptive", "no_completion"):
                by_method[method] = {
                    (row["model_seed"], row["mask_seed"]): row
                    for row in rows
                    if row["dataset"] == dataset
                    and row["missing_rate"] == rate
                    and row["completion"] == method
                }
            pairs = sorted(
                set(by_method["adaptive"]) & set(by_method["no_completion"])
            )
            if len(pairs) != 3:
                raise ValueError(
                    f"{dataset} rate={rate}: expected 3 matched pairs, "
                    f"found {len(pairs)}"
                )
            for metric in ("fine_RMSE", "fine_MAE"):
                adaptive = np.asarray(
                    [by_method["adaptive"][pair][metric] for pair in pairs],
                    dtype=np.float64,
                )
                no_completion = np.asarray(
                    [
                        by_method["no_completion"][pair][metric]
                        for pair in pairs
                    ],
                    dtype=np.float64,
                )
                differences = adaptive - no_completion
                if np.allclose(differences, 0):
                    t_statistic, t_p = 0.0, 1.0
                    wilcoxon_statistic, wilcoxon_p = 0.0, 1.0
                    cohen_dz = 0.0
                else:
                    t_result = stats.ttest_rel(adaptive, no_completion)
                    t_statistic = float(t_result.statistic)
                    t_p = float(t_result.pvalue)
                    wilcoxon = stats.wilcoxon(
                        adaptive,
                        no_completion,
                        alternative="two-sided",
                        method="auto",
                    )
                    wilcoxon_statistic = float(wilcoxon.statistic)
                    wilcoxon_p = float(wilcoxon.pvalue)
                    difference_std = differences.std(ddof=1)
                    cohen_dz = (
                        float(differences.mean() / difference_std)
                        if difference_std > 0
                        else math.copysign(float("inf"), differences.mean())
                    )
                baseline_mean = float(no_completion.mean())
                improvement = (
                    100.0
                    * (baseline_mean - float(adaptive.mean()))
                    / baseline_mean
                    if baseline_mean != 0
                    else float("nan")
                )
                output.append(
                    {
                        "dataset": dataset,
                        "missing_rate": rate,
                        "metric": metric,
                        "n": len(pairs),
                        "adaptive_mean": f"{adaptive.mean():.6f}",
                        "no_completion_mean": f"{no_completion.mean():.6f}",
                        "mean_difference_adaptive_minus_no_completion": (
                            f"{differences.mean():.6f}"
                        ),
                        "relative_improvement_percent": f"{improvement:.4f}",
                        "adaptive_lower_seed_count": int(
                            np.sum(adaptive < no_completion)
                        ),
                        "paired_t_statistic": f"{t_statistic:.6f}",
                        "paired_t_p": f"{t_p:.6f}",
                        "wilcoxon_statistic": f"{wilcoxon_statistic:.6f}",
                        "wilcoxon_p": f"{wilcoxon_p:.6f}",
                        "cohen_dz": (
                            f"{cohen_dz:.6f}"
                            if math.isfinite(cohen_dz)
                            else str(cohen_dz)
                        ),
                        "interpretation": "descriptive; n=3 has low power",
                    }
                )
    return output


def mean_std(summary: list[dict], dataset, rate, method, metric) -> str:
    row = next(
        item
        for item in summary
        if item["dataset"] == dataset
        and item["missing_rate"] == rate
        and item["completion"] == method
        and item["metric"] == metric
    )
    return f"${float(row['mean']):.3f}\\pm{float(row['std']):.3f}$"


def write_latex(path: Path, rows: list[dict], summary: list[dict]) -> None:
    datasets = sorted(
        {row["dataset"] for row in rows},
        key=lambda value: DATASET_ORDER.get(value, 999),
    )
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        (
            r"\caption{End-to-end sparse-input evaluation under the unified "
            r"MainSeed-RawCount-v2 protocol. Results are mean $\pm$ standard "
            r"deviation over three matched model/mask seeds; lower is better.}"
        ),
        r"\label{tab:sparse-end-to-end}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        (
            r"Dataset & Missing & No completion & Adaptive & "
            r"No completion & Adaptive \\"
        ),
            r" & rate & RMSE & RMSE & MAE & MAE \\",
        r"\midrule",
    ]
    for dataset in datasets:
        rates = sorted(
            {
                row["missing_rate"]
                for row in rows
                if row["dataset"] == dataset
            }
        )
        for rate in rates:
            adaptive_methods = {
                row["completion_method"]
                for row in rows
                if row["dataset"] == dataset
                and row["missing_rate"] == rate
                and row["completion"] == "adaptive"
            }
            selection = "/".join(sorted(adaptive_methods)).replace("_", r"\_")
            no_rmse = mean_std(
                summary, dataset, rate, "no_completion", "fine_RMSE"
            )
            adaptive_rmse = mean_std(
                summary, dataset, rate, "adaptive", "fine_RMSE"
            )
            no_mae = mean_std(
                summary, dataset, rate, "no_completion", "fine_MAE"
            )
            adaptive_mae = mean_std(
                summary, dataset, rate, "adaptive", "fine_MAE"
            )
            lines.append(
                f"{dataset} & {100 * rate:.0f}\\% & {no_rmse} & "
                f"{adaptive_rmse} & {no_mae} & {adaptive_mae} \\\\"
            )
        if dataset != datasets[-1]:
            lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\parbox{\textwidth}{\footnotesize\textit{Note.} "
                r"Completion is causal: no operator uses an observation after "
                r"the time being filled. Final RMSE/MAE are computed on "
                r"raw-count future fine-grained maps. Completion MSE, the "
                r"validation-fixed operator selected for each setting, and "
                r"all seed-level values are provided in the accompanying "
                r"result artifact. Statistical tests are descriptive because "
                r"only three matched seeds are available.}"
            ),
            r"\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(
    path: Path, rows: list[dict], tests: list[dict], expected_rows: int
) -> None:
    nonzero = [
        row for row in tests if float(row["missing_rate"]) > 0
    ]
    improved = sum(
        float(row["mean_difference_adaptive_minus_no_completion"]) < 0
        for row in nonzero
    )
    lines = [
        "# End-to-End Sparse-Input Audit",
        "",
        f"- Seed-level rows: {len(rows)}/{expected_rows}.",
        "- Protocol: MainSeed-RawCount-v2.",
        "- Methods: adaptive completion and no completion.",
        "- Missing rates: 0%, 10%, 30%, 50%, and 70%.",
        "- Seeds: 2024, 2025, and 2026, matched for model and mask.",
        "- Completion is causal and never uses future test observations.",
        (
            f"- Adaptive completion has a lower mean in {improved}/"
            f"{len(nonzero)} nonzero-rate dataset-metric comparisons."
        ),
        "",
        (
            "Paired t-tests, Wilcoxon tests, Cohen's d_z, and seed-direction "
            "counts are reported as descriptive evidence because n=3 has low "
            "inferential power."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "revision/round2/sparse_pipeline/sparse_pipeline_seed_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision/round2/sparse_pipeline/analysis"),
    )
    parser.add_argument("--expected-rows", type=int, default=150)
    args = parser.parse_args()

    rows = normalized_rows(read_rows(args.input))
    audit(rows, args.expected_rows)
    summary = summarize(rows)
    tests = paired_tests(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(
        args.output_dir / "sparse_end_to_end_seed_metrics.csv",
        rows,
        list(rows[0]),
    )
    write_rows(
        args.output_dir / "sparse_end_to_end_mean_std.csv",
        summary,
        list(summary[0]),
    )
    write_rows(
        args.output_dir / "sparse_end_to_end_paired_tests.csv",
        tests,
        list(tests[0]),
    )
    write_latex(
        args.output_dir / "sparse_end_to_end_table.tex", rows, summary
    )
    write_markdown(
        args.output_dir / "sparse_end_to_end_audit.md",
        rows,
        tests,
        args.expected_rows,
    )
    print(
        f"Validated {len(rows)} rows; wrote {len(summary)} summaries and "
        f"{len(tests)} paired tests to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
