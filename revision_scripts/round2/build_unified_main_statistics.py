#!/usr/bin/env python3
"""Build unified main-result and descriptive paired-test tables."""

from __future__ import annotations

import argparse
import csv
import math
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
METHOD_ORDER = {
    "FUPSI": 0,
    "HRSTT (reimplementation)": 1,
    "UrbanFM": 2,
    "FODE": 3,
    "HA-Mean": 4,
}
SEEDS = (2024, 2025, 2026)
EXPECTED_METHODS = {
    "TaxiBJ P1": {
        "FUPSI",
        "HRSTT (reimplementation)",
        "UrbanFM",
        "FODE",
        "HA-Mean",
    },
    "TaxiBJ P2": {
        "FUPSI",
        "HRSTT (reimplementation)",
        "UrbanFM",
        "FODE",
        "HA-Mean",
    },
    "TaxiBJ P3": {
        "FUPSI",
        "HRSTT (reimplementation)",
        "UrbanFM",
        "FODE",
        "HA-Mean",
    },
    "TaxiBJ P4": {
        "FUPSI",
        "HRSTT (reimplementation)",
        "UrbanFM",
        "FODE",
        "HA-Mean",
    },
    "BikeNYC": {
        "FUPSI",
        "HRSTT (reimplementation)",
        "UrbanFM",
        "FODE",
        "HA-Mean",
    },
}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_rows(rows: list[dict]) -> list[dict]:
    normalized = []
    for row in rows:
        method = row.get("method", "")
        if method == "HRSTT_reimplementation":
            method = "HRSTT (reimplementation)"
        dataset = row.get("paper_dataset") or row.get("dataset")
        normalized_row = {
            "dataset": dataset,
            "method": method,
            "seed": int(row["seed"]),
            "RMSE": float(row["RMSE"]),
            "MAE": float(row["MAE"]),
        }
        if not all(
            math.isfinite(normalized_row[metric]) and normalized_row[metric] >= 0
            for metric in ("RMSE", "MAE")
        ):
            raise ValueError(f"Invalid metric row: {normalized_row}")
        normalized.append(normalized_row)
    return normalized


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def format_mean_std(mean: float, std: float, rank: int) -> str:
    value = rf"${mean:.4f}\pm{std:.4f}$"
    if rank == 0:
        return rf"\textbf{{{value}}}"
    if rank == 1:
        return rf"\underline{{{value}}}"
    return value


def format_p_latex(value: float) -> str:
    return r"$<0.0001$" if value < 0.0001 else f"${value:.4f}$"


def validate_exact_result_matrix(rows: list[dict]) -> None:
    expected = {
        (dataset, method, seed)
        for dataset, methods in EXPECTED_METHODS.items()
        for method in methods
        for seed in SEEDS
    }
    discovered = {
        (row["dataset"], row["method"], row["seed"]) for row in rows
    }
    if discovered != expected:
        raise ValueError(
            "Unified result matrix mismatch: "
            f"missing={sorted(expected - discovered)}, "
            f"extra={sorted(discovered - expected)}"
        )
    if len(rows) != len(expected):
        raise ValueError(
            f"Expected exactly {len(expected)} unique seed rows, found {len(rows)}."
        )


def write_main_latex(path: Path, mean_std_rows: list[dict]) -> None:
    grouped: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    for row in mean_std_rows:
        grouped.setdefault((row["dataset"], row["method"]), {})[
            row["metric"]
        ] = (float(row["mean"]), float(row["std"]))

    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        (
            r"\caption{Unified raw-count main results under the "
            r"MainSeed-RawCount-v2 protocol. Results are mean $\pm$ standard "
            r"deviation over seeds 2024, 2025, and 2026; lower is better.}"
        ),
        r"\label{tab:unified-main-results}",
        r"\small",
        r"\setlength{\tabcolsep}{8pt}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Dataset & Method & RMSE & MAE \\",
        r"\midrule",
    ]
    datasets = sorted(
        {dataset for dataset, _method in grouped},
        key=lambda value: (DATASET_ORDER.get(value, 999), value),
    )
    for dataset_index, dataset in enumerate(datasets):
        methods = sorted(
            [
                method
                for candidate_dataset, method in grouped
                if candidate_dataset == dataset
                and {"RMSE", "MAE"}.issubset(grouped[(dataset, method)])
            ],
            key=lambda value: (METHOD_ORDER.get(value, 999), value),
        )
        ranks: dict[str, dict[str, int]] = {"RMSE": {}, "MAE": {}}
        for metric in ("RMSE", "MAE"):
            unique_values = sorted(
                {
                    round(grouped[(dataset, method)][metric][0], 12)
                    for method in methods
                }
            )
            ranks[metric] = {
                method: unique_values.index(
                    round(grouped[(dataset, method)][metric][0], 12)
                )
                for method in methods
            }
        for method in methods:
            rmse_mean, rmse_std = grouped[(dataset, method)]["RMSE"]
            mae_mean, mae_std = grouped[(dataset, method)]["MAE"]
            lines.append(
                " & ".join(
                    [
                        latex_escape(dataset),
                        latex_escape(method),
                        format_mean_std(
                            rmse_mean, rmse_std, ranks["RMSE"][method]
                        ),
                        format_mean_std(
                            mae_mean, mae_std, ranks["MAE"][method]
                        ),
                    ]
                )
                + r" \\"
            )
        if dataset_index != len(datasets) - 1:
            lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\parbox{0.96\textwidth}{\footnotesize\textit{Note.} "
                r"HRSTT is a documented reimplementation because no verified "
                r"official implementation was identified. Super-resolution "
                r"baselines receive the same seed-matched FUPSI-predicted "
                r"coarse maps. Best results are bold and second-best results "
                r"are underlined.}"
            ),
            r"\end{table*}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hrstt_statistics_latex(path: Path, paired_rows: list[dict]) -> None:
    rows = [
        row
        for row in paired_rows
        if row["comparison"] == "FUPSI vs HRSTT (reimplementation)"
    ]
    rows.sort(
        key=lambda row: (
            DATASET_ORDER.get(row["dataset"], 999),
            0 if row["metric"] == "RMSE" else 1,
        )
    )
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        (
            r"\caption{Descriptive paired comparison with HRSTT "
            r"(reimplementation). Differences are FUPSI minus HRSTT; negative "
            r"values favor FUPSI. Statistical evidence is interpreted "
            r"cautiously because only three paired seeds are available.}"
        ),
        r"\label{tab:hrstt-paired-statistics}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        (
            r"Dataset & Metric & Mean diff. & $t$-test $p$ & Wilcoxon $p$ "
            r"& Cohen's $d_z$ & FUPSI lower & $n$ \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        dz = float(row["cohen_dz"])
        dz_text = f"{dz:.3f}" if math.isfinite(dz) else "--"
        lines.append(
            " & ".join(
                [
                    latex_escape(row["dataset"]),
                    row["metric"],
                    f"{float(row['mean_difference_FUPSI_minus_baseline']):.4f}",
                    format_p_latex(float(row["paired_t_p"])),
                    format_p_latex(float(row["wilcoxon_p"])),
                    dz_text,
                    f"{row['FUPSI_lower_seed_count']}/{row['n']}",
                    str(row["n"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--existing",
        type=Path,
        default=None,
        help="Optional legacy/extra seed-level baseline CSV.",
    )
    parser.add_argument(
        "--sr",
        type=Path,
        default=Path(
            "revision/round2/sr_baselines/summary/"
            "sr_baseline_seed_metrics.csv"
        ),
    )
    parser.add_argument(
        "--hamean",
        type=Path,
        default=Path("revision/round2/hamean_seed_metrics.csv"),
    )
    parser.add_argument(
        "--fupsi",
        type=Path,
        default=Path("revision/round2/fupsi_seed_metrics.csv"),
    )
    parser.add_argument(
        "--hrstt",
        type=Path,
        default=Path("revision/round2/hrstt/summary/hrstt_seed_metrics.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision/round2/main_statistics"),
    )
    args = parser.parse_args()

    rows = []
    for path in (args.existing, args.sr, args.hamean):
        if path is not None:
            rows.extend(normalize_rows(read_csv(path)))
    current_fupsi = normalize_rows(read_csv(args.fupsi))
    for row in current_fupsi:
        row["method"] = "FUPSI"
    rows.extend(current_fupsi)
    rows.extend(normalize_rows(read_csv(args.hrstt)))
    if not rows:
        raise FileNotFoundError("No seed-level main results were found.")

    deduplicated = {}
    for row in rows:
        key = (row["dataset"], row["method"], row["seed"])
        if key in deduplicated:
            raise ValueError(f"Duplicate unified result row: {key}")
        deduplicated[key] = row
    rows = list(deduplicated.values())
    validate_exact_result_matrix(rows)
    write_csv(
        args.output_dir / "unified_seed_metrics.csv",
        sorted(rows, key=lambda row: (row["dataset"], row["method"], row["seed"])),
        ["dataset", "method", "seed", "RMSE", "MAE"],
    )

    mean_std_rows = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for method in sorted(
            {row["method"] for row in rows if row["dataset"] == dataset}
        ):
            subset = [
                row
                for row in rows
                if row["dataset"] == dataset and row["method"] == method
            ]
            for metric in ("RMSE", "MAE"):
                values = np.array(
                    [row[metric] for row in subset], dtype=np.float64
                )
                mean_std_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        "runs": len(values),
                        "mean": f"{values.mean():.6f}",
                        "std": f"{values.std(ddof=1) if len(values) > 1 else 0.0:.6f}",
                        "mean_std": (
                            f"{values.mean():.4f} +/- "
                            f"{values.std(ddof=1) if len(values) > 1 else 0.0:.4f}"
                        ),
                    }
                )
    write_csv(
        args.output_dir / "unified_mean_std.csv",
        mean_std_rows,
        ["dataset", "method", "metric", "runs", "mean", "std", "mean_std"],
    )
    write_main_latex(
        args.output_dir / "unified_main_results_table.tex", mean_std_rows
    )

    gain_rows = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for metric in ("RMSE", "MAE"):
            summaries = [
                row
                for row in mean_std_rows
                if row["dataset"] == dataset and row["metric"] == metric
            ]
            fupsi = next(
                (row for row in summaries if row["method"] == "FUPSI"), None
            )
            baselines = [
                row for row in summaries if row["method"] != "FUPSI"
            ]
            if fupsi is None or not baselines:
                continue
            best = min(baselines, key=lambda row: float(row["mean"]))
            fupsi_mean = float(fupsi["mean"])
            baseline_mean = float(best["mean"])
            absolute_gain = baseline_mean - fupsi_mean
            relative_gain = (
                100.0 * absolute_gain / baseline_mean
                if baseline_mean != 0
                else float("nan")
            )
            gain_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "best_baseline": best["method"],
                    "FUPSI_mean": f"{fupsi_mean:.6f}",
                    "best_baseline_mean": f"{baseline_mean:.6f}",
                    "absolute_gain_baseline_minus_FUPSI": (
                        f"{absolute_gain:.6f}"
                    ),
                    "relative_gain_percent": f"{relative_gain:.4f}",
                    "FUPSI_is_better": str(absolute_gain > 0).lower(),
                }
            )
    write_csv(
        args.output_dir / "unified_absolute_relative_gains.csv",
        gain_rows,
        [
            "dataset",
            "metric",
            "best_baseline",
            "FUPSI_mean",
            "best_baseline_mean",
            "absolute_gain_baseline_minus_FUPSI",
            "relative_gain_percent",
            "FUPSI_is_better",
        ],
    )

    paired_rows = []
    for dataset in sorted({row["dataset"] for row in rows}):
        fupsi = {
            row["seed"]: row
            for row in rows
            if row["dataset"] == dataset and row["method"] == "FUPSI"
        }
        methods = sorted(
            {
                row["method"]
                for row in rows
                if row["dataset"] == dataset and row["method"] != "FUPSI"
            }
        )
        for method in methods:
            baseline = {
                row["seed"]: row
                for row in rows
                if row["dataset"] == dataset and row["method"] == method
            }
            seeds = sorted(set(fupsi) & set(baseline))
            if len(seeds) < 2:
                continue
            for metric in ("RMSE", "MAE"):
                fupsi_values = np.array(
                    [fupsi[seed][metric] for seed in seeds], dtype=np.float64
                )
                baseline_values = np.array(
                    [baseline[seed][metric] for seed in seeds], dtype=np.float64
                )
                differences = fupsi_values - baseline_values
                t_result = stats.ttest_rel(fupsi_values, baseline_values)
                difference_std = differences.std(ddof=1)
                cohen_dz = (
                    differences.mean() / difference_std
                    if difference_std > 0
                    else math.copysign(float("inf"), differences.mean())
                    if differences.mean() != 0
                    else 0.0
                )
                if np.allclose(differences, 0):
                    wilcoxon_statistic, wilcoxon_p = 0.0, 1.0
                else:
                    wilcoxon = stats.wilcoxon(
                        fupsi_values,
                        baseline_values,
                        alternative="two-sided",
                        method="auto",
                    )
                    wilcoxon_statistic = float(wilcoxon.statistic)
                    wilcoxon_p = float(wilcoxon.pvalue)
                paired_rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "comparison": f"FUPSI vs {method}",
                        "paired_seeds": ",".join(str(seed) for seed in seeds),
                        "n": len(seeds),
                        "FUPSI_mean": f"{fupsi_values.mean():.6f}",
                        "baseline_mean": f"{baseline_values.mean():.6f}",
                        "mean_difference_FUPSI_minus_baseline": f"{differences.mean():.6f}",
                        "FUPSI_lower_seed_count": int(
                            np.sum(fupsi_values < baseline_values)
                        ),
                        "paired_t_statistic": f"{float(t_result.statistic):.6f}",
                        "paired_t_p": f"{float(t_result.pvalue):.6f}",
                        "cohen_dz": (
                            f"{cohen_dz:.6f}"
                            if math.isfinite(cohen_dz)
                            else str(cohen_dz)
                        ),
                        "wilcoxon_statistic": f"{wilcoxon_statistic:.6f}",
                        "wilcoxon_p": f"{wilcoxon_p:.6f}",
                        "interpretation": "descriptive; n=3 has low inferential power",
                    }
                )
    fields = list(paired_rows[0]) if paired_rows else []
    if paired_rows:
        write_csv(
            args.output_dir / "unified_paired_tests.csv", paired_rows, fields
        )
        write_hrstt_statistics_latex(
            args.output_dir / "hrstt_paired_statistics_table.tex", paired_rows
        )
    print(
        f"Wrote {len(rows)} seed rows, {len(mean_std_rows)} summaries, "
        f"and {len(paired_rows)} paired comparisons."
    )


if __name__ == "__main__":
    main()
