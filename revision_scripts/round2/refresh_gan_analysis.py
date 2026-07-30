#!/usr/bin/env python3
"""Refresh corrected GAN comparisons with the fresh round-two noGAN runs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats


DATASETS = {
    "TaxiBJ_P1": "TaxiBJ P1",
    "TaxiBJ_P2": "TaxiBJ P2",
    "TaxiBJ_P3": "TaxiBJ P3",
    "TaxiBJ_P4": "TaxiBJ P4",
    "BikeNYC": "BikeNYC",
}
SEEDS = (2024, 2025, 2026)


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nogan",
        type=Path,
        default=Path("revision/round2/fupsi_seed_metrics.csv"),
    )
    parser.add_argument(
        "--gan-root",
        type=Path,
        default=Path("revision/gan/GANStableMainE300/raw_metrics"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision/round2/gan_analysis"),
    )
    args = parser.parse_args()

    values = {}
    seed_rows = []
    for row in read_rows(args.nogan):
        dataset_key = row["dataset_key"]
        seed = int(row["seed"])
        for metric in ("RMSE", "MAE"):
            values[("noGAN", dataset_key, seed, metric)] = float(row[metric])
        seed_rows.append(
            {
                "dataset": DATASETS[dataset_key],
                "variant": "noGAN",
                "seed": seed,
                "RMSE": f"{float(row['RMSE']):.6f}",
                "MAE": f"{float(row['MAE']):.6f}",
                "source": row["source"],
            }
        )

    for path in sorted(args.gan_root.rglob("test_metrics.csv")):
        rows = read_rows(path)
        if len(rows) != 1:
            raise ValueError(f"{path}: expected one row, found {len(rows)}")
        row = rows[0]
        prefix = "GANStableMainE300_"
        if not row["dataset"].startswith(prefix):
            continue
        dataset_key = row["dataset"].removeprefix(prefix)
        seed = int(row["seed"])
        for metric in ("RMSE", "MAE"):
            values[("GAN", dataset_key, seed, metric)] = float(row[metric])
        seed_rows.append(
            {
                "dataset": DATASETS[dataset_key],
                "variant": "GAN",
                "seed": seed,
                "RMSE": f"{float(row['RMSE']):.6f}",
                "MAE": f"{float(row['MAE']):.6f}",
                "source": path.relative_to(args.gan_root).as_posix(),
            }
        )

    expected = {
        (variant, dataset_key, seed, metric)
        for variant in ("noGAN", "GAN")
        for dataset_key in DATASETS
        for seed in SEEDS
        for metric in ("RMSE", "MAE")
    }
    if set(values) != expected:
        raise RuntimeError(
            f"Incomplete GAN comparison: missing={sorted(expected - set(values))}"
        )

    summaries = []
    tests = []
    summary_lookup = {}
    for dataset_key, paper_name in DATASETS.items():
        for metric in ("RMSE", "MAE"):
            samples = {}
            for variant in ("noGAN", "GAN"):
                array = np.asarray(
                    [
                        values[(variant, dataset_key, seed, metric)]
                        for seed in SEEDS
                    ],
                    dtype=np.float64,
                )
                samples[variant] = array
                summary_lookup[(variant, dataset_key, metric)] = (
                    float(array.mean()),
                    float(array.std(ddof=1)),
                )
                summaries.append(
                    {
                        "dataset": paper_name,
                        "variant": variant,
                        "metric": metric,
                        "runs": 3,
                        "mean": f"{array.mean():.6f}",
                        "std": f"{array.std(ddof=1):.6f}",
                    }
                )
            differences = samples["noGAN"] - samples["GAN"]
            difference_std = differences.std(ddof=1)
            t_result = stats.ttest_rel(samples["noGAN"], samples["GAN"])
            wilcoxon = stats.wilcoxon(
                samples["noGAN"], samples["GAN"], method="auto"
            )
            tests.append(
                {
                    "dataset": paper_name,
                    "metric": metric,
                    "n": 3,
                    "mean_difference_noGAN_minus_GAN": (
                        f"{differences.mean():.6f}"
                    ),
                    "paired_t_p": f"{float(t_result.pvalue):.6f}",
                    "wilcoxon_p": f"{float(wilcoxon.pvalue):.6f}",
                    "cohen_dz": (
                        f"{differences.mean() / difference_std:.6f}"
                        if difference_std > 0
                        else "--"
                    ),
                    "noGAN_lower_seed_count": int(
                        np.sum(samples["noGAN"] < samples["GAN"])
                    ),
                    "interpretation": "descriptive; n=3 has low power",
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "gan_seed_metrics.csv", seed_rows)
    write_rows(args.output_dir / "gan_mean_std.csv", summaries)
    write_rows(args.output_dir / "gan_paired_tests.csv", tests)

    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        (
            r"\caption{Optional GAN stability analysis under the corrected "
            r"MainSeed-RawCount-v2 protocol. Results are mean $\pm$ standard "
            r"deviation over three seeds; lower is better.}"
        ),
        r"\label{tab:gan-stability}",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Dataset & Metric & FUPSI-noGAN & FUPSI-GAN & Lower variant \\",
        r"\midrule",
    ]
    markdown = [
        "# Refreshed GAN Stability Audit",
        "",
        "| Dataset | Metric | noGAN | GAN | Lower |",
        "|---|---|---:|---:|---|",
    ]
    for dataset_key, paper_name in DATASETS.items():
        for metric in ("RMSE", "MAE"):
            nogan = summary_lookup[("noGAN", dataset_key, metric)]
            gan = summary_lookup[("GAN", dataset_key, metric)]
            lower = "noGAN" if nogan[0] < gan[0] else "GAN"
            label = paper_name if metric == "RMSE" else ""
            lines.append(
                f"{label} & {metric} & ${nogan[0]:.4f}\\pm{nogan[1]:.4f}$ "
                f"& ${gan[0]:.4f}\\pm{gan[1]:.4f}$ & {lower} \\\\"
            )
            markdown.append(
                f"| {paper_name} | {metric} | {nogan[0]:.4f} +/- "
                f"{nogan[1]:.4f} | {gan[0]:.4f} +/- {gan[1]:.4f} | "
                f"{lower} |"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\parbox{0.96\textwidth}{\footnotesize\textit{Note.} The "
                r"noGAN side uses the fresh round-two FUPSI rerun. The GAN "
                r"side uses the previously regenerated corrected BCE-GAN "
                r"runs. Tests are descriptive because $n=3$.}"
            ),
            r"\end{table*}",
        ]
    )
    markdown.extend(
        [
            "",
            "The noGAN rows come from the fresh round-two rerun. The GAN rows "
            "come from the corrected residual/BCE-GAN experiment. Statistical "
            "tests are descriptive because n=3 has low inferential power.",
        ]
    )
    (args.output_dir / "gan_stability_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (args.output_dir / "gan_stability_audit.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print("Validated 30 GAN/noGAN seed rows and wrote refreshed analysis.")


if __name__ == "__main__":
    main()
