#!/usr/bin/env python3
"""Compare corrected BCE-GAN training with the audited noGAN runs."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


WORKSPACE = Path(__file__).resolve().parents[1]
GAN_ROOT = WORKSPACE / "revision" / "gan" / "GANStableMainE300"
if not GAN_ROOT.exists():
    GAN_ROOT = WORKSPACE / "results" / "gan"
OUTPUT = GAN_ROOT / "analysis"
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
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical_gan_dataset(alias: str) -> str:
    prefix = "GANStableMainE300_"
    if not alias.startswith(prefix):
        raise ValueError(alias)
    dataset = "MainSeed_" + alias.removeprefix(prefix)
    if dataset not in PAPER_NAMES:
        raise ValueError(dataset)
    return dataset


def load_seed_metrics() -> tuple[list[dict[str, object]], dict[tuple[str, str, int, str], float]]:
    records: list[dict[str, object]] = []
    values: dict[tuple[str, str, int, str], float] = {}

    nogan_path = (
        WORKSPACE / "revision" / "statistics" / "ResidualMainE300P5" / "analysis"
        / "residual_main_seed_metrics.csv"
    )
    if not nogan_path.exists():
        nogan_path = WORKSPACE / "results" / "main" / "corrected_seed_metrics.csv"
    for raw in read_csv(nogan_path):
        if "method" in raw and raw["method"] not in {"FUPSI", "FUPSI-residual"}:
            continue
        dataset = raw["dataset"]
        seed = int(raw["seed"])
        row: dict[str, object] = {
            "dataset": dataset, "paper_dataset": PAPER_NAMES[dataset],
            "variant": "noGAN", "seed": seed,
            "source": raw.get("source", "included anonymous result evidence"),
        }
        for metric in METRICS:
            value = float(raw[metric])
            row[metric] = value
            values[("noGAN", dataset, seed, metric)] = value
        records.append(row)

    for path in sorted((GAN_ROOT / "raw_metrics").rglob("test_metrics.csv")):
        source = read_csv(path)
        if len(source) != 1:
            raise ValueError(f"Expected one metric row in {path}")
        raw = source[0]
        dataset = canonical_gan_dataset(raw["dataset"])
        seed = int(raw["seed"])
        row = {
            "dataset": dataset, "paper_dataset": PAPER_NAMES[dataset],
            "variant": "GAN", "seed": seed, "source": str(path.relative_to(WORKSPACE)),
        }
        for metric in METRICS:
            value = float(raw[metric])
            row[metric] = value
            values[("GAN", dataset, seed, metric)] = value
        records.append(row)

    expected = {
        (variant, dataset, seed)
        for variant in ("GAN", "noGAN") for dataset in PAPER_NAMES for seed in SEEDS
    }
    found = {(str(row["variant"]), str(row["dataset"]), int(row["seed"])) for row in records}
    if found != expected:
        raise RuntimeError(f"Incomplete GAN evidence: missing={sorted(expected-found)}, extra={sorted(found-expected)}")
    records.sort(key=lambda row: (str(row["dataset"]), str(row["variant"]), int(row["seed"])))
    return records, values


def summarize_metrics(values: dict[tuple[str, str, int, str], float]) -> tuple[list[dict[str, object]], dict[tuple[str, str, str], tuple[float, float]]]:
    rows: list[dict[str, object]] = []
    lookup: dict[tuple[str, str, str], tuple[float, float]] = {}
    for dataset in PAPER_NAMES:
        for variant in ("noGAN", "GAN"):
            for metric in METRICS:
                sample = np.asarray([values[(variant, dataset, seed, metric)] for seed in SEEDS])
                mean, std = float(sample.mean()), float(sample.std(ddof=1))
                lookup[(variant, dataset, metric)] = (mean, std)
                rows.append({
                    "dataset": dataset, "paper_dataset": PAPER_NAMES[dataset],
                    "variant": variant, "metric": metric, "runs": 3,
                    "mean": f"{mean:.10f}", "std": f"{std:.10f}",
                    "cv_percent": f"{std/mean*100:.6f}" if mean else "",
                    "mean_std": f"{mean:.4f} +/- {std:.4f}",
                })
    return rows, lookup


def paired_tests(values: dict[tuple[str, str, int, str], float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in PAPER_NAMES:
        for metric in PRIMARY:
            gan = np.asarray([values[("GAN", dataset, seed, metric)] for seed in SEEDS])
            nogan = np.asarray([values[("noGAN", dataset, seed, metric)] for seed in SEEDS])
            delta = nogan - gan
            t_result = stats.ttest_rel(nogan, gan)
            try:
                w_result = stats.wilcoxon(nogan, gan, method="exact")
                w_stat, w_p = float(w_result.statistic), float(w_result.pvalue)
            except ValueError:
                w_stat, w_p = math.nan, math.nan
            rows.append({
                "dataset": dataset, "paper_dataset": PAPER_NAMES[dataset], "metric": metric,
                "gan_mean": f"{gan.mean():.10f}", "nogan_mean": f"{nogan.mean():.10f}",
                "absolute_gain_nogan_minus_gan": f"{delta.mean():.10f}",
                "relative_gain_percent": f"{delta.mean()/nogan.mean()*100:.6f}",
                "gan_seed_wins": int(np.sum(gan < nogan)),
                "nogan_seed_wins": int(np.sum(nogan < gan)),
                "paired_t_p": f"{float(t_result.pvalue):.10f}",
                "wilcoxon_stat": "" if math.isnan(w_stat) else f"{w_stat:.6f}",
                "wilcoxon_p": "" if math.isnan(w_p) else f"{w_p:.6f}",
            })
    return rows


def load_histories() -> dict[str, list[list[dict[str, str]]]]:
    histories: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    for path in sorted((GAN_ROOT / "training_history").rglob("training_history.csv")):
        match = path.parts[-3]
        dataset = canonical_gan_dataset(match)
        rows = read_csv(path)
        if len(rows) != 300:
            raise RuntimeError(f"Expected 300 epochs in {path}, found {len(rows)}")
        numeric = np.asarray([[float(row[key]) for key in ("d_loss", "g_loss", "adv_loss", "valid_rmse")] for row in rows])
        if not np.isfinite(numeric).all():
            raise RuntimeError(f"Non-finite loss history in {path}")
        histories[dataset].append(rows)
    if any(len(histories[dataset]) != 3 for dataset in PAPER_NAMES):
        raise RuntimeError({dataset: len(histories[dataset]) for dataset in PAPER_NAMES})
    return histories


def plot_p4(histories: dict[str, list[list[dict[str, str]]]]) -> None:
    runs = histories["MainSeed_TaxiBJ_P4"]
    epochs = np.arange(1, 301)
    panels = (
        ("d_loss", "Discriminator BCE"), ("g_loss", "Generator total loss"),
        ("adv_loss", "Generator adversarial BCE"), ("valid_rmse", "Validation RMSE (normalized)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.8), constrained_layout=True)
    for axis, (key, label) in zip(axes.flat, panels):
        matrix = np.asarray([[float(row[key]) for row in run] for run in runs])
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=1)
        axis.plot(epochs, mean, color="#1756A9", linewidth=1.6)
        axis.fill_between(epochs, mean-std, mean+std, color="#91BCEB", alpha=0.45, linewidth=0)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(label)
        axis.grid(True, linewidth=0.4, alpha=0.35)
    fig.suptitle("GAN training stability on TaxiBJ P4 (mean +/- std over three seeds)", fontsize=11)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "gan_stability_curves_p4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_paper_outputs(summary: dict[tuple[str, str, str], tuple[float, float]], tests: list[dict[str, object]]) -> None:
    test_lookup = {(row["dataset"], row["metric"]): row for row in tests}
    latex = [
        r"\begin{table*}[!htbp]", r"\centering",
        r"\caption{GAN stability analysis under the raw-count MainSeed protocol. Results are mean $\pm$ standard deviation over three seeds; lower is better.}",
        r"\label{tab:gan-stability}", r"\small", r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{llccc}", r"\toprule",
        r"Dataset & Metric & FUPSI-noGAN & FUPSI-GAN & Lower variant \\", r"\midrule",
    ]
    markdown = [
        "# GAN versus noGAN stability analysis", "",
        "Both variants use the same residual-enabled architecture, pretraining checkpoints, splits, 300-epoch joint-training budget, and seeds.", "",
        "| Dataset | Metric | noGAN | GAN | Lower variant |", "|---|---|---:|---:|---|",
    ]
    for dataset, paper in PAPER_NAMES.items():
        for metric in ("RMSE", "MAE"):
            nogan = summary[("noGAN", dataset, metric)]
            gan = summary[("GAN", dataset, metric)]
            lower = "GAN" if gan[0] < nogan[0] else "noGAN"
            label = paper if metric == "RMSE" else ""
            latex.append(
                f"{label} & {metric} & ${nogan[0]:.4f}\\pm{nogan[1]:.4f}$ & "
                f"${gan[0]:.4f}\\pm{gan[1]:.4f}$ & {lower} \\\\"
            )
            markdown.append(
                f"| {paper} | {metric} | {nogan[0]:.4f} +/- {nogan[1]:.4f} | "
                f"{gan[0]:.4f} +/- {gan[1]:.4f} | {lower} |"
            )
    latex.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\parbox{\textwidth}{\footnotesize\textit{Note.} The noGAN results are the audited ResidualMainE300P5 runs. MAPE is excluded because zero and near-zero targets make it unstable. With only three paired seeds, statistical tests are descriptive.}",
        r"\end{table*}",
    ])
    markdown.extend([
        "", "MAPE is excluded from the primary conclusion because of zero and near-zero denominators.",
        "All paired tests use n=3 and are treated as descriptive rather than definitive significance evidence.",
    ])
    (OUTPUT / "gan_stability_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    (OUTPUT / "gan_stability_audit.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def main() -> None:
    records, values = load_seed_metrics()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "gan_nogan_seed_metrics.csv", records)
    summary_rows, summary = summarize_metrics(values)
    write_csv(OUTPUT / "gan_nogan_mean_std.csv", summary_rows)
    tests = paired_tests(values)
    write_csv(OUTPUT / "gan_nogan_paired_tests.csv", tests)
    histories = load_histories()
    plot_p4(histories)
    write_paper_outputs(summary, tests)
    print(f"Wrote GAN stability analysis to {OUTPUT}")


if __name__ == "__main__":
    main()
