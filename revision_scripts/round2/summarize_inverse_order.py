#!/usr/bin/env python3
"""Summarize the corrected TaxiBJ P4 prediction/SR order experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


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
        "--fupsi",
        type=Path,
        default=Path("revision/round2/fupsi_seed_metrics.csv"),
    )
    parser.add_argument(
        "--inverse-root",
        type=Path,
        default=Path("revision/round2/inverse_order"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision/round2/inverse_order/summary"),
    )
    args = parser.parse_args()

    values: dict[tuple[str, int, str], float] = {}
    seed_rows: list[dict] = []
    fupsi_rows = [
        row for row in read_rows(args.fupsi) if row["dataset_key"] == "TaxiBJ_P4"
    ]
    if len(fupsi_rows) != 3:
        raise ValueError(f"Expected 3 FUPSI P4 rows, found {len(fupsi_rows)}")
    fupsi_seeds: set[int] = set()
    for row in fupsi_rows:
        seed = int(row["seed"])
        if seed not in SEEDS:
            raise ValueError(f"Unexpected FUPSI seed {seed}")
        if seed in fupsi_seeds:
            raise ValueError(f"Duplicate FUPSI P4 seed {seed}")
        fupsi_seeds.add(seed)
        for metric in ("RMSE", "MAE"):
            value = float(row[metric])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid FUPSI {metric} for seed {seed}: {value}")
            values[("FUPSI", seed, metric)] = value
        seed_rows.append(
            {
                "dataset": "TaxiBJ P4",
                "method": "FUPSI (prediction before SR)",
                "seed": seed,
                "RMSE": f"{float(row['RMSE']):.6f}",
                "MAE": f"{float(row['MAE']):.6f}",
                "source": row["source"],
            }
        )

    inverse_paths = sorted(args.inverse_root.rglob("test_metrics.csv"))
    if len(inverse_paths) != 3:
        raise ValueError(
            f"Expected 3 inverse-order metric files, found {len(inverse_paths)}"
        )
    inverse_seeds: set[int] = set()
    for path in inverse_paths:
        rows = read_rows(path)
        if len(rows) != 1:
            raise ValueError(f"{path}: expected one row, found {len(rows)}")
        row = rows[0]
        if row["dataset_key"] != "TaxiBJ_P4":
            raise ValueError(f"{path}: unexpected dataset {row['dataset_key']}")
        seed = int(row["seed"])
        if seed not in SEEDS:
            raise ValueError(f"{path}: unexpected seed {seed}")
        if seed in inverse_seeds:
            raise ValueError(f"Duplicate inverse-order seed {seed}")
        inverse_seeds.add(seed)
        if row["method"] != "FUPSI_IN_reimplementation":
            raise ValueError(f"{path}: unexpected method {row['method']!r}")
        metadata_path = path.with_name("run_metadata.json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing inverse-order metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("protocol") != "MainSeed-RawCount-v2"
            or metadata.get("order")
            != "super-resolution before fine-grid prediction"
            or float(metadata.get("scaler_x", -1)) != 1500.0
            or float(metadata.get("scaler_y", -1)) != 100.0
            or metadata.get("residual_flag") is not True
        ):
            raise ValueError(f"{metadata_path}: protocol/configuration mismatch")
        for metric in ("RMSE", "MAE"):
            value = float(row[metric])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{path}: invalid {metric}={value}")
            values[("Inverse order", seed, metric)] = value
        seed_rows.append(
            {
                "dataset": "TaxiBJ P4",
                "method": "Inverse order (SR before prediction)",
                "seed": seed,
                "RMSE": f"{float(row['RMSE']):.6f}",
                "MAE": f"{float(row['MAE']):.6f}",
                "source": path.relative_to(args.inverse_root).as_posix(),
            }
        )

    expected = {
        (method, seed, metric)
        for method in ("FUPSI", "Inverse order")
        for seed in SEEDS
        for metric in ("RMSE", "MAE")
    }
    if set(values) != expected:
        raise RuntimeError(f"Incomplete order evidence: {expected - set(values)}")
    if fupsi_seeds != set(SEEDS) or inverse_seeds != set(SEEDS):
        raise RuntimeError("Order study does not contain exactly the three formal seeds")

    summaries: list[dict] = []
    tests: list[dict] = []
    lookup: dict[tuple[str, str], tuple[float, float]] = {}
    for metric in ("RMSE", "MAE"):
        samples = {}
        for method in ("FUPSI", "Inverse order"):
            array = np.asarray(
                [values[(method, seed, metric)] for seed in SEEDS],
                dtype=np.float64,
            )
            samples[method] = array
            lookup[(method, metric)] = (
                float(array.mean()),
                float(array.std(ddof=1)),
            )
            summaries.append(
                {
                    "dataset": "TaxiBJ P4",
                    "method": method,
                    "metric": metric,
                    "runs": len(array),
                    "mean": f"{array.mean():.6f}",
                    "std": f"{array.std(ddof=1):.6f}",
                }
            )

        differences = samples["FUPSI"] - samples["Inverse order"]
        difference_std = float(differences.std(ddof=1))
        t_result = stats.ttest_rel(samples["FUPSI"], samples["Inverse order"])
        if np.allclose(differences, 0.0):
            wilcoxon_statistic, wilcoxon_p = 0.0, 1.0
        else:
            wilcoxon = stats.wilcoxon(
                samples["FUPSI"], samples["Inverse order"], method="auto"
            )
            wilcoxon_statistic = float(wilcoxon.statistic)
            wilcoxon_p = float(wilcoxon.pvalue)
        if difference_std > 0:
            cohen_dz = float(differences.mean() / difference_std)
        elif np.isclose(differences.mean(), 0.0):
            cohen_dz = 0.0
        else:
            cohen_dz = math.copysign(math.inf, float(differences.mean()))
        tests.append(
            {
                "dataset": "TaxiBJ P4",
                "comparison": "FUPSI minus inverse order",
                "metric": metric,
                "n": len(SEEDS),
                "mean_difference": f"{differences.mean():.6f}",
                "paired_t_statistic": f"{float(t_result.statistic):.6f}",
                "paired_t_p": f"{float(t_result.pvalue):.6f}",
                "wilcoxon_statistic": f"{wilcoxon_statistic:.6f}",
                "wilcoxon_p": f"{wilcoxon_p:.6f}",
                "cohen_dz": (
                    f"{cohen_dz:.6f}" if math.isfinite(cohen_dz) else str(cohen_dz)
                ),
                "fupsi_lower_seed_count": int(np.sum(differences < 0)),
                "inverse_lower_seed_count": int(np.sum(differences > 0)),
                "tie_seed_count": int(np.sum(np.isclose(differences, 0.0))),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "order_seed_metrics.csv", seed_rows)
    write_rows(args.output_dir / "order_mean_std.csv", summaries)
    write_rows(args.output_dir / "order_paired_tests.csv", tests)

    table = [
        r"\begin{table}[htb]",
        r"\centering",
        r"\caption{Prediction-before-super-resolution order study on TaxiBJ P4. Results are mean $\pm$ standard deviation over three matched seeds; lower is better.}",
        r"\label{tab:order-study}",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Order & RMSE & MAE \\",
        r"\midrule",
    ]
    for method, label in (
        ("FUPSI", "Prediction before SR"),
        ("Inverse order", "SR before prediction"),
    ):
        values_text = []
        for metric in ("RMSE", "MAE"):
            mean, std = lookup[(method, metric)]
            values_text.append(f"${mean:.4f}\\pm{std:.4f}$")
        table.append(f"{label} & {values_text[0]} & {values_text[1]} \\\\")
    table.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\parbox{\linewidth}{\footnotesize\textit{Note.} Paired tests and effect sizes are reported in the accompanying seed-level evidence. With only three seeds, they are treated as descriptive rather than definitive evidence.}",
            r"\end{table}",
            "",
        ]
    )
    (args.output_dir / "order_study_table.tex").write_text(
        "\n".join(table), encoding="utf-8"
    )
    audit = {
        "protocol": "MainSeed-RawCount-v2",
        "dataset": "TaxiBJ P4",
        "methods": [
            "FUPSI (prediction before SR)",
            "Inverse order (SR before prediction)",
        ],
        "seeds": list(SEEDS),
        "expected_inverse_runs": len(SEEDS),
        "inverse_runs": len(inverse_seeds),
        "normalization": {
            "coarse_divisor": 1500,
            "fine_divisor": 100,
            "metrics": "raw counts after inverse scaling",
        },
        "statistical_interpretation": "descriptive; n=3 has low power",
    }
    (args.output_dir / "order_study_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(
        f"Validated 3/3 inverse-order runs and wrote order analysis to "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
