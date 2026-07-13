#!/usr/bin/env python
"""Create mean/std and paired significance summaries for revision tables.

Expected input CSV columns:

    dataset,method,seed,metric,value

Optional columns are preserved by grouping only on dataset and metric.
The script compares --method-a against --method-b for each dataset/metric
using paired values by seed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def paired_test(a: np.ndarray, b: np.ndarray) -> Dict[str, float | str]:
    diff = a - b
    result: Dict[str, float | str] = {
        "mean_diff": float(np.mean(diff)),
        "n_pairs": int(len(diff)),
    }
    try:
        from scipy.stats import ttest_rel, wilcoxon

        result["paired_t_p"] = float(ttest_rel(a, b).pvalue)
        if np.allclose(diff, 0):
            result["wilcoxon_p"] = "all_equal"
        else:
            result["wilcoxon_p"] = float(wilcoxon(a, b).pvalue)
    except Exception as exc:
        result["paired_t_p"] = f"NA: {exc}"
        result["wilcoxon_p"] = f"NA: {exc}"
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--method-a", required=True, help="Usually FUPSI")
    parser.add_argument("--method-b", required=True, help="Comparison baseline")
    parser.add_argument("--summary-csv", type=Path, default=Path("revision/stat_summary.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("revision/stat_tests.csv"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = pd.read_csv(args.input_csv)
    required = {"dataset", "method", "seed", "metric", "value"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    summary = (
        data.groupby(["dataset", "method", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)

    rows: List[Dict[str, object]] = []
    for (dataset, metric), group in data.groupby(["dataset", "metric"]):
        pivot = group.pivot_table(index="seed", columns="method", values="value", aggfunc="mean")
        if args.method_a not in pivot.columns or args.method_b not in pivot.columns:
            continue
        paired = pivot[[args.method_a, args.method_b]].dropna()
        if len(paired) < 2:
            continue
        test = paired_test(
            paired[args.method_a].to_numpy(dtype=float),
            paired[args.method_b].to_numpy(dtype=float),
        )
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "method_a": args.method_a,
                "method_b": args.method_b,
                **test,
            }
        )
    pd.DataFrame(rows).to_csv(args.test_csv, index=False)
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.test_csv}")


if __name__ == "__main__":
    main()
