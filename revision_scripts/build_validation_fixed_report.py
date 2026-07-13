#!/usr/bin/env python
"""Rebuild adaptive_completion rows from a validation-fixed selection plan.

The expanded missing-rate script evaluates each single completion baseline
independently. A validation-fixed adaptive selector is deterministic: for a
given dataset and missing rate, it chooses one of those already evaluated
single baselines. This utility copies the selected baseline rows and relabels
them as adaptive_completion, avoiding an expensive rerun of KNN/SVD baselines.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_plan(path: Path) -> Dict[Tuple[str, float], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_plan = payload.get("adaptive_plan", payload)
    plan: Dict[Tuple[str, float], str] = {}
    for dataset, by_rate in raw_plan.items():
        for rate, method in by_rate.items():
            plan[(str(dataset), float(rate))] = str(method)
    return plan


def copy_as_adaptive(row: Dict[str, object], selected_method: str) -> Dict[str, object]:
    cloned = copy.deepcopy(row)
    cloned["selected_completion_method"] = selected_method
    cloned["method"] = "adaptive_completion"
    return cloned


def rebuild_rows(rows: Iterable[Dict[str, object]], plan: Dict[Tuple[str, float], str]) -> List[Dict[str, object]]:
    rows = list(rows)
    by_key: Dict[Tuple[str, float, int | None, str], Dict[str, object]] = {}
    for row in rows:
        seed = row.get("seed")
        key = (
            str(row["dataset"]),
            float(row["missing_rate"]),
            int(seed) if seed is not None else None,
            str(row["method"]),
        )
        by_key[key] = row

    rebuilt: List[Dict[str, object]] = []
    for row in rows:
        dataset = str(row["dataset"])
        rate = float(row["missing_rate"])
        method = str(row["method"])
        if method != "adaptive_completion":
            rebuilt.append(row)
            continue

        selected = plan[(dataset, rate)]
        seed = row.get("seed")
        seed_key = int(seed) if seed is not None else None
        source = by_key[(dataset, rate, seed_key, selected)]
        rebuilt.append(copy_as_adaptive(source, selected))
    return rebuilt


def rebuild_summary(summary: Iterable[Dict[str, object]], plan: Dict[Tuple[str, float], str]) -> List[Dict[str, object]]:
    rows = list(summary)
    by_key = {
        (str(row["dataset"]), float(row["missing_rate"]), str(row["method"])): row
        for row in rows
    }
    rebuilt: List[Dict[str, object]] = []
    for row in rows:
        dataset = str(row["dataset"])
        rate = float(row["missing_rate"])
        method = str(row["method"])
        if method != "adaptive_completion":
            rebuilt.append(row)
            continue

        selected = plan[(dataset, rate)]
        source = by_key[(dataset, rate, selected)]
        rebuilt.append(copy_as_adaptive(source, selected))
    return rebuilt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--adaptive-plan", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = json.loads(args.base_report.read_text(encoding="utf-8"))
    plan = load_plan(args.adaptive_plan)

    output = copy.deepcopy(report)
    output["runs"] = rebuild_rows(report["runs"], plan)
    output["summary"] = rebuild_summary(report["summary"], plan)
    output.setdefault("experiment_info", {})["adaptive_plan"] = {
        f"{dataset}:{rate:g}": method for (dataset, rate), method in sorted(plan.items())
    }
    output.setdefault("experiment_info", {})["adaptive_plan_source"] = str(args.adaptive_plan)
    output.setdefault("experiment_info", {})[
        "adaptive_report_note"
    ] = "adaptive_completion rows are copied from validation-selected single-baseline rows."

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_report}")


if __name__ == "__main__":
    main()
