#!/usr/bin/env python3
"""Run formal FUPSI seed-rerun tasks with skip/resume support.

This runner is safer than executing one long PowerShell file on CPU-only
machines. It skips completed outputs, writes per-task logs, and can be limited
with --max-tasks for overnight or staged execution.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "revision" / "statistics"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    alias: str
    upscale_factor: int
    map_height: int
    map_width: int
    day_len: int
    len_period: int
    n_heads: int
    num_layers: int
    lamda_p: float


DATASETS = [
    DatasetConfig("TaxiBJ_P1", "MainSeed_TaxiBJ_P1", 4, 8, 8, 48, 5, 4, 4, 0.01),
    DatasetConfig("TaxiBJ_P2", "MainSeed_TaxiBJ_P2", 4, 8, 8, 48, 1, 2, 1, 0.9),
    DatasetConfig("TaxiBJ_P3", "MainSeed_TaxiBJ_P3", 4, 8, 8, 48, 2, 4, 1, 0.01),
    DatasetConfig("TaxiBJ_P4", "MainSeed_TaxiBJ_P4", 4, 8, 8, 48, 3, 2, 1, 0.01),
    DatasetConfig("BikeNYC", "MainSeed_BikeNYC", 2, 8, 4, 24, 5, 4, 1, 0.9),
]


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_strs(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def common_args(cfg: DatasetConfig, dataset_alias: str, seed: int, epochs: int, batch_size: int, include_pretrain_lr: bool) -> list[str]:
    args = [
        "--dataset", dataset_alias,
        "--num_epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--upscale_factor", str(cfg.upscale_factor),
        "--map_height", str(cfg.map_height),
        "--map_width", str(cfg.map_width),
        "--day_len", str(cfg.day_len),
        "--nb_flow", "2",
        "--len_closeness", "3",
        "--len_period", str(cfg.len_period),
        "--len_trend", "0",
        "--n_heads", str(cfg.n_heads),
        "--num_layers", str(cfg.num_layers),
        "--skip_dim", "128",
        "--scaler_X", "1500",
        "--scaler_Y", "100",
        "--n_residuals", "8",
        "--base_channels", "64",
        "--lamda_s", "0.1",
        "--lamda_p", str(cfg.lamda_p),
        "--lr_pre", "0.000001",
        "--lr_sr", "0.0001",
        "--seed", str(seed),
    ]
    if include_pretrain_lr:
        args[-4:-4] = ["--lr", "0.001"]
    return args


def pretrain_output(code_root: Path, dataset_alias: str, cfg: DatasetConfig, seed: int) -> Path:
    return (
        code_root
        / "saved_model"
        / "separate"
        / dataset_alias
        / f"seed{seed}"
        / "cpt_noext"
        / f"3-{cfg.len_period}-0_{cfg.n_heads}_{cfg.num_layers}_128"
        / "final_model.pt"
    )


def pretrain_marker(code_root: Path, dataset_alias: str, cfg: DatasetConfig, seed: int, epochs: int, batch_size: int) -> Path:
    return pretrain_output(code_root, dataset_alias, cfg, seed).with_suffix(f".e{epochs}_b{batch_size}.done")


def stage_root(code_root: Path, dataset_alias: str, cfg: DatasetConfig, seed: int, epochs: int) -> Path:
    return (
        code_root
        / "saved_model"
        / "to_stage"
        / "no_ext(r)"
        / dataset_alias
        / f"seed{seed}"
        / f"{cfg.lamda_p}_0.1"
        / "-4-6"
        / f"8-64-{epochs}_3{cfg.len_period}0_{cfg.n_heads}_{cfg.num_layers}"
    )


def train_outputs(code_root: Path, dataset_alias: str, cfg: DatasetConfig, seed: int, epochs: int) -> list[Path]:
    root = stage_root(code_root, dataset_alias, cfg, seed, epochs)
    return [
        root / "cpt" / "final_model.pt",
        root / "Generator" / "final_model.pt",
        root / "Discriminator" / "final_model.pt",
    ]


def test_output(code_root: Path, dataset_alias: str, cfg: DatasetConfig, seed: int, epochs: int) -> Path:
    return stage_root(code_root, dataset_alias, cfg, seed, epochs) / "Generator" / "test_metrics.csv"


def command_for(
    stage: str,
    cfg: DatasetConfig,
    dataset_alias: str,
    seed: int,
    epochs: int,
    batch_size: int,
    train_script: str,
    test_script: str,
    lambda_adv: float,
) -> list[str]:
    if stage == "pretrain":
        return [sys.executable, "-u", train_script, *common_args(cfg, dataset_alias, seed, epochs, batch_size, True), "--train_pre_flag", "True"]
    if stage == "train":
        return [
            sys.executable, "-u", train_script,
            *common_args(cfg, dataset_alias, seed, epochs, batch_size, True),
            "--lambda_adv", str(lambda_adv),
        ]
    if stage == "test":
        return [sys.executable, "-u", test_script, *common_args(cfg, dataset_alias, seed, epochs, batch_size, False)]
    raise ValueError(stage)


def is_complete(stage: str, code_root: Path, dataset_alias: str, cfg: DatasetConfig, seed: int, epochs: int, batch_size: int) -> bool:
    if stage == "pretrain":
        # The original pretrain checkpoint path does not include the epoch
        # count, so use a runner-owned marker to avoid treating short smoke
        # runs as completed formal pretraining.
        return pretrain_output(code_root, dataset_alias, cfg, seed).exists() and pretrain_marker(code_root, dataset_alias, cfg, seed, epochs, batch_size).exists()
    if stage == "train":
        return all(path.exists() for path in train_outputs(code_root, dataset_alias, cfg, seed, epochs))
    if stage == "test":
        return test_output(code_root, dataset_alias, cfg, seed, epochs).exists()
    raise ValueError(stage)


def write_status(status_rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Main-Seed Task Queue Status",
        "",
        "| Dataset | Seed | Stage | Status | Seconds | Log |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in status_rows:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {row['stage']} | {row['status']} | "
            f"{row['seconds']} | `{row['log']}` |"
        )
    (OUT_DIR / "main_seed_task_queue_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_history(row: dict[str, object]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "main_seed_task_queue_history.csv"
    fieldnames = ["timestamp", "dataset", "seed", "stage", "epochs", "batch_size", "status", "seconds", "log"]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=ROOT / "fupsi")
    parser.add_argument("--datasets", default=",".join(cfg.alias for cfg in DATASETS))
    parser.add_argument("--seeds", default="2024,2025,2026")
    parser.add_argument("--stages", default="pretrain,train,test")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tasks", type=int, default=0, help="0 means no limit")
    parser.add_argument("--namespace", default="", help="Prefix for isolated data and checkpoint aliases")
    parser.add_argument("--train-script", default="main_seed_train.py")
    parser.add_argument("--test-script", default="main_seed_test.py")
    parser.add_argument("--lambda-adv", type=float, default=0.0, help="Optional adversarial-loss weight for the train stage")
    parser.add_argument("--status-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    code_root = args.code_root.resolve()
    if args.status_dir is not None:
        OUT_DIR = args.status_dir.resolve()
    selected_datasets = parse_csv_strs(args.datasets)
    selected_seeds = parse_csv_ints(args.seeds)
    selected_stages = [stage for stage in args.stages.split(",") if stage.strip()]
    selected_stages = [stage.strip() for stage in selected_stages]

    logs_dir = code_root / "revision_main_seed_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    configs = [cfg for cfg in DATASETS if cfg.alias in selected_datasets or cfg.name in selected_datasets]
    if not configs:
        raise SystemExit(f"No dataset selected from: {args.datasets}")

    ran = 0
    status_rows: list[dict[str, object]] = []
    for cfg in configs:
        dataset_alias = f"{args.namespace}_{cfg.alias.removeprefix('MainSeed_')}" if args.namespace else cfg.alias
        for seed in selected_seeds:
            for stage in selected_stages:
                label = f"{dataset_alias}_seed{seed}_{stage}_e{args.epochs}_b{args.batch_size}"
                log_path = logs_dir / f"{label}.log"
                if is_complete(stage, code_root, dataset_alias, cfg, seed, args.epochs, args.batch_size):
                    status_rows.append({"dataset": dataset_alias, "seed": seed, "stage": stage, "status": "skipped_existing", "seconds": 0, "log": log_path})
                    append_history(
                        {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "dataset": dataset_alias,
                            "seed": seed,
                            "stage": stage,
                            "epochs": args.epochs,
                            "batch_size": args.batch_size,
                            "status": "skipped_existing",
                            "seconds": 0,
                            "log": log_path,
                        }
                    )
                    continue
                if args.max_tasks and ran >= args.max_tasks:
                    status_rows.append({"dataset": dataset_alias, "seed": seed, "stage": stage, "status": "pending_max_tasks", "seconds": "", "log": log_path})
                    continue
                cmd = command_for(
                    stage, cfg, dataset_alias, seed, args.epochs, args.batch_size,
                    args.train_script, args.test_script, args.lambda_adv,
                )
                if args.dry_run:
                    status_rows.append({"dataset": dataset_alias, "seed": seed, "stage": stage, "status": "dry_run", "seconds": "", "log": log_path})
                    print("DRY RUN:", " ".join(cmd))
                    ran += 1
                    continue
                print(f"Running {label}")
                start = time.perf_counter()
                with log_path.open("w", encoding="utf-8") as log_file:
                    log_file.write("$ " + " ".join(cmd) + "\n\n")
                    log_file.flush()
                    proc = subprocess.run(cmd, cwd=code_root, stdout=log_file, stderr=subprocess.STDOUT, text=True)
                seconds = round(time.perf_counter() - start, 2)
                status = "completed" if proc.returncode == 0 else f"failed_{proc.returncode}"
                if proc.returncode == 0 and stage == "pretrain":
                    pretrain_marker(code_root, dataset_alias, cfg, seed, args.epochs, args.batch_size).write_text(
                        f"completed {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                        encoding="utf-8",
                    )
                status_rows.append({"dataset": dataset_alias, "seed": seed, "stage": stage, "status": status, "seconds": seconds, "log": log_path})
                append_history(
                    {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "dataset": dataset_alias,
                        "seed": seed,
                        "stage": stage,
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "status": status,
                        "seconds": seconds,
                        "log": log_path,
                    }
                )
                write_status(status_rows)
                ran += 1
                if proc.returncode != 0:
                    raise SystemExit(f"Task failed: {label}. See {log_path}")
    write_status(status_rows)
    print(f"Wrote queue status to {OUT_DIR / 'main_seed_task_queue_status.md'}")


if __name__ == "__main__":
    main()
