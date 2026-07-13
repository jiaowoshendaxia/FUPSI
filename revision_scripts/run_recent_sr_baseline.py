#!/usr/bin/env python3
"""Train and evaluate recent FUFI baselines on MainSeed data."""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from run_sr_baseline_adapter import DATASETS, aligned_pairs, load_test_arrays, metric_values


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CODE_ROOT = ROOT / "fupsi"
DEFAULT_PLGF_ROOT = ROOT / "revision" / "external_baselines" / "PLGF"
DEFAULT_OUTPUT = ROOT / "revision" / "baseline" / "recent_sr_results"
METRICS = ("RMSE", "MAE", "MAPE", "RMSE_c", "MAE_c", "MAPE_c")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def import_plgf(plgf_root: Path):
    sys.path.insert(0, str(plgf_root))
    from util.PLGF import PLGF
    from util.focalloss import FocalL1Loss
    return PLGF, FocalL1Loss


def loader(coarse: np.ndarray, fine: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x = torch.from_numpy(coarse).float()
    y = torch.from_numpy(fine).float()
    # The controlled MainSeed reruns do not use external factors. Zeros keep
    # the released PLGF interface intact without granting extra information.
    ext = torch.zeros((len(x), 7), dtype=torch.float32)
    return DataLoader(
        TensorDataset(x, y, ext), batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_model(args: argparse.Namespace, cfg) -> torch.nn.Module:
    PLGF, _ = import_plgf(args.plgf_root)
    return PLGF(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        img_width=cfg.coarse_w,
        img_height=cfg.coarse_h,
        num_layers=args.num_layers,
        upscale_factor=cfg.scale,
    )


def validate(model, data_loader: DataLoader, criterion, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for coarse, fine, ext in data_loader:
            coarse, fine, ext = coarse.to(device), fine.to(device), ext.to(device)
            loss = criterion(model(coarse, ext), fine)
            total += loss.item() * len(coarse)
            count += len(coarse)
    return total / max(count, 1)


def predict(model, coarse: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(coarse).float()
    ext = torch.zeros((len(x), 7), dtype=torch.float32)
    data_loader = DataLoader(TensorDataset(x, ext), batch_size=batch_size, shuffle=False, num_workers=0)
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for coarse_batch, ext_batch in data_loader:
            output.append(model(coarse_batch.to(device), ext_batch.to(device)).cpu().numpy())
    return np.concatenate(output, axis=0)


def checkpoint_dir(args: argparse.Namespace, alias: str, seed: int) -> Path:
    return args.output_dir / "PLGF" / alias / f"seed{seed}" / f"epochs{args.epochs}"


def train_one(args: argparse.Namespace, alias: str, seed: int) -> dict[str, Any]:
    cfg = DATASETS[alias]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model(args, cfg).to(device)
    _, FocalL1Loss = import_plgf(args.plgf_root)
    criterion = FocalL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    train_x, train_y = aligned_pairs(args.code_root, cfg, "train")
    valid_x, valid_y = aligned_pairs(args.code_root, cfg, "valid")
    train_loader = loader(train_x, train_y, args.batch_size, True)
    valid_loader = loader(valid_x, valid_y, args.batch_size, False)

    run_dir = checkpoint_dir(args, alias, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best_model.pt"
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        count = 0
        for coarse, fine, ext in train_loader:
            coarse, fine, ext = coarse.to(device), fine.to(device), ext.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(coarse, ext), fine)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_total += loss.item() * len(coarse)
            count += len(coarse)
        valid_loss = validate(model, valid_loader, criterion, device)
        scheduler.step(valid_loss)
        if valid_loss < best_val:
            best_val = valid_loss
            best_epoch = epoch
            stale = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "valid_loss": valid_loss},
                best_path,
            )
        else:
            stale += 1
        history.append({
            "epoch": epoch, "train_focal_l1": f"{train_total/max(count,1):.10f}",
            "valid_focal_l1": f"{valid_loss:.10f}", "best_epoch": best_epoch,
            "learning_rate": f"{optimizer.param_groups[0]['lr']:.10g}",
            "elapsed_seconds": f"{time.time()-start:.2f}",
        })
        write_csv(run_dir / "training_history.csv", history)
        if epoch == 1 or epoch % 10 == 0:
            print(f"PLGF {alias} seed{seed} epoch {epoch}/{args.epochs} val={valid_loss:.6f} best={best_epoch}", flush=True)
        if stale >= args.patience:
            break

    return {
        "dataset": cfg.paper_name, "alias": alias, "method": "PLGF", "seed": seed,
        "mode": "train", "best_epoch": best_epoch, "best_valid_loss": f"{best_val:.10f}",
        "checkpoint": str(best_path), "epochs_completed": len(history),
    }


def evaluate_one(args: argparse.Namespace, alias: str, seed: int) -> dict[str, Any]:
    cfg = DATASETS[alias]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model(args, cfg).to(device)
    best_path = checkpoint_dir(args, alias, seed) / "best_model.pt"
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    fupsi_alias = f"{args.fupsi_namespace}_{alias.removeprefix('MainSeed_')}" if args.fupsi_namespace else None
    pred_coarse, true_coarse, true_fine, _, source = load_test_arrays(
        args.code_root, cfg, seed, args.fupsi_epochs, fupsi_alias
    )
    pred_fine = predict(model, pred_coarse, args.batch_size, device)
    metrics = metric_values(pred_fine, true_fine, pred_coarse, true_coarse)
    run_dir = checkpoint_dir(args, alias, seed)
    np.save(run_dir / "pred_fine.npy", pred_fine)
    row = {
        "dataset": cfg.paper_name, "alias": alias, "method": "PLGF", "seed": seed, "mode": "test",
        **{metric: f"{metrics[metric]:.6f}" for metric in METRICS},
        "best_epoch": state["epoch"], "best_valid_loss": f"{state['valid_loss']:.10f}",
        "fupsi_generator_dir": str(source), "checkpoint": str(best_path),
    }
    write_csv(run_dir / "test_metrics.csv", [row])
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    parser.add_argument("--plgf-root", type=Path, default=DEFAULT_PLGF_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", default="MainSeed_TaxiBJ_P4")
    parser.add_argument("--seeds", default="2024,2025,2026")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--fupsi-epochs", type=int, default=300)
    parser.add_argument("--fupsi-namespace", default="")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.train_only and args.eval_only:
        raise SystemExit("Choose only one of --train-only or --eval-only")
    args.code_root = args.code_root.resolve()
    args.plgf_root = args.plgf_root.resolve()
    args.output_dir = args.output_dir.resolve()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    for alias in datasets:
        for seed in seeds:
            if not args.eval_only:
                rows.append(train_one(args, alias, seed))
            if not args.train_only:
                rows.append(evaluate_one(args, alias, seed))
    write_csv(args.output_dir / "plgf_runs_latest.csv", rows)
    print(f"Wrote {len(rows)} PLGF run records", flush=True)


if __name__ == "__main__":
    main()
