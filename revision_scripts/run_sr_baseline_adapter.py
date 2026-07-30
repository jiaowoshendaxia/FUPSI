#!/usr/bin/env python3
"""Run SR-only baselines under the MainSeed protocol.

The first supported adapter is UrbanFM from the downloaded CUFAR repository.
It trains on aligned MainSeed target pairs and tests on FUPSI predicted coarse
maps, which keeps the SR-only baseline consistent with the paper's
prediction-before-super-resolution pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CODE_ROOT = ROOT / "fupsi" if (ROOT / "fupsi").exists() else ROOT
DEFAULT_CUFAR_ROOT = ROOT / "revision" / "external_baselines" / "CUFAR"
DEFAULT_OUT = ROOT / "revision" / "baseline" / "sr_adapter_results"


@dataclass(frozen=True)
class DatasetConfig:
    paper_name: str
    alias: str
    scale: int
    coarse_h: int
    coarse_w: int
    len_closeness: int
    len_period: int
    len_trend: int
    day_len: int
    scaler_x: int
    scaler_y: int
    lamda_p: str
    lamda_s: str
    n_residuals: int
    base_channels: int
    distant_len: int
    heads: int
    layers: int
    fupsi_alias: str | None = None


DATASETS: dict[str, DatasetConfig] = {
    "MainSeed_TaxiBJ_P1": DatasetConfig("TaxiBJ P1", "MainSeed_TaxiBJ_P1", 4, 8, 8, 3, 5, 0, 48, 1500, 100, "0.01", "0.1", 8, 64, 5, 4, 4),
    "MainSeed_TaxiBJ_P2": DatasetConfig("TaxiBJ P2", "MainSeed_TaxiBJ_P2", 4, 8, 8, 3, 1, 0, 48, 1500, 100, "0.9", "0.1", 8, 64, 1, 2, 1),
    "MainSeed_TaxiBJ_P3": DatasetConfig("TaxiBJ P3", "MainSeed_TaxiBJ_P3", 4, 8, 8, 3, 2, 0, 48, 1500, 100, "0.01", "0.1", 8, 64, 2, 4, 1),
    "MainSeed_TaxiBJ_P4": DatasetConfig("TaxiBJ P4", "MainSeed_TaxiBJ_P4", 4, 8, 8, 3, 3, 0, 48, 1500, 100, "0.01", "0.1", 8, 64, 3, 2, 1),
    "MainSeed_BikeNYC": DatasetConfig("BikeNYC", "MainSeed_BikeNYC", 2, 8, 4, 3, 5, 0, 24, 1500, 100, "0.9", "0.1", 8, 64, 5, 4, 1),
    "MainSeed_ChicagoTaxi2024": DatasetConfig(
        "Chicago Taxi 2024",
        "MainSeed_ChicagoTaxi2024",
        2,
        16,
        16,
        3,
        3,
        0,
        24,
        500,
        500,
        "0.01",
        "0.1",
        8,
        64,
        3,
        2,
        1,
        "ChicagoResidualE300_MainSeed_ChicagoTaxi2024",
    ),
}


METRICS = ["RMSE", "MAE", "MAPE", "RMSE_c", "MAE_c", "MAPE_c"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def offset(cfg: DatasetConfig) -> int:
    return max(cfg.len_closeness, cfg.len_period * cfg.day_len, cfg.len_trend * cfg.day_len * 7)


def flatten_channels(array: np.ndarray) -> np.ndarray:
    if array.ndim != 4:
        raise ValueError(f"Expected [N,C,H,W], got {array.shape}")
    n, c, h, w = array.shape
    return array.reshape(n * c, 1, h, w)


def aligned_pairs(code_root: Path, cfg: DatasetConfig, split: str) -> tuple[np.ndarray, np.ndarray]:
    split_dir = code_root / "data" / cfg.alias / split
    coarse = np.load(split_dir / "X.npy").astype(np.float32)
    fine = np.load(split_dir / "Y.npy").astype(np.float32)
    start = offset(cfg)
    if len(coarse) <= start:
        raise ValueError(f"{cfg.alias}/{split} has no samples after offset={start}")
    return flatten_channels(coarse[start:]), flatten_channels(fine[start:])


def make_loader(
    coarse: np.ndarray,
    fine: np.ndarray,
    cfg: DatasetConfig,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x = torch.from_numpy(coarse / cfg.scaler_x).float()
    y = torch.from_numpy(fine / cfg.scaler_y).float()
    ext = torch.zeros((len(x), 7), dtype=torch.float32)
    return DataLoader(TensorDataset(x, y, ext), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def fupsi_generator_dir(
    code_root: Path,
    cfg: DatasetConfig,
    seed: int,
    epochs: int,
    alias_override: str | None = None,
) -> Path:
    fupsi_alias = alias_override or cfg.fupsi_alias or cfg.alias
    pattern = (
        code_root
        / "saved_model"
        / "to_stage"
        / "no_ext(r)"
        / fupsi_alias
        / f"seed{seed}"
        / f"{cfg.lamda_p}_{cfg.lamda_s}"
        / "-4-6"
        / f"{cfg.n_residuals}-{cfg.base_channels}-{epochs}_3{cfg.distant_len}0_{cfg.heads}_{cfg.layers}"
        / "Generator"
    )
    if pattern.exists():
        return pattern
    matches = sorted(
        (code_root / "saved_model" / "to_stage" / "no_ext(r)" / fupsi_alias / f"seed{seed}").glob(
            f"**/{cfg.n_residuals}-{cfg.base_channels}-{epochs}_*/Generator"
        )
    )
    if not matches:
        raise FileNotFoundError(f"Cannot find FUPSI generator dir for {fupsi_alias} seed{seed}")
    return matches[-1]


def load_test_arrays(
    code_root: Path,
    cfg: DatasetConfig,
    seed: int,
    epochs: int,
    alias_override: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]:
    gen_dir = fupsi_generator_dir(code_root, cfg, seed, epochs, alias_override)
    required = ["test_coarse.npy", "true_coarse.npy", "true_fine.npy"]
    missing = [name for name in required if not (gen_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {gen_dir}")
    pred_coarse = np.load(gen_dir / "test_coarse.npy").astype(np.float32)
    true_coarse = np.load(gen_dir / "true_coarse.npy").astype(np.float32)
    true_fine = np.load(gen_dir / "true_fine.npy").astype(np.float32)
    return (
        flatten_channels(pred_coarse),
        flatten_channels(true_coarse),
        flatten_channels(true_fine),
        pred_coarse,
        gen_dir,
    )


def metric_values(pred_fine: np.ndarray, true_fine: np.ndarray, pred_coarse: np.ndarray, true_coarse: np.ndarray) -> dict[str, float]:
    def mse(pred: np.ndarray, real: np.ndarray) -> float:
        return float(np.mean((pred.astype(np.float64) - real.astype(np.float64)) ** 2))

    def mae(pred: np.ndarray, real: np.ndarray) -> float:
        return float(np.mean(np.abs(pred.astype(np.float64) - real.astype(np.float64))))

    def mape(pred: np.ndarray, real: np.ndarray, eps: float = 1e-6) -> float:
        return float(np.mean(np.abs((real.astype(np.float64) - pred.astype(np.float64)) / np.maximum(np.abs(real.astype(np.float64)), eps))))

    return {
        "RMSE": math.sqrt(mse(pred_fine, true_fine)),
        "MAE": mae(pred_fine, true_fine),
        "MAPE": mape(pred_fine, true_fine),
        "RMSE_c": math.sqrt(mse(pred_coarse, true_coarse)),
        "MAE_c": mae(pred_coarse, true_coarse),
        "MAPE_c": mape(pred_coarse, true_coarse),
    }


def import_model_class(cufar_root: Path, model_name: str) -> type[torch.nn.Module]:
    sys.path.insert(0, str(cufar_root))
    module = importlib.import_module(f"model.{model_name}")
    return getattr(module, model_name)


def build_model(cufar_root: Path, model_name: str, cfg: DatasetConfig, base_channels: int) -> torch.nn.Module:
    if cfg.scale not in {2, 4}:
        raise ValueError(f"{model_name} adapter supports scale 2 or 4; got {cfg.scale} for {cfg.alias}")
    cls = import_model_class(cufar_root, model_name)
    if model_name in {"UrbanFM", "FODE", "UrbanODE"}:
        model = cls(
            in_channels=1,
            out_channels=1,
            n_residual_blocks=16,
            base_channels=base_channels,
            img_width=cfg.coarse_w,
            img_height=cfg.coarse_h,
            ext_flag=False,
            scaler_X=cfg.scaler_x,
            scaler_Y=cfg.scaler_y,
        )
        # The released FODE/UrbanODE code fixes LayerNorm to a 32x32
        # coarse grid. Replace only that shape-dependent layer so the
        # published architecture can be evaluated on the paper's 8x8 grid.
        if model_name in {"FODE", "UrbanODE"} and isinstance(model.conv2[0], torch.nn.LayerNorm):
            model.conv2[0] = torch.nn.LayerNorm([base_channels, cfg.coarse_h, cfg.coarse_w])

        # CUFAR's released SR models contain two x2 PixelShuffle stages and
        # x4 distribution recovery. BikeNYC uses scale 2, so retain one stage
        # and replace only the scale-dependent normalization/recovery layers.
        if cfg.scale == 2:
            module = importlib.import_module(f"model.{model_name}")
            stage_modules = list(model.upsampling.children())
            if len(stage_modules) < 4:
                raise ValueError(f"Cannot adapt {model_name} upsampling stack to scale 2")
            model.upsampling = torch.nn.Sequential(*stage_modules[:4])
            model.den_softmax = module.N2_Normalization(2)
            model.recover = module.Recover_from_density(2)
        return model
    raise ValueError(f"Unsupported model_name={model_name}")


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: DatasetConfig,
    device: torch.device,
) -> float:
    model.eval()
    total_mse = 0.0
    n = 0
    with torch.no_grad():
        for coarse, fine, ext in loader:
            coarse = coarse.to(device)
            fine = fine.to(device)
            ext = ext.to(device)
            pred = model(coarse, ext) * cfg.scaler_y
            real = fine * cfg.scaler_y
            total_mse += F.mse_loss(pred, real, reduction="sum").item()
            n += int(np.prod(real.shape))
    return total_mse / max(n, 1)


def predict_test(
    model: torch.nn.Module,
    pred_coarse: np.ndarray,
    cfg: DatasetConfig,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    x = torch.from_numpy(pred_coarse / cfg.scaler_x).float()
    ext = torch.zeros((len(x), 7), dtype=torch.float32)
    loader = DataLoader(TensorDataset(x, ext), batch_size=batch_size, shuffle=False, num_workers=0)
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for coarse, ext_batch in loader:
            pred = model(coarse.to(device), ext_batch.to(device)) * cfg.scaler_y
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_source_hashes(generator_dir: Path) -> dict[str, str]:
    paths = {
        "test_coarse_sha256": generator_dir / "test_coarse.npy",
        "true_coarse_sha256": generator_dir / "true_coarse.npy",
        "true_fine_sha256": generator_dir / "true_fine.npy",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shared evaluation arrays: {missing}")
    return {name: file_sha256(path) for name, path in paths.items()}


def run_one(args: argparse.Namespace, cfg: DatasetConfig, seed: int, model_name: str) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    run_dir = (
        args.output_dir
        / model_name
        / cfg.alias
        / f"seed{seed}"
        / f"epochs{args.epochs}"
    )
    metrics_path = run_dir / "test_metrics.csv"
    if args.skip_existing and metrics_path.exists():
        with metrics_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ValueError(
                f"Expected one existing metrics row in {metrics_path}, "
                f"received {len(rows)}"
            )
        print(f"Skipping completed run: {metrics_path}", flush=True)
        return rows[0]
    fupsi_alias = (
        f"{args.fupsi_namespace}_{cfg.alias.removeprefix('MainSeed_')}"
        if args.fupsi_namespace
        else None
    )
    if args.eval_only:
        test_coarse, true_coarse, true_fine, _, gen_dir = load_test_arrays(
            args.code_root, cfg, seed, args.fupsi_epochs, fupsi_alias
        )
        model = build_model(args.cufar_root, model_name, cfg, args.base_channels).to(device)
        checkpoint_dir = args.checkpoint_root / model_name / cfg.alias / f"seed{seed}" / f"epochs{args.epochs}"
        best_path = checkpoint_dir / "best_model.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"Missing trained checkpoint: {best_path}")
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        pred_fine = predict_test(model, test_coarse, cfg, args.batch_size, device)
        metrics = metric_values(pred_fine, true_fine, test_coarse, true_coarse)
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_row = {
            "dataset": cfg.paper_name,
            "alias": cfg.alias,
            "method": model_name,
            "seed": seed,
            "mode": "test",
            **{metric: f"{metrics[metric]:.6f}" for metric in METRICS},
            "best_epoch": state.get("epoch", ""),
            "best_valid_mse": f"{float(state.get('val_mse', float('nan'))):.8f}",
            "fupsi_generator_dir": str(gen_dir),
            "checkpoint": str(best_path),
            "run_dir": str(run_dir),
            **evaluation_source_hashes(gen_dir),
        }
        np.save(run_dir / "pred_fine.npy", pred_fine)
        write_csv(run_dir / "test_metrics.csv", [metrics_row], list(metrics_row.keys()))
        print(f"Evaluated {model_name} {cfg.alias} seed{seed} on {fupsi_alias or cfg.alias}", flush=True)
        return metrics_row

    train_x, train_y = aligned_pairs(args.code_root, cfg, "train")
    valid_x, valid_y = aligned_pairs(args.code_root, cfg, "valid")
    train_loader = make_loader(train_x, train_y, cfg, args.batch_size, True)
    valid_loader = make_loader(valid_x, valid_y, cfg, args.batch_size, False)
    test_coarse, true_coarse, true_fine, _, gen_dir = load_test_arrays(
        args.code_root, cfg, seed, args.fupsi_epochs, fupsi_alias
    )

    model = build_model(args.cufar_root, model_name, cfg, args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.csv"
    best_path = run_dir / "best_model.pt"
    best_val = float("inf")
    best_epoch = -1
    start_time = time.time()
    log_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n = 0
        for coarse, fine, ext in train_loader:
            coarse = coarse.to(device)
            fine = fine.to(device)
            ext = ext.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(coarse, ext) * cfg.scaler_y
            real = fine * cfg.scaler_y
            loss = F.mse_loss(pred, real)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(coarse)
            n += len(coarse)
        val_mse = evaluate_model(model, valid_loader, cfg, device)
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_mse": val_mse}, best_path)
        log_rows.append(
            {
                "epoch": epoch,
                "train_mse": f"{train_loss / max(n, 1):.8f}",
                "valid_mse": f"{val_mse:.8f}",
                "best_epoch": best_epoch,
                "elapsed_sec": f"{time.time() - start_time:.2f}",
            }
        )
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"{model_name} {cfg.alias} seed{seed} epoch {epoch}/{args.epochs} train_mse={train_loss/max(n,1):.6f} val_mse={val_mse:.6f} best={best_epoch}", flush=True)
        write_csv(log_path, log_rows, ["epoch", "train_mse", "valid_mse", "best_epoch", "elapsed_sec"])

    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    pred_fine = predict_test(model, test_coarse, cfg, args.batch_size, device)
    metrics = metric_values(pred_fine, true_fine, test_coarse, true_coarse)

    np.save(run_dir / "pred_fine.npy", pred_fine)
    metrics_row = {
        "dataset": cfg.paper_name,
        "alias": cfg.alias,
        "method": model_name,
        "seed": seed,
        "mode": "test",
        **{metric: f"{metrics[metric]:.6f}" for metric in METRICS},
        "best_epoch": best_epoch,
        "best_valid_mse": f"{best_val:.8f}",
        "fupsi_generator_dir": str(gen_dir),
        "run_dir": str(run_dir),
        **evaluation_source_hashes(gen_dir),
    }
    write_csv(run_dir / "test_metrics.csv", [metrics_row], list(metrics_row.keys()))
    return metrics_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    parser.add_argument("--cufar-root", type=Path, default=DEFAULT_CUFAR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--datasets", default="MainSeed_TaxiBJ_P4")
    parser.add_argument("--seeds", default="2026")
    parser.add_argument("--models", default="UrbanFM")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--fupsi-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fupsi-namespace", default="")
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    args.code_root = args.code_root.resolve()
    args.cufar_root = args.cufar_root.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        cfg = DATASETS[dataset]
        for model_name in models:
            for seed in seeds:
                rows.append(run_one(args, cfg, seed, model_name))

    run_tag = "_".join(datasets + models + [str(seed) for seed in seeds])
    combined_path = args.output_dir / f"sr_adapter_metrics_{run_tag}.csv"
    if rows:
        write_csv(combined_path, rows, list(rows[0].keys()))
    print(f"Wrote {combined_path}")


if __name__ == "__main__":
    main()
