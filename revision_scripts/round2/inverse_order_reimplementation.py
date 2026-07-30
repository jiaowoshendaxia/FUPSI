#!/usr/bin/env python3
"""Regenerate the super-resolution-before-prediction order baseline.

The inverse-order pipeline first learns a residual distributional
super-resolution model for each observed coarse map, applies it to the complete
historical sequence, and then forecasts the next fine-grid map with the same
temporal Transformer family used by FUPSI. It is evaluated only as a controlled
order study under MainSeed-RawCount-v2.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from hrstt_reimplementation import DATASETS, DatasetConfig, load_split, set_seed


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def direct_sr_dataset(
    x: np.ndarray, y: np.ndarray, scaler_x: float, scaler_y: float
) -> TensorDataset:
    ext = np.zeros((len(x),), dtype=np.float32)
    return TensorDataset(
        torch.from_numpy(x / scaler_x),
        torch.from_numpy(y / scaler_y),
        torch.from_numpy(ext),
    )


def build_fine_temporal_dataset(
    generated_fine: np.ndarray,
    true_fine: np.ndarray,
    cfg: DatasetConfig,
    scaler_y: float,
) -> TensorDataset:
    max_lag = max(
        cfg.len_closeness,
        cfg.len_period * cfg.day_len,
        cfg.len_trend * cfg.day_len * 7,
    )
    block_len = max_lag + 1
    count = len(generated_fine) - block_len + 1
    if count <= 0:
        raise ValueError(
            f"{cfg.paper_name}: {len(generated_fine)} slots cannot support "
            f"lag {max_lag}"
        )
    channels, height, width = generated_fine.shape[1:]

    def empty_view(length: int) -> np.ndarray:
        return np.empty(
            (count, length, channels, height, width), dtype=np.float32
        )

    xc = empty_view(cfg.len_closeness)
    xp = empty_view(cfg.len_period)
    xt = empty_view(cfg.len_trend)
    target = np.empty((count, channels, height, width), dtype=np.float32)
    for sample_index in range(count):
        target_index = sample_index + block_len - 1
        for offset in range(cfg.len_closeness):
            xc[sample_index, offset] = generated_fine[
                target_index - (cfg.len_closeness - offset)
            ]
        for offset in range(cfg.len_period):
            xp[sample_index, offset] = generated_fine[
                target_index - (cfg.len_period - offset) * cfg.day_len
            ]
        for offset in range(cfg.len_trend):
            xt[sample_index, offset] = generated_fine[
                target_index - (cfg.len_trend - offset) * cfg.day_len * 7
            ]
        target[sample_index] = true_fine[target_index]
    ext = np.zeros((count,), dtype=np.float32)
    return TensorDataset(
        torch.from_numpy(xc / scaler_y),
        torch.from_numpy(xp / scaler_y),
        torch.from_numpy(xt / scaler_y),
        torch.from_numpy(ext),
        torch.from_numpy(target / scaler_y),
    )


@torch.no_grad()
def evaluate_sr(
    generator,
    loader: DataLoader,
    device: torch.device,
    scaler_y: float,
) -> tuple[float, float]:
    generator.eval()
    squared = 0.0
    absolute = 0.0
    values = 0
    for coarse, fine, ext in loader:
        prediction = generator(coarse.to(device), ext.to(device))
        error = (prediction - fine.to(device)) * scaler_y
        squared += float(torch.sum(error * error).cpu())
        absolute += float(torch.sum(torch.abs(error)).cpu())
        values += error.numel()
    return math.sqrt(squared / values), absolute / values


def train_sr_epoch(generator, loader, optimizer, device) -> float:
    generator.train()
    total = 0.0
    samples = 0
    for coarse, fine, ext in loader:
        coarse = coarse.to(device)
        fine = fine.to(device)
        ext = ext.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = generator(coarse, ext)
        loss = F.mse_loss(prediction, fine)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * len(coarse)
        samples += len(coarse)
    return total / samples


@torch.no_grad()
def generate_fine_series(
    generator,
    x: np.ndarray,
    device: torch.device,
    scaler_x: float,
    scaler_y: float,
    batch_size: int,
) -> np.ndarray:
    generator.eval()
    dataset = TensorDataset(
        torch.from_numpy(x / scaler_x),
        torch.zeros((len(x),), dtype=torch.float32),
    )
    output = []
    for coarse, ext in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        prediction = generator(coarse.to(device), ext.to(device))
        output.append(prediction.cpu().numpy() * scaler_y)
    return np.concatenate(output).astype(np.float32)


def train_predictor_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    samples = 0
    for xc, xp, xt, ext, target in loader:
        xc = xc.to(device)
        xp = xp.to(device)
        xt = xt.to(device)
        ext = ext.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(xc, xp, xt, ext)
        loss = F.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * len(xc)
        samples += len(xc)
    return total / samples


@torch.no_grad()
def evaluate_predictor(
    model,
    loader,
    device,
    scaler_y: float,
    save_predictions: bool = False,
):
    model.eval()
    squared = 0.0
    absolute = 0.0
    values = 0
    samples = 0
    predictions = []
    targets = []
    for xc, xp, xt, ext, target in loader:
        prediction = model(
            xc.to(device), xp.to(device), xt.to(device), ext.to(device)
        )
        target = target.to(device)
        error = (prediction - target) * scaler_y
        squared += float(torch.sum(error * error).cpu())
        absolute += float(torch.sum(torch.abs(error)).cpu())
        values += error.numel()
        samples += len(xc)
        if save_predictions:
            predictions.append(prediction.cpu().numpy() * scaler_y)
            targets.append(target.cpu().numpy() * scaler_y)
    arrays = None
    if save_predictions:
        arrays = (np.concatenate(predictions), np.concatenate(targets))
    return math.sqrt(squared / values), absolute / values, samples, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="TaxiBJ_P4")
    parser.add_argument("--data-prefix", default="MainSeed")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--sr-epochs", type=int, default=300)
    parser.add_argument("--prediction-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--halving-interval", type=int, default=20)
    parser.add_argument("--scaler-x", type=float, default=1500.0)
    parser.add_argument("--scaler-y", type=float, default=100.0)
    parser.add_argument("--n-residuals", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("revision/round2/inverse_order"),
    )
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = DATASETS[args.dataset]
    code_root = args.code_root.resolve()
    sys.path.insert(0, str(code_root))
    try:
        from prediction import TransAm
        from UrbanSG import Generator
    finally:
        sys.path.pop(0)

    data_alias = f"{args.data_prefix}_{cfg.key}"
    data_root = code_root / "data" / data_alias
    arrays = {
        split: load_split(data_root, split)
        for split in ("train", "valid", "test")
    }
    expected_coarse = (2, cfg.map_height, cfg.map_width)
    expected_fine = (
        2,
        cfg.map_height * cfg.upscale_factor,
        cfg.map_width * cfg.upscale_factor,
    )
    for split, (x, y) in arrays.items():
        if x.shape[1:] != expected_coarse or y.shape[1:] != expected_fine:
            raise ValueError(
                f"{cfg.paper_name} {split}: X={x.shape}, Y={y.shape}, "
                f"expected {expected_coarse}/{expected_fine}"
            )
    output_dir = args.output_root.resolve() / cfg.key / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sr_datasets = {
        split: direct_sr_dataset(x, y, args.scaler_x, args.scaler_y)
        for split, (x, y) in arrays.items()
    }
    train_generator = torch.Generator().manual_seed(args.seed)
    sr_loaders = {
        "train": DataLoader(
            sr_datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            generator=train_generator,
        ),
        "valid": DataLoader(
            sr_datasets["valid"], batch_size=args.batch_size, shuffle=False
        ),
    }
    sr_model = Generator(
        scale_factor=cfg.upscale_factor,
        n_residual_block=args.n_residuals,
        base_channel=args.base_channels,
        scaler_x=args.scaler_x,
        scaler_y=args.scaler_y,
        ext_flag=False,
        residual_flag=True,
        in_channel=2,
    ).to(device)
    sr_optimizer = torch.optim.Adam(
        sr_model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999)
    )
    sr_scheduler = torch.optim.lr_scheduler.StepLR(
        sr_optimizer, step_size=args.halving_interval, gamma=0.5
    )
    sr_history = []
    best_sr_rmse = float("inf")
    best_sr_epoch = 0
    sr_checkpoint = output_dir / "best_sr_model.pt"
    for epoch in range(1, args.sr_epochs + 1):
        train_loss = train_sr_epoch(
            sr_model, sr_loaders["train"], sr_optimizer, device
        )
        validation_rmse, validation_mae = evaluate_sr(
            sr_model, sr_loaders["valid"], device, args.scaler_y
        )
        row = {
            "epoch": epoch,
            "learning_rate": sr_optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "validation_RMSE": validation_rmse,
            "validation_MAE": validation_mae,
        }
        sr_history.append(row)
        write_csv(output_dir / "sr_history.csv", sr_history, list(row))
        if validation_rmse < best_sr_rmse:
            best_sr_rmse = validation_rmse
            best_sr_epoch = epoch
            torch.save(
                {
                    "model_state": sr_model.state_dict(),
                    "epoch": epoch,
                    "validation_RMSE": validation_rmse,
                },
                sr_checkpoint,
            )
        sr_scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.sr_epochs:
            print(
                f"SR {cfg.paper_name} seed={args.seed} epoch={epoch}/"
                f"{args.sr_epochs} loss={train_loss:.6f} "
                f"val_RMSE={validation_rmse:.6f}"
            )

    sr_model.load_state_dict(
        torch.load(sr_checkpoint, map_location=device)["model_state"]
    )
    generated = {
        split: generate_fine_series(
            sr_model,
            arrays[split][0],
            device,
            args.scaler_x,
            args.scaler_y,
            args.batch_size,
        )
        for split in ("train", "valid", "test")
    }
    fine_datasets = {
        split: build_fine_temporal_dataset(
            generated[split], arrays[split][1], cfg, args.scaler_y
        )
        for split in ("train", "valid", "test")
    }
    prediction_generator = torch.Generator().manual_seed(args.seed)
    prediction_loaders = {
        "train": DataLoader(
            fine_datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            generator=prediction_generator,
        ),
        "valid": DataLoader(
            fine_datasets["valid"], batch_size=args.batch_size, shuffle=False
        ),
        "test": DataLoader(
            fine_datasets["test"], batch_size=args.batch_size, shuffle=False
        ),
    }
    predictor = TransAm(
        in_channel=2,
        feature_size=64,
        hid_dim=128,
        n_heads=2 if cfg.key in {"TaxiBJ_P2", "TaxiBJ_P4"} else 4,
        dim_head=8,
        skip_dim=128,
        num_layers=1 if cfg.key != "TaxiBJ_P1" else 4,
        len_clossness=cfg.len_closeness,
        len_period=cfg.len_period,
        len_trend=cfg.len_trend,
        map_heigh=cfg.map_height * cfg.upscale_factor,
        map_width=cfg.map_width * cfg.upscale_factor,
        ext_flag=False,
        external_dim=7,
        dropout=0,
    ).to(device)
    predictor_optimizer = torch.optim.Adam(
        predictor.parameters(), lr=args.learning_rate, betas=(0.9, 0.999)
    )
    predictor_scheduler = torch.optim.lr_scheduler.StepLR(
        predictor_optimizer, step_size=args.halving_interval, gamma=0.5
    )
    predictor_history = []
    best_prediction_rmse = float("inf")
    best_prediction_epoch = 0
    predictor_checkpoint = output_dir / "best_fine_predictor.pt"
    for epoch in range(1, args.prediction_epochs + 1):
        train_loss = train_predictor_epoch(
            predictor,
            prediction_loaders["train"],
            predictor_optimizer,
            device,
        )
        validation_rmse, validation_mae, _, _ = evaluate_predictor(
            predictor,
            prediction_loaders["valid"],
            device,
            args.scaler_y,
        )
        row = {
            "epoch": epoch,
            "learning_rate": predictor_optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "validation_RMSE": validation_rmse,
            "validation_MAE": validation_mae,
        }
        predictor_history.append(row)
        write_csv(output_dir / "prediction_history.csv", predictor_history, list(row))
        if validation_rmse < best_prediction_rmse:
            best_prediction_rmse = validation_rmse
            best_prediction_epoch = epoch
            torch.save(
                {
                    "model_state": predictor.state_dict(),
                    "epoch": epoch,
                    "validation_RMSE": validation_rmse,
                },
                predictor_checkpoint,
            )
        predictor_scheduler.step()
        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == args.prediction_epochs
        ):
            print(
                f"Fine prediction {cfg.paper_name} seed={args.seed} "
                f"epoch={epoch}/{args.prediction_epochs} loss={train_loss:.6f} "
                f"val_RMSE={validation_rmse:.6f}"
            )

    predictor.load_state_dict(
        torch.load(predictor_checkpoint, map_location=device)["model_state"]
    )
    test_rmse, test_mae, test_samples, prediction_arrays = evaluate_predictor(
        predictor,
        prediction_loaders["test"],
        device,
        args.scaler_y,
        args.save_predictions,
    )
    metrics = {
        "dataset": cfg.paper_name,
        "dataset_key": args.dataset,
        "method": "FUPSI_IN_reimplementation",
        "seed": args.seed,
        "RMSE": f"{test_rmse:.6f}",
        "MAE": f"{test_mae:.6f}",
        "best_sr_epoch": best_sr_epoch,
        "best_prediction_epoch": best_prediction_epoch,
        "validation_RMSE": f"{best_prediction_rmse:.6f}",
        "test_samples": test_samples,
        "device": str(device),
    }
    write_csv(output_dir / "test_metrics.csv", [metrics], list(metrics))
    if prediction_arrays is not None:
        np.save(output_dir / "test_fine_prediction.npy", prediction_arrays[0])
        np.save(output_dir / "test_fine_target.npy", prediction_arrays[1])
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "protocol": "MainSeed-RawCount-v2",
                "order": "super-resolution before fine-grid prediction",
                "dataset": args.dataset,
                "data_alias": data_alias,
                "seed": args.seed,
                "scaler_x": args.scaler_x,
                "scaler_y": args.scaler_y,
                "sr_epochs": args.sr_epochs,
                "prediction_epochs": args.prediction_epochs,
                "residual_flag": True,
                "checkpoint_selection": "validation RMSE for both stages",
                "normalization": {
                    "coarse_divisor": args.scaler_x,
                    "fine_divisor": args.scaler_y,
                    "metrics": "raw counts after inverse scaling",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
