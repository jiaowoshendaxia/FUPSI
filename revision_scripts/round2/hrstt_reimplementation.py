#!/usr/bin/env python3
"""Documented HRSTT reimplementation for the unified MainSeed protocol.

The original HRSTT paper describes three-view temporal residual encoders,
Transformer-based spatio-temporal modeling, external-factor fusion, and
distributional upsampling. No validated official implementation was found
during the reproducibility audit. This script therefore provides an explicit
reimplementation for the flow-only controlled comparison:

* the same chronological splits and temporal windows as FUPSI;
* residual convolutional encoding for closeness/period/trend views;
* a Transformer encoder over coarse-grid spatial tokens for each view;
* learned view fusion;
* coarse-flow prediction and N2-normalized distributional upsampling;
* validation-selected checkpoints and raw-count fine-grid RMSE/MAE.

The manuscript and response letter must label results from this script as
"HRSTT (reimplementation)" rather than implying use of official code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class DatasetConfig:
    paper_name: str
    key: str
    upscale_factor: int
    map_height: int
    map_width: int
    day_len: int
    len_closeness: int
    len_period: int
    len_trend: int


DATASETS = {
    "TaxiBJ_P1": DatasetConfig("TaxiBJ P1", "TaxiBJ_P1", 4, 8, 8, 48, 3, 5, 0),
    "TaxiBJ_P2": DatasetConfig("TaxiBJ P2", "TaxiBJ_P2", 4, 8, 8, 48, 3, 1, 0),
    "TaxiBJ_P3": DatasetConfig("TaxiBJ P3", "TaxiBJ_P3", 4, 8, 8, 48, 3, 2, 0),
    "TaxiBJ_P4": DatasetConfig("TaxiBJ P4", "TaxiBJ_P4", 4, 8, 8, 48, 3, 3, 0),
    "BikeNYC": DatasetConfig("BikeNYC", "BikeNYC", 2, 8, 4, 24, 3, 5, 0),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_split(data_root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    root = data_root / split
    x = np.load(root / "X.npy").astype(np.float32)
    y = np.load(root / "Y.npy").astype(np.float32)
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError(f"Expected [T,C,H,W], got X={x.shape}, Y={y.shape}")
    if len(x) != len(y):
        raise ValueError(f"Unaligned X/Y arrays: {len(x)} versus {len(y)}")
    return x, y


def build_temporal_dataset(
    x: np.ndarray,
    y: np.ndarray,
    cfg: DatasetConfig,
    scaler_x: float,
    scaler_y: float,
) -> TensorDataset:
    max_lag = max(
        cfg.len_closeness,
        cfg.len_period * cfg.day_len,
        cfg.len_trend * cfg.day_len * 7,
    )
    block_len = max_lag + 1
    sample_count = len(x) - block_len + 1
    if sample_count <= 0:
        raise ValueError(
            f"{cfg.paper_name}: {len(x)} slots cannot support lag {max_lag}"
        )
    channels = x.shape[1]

    def empty_view(length: int) -> np.ndarray:
        return np.empty(
            (
                sample_count,
                length,
                channels,
                cfg.map_height,
                cfg.map_width,
            ),
            dtype=np.float32,
        )

    xc = empty_view(cfg.len_closeness)
    xp = empty_view(cfg.len_period)
    xt = empty_view(cfg.len_trend)
    coarse_target = np.empty(
        (sample_count, channels, cfg.map_height, cfg.map_width), dtype=np.float32
    )
    fine_target = np.empty(
        (
            sample_count,
            channels,
            cfg.map_height * cfg.upscale_factor,
            cfg.map_width * cfg.upscale_factor,
        ),
        dtype=np.float32,
    )

    for sample_index in range(sample_count):
        target_index = sample_index + block_len - 1
        for offset in range(cfg.len_closeness):
            xc[sample_index, offset] = x[
                target_index - (cfg.len_closeness - offset)
            ]
        for offset in range(cfg.len_period):
            xp[sample_index, offset] = x[
                target_index - (cfg.len_period - offset) * cfg.day_len
            ]
        for offset in range(cfg.len_trend):
            xt[sample_index, offset] = x[
                target_index - (cfg.len_trend - offset) * cfg.day_len * 7
            ]
        coarse_target[sample_index] = x[target_index]
        fine_target[sample_index] = y[target_index]

    return TensorDataset(
        torch.from_numpy(xc / scaler_x),
        torch.from_numpy(xp / scaler_x),
        torch.from_numpy(xt / scaler_x),
        torch.from_numpy(coarse_target / scaler_x),
        torch.from_numpy(fine_target / scaler_y),
    )


class ResidualUnit(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.PReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ViewEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        feature_dim: int,
        map_height: int,
        map_width: int,
        residual_blocks: int,
        transformer_layers: int,
        transformer_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.map_height = map_height
        self.map_width = map_width
        self.input = nn.Sequential(
            nn.Conv2d(input_channels, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.PReLU(feature_dim),
        )
        self.residual = nn.Sequential(
            *(ResidualUnit(feature_dim) for _ in range(residual_blocks))
        )
        self.position = nn.Parameter(
            torch.zeros(1, map_height * map_width, feature_dim)
        )
        nn.init.trunc_normal_(self.position, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=transformer_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, view: torch.Tensor) -> torch.Tensor:
        batch, length, channels, height, width = view.shape
        encoded = self.residual(self.input(view.reshape(batch, length * channels, height, width)))
        tokens = encoded.flatten(2).transpose(1, 2) + self.position
        tokens = self.output_norm(self.transformer(tokens))
        return tokens.transpose(1, 2).reshape(
            batch, -1, self.map_height, self.map_width
        )


class N2Normalization(nn.Module):
    def __init__(self, upscale_factor: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor
        self.epsilon = epsilon

    def forward(self, raw_density: torch.Tensor) -> torch.Tensor:
        nonnegative = F.relu(raw_density)
        coarse_sum = F.avg_pool2d(
            nonnegative,
            kernel_size=self.upscale_factor,
            stride=self.upscale_factor,
        ) * (self.upscale_factor**2)
        expanded_sum = F.interpolate(
            coarse_sum,
            scale_factor=self.upscale_factor,
            mode="nearest",
        )
        return nonnegative / (expanded_sum + self.epsilon)


class DistributionalUpsampling(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        channels: int,
        upscale_factor: int,
        scaler_x: float,
        scaler_y: float,
    ) -> None:
        super().__init__()
        if upscale_factor not in (2, 4):
            raise ValueError(f"Unsupported upscale factor: {upscale_factor}")
        blocks: list[nn.Module] = []
        for _ in range(int(math.log2(upscale_factor))):
            blocks.extend(
                [
                    nn.Conv2d(feature_dim, feature_dim * 4, 3, padding=1),
                    nn.PixelShuffle(2),
                    nn.PReLU(feature_dim),
                ]
            )
        self.upsampling = nn.Sequential(*blocks)
        self.density_head = nn.Conv2d(feature_dim, channels, 3, padding=1)
        self.normalize = N2Normalization(upscale_factor)
        self.upscale_factor = upscale_factor
        self.scaler_ratio = scaler_x / scaler_y

    def forward(
        self, fused_features: torch.Tensor, coarse_prediction: torch.Tensor
    ) -> torch.Tensor:
        raw_density = self.density_head(self.upsampling(fused_features))
        density = self.normalize(raw_density)
        coarse_upsampled = F.interpolate(
            F.relu(coarse_prediction),
            scale_factor=self.upscale_factor,
            mode="nearest",
        )
        return coarse_upsampled * self.scaler_ratio * density


class HRSTTReimplementation(nn.Module):
    def __init__(
        self,
        cfg: DatasetConfig,
        channels: int,
        feature_dim: int,
        residual_blocks: int,
        transformer_layers: int,
        transformer_heads: int,
        dropout: float,
        scaler_x: float,
        scaler_y: float,
    ) -> None:
        super().__init__()
        view_lengths = {
            "closeness": cfg.len_closeness,
            "period": cfg.len_period,
            "trend": cfg.len_trend,
        }
        self.view_names = [
            name for name, length in view_lengths.items() if length > 0
        ]
        self.encoders = nn.ModuleDict(
            {
                name: ViewEncoder(
                    input_channels=view_lengths[name] * channels,
                    feature_dim=feature_dim,
                    map_height=cfg.map_height,
                    map_width=cfg.map_width,
                    residual_blocks=residual_blocks,
                    transformer_layers=transformer_layers,
                    transformer_heads=transformer_heads,
                    dropout=dropout,
                )
                for name in self.view_names
            }
        )
        self.view_logits = nn.Parameter(torch.zeros(len(self.view_names)))
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.PReLU(feature_dim),
        )
        self.coarse_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.PReLU(feature_dim),
            nn.Conv2d(feature_dim, channels, 1),
            nn.ReLU(),
        )
        self.distributional_upsampling = DistributionalUpsampling(
            feature_dim,
            channels,
            cfg.upscale_factor,
            scaler_x,
            scaler_y,
        )

    def forward(
        self, xc: torch.Tensor, xp: torch.Tensor, xt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        view_inputs = {"closeness": xc, "period": xp, "trend": xt}
        encoded = [
            self.encoders[name](view_inputs[name]) for name in self.view_names
        ]
        weights = torch.softmax(self.view_logits, dim=0)
        fused = sum(weight * feature for weight, feature in zip(weights, encoded))
        fused = self.fusion(fused)
        coarse_prediction = self.coarse_head(fused)
        fine_prediction = self.distributional_upsampling(
            fused, coarse_prediction
        )
        return fine_prediction, coarse_prediction


def raw_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler_y: float,
) -> tuple[float, float, int]:
    model.eval()
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    value_count = 0
    sample_count = 0
    with torch.no_grad():
        for xc, xp, xt, _coarse_target, fine_target in loader:
            xc = xc.to(device)
            xp = xp.to(device)
            xt = xt.to(device)
            fine_target = fine_target.to(device)
            fine_prediction, _ = model(xc, xp, xt)
            error = (fine_prediction - fine_target) * scaler_y
            squared_error_sum += float(torch.sum(error * error).cpu())
            absolute_error_sum += float(torch.sum(torch.abs(error)).cpu())
            value_count += error.numel()
            sample_count += len(xc)
    return (
        math.sqrt(squared_error_sum / value_count),
        absolute_error_sum / value_count,
        sample_count,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_coarse: float,
) -> tuple[float, float, float]:
    model.train()
    fine_sum = 0.0
    coarse_sum = 0.0
    total_sum = 0.0
    sample_count = 0
    for xc, xp, xt, coarse_target, fine_target in loader:
        xc = xc.to(device)
        xp = xp.to(device)
        xt = xt.to(device)
        coarse_target = coarse_target.to(device)
        fine_target = fine_target.to(device)
        optimizer.zero_grad(set_to_none=True)
        fine_prediction, coarse_prediction = model(xc, xp, xt)
        fine_loss = F.mse_loss(fine_prediction, fine_target)
        coarse_loss = F.mse_loss(coarse_prediction, coarse_target)
        loss = fine_loss + lambda_coarse * coarse_loss
        loss.backward()
        optimizer.step()
        batch = len(xc)
        fine_sum += float(fine_loss.detach().cpu()) * batch
        coarse_sum += float(coarse_loss.detach().cpu()) * batch
        total_sum += float(loss.detach().cpu()) * batch
        sample_count += batch
    return (
        total_sum / sample_count,
        fine_sum / sample_count,
        coarse_sum / sample_count,
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--data-prefix", default="MainSeed")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--halving-interval", type=int, default=20)
    parser.add_argument("--lambda-coarse", type=float, default=0.01)
    parser.add_argument("--scaler-x", type=float, default=1500.0)
    parser.add_argument("--scaler-y", type=float, default=100.0)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("revision/round2/hrstt"),
    )
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = DATASETS[args.dataset]
    code_root = args.code_root.resolve()
    data_alias = f"{args.data_prefix}_{cfg.key}"
    data_root = code_root / "data" / data_alias
    output_dir = args.output_root.resolve() / cfg.key / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {}
    for split in ("train", "valid", "test"):
        x, y = load_split(data_root, split)
        expected_x = (2, cfg.map_height, cfg.map_width)
        expected_y = (
            2,
            cfg.map_height * cfg.upscale_factor,
            cfg.map_width * cfg.upscale_factor,
        )
        if x.shape[1:] != expected_x or y.shape[1:] != expected_y:
            raise ValueError(
                f"{cfg.paper_name} {split}: X={x.shape}, Y={y.shape}, "
                f"expected {expected_x}/{expected_y}"
            )
        datasets[split] = build_temporal_dataset(
            x, y, cfg, args.scaler_x, args.scaler_y
        )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.num_workers,
        ),
        "valid": DataLoader(
            datasets["valid"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HRSTTReimplementation(
        cfg=cfg,
        channels=2,
        feature_dim=args.feature_dim,
        residual_blocks=args.residual_blocks,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        dropout=args.dropout,
        scaler_x=args.scaler_x,
        scaler_y=args.scaler_y,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.halving_interval,
        gamma=0.5,
    )

    best_validation_rmse = float("inf")
    best_epoch = 0
    history: list[dict] = []
    checkpoint_path = output_dir / "best_model.pt"
    serializable_arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    for epoch in range(1, args.epochs + 1):
        total_loss, fine_loss, coarse_loss = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            args.lambda_coarse,
        )
        validation_rmse, validation_mae, _ = raw_metrics(
            model, loaders["valid"], device, args.scaler_y
        )
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_total_loss": total_loss,
            "train_fine_loss": fine_loss,
            "train_coarse_loss": coarse_loss,
            "validation_RMSE": validation_rmse,
            "validation_MAE": validation_mae,
        }
        history.append(row)
        write_csv(output_dir / "training_history.csv", history, list(row))
        if validation_rmse < best_validation_rmse:
            best_validation_rmse = validation_rmse
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "dataset": args.dataset,
                    "data_alias": data_alias,
                    "seed": args.seed,
                    "epoch": epoch,
                    "validation_RMSE": validation_rmse,
                    "config": asdict(cfg),
                    "arguments": serializable_arguments,
                },
                checkpoint_path,
            )
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"{cfg.paper_name} seed={args.seed} epoch={epoch}/{args.epochs} "
                f"loss={total_loss:.6f} val_RMSE={validation_rmse:.6f} "
                f"val_MAE={validation_mae:.6f} best={best_validation_rmse:.6f}"
            )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_rmse, test_mae, test_samples = raw_metrics(
        model, loaders["test"], device, args.scaler_y
    )
    metrics = {
        "dataset": cfg.paper_name,
        "dataset_key": args.dataset,
        "method": "HRSTT_reimplementation",
        "seed": args.seed,
        "RMSE": f"{test_rmse:.6f}",
        "MAE": f"{test_mae:.6f}",
        "best_epoch": best_epoch,
        "validation_RMSE": f"{best_validation_rmse:.6f}",
        "test_samples": test_samples,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "checkpoint": str(checkpoint_path),
    }
    write_csv(output_dir / "test_metrics.csv", [metrics], list(metrics))

    metadata = {
        "protocol": "MainSeed-RawCount-v2",
        "method": "HRSTT documented reimplementation",
        "official_code_used": False,
        "dataset_config": asdict(cfg),
        "arguments": serializable_arguments,
        "normalization": {
            "coarse_divisor": args.scaler_x,
            "fine_divisor": args.scaler_y,
            "metrics": "raw counts after inverse scaling",
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    if args.save_predictions:
        model.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for xc, xp, xt, _coarse_target, fine_target in loaders["test"]:
                fine_prediction, _ = model(
                    xc.to(device), xp.to(device), xt.to(device)
                )
                predictions.append((fine_prediction.cpu().numpy() * args.scaler_y))
                targets.append(fine_target.numpy() * args.scaler_y)
        np.save(output_dir / "test_fine_prediction.npy", np.concatenate(predictions))
        np.save(output_dir / "test_fine_target.npy", np.concatenate(targets))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
