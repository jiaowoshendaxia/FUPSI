#!/usr/bin/env python3
"""Evaluate the full FUPSI pipeline under synthetic sparse inputs.

The evaluator follows the frozen MainSeed-RawCount-v2 protocol:

    mask historical coarse maps
      -> select completion on validation data
      -> complete validation/test histories
      -> coarse forecasting
      -> residual super-resolution
      -> raw-count fine-grid RMSE and MAE

Each model seed is paired with the same mask seed. Completion selection is
performed independently for every dataset and missing rate by averaging
validation completion MSE over the requested seeds. The selected operator is
then fixed for all test evaluations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_SEEDS = (2024, 2025, 2026)
DEFAULT_RATES = (0.0, 0.1, 0.3, 0.5, 0.7)
VALIDATION_CANDIDATES = (
    "no_completion",
    "zero_fill",
    "mean_fill",
    "linear_interpolation",
    "knn_fill",
    "svd_completion",
)


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
    n_heads: int
    num_layers: int
    lamda_p: float


DATASETS = {
    "TaxiBJ_P1": DatasetConfig(
        "TaxiBJ P1", "TaxiBJ_P1", 4, 8, 8, 48, 3, 5, 0, 4, 4, 0.01
    ),
    "TaxiBJ_P2": DatasetConfig(
        "TaxiBJ P2", "TaxiBJ_P2", 4, 8, 8, 48, 3, 1, 0, 2, 1, 0.9
    ),
    "TaxiBJ_P3": DatasetConfig(
        "TaxiBJ P3", "TaxiBJ_P3", 4, 8, 8, 48, 3, 2, 0, 4, 1, 0.01
    ),
    "TaxiBJ_P4": DatasetConfig(
        "TaxiBJ P4", "TaxiBJ_P4", 4, 8, 8, 48, 3, 3, 0, 2, 1, 0.01
    ),
    "BikeNYC": DatasetConfig(
        "BikeNYC", "BikeNYC", 2, 8, 4, 24, 3, 5, 0, 4, 1, 0.9
    ),
}


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_str_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_mask(shape: tuple[int, ...], missing_rate: float, seed: int) -> np.ndarray:
    if missing_rate <= 0:
        return np.ones(shape, dtype=np.float32)
    rng = np.random.default_rng(seed)
    return (rng.random(shape) >= missing_rate).astype(np.float32)


def no_completion(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return data * mask


def zero_fill(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return data * mask


def mean_fill(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill from past-only cell means with a current-frame spatial fallback."""
    output = np.empty_like(data, dtype=np.float32)
    running_sum = np.zeros(data.shape[1:], dtype=np.float64)
    running_count = np.zeros(data.shape[1:], dtype=np.int64)
    for time_index in range(data.shape[0]):
        frame = data[time_index]
        observed = mask[time_index] > 0
        fallback = np.zeros((data.shape[1], 1, 1), dtype=np.float32)
        for channel in range(data.shape[1]):
            channel_observed = observed[channel]
            if channel_observed.any():
                fallback[channel, 0, 0] = float(
                    frame[channel][channel_observed].mean()
                )
        historical_mean = np.divide(
            running_sum,
            running_count,
            out=np.broadcast_to(fallback, frame.shape).astype(np.float64).copy(),
            where=running_count > 0,
        )
        output[time_index] = np.where(observed, frame, historical_mean)
        running_sum[observed] += frame[observed]
        running_count[observed] += 1
    return output.astype(np.float32)


def linear_interpolation(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Causal linear temporal filling using only earlier observations."""
    output = np.empty_like(data, dtype=np.float32)
    spatial_shape = data.shape[1:]
    last_time = np.full(spatial_shape, -1, dtype=np.int64)
    previous_time = np.full(spatial_shape, -1, dtype=np.int64)
    last_value = np.zeros(spatial_shape, dtype=np.float64)
    previous_value = np.zeros(spatial_shape, dtype=np.float64)

    for time_index in range(data.shape[0]):
        frame = data[time_index]
        observed = mask[time_index] > 0
        fallback = np.zeros((data.shape[1], 1, 1), dtype=np.float64)
        for channel in range(data.shape[1]):
            channel_observed = observed[channel]
            if channel_observed.any():
                fallback[channel, 0, 0] = float(
                    frame[channel][channel_observed].mean()
                )

        estimate = np.broadcast_to(fallback, frame.shape).copy()
        one_previous = last_time >= 0
        estimate[one_previous] = last_value[one_previous]
        two_previous = previous_time >= 0
        denominator = np.maximum(last_time - previous_time, 1)
        slope = (last_value - previous_value) / denominator
        estimate[two_previous] = (
            last_value[two_previous]
            + slope[two_previous] * (time_index - last_time[two_previous])
        )
        estimate = np.maximum(estimate, 0.0)
        output[time_index] = np.where(observed, frame, estimate)

        previous_time[observed] = last_time[observed]
        previous_value[observed] = last_value[observed]
        last_time[observed] = time_index
        last_value[observed] = frame[observed]
    return output.astype(np.float32)


def knn_fill(data: np.ndarray, mask: np.ndarray, radius: int = 2) -> np.ndarray:
    from scipy.ndimage import convolve

    output = data.copy()
    offsets = np.arange(-radius, radius + 1)
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    for row_index, row_offset in enumerate(offsets):
        for col_index, col_offset in enumerate(offsets):
            distance = abs(int(row_offset)) + abs(int(col_offset))
            kernel[row_index, col_index] = 1.0 / (distance + 1.0)

    for time_index in range(data.shape[0]):
        for channel in range(data.shape[1]):
            frame = data[time_index, channel]
            observed = mask[time_index, channel] > 0
            observed_float = observed.astype(np.float32)
            weighted_sum = convolve(frame * observed_float, kernel, mode="nearest")
            weights = convolve(observed_float, kernel, mode="nearest")
            fallback = float(frame[observed].mean()) if observed.any() else 0.0
            estimate = np.divide(
                weighted_sum,
                weights,
                out=np.full_like(frame, fallback),
                where=weights > 0,
            )
            output[time_index, channel, ~observed] = estimate[~observed]
    return output.astype(np.float32)


def svd_completion(data: np.ndarray, mask: np.ndarray, rank: int = 8) -> np.ndarray:
    output = data.copy()
    for time_index in range(data.shape[0]):
        for channel in range(data.shape[1]):
            matrix = data[time_index, channel]
            observed = mask[time_index, channel] > 0
            frame_mean = float(matrix[observed].mean()) if observed.any() else 0.0
            initialized = np.where(observed, matrix, frame_mean)
            u, singular_values, vh = np.linalg.svd(
                initialized, full_matrices=False
            )
            used_rank = max(1, min(rank, len(singular_values)))
            reconstructed = (
                u[:, :used_rank] * singular_values[:used_rank]
            ) @ vh[:used_rank]
            output[time_index, channel] = np.where(
                observed, matrix, reconstructed
            )
    return output.astype(np.float32)


COMPLETION_METHODS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "no_completion": no_completion,
    "zero_fill": zero_fill,
    "mean_fill": mean_fill,
    "linear_interpolation": linear_interpolation,
    "knn_fill": knn_fill,
    "svd_completion": svd_completion,
}


def completion_mse(
    original: np.ndarray, completed: np.ndarray, mask: np.ndarray
) -> float:
    missing = mask <= 0
    if not missing.any():
        return 0.0
    error = completed[missing] - original[missing]
    return float(np.mean(error * error))


def load_split(data_root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    split_root = data_root / split
    x = np.load(split_root / "X.npy").astype(np.float32)
    y = np.load(split_root / "Y.npy").astype(np.float32)
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError(f"Expected [T,C,H,W], received X={x.shape}, Y={y.shape}")
    if len(x) != len(y):
        raise ValueError(f"Unaligned X/Y lengths: {len(x)} versus {len(y)}")
    return x, y


def build_temporal_dataset(
    completed_x: np.ndarray,
    y: np.ndarray,
    cfg: DatasetConfig,
    scaler_x: float,
    scaler_y: float,
) -> TensorDataset:
    length_offsets = np.array(
        [
            cfg.len_closeness,
            cfg.len_period * cfg.day_len,
            cfg.len_trend * cfg.day_len * 7,
        ]
    )
    max_lag = int(length_offsets.max())
    block_len = max_lag + 1
    sample_count = len(completed_x) - block_len + 1
    if sample_count <= 0:
        raise ValueError(
            f"{cfg.paper_name}: split length {len(completed_x)} is shorter than "
            f"the required temporal block {block_len}"
        )

    channels = completed_x.shape[1]
    xc = np.empty(
        (
            sample_count,
            cfg.len_closeness,
            channels,
            cfg.map_height,
            cfg.map_width,
        ),
        dtype=np.float32,
    )
    xp = np.empty(
        (
            sample_count,
            cfg.len_period,
            channels,
            cfg.map_height,
            cfg.map_width,
        ),
        dtype=np.float32,
    )
    xt = np.empty(
        (
            sample_count,
            cfg.len_trend,
            channels,
            cfg.map_height,
            cfg.map_width,
        ),
        dtype=np.float32,
    )
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
            source_index = target_index - (cfg.len_closeness - offset)
            xc[sample_index, offset] = completed_x[source_index]
        for offset in range(cfg.len_period):
            source_index = target_index - (cfg.len_period - offset) * cfg.day_len
            xp[sample_index, offset] = completed_x[source_index]
        for offset in range(cfg.len_trend):
            source_index = (
                target_index - (cfg.len_trend - offset) * cfg.day_len * 7
            )
            xt[sample_index, offset] = completed_x[source_index]
        coarse_target[sample_index] = completed_x[target_index]
        fine_target[sample_index] = y[target_index]

    ext = np.zeros((sample_count,), dtype=np.float32)
    return TensorDataset(
        torch.from_numpy(xc / scaler_x),
        torch.from_numpy(xp / scaler_x),
        torch.from_numpy(xt / scaler_x),
        torch.from_numpy(ext),
        torch.from_numpy(coarse_target / scaler_x),
        torch.from_numpy(fine_target / scaler_y),
    )


def model_checkpoint_paths(
    code_root: Path,
    dataset_alias: str,
    cfg: DatasetConfig,
    seed: int,
    epochs: int,
    n_residuals: int,
    base_channels: int,
    lamda_s: float,
) -> tuple[Path, Path]:
    suffix = (
        f"{n_residuals}-{base_channels}-{epochs}_"
        f"{cfg.len_closeness}{cfg.len_period}{cfg.len_trend}_"
        f"{cfg.n_heads}_{cfg.num_layers}"
    )
    root = (
        code_root
        / "saved_model"
        / "to_stage"
        / "no_ext(r)"
        / dataset_alias
        / f"seed{seed}"
        / f"{cfg.lamda_p}_{lamda_s}"
        / "-4-6"
        / suffix
    )
    return root / "cpt" / "final_model.pt", root / "Generator" / "final_model.pt"


def load_models(
    code_root: Path,
    dataset_alias: str,
    cfg: DatasetConfig,
    seed: int,
    device: torch.device,
    epochs: int,
    scaler_x: float,
    scaler_y: float,
    n_residuals: int,
    base_channels: int,
    lamda_s: float,
    residual_flag: bool,
):
    sys.path.insert(0, str(code_root))
    try:
        from prediction import TransAm
        from UrbanSG import Generator
    finally:
        sys.path.pop(0)

    predictor = TransAm(
        in_channel=2,
        feature_size=64,
        hid_dim=128,
        n_heads=cfg.n_heads,
        dim_head=8,
        skip_dim=128,
        num_layers=cfg.num_layers,
        len_clossness=cfg.len_closeness,
        len_period=cfg.len_period,
        len_trend=cfg.len_trend,
        map_heigh=cfg.map_height,
        map_width=cfg.map_width,
        ext_flag=False,
        external_dim=7,
        dropout=0,
    )
    generator = Generator(
        scale_factor=cfg.upscale_factor,
        n_residual_block=n_residuals,
        base_channel=base_channels,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        ext_flag=False,
        residual_flag=residual_flag,
        in_channel=2,
    )
    predictor_path, generator_path = model_checkpoint_paths(
        code_root,
        dataset_alias,
        cfg,
        seed,
        epochs,
        n_residuals,
        base_channels,
        lamda_s,
    )
    if not predictor_path.exists() or not generator_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for {dataset_alias} seed {seed}: "
            f"{predictor_path} / {generator_path}"
        )
    predictor.load_state_dict(torch.load(predictor_path, map_location=device))
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    predictor.to(device).eval()
    generator.to(device).eval()
    return predictor, generator, predictor_path, generator_path


@torch.no_grad()
def evaluate_pipeline(
    predictor,
    generator,
    dataset: TensorDataset,
    device: torch.device,
    scaler_y: float,
    batch_size: int,
) -> tuple[float, float, int]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    value_count = 0
    for xc, xp, xt, ext, _coarse_target, fine_target in loader:
        xc = xc.to(device)
        xp = xp.to(device)
        xt = xt.to(device)
        ext = ext.to(device)
        fine_target = fine_target.to(device)
        coarse_prediction = predictor(xc, xp, xt, ext)
        fine_prediction = generator(coarse_prediction, ext)
        error = (fine_prediction - fine_target) * scaler_y
        squared_error_sum += float(torch.sum(error * error).cpu())
        absolute_error_sum += float(torch.sum(torch.abs(error)).cpu())
        value_count += error.numel()
    rmse = math.sqrt(squared_error_sum / value_count)
    mae = absolute_error_sum / value_count
    return rmse, mae, len(dataset)


def select_completion_methods(
    code_root: Path,
    data_prefix: str,
    dataset_keys: Iterable[str],
    rates: Iterable[float],
    seeds: Iterable[int],
    output_path: Path,
) -> dict[tuple[str, float], str]:
    selected: dict[tuple[str, float], str] = {}
    evidence: dict[str, dict[str, dict[str, float] | str]] = {}
    for dataset_key in dataset_keys:
        cfg = DATASETS[dataset_key]
        data_alias = f"{data_prefix}_{cfg.key}"
        x_validation, _ = load_split(code_root / "data" / data_alias, "valid")
        evidence[dataset_key] = {}
        for rate in rates:
            if rate <= 0:
                selected[(dataset_key, rate)] = "no_completion"
                evidence[dataset_key][str(rate)] = {
                    "selected": "no_completion",
                    "no_completion": 0.0,
                }
                continue
            method_scores: dict[str, list[float]] = {
                method: [] for method in VALIDATION_CANDIDATES
            }
            for seed in seeds:
                mask = generate_mask(x_validation.shape, rate, seed)
                for method in VALIDATION_CANDIDATES:
                    completed = COMPLETION_METHODS[method](x_validation, mask)
                    method_scores[method].append(
                        completion_mse(x_validation, completed, mask)
                    )
            means = {
                method: float(np.mean(values))
                for method, values in method_scores.items()
            }
            best_method = min(means, key=lambda method: (means[method], method))
            selected[(dataset_key, rate)] = best_method
            evidence[dataset_key][str(rate)] = {
                "selected": best_method,
                **means,
            }
            print(
                f"validation selection: {cfg.paper_name} rate={rate:.1f} "
                f"method={best_method} mse={means[best_method]:.6f}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return selected


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    groups: dict[tuple[str, float, str, str], list[float]] = {}
    for row in rows:
        for metric in ("RMSE", "MAE"):
            key = (row["dataset"], row["missing_rate"], row["method"], metric)
            groups.setdefault(key, []).append(float(row[metric]))
    for (dataset, rate, method, metric), values in sorted(groups.items()):
        output.append(
            {
                "dataset": dataset,
                "missing_rate": rate,
                "method": method,
                "metric": metric,
                "runs": len(values),
                "mean": f"{np.mean(values):.6f}",
                "std": f"{np.std(values, ddof=1) if len(values) > 1 else 0.0:.6f}",
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-prefix", default="ResidualMainE300P5")
    parser.add_argument("--model-prefix", default="ResidualMainE300P5")
    parser.add_argument(
        "--datasets",
        default="TaxiBJ_P1,TaxiBJ_P2,TaxiBJ_P3,TaxiBJ_P4,BikeNYC",
    )
    parser.add_argument("--seeds", default="2024,2025,2026")
    parser.add_argument("--rates", default="0,0.1,0.3,0.5,0.7")
    parser.add_argument(
        "--methods",
        default="adaptive,no_completion",
        help="Comma-separated test methods. Use adaptive or a completion method.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--scaler-x", type=float, default=1500.0)
    parser.add_argument("--scaler-y", type=float, default=100.0)
    parser.add_argument("--n-residuals", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--lamda-s", type=float, default=0.1)
    parser.add_argument(
        "--residual-flag",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision/round2/sparse_pipeline"),
    )
    args = parser.parse_args()

    code_root = args.code_root.resolve()
    output_dir = args.output_dir.resolve()
    dataset_keys = parse_str_list(args.datasets)
    seeds = parse_int_list(args.seeds)
    rates = parse_float_list(args.rates)
    methods = parse_str_list(args.methods)
    for dataset_key in dataset_keys:
        if dataset_key not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_key}")
    for method in methods:
        if method != "adaptive" and method not in COMPLETION_METHODS:
            raise ValueError(f"Unknown method: {method}")

    selected = select_completion_methods(
        code_root,
        args.data_prefix,
        dataset_keys,
        rates,
        seeds,
        output_dir / "validation_selection.json",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    rows: list[dict] = []
    for dataset_key in dataset_keys:
        cfg = DATASETS[dataset_key]
        data_alias = f"{args.data_prefix}_{cfg.key}"
        model_alias = f"{args.model_prefix}_{cfg.key}"
        data_root = code_root / "data" / data_alias
        x_test, y_test = load_split(data_root, "test")
        expected_coarse = (2, cfg.map_height, cfg.map_width)
        expected_fine = (
            2,
            cfg.map_height * cfg.upscale_factor,
            cfg.map_width * cfg.upscale_factor,
        )
        if x_test.shape[1:] != expected_coarse or y_test.shape[1:] != expected_fine:
            raise ValueError(
                f"{cfg.paper_name}: protocol mismatch X={x_test.shape}, "
                f"Y={y_test.shape}, expected {expected_coarse}/{expected_fine}"
            )

        for seed in seeds:
            set_seed(seed)
            predictor, generator, predictor_path, generator_path = load_models(
                code_root,
                model_alias,
                cfg,
                seed,
                device,
                args.epochs,
                args.scaler_x,
                args.scaler_y,
                args.n_residuals,
                args.base_channels,
                args.lamda_s,
                args.residual_flag,
            )
            for rate in rates:
                mask = generate_mask(x_test.shape, rate, seed)
                for reported_method in methods:
                    completion_method = (
                        selected[(dataset_key, rate)]
                        if reported_method == "adaptive"
                        else reported_method
                    )
                    completed_x = COMPLETION_METHODS[completion_method](x_test, mask)
                    temporal_dataset = build_temporal_dataset(
                        completed_x,
                        y_test,
                        cfg,
                        args.scaler_x,
                        args.scaler_y,
                    )
                    rmse, mae, samples = evaluate_pipeline(
                        predictor,
                        generator,
                        temporal_dataset,
                        device,
                        args.scaler_y,
                        args.batch_size,
                    )
                    row = {
                        "dataset": cfg.paper_name,
                        "dataset_key": dataset_key,
                        "completion": reported_method,
                        "model_seed": seed,
                        "mask_seed": seed,
                        "fine_RMSE": f"{rmse:.6f}",
                        "fine_MAE": f"{mae:.6f}",
                        "seed": seed,
                        "missing_rate": rate,
                        "method": reported_method,
                        "completion_method": completion_method,
                        "completion_MSE": f"{completion_mse(x_test, completed_x, mask):.6f}",
                        "RMSE": f"{rmse:.6f}",
                        "MAE": f"{mae:.6f}",
                        "test_samples": samples,
                        "model_alias": model_alias,
                        "predictor_checkpoint": str(predictor_path),
                        "generator_checkpoint": str(generator_path),
                    }
                    rows.append(row)
                    print(
                        f"{cfg.paper_name} seed={seed} rate={rate:.1f} "
                        f"method={reported_method}/{completion_method} "
                        f"RMSE={rmse:.6f} MAE={mae:.6f}"
                    )
                    write_csv(
                        output_dir / "sparse_pipeline_seed_metrics.csv",
                        rows,
                        list(row),
                    )

    summary_rows = summarize(rows)
    write_csv(
        output_dir / "sparse_pipeline_mean_std.csv",
        summary_rows,
        ["dataset", "missing_rate", "method", "metric", "runs", "mean", "std"],
    )
    metadata = {
        "protocol": "MainSeed-RawCount-v2",
        "device": str(device),
        "data_prefix": args.data_prefix,
        "model_prefix": args.model_prefix,
        "datasets": list(dataset_keys),
        "seeds": list(seeds),
        "rates": list(rates),
        "methods": list(methods),
        "residual_flag": args.residual_flag,
        "causal_completion": True,
        "causal_completion_note": (
            "Every filled value uses only observations available at or before "
            "that time index; no future test observation is used."
        ),
        "normalization": {
            "coarse_divisor": args.scaler_x,
            "fine_divisor": args.scaler_y,
            "metrics": "raw counts after inverse scaling",
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(rows)} seed-level rows to {output_dir}")


if __name__ == "__main__":
    main()
