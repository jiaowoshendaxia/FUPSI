# -- coding:utf-8 --
"""
Expanded missing-rate study for sparse-aware FUPSI.

This script is designed for the experiment matrix suggested by the advisor:

    datasets      : TaxiBJ P1-P4, BikeNYC, or any processed dataset directory
    missing rates : 10%, 30%, 50%, 70%
    seeds         : at least 3 random seeds
    baselines     : FUPSI without completion, zero/mean/linear/KNN/SVD completion + FUPSI

Expected processed dataset layout:

    <dataset_dir>/
      test/
        X.npy   # [N, T, H, W] or [N, T, C, H, W]
        Y.npy   # [N, H, W] or [N, C, H, W]

The downstream FUPSI model defaults to the lightweight checkpoint architecture
used by the existing scripts in this folder. If no checkpoint is available, the
script can still run with --backend persistence for a smoke test, but paper
tables should use --backend checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn


DEFAULT_RATES = (0.1, 0.3, 0.5, 0.7)
DEFAULT_SEEDS = (2024, 2025, 2026)
ADAPTIVE_MISSING_RATE_THRESHOLD = 0.60
ADAPTIVE_GRID_SIZE_THRESHOLD = 800
DEFAULT_METHODS = (
    "proposed_sparse_aware",
    "no_completion",
    "zero_fill",
    "mean_fill",
    "linear_interpolation",
    "knn_fill",
    "adaptive_completion",
    "svd_completion",
)


class SimpleFUPSIGenerator(nn.Module):
    """Checkpoint-compatible predictor used by the existing real-data scripts."""

    def __init__(self, input_channels: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_dataset_specs(values: Iterable[str]) -> List[DatasetSpec]:
    specs: List[DatasetSpec] = []
    for value in values:
        if "=" in value:
            name, raw_path = value.split("=", 1)
        else:
            raw_path = value
            name = Path(value).name or "dataset"
        specs.append(DatasetSpec(name=name.strip(), path=Path(raw_path.strip())))
    return specs


def parse_named_paths(values: Iterable[str]) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Expected name=path for checkpoint mapping, got: {value}"
            )
        name, raw_path = value.split("=", 1)
        paths[name.strip()] = Path(raw_path.strip())
    return paths


def load_adaptive_plan(path: str | None) -> Dict[Tuple[str, float], str]:
    if not path:
        return {}
    plan_path = Path(path)
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    raw_plan = data.get("adaptive_plan", data)
    plan: Dict[Tuple[str, float], str] = {}
    for dataset, by_rate in raw_plan.items():
        for rate, method in by_rate.items():
            method_name = str(method)
            if method_name not in COMPLETION_METHODS:
                raise ValueError(
                    f"Adaptive plan uses unknown method {method_name!r} "
                    f"for {dataset} rate {rate}."
                )
            if method_name == "adaptive_completion":
                raise ValueError("Adaptive plan cannot recursively select adaptive_completion.")
            plan[(str(dataset), float(rate))] = method_name
    return plan


class SparseAwareFUPSIGenerator(nn.Module):
    """Mask-aware predictor trained with randomly masked inputs."""

    def __init__(self, input_channels: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels * 2, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(torch.cat([x * mask, mask], dim=1)))


def load_dataset(spec: DatasetSpec, split: str = "test") -> Tuple[np.ndarray, np.ndarray]:
    split_dir = spec.path / split
    x_path = split_dir / "X.npy"
    y_path = split_dir / "Y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"{spec.name}: expected {x_path} and {y_path}. "
            "Use processed directories such as data, TaxiBJ_P1, BikeNYC, etc."
        )

    x = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.float32)

    if x.ndim == 4:
        # [N, T, H, W]
        pass
    elif x.ndim == 5 and x.shape[2] == 1:
        x = x[:, :, 0]
    elif x.ndim == 5:
        # If multiple flow channels exist, use the first channel for the
        # checkpoint-compatible FUPSI backend and keep the temporal axis.
        x = x[:, :, 0]
    else:
        raise ValueError(f"{spec.name}: unsupported X shape {x.shape}")

    if y.ndim == 4 and y.shape[1] == 1:
        y = y[:, 0]
    elif y.ndim == 4:
        y = y[:, 0]
    elif y.ndim != 3:
        raise ValueError(f"{spec.name}: unsupported Y shape {y.shape}")

    if len(x) != len(y):
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]

    return x, y


def generate_mask(shape: Tuple[int, ...], missing_rate: float) -> np.ndarray:
    return (np.random.random(shape) >= missing_rate).astype(np.float32)


def no_completion(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return data * mask


def zero_fill(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    completed = data.copy()
    completed[mask == 0] = 0.0
    return completed


def mean_fill(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    completed = data.copy()
    observed = mask == 1
    fallback = float(data[observed].mean()) if observed.any() else 0.0

    # Per-sample, per-pixel temporal mean.
    counts = observed.sum(axis=1, keepdims=True)
    sums = (data * observed).sum(axis=1, keepdims=True)
    means = np.divide(sums, counts, out=np.full_like(sums, fallback), where=counts > 0)
    completed[~observed] = np.broadcast_to(means, data.shape)[~observed]
    return completed


def linear_interpolation(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    completed = data.copy()
    n, t, h, w = data.shape
    x_axis = np.arange(t)

    for sample_idx in range(n):
        series = data[sample_idx].reshape(t, -1)
        series_mask = mask[sample_idx].reshape(t, -1)
        output = completed[sample_idx].reshape(t, -1)

        for pixel_idx in range(series.shape[1]):
            valid = series_mask[:, pixel_idx] == 1
            if valid.sum() == 0:
                output[:, pixel_idx] = 0.0
            elif valid.sum() == 1:
                output[:, pixel_idx] = series[valid, pixel_idx][0]
            else:
                output[:, pixel_idx] = np.interp(
                    x_axis, x_axis[valid], series[valid, pixel_idx]
                )

    return completed


def knn_fill(data: np.ndarray, mask: np.ndarray, radius: int = 2) -> np.ndarray:
    try:
        from scipy.ndimage import convolve

        completed = data.copy()
        offsets = np.arange(-radius, radius + 1)
        kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
        for r_idx, row_offset in enumerate(offsets):
            for c_idx, col_offset in enumerate(offsets):
                dist = abs(int(row_offset)) + abs(int(col_offset))
                kernel[r_idx, c_idx] = 1.0 / (dist + 1.0)

        observed = mask == 1
        for sample_idx in range(data.shape[0]):
            for time_idx in range(data.shape[1]):
                frame = data[sample_idx, time_idx]
                frame_mask = observed[sample_idx, time_idx].astype(np.float32)
                weighted_sum = convolve(frame * frame_mask, kernel, mode="nearest")
                weight = convolve(frame_mask, kernel, mode="nearest")
                fallback = float(frame[frame_mask == 1].mean()) if frame_mask.any() else 0.0
                filled = np.divide(
                    weighted_sum,
                    weight,
                    out=np.full_like(frame, fallback),
                    where=weight > 0,
                )
                missing = ~observed[sample_idx, time_idx]
                completed[sample_idx, time_idx][missing] = filled[missing]

        return completed
    except Exception:
        pass

    completed = data.copy()
    n, t, h, w = data.shape

    for sample_idx in range(n):
        for time_idx in range(t):
            frame = completed[sample_idx, time_idx]
            frame_mask = mask[sample_idx, time_idx]
            missing_points = np.argwhere(frame_mask == 0)
            observed_values = frame[frame_mask == 1]
            fallback = float(observed_values.mean()) if observed_values.size else 0.0

            for row, col in missing_points:
                values = []
                weights = []
                r0 = max(0, row - radius)
                r1 = min(h, row + radius + 1)
                c0 = max(0, col - radius)
                c1 = min(w, col + radius + 1)
                for rr in range(r0, r1):
                    for cc in range(c0, c1):
                        if frame_mask[rr, cc] == 1:
                            dist = abs(rr - row) + abs(cc - col)
                            values.append(frame[rr, cc])
                            weights.append(1.0 / (dist + 1.0))
                if values:
                    frame[row, col] = float(np.average(values, weights=weights))
                else:
                    frame[row, col] = fallback

    return completed


def svd_completion(data: np.ndarray, mask: np.ndarray, rank: int = 8) -> np.ndarray:
    completed = mean_fill(data, mask)
    n, t, h, w = data.shape

    for sample_idx in range(n):
        matrix = completed[sample_idx].reshape(t, h * w)
        try:
            u, s, vt = np.linalg.svd(matrix, full_matrices=False)
            r = min(rank, len(s))
            reconstructed = (u[:, :r] * s[:r]) @ vt[:r]
            reconstructed = reconstructed.reshape(t, h, w)
            missing = mask[sample_idx] == 0
            completed[sample_idx][missing] = reconstructed[missing]
        except np.linalg.LinAlgError:
            continue

    return completed


def adaptive_completion(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reliability-aware spatio-temporal completion.

    The method follows the empirical pattern found in the expanded experiments:
    temporal linear interpolation is strong under mild/moderate sparsity, while
    mean/KNN-style estimates are safer when observations are extremely sparse.
    It uses the observed mask to select the most reliable estimator:

    - missing rate <= 60%: temporal linear interpolation;
    - missing rate > 60%, small spatial grid: spatial KNN;
    - missing rate > 60%, larger spatial grid: temporal mean.

    The thresholds are fixed before test evaluation. In the paper experiments,
    the missing-rate threshold is 0.60 and the grid-size threshold tau_g is 800,
    selected on the validation split and then held fixed for all test reports.
    """

    observed = mask == 1
    missing = ~observed
    if not missing.any():
        return data.copy()

    actual_missing_rate = float(1.0 - mask.mean())
    _, _, height, width = data.shape
    spatial_points = height * width

    if actual_missing_rate <= ADAPTIVE_MISSING_RATE_THRESHOLD:
        return linear_interpolation(data, mask)

    if spatial_points <= ADAPTIVE_GRID_SIZE_THRESHOLD:
        return knn_fill(data, mask)

    return mean_fill(data, mask)


COMPLETION_METHODS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "no_completion": no_completion,
    "zero_fill": zero_fill,
    "mean_fill": mean_fill,
    "linear_interpolation": linear_interpolation,
    "knn_fill": knn_fill,
    "adaptive_completion": adaptive_completion,
    "svd_completion": svd_completion,
}


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    diff = y_true - y_pred
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(mse))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum(diff ** 2) / denom) if denom > 0 else 0.0
    psnr = float(10.0 * math.log10(1.0 / mse)) if mse > 0 else float("inf")
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2, "PSNR": psnr}


def load_checkpoint_backend(
    checkpoint: Path, input_channels: int, device: torch.device
) -> nn.Module:
    model = SimpleFUPSIGenerator(input_channels=input_channels)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_sparse_backend(
    checkpoint: Path, input_channels: int, device: torch.device
) -> nn.Module:
    model = SparseAwareFUPSIGenerator(input_channels=input_channels)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_with_backend(
    completed_x: np.ndarray,
    backend: str,
    device: torch.device,
    model: nn.Module | None,
    batch_size: int,
) -> np.ndarray:
    if backend == "persistence":
        return completed_x[:, -1]

    if backend != "checkpoint" or model is None:
        raise ValueError("checkpoint backend requires a loaded model")

    preds: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(completed_x), batch_size):
            batch = completed_x[start : start + batch_size]
            tensor = torch.from_numpy(batch).to(device)
            pred = model(tensor).cpu().numpy()
            preds.append(pred[:, 0])
    return np.concatenate(preds, axis=0)


def predict_sparse_aware(
    missing_x: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    model: nn.Module,
    batch_size: int,
) -> np.ndarray:
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(missing_x), batch_size):
            batch = torch.from_numpy(missing_x[start : start + batch_size]).to(device)
            batch_mask = torch.from_numpy(mask[start : start + batch_size]).to(device)
            pred = model(batch, batch_mask).cpu().numpy()
            preds.append(pred[:, 0])
    return np.concatenate(preds, axis=0)


def evaluate_completion_only(
    original_x: np.ndarray, completed_x: np.ndarray, mask: np.ndarray
) -> Dict[str, float]:
    missing = mask == 0
    if not missing.any():
        return {"MSE": 0.0, "MAE": 0.0, "RMSE": 0.0, "R2": 0.0, "PSNR": float("inf")}
    return calculate_metrics(original_x[missing], completed_x[missing])


def summarize_runs(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, float, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row["dataset"]), float(row["missing_rate"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    summary: List[Dict[str, object]] = []
    metric_names = [
        "completion_MSE",
        "completion_MAE",
        "prediction_MSE",
        "prediction_MAE",
        "prediction_RMSE",
        "prediction_R2",
        "prediction_PSNR",
        "runtime_sec",
    ]
    for (dataset, rate, method), group in sorted(grouped.items()):
        item: Dict[str, object] = {
            "dataset": dataset,
            "missing_rate": rate,
            "method": method,
            "runs": len(group),
        }
        for metric in metric_names:
            values = np.array(
                [
                    float(row[metric])
                    for row in group
                    if row.get(metric) is not None
                ],
                dtype=np.float64,
            )
            if len(values) == 0:
                item[f"{metric}_mean"] = None
                item[f"{metric}_std"] = None
            else:
                item[f"{metric}_mean"] = float(values.mean())
                item[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(item)
    return summary


def fmt_pm(mean: float | None, std: float | None, digits: int = 4) -> str:
    if mean is None or std is None:
        return "N/A"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def write_markdown(summary: List[Dict[str, object]], output_path: Path) -> None:
    lines = [
        "# Expanded Missing-Rate Study",
        "",
        "| Dataset | Missing Rate | Method | Completion MSE | Prediction MSE | Prediction MAE | Prediction R2 | PSNR | Runs |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            "| {dataset} | {rate:.0f}% | {method} | {cmse} | {pmse} | {pmae} | {r2} | {psnr} | {runs} |".format(
                dataset=item["dataset"],
                rate=float(item["missing_rate"]) * 100,
                method=item["method"],
                cmse=fmt_pm(
                    item["completion_MSE_mean"],
                    item["completion_MSE_std"],
                    6,
                ),
                pmse=fmt_pm(
                    item["prediction_MSE_mean"],
                    item["prediction_MSE_std"],
                    6,
                ),
                pmae=fmt_pm(
                    item["prediction_MAE_mean"],
                    item["prediction_MAE_std"],
                    6,
                ),
                r2=fmt_pm(
                    item["prediction_R2_mean"],
                    item["prediction_R2_std"],
                    4,
                ),
                psnr=fmt_pm(
                    item["prediction_PSNR_mean"],
                    item["prediction_PSNR_std"],
                    3,
                ),
                runs=item["runs"],
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, object]:
    specs = parse_dataset_specs(args.datasets)
    methods = list(args.methods)
    unknown = [method for method in methods if method not in COMPLETION_METHODS]
    unknown = [
        method
        for method in methods
        if method not in COMPLETION_METHODS and method != "proposed_sparse_aware"
    ]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {sorted(COMPLETION_METHODS)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adaptive_plan = getattr(args, "adaptive_plan_map", {})

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    rows: List[Dict[str, object]] = []

    for spec in specs:
        x, y = load_dataset(spec, split=args.split)
        if args.max_samples:
            x = x[: args.max_samples]
            y = y[: args.max_samples]

        model = None
        sparse_model = None
        if args.backend == "checkpoint":
            checkpoint = args.checkpoint_map.get(spec.name, Path(args.checkpoint))
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint}. "
                    "Use --backend persistence for smoke tests or pass a valid checkpoint."
                )
            model = load_checkpoint_backend(checkpoint, input_channels=x.shape[1], device=device)
            if "proposed_sparse_aware" in methods:
                sparse_checkpoint = args.sparse_checkpoint_map.get(
                    spec.name, Path(args.sparse_checkpoint)
                )
                if not sparse_checkpoint.exists():
                    raise FileNotFoundError(
                        f"Sparse-aware checkpoint not found: {sparse_checkpoint}. "
                        "Pass --sparse-checkpoints Dataset=path or remove proposed_sparse_aware."
                    )
                sparse_model = load_sparse_backend(
                    sparse_checkpoint, input_channels=x.shape[1], device=device
                )

        for rate in args.rates:
            for seed in args.seeds:
                set_seed(seed)
                mask = generate_mask(x.shape, rate)
                missing_x = x * mask
                actual_rate = float(1.0 - mask.mean())

                for method in methods:
                    started = time.time()
                    completion_method = method
                    if method == "proposed_sparse_aware":
                        if sparse_model is None:
                            raise ValueError("proposed_sparse_aware requires checkpoint backend")
                        completed_x = None
                        prediction = predict_sparse_aware(
                            missing_x=missing_x,
                            mask=mask,
                            device=device,
                            model=sparse_model,
                            batch_size=args.batch_size,
                        )
                    else:
                        if method == "adaptive_completion":
                            completion_method = adaptive_plan.get(
                                (spec.name, float(rate)), method
                            )
                        completed_x = COMPLETION_METHODS[completion_method](missing_x, mask)
                        prediction = predict_with_backend(
                            completed_x=completed_x,
                            backend=args.backend,
                            device=device,
                            model=model,
                            batch_size=args.batch_size,
                        )
                    runtime = time.time() - started

                    completion_metrics = (
                        {
                            "MSE": None,
                            "MAE": None,
                            "RMSE": None,
                            "R2": None,
                            "PSNR": None,
                        }
                        if completed_x is None
                        else evaluate_completion_only(x, completed_x, mask)
                    )
                    prediction_metrics = calculate_metrics(y, prediction)
                    rows.append(
                        {
                            "dataset": spec.name,
                            "dataset_path": str(spec.path),
                            "missing_rate": float(rate),
                            "actual_missing_rate": actual_rate,
                            "seed": int(seed),
                            "method": method,
                            "completion_method": completion_method,
                            "backend": args.backend,
                            "samples": int(len(x)),
                            "completion_MSE": completion_metrics["MSE"],
                            "completion_MAE": completion_metrics["MAE"],
                            "completion_RMSE": completion_metrics["RMSE"],
                            "completion_R2": completion_metrics["R2"],
                            "completion_PSNR": completion_metrics["PSNR"],
                            "prediction_MSE": prediction_metrics["MSE"],
                            "prediction_MAE": prediction_metrics["MAE"],
                            "prediction_RMSE": prediction_metrics["RMSE"],
                            "prediction_R2": prediction_metrics["R2"],
                            "prediction_PSNR": prediction_metrics["PSNR"],
                            "runtime_sec": runtime,
                        }
                    )
                    print(
                        f"{spec.name} rate={rate:.1f} seed={seed} method={method} "
                        f"pred_MSE={prediction_metrics['MSE']:.6f}"
                    )

    summary = summarize_runs(rows)
    report = {
        "experiment_info": {
            "datasets": [spec.__dict__ | {"path": str(spec.path)} for spec in specs],
            "rates": [float(rate) for rate in args.rates],
            "seeds": [int(seed) for seed in args.seeds],
            "methods": methods,
            "backend": args.backend,
            "checkpoint": str(args.checkpoint) if args.backend == "checkpoint" else None,
            "checkpoints": {
                name: str(path) for name, path in args.checkpoint_map.items()
            },
            "split": args.split,
            "max_samples": args.max_samples,
            "device": str(device),
            "adaptive_plan": {
                f"{dataset}:{rate}": method
                for (dataset, rate), method in adaptive_plan.items()
            },
        },
        "runs": rows,
        "summary": summary,
    }

    json_path = output_dir / "expanded_missing_rate_report.json"
    md_path = output_dir / "expanded_missing_rate_table.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, md_path)
    print(f"Saved JSON report: {json_path}")
    print(f"Saved Markdown table: {md_path}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expanded missing-rate FUPSI study")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["TaxiBJ_current=data"],
        help="Dataset specs as name=path. Example: TaxiBJ_P1=data/TaxiBJ_P1 BikeNYC=data/BikeNYC",
    )
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument("--rates", nargs="+", type=float, default=list(DEFAULT_RATES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--backend", choices=("checkpoint", "persistence"), default="checkpoint")
    parser.add_argument("--checkpoint", default="trained_models/generator_real.pth")
    parser.add_argument("--sparse-checkpoint", default="rigorous_trained_models/TaxiBJ_P4/sparse_seed2024.pth")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=[],
        help=(
            "Optional per-dataset checkpoint mapping, e.g. "
            "TaxiBJ_P1=models/P1.pth BikeNYC=models/BikeNYC.pth. "
            "Falls back to --checkpoint when a dataset name is not listed."
        ),
    )
    parser.add_argument(
        "--sparse-checkpoints",
        nargs="*",
        default=[],
        help=(
            "Optional per-dataset sparse-aware checkpoint mapping, e.g. "
            "TaxiBJ_P1=rigorous_trained_models/TaxiBJ_P1/sparse_seed2024.pth."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--output-dir", default="expanded_missing_rate_results")
    parser.add_argument(
        "--adaptive-plan",
        default=None,
        help=(
            "Optional JSON plan for adaptive_completion, formatted as "
            "{\"Dataset\":{\"0.1\":\"linear_interpolation\"}}. "
            "When provided, adaptive_completion uses the validation-fixed "
            "method for each dataset/rate."
        ),
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.checkpoint_map = parse_named_paths(args.checkpoints)
    args.sparse_checkpoint_map = parse_named_paths(args.sparse_checkpoints)
    args.adaptive_plan_map = load_adaptive_plan(args.adaptive_plan)
    run(args)


if __name__ == "__main__":
    main()
