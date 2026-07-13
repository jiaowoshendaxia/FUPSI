#!/usr/bin/env python3
"""Benchmark FUPSI and reproducible SR baselines under MainSeed shapes.

UrbanFM and FODE are single-channel released models. The adapter evaluates one
two-channel urban-flow sample by flattening its two channels into a batch of two,
so MACs, latency, and memory include both channels.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CODE_ROOT = Path("/root/autodl-tmp/fupsi")
DEFAULT_CUFAR_ROOT = DEFAULT_CODE_ROOT / "revision" / "external_baselines" / "CUFAR"
DEFAULT_OUTPUT = DEFAULT_CODE_ROOT / "revision" / "complexity" / "sr_pipeline_mainseed"


@dataclass(frozen=True)
class Config:
    alias: str
    paper_name: str
    coarse_h: int
    coarse_w: int
    scale: int
    recent: int
    distant: int
    trend: int
    heads: int
    layers: int


CONFIGS = [
    Config("MainSeed_TaxiBJ_P1", "TaxiBJ P1", 8, 8, 4, 3, 5, 0, 4, 4),
    Config("MainSeed_TaxiBJ_P2", "TaxiBJ P2", 8, 8, 4, 3, 1, 0, 2, 1),
    Config("MainSeed_TaxiBJ_P3", "TaxiBJ P3", 8, 8, 4, 3, 2, 0, 4, 1),
    Config("MainSeed_TaxiBJ_P4", "TaxiBJ P4", 8, 8, 4, 3, 3, 0, 2, 1),
    Config("MainSeed_BikeNYC", "BikeNYC", 8, 4, 2, 3, 5, 0, 4, 1),
]


class FlattenedSRWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, coarse: torch.Tensor, ext: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = coarse.shape
        flat = coarse.reshape(batch * channels, 1, height, width)
        flat_ext = ext.repeat_interleave(channels, dim=0)
        output = self.model(flat, flat_ext)
        return output.reshape(batch, channels, output.shape[-2], output.shape[-1])


class Pipeline(nn.Module):
    def __init__(self, predictor: nn.Module, sr_model: nn.Module):
        super().__init__()
        self.predictor = predictor
        self.sr_model = sr_model

    def forward(self, xc: torch.Tensor, xp: torch.Tensor, xt: torch.Tensor, ext: torch.Tensor) -> torch.Tensor:
        return self.sr_model(self.predictor(xc, xp, xt, ext), ext)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def estimate_conv_linear_macs(model: nn.Module, inputs: Sequence[torch.Tensor]) -> int:
    macs = 0
    handles = []

    def conv_hook(module: nn.Conv2d, _inputs, output):
        nonlocal macs
        if not isinstance(output, torch.Tensor):
            return
        batch, out_channels, out_h, out_w = output.shape
        kernel_h, kernel_w = module.kernel_size
        in_per_group = module.in_channels // module.groups
        macs += batch * out_channels * out_h * out_w * in_per_group * kernel_h * kernel_w

    def linear_hook(module: nn.Linear, _inputs, output):
        nonlocal macs
        if isinstance(output, torch.Tensor):
            macs += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    model.eval()
    with torch.no_grad():
        model(*inputs)
    for handle in handles:
        handle.remove()
    return int(macs)


def attention_macs(config: Config, batch_size: int) -> int:
    grid = config.coarse_h * config.coarse_w
    lengths = [config.recent]
    if config.distant:
        lengths.append(config.distant)
    if config.trend:
        lengths.append(config.trend)
    dim_head = 8
    return int(sum(2 * batch_size * grid * config.heads * length * length * dim_head * config.layers for length in lengths))


def make_inputs(config: Config, batch_size: int, device: torch.device):
    channels = 2
    h, w = config.coarse_h, config.coarse_w
    xc = torch.randn(batch_size, config.recent, channels, h, w, device=device)
    xp = torch.randn(batch_size, config.distant, channels, h, w, device=device)
    xt = torch.randn(batch_size, config.trend, channels, h, w, device=device)
    ext = torch.zeros(batch_size, 7, device=device)
    coarse = torch.randn(batch_size, channels, h, w, device=device)
    return (xc, xp, xt, ext), (coarse, ext)


def latency_ms(model: nn.Module, inputs: Sequence[torch.Tensor], warmup: int, repeats: int, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeats


def inference_profile(model: nn.Module, inputs: Sequence[torch.Tensor], warmup: int, repeats: int, device: torch.device) -> tuple[float, float]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    measured_latency = latency_ms(model, inputs, warmup, repeats, device)
    peak = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else math.nan
    return measured_latency, peak


def training_step_profile(
    model: nn.Module,
    inputs: Sequence[torch.Tensor],
    target: torch.Tensor,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        prediction = model(*inputs)
        loss = F.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(repeats):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    measured = (time.perf_counter() - start) * 1000.0 / repeats
    peak = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else math.nan
    return measured, peak


def build_models(config: Config, method: str, code_root: Path, cufar_root: Path, device: torch.device):
    sys.path.insert(0, str(code_root / "revision_scripts"))
    from measure_fupsi_complexity import FUPSIConfig, build_modules
    from run_sr_baseline_adapter import DATASETS, build_model

    fupsi_config = FUPSIConfig(
        config.paper_name,
        config.coarse_h,
        config.coarse_w,
        config.scale,
        config.recent,
        config.distant,
        config.trend,
        config.heads,
        config.layers,
    )
    predictor, fupsi_generator, _, _ = build_modules(fupsi_config, code_root, device)
    if method == "FUPSI":
        sr_model = fupsi_generator
    else:
        released_model = build_model(cufar_root, method, DATASETS[config.alias], 64).to(device)
        sr_model = FlattenedSRWrapper(released_model).to(device)
    return predictor, sr_model, Pipeline(predictor, sr_model).to(device)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    parser.add_argument("--cufar-root", type=Path, default=DEFAULT_CUFAR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--train-warmup", type=int, default=3)
    parser.add_argument("--train-repeats", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--datasets", default=",".join(config.alias for config in CONFIGS))
    parser.add_argument("--methods", default="FUPSI,UrbanFM,FODE")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal same-hardware benchmark")
    device = torch.device("cuda")
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    gpu_name = torch.cuda.get_device_name(0)
    rows: list[dict[str, Any]] = []
    selected_datasets = {item.strip() for item in args.datasets.split(",") if item.strip()}
    selected_methods = {item.strip() for item in args.methods.split(",") if item.strip()}

    for config in CONFIGS:
        if config.alias not in selected_datasets:
            continue
        methods = [method for method in (["FUPSI", "UrbanFM"] + (["FODE"] if "TaxiBJ" in config.alias else [])) if method in selected_methods]
        for method in methods:
            print(f"Profiling {config.paper_name} {method}", flush=True)
            predictor, sr_model, pipeline = build_models(config, method, args.code_root, args.cufar_root, device)
            inference_pipeline_inputs, inference_sr_inputs = make_inputs(config, 1, device)
            sr_macs = estimate_conv_linear_macs(sr_model, inference_sr_inputs)
            pipeline_macs = estimate_conv_linear_macs(pipeline, inference_pipeline_inputs) + attention_macs(config, 1)
            sr_latency, sr_peak = inference_profile(sr_model, inference_sr_inputs, args.warmup, args.repeats, device)
            pipeline_latency, pipeline_peak = inference_profile(pipeline, inference_pipeline_inputs, args.warmup, args.repeats, device)

            train_inputs, _ = make_inputs(config, args.train_batch_size, device)
            target = torch.randn(
                args.train_batch_size,
                2,
                config.coarse_h * config.scale,
                config.coarse_w * config.scale,
                device=device,
            )
            train_ms, train_peak = training_step_profile(
                pipeline, train_inputs, target, args.train_warmup, args.train_repeats, device
            )
            rows.append(
                {
                    "dataset": config.paper_name,
                    "method": method,
                    "coarse_grid": f"{config.coarse_h}x{config.coarse_w}",
                    "fine_grid": f"{config.coarse_h * config.scale}x{config.coarse_w * config.scale}",
                    "scale": config.scale,
                    "sr_params": count_parameters(sr_model),
                    "sr_params_m": f"{count_parameters(sr_model) / 1e6:.6f}",
                    "sr_macs": sr_macs,
                    "sr_macs_g": f"{sr_macs / 1e9:.6f}",
                    "sr_latency_ms_batch1": f"{sr_latency:.6f}",
                    "sr_peak_memory_mb_batch1": f"{sr_peak:.6f}",
                    "pipeline_params": count_parameters(pipeline),
                    "pipeline_params_m": f"{count_parameters(pipeline) / 1e6:.6f}",
                    "pipeline_macs": pipeline_macs,
                    "pipeline_macs_g": f"{pipeline_macs / 1e9:.6f}",
                    "pipeline_latency_ms_batch1": f"{pipeline_latency:.6f}",
                    "pipeline_peak_memory_mb_batch1": f"{pipeline_peak:.6f}",
                    "train_step_ms_batch8_no_discriminator": f"{train_ms:.6f}",
                    "train_peak_memory_mb_batch8_no_discriminator": f"{train_peak:.6f}",
                    "gpu": gpu_name,
                    "inference_warmup": args.warmup,
                    "inference_repeats": args.repeats,
                    "train_warmup": args.train_warmup,
                    "train_repeats": args.train_repeats,
                }
            )
            del pipeline, sr_model, predictor, train_inputs, target
            gc.collect()
            torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "sr_pipeline_complexity_detailed.csv", rows)

    md = [
        "# MainSeed SR pipeline complexity",
        "",
        "| Dataset | Method | Pipeline params (M) | Pipeline MACs (G) | Inference (ms) | Infer peak (MB) | Train step (ms) | Train peak (MB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['dataset']} | {row['method']} | {float(row['pipeline_params_m']):.3f} | "
            f"{float(row['pipeline_macs_g']):.3f} | {float(row['pipeline_latency_ms_batch1']):.3f} | "
            f"{float(row['pipeline_peak_memory_mb_batch1']):.1f} | "
            f"{float(row['train_step_ms_batch8_no_discriminator']):.3f} | "
            f"{float(row['train_peak_memory_mb_batch8_no_discriminator']):.1f} |"
        )
    md.extend(
        [
            "",
            "Notes:",
            "",
            "- All methods use identical MainSeed input shapes and the same RTX 4090.",
            "- Baseline SR MACs/latency include both flow channels through the released single-channel interface.",
            "- Pipeline values include the shared FUPSI coarse predictor followed by the named SR method.",
            "- Training-step measurements cover predictor plus SR forward/backward/Adam update and exclude the FUPSI discriminator.",
        ]
    )
    (args.output_dir / "sr_pipeline_complexity.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir / 'sr_pipeline_complexity_detailed.csv'}", flush=True)


if __name__ == "__main__":
    main()
