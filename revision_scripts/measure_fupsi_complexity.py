#!/usr/bin/env python
"""Measure FUPSI complexity for the major-revision response.

The script instantiates the original FUPSI prediction and super-resolution
modules with the hyperparameters reported in the manuscript. It uses synthetic
inputs with the corresponding coarse/fine grid sizes and reports parameter
counts, estimated MACs, and single-sample latency.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import types
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FUPSIConfig:
    name: str
    coarse_h: int
    coarse_w: int
    scale: int
    len_recent: int
    len_distant: int
    len_trend: int
    heads: int
    transformer_layers: int
    channels: int = 2
    feature_size: int = 64
    hidden_dim: int = 128
    dim_head: int = 8
    skip_dim: int = 128
    residual_blocks: int = 8
    base_channels: int = 64
    external_dim: int = 7


MANUSCRIPT_CONFIGS: List[FUPSIConfig] = [
    FUPSIConfig("TaxiBJ P1", 32, 32, 4, 3, 5, 0, 4, 4),
    FUPSIConfig("TaxiBJ P2", 32, 32, 4, 3, 1, 0, 2, 1),
    FUPSIConfig("TaxiBJ P3", 32, 32, 4, 3, 2, 0, 4, 1),
    FUPSIConfig("TaxiBJ P4", 32, 32, 4, 3, 3, 0, 2, 1),
    FUPSIConfig("BikeNYC", 40, 16, 2, 3, 5, 0, 4, 1),
]


class Pipeline(nn.Module):
    def __init__(self, predictor: nn.Module, generator: nn.Module):
        super().__init__()
        self.predictor = predictor
        self.generator = generator

    def forward(
        self,
        xc: torch.Tensor,
        xp: torch.Tensor,
        xt: torch.Tensor,
        ext: torch.Tensor,
    ) -> torch.Tensor:
        coarse_pred = self.predictor(xc, xp, xt, ext)
        return self.generator(coarse_pred, ext)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def _product(items: Iterable[int]) -> int:
    result = 1
    for item in items:
        result *= int(item)
    return result


def estimate_conv_linear_macs(model: nn.Module, inputs: Sequence[torch.Tensor]) -> int:
    """Estimate MACs for Conv2d and Linear layers using forward hooks."""

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
        if not isinstance(output, torch.Tensor):
            return
        output_elements = output.numel()
        macs += output_elements * module.in_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        _ = model(*inputs)

    for handle in handles:
        handle.remove()
    return int(macs)


def estimate_temporal_attention_macs(config: FUPSIConfig, batch_size: int) -> int:
    """Estimate the two explicit matmul operations inside temporal attention."""

    grid_cells = config.coarse_h * config.coarse_w
    branch_lengths = [config.len_recent]
    if config.len_distant > 0:
        branch_lengths.append(config.len_distant)
    if config.len_trend > 0:
        branch_lengths.append(config.len_trend)

    total = 0
    for length in branch_lengths:
        qk = batch_size * grid_cells * config.heads * length * length * config.dim_head
        av = batch_size * grid_cells * config.heads * length * length * config.dim_head
        total += (qk + av) * config.transformer_layers
    return int(total)


def format_millions(value: int) -> float:
    return value / 1_000_000.0


def format_giga(value: int) -> float:
    return value / 1_000_000_000.0


def build_modules(config: FUPSIConfig, code_root: Path, device: torch.device):
    if "einops" not in sys.modules:
        try:
            import einops  # noqa: F401
        except Exception:
            einops_stub = types.ModuleType("einops")

            def rearrange(x: torch.Tensor, pattern: str, **_kwargs) -> torch.Tensor:
                if pattern.strip() != "b n h t d -> b t n (h d)":
                    raise NotImplementedError(f"Unsupported local einops pattern: {pattern}")
                b, n, h, t, d = x.shape
                return x.permute(0, 3, 1, 2, 4).reshape(b, t, n, h * d)

            einops_stub.rearrange = rearrange
            sys.modules["einops"] = einops_stub

    sys.path.insert(0, str(code_root))
    from prediction import TransAm
    from UrbanSG import Discriminator, Generator

    # PositionalEncoding prints a buffer shape at construction time. Silence it
    # so the generated CSV/Markdown remain clean.
    with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stdout(devnull):
        predictor = TransAm(
            in_channel=config.channels,
            feature_size=config.feature_size,
            hid_dim=config.hidden_dim,
            n_heads=config.heads,
            dim_head=config.dim_head,
            skip_dim=config.skip_dim,
            num_layers=config.transformer_layers,
            len_clossness=config.len_recent,
            len_period=config.len_distant,
            len_trend=config.len_trend,
            external_dim=config.external_dim,
            map_heigh=config.coarse_h,
            map_width=config.coarse_w,
            dropout=0.1,
            ext_flag=False,
        )

    generator = Generator(
        scale_factor=config.scale,
        n_residual_block=config.residual_blocks,
        in_channel=config.channels,
        base_channel=config.base_channels,
        ext_flag=False,
        residual_flag=True,
    )
    discriminator = Discriminator(in_channel=config.channels, ext_flag=False)
    pipeline = Pipeline(predictor, generator)
    return (
        predictor.to(device),
        generator.to(device),
        discriminator.to(device),
        pipeline.to(device),
    )


def make_inputs(config: FUPSIConfig, batch_size: int, device: torch.device):
    c = config.channels
    h = config.coarse_h
    w = config.coarse_w
    xc = torch.randn(batch_size, config.len_recent, c, h, w, device=device)
    xp = torch.randn(batch_size, max(config.len_distant, 0), c, h, w, device=device)
    xt = torch.randn(batch_size, max(config.len_trend, 0), c, h, w, device=device)
    ext = torch.zeros(batch_size, config.external_dim, device=device)
    coarse = torch.randn(batch_size, c, h, w, device=device)
    fine = torch.randn(batch_size, c, h * config.scale, w * config.scale, device=device)
    return {
        "predictor": (xc, xp, xt, ext),
        "generator": (coarse, ext),
        "discriminator": (fine, ext),
        "pipeline": (xc, xp, xt, ext),
    }


def measure_latency_ms(
    model: nn.Module,
    inputs: Sequence[torch.Tensor],
    warmup: int,
    repeats: int,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
    return (end - start) * 1000.0 / repeats


def measure_one_module(
    config: FUPSIConfig,
    module_name: str,
    model: nn.Module,
    inputs: Sequence[torch.Tensor],
    warmup: int,
    repeats: int,
    device: torch.device,
    extra_macs: int = 0,
) -> Dict[str, object]:
    total_params, trainable_params = count_parameters(model)
    conv_linear_macs = estimate_conv_linear_macs(model, inputs)
    total_macs = conv_linear_macs + extra_macs
    peak_memory_mb = ""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    latency_ms = measure_latency_ms(model, inputs, warmup, repeats, device)
    if device.type == "cuda":
        peak_memory_mb = f"{torch.cuda.max_memory_allocated() / (1024 ** 2):.2f}"

    return {
        "dataset": config.name,
        "module": module_name,
        "coarse_grid": f"{config.coarse_h}x{config.coarse_w}",
        "fine_grid": f"{config.coarse_h * config.scale}x{config.coarse_w * config.scale}",
        "scale": config.scale,
        "len_recent": config.len_recent,
        "len_distant": config.len_distant,
        "heads": config.heads,
        "transformer_layers": config.transformer_layers,
        "params": total_params,
        "trainable_params": trainable_params,
        "params_m": f"{format_millions(total_params):.3f}",
        "param_memory_mb_fp32": f"{total_params * 4 / (1024 ** 2):.2f}",
        "conv_linear_macs": conv_linear_macs,
        "attention_macs_est": extra_macs,
        "macs_total": total_macs,
        "macs_g": f"{format_giga(total_macs):.3f}",
        "latency_ms": f"{latency_ms:.3f}",
        "peak_memory_mb": peak_memory_mb,
        "device": str(device),
        "warmup": warmup,
        "repeats": repeats,
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Dataset", "Module", "Params (M)", "MACs (G)", "Latency (ms)", "Param memory (MB)"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {module} | {params_m} | {macs_g} | {latency_ms} | {param_memory_mb_fp32} |".format(
                **row
            )
        )
    lines.append("")
    lines.append(
        "Note: MACs include Conv2d/Linear operations measured by PyTorch hooks and the two temporal-attention matmuls estimated from the manuscript hyperparameters. Latency is measured with batch size 1 on the reported device."
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_compact_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_dataset: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), {})[str(row["module"])] = row

    compact = []
    for dataset, modules in by_dataset.items():
        pred = modules["UFP predictor"]
        gen = modules["SR generator"]
        disc = modules["GAN discriminator"]
        pipe = modules["Inference pipeline"]
        compact.append(
            {
                "Dataset": dataset,
                "UFP Params (M)": pred["params_m"],
                "SR Params (M)": gen["params_m"],
                "Disc Params (M)": disc["params_m"],
                "Inference Params (M)": pipe["params_m"],
                "Inference MACs (G)": pipe["macs_g"],
                "Inference Latency (ms)": pipe["latency_ms"],
                "Device": pipe["device"],
            }
        )
    return compact


def write_compact_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_compact_latex(path: Path, rows: List[Dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\caption{Model complexity of FUPSI under the manuscript hyperparameter settings. MACs include convolution, linear, and temporal-attention matmul operations. Latency is measured for batch size 1.}",
        r"\label{tab:complexity}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Dataset & UFP & SR & Disc. & Infer. params & Infer. MACs & Latency \\",
        r" & (M) & (M) & (M) & (M) & (G) & (ms) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['Dataset']} & {row['UFP Params (M)']} & {row['SR Params (M)']} & "
            f"{row['Disc Params (M)']} & {row['Inference Params (M)']} & "
            f"{row['Inference MACs (G)']} & {row['Inference Latency (ms)']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parents[1] / "fupsi")
    parser.add_argument("--output-dir", type=Path, default=Path("revision/complexity"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    device = torch.device(args.device)
    rows: List[Dict[str, object]] = []

    for config in MANUSCRIPT_CONFIGS:
        predictor, generator, discriminator, pipeline = build_modules(config, args.code_root, device)
        inputs = make_inputs(config, args.batch_size, device)
        attn_macs = estimate_temporal_attention_macs(config, args.batch_size)

        rows.append(
            measure_one_module(
                config,
                "UFP predictor",
                predictor,
                inputs["predictor"],
                args.warmup,
                args.repeats,
                device,
                extra_macs=attn_macs,
            )
        )
        rows.append(
            measure_one_module(
                config,
                "SR generator",
                generator,
                inputs["generator"],
                args.warmup,
                args.repeats,
                device,
            )
        )
        rows.append(
            measure_one_module(
                config,
                "GAN discriminator",
                discriminator,
                inputs["discriminator"],
                args.warmup,
                args.repeats,
                device,
            )
        )
        rows.append(
            measure_one_module(
                config,
                "Inference pipeline",
                pipeline,
                inputs["pipeline"],
                args.warmup,
                args.repeats,
                device,
                extra_macs=attn_macs,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = args.output_dir / "fupsi_complexity_detailed.csv"
    detail_json = args.output_dir / "fupsi_complexity_detailed.json"
    detail_md = args.output_dir / "fupsi_complexity_detailed.md"
    compact_csv = args.output_dir / "fupsi_complexity_compact.csv"
    compact_tex = args.output_dir / "fupsi_complexity_compact_table.tex"

    write_csv(detail_csv, rows)
    detail_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(detail_md, rows)
    compact_rows = build_compact_rows(rows)
    write_compact_csv(compact_csv, compact_rows)
    write_compact_latex(compact_tex, compact_rows)

    print(f"Wrote {detail_csv}")
    print(f"Wrote {detail_md}")
    print(f"Wrote {compact_csv}")
    print(f"Wrote {compact_tex}")


if __name__ == "__main__":
    main()
