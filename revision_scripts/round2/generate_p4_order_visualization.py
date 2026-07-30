#!/usr/bin/env python3
"""Generate an auditable TaxiBJ P4 order-comparison visualization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_fupsi_generator(
    code_root: Path, namespace: str, seed: int
) -> Path:
    seed_root = (
        code_root
        / "saved_model"
        / "to_stage"
        / "no_ext(r)"
        / f"{namespace}_TaxiBJ_P4"
        / f"seed{seed}"
    )
    matches = sorted(seed_root.rglob("test_metrics.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one TaxiBJ P4 seed {seed} result under {seed_root}, "
            f"found {len(matches)}"
        )
    return matches[0].parent


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    array = np.load(path).astype(np.float32)
    if array.ndim != 4:
        raise ValueError(f"{path}: expected [N,C,H,W], received {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: contains non-finite values")
    return array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument("--namespace", default="ResidualMainE300P5")
    parser.add_argument(
        "--inverse-root",
        type=Path,
        default=Path("revision/round2/inverse_order"),
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision/round2/visualization"),
    )
    args = parser.parse_args()

    code_root = args.code_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fupsi_root = find_fupsi_generator(code_root, args.namespace, args.seed)
    inverse_root = (
        args.inverse_root.resolve() / "TaxiBJ_P4" / f"seed{args.seed}"
    )

    paths = {
        "fupsi_prediction": fupsi_root / "test_fine.npy",
        "fupsi_target": fupsi_root / "true_fine.npy",
        "inverse_prediction": inverse_root / "test_fine_prediction.npy",
        "inverse_target": inverse_root / "test_fine_target.npy",
    }
    arrays = {name: load_array(path) for name, path in paths.items()}
    shape = arrays["fupsi_target"].shape
    if any(array.shape != shape for array in arrays.values()):
        raise ValueError(
            "Order-visualization arrays are not shape aligned: "
            + ", ".join(f"{name}={array.shape}" for name, array in arrays.items())
        )
    target_delta = float(
        np.max(np.abs(arrays["fupsi_target"] - arrays["inverse_target"]))
    )
    if target_delta > 1e-5:
        raise ValueError(
            f"FUPSI and inverse-order targets differ (max abs={target_delta})"
        )

    target = arrays["fupsi_target"]
    fupsi = arrays["fupsi_prediction"]
    inverse = arrays["inverse_prediction"]
    per_sample_rmse = np.sqrt(np.mean((fupsi - target) ** 2, axis=(1, 2, 3)))
    median_error = float(np.median(per_sample_rmse))
    sample_index = int(np.argmin(np.abs(per_sample_rmse - median_error)))

    channel_names = ("Inflow", "Outflow")
    fig, axes = plt.subplots(2, 5, figsize=(12.2, 4.7), constrained_layout=True)
    for channel, channel_name in enumerate(channel_names):
        truth_map = target[sample_index, channel]
        fupsi_map = fupsi[sample_index, channel]
        inverse_map = inverse[sample_index, channel]
        fupsi_error = np.abs(fupsi_map - truth_map)
        inverse_error = np.abs(inverse_map - truth_map)
        flow_min = float(min(truth_map.min(), fupsi_map.min(), inverse_map.min()))
        flow_max = float(max(truth_map.max(), fupsi_map.max(), inverse_map.max()))
        error_max = float(max(fupsi_error.max(), inverse_error.max(), 1e-6))

        panels = (
            (truth_map, "Ground truth", "viridis", flow_min, flow_max),
            (fupsi_map, "FUPSI", "viridis", flow_min, flow_max),
            (fupsi_error, "FUPSI abs. error", "magma", 0.0, error_max),
            (inverse_map, "Inverse order", "viridis", flow_min, flow_max),
            (
                inverse_error,
                "Inverse abs. error",
                "magma",
                0.0,
                error_max,
            ),
        )
        flow_handle = None
        error_handle = None
        for column, (image, title, cmap, vmin, vmax) in enumerate(panels):
            axis = axes[channel, column]
            handle = axis.imshow(
                image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
            )
            if column in (0, 1, 3):
                flow_handle = handle
            else:
                error_handle = handle
            axis.set_title(title, fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(channel_name, fontsize=9)
        fig.colorbar(
            flow_handle,
            ax=[axes[channel, column] for column in (0, 1, 3)],
            fraction=0.018,
            pad=0.012,
            label="Flow count",
        )
        fig.colorbar(
            error_handle,
            ax=[axes[channel, column] for column in (2, 4)],
            fraction=0.026,
            pad=0.012,
            label="Absolute error",
        )
    png_path = output_dir / "TaxiBJ_P4_order_comparison.png"
    pdf_path = output_dir / "TaxiBJ_P4_order_comparison.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    def evidence_path(path: Path) -> str:
        try:
            return path.relative_to(code_root).as_posix()
        except ValueError:
            return path.name

    selected = {
        "protocol": "MainSeed-RawCount-v2",
        "dataset": "TaxiBJ P4",
        "seed": args.seed,
        "selection_rule": (
            "test sample whose corrected FUPSI per-sample RMSE is nearest "
            "the median corrected FUPSI per-sample RMSE"
        ),
        "sample_index": sample_index,
        "median_fupsi_sample_RMSE": median_error,
        "selected_fupsi_sample_RMSE": float(per_sample_rmse[sample_index]),
        "selected_inverse_sample_RMSE": float(
            np.sqrt(np.mean((inverse[sample_index] - target[sample_index]) ** 2))
        ),
        "target_max_abs_difference": target_delta,
        "source_files": {
            name: {"path": evidence_path(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "outputs": {
            "png": png_path.name,
            "png_sha256": sha256(png_path),
            "pdf": pdf_path.name,
            "pdf_sha256": sha256(pdf_path),
        },
    }
    (output_dir / "TaxiBJ_P4_order_comparison.metadata.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote P4 order visualization for sample {sample_index} to "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
