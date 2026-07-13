# -- coding:utf-8 --
"""Prepare TaxiBJ P1-P4 and BikeNYC datasets for missing-rate experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np


DATASETS: Dict[str, str] = {
    "TaxiBJ_P1": "TaxiBJ and BikeNYC/TaxiBJ/BJ13_M32x32_T30_InOut.h5",
    "TaxiBJ_P2": "TaxiBJ and BikeNYC/TaxiBJ/BJ14_M32x32_T30_InOut.h5",
    "TaxiBJ_P3": "TaxiBJ and BikeNYC/TaxiBJ/BJ15_M32x32_T30_InOut.h5",
    "TaxiBJ_P4": "TaxiBJ and BikeNYC/TaxiBJ/BJ16_M32x32_T30_InOut.h5",
    "BikeNYC": "TaxiBJ and BikeNYC/BikeNYC/NYC14_M16x8_T60_NewEnd.h5",
}


def create_sequences(data: np.ndarray, window_size: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    x_seq = []
    y_seq = []
    stop = len(data) - window_size - horizon + 1
    for idx in range(stop):
        x_seq.append(data[idx : idx + window_size])
        y_seq.append(data[idx + window_size + horizon - 1])
    return np.asarray(x_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


def prepare_dataset(name: str, source: Path, output_root: Path, window_size: int, horizon: int) -> None:
    with h5py.File(source, "r") as handle:
        raw = handle["data"][:].astype(np.float32)

    # Use inflow channel to match the existing single-channel FUPSI checkpoint.
    flow = raw[:, 0]
    flow_min = float(flow.min())
    flow_max = float(flow.max())
    if flow_max > flow_min:
        flow = (flow - flow_min) / (flow_max - flow_min)
    else:
        flow = np.zeros_like(flow, dtype=np.float32)

    train_size = int(len(flow) * 0.8)
    train_flow = flow[:train_size]
    test_flow = flow[train_size:]

    x_train, y_train = create_sequences(train_flow, window_size, horizon)
    x_test, y_test = create_sequences(test_flow, window_size, horizon)

    dataset_dir = output_root / name
    for split_name, x_data, y_data in (
        ("train", x_train, y_train),
        ("test", x_test, y_test),
    ):
        split_dir = dataset_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / "X.npy", x_data)
        np.save(split_dir / "Y.npy", y_data)

    print(
        f"{name}: raw={raw.shape}, train={x_train.shape}->{y_train.shape}, "
        f"test={x_test.shape}->{y_test.shape}, range=[{flow_min:.3f}, {flow_max:.3f}]"
    )


def main() -> None:
    output_root = Path("data")
    for name, source in DATASETS.items():
        prepare_dataset(
            name=name,
            source=Path(source),
            output_root=output_root,
            window_size=8,
            horizon=1,
        )


if __name__ == "__main__":
    main()
