#!/usr/bin/env python
"""Prepare Chicago Taxi Trips 2024+ as a FUPSI-style grid dataset.

The script can either read a local CSV exported from the City of Chicago
portal or download rows from the public Socrata CSV endpoint.

Output layout:

    output_dir/
      metadata.json
      raw_fine_series.npy   # [T, H_fine, W_fine], unnormalized pickup counts
      raw_coarse_series.npy # [T, H_coarse, W_coarse], unnormalized pickup counts
      fine_series.npy       # [T, H_fine, W_fine]
      coarse_series.npy     # [T, H_coarse, W_coarse]
      train/X.npy           # [N, history, H_coarse, W_coarse]
      train/Y.npy           # [N, H_coarse, W_coarse] by default
      valid/X.npy
      valid/Y.npy
      test/X.npy
      test/Y.npy

Default task:

    history coarse maps -> next coarse pickup-flow map

This default is compatible with the existing missing-rate and checkpoint
scripts. The script still saves both raw/normalized fine and coarse series so
the future full super-resolution task can use history coarse maps -> next fine
pickup-flow map. The default grid is fine 32 x 32 and coarse 16 x 16, so the
upscaling factor is 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from urllib.parse import urlencode

import numpy as np
import pandas as pd


SOCRATA_CSV_URL = "https://data.cityofchicago.org/resource/ajtu-isnz.csv"
DEFAULT_COLUMNS = [
    "trip_start_timestamp",
    "pickup_centroid_latitude",
    "pickup_centroid_longitude",
    "dropoff_centroid_latitude",
    "dropoff_centroid_longitude",
]


def read_local_csv(path: Path, columns: Iterable[str], max_rows: Optional[int]) -> pd.DataFrame:
    frames = []
    remaining = max_rows
    total = 0
    for chunk in pd.read_csv(path, usecols=lambda col: col in columns, chunksize=200_000):
        frames.append(chunk)
        total += len(chunk)
        print(f"Read {total} rows from local CSV...", flush=True)
        if remaining is not None:
            remaining -= len(chunk)
            if remaining <= 0:
                break
    if not frames:
        raise ValueError(f"No rows read from {path}")
    data = pd.concat(frames, ignore_index=True)
    if max_rows is not None:
        data = data.iloc[:max_rows].copy()
    return data


def download_socrata_rows(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    offset = 0
    remaining = args.max_rows
    selected = ",".join(DEFAULT_COLUMNS)
    while remaining is None or remaining > 0:
        limit = args.page_size if remaining is None else min(args.page_size, remaining)
        params = {
            "$select": selected,
            "$limit": limit,
            "$offset": offset,
            "$order": "trip_start_timestamp ASC",
            "$where": (
                f"trip_start_timestamp between '{args.start_date}T00:00:00' "
                f"and '{args.end_date}T23:59:59'"
            ),
        }
        query = f"{SOCRATA_CSV_URL}?{urlencode(params)}"
        chunk = pd.read_csv(query)
        if chunk.empty:
            break
        frames.append(chunk)
        offset += len(chunk)
        print(f"Downloaded {offset} Chicago taxi rows...", flush=True)
        if remaining is not None:
            remaining -= len(chunk)
        if len(chunk) < limit:
            break
    if not frames:
        raise ValueError("No rows downloaded. Check network access/date range.")
    return pd.concat(frames, ignore_index=True)


def clean_rows(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    data = data.copy()
    data["trip_start_timestamp"] = pd.to_datetime(
        data["trip_start_timestamp"], errors="coerce"
    )
    data = data.dropna(
        subset=[
            "trip_start_timestamp",
            "pickup_centroid_latitude",
            "pickup_centroid_longitude",
        ]
    )
    data = data[
        (data["trip_start_timestamp"] >= pd.Timestamp(args.start_date))
        & (data["trip_start_timestamp"] <= pd.Timestamp(args.end_date) + pd.Timedelta(days=1))
    ]
    lat = data["pickup_centroid_latitude"].astype(float)
    lon = data["pickup_centroid_longitude"].astype(float)
    data = data[(lat.between(-90, 90)) & (lon.between(-180, 180))].copy()
    return data


def infer_bbox(data: pd.DataFrame, q: float) -> Tuple[float, float, float, float]:
    lat = data["pickup_centroid_latitude"].astype(float)
    lon = data["pickup_centroid_longitude"].astype(float)
    return (
        float(lat.quantile(q)),
        float(lat.quantile(1.0 - q)),
        float(lon.quantile(q)),
        float(lon.quantile(1.0 - q)),
    )


def aggregate_pickup_series(
    data: pd.DataFrame,
    bbox: Tuple[float, float, float, float],
    grid: Tuple[int, int],
    freq: str,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    min_lat, max_lat, min_lon, max_lon = bbox
    height, width = grid
    data = data.copy()
    data["slot"] = data["trip_start_timestamp"].dt.floor(freq)
    data = data[
        data["pickup_centroid_latitude"].between(min_lat, max_lat)
        & data["pickup_centroid_longitude"].between(min_lon, max_lon)
    ].copy()
    if data.empty:
        raise ValueError("No rows remain inside the selected bounding box.")

    lat_norm = (data["pickup_centroid_latitude"].astype(float) - min_lat) / (max_lat - min_lat)
    lon_norm = (data["pickup_centroid_longitude"].astype(float) - min_lon) / (max_lon - min_lon)
    data["row"] = np.clip((height - 1 - np.floor(lat_norm * height)).astype(int), 0, height - 1)
    data["col"] = np.clip(np.floor(lon_norm * width).astype(int), 0, width - 1)

    slots = pd.date_range(data["slot"].min(), data["slot"].max(), freq=freq)
    slot_index: Dict[pd.Timestamp, int] = {slot: idx for idx, slot in enumerate(slots)}
    series = np.zeros((len(slots), height, width), dtype=np.float32)

    grouped = data.groupby(["slot", "row", "col"]).size().reset_index(name="count")
    for row in grouped.itertuples(index=False):
        t = slot_index.get(row.slot)
        if t is not None:
            series[t, int(row.row), int(row.col)] = float(row.count)
    return series, slots


def downsample_sum(fine: np.ndarray, factor: int) -> np.ndarray:
    if fine.shape[1] % factor != 0 or fine.shape[2] % factor != 0:
        raise ValueError("Fine grid dimensions must be divisible by factor.")
    t, h, w = fine.shape
    return fine.reshape(t, h // factor, factor, w // factor, factor).sum(axis=(2, 4))


def minmax_normalize(series: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    min_value = float(series.min())
    max_value = float(series.max())
    if max_value > min_value:
        normalized = (series - min_value) / (max_value - min_value)
    else:
        normalized = np.zeros_like(series, dtype=np.float32)
    return normalized.astype(np.float32), {"min": min_value, "max": max_value}


def make_samples(coarse: np.ndarray, fine: np.ndarray, history: int) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for idx in range(history, len(fine)):
        xs.append(coarse[idx - history : idx])
        ys.append(fine[idx])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def split_and_save(x: np.ndarray, y: np.ndarray, output_dir: Path) -> Dict[str, int]:
    n = len(x)
    train_end = int(n * 0.7)
    valid_end = int(n * 0.8)
    splits = {
        "train": (0, train_end),
        "valid": (train_end, valid_end),
        "test": (valid_end, n),
    }
    counts = {}
    for name, (start, end) in splits.items():
        split_dir = output_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / "X.npy", x[start:end])
        np.save(split_dir / "Y.npy", y[start:end])
        counts[name] = int(end - start)
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/ChicagoTaxi2024"))
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2024-03-31")
    parser.add_argument("--max-rows", type=int, default=2_000_000)
    parser.add_argument("--page-size", type=int, default=50_000)
    parser.add_argument("--freq", default="1h")
    parser.add_argument("--fine-height", type=int, default=32)
    parser.add_argument("--fine-width", type=int, default=32)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument(
        "--task-resolution",
        choices=("coarse", "fine"),
        default="coarse",
        help="Resolution used for train/valid/test X.npy and Y.npy. Use coarse for compatibility with existing missing-rate scripts.",
    )
    parser.add_argument("--bbox-quantile", type=float, default=0.01)
    parser.add_argument("--min-lat", type=float, default=None)
    parser.add_argument("--max-lat", type=float, default=None)
    parser.add_argument("--min-lon", type=float, default=None)
    parser.add_argument("--max-lon", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_rows is not None and args.max_rows <= 0:
        args.max_rows = None
    if args.input_csv is not None:
        raw = read_local_csv(args.input_csv, DEFAULT_COLUMNS, args.max_rows)
    else:
        raw = download_socrata_rows(args)
    data = clean_rows(raw, args)
    if args.min_lat is None:
        bbox = infer_bbox(data, args.bbox_quantile)
    else:
        bbox = (args.min_lat, args.max_lat, args.min_lon, args.max_lon)

    raw_fine, slots = aggregate_pickup_series(
        data=data,
        bbox=bbox,
        grid=(args.fine_height, args.fine_width),
        freq=args.freq,
    )
    raw_coarse = downsample_sum(raw_fine, args.factor)
    fine, fine_norm = minmax_normalize(raw_fine)
    coarse, coarse_norm = minmax_normalize(raw_coarse)

    task_series = coarse if args.task_resolution == "coarse" else fine
    x, y = make_samples(task_series, task_series, args.history)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "raw_fine_series.npy", raw_fine)
    np.save(args.output_dir / "raw_coarse_series.npy", raw_coarse)
    np.save(args.output_dir / "fine_series.npy", fine)
    np.save(args.output_dir / "coarse_series.npy", coarse)
    counts = split_and_save(x, y, args.output_dir)

    metadata = {
        "source": "City of Chicago Taxi Trips (2024+), Socrata dataset ajtu-isnz",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "freq": args.freq,
        "bbox": {
            "min_lat": bbox[0],
            "max_lat": bbox[1],
            "min_lon": bbox[2],
            "max_lon": bbox[3],
        },
        "fine_grid": [args.fine_height, args.fine_width],
        "coarse_grid": [int(args.fine_height / args.factor), int(args.fine_width / args.factor)],
        "factor": args.factor,
        "history": args.history,
        "task_resolution": args.task_resolution,
        "slots": int(len(slots)),
        "samples": int(len(x)),
        "split_counts": counts,
        "max_rows": args.max_rows,
        "fine_normalization": fine_norm,
        "coarse_normalization": coarse_norm,
        "nonzero_slots": int((raw_fine.reshape(raw_fine.shape[0], -1).sum(axis=1) > 0).sum()),
        "mean_total_pickups_per_slot": float(raw_fine.reshape(raw_fine.shape[0], -1).sum(axis=1).mean()),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
