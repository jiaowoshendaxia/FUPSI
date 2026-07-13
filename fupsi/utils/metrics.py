"""Metric helpers used by the original FUPSI training scripts."""

from __future__ import annotations

import numpy as np


def _as_float_array(value):
    return np.asarray(value, dtype=np.float64)


def get_MSE(pred, real):
    pred_arr = _as_float_array(pred)
    real_arr = _as_float_array(real)
    return float(np.mean((pred_arr - real_arr) ** 2))


def get_MAE(pred, real):
    pred_arr = _as_float_array(pred)
    real_arr = _as_float_array(real)
    return float(np.mean(np.abs(pred_arr - real_arr)))


def get_MAPE(pred, real, eps=1e-6):
    pred_arr = _as_float_array(pred)
    real_arr = _as_float_array(real)
    denom = np.maximum(np.abs(real_arr), eps)
    return float(np.mean(np.abs((real_arr - pred_arr) / denom)))
