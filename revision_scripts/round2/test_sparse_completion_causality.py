#!/usr/bin/env python3
"""Verify that completion outputs do not depend on future observations."""

from __future__ import annotations

import numpy as np

from evaluate_sparse_pipeline import COMPLETION_METHODS


def main() -> None:
    rng = np.random.default_rng(2026)
    data = rng.random((16, 2, 8, 8), dtype=np.float32) * 100.0
    mask = (rng.random(data.shape) >= 0.5).astype(np.float32)

    for method_name, method in COMPLETION_METHODS.items():
        reference = method(data, mask)
        if not np.isfinite(reference).all():
            raise AssertionError(f"{method_name}: non-finite output")
        if (reference < 0).any():
            raise AssertionError(f"{method_name}: negative output")
        if not np.allclose(reference[mask > 0], data[mask > 0]):
            raise AssertionError(f"{method_name}: observed entries changed")

        for cutoff in (1, 4, 8, 12):
            perturbed = data.copy()
            perturbed[cutoff:] += rng.random(
                perturbed[cutoff:].shape, dtype=np.float32
            ) * 10000.0
            candidate = method(perturbed, mask)
            if not np.allclose(
                reference[:cutoff], candidate[:cutoff], atol=1e-4
            ):
                raise AssertionError(
                    f"{method_name}: output before t={cutoff} depends on "
                    "future observations"
                )
        print(f"PASS {method_name}")

    print("All completion methods passed the causal no-future-leakage audit.")


if __name__ == "__main__":
    main()
