#!/usr/bin/env python3
"""The deterministic random Fourier projection shared by the NumPy and Dask arms.

Mirrors `uniform_stream`/`standard_normal` in implementation.dml exactly. Every
intermediate of the congruential stream stays below 2^53, so FP64 arithmetic is
exact and all three arms build the same projection bit for bit; only `log` and
`cos` can differ, and only in the last ulp, which is far below the plan's
cross-arm agreement tolerance.
"""

import numpy as np

MODULUS = 2147483647.0


def uniform_stream(key):
    """Three Lehmer rounds mapped onto (0, 1)."""
    state = np.mod(key * 48271.0, MODULUS)
    state = np.mod((state + 12345.0) * 48271.0, MODULUS)
    state = np.mod((state + 6364136.0) * 69621.0, MODULUS)
    return (state + 0.5) / MODULUS


def cell_keys(rows, cols, base):
    """Row-major cell indices of a rows-by-cols matrix, shifted by one scalar."""
    return (np.arange(rows, dtype=np.float64)[:, None] * cols
            + np.arange(cols, dtype=np.float64)[None, :] + base)


def scaled_normal(rows, cols, base, deviation):
    """Box-Muller with the deviation folded under the root, as in the DML arm."""
    first = uniform_stream(cell_keys(rows, cols, base))
    second = uniform_stream(cell_keys(rows, cols, base + rows * cols))
    amplitude = np.sqrt(np.log(first) * (-2.0 * deviation * deviation))
    return amplitude * np.cos(second * (2.0 * np.pi))


def projection(cols, features, gamma, seed):
    """The scaled projection and the feature scale for the paired [cos, sin] map.

    `features` counts output columns, so the projection is half that wide; see the
    note on the pairing in implementation.dml.
    """
    if features % 2 or features < 4:
        raise ValueError("features must be even and at least four")
    half = features // 2
    weights = scaled_normal(cols, half, seed, np.sqrt(2.0 * gamma))
    return weights, half, np.sqrt(1.0 / half)
