"""Fixed, parameter-free input preparation for S3T.

Everything here is numpy so the data loader can run it on the CPU workers and
the same function serves pretraining, training and inference.

Order: log radiance -> per-frame robust scaling -> sub-pixel band alignment ->
three features per band (level, shape, local contrast).
"""

from __future__ import annotations

import numpy as np

CELL = 4
N_BANDS = CELL * CELL
# Wavelength order of the 16 mosaic positions, as recovered from adjacent-band
# correlation on the training frames. Mosaic index is not wavelength order:
# bands 4 and 11 sit at the far end of the chain. Used where "contiguous in
# wavelength" matters (spectral masking, spectral-gradient loss).
BAND_CHAIN = (15, 13, 14, 12, 10, 8, 9, 7, 6, 1, 0, 2, 3, 5, 4, 11)
FEATURES = ("level", "shape", "contrast")


def band_offsets(cell: int = CELL):
    """(dy, dx) in cube pixels that moves each band onto the block centre.

    X2Cube puts raw pixel img[cell*i + k//cell, cell*j + k%cell] at cube[i, j, k],
    so band k is physically sampled (k//cell)/cell, (k%cell)/cell of a cube
    pixel below/right of the block origin. The block centre is at
    (cell-1)/(2*cell). Resampling band k at i + offset aligns all 16 bands.
    """
    c = (cell - 1) / (2 * cell)
    return [(c - (k // cell) / cell, c - (k % cell) / cell) for k in range(cell * cell)]


def _shift_axis(x: np.ndarray, d: float, axis: int) -> np.ndarray:
    """Linear resample of x at index + d along axis, edges replicated. |d| < 1."""
    if d == 0:
        return x
    n = x.shape[axis]
    idx = np.arange(n)
    nb = np.clip(idx + (1 if d > 0 else -1), 0, n - 1)
    a = abs(d)
    return (1 - a) * x + a * np.take(x, nb, axis=axis)


def align_bands(cube: np.ndarray) -> np.ndarray:
    """(H, W, 16) -> (H, W, 16), each band resampled onto the common block centre."""
    out = np.empty(cube.shape, dtype=np.float32)
    for k, (dy, dx) in enumerate(band_offsets()):
        b = cube[:, :, k].astype(np.float32)
        out[:, :, k] = _shift_axis(_shift_axis(b, dy, 0), dx, 1)
    return out


def box_sum(x: np.ndarray, k: int) -> np.ndarray:
    """Sum over a k x k window (k odd) per channel, reflect-padded. x: (H, W, C)."""
    r = k // 2
    p = np.pad(x, ((r, r), (r, r), (0, 0)), mode="reflect")
    s = np.cumsum(np.cumsum(p, 0, dtype=np.float64), 1)
    s = np.pad(s, ((1, 0), (1, 0), (0, 0)))
    h, w = x.shape[:2]
    return (s[k:k + h, k:k + w] - s[:h, k:k + w] - s[k:k + h, :w] + s[:h, :w]).astype(np.float32)


def annulus_mean(x: np.ndarray, inner: int = 31, outer: int = 63) -> np.ndarray:
    """Mean over an outer window with the inner window removed (guard region).

    The measured best background estimate for the grey classes: a plain local
    mean includes the object and cancels the contrast it is meant to expose.
    """
    return (box_sum(x, outer) - box_sum(x, inner)) / float(outer * outer - inner * inner)


def normalise_frame(cube: np.ndarray, lo: float = 2, hi: float = 98, sample: int = 65536,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """log1p radiance, scaled so the frame's P2..P98 maps to 0..1 (not clipped).

    Radiance, not reflectance: a per-frame scale removes exposure differences
    while keeping every within-frame brightness ratio, which is the one cue the
    grey classes have. Values outside P2..P98 are kept (clipping them would
    flatten exactly the dark objects we care about).
    """
    L = np.log1p(cube.astype(np.float32))
    flat = L.reshape(-1)
    if flat.size > sample:
        rng = rng or np.random.default_rng(0)
        flat = flat[rng.integers(0, flat.size, sample)]
    p_lo, p_hi = np.percentile(flat, [lo, hi])
    return (L - p_lo) / max(float(p_hi - p_lo), 1e-6)


def features(cube: np.ndarray, align: bool = True) -> np.ndarray:
    """(H, W, 16) raw cube -> (3, 16, H, W) float32: level, shape, contrast.

    level     normalised log radiance (brightness kept -- no per-token norm later)
    shape     level minus its per-pixel band mean: the spectrum with brightness removed
    contrast  level minus its 31/63 annular background mean: local log ratio
    """
    L = normalise_frame(cube)
    if align:
        L = align_bands(L)
    shape = L - L.mean(-1, keepdims=True)
    contrast = L - annulus_mean(L)
    return np.stack([L, shape, contrast]).transpose(0, 3, 1, 2).astype(np.float32)
