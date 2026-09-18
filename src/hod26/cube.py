"""Hyperspectral cube decoding for HOD26.

Competition images are 16-bit grayscale PNGs holding a 4x4 spectral mosaic
(XIMEA snapshot-mosaic layout): every 4x4 pixel block carries 16 bands.
A (4H, 4W) PNG therefore decodes to an (H, W, 16) cube, which is the
resolution the Pascal VOC boxes are expressed in.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

CELL = 4
N_BANDS = CELL * CELL


def read_raw(path) -> np.ndarray:
    """Read the mosaic PNG as a 2-D uint16 array."""
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected 2-D mosaic, got shape {arr.shape}")
    return arr


def x2cube(img: np.ndarray, cell: int = CELL) -> np.ndarray:
    """De-mosaic a (4H, 4W) frame into an (H, W, 16) cube.

    Band k of cube[i, j] is the pixel at img[cell*i + k // cell, cell*j + k % cell],
    which is exactly what the organizers' ``pseudo_rgb_demo.X2Cube`` produces,
    but as a strided view-and-copy instead of a gather over flat indices.
    """
    m, n = img.shape
    if m % cell or n % cell:
        raise ValueError(f"frame {img.shape} not divisible by cell size {cell}")
    # (H, cell, W, cell) -> (H, W, cell, cell) -> (H, W, cell*cell)
    tiled = img.reshape(m // cell, cell, n // cell, cell)
    return np.ascontiguousarray(tiled.transpose(0, 2, 1, 3)).reshape(
        m // cell, n // cell, cell * cell
    )


def load_cube(path) -> np.ndarray:
    """Read a mosaic PNG straight to an (H, W, 16) uint16 cube."""
    return x2cube(read_raw(path))


def stretch(band: np.ndarray, lo_pct: float = 0.0, hi_pct: float = 100.0) -> np.ndarray:
    """Percentile contrast stretch of one band to uint8.

    ``lo_pct=0, hi_pct=100`` reproduces the organizers' min-max demo; tighter
    percentiles are robust to the hot pixels common in 16-bit sensor data.
    """
    b = band.astype(np.float32)
    lo = np.percentile(b, lo_pct) if lo_pct > 0 else float(b.min())
    hi = np.percentile(b, hi_pct) if hi_pct < 100 else float(b.max())
    if hi <= lo:
        return np.zeros(b.shape, np.uint8)
    return (np.clip((b - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def pseudo_rgb(cube: np.ndarray, bands=(0, 1, 2), **kw) -> np.ndarray:
    """Per-band-stretched 3-channel composite — the organizers' demo default."""
    return np.dstack([stretch(cube[:, :, b], **kw) for b in bands])
