"""Spectral and spatial augmentation for hyperspectral detection.

These act on the (H, W, 16) cube, before it is projected to the channels the
detector sees, because the competition's hardest confusions are spectral: the
classes come in material pairs of the same shape and size, and the measured
bottleneck is exactly there. An augmentation applied after projection can only
perturb what the projection already kept.

Three operators, all randomised per call:

* Savitzky-Golay smoothing along the spectral axis, standing in for the
  atmospheric and sensor variation a second capture would show. It replaces each
  band by the value of a low-order polynomial least-squares fitted over a window
  of neighbouring bands, so band shape survives while noise does not.
* Same-class spectral SMOTE, interpolating an object's spectra toward another
  object of its own class. With 16 bands and few hundred instances per class,
  the spectral manifold is sparsely sampled; interpolating between members fills
  it in without inventing a new material.
* Superpixel CutMix, transplanting whole perceptually-coherent regions between
  frames so that context varies while object interiors stay internally
  consistent -- a rectangular cut would slice objects mid-spectrum.
"""

from __future__ import annotations

import numpy as np

from .voc import Box


def savgol_coeffs(window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay smoothing coefficients for the centre of a window.

    Derived rather than imported so the kernel needs no SciPy: the smoothed
    centre value is the first row of the pseudo-inverse of the Vandermonde
    design matrix over the window's offsets.
    """
    if window % 2 == 0 or window < 3:
        raise ValueError(f"window must be odd and >= 3, got {window}")
    if polyorder >= window:
        raise ValueError(f"polyorder {polyorder} must be < window {window}")
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vander(x, polyorder + 1, increasing=True)
    return np.linalg.pinv(A)[0]


def savgol_spectral(cube: np.ndarray, window: int = 5, polyorder: int = 2,
                    order=None) -> np.ndarray:
    """Smooth each pixel's spectrum along the band axis.

    Edges are handled by reflection, which keeps the endpoints from being pulled
    toward zero the way zero-padding would. ``order`` lists the bands in
    wavelength order when that differs from index order; the smoothing then
    runs along it and the result is put back in index order.
    """
    if cube.shape[2] < window:
        return cube
    if order is not None:
        order = np.asarray(order)
        out = np.empty(cube.shape, np.float32)
        out[:, :, order] = savgol_spectral(cube[:, :, order], window, polyorder)
        return out
    coeffs = savgol_coeffs(window, polyorder)
    half = window // 2
    padded = np.pad(cube.astype(np.float32), ((0, 0), (0, 0), (half, half)), mode="reflect")
    out = np.zeros(cube.shape, np.float32)
    for k, c in enumerate(coeffs):
        out += c * padded[:, :, k:k + cube.shape[2]]
    return np.clip(out, 0, None).astype(cube.dtype)


def spectral_smote(cube: np.ndarray, boxes: list[Box], donors: dict[int, np.ndarray],
                   alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Interpolate each object's spectra toward a donor of the same class.

    ``donors`` maps class id to a donor spectrum (16,). Only pixels inside a box
    move, and each object draws its own interpolation weight in [0, alpha], so a
    frame yields a spread of synthetic material states rather than one.
    """
    if alpha <= 0 or not boxes:
        return cube
    out = cube.astype(np.float32).copy()
    for b in boxes:
        donor = donors.get(b.cls_id)
        if donor is None:
            continue
        patch = out[b.y1:b.y2, b.x1:b.x2, :]
        if patch.size == 0:
            continue
        lam = float(rng.uniform(0.0, alpha))
        # Match the donor's overall level first, so the interpolation moves band
        # *shape* -- the discriminative part -- and not brightness.
        scale = patch.mean() / (donor.mean() + 1e-6)
        out[b.y1:b.y2, b.x1:b.x2, :] = patch + lam * (donor * scale - patch)
    return np.clip(out, 0, None).astype(cube.dtype)


def _superpixels(shape: tuple[int, int], n_blocks: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Irregular region labels covering the frame.

    A jittered grid: SLIC would follow image content more faithfully, but this
    needs no scikit-image inside the kernel and still produces regions whose
    boundaries do not line up across frames, which is the property CutMix needs.
    """
    h, w = shape
    rows = max(1, int(np.sqrt(n_blocks * h / max(w, 1))))
    cols = max(1, n_blocks // max(rows, 1))
    ys = np.linspace(0, h, rows + 1).astype(int)
    xs = np.linspace(0, w, cols + 1).astype(int)
    labels = np.zeros((h, w), np.int32)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            y0, y1 = ys[i], ys[i + 1]
            x0, x1 = xs[j], xs[j + 1]
            jy = rng.integers(-(y1 - y0) // 6 or 1, ((y1 - y0) // 6 or 1) + 1)
            jx = rng.integers(-(x1 - x0) // 6 or 1, ((x1 - x0) // 6 or 1) + 1)
            labels[max(0, y0 + jy):min(h, y1 + jy), max(0, x0 + jx):min(w, x1 + jx)] = idx
            idx += 1
    return labels


def superpixel_cutmix(cube: np.ndarray, boxes: list[Box], other: np.ndarray,
                      prob: float, n_blocks: int, rng: np.random.Generator
                      ) -> tuple[np.ndarray, list[Box]]:
    """Paste regions of ``other`` into ``cube``, keeping annotated objects intact.

    Regions overlapping a ground-truth box are skipped, so every surviving label
    still describes what is underneath it. That keeps the operator honest under
    a rule that forbids forged annotations: no box is invented, moved or
    silently emptied.
    """
    if prob <= 0 or other is None:
        return cube, boxes
    h, w, _ = cube.shape
    oh, ow, _ = other.shape
    # Frame sizes vary across this dataset, so a donor is often smaller than its
    # target in one axis. Tiling by reflection covers the shortfall without
    # resampling, which would blur the very spectra the paste is meant to vary.
    if oh < h or ow < w:
        other = np.pad(other,
                       ((0, max(0, h - oh)), (0, max(0, w - ow)), (0, 0)),
                       mode="reflect")
        oh, ow, _ = other.shape

    top = int(rng.integers(0, oh - h + 1))
    left = int(rng.integers(0, ow - w + 1))
    donor = other[top:top + h, left:left + w, :]

    labels = _superpixels((h, w), n_blocks, rng)
    protected = np.zeros((h, w), bool)
    for b in boxes:
        protected[b.y1:b.y2, b.x1:b.x2] = True

    out = cube.copy()
    for lab in np.unique(labels):
        if rng.random() >= prob:
            continue
        region = labels == lab
        if (region & protected).any():
            continue
        out[region] = donor[region]
    return out, boxes


def _feather(h: int, w: int, margin: int) -> np.ndarray:
    """1 over the inner (h - 2m) x (w - 2m) box, falling linearly to 0 across the margin."""
    if margin <= 0:
        return np.ones((h, w), np.float32)
    ys = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)
    xs = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32)
    ry = np.clip((ys + 0.5) / margin, 0.0, 1.0)
    rx = np.clip((xs + 0.5) / margin, 0.0, 1.0)
    return np.minimum(ry[:, None], rx[None, :])


def _covered(b: Box, by: Box) -> float:
    ix = max(0, min(b.x2, by.x2) - max(b.x1, by.x1))
    iy = max(0, min(b.y2, by.y2) - max(b.y1, by.y1))
    return ix * iy / max(1, (b.x2 - b.x1) * (b.y2 - b.y1))


def crowd_paste(cube: np.ndarray, boxes: list[Box], pool: dict, rng: np.random.Generator,
                anchor_classes, load, n_max: int = 3, margin: int = 4,
                max_cover: float = 0.7) -> tuple[np.ndarray, list[Box]]:
    """Paste instances of the crowded classes next to, and partly over, their own kind.

    stone_block / people / e-bike / car are the only classes whose boxes overlap
    other boxes (10-32% of instances against 0-3%) -- people x people, e-bike x
    e-bike, car x car -- and stone_block has only 42 scenes. Each paste takes a
    real instance from another street frame, with a ``margin`` of its own
    background feathered into the destination, sets it beside an anchor *of the
    same class* (a person by a person, an e-bike in a row of e-bikes; never a
    car over a pedestrian) with bottoms roughly aligned and 15-45% of its width
    overlapping the anchor and scales it per band so its surrounding
    background matches the destination's -- which keeps the object-to-background
    contrast, the only cue these grey classes carry, what it was. Existing boxes
    the paste covers by more than ``max_cover`` are dropped (they are no longer
    visible); partly covered ones stay, as in the real crowds.

    pool: {cls_id: [(source, (x1, y1, x2, y2)), ...]}; load(source) -> cube.
    Frames without an anchor are returned unchanged (tabletop scenes).
    """
    anchors = [b for b in boxes if b.cls_id in anchor_classes]
    classes = sorted({b.cls_id for b in anchors} & {c for c in pool if pool[c]})
    if not classes:
        return cube, boxes
    out = cube.astype(np.float32, copy=True)
    boxes = list(boxes)
    H, W = cube.shape[:2]
    m = int(margin)
    for _ in range(int(rng.integers(1, n_max + 1))):
        c = classes[int(rng.integers(len(classes)))]
        src_key, (x1, y1, x2, y2) = pool[c][int(rng.integers(len(pool[c])))]
        w, h = x2 - x1, y2 - y1
        if w < 4 or h < 4:
            continue
        src = load(src_key)
        if y1 - m < 0 or x1 - m < 0 or y2 + m > src.shape[0] or x2 + m > src.shape[1]:
            continue
        same = [b for b in anchors if b.cls_id == c]
        a = same[int(rng.integers(len(same)))]
        ov = int(round(float(rng.uniform(0.15, 0.45)) * w))
        nx1 = a.x2 - ov if rng.random() < 0.5 else a.x1 - w + ov
        jitter = max(1, h // 6)
        ny2 = a.y2 + int(rng.integers(-jitter, jitter + 1))
        ny1 = ny2 - h
        if nx1 - m < 0 or ny1 - m < 0 or nx1 + w + m > W or ny2 + m > H:
            continue
        patch = src[y1 - m:y2 + m, x1 - m:x2 + m].astype(np.float32)
        dst = out[ny1 - m:ny2 + m, nx1 - m:nx1 + w + m]
        ring = np.ones(patch.shape[:2], bool)
        ring[m:m + h, m:m + w] = False
        if m > 0 and ring.any():
            ratio = (dst[ring].mean(0) + 1e-3) / (patch[ring].mean(0) + 1e-3)
            patch = patch * np.clip(ratio, 0.5, 2.0)[None, None, :]
        alpha = _feather(h + 2 * m, w + 2 * m, m)[:, :, None]
        out[ny1 - m:ny2 + m, nx1 - m:nx1 + w + m] = alpha * patch + (1.0 - alpha) * dst
        new = Box(c, nx1, ny1, nx1 + w, ny2)
        boxes = [b for b in boxes if _covered(b, new) <= max_cover] + [new]
    if np.issubdtype(cube.dtype, np.integer):
        info = np.iinfo(cube.dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(cube.dtype), boxes
