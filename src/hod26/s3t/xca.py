"""S3T-X: the spectral encoder with cross-covariance (channel) attention.

Why this replaces the band-token encoder
----------------------------------------
The band-token encoder kept 16 tokens x 64 dims = 1024 numbers per position at
stride 2 -- more than twenty times the 48 numbers a pixel carries (16 bands x
level/shape/contrast). Every LayerNorm, linear layer and copy scaled with that
state, and a 1024^2 detector input carried ~1.05M tokens: 3.3 s of a 3.65 s
training step on a T4. Attention itself was ~2% of the arithmetic, so a
cheaper attention (linear or otherwise) could not fix it; the state could.

What the band tokens bought was *content-dependent mixing between bands*.
Cross-covariance attention buys the same thing on a per-pixel vector:

    XCA(Q, K, V) = V . softmax(K^T Q / tau),   Q, K l2-normalised over pixels

(XCiT, NeurIPS 2021; Restormer's MDTA, CVPR 2022; MST++'s spectral-wise
attention, NTIRE 2022). The d x d matrix is a normalised cross-covariance of
the features over the pixels it is computed on, the cost is linear in pixels,
and each pixel keeps one d-dim vector.

Why it fits this data
---------------------
Computed over a *local window*, K^T Q is the local background's spectral
covariance, and applying a softmax of it to V is a learned, soft analogue of
local whitening -- the operation behind the local RX detector, which scores a
pixel by its Mahalanobis distance to the local background because small
targets differ only from their surroundings. The grey detection classes are
exactly that case: the CPU scans found them separable only against a local
background (annular contrast; local RX was the best transform for e-bike and
car). So the first blocks use 16 x 16 windows (~32 native px, the scale of the
31 px annulus that worked) and the last blocks the whole image, for
scene-level normalisation such as illumination.

Layout: channels_last (NHWC) throughout, so the per-pixel LayerNorm acts on
the contiguous last dimension and the 1x1 / depthwise convs use cuDNN's NHWC
kernels without transposes.

Masked pretraining: pass `vis` (B, 1, h, w), 1 for visible positions. Keys and
queries are zeroed at masked positions before the covariance, so the
statistics come from visible pixels only; the state is re-zeroed at masked
positions after every block and the normalised input of every branch before
its convolutions (the sparse-conv trick of FCMAE), so no visible output ever
reads a masked position -- not even the constant a LayerNorm makes of a zero.
`band_vis` (B, C) marks the bands present; a missing band adds a learned
vector to the stem output, so a masked band is not mistaken for a zero one.

No BatchNorm anywhere: under a 75% mask its batch statistics would be those
of mostly-zero maps, and a detector's 2-image batches are no better.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import N_BANDS


def ln_nhwc(x: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
    """Per-pixel LayerNorm of an NCHW-shaped tensor stored channels_last."""
    return F.layer_norm(x.permute(0, 2, 3, 1), ln.normalized_shape, ln.weight, ln.bias,
                        ln.eps).permute(0, 3, 1, 2)


def _windows(t: torch.Tensor, win: int | None, heads: int):
    """(B, C, H, W) -> (B*nW, heads, C/heads, N) over windows (or the whole map)."""
    b, c, h, w = t.shape
    if win is None:
        return t.reshape(b, heads, c // heads, h * w)
    t = t.reshape(b, heads, c // heads, h // win, win, w // win, win)
    t = t.permute(0, 3, 5, 1, 2, 4, 6)                      # B, nH, nW, heads, dh, win, win
    return t.reshape(-1, heads, c // heads, win * win)


def _unwindows(t: torch.Tensor, win: int | None, b: int, c: int, h: int, w: int):
    if win is None:
        return t.reshape(b, c, h, w)
    heads, dh = t.shape[1], t.shape[2]
    t = t.reshape(b, h // win, w // win, heads, dh, win, win).permute(0, 3, 4, 1, 5, 2, 6)
    return t.reshape(b, c, h, w)


class XCAttention(nn.Module):
    """Multi-Dconv cross-covariance attention, over windows or the whole map."""

    def __init__(self, dim: int, heads: int = 4, window: int | None = 16):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.window = heads, window
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=False)
        self.qkv_dw = nn.Conv2d(3 * dim, 3 * dim, 3, padding=1, groups=3 * dim, bias=False)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x, vis=None):
        b, c, h, w = x.shape
        win = self.window
        pad_h = pad_w = 0
        if win is not None and (h % win or w % win):
            pad_h, pad_w = (-h) % win, (-w) % win
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        if vis is not None:
            # statistics from visible pixels only
            q, k = q * vis, k * vis
        if pad_h or pad_w:
            q, k, v = (F.pad(t, (0, pad_w, 0, pad_h)) for t in (q, k, v))
        hp, wp = h + pad_h, w + pad_w
        q = F.normalize(_windows(q, win, self.heads), dim=-1)
        k = F.normalize(_windows(k, win, self.heads), dim=-1)
        v = _windows(v, win, self.heads)
        attn = (q @ k.transpose(-2, -1)) * self.temperature        # (B*nW, heads, dh, dh)
        out = attn.softmax(-1).to(v.dtype) @ v                       # (B*nW, heads, dh, N)
        out = _unwindows(out, win, b, c, hp, wp)[:, :, :h, :w]
        return self.proj(out.contiguous(memory_format=torch.channels_last))


class GDFN(nn.Module):
    """Gated-Dconv feed-forward (Restormer): GELU(a) * b after a depthwise 3x3."""

    def __init__(self, dim: int, expand: float = 2.0):
        super().__init__()
        hid = int(dim * expand)
        self.pw_in = nn.Conv2d(dim, 2 * hid, 1, bias=False)
        self.dw = nn.Conv2d(2 * hid, 2 * hid, 3, padding=1, groups=2 * hid, bias=False)
        self.pw_out = nn.Conv2d(hid, dim, 1, bias=False)

    def forward(self, x):
        a, g = self.dw(self.pw_in(x)).chunk(2, dim=1)
        return self.pw_out(F.gelu(a) * g)


class XCABlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, window: int | None = 16, ls: float = 0.1):
        super().__init__()
        self.n1, self.attn = nn.LayerNorm(dim), XCAttention(dim, heads, window)
        self.n2, self.ffn = nn.LayerNorm(dim), GDFN(dim)
        self.g1 = nn.Parameter(torch.full((1, dim, 1, 1), ls))
        self.g2 = nn.Parameter(torch.full((1, dim, 1, 1), ls))

    def forward(self, x, vis=None):
        h = ln_nhwc(x, self.n1)
        x = x + self.g1 * self.attn(h if vis is None else h * vis, vis)
        h = ln_nhwc(x, self.n2)
        x = x + self.g2 * self.ffn(h if vis is None else h * vis)
        return x if vis is None else x * vis


class XCAEncoder(nn.Module):
    """(B, F, C, H, W) features -> (B, D, H/2, W/2), one D-dim vector per position."""

    arch = "xca"

    def __init__(self, n_feats: int = 3, n_bands: int = N_BANDS, dim: int = 64, depth: int = 4,
                 heads: int = 4, windows=(16, 16, None, None), stride: int = 2):
        super().__init__()
        self.n_feats, self.n_bands, self.dim, self.stride = n_feats, n_bands, dim, stride
        self.depth, self.heads = depth, heads
        windows = list(windows) + [None] * max(0, depth - len(windows))
        self.windows = [None if w in (None, 0) else int(w) for w in windows[:depth]]
        self.stem = nn.Sequential(
            nn.Conv2d(n_feats * n_bands, dim, 3, stride=stride, padding=1),
            nn.GELU(), nn.Conv2d(dim, dim, 1))
        self.band_missing = nn.Linear(n_bands, dim, bias=False)
        nn.init.trunc_normal_(self.band_missing.weight, std=0.02)
        self.blocks = nn.ModuleList(XCABlock(dim, heads, w) for w in self.windows)
        self.norm = nn.LayerNorm(dim)
        self.grad_ckpt = False

    def config(self):
        return {"arch": "xca", "dim": self.dim, "depth": self.depth, "heads": self.heads,
                "windows": self.windows, "stride": self.stride}

    def grid(self, h, w):
        return -(-h // self.stride), -(-w // self.stride)

    def stem_map(self, x, vis=None, band_vis=None):
        b, f, c, h, w = x.shape
        y = self.stem(x.reshape(b, f * c, h, w).contiguous(memory_format=torch.channels_last))
        if band_vis is not None:
            y = y + self.band_missing((~band_vis.bool()).to(y.dtype))[:, :, None, None]
        return y if vis is None else y * vis

    def forward(self, x, vis=None, band_vis=None):
        y = self.stem_map(x, vis, band_vis)
        ckpt = self.grad_ckpt and self.training and torch.is_grad_enabled()
        for blk in self.blocks:
            if ckpt:
                y = torch.utils.checkpoint.checkpoint(blk, y, vis, use_reentrant=False)
            else:
                y = blk(y, vis)
        return ln_nhwc(y, self.norm)


def build_encoder(cfg: dict | None = None):
    """Encoder from a checkpoint's config: 'xca' (S3T-X) or the band-token one."""
    cfg = dict(cfg or {})
    if cfg.get("arch", "tokens") == "xca":
        return XCAEncoder(dim=cfg.get("dim", 64), depth=cfg.get("depth", 4), heads=cfg.get("heads", 4),
                          windows=cfg.get("windows", (16, 16, None, None)), stride=cfg.get("stride", 2))
    from .spectral import SpectralEncoder
    return SpectralEncoder(dim=cfg.get("dim", 64), depth=cfg.get("depth", 4), heads=cfg.get("heads", 4))
