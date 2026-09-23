"""The spectral encoder: a per-pixel Transformer over the 16 band tokens.

This is S3M's spectral stream with the Mamba replaced by self-attention. With
16 bands the sequence is short enough that attention costs nothing Mamba would
save, it is bidirectional by construction, and it needs no custom CUDA build.

Fixes over the S3M design it follows:
- no per-token LayerNorm after the stem (it cancels a patch's brightness, the
  only cue the grey classes have) -- BatchNorm there, pre-norm only inside the
  residual branches, so the residual stream keeps absolute level;
- no IWS: a content-independent per-band scale is cancelled by the next
  normalisation, and for a fixed sensor it is 16 constants anyway;
- zero-ish initialised residual branches (LayerScale) where S3M initialised
  sigmoid gates at 0 -- sigmoid(0) is 0.5, not "no effect";
- stride-2 overlapping stem instead of a stride-4 patchify.

Token layout throughout: (B, S, C, D) -- S spatial positions, C bands, D dims.
Every block is independent across S except SpatialMix, which is what lets
masked pretraining compute only the visible positions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import N_BANDS


class SpectralAttention(nn.Module):
    """Multi-head self-attention across the band axis of (N, C, D) tokens.

    Goes through F.scaled_dot_product_attention, so PyTorch picks the fastest
    kernel the GPU supports (flash on sm80+, memory-efficient on a T4).
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.hd = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        n, c, d = t.shape
        q, k, v = self.qkv(t).view(n, c, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(n, c, d))


class SpectralBlock(nn.Module):
    """Pre-norm Transformer block over bands, applied to every position alike."""

    def __init__(self, dim: int, heads: int, mlp: int = 4, ls: float = 0.1):
        super().__init__()
        self.n1, self.attn = nn.LayerNorm(dim), SpectralAttention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp * dim), nn.GELU(), nn.Linear(mlp * dim, dim))
        self.g1 = nn.Parameter(torch.full((dim,), ls))
        self.g2 = nn.Parameter(torch.full((dim,), ls))

    def forward(self, t: torch.Tensor) -> torch.Tensor:          # (B, S, C, D)
        b, s, c, d = t.shape
        x = t.reshape(b * s, c, d)
        x = x + self.g1 * self.attn(self.n1(x))
        x = x + self.g2 * self.mlp(self.n2(x))
        return x.view(b, s, c, d)


def scatter_dense(t: torch.Tensor, idx: torch.Tensor | None, s: int) -> torch.Tensor:
    """(B, Sv, C, D) visible tokens -> (B, S, C, D) with zeros elsewhere."""
    if idx is None:
        return t
    b, _, c, d = t.shape
    dense = t.new_zeros(b, s, c, d)
    return dense.scatter(1, idx[:, :, None, None].expand(-1, -1, c, d), t)


def gather(t: torch.Tensor, idx: torch.Tensor | None) -> torch.Tensor:
    if idx is None:
        return t
    c, d = t.shape[2:]
    return t.gather(1, idx[:, :, None, None].expand(-1, -1, c, d))


class SpatialMix(nn.Module):
    """Local spatial context: a depthwise k x k conv per band, weights shared.

    Masked positions enter as zeros (the sparse-conv trick of FCMAE), so a
    visible token never reads a masked token's content.
    """

    def __init__(self, dim: int, k: int = 7, ls: float = 0.1):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.g = nn.Parameter(torch.full((dim,), ls))

    def forward(self, t, idx, hw):
        h, w = hw
        dense = scatter_dense(t, idx, h * w)
        b, s, c, d = dense.shape
        x = dense.permute(0, 2, 3, 1).reshape(b * c, d, h, w)
        y = self.dw(x).view(b, c, d, s).permute(0, 3, 1, 2)
        y = gather(y, idx)
        return t + self.g * self.pw(self.norm(y))


class SpectralPool(nn.Module):
    """Content-aware weighted sum over bands: (B, S, C, D) -> (B, S, D)."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)
        self.value = nn.Linear(dim, dim)

    def forward(self, t):
        a = self.score(self.norm(t)).softmax(dim=2)             # (B, S, C, 1)
        return (a * self.value(t)).sum(2)


class SpectralEncoder(nn.Module):
    """(B, F, C, H, W) features -> band tokens (B, S, C, D) and a (B, D, H/2, W/2) map."""

    def __init__(self, n_feats: int = 3, n_bands: int = N_BANDS, dim: int = 64,
                 depth: int = 4, heads: int = 4, mix_every: int = 2, stride: int = 2):
        super().__init__()
        self.stride, self.n_bands, self.dim = stride, n_bands, dim
        self.stem = nn.Sequential(
            nn.Conv2d(n_feats, dim, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(dim), nn.GELU(), nn.Conv2d(dim, dim, 1))
        self.band_pe = nn.Parameter(torch.zeros(1, 1, n_bands, dim))
        nn.init.trunc_normal_(self.band_pe, std=0.02)
        self.band_mask_token = nn.Parameter(torch.zeros(dim))
        layers = []
        for i in range(depth):
            layers.append(SpectralBlock(dim, heads))
            if (i + 1) % mix_every == 0:
                layers.append(SpatialMix(dim))
        self.layers = nn.ModuleList(layers)
        self.pool = SpectralPool(dim)
        self.norm = nn.LayerNorm(dim)

    def grid(self, h: int, w: int):
        return math.ceil(h / self.stride), math.ceil(w / self.stride)

    def tokens(self, x, idx=None, band_mask=None):
        """x (B, F, C, H, W); idx (B, Sv) visible positions or None; band_mask (B, C) bool."""
        b, f, c, h, w = x.shape
        y = self.stem(x.transpose(1, 2).reshape(b * c, f, h, w))
        gh, gw = y.shape[-2:]
        t = y.view(b, c, self.dim, gh * gw).permute(0, 3, 1, 2)   # (B, S, C, D)
        t = gather(t, idx)
        if band_mask is not None:
            t = torch.where(band_mask[:, None, :, None], self.band_mask_token.to(t.dtype), t)
        t = t + self.band_pe
        for layer in self.layers:
            t = layer(t, idx, (gh, gw)) if isinstance(layer, SpatialMix) else layer(t)
        return t, (gh, gw)

    def forward(self, x):
        t, (gh, gw) = self.tokens(x)
        p = self.norm(self.pool(t))                               # (B, S, D)
        return p.transpose(1, 2).reshape(x.shape[0], self.dim, gh, gw)
