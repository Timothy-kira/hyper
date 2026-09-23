"""Masked-autoencoder pretraining for the spectral encoder.

Two orthogonal masks, as in S3M: a tube mask over space (a masked unit is gone
in every band, so it cannot be peeked at through another band) and a
contiguous run of bands masked everywhere (contiguous in *wavelength* order,
which is not mosaic order). The detector is not involved: this pretrains only
the spectral encoder, with a light decoder that is thrown away afterwards.

The encoder computes only the visible positions -- its blocks are per-position
except SpatialMix, which sees masked positions as zeros -- so a 0.75 spatial
mask cuts the encoder's token count to a quarter.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .preprocess import BAND_CHAIN, N_BANDS
from .spectral import SpatialMix, SpectralBlock, SpectralEncoder, scatter_dense


def spatial_mask(b: int, gh: int, gw: int, unit: int, ratio: float, device, generator=None):
    """Visible token indices (B, Sv), sorted; exactly the same count per sample.

    Masking is in units of unit x unit tokens (a random single token would be
    trivially interpolated from its neighbours).
    """
    uh, uw = gh // unit, gw // unit
    n_units = uh * uw
    keep = max(1, round((1 - ratio) * n_units))
    noise = torch.rand(b, n_units, device=device, generator=generator)
    kept = noise.argsort(1)[:, :keep]                               # (B, keep) unit ids
    ui, uj = kept // uw, kept % uw
    di = torch.arange(unit, device=device)
    rows = (ui[:, :, None, None] * unit + di[None, None, :, None])
    cols = (uj[:, :, None, None] * unit + di[None, None, None, :])
    idx = (rows * gw + cols).reshape(b, -1)
    return idx.sort(1).values


def band_mask(b: int, ratio: float, device, n_bands: int = N_BANDS, generator=None):
    """(B, C) bool: a contiguous run (in wavelength order) of masked bands per sample."""
    m = max(1, int(ratio * n_bands))
    chain = torch.tensor(BAND_CHAIN, device=device)
    start = torch.randint(0, n_bands - m + 1, (b,), device=device, generator=generator)
    pos = start[:, None] + torch.arange(m, device=device)[None]
    out = torch.zeros(b, n_bands, dtype=torch.bool, device=device)
    return out.scatter(1, chain[pos], True)


class S3TMAE(nn.Module):
    def __init__(self, encoder: SpectralEncoder | None = None, dec_dim: int = 32,
                 dec_depth: int = 2, unit: int = 4, mask_ratio: float = 0.75,
                 band_ratio: float = 0.15, w_l1: float = 0.5, w_grad: float = 0.5):
        super().__init__()
        self.enc = encoder or SpectralEncoder()
        d, s = self.enc.dim, self.enc.stride
        self.unit, self.mask_ratio, self.band_ratio = unit, mask_ratio, band_ratio
        self.w_l1, self.w_grad = w_l1, w_grad
        self.proj = nn.Linear(d, dec_dim)
        self.proj_pool = nn.Linear(d, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(dec_dim))
        self.dec_pe = nn.Parameter(torch.zeros(1, 1, self.enc.n_bands, dec_dim))
        nn.init.trunc_normal_(self.dec_pe, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        blocks = []
        for _ in range(dec_depth):
            blocks += [SpatialMix(dec_dim), SpectralBlock(dec_dim, heads=2)]
        self.dec = nn.ModuleList(blocks)
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.head = nn.Linear(dec_dim, s * s)          # the level of each pixel in the token
        self.register_buffer("chain", torch.tensor(BAND_CHAIN), persistent=False)

    def target(self, x):
        """Level feature (B, F, C, H, W)[:, 0] -> (B, S, C, s*s) per-token pixel values."""
        s = self.enc.stride
        lv = x[:, 0]
        b, c, h, w = lv.shape
        t = lv.view(b, c, h // s, s, w // s, s).permute(0, 2, 4, 1, 3, 5)
        return t.reshape(b, (h // s) * (w // s), c, s * s)

    def forward(self, x, generator=None):
        b, f, c, h, w = x.shape
        s, u = self.enc.stride, self.unit
        assert h % (s * u) == 0 and w % (s * u) == 0, "crop must tile into mask units"
        gh, gw = h // s, w // s
        idx = spatial_mask(b, gh, gw, u, self.mask_ratio, x.device, generator)
        bm = band_mask(b, self.band_ratio, x.device, c, generator)

        # Zero the masked pixels before the stem, so no conv reads across a unit edge.
        keep_tok = torch.zeros(b, gh * gw, device=x.device, dtype=x.dtype)
        keep_tok.scatter_(1, idx, 1.0)
        keep_px = keep_tok.view(b, 1, 1, gh, 1, gw, 1).expand(b, 1, 1, gh, s, gw, s)
        x_in = x * keep_px.reshape(b, 1, 1, h, w)
        x_in = x_in.masked_fill(bm[:, None, :, None, None], 0.0)

        t, _ = self.enc.tokens(x_in, idx, bm)                    # (B, Sv, C, D)
        pooled = self.enc.norm(self.enc.pool(t))                 # (B, Sv, D)
        z = self.proj(t) + self.proj_pool(pooled)[:, :, None, :]
        dense = scatter_dense(z, idx, gh * gw)
        vis = keep_tok.bool()[:, :, None, None]
        z = torch.where(vis, dense, self.mask_token.to(dense.dtype)) + self.dec_pe
        for blk in self.dec:
            z = blk(z, None, (gh, gw)) if isinstance(blk, SpatialMix) else blk(z)
        pred = self.head(self.dec_norm(z))                       # (B, S, C, s*s)

        tgt = self.target(x)
        # Loss where something was hidden: every band of a masked position, and
        # the masked bands of a visible one.
        m = (~keep_tok.bool())[:, :, None] | bm[:, None, :]      # (B, S, C)
        m = m.to(pred.dtype)[..., None]
        denom = m.sum() * pred.shape[-1] + 1e-6
        # Per-token normalised MSE (MAE's target), scaled by the target token's spread.
        # Floor: in a flat region the spread is ~0 and the ratio would explode.
        sd = tgt.std((2, 3), keepdim=True) + 0.05
        l_norm = (((pred - tgt) / sd) ** 2 * m).sum() / denom
        # Plain L1 on the level itself, so a flat mean-valued spectrum cannot score well.
        l_l1 = ((pred - tgt).abs() * m).sum() / denom
        # Spectral gradient along wavelength order, on the spatially masked tokens.
        pm = (~keep_tok.bool()).to(pred.dtype)[:, :, None, None]
        dp = pred[:, :, self.chain].diff(dim=2)
        dt = tgt[:, :, self.chain].diff(dim=2)
        l_grad = (((dp - dt) ** 2) * pm).sum() / (pm.sum() * dp.shape[2] * dp.shape[3] + 1e-6)
        loss = l_norm + self.w_l1 * l_l1 + self.w_grad * l_grad
        return loss, {"norm": l_norm.detach(), "l1": l_l1.detach(), "grad": l_grad.detach()}, pred, idx, bm
