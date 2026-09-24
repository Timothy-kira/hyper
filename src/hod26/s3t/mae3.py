"""MAE v3: pretraining for S3T-X, on features computed from the visible part only.

Two leaks in v1/v2, both from computing the features *before* masking:

1. shape = level minus the mean of all 16 bands. A visible band's level and
   shape give that mean, so with one band masked its value is exactly
   recoverable from any visible band of the same pixel -- the spectral holes,
   the objective meant to teach band relations, could be solved by
   arithmetic. With k masked bands their sum was given away.
2. contrast = level minus the 31/63 annulus mean, taken over every pixel. The
   ring of a visible pixel next to a hole summed the hidden pixels, and the
   rings of neighbouring visible pixels differ by thin strips of them.

v3 takes the aligned level and masks it first; `observed_features` then takes
every statistic over what is observed (present bands, observed ring pixels).
The detector computes its input with the same function, with nothing masked.

The rest follows v2: 2x2-token spatial units, 1-4 masked bands (contiguous in
wavelength 70% of the time), a ratio curriculum, split spatial / spectral
losses with the same reference ratios and grey rate. The decoder: global
self-attention over all positions with a normalised 2-D sin-cos code (so no
masked unit is out of reach), one windowed cross-covariance block, and a
linear head to 16 bands x 2 x 2 pixels per position.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .front import observed_features
from .mae2 import GlobalBlock, S3TMAE2, band_mask2, sincos_2d, spatial_mask2
from .preprocess import BAND_CHAIN, N_BANDS
from .xca import XCABlock, XCAEncoder, ln_nhwc


class S3TMAE3(nn.Module):
    def __init__(self, encoder: XCAEncoder | None = None, dec_dim: int = 64, glob_depth: int = 2,
                 unit: int = 2, band_lo: int = 1, band_hi: int = 4, band_scatter: float = 0.3,
                 w_band: float = 2.0, w_l1: float = 0.5, w_grad: float = 0.5):
        super().__init__()
        self.enc = encoder or XCAEncoder()
        d, s, c = self.enc.dim, self.enc.stride, self.enc.n_bands
        self.unit = unit
        self.band_lo, self.band_hi, self.band_scatter = band_lo, band_hi, band_scatter
        self.w_band, self.w_l1, self.w_grad = w_band, w_l1, w_grad
        self.dec_dim = dec_dim
        self.proj = nn.Linear(d, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(dec_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.glob = nn.ModuleList(GlobalBlock(dec_dim) for _ in range(glob_depth))
        self.local = XCABlock(dec_dim, heads=4, window=8)
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.head = nn.Linear(dec_dim, c * s * s)
        self.register_buffer("chain", torch.tensor(BAND_CHAIN), persistent=False)

    # v2's target, losses and reference ratios, unchanged (they read level from x[:, 0]).
    target = S3TMAE2.target
    _loss = S3TMAE2._loss
    references = S3TMAE2.references

    def masks(self, b, gh, gw, ratio, device):
        idx = spatial_mask2(b, gh, gw, self.unit, ratio, device)
        bm = band_mask2(b, device, self.enc.n_bands, self.band_lo, self.band_hi, self.band_scatter)
        keep_tok = torch.zeros(b, gh * gw, device=device)
        keep_tok.scatter_(1, idx, 1.0)
        return idx, bm, keep_tok

    def forward(self, level, ratio: float = 0.75):
        """level (B, 16, H, W): the aligned level the detector also reads."""
        b, c, h, w = level.shape
        s, u = self.enc.stride, self.unit
        assert h % (s * u) == 0 and w % (s * u) == 0, "crop must tile into mask units"
        gh, gw = h // s, w // s
        level = level.float()
        idx, bm, keep_tok = self.masks(b, gh, gw, ratio, level.device)
        vis_tok = keep_tok.view(b, 1, gh, gw)
        vis_px = vis_tok.repeat_interleave(s, 2).repeat_interleave(s, 3)
        feats = observed_features(level, vis_px, ~bm)                  # fp32, outside autocast's reach
        pred = self._predict(feats, vis_tok, ~bm, keep_tok, gh, gw)    # (B, S, C, s*s)
        loss, parts = self._loss(pred.float(), level[:, None], keep_tok.to(pred.dtype), bm, gh, gw)
        return loss, parts, pred, idx, bm

    def _predict(self, feats, vis_tok, band_vis, keep_tok, gh, gw):
        b = feats.shape[0]
        s, c = self.enc.stride, self.enc.n_bands
        f = self.enc(feats, vis_tok.to(feats.dtype), band_vis)       # (B, D, gh, gw), 0 where masked
        z = self.proj(f.flatten(2).transpose(1, 2))                    # (B, S, dd)
        z = torch.where(keep_tok.bool()[..., None], z, self.mask_token.to(z.dtype))
        z = z + sincos_2d(gh, gw, self.dec_dim, z.device, z.dtype)
        for blk in self.glob:
            z = blk(z)
        z = z.transpose(1, 2).reshape(b, self.dec_dim, gh, gw).contiguous(memory_format=torch.channels_last)
        z = ln_nhwc(self.local(z), self.dec_norm)
        pred = self.head(z.flatten(2).transpose(1, 2))                  # (B, S, C*s*s)
        return pred.view(b, gh * gw, c, s * s)


def build_mae(version: int, encoder):
    if version == 3:
        return S3TMAE3(encoder)
    if version == 2:
        return S3TMAE2(encoder)
    from .mae import S3TMAE
    return S3TMAE(encoder)


__all__ = ["S3TMAE3", "build_mae", "N_BANDS"]
