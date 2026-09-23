"""MAE v2 for the S3T spectral encoder.

What v1 got wrong, and what changes here. The encoder is untouched -- same
class, same weights layout -- so v2 continues from v1's encoder and the
detector loads either.

1. Grey blocks. v1's decoder saw 3 tokens around each position (one 7x7
   SpatialMix) and had no spatial position code. With 4x4-token mask units at
   75%, 28.9% of masked tokens were further than that from any visible token:
   they decoded to one flat value, and sent the encoder no gradient at all.
   v2's decoder pools the bands at each position, runs 2 layers of *global*
   self-attention over all positions with a normalised 2-D sin-cos code, and
   adds the result back to every band token before the per-band head.
2. The objective is split. Spatial holes (a masked position, every band) and
   spectral holes (masked bands at a visible position) are scored separately.
   The spectral holes are what the grey detection classes need, and they are
   filled by the encoder's per-pixel band attention -- so they train the part
   of the model the detector keeps. They get weight 2.
3. Stronger band masking: 1-4 bands per sample; contiguous in wavelength order
   70% of the time, scattered 30%.
4. Smaller spatial units (2x2 tokens = 4x4 native px) and a mask-ratio
   curriculum the trainer steps from 0.5 to 0.75.
5. Two reference points are logged with the losses, so progress can be told
   from noise: spectral holes vs linear interpolation between the nearest
   unmasked bands in wavelength order; spatial holes vs the crop's visible
   mean per band. Below 1.0 means better than the trivial fill. And a grey
   rate: the share of masked units whose prediction varies less than a tenth
   as much as the truth does.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import BAND_CHAIN, N_BANDS
from .spectral import SpatialMix, SpectralBlock, SpectralEncoder, scatter_dense


def spatial_mask2(b: int, gh: int, gw: int, unit: int, ratio: float, device):
    """Visible token indices (B, Sv), same count per sample, units of unit x unit tokens."""
    uh, uw = gh // unit, gw // unit
    n_units = uh * uw
    keep = max(1, round((1 - ratio) * n_units))
    kept = torch.rand(b, n_units, device=device).argsort(1)[:, :keep]
    ui, uj = kept // uw, kept % uw
    di = torch.arange(unit, device=device)
    rows = ui[:, :, None, None] * unit + di[None, None, :, None]
    cols = uj[:, :, None, None] * unit + di[None, None, None, :]
    return (rows * gw + cols).reshape(b, -1).sort(1).values


def band_mask2(b: int, device, n_bands: int = N_BANDS, lo: int = 1, hi: int = 4,
               p_scatter: float = 0.3):
    """(B, C) bool. 1..4 bands per sample; a contiguous run in wavelength order,
    or (with p_scatter) that many bands anywhere."""
    chain = torch.tensor(BAND_CHAIN, device=device)
    n = torch.randint(lo, hi + 1, (b,), device=device)
    scatter = torch.rand(b, device=device) < p_scatter
    out = torch.zeros(b, n_bands, dtype=torch.bool, device=device)
    for i in range(b):                     # b is a batch of crops; this is cheap
        k = int(n[i])
        if scatter[i]:
            pick = torch.randperm(n_bands, device=device)[:k]
        else:
            s = int(torch.randint(0, n_bands - k + 1, (1,)))
            pick = chain[s:s + k]
        out[i, pick] = True
    return out


def sincos_2d(h: int, w: int, dim: int, device, dtype=torch.float32) -> torch.Tensor:
    """(h*w, dim) code of normalised (y, x): the same for any grid size."""
    nf = dim // 4
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    freq = math.pi * 2.0 ** torch.arange(nf, device=device, dtype=torch.float32)
    ay, ax = gy.reshape(-1, 1) * freq, gx.reshape(-1, 1) * freq
    return torch.cat([ay.sin(), ay.cos(), ax.sin(), ax.cos()], 1).to(dtype)


class GlobalBlock(nn.Module):
    """Pre-norm self-attention over all spatial positions (B, S, D), SDPA."""

    def __init__(self, dim: int, heads: int = 4, mlp: int = 2):
        super().__init__()
        self.heads, self.hd = heads, dim // heads
        self.n1, self.qkv, self.proj = nn.LayerNorm(dim), nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp * dim), nn.GELU(), nn.Linear(mlp * dim, dim))

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, s, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
        x = x + self.proj(o.transpose(1, 2).reshape(b, s, d))
        return x + self.mlp(self.n2(x))


def interp_bands(level: torch.Tensor, bm: torch.Tensor) -> torch.Tensor:
    """Fill masked bands by linear interpolation along wavelength order.

    level (B, C, ...) and bm (B, C) bool. Ends take the nearest unmasked band.
    The trivial baseline the spectral-hole loss must beat.
    """
    chain = list(BAND_CHAIN)
    out = level.clone()
    for i in range(level.shape[0]):
        m = [bool(bm[i, c]) for c in chain]
        known = [j for j, mm in enumerate(m) if not mm]
        for j, mm in enumerate(m):
            if not mm:
                continue
            lo = max((k for k in known if k < j), default=None)
            hi = min((k for k in known if k > j), default=None)
            if lo is None:
                out[i, chain[j]] = level[i, chain[hi]]
            elif hi is None:
                out[i, chain[j]] = level[i, chain[lo]]
            else:
                t = (j - lo) / (hi - lo)
                out[i, chain[j]] = (1 - t) * level[i, chain[lo]] + t * level[i, chain[hi]]
    return out


class S3TMAE2(nn.Module):
    def __init__(self, encoder: SpectralEncoder | None = None, dec_dim: int = 32,
                 glob_dim: int = 64, glob_depth: int = 2, unit: int = 2,
                 band_lo: int = 1, band_hi: int = 4, band_scatter: float = 0.3,
                 w_band: float = 2.0, w_l1: float = 0.5, w_grad: float = 0.5):
        super().__init__()
        self.enc = encoder or SpectralEncoder()
        d, s = self.enc.dim, self.enc.stride
        self.unit = unit
        self.band_lo, self.band_hi, self.band_scatter = band_lo, band_hi, band_scatter
        self.w_band, self.w_l1, self.w_grad = w_band, w_l1, w_grad
        self.glob_dim = glob_dim
        self.proj = nn.Linear(d, dec_dim)
        self.proj_pool = nn.Linear(d, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(dec_dim))
        self.dec_pe = nn.Parameter(torch.zeros(1, 1, self.enc.n_bands, dec_dim))
        nn.init.trunc_normal_(self.dec_pe, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.to_glob = nn.Linear(dec_dim, glob_dim)
        self.glob = nn.ModuleList(GlobalBlock(glob_dim) for _ in range(glob_depth))
        self.from_glob = nn.Linear(glob_dim, dec_dim)
        self.spec = SpectralBlock(dec_dim, heads=2, mlp=2)
        self.mix = SpatialMix(dec_dim)
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.head = nn.Linear(dec_dim, s * s)
        self.register_buffer("chain", torch.tensor(BAND_CHAIN), persistent=False)

    def target(self, x):
        s = self.enc.stride
        lv = x[:, 0]
        b, c, h, w = lv.shape
        t = lv.view(b, c, h // s, s, w // s, s).permute(0, 2, 4, 1, 3, 5)
        return t.reshape(b, (h // s) * (w // s), c, s * s)

    def forward(self, x, ratio: float = 0.75):
        b, f, c, h, w = x.shape
        s, u = self.enc.stride, self.unit
        assert h % (s * u) == 0 and w % (s * u) == 0, "crop must tile into mask units"
        gh, gw = h // s, w // s
        S = gh * gw
        idx = spatial_mask2(b, gh, gw, u, ratio, x.device)
        bm = band_mask2(b, x.device, c, self.band_lo, self.band_hi, self.band_scatter)

        keep_tok = torch.zeros(b, S, device=x.device, dtype=x.dtype)
        keep_tok.scatter_(1, idx, 1.0)
        keep_px = keep_tok.view(b, 1, 1, gh, 1, gw, 1).expand(b, 1, 1, gh, s, gw, s)
        x_in = (x * keep_px.reshape(b, 1, 1, h, w)).masked_fill(bm[:, None, :, None, None], 0.0)

        t, _ = self.enc.tokens(x_in, idx, bm)                     # (B, Sv, C, D)
        pooled = self.enc.norm(self.enc.pool(t))
        z = self.proj(t) + self.proj_pool(pooled)[:, :, None, :]
        dense = scatter_dense(z, idx, S)
        vis = keep_tok.bool()[:, :, None, None]
        z = torch.where(vis, dense, self.mask_token.to(dense.dtype)) + self.dec_pe
        # Global context: every position, visible or not, sees every other.
        g = self.to_glob(z.mean(2)) + sincos_2d(gh, gw, self.glob_dim, x.device, z.dtype)
        for blk in self.glob:
            g = blk(g)
        z = z + self.from_glob(g)[:, :, None, :]
        z = self.mix(z, None, (gh, gw))
        z = self.spec(z)
        pred = self.head(self.dec_norm(z))                          # (B, S, C, s*s)

        tgt = self.target(x)
        sd = tgt.std((2, 3), keepdim=True) + 0.05
        hole_s = (~keep_tok.bool())[:, :, None].expand(b, S, c)       # masked positions, all bands
        hole_b = keep_tok.bool()[:, :, None] & bm[:, None, :]          # visible positions, masked bands

        def region(m):
            mm = m.to(pred.dtype)[..., None]
            n = mm.sum() * pred.shape[-1] + 1e-6
            return (((pred - tgt) / sd) ** 2 * mm).sum() / n, ((pred - tgt).abs() * mm).sum() / n

        norm_s, l1_s = region(hole_s)
        norm_b, l1_b = region(hole_b)
        pm = (~keep_tok.bool()).to(pred.dtype)[:, :, None, None]
        dp, dt = pred[:, :, self.chain].diff(dim=2), tgt[:, :, self.chain].diff(dim=2)
        grad = (((dp - dt) ** 2) * pm).sum() / (pm.sum() * dp.shape[2] * dp.shape[3] + 1e-6)
        loss = (norm_s + self.w_l1 * l1_s) + self.w_band * (norm_b + self.w_l1 * l1_b) + self.w_grad * grad

        with torch.no_grad():
            parts = {"norm_s": norm_s.detach(), "l1_s": l1_s.detach(), "norm_b": norm_b.detach(),
                     "l1_b": l1_b.detach(), "grad": grad.detach()}
            parts.update(self.references(pred.detach(), tgt, keep_tok, bm, hole_s, hole_b, gh, gw))
        return loss, parts, pred, idx, bm

    @torch.no_grad()
    def references(self, pred, tgt, keep_tok, bm, hole_s, hole_b, gh, gw):
        """Model error / trivial-fill error on each hole type, and the grey rate."""
        b, S, c, p = tgt.shape
        # spectral holes vs interpolation along wavelength
        base_b = interp_bands(tgt.permute(0, 2, 1, 3), bm).permute(0, 2, 1, 3)
        mb = hole_b.to(pred.dtype)[..., None]
        band_ratio = ((pred - tgt).abs() * mb).sum() / (((base_b - tgt).abs() * mb).sum() + 1e-9)
        # spatial holes vs the crop's visible mean, per band
        vis = keep_tok[:, :, None, None]
        vis_mean = (tgt * vis).sum((1, 3), keepdim=True) / (vis.sum(1, keepdim=True) * p + 1e-6)
        ms = hole_s.to(pred.dtype)[..., None]
        spat_ratio = ((pred - tgt).abs() * ms).sum() / (((vis_mean - tgt).abs() * ms).sum() + 1e-9)
        # grey rate: masked units whose prediction is nearly flat relative to the truth
        u, s = self.unit, self.enc.stride

        def units(t):
            g = t.view(b, gh // u, u, gw // u, u, c, p)                # token grid -> units
            return g.permute(0, 1, 3, 5, 2, 4, 6).reshape(b, (gh // u) * (gw // u), c, u * u * p)

        pu, tu = units(pred), units(tgt)
        masked_unit = units((~keep_tok.bool()).to(pred.dtype)[:, :, None, None].expand(b, S, c, p))[..., 0, 0] > 0.5
        flat = pu.std(-1).mean(-1) < 0.1 * tu.std(-1).mean(-1)
        grey = (flat & masked_unit).sum() / masked_unit.sum().clamp(min=1)
        return {"band_vs_interp": band_ratio, "spat_vs_mean": spat_ratio, "grey": grey.to(pred.dtype)}
