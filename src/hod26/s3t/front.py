"""S3T in front of a COCO-pretrained RT-DETR.

The detector keeps its pretrained spatial machinery; S3T adds what it cannot
see. Two paths out of the spectral encoder:

1. main: the encoder's 64-d features, upsampled to full resolution, go into
   the pretrained stem *beside* a plain 16->3 projection of the bands. The
   stem's first conv is widened from 3 to 3 + 64 input channels: the first 3
   keep their COCO weights, the new 64 start at zero (I3D-style inflation with
   a ControlNet-style zero init). Nothing is squeezed through 3 channels any
   more, and step 0 is still exactly the pretrained stem on the projection.
   (widen=False keeps the older form: features mapped to 3 channels by a
   zero-init 1x1 and added to the projection -- a 64 -> 3 bottleneck);
2. side: the same features pooled to P3/P4/P5 and added, through zero-
   initialised 1x1 convs, to the hybrid encoder's input projections. At P3
   and P4 the pooled spectral features first cross-attend to AIFI's output --
   the detector's global, COCO-pretrained context -- which is already
   computed by then (layer 11 runs before 14 and 19). That is the
   spatial -> spectral direction; the two paths above are spectral -> spatial.

Both new outputs start at zero, so step 0 is the projection alone and the
pretrained detector sees a sane image; the spectral features are phased in by
gradient rather than dropped in at full strength.

S3TXFront is the same idea for the S3T-X encoder (xca.py), wired where the
grids already agree: its stride-4 output joins the HGStem output (stride 4)
through a zero-init 1x1 instead of being upsampled to the input and pushed
through a widened full-resolution conv, and a learned strided pyramid replaces
the average pooling for the side injections.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import LEVEL_LO, LEVEL_SPAN
from .spectral import SpectralEncoder


def _box_mean(x: torch.Tensor, k: int) -> torch.Tensor:
    """k x k window mean over (N, C, H, W), reflect-padded, via an integral image."""
    r = k // 2
    h, w = x.shape[-2:]
    if r >= h or r >= w:                       # tiny input: reflect needs r < size
        x = F.pad(x, (r, r, r, r), mode="replicate")
    else:
        x = F.pad(x, (r, r, r, r), mode="reflect")
    s = x.double().cumsum(-1).cumsum(-2)
    s = F.pad(s, (1, 0, 1, 0))
    out = s[..., k:k + h, k:k + w] - s[..., :h, k:k + w] - s[..., k:k + h, :w] + s[..., :h, :w]
    return (out / (k * k)).to(x.dtype)


def level_features(level: torch.Tensor, inner: int = 31, outer: int = 63) -> torch.Tensor:
    """(B, 16, H, W) aligned level -> (B, 3, 16, H, W): level, shape, contrast.

    The torch twin of preprocess.features (minus the alignment, which the
    renderer already applied), so fine-tuning sees what pretraining saw.
    """
    shape = level - level.mean(1, keepdim=True)
    ring = (_box_mean(level, outer) * outer * outer - _box_mean(level, inner) * inner * inner) \
        / float(outer * outer - inner * inner)
    return torch.stack([level, shape, level - ring], 1)


def _box_sum(x: torch.Tensor, k: int) -> torch.Tensor:
    """k x k window sums over (N, C, H, W), zero outside: two 1-D average pools."""
    r = k // 2
    y = F.avg_pool2d(x, (1, k), stride=1, padding=(0, r), count_include_pad=True)
    return F.avg_pool2d(y, (k, 1), stride=1, padding=(r, 0), count_include_pad=True) * (k * k)


def observed_features(level: torch.Tensor, vis: torch.Tensor | None = None,
                      band_vis: torch.Tensor | None = None, inner: int = 31,
                      outer: int = 63) -> torch.Tensor:
    """(B, 16, H, W) aligned level -> (B, 3, 16, H, W) from the observed part only.

    level, shape and contrast as in level_features, but every statistic is
    taken over what is observed, so none of the three carries anything of a
    masked pixel or band:
      shape     level minus the mean of the *present* bands (band_vis, (B, 16));
      contrast  level minus the mean of the *observed* pixels of the 31/63
                annulus (vis, (B, 1, H, W)): a normalised convolution. Outside
                the image counts as unobserved, so a border pixel's ring is the
                part of it inside the image.
    Masked pixels and bands come out as zero. With nothing masked this is the
    detector's input; S3T-X is fine-tuned on exactly what it was pretrained on.
    """
    b, c, h, w = level.shape
    level = level.float()
    m = torch.ones(b, 1, h, w, device=level.device) if vis is None else vis.float()
    bv = torch.ones(b, c, device=level.device) if band_vis is None else band_vis.float()
    bv = bv[:, :, None, None]
    obs = m * bv                                                       # (B, C, H, W)
    lv = level * obs
    mean_b = lv.sum(1, keepdim=True) / bv.sum(1, keepdim=True).clamp(min=1.0)
    shape = (level - mean_b) * obs
    num = _box_sum(lv, outer) - _box_sum(lv, inner)
    den = _box_sum(m, outer) - _box_sum(m, inner)
    ok = den > 0.5
    ring = num / den.clamp(min=1.0)
    contrast = torch.where(ok, level - ring, torch.zeros_like(level)) * obs
    return torch.stack([lv, shape, contrast], 1)


class S3TFront(nn.Module):
    """(B, 16, H, W) in [0, 1] -> (B, 3, H, W) for the pretrained stem.

    scale: the encoder runs on the input resized by this factor. Ultralytics
    upsamples the native 493x241 cube to imgsz (about 2x at 1024), and the
    encoder was pretrained at native pixel scale, so 0.5 puts it back there --
    and quarters its memory.
    """

    def __init__(self, encoder: SpectralEncoder, projection=None, scale: float = 0.5,
                 grad_ckpt: bool = True, amp: bool = True, widen: bool = True,
                 ckpt_chunks: int = 8, fast_kernels: bool = False, train_encoder: bool = True):
        super().__init__()
        n, d = encoder.n_bands, encoder.dim
        self.enc, self.scale, self.grad_ckpt, self.amp = encoder, scale, grad_ckpt, amp
        # 0: one checkpoint around the whole encoder (the form that OOM'd: its
        # backward recomputes every layer at once, ~13 GB per 1024^2 image).
        # N > 0: checkpoint each layer separately, and split the per-position
        # layers (spectral blocks, the pool, the stem) into N chunks of
        # positions, so a backward holds one chunk of one layer at a time.
        self.ckpt_chunks = int(ckpt_chunks)
        # Kernel choices that keep the arithmetic: 16-token attention as batched
        # matmuls, depthwise convs in channels_last. Weights are untouched.
        if fast_kernels:
            from .spectral import SpatialMix, SpectralAttention
            for m in encoder.modules():
                if isinstance(m, SpectralAttention):
                    m.impl = "bmm"
                elif isinstance(m, SpatialMix):
                    m.channels_last = True
        # False: the encoder is a fixed feature extractor during detection (no
        # backward through it, no recomputation); the stem channels and the
        # injections still train. An option to measure, not the default.
        self.train_encoder = bool(train_encoder)
        if not self.train_encoder:
            for p_ in encoder.parameters():
                p_.requires_grad_(False)
        self.widen = widen
        # Channels this front hands the stem: 3 (projection) + d when widened.
        self.out_channels = 3 + d if widen else 3
        self.base = nn.Conv2d(n, 3, 1, bias=False)
        with torch.no_grad():
            if projection is not None:
                self.base.weight.copy_(torch.as_tensor(projection, dtype=torch.float32).view(3, n, 1, 1))
            else:
                self.base.weight.fill_(1.0 / n)
        if not widen:
            self.head = nn.Conv2d(d, 3, 1)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def __getstate__(self):
        # The side features of the last forward are a graph-carrying tensor:
        # never pickle or deepcopy them into a checkpoint or the EMA.
        state = self.__dict__.copy()
        state.pop("_side", None)
        state.pop("_ctx", None)
        return state

    def _encode(self, x):
        level = x * LEVEL_SPAN + LEVEL_LO
        if self.scale != 1.0:
            level = F.interpolate(level, scale_factor=self.scale, mode="bilinear",
                                  align_corners=False, antialias=True)
        feats = level_features(level)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp and x.is_cuda):
            if not getattr(self, "train_encoder", True):
                with torch.no_grad():
                    return self.enc(feats).float()
            ckpt = self.grad_ckpt and self.training and torch.is_grad_enabled()
            if ckpt and self.ckpt_chunks > 0:
                return self._encode_chunked(feats).float()
            if ckpt:
                return torch.utils.checkpoint.checkpoint(self.enc, feats, use_reentrant=False).float()
            return self.enc(feats).float()

    def _encode_chunked(self, feats):
        """SpectralEncoder.forward, checkpointed layer by layer and chunk by chunk.

        Same modules, same order, same arithmetic as enc(feats); only what is
        kept for backward changes. Spectral blocks, the band pool and the band
        embedding act on each position independently, so they run on slices of
        the position axis; SpatialMix needs the whole grid and is checkpointed
        whole (it is the cheaper layer).
        """
        from torch.utils.checkpoint import checkpoint
        from .spectral import SpatialMix

        enc, n = self.enc, self.ckpt_chunks
        b, f, c, h, w = feats.shape

        def stem(z):
            y = enc.stem(z.transpose(1, 2).reshape(b * c, f, h, w))
            return y

        y = checkpoint(stem, feats, use_reentrant=False)
        gh, gw = y.shape[-2:]
        t = y.view(b, c, enc.dim, gh * gw).permute(0, 3, 1, 2) + enc.band_pe   # (B, S, C, D)

        def per_chunk(fn, t):
            s = t.shape[1]
            step = -(-s // n)
            return torch.cat([checkpoint(fn, t[:, i:i + step], use_reentrant=False)
                              for i in range(0, s, step)], 1)

        for layer in enc.layers:
            if isinstance(layer, SpatialMix):
                t = checkpoint(layer, t, None, (gh, gw), use_reentrant=False)
            else:
                t = per_chunk(layer, t)
        p = per_chunk(lambda z: enc.norm(enc.pool(z)), t)                   # (B, S, D)
        return p.transpose(1, 2).reshape(b, enc.dim, gh, gw)

    def side_at(self, size):
        """The spectral features on a detector level's grid, or None before a forward."""
        f = self.__dict__.get("_side")
        return None if f is None else F.adaptive_avg_pool2d(f, size)

    def forward(self, x):
        f = self._encode(x)                                     # (B, D, h, w)
        self.__dict__["_side"] = f
        if self.widen:
            up = F.interpolate(f, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return torch.cat([self.base(x), up.to(x.dtype)], 1)   # (B, 3 + D, H, W)
        up = F.interpolate(self.head(f), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.base(x) + up


class SpectralPyramid(nn.Module):
    """Stride-4 spectral features -> strides 8, 16, 32 (the detector's P3, P4, P5).

    Strided 3x3 convolutions rather than average pooling: a target a few pixels
    wide is averaged into its background by a pool, and the grey classes differ
    from their background only in spectrum. Each level ends in a per-pixel
    LayerNorm so every injection sees unit-scale input.
    """

    def __init__(self, dim: int, levels: int = 3):
        super().__init__()
        self.down = nn.ModuleList(nn.Conv2d(dim, dim, 3, stride=2, padding=1) for _ in range(levels))
        self.norm = nn.ModuleList(nn.LayerNorm(dim) for _ in range(levels))

    def forward(self, f):
        from .xca import ln_nhwc
        out = []
        for conv, ln in zip(self.down, self.norm):
            f = F.gelu(conv(f))
            out.append(ln_nhwc(f, ln))
        return out


class S3TXFront(nn.Module):
    """S3T-X in front of the detector: fused where the grids already agree.

    (B, 16, H, W) in [0, 1] -> (B, 3, H, W): the plain 16 -> 3 projection, for
    the pretrained stem, unchanged. The spectral features never pass through
    three channels and are never upsampled:

    * the encoder runs at `scale` (0.5: native pixel scale) with stride 2, so
      its grid is the input's stride 4 -- exactly the grid of RT-DETR's HGStem
      output. `fuse_stem` adds them there through a zero-initialised 1x1 conv
      (D -> the stem's 48 channels); the wrapper around the stem calls it.
    * a learned pyramid takes them to strides 8/16/32 for the side injections
      (`side_at`), which may first read AIFI's global context (ContextInject).

    Every new path starts at zero: step 0 is the pretrained detector on the
    projection, exactly.

    upsample: the data loader delivers the input at 1/upsample of the
    detector's resolution and the 2x happens here, on the GPU. The cube is
    493x241 natively, so a 512 loader loses nothing; rendering, mosaic and
    affine augmentation at 1024 cost the 4 vCPUs of a T4 pair 3.5x more per
    sample (145 vs 41 ms) and left the GPUs 40% idle. The 16->3 projection is
    applied first and only its 3 channels are resized (a 1x1 conv commutes with
    bilinear interpolation), and the encoder, which wants native scale, reads
    the loader's input directly (scale * upsample = 1: no resize at all).
    """

    def __init__(self, encoder, projection=None, scale: float = 0.5, grad_ckpt: bool = True,
                 amp: bool = True, stem_ch: int = 48, train_encoder: bool = True, upsample: int = 1):
        super().__init__()
        n, d = encoder.n_bands, encoder.dim
        self.enc, self.scale, self.amp = encoder, scale, amp
        self.upsample = int(upsample)
        encoder.grad_ckpt = bool(grad_ckpt)
        self.train_encoder = bool(train_encoder)
        if not self.train_encoder:
            for p_ in encoder.parameters():
                p_.requires_grad_(False)
        self.out_channels = 3
        self.base = nn.Conv2d(n, 3, 1, bias=False)
        with torch.no_grad():
            if projection is not None:
                self.base.weight.copy_(torch.as_tensor(projection, dtype=torch.float32).view(3, n, 1, 1))
            else:
                self.base.weight.fill_(1.0 / n)
        self.pyramid = SpectralPyramid(d)
        self.fuse = nn.Conv2d(d, stem_ch, 1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def __getstate__(self):
        state = self.__dict__.copy()
        for k in ("_side", "_pyr", "_ctx"):
            state.pop(k, None)
        return state

    def _encode(self, x):
        # Features in fp32 whatever the model's dtype: the ring means and the
        # per-band statistics are sums over many pixels.
        level = x.float() * LEVEL_SPAN + LEVEL_LO
        s = self.scale * getattr(self, "upsample", 1)
        if s != 1.0:
            level = F.interpolate(level, scale_factor=s, mode="bilinear",
                                  align_corners=False, antialias=s < 1)
        feats = observed_features(level)
        wdt = self.fuse.weight.dtype
        amp = self.amp and x.is_cuda
        if not amp:
            feats = feats.to(wdt)
        # Under autocast the encoder's LayerNorms, normalisations and softmaxes
        # run in fp32 even in an fp16 model (ultralytics validates and predicts
        # with model.half()), so the output can come back fp32: hand it on in
        # the dtype of the layers that consume it.
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            if not self.train_encoder:
                with torch.no_grad():
                    f = self.enc(feats)
            else:
                f = self.enc(feats)
            pyr = self.pyramid(f.to(wdt) if not amp else f)
        return f.to(wdt), [p_.to(wdt) for p_ in pyr]

    def side_at(self, size):
        pyr = self.__dict__.get("_pyr")
        if pyr is None:
            return None
        size = tuple(size)
        for p_ in pyr:
            if tuple(p_.shape[-2:]) == size:
                return p_
        # an input size the strides do not divide: the finest level at least
        # as large as the target, pooled onto it
        big = [p_ for p_ in pyr if p_.shape[-2] >= size[0] and p_.shape[-1] >= size[1]]
        return F.adaptive_avg_pool2d(big[-1] if big else pyr[0], size)

    def fuse_stem(self, y):
        """HGStem output (B, 48, H/4, W/4) plus the spectral features on the same grid."""
        f = self.__dict__.get("_side")
        if f is None:
            return y
        if f.shape[-2:] != y.shape[-2:]:
            f = F.interpolate(f.float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
        return y + self.fuse(f.to(self.fuse.weight.dtype)).to(y.dtype)

    def forward(self, x):
        f, pyr = self._encode(x)
        self.__dict__["_side"], self.__dict__["_pyr"] = f, pyr
        y = self.base(x)
        u = getattr(self, "upsample", 1)
        if u > 1:
            y = F.interpolate(y, scale_factor=u, mode="bilinear", align_corners=False)
        return y


def widen_first_conv(block: nn.Module, extra: int) -> nn.Conv2d:
    """Give the block's first conv `extra` more input channels, zero-initialised.

    The original input channels keep their (pretrained) weights, so on inputs
    whose extra channels are anything at all the output is unchanged at step 0;
    gradient then decides how much of the new channels to use.
    """
    old = next(m for m in block.modules() if isinstance(m, nn.Conv2d))
    new = nn.Conv2d(old.in_channels + extra, old.out_channels, old.kernel_size, old.stride,
                    old.padding, old.dilation, old.groups, bias=old.bias is not None,
                    padding_mode=old.padding_mode).to(old.weight.device, old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :old.in_channels] = old.weight
        if old.bias is not None:
            new.bias.copy_(old.bias)
    for parent in block.modules():
        for name, child in parent._modules.items():
            if child is old:
                parent._modules[name] = new
                return new
    raise RuntimeError("first conv not found in its block")


class Inject(nn.Module):
    """Wraps one detector layer and adds the pooled spectral features to its output."""

    def __init__(self, layer: nn.Module, front: S3TFront, out_ch: int):
        super().__init__()
        self.layer = layer
        self.proj = nn.Conv2d(front.enc.dim, out_ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        # Not a submodule: registering it here would put the front's weights in
        # the state_dict twice. Pickle and deepcopy keep the identity anyway.
        self.__dict__["front"] = front
        for attr in ("i", "f", "type", "np"):
            if hasattr(layer, attr):
                setattr(self, attr, getattr(layer, attr))

    def forward(self, x):
        y = self.layer(x)
        f = self.front.side_at(y.shape[-2:])
        if f is None:
            return y
        return y + self.proj(f.to(y.dtype))



def sincos_2d(h: int, w: int, dim: int, like: torch.Tensor) -> torch.Tensor:
    """(h*w, dim) sine-cosine encoding of *normalised* (y, x) in (0, 1).

    Normalised rather than integer positions, so a P3 query and a P5 key at the
    same place in the image get the same code whatever the grid sizes.
    """
    nf = dim // 4
    ys = (torch.arange(h, device=like.device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=like.device, dtype=torch.float32) + 0.5) / w
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    freq = torch.pi * 2.0 ** torch.arange(nf, device=like.device, dtype=torch.float32)
    ay = gy.reshape(-1, 1) * freq
    ax = gx.reshape(-1, 1) * freq
    return torch.cat([ay.sin(), ay.cos(), ax.sin(), ax.cos()], 1).to(like.dtype)


class Tap(nn.Module):
    """Wraps AIFI and leaves its output (the global context) on the front."""

    def __init__(self, layer: nn.Module, front: S3TFront):
        super().__init__()
        self.layer = layer
        self.__dict__["front"] = front
        for attr in ("i", "f", "type", "np"):
            if hasattr(layer, attr):
                setattr(self, attr, getattr(layer, attr))

    def forward(self, x):
        y = self.layer(x)
        self.front.__dict__["_ctx"] = y
        return y


class ContextInject(Inject):
    """Inject, after letting the spectral features read the detector's global context.

    Queries: the spectral features pooled to this level's grid. Keys/values:
    AIFI's tokens (P5, about 32x16 at imgsz 1024, 256-d). One multi-head
    cross-attention with normalised 2-D sin-cos positions on both sides, added
    residually, then the zero-initialised 1x1 projection as in Inject -- so
    step 0 is still exactly the pretrained detector.
    """

    def __init__(self, layer: nn.Module, front: S3TFront, out_ch: int, ctx_ch: int = 256,
                 dk: int = 64, heads: int = 4):
        super().__init__(layer, front, out_ch)
        d = front.enc.dim
        self.heads, self.dk = heads, dk
        self.nq, self.nk = nn.LayerNorm(d), nn.LayerNorm(ctx_ch)
        self.q, self.k, self.v = nn.Linear(d, dk), nn.Linear(ctx_ch, dk), nn.Linear(ctx_ch, dk)
        self.o = nn.Linear(dk, d)

    def forward(self, x):
        y = self.layer(x)
        h, w = y.shape[-2:]
        fp = self.front.side_at((h, w))
        if fp is None:
            return y
        fp = fp.to(y.dtype)                                          # (B, D, h, w)
        ctx = self.front.__dict__.get("_ctx")
        if ctx is not None:
            b, d = fp.shape[:2]
            tq = fp.flatten(2).transpose(1, 2)                        # (B, N, D)
            kv = ctx.flatten(2).transpose(1, 2).to(y.dtype)           # (B, M, C)
            q = self.q(self.nq(tq)) + sincos_2d(h, w, self.dk, tq)
            k = self.k(self.nk(kv)) + sincos_2d(ctx.shape[-2], ctx.shape[-1], self.dk, kv)
            v = self.v(self.nk(kv))
            split = lambda t: t.view(b, t.shape[1], self.heads, self.dk // self.heads).transpose(1, 2)
            o = F.scaled_dot_product_attention(split(q), split(k), split(v))
            o = o.transpose(1, 2).reshape(b, -1, self.dk)
            fp = (tq + self.o(o)).transpose(1, 2).reshape(b, d, h, w)
        return y + self.proj(fp)
