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


class S3TFront(nn.Module):
    """(B, 16, H, W) in [0, 1] -> (B, 3, H, W) for the pretrained stem.

    scale: the encoder runs on the input resized by this factor. Ultralytics
    upsamples the native 493x241 cube to imgsz (about 2x at 1024), and the
    encoder was pretrained at native pixel scale, so 0.5 puts it back there --
    and quarters its memory.
    """

    def __init__(self, encoder: SpectralEncoder, projection=None, scale: float = 0.5,
                 grad_ckpt: bool = True, amp: bool = True, widen: bool = True):
        super().__init__()
        n, d = encoder.n_bands, encoder.dim
        self.enc, self.scale, self.grad_ckpt, self.amp = encoder, scale, grad_ckpt, amp
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
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                return torch.utils.checkpoint.checkpoint(self.enc, feats, use_reentrant=False).float()
            return self.enc(feats).float()

    def forward(self, x):
        f = self._encode(x)                                     # (B, D, h, w)
        self.__dict__["_side"] = f
        if self.widen:
            up = F.interpolate(f, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return torch.cat([self.base(x), up.to(x.dtype)], 1)   # (B, 3 + D, H, W)
        up = F.interpolate(self.head(f), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.base(x) + up


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
        f = self.front.__dict__.get("_side")
        if f is None:
            return y
        return y + self.proj(F.adaptive_avg_pool2d(f, y.shape[-2:]).to(y.dtype))



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
        f = self.front.__dict__.get("_side")
        if f is None:
            return y
        h, w = y.shape[-2:]
        fp = F.adaptive_avg_pool2d(f, (h, w)).to(y.dtype)          # (B, D, h, w)
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
