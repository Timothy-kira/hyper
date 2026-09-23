"""S3T in front of a COCO-pretrained RT-DETR.

The detector keeps its pretrained spatial machinery; S3T adds what it cannot
see. Two paths out of the spectral encoder:

1. main: a 1x1 map of the encoder's features to 3 channels, upsampled and
   added to a plain 16->3 projection of the bands, feeding the pretrained stem;
2. side: the same features pooled to P3/P4/P5 and added, through zero-
   initialised 1x1 convs, to the hybrid encoder's input projections.

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
                 grad_ckpt: bool = True, amp: bool = True):
        super().__init__()
        n, d = encoder.n_bands, encoder.dim
        self.enc, self.scale, self.grad_ckpt, self.amp = encoder, scale, grad_ckpt, amp
        self.base = nn.Conv2d(n, 3, 1, bias=False)
        with torch.no_grad():
            if projection is not None:
                self.base.weight.copy_(torch.as_tensor(projection, dtype=torch.float32).view(3, n, 1, 1))
            else:
                self.base.weight.fill_(1.0 / n)
        self.head = nn.Conv2d(d, 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def __getstate__(self):
        # The side features of the last forward are a graph-carrying tensor:
        # never pickle or deepcopy them into a checkpoint or the EMA.
        state = self.__dict__.copy()
        state.pop("_side", None)
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
        up = F.interpolate(self.head(f), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.base(x) + up


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
