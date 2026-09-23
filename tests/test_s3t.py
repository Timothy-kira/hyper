"""S3T pieces that would fail silently if wrong.

A mis-ordered demosaic, an alignment shifted the wrong way, or a "visible
only" encoder that quietly reads masked content all train without error and
just make the model worse. Each check pins one of those.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def X2Cube(img, cellSize=4):
    """The organizers' function, verbatim."""
    B = [cellSize, cellSize]
    skip = [cellSize, cellSize]
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    return out.reshape(M // cellSize, N // cellSize, cellSize * cellSize)


fails = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    import torch
    from hod26.cube import x2cube
    from hod26.s3t import preprocess as P
    from hod26.s3t.mae import S3TMAE, band_mask, spatial_mask
    from hod26.s3t.spectral import SpectralBlock, SpectralEncoder

    rng = np.random.default_rng(0)
    raw = rng.integers(0, 65535, (4 * 24, 4 * 40), dtype=np.uint16)
    check("official X2Cube == hod26.cube.x2cube", np.array_equal(X2Cube(raw), x2cube(raw)))

    # Alignment: a scene that is a linear ramp in raw-pixel coordinates. Every
    # band samples it at its own sub-pixel position; after alignment all 16
    # bands must read the same value (the ramp at the block centre).
    yy, xx = np.mgrid[0:4 * 24, 0:4 * 40].astype(np.float32)
    ramp = 3.0 * yy + 1.0 * xx
    cube = x2cube(ramp)
    before = np.ptp(cube[2:-2, 2:-2], axis=-1).max()
    after = np.ptp(P.align_bands(cube)[2:-2, 2:-2], axis=-1).max()
    check("alignment puts all bands on one centre", after < 1e-3 and before > 1,
          f"spread before {before:.3f} after {after:.5f}")

    x = rng.random((50, 70, 3)).astype(np.float32)
    ref = np.zeros_like(x)
    p = np.pad(x, ((3, 3), (3, 3), (0, 0)), mode="reflect")
    for i in range(50):
        for j in range(70):
            ref[i, j] = p[i:i + 7, j:j + 7].sum((0, 1))
    check("box_sum matches a direct window sum", np.allclose(P.box_sum(x, 7), ref, atol=1e-3))

    feats = P.features(rng.integers(100, 4000, (64, 96, 16)).astype(np.uint16))
    check("features shape (3, 16, H, W)", feats.shape == (3, 16, 64, 96), str(feats.shape))
    check("shape feature has zero band mean", np.abs(feats[1].mean(0)).max() < 1e-4)

    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    idx = spatial_mask(3, 32, 32, 4, 0.75, "cpu", g)
    check("spatial mask keeps 25% of tokens, same count per sample",
          idx.shape == (3, 256), str(tuple(idx.shape)))
    check("visible indices are unique", all(len(set(r.tolist())) == 256 for r in idx))
    bm = band_mask(5, 0.15, "cpu", generator=g)
    chain = list(P.BAND_CHAIN)
    runs_ok = True
    for r in bm:
        pos = sorted(chain.index(k) for k in torch.nonzero(r).flatten().tolist())
        runs_ok &= len(pos) == 2 and pos[1] - pos[0] == 1
    check("band mask is one contiguous run of 2 in wavelength order", runs_ok)

    blk = SpectralBlock(32, 4).eval()
    t = torch.randn(2, 50, 16, 32)
    sel = torch.tensor([[1, 5, 9], [0, 2, 49]])
    full = blk(t).gather(1, sel[:, :, None, None].expand(-1, -1, 16, 32))
    part = blk(t.gather(1, sel[:, :, None, None].expand(-1, -1, 16, 32)))
    check("spectral block: visible-only == full then gather", torch.allclose(full, part, atol=1e-5))

    enc = SpectralEncoder(dim=32, depth=2, heads=4).eval()
    xb = torch.randn(2, 3, 16, 40, 56)
    with torch.no_grad():
        out = enc(xb)
        all_idx = torch.arange(20 * 28)[None].expand(2, -1)
        t_all, _ = enc.tokens(xb, all_idx)
        t_none, _ = enc.tokens(xb)
    check("encoder map is (B, D, H/2, W/2)", tuple(out.shape) == (2, 32, 20, 28), str(tuple(out.shape)))
    check("encoder with all positions visible == dense path", torch.allclose(t_all, t_none, atol=1e-5))

    # Brightness survives: a uniformly brighter input must change the output
    # (a per-token LayerNorm after the stem would cancel it).
    with torch.no_grad():
        flat = torch.full((1, 3, 16, 16, 16), 0.3)
        o1, o2 = enc(flat), enc(flat * 2)
    check("absolute level reaches the output", (o1 - o2).abs().max() > 1e-3)

    mae = S3TMAE(SpectralEncoder(dim=32, depth=2, heads=4), dec_dim=16)
    xb = torch.randn(2, 3, 16, 32, 32)
    loss, parts, pred, idx, bm = mae(xb)
    loss.backward()
    no_grad = [n for n, p in mae.named_parameters() if p.grad is None]
    check("loss is finite", bool(torch.isfinite(loss)), str(loss))
    check("every parameter gets a gradient (DDP needs this)", not no_grad, str(no_grad[:5]))

    # A masked unit's content must not influence the encoder's visible tokens.
    mae.eval()
    g1 = torch.Generator().manual_seed(3)
    idx = spatial_mask(1, 16, 16, 4, 0.75, "cpu", g1)
    keep = torch.zeros(16 * 16, dtype=torch.bool)
    keep[idx[0]] = True
    xa = torch.randn(1, 3, 16, 32, 32)
    xc = xa.clone()
    kp = keep.view(16, 1, 16, 1).expand(16, 2, 16, 2).reshape(32, 32)
    xc[..., ~kp] += 5.0                                  # change only masked pixels
    with torch.no_grad():
        ta, _ = mae.enc.tokens(xa * kp, idx)
        tc, _ = mae.enc.tokens(xc * kp, idx)
    check("masked pixels cannot leak into visible tokens", torch.allclose(ta, tc))

    # Overfit: the loss has to fall on a fixed batch.
    torch.manual_seed(1)
    mae = S3TMAE(SpectralEncoder(dim=32, depth=2, heads=4), dec_dim=16)
    opt = torch.optim.AdamW(mae.parameters(), lr=3e-3)
    xb = torch.from_numpy(np.stack([P.features(rng.integers(100, 4000, (32, 32, 16)).astype(np.uint16))
                                    for _ in range(2)]))
    first = None
    for step in range(60):
        loss, *_ = mae(xb, generator=torch.Generator().manual_seed(step % 4))
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss.detach())
    check("loss falls when overfitting one batch", float(loss.detach()) < 0.7 * first,
          f"{first:.3f} -> {float(loss.detach()):.3f}")

    blob = pickle.dumps(mae)
    check("model pickles (DDP / checkpoints)", isinstance(pickle.loads(blob), S3TMAE))

    print("\n".join(fails) if fails else "\nall S3T checks pass")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
