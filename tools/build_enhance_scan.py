#!/usr/bin/env python3
"""CPU kernel: do candidate input transforms make the deficit classes
separable from their background, without costing the other fourteen?

handoff/DIAGNOSIS.md found stone_block, people, e-bike and car are grey against
their local background: no class-wide spectral rule separates them (grouped
AUC 0.57-0.68), and the box edge carries almost no spectral or brightness step.
This tests input-side transforms taken from the hyperspectral detection
literature -- local contrast/ratio, local RX, spectral derivatives, background
whitening -- by the question a detector's first layer faces:

  pixel AUC   one linear rule per class (LDA), fitted on pixels from some
              frames and scored on pixels from others (5 folds grouped by
              frame), core pixels vs ring pixels
  edge d'     that same rule's step across the box boundary (2 px inside vs
              2 px outside), in units of its spread over the ring -- how
              visible the edge a box regressor has to find becomes

No GPU (`enable_gpu: false`). Reuses the helpers of build_spectral_scan.py.

    python3 tools/build_enhance_scan.py --out-dir /tmp/enh && kaggle kernels push -p /tmp/enh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from build_spectral_scan import function_source, strip  # noqa: E402

HEAD = '''"""Input-transform separability scan. Assembled by tools/build_enhance_scan.py. CPU only."""
import json, math, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2
'''

BODY = r'''

CACHE_SEED, VAL_FRACTION = 20260918, 0.2
DEFICIT = ("stone_block", "people", "e-bike", "car")
CHAIN = [15, 13, 14, 12, 10, 8, 9, 7, 6, 1, 0, 2, 3, 5, 4, 11]   # spectral order, from the band-correlation scan
CONTROL_CAP = 250            # instances per control class; every deficit instance is kept
PX, EPX = 40, 24             # pixels sampled per region per instance
RNG = np.random.default_rng(0)
T0 = time.time()

INPUT = Path("/kaggle/input")
ann_dir = next(p for p in INPUT.rglob("train/annotations") if p.is_dir())
img_dir = ann_dir.parent / "images"
ids = sorted(int(p.stem) for p in ann_dir.glob("*.xml"))
anns = {pid: parse(ann_dir / f"{pid}.xml") for pid in ids}
count = defaultdict(int)
keep = {}
for pid in RNG.permutation(ids):
    for i, b in enumerate(anns[pid].boxes):
        c = CLASSES[b.cls_id]
        if c in DEFICIT or count[c] < CONTROL_CAP:
            keep[(int(pid), i)] = True
            count[c] += 1
frames = sorted({pid for pid, _ in keep})
print(f"{len(keep)} instances over {len(frames)} frames", flush=True)


def box_mean(x, k):
    return cv2.blur(x, (k, k), borderType=cv2.BORDER_REFLECT)


# ---- pass 1: global background statistics for whitening / RX ----------------
bg, res = [], []
for pid in frames[::15]:
    X = np.log1p(load_planar(img_dir / f"{pid}.png").astype(np.float32))
    M = np.stack([box_mean(X[:, :, b], 31) for b in range(16)], -1)
    ys, xs = RNG.integers(0, X.shape[0], 400), RNG.integers(0, X.shape[1], 400)
    bg.append(X[ys, xs]); res.append((X - M)[ys, xs])
bg, res = np.concatenate(bg), np.concatenate(res)
mu_g = bg.mean(0)
ev, evec = np.linalg.eigh(np.cov(bg.T))
W_white = evec / np.sqrt(ev + 1e-6 * ev.max())
Sinv_res = np.linalg.inv(np.cov(res.T) + 1e-6 * np.eye(16))
print(f"background stats from {len(bg)} pixels", flush=True)


def annulus_mean(x, inner, outer):
    """Mean over an outer x outer window with the inner x inner centre removed.

    A plain local mean includes the object itself: for a 20-45 px object a
    31 px window is mostly object, so object / local-mean cancels the very
    contrast it is meant to expose. Dual-window background estimation, as in
    local RX, keeps a guard region out of the estimate.
    """
    so = cv2.blur(x, (outer, outer), borderType=cv2.BORDER_REFLECT) * outer * outer
    si = cv2.blur(x, (inner, inner), borderType=cv2.BORDER_REFLECT) * inner * inner
    return (so - si) / (outer * outer - inner * inner)


def transforms(cube):
    X = cube.astype(np.float32)
    L = np.log1p(X)
    shape = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-6)
    MA = np.stack([annulus_mean(L[:, :, b], 31, 63) for b in range(16)], -1)
    SA = np.sqrt(np.maximum(np.stack([annulus_mean(L[:, :, b] ** 2, 31, 63) for b in range(16)], -1) - MA ** 2, 1e-6))
    MB = np.stack([annulus_mean(L[:, :, b], 47, 95) for b in range(16)], -1)
    la, lb = L - MA, L - MB
    return {
        "raw16 (baseline)": L,
        "shape+raw": np.concatenate([shape, L], -1),
        "lratio_ann63": la,
        "lratio_ann95": lb,
        "lcn_ann63": la / SA,
        "lrx_ann63": np.einsum("hwi,ij,hwj->hw", la, Sinv_res, la)[..., None],
        "shape+lr_ann63": np.concatenate([shape, la], -1),
        "raw+lr_ann63+lr95": np.concatenate([L, la, lb], -1),
    }


samples = defaultdict(lambda: defaultdict(lambda: {"core": [], "ring": [], "ie": [], "oe": [], "gid": []}))
for n, pid in enumerate(frames):
    cube = load_planar(img_dir / f"{pid}.png")
    H, W, _ = cube.shape
    T = transforms(cube)
    gt = np.zeros((H, W), bool)
    for b in anns[pid].boxes:
        gt[max(0, b.y1):b.y2, max(0, b.x1):b.x2] = True
    yy, xx = np.mgrid[0:H, 0:W]
    for i, b in enumerate(anns[pid].boxes):
        if (pid, i) not in keep:
            continue
        x1, y1, x2, y2 = max(0, b.x1), max(0, b.y1), min(W, b.x2), min(H, b.y2)
        w, h = x2 - x1, y2 - y1
        if w < 4 or h < 4:
            continue
        sx, sy = max(1, round(.15 * w)), max(1, round(.15 * h))
        r = max(4, round(.5 * min(w, h)))
        X1, Y1, X2, Y2 = max(0, x1 - r), max(0, y1 - r), min(W, x2 + r), min(H, y2 + r)
        ry, rx = yy[Y1:Y2, X1:X2], xx[Y1:Y2, X1:X2]
        inside = (ry >= y1) & (ry < y2) & (rx >= x1) & (rx < x2)
        core = (ry >= y1 + sy) & (ry < y2 - sy) & (rx >= x1 + sx) & (rx < x2 - sx)
        ring = ~gt[Y1:Y2, X1:X2]
        d_in = np.minimum.reduce([ry - y1, y2 - 1 - ry, rx - x1, x2 - 1 - rx]) + 1
        d_out = np.maximum.reduce([y1 - ry, ry - (y2 - 1), x1 - rx, rx - (x2 - 1)])
        ie, oe = inside & (d_in <= 2), (~inside) & (d_out <= 2) & ring
        sel = {}
        for key, m, k in (("core", core, PX), ("ring", ring, PX), ("ie", ie, EPX), ("oe", oe, EPX)):
            idx = np.flatnonzero(m)
            if len(idx) < 4:
                sel = None
                break
            sel[key] = RNG.choice(idx, min(k, len(idx)), replace=False)
        if sel is None:
            continue
        c = CLASSES[b.cls_id]
        for name, F in T.items():
            f = F[Y1:Y2, X1:X2].reshape(-1, F.shape[-1])
            s = samples[c][name]
            for key in ("core", "ring", "ie", "oe"):
                s[key].append(f[sel[key]])
            s["gid"].append(pid)
    if (n + 1) % 300 == 0:
        print(f"  {n + 1}/{len(frames)} frames, {time.time() - T0:.0f}s", flush=True)


def auc(p, q):
    x = np.r_[p, q]
    o = x.argsort()
    rk = np.empty(len(x)); rk[o] = np.arange(1, len(x) + 1)
    return (rk[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(q))


def evaluate(s):
    g = np.array(s["gid"])
    fr = np.unique(g); RNG.shuffle(fr)
    aucs, dps = [], []
    for fold in np.array_split(fr, 5):
        te = np.isin(g, fold)
        if te.all() or not te.any():
            continue
        cat = lambda key, m: np.concatenate([a for a, t in zip(s[key], m) if t])  # noqa: E731
        A, B = cat("core", ~te), cat("ring", ~te)
        d = A.shape[1]
        Sw = (np.cov(A.T, bias=True) * len(A) + np.cov(B.T, bias=True) * len(B)) / (len(A) + len(B))
        Sw = np.atleast_2d(Sw) + np.eye(d) * (1e-3 * np.trace(np.atleast_2d(Sw)) / d + 1e-12)
        w = np.linalg.solve(Sw, A.mean(0) - B.mean(0))
        aucs.append(auc(cat("core", te) @ w, cat("ring", te) @ w))
        for k in np.flatnonzero(te):
            rp = s["ring"][k] @ w
            sd = rp.std() + 1e-9
            dps.append(((s["ie"][k] @ w).mean() - (s["oe"][k] @ w).mean()) / sd)
    return float(np.mean(aucs)), float(np.median(dps))


names = list(next(iter(samples.values())).keys())
results = {}
for c in samples:
    results[c] = {n: evaluate(samples[c][n]) for n in names}

order = [c for c in DEFICIT if c in results] + sorted(c for c in results if c not in DEFICIT)
print(f"\n=== class-wide pixel AUC, core vs ring (grouped 5-fold by frame) ===")
print(f"{'class':<15}" + "".join(f"{n[:13]:>14}" for n in names))
for c in order:
    print(f"{('*' if c in DEFICIT else ' ') + c:<15}" + "".join(f"{results[c][n][0]:>14.3f}" for n in names))
print(f"\n=== edge d' across the box boundary, same rule (median) ===")
print(f"{'class':<15}" + "".join(f"{n[:13]:>14}" for n in names))
for c in order:
    print(f"{('*' if c in DEFICIT else ' ') + c:<15}" + "".join(f"{results[c][n][1]:>14.2f}" for n in names))
print("\n=== summary: mean over deficit 4 / mean over other 14 ===")
for n in names:
    d = [results[c][n] for c in DEFICIT if c in results]
    o = [results[c][n] for c in results if c not in DEFICIT]
    print(f"  {n:<18} AUC {np.mean([x[0] for x in d]):.3f} / {np.mean([x[0] for x in o]):.3f}   "
          f"edge d' {np.mean([x[1] for x in d]):5.2f} / {np.mean([x[1] for x in o]):5.2f}")
Path("/kaggle/working/enhance_scan.json").write_text(json.dumps(results, indent=1))
print(f"\ndone in {time.time() - T0:.0f}s")
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--slug", default="qwyi123/hod26-enhance-scan2")
    ap.add_argument("--dataset", default="xishengfeng/hod26-planar")
    args = ap.parse_args()
    parts = [HEAD, strip("src/hod26/voc.py"), strip("src/hod26/cube.py"),
             function_source("tools/error_analysis.py", "split_ids"), BODY]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "enhance_scan.py").write_text("\n\n".join(parts))
    (args.out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": args.slug, "title": args.slug.split("/")[-1].replace("-", " ").title(),
        "code_file": "enhance_scan.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_internet": False,
        "competition_sources": [], "dataset_sources": [args.dataset], "kernel_sources": [],
    }, indent=2))
    print(f"wrote {args.out_dir / 'enhance_scan.py'} (CPU only)")


if __name__ == "__main__":
    main()
