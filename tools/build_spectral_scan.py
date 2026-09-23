#!/usr/bin/env python3
"""Assemble a CPU-only kernel that asks whether each object is spectrally
separable from the background immediately around it.

The error decomposition (handoff/DIAGNOSIS.md) says the four deficit classes
are *found* but *boxed loosely*. It does not say why. One candidate is that the
boundary is invisible in the data itself: if an object's pixels and the pixels
just outside its box have the same spectrum, no detector can place the edge
precisely. Another is that the data separates them and the pipeline's fixed
16 -> 8 Gaussian SRF average, which assumes adjacent band *indices* are
adjacent *wavelengths*, throws the difference away. This measures both, per
annotated instance, over every training frame:

  core      the box shrunk by 15% per side, clear of annotation slop
  ring      a band outside the box, every annotated box removed from it
  AUC       object-vs-ring pixel separability, cross-validated LDA with a
            4x4-block checkerboard split so neighbouring pixels never sit on
            both sides of it -- on brightness alone, on spectral shape alone
            (L2-normalised, brightness removed), on raw 16 bands, and on the
            8 channels the fixed SRF bank hands the network
  edge      spectral angle and brightness step between the 2 px just inside
            the box and the 2 px just outside it
  clutter   how much the ring and the core vary within themselves, so a
            contrast can be read against the noise it has to beat

No GPU: `enable_gpu: false`. Assembled from src/hod26 and tools/error_analysis
rather than restated, so the frames and the split are exactly the ones the
training pipeline uses.

    python3 tools/build_spectral_scan.py --out-dir /tmp/scan
    kaggle kernels push -p /tmp/scan
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DROP = re.compile(r"^\s*(from\s+\.|from\s+__future__\s+import|from hod26|"
                  r"sys\.path\.insert|import sys$)")


def strip(path: str) -> str:
    return "\n".join(ln for ln in (REPO / path).read_text().splitlines()
                     if not DROP.match(ln))


def function_source(path: str, name: str) -> str:
    src = (REPO / path).read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise KeyError(f"{name} not found in {path}")


HEAD = '''"""Spectral separability of every annotated object from its local background.

Assembled by tools/build_spectral_scan.py -- do not edit directly. CPU only.
"""
import json, math, statistics, time
from collections import defaultdict
from pathlib import Path
import numpy as np
'''

BODY = r'''

CACHE_SEED, VAL_FRACTION = 20260918, 0.2
DEFICIT = ("stone_block", "people", "e-bike", "car")
BANK = gaussian_srf_bank(16, 8, 2.0)            # the fixed front end, (8, 16)
RNG = np.random.default_rng(0)
T0 = time.time()

INPUT = Path("/kaggle/input")
ann_dir = next(p for p in INPUT.rglob("train/annotations") if p.is_dir())
img_dir = ann_dir.parent / "images"
print(f"annotations {ann_dir}\nimages      {img_dir}", flush=True)
ids = sorted(int(p.stem) for p in ann_dir.glob("*.xml"))
_, val_ids = split_ids(ids)
VAL = set(val_ids)


def auc_dir(pos, neg):
    """P(pos score > neg score), ties split -- directional, not symmetrised."""
    x = np.concatenate([pos, neg])
    order = x.argsort(kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    # average ranks over ties
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def auc_sym(pos, neg):
    a = auc_dir(pos, neg)
    return max(a, 1 - a)


def lda_cv(P, N, bP, bN):
    """Two-fold LDA, folds = 4x4-block checkerboard parity. Directional AUC on the held fold."""
    d = P.shape[1]
    out = []
    for f in (0, 1):
        tp, tn, ep, en = P[bP != f], N[bN != f], P[bP == f], N[bN == f]
        if min(len(tp), len(tn), len(ep), len(en)) < 8:
            return None
        m1, m0 = tp.mean(0), tn.mean(0)
        Sw = (np.cov(tp.T, bias=True) * len(tp) + np.cov(tn.T, bias=True) * len(tn)) / (len(tp) + len(tn))
        Sw = np.atleast_2d(Sw)
        Sw = Sw + np.eye(d) * (1e-3 * np.trace(Sw) / d + 1e-12)
        w = np.linalg.solve(Sw, m1 - m0)
        out.append(auc_dir(ep @ w, en @ w))
    return float(np.mean(out))


def angle(a, b):
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def spread(X):
    """Median per-pixel spectral angle to the set's own mean -- its clutter."""
    m = X.mean(0)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    c = np.clip(Xn @ (m / (np.linalg.norm(m) + 1e-12)), -1, 1)
    return float(np.degrees(np.median(np.arccos(c))))


def sub(idx, n):
    return idx if len(idx) <= n else RNG.choice(idx, n, replace=False)


records = []
band_sample = []
t_last = time.time()
for k, pid in enumerate(ids):
    ann = parse(ann_dir / f"{pid}.xml")
    if not ann.boxes:
        continue
    cube = load_planar(img_dir / f"{pid}.png").astype(np.float32)   # (H, W, 16)
    H, W, _ = cube.shape
    if len(band_sample) < 400000:
        ys = RNG.integers(0, H, 200); xs = RNG.integers(0, W, 200)
        band_sample.append(cube[ys, xs])
    gt = np.zeros((H, W), bool)
    boxes = []
    for b in ann.boxes:
        x1, y1 = max(0, b.x1), max(0, b.y1)
        x2, y2 = min(W, b.x2), min(H, b.y2)
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            boxes.append((b.cls_id, x1, y1, x2, y2))
            gt[y1:y2, x1:x2] = True
    yy, xx = np.mgrid[0:H, 0:W]
    block = ((yy // 4) + (xx // 4)) % 2
    for cls_id, x1, y1, x2, y2 in boxes:
        w, h = x2 - x1, y2 - y1
        sx, sy = max(1, round(0.15 * w)), max(1, round(0.15 * h))
        r = max(4, round(0.5 * min(w, h)))
        X1, Y1, X2, Y2 = max(0, x1 - r), max(0, y1 - r), min(W, x2 + r), min(H, y2 + r)
        reg = (slice(Y1, Y2), slice(X1, X2))
        ry, rx = yy[reg], xx[reg]
        inside = (ry >= y1) & (ry < y2) & (rx >= x1) & (rx < x2)
        core = (ry >= y1 + sy) & (ry < y2 - sy) & (rx >= x1 + sx) & (rx < x2 - sx)
        ring = ~gt[reg]
        d_in = np.minimum.reduce([ry - y1, y2 - 1 - ry, rx - x1, x2 - 1 - rx]) + 1
        d_out = np.maximum.reduce([y1 - ry, ry - (y2 - 1), x1 - rx, rx - (x2 - 1)])
        inner_edge = inside & (d_in <= 2)
        outer_edge = (~inside) & (d_out <= 2) & ring
        C, R = cube[reg][core], cube[reg][ring]
        if len(C) < 12 or len(R) < 16:
            continue
        ci = sub(np.arange(len(C)), 3000); ri = sub(np.arange(len(R)), min(3000, 4 * len(C) + 64))
        C, R = C[ci], R[ri]
        bC, bR = block[reg][core][ci], block[reg][ring][ri]

        def views(X):
            nrm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
            s8 = X @ BANK.T
            return {"bright": X.sum(1, keepdims=True), "shape": X / nrm, "raw": X,
                    "srf8": s8, "srf8_shape": s8 / (np.linalg.norm(s8, axis=1, keepdims=True) + 1e-9)}
        vC, vR = views(C), views(R)
        rec = {"id": pid, "cls": CLASSES[cls_id], "box": [x1, y1, x2, y2],
               "val": pid in VAL, "w": w, "h": h, "n_core": int(len(C)), "n_ring": int(len(R))}
        rec["auc_bright"] = auc_sym(vC["bright"][:, 0], vR["bright"][:, 0])
        for key in ("shape", "raw", "srf8", "srf8_shape"):
            rec["auc_" + key] = lda_cv(vC[key], vR[key], bC, bR)
        mC, mR = C.mean(0), R.mean(0)
        rec["sam_core_ring"] = angle(mC, mR)
        rec["log_bright"] = float(math.log((mC.sum() + 1e-6) / (mR.sum() + 1e-6)))
        rec["clutter_core"] = spread(C)
        rec["clutter_ring"] = spread(R)
        rec["band_contrast"] = [float(v) for v in (mC - mR) / (mR + 1e-6)]
        rec["core_spec"] = [float(v) for v in mC / (np.linalg.norm(mC) + 1e-12)]
        rec["ring_spec"] = [float(v) for v in mR / (np.linalg.norm(mR) + 1e-12)]
        IE, OE = cube[reg][inner_edge], cube[reg][outer_edge]
        if len(IE) >= 4 and len(OE) >= 4:
            rec["edge_sam"] = angle(IE.mean(0), OE.mean(0))
            rec["edge_log_bright"] = float(math.log((IE.mean(0).sum() + 1e-6) / (OE.mean(0).sum() + 1e-6)))
        records.append(rec)
    if time.time() - t_last > 60:
        print(f"  {k + 1}/{len(ids)} frames, {len(records)} objects, {time.time() - T0:.0f}s", flush=True)
        t_last = time.time()

print(f"\n{len(records)} objects from {len(ids)} frames in {time.time() - T0:.0f}s\n", flush=True)

# ---- band order: is index order spectral order? ---------------------------
S = np.concatenate(band_sample)
S = S / (S.sum(1, keepdims=True) + 1e-9)            # shape only, so illumination does not dominate
Cm = np.corrcoef(S.T)
print("=== band-to-band correlation of normalised spectra (index order) ===")
print("     " + "".join(f"{j:>6}" for j in range(16)))
for i in range(16):
    print(f"{i:>4} " + "".join(f"{Cm[i, j]:>6.2f}" for j in range(16)))
adj = np.mean([Cm[i, i + 1] for i in range(15)])
rand = np.mean([Cm[i, j] for i in range(16) for j in range(16) if abs(i - j) > 3])
print(f"\nmean corr, index-adjacent pairs: {adj:.3f}   index-distant (|i-j|>3): {rand:.3f}")
# greedy chain through strongest correlations -- a guess at the spectral order
order = [int(np.argmin(Cm.sum(0)))]
while len(order) < 16:
    last = order[-1]
    cand = [(Cm[last, j], j) for j in range(16) if j not in order]
    order.append(max(cand)[1])
print(f"order by correlation chain: {order}\n")

# ---- per-class summary -------------------------------------------------------
def q(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return "   -     -     -  "
    a = np.percentile(v, [25, 50, 75])
    return f"{a[0]:5.3f} {a[1]:5.3f} {a[2]:5.3f}"

by = defaultdict(list)
for r in records:
    by[r["cls"]].append(r)
names = sorted(by, key=lambda c: (c not in DEFICIT, np.median([r["auc_raw"] for r in by[c] if r["auc_raw"] is not None] or [0])))
print("=== object-vs-local-background separability, AUC quartiles (25/50/75) ===")
print("  (0.5 = indistinguishable, 1.0 = perfectly separable; LDA ones are cross-validated)")
print(f"{'class':<15}{'n':>5}  {'brightness only':<18}{'shape only (16)':<18}{'raw 16 bands':<18}{'SRF 8 (net input)':<18}")
for c in names:
    rs = by[c]
    print(f"{('*' if c in DEFICIT else ' ') + c:<15}{len(rs):>5}  {q([r['auc_bright'] for r in rs]):<18}"
          f"{q([r['auc_shape'] for r in rs]):<18}{q([r['auc_raw'] for r in rs]):<18}{q([r['auc_srf8'] for r in rs]):<18}")

print("\n=== contrast against clutter (medians) ===")
print(f"{'class':<15}{'SAM core-ring':>14}{'clutter ring':>13}{'clutter core':>13}{'SAM/clutter':>12}{'log bright':>11}{'edge SAM':>9}{'edge logB':>10}")
for c in names:
    rs = by[c]
    med = lambda k: float(np.median([r[k] for r in rs if r.get(k) is not None]))
    ratio = float(np.median([r["sam_core_ring"] / (0.5 * (r["clutter_ring"] + r["clutter_core"]) + 1e-9) for r in rs]))
    print(f"{('*' if c in DEFICIT else ' ') + c:<15}{med('sam_core_ring'):>14.2f}{med('clutter_ring'):>13.2f}{med('clutter_core'):>13.2f}"
          f"{ratio:>12.2f}{med('log_bright'):>11.3f}{med('edge_sam'):>9.2f}{med('edge_log_bright'):>10.3f}")

print("\n=== per-band contrast (core-ring)/ring, class median, by band index ===")
print(f"{'class':<15}" + "".join(f"{j:>6}" for j in range(16)))
for c in names:
    bc = np.median(np.array([r["band_contrast"] for r in by[c]]), 0)
    print(f"{('*' if c in DEFICIT else ' ') + c:<15}" + "".join(f"{v:>6.2f}" for v in bc))

print("\n=== mean normalised spectrum, core vs ring (deficit classes) ===")
for c in DEFICIT:
    if c not in by:
        continue
    cs = np.mean([r["core_spec"] for r in by[c]], 0); rsp = np.mean([r["ring_spec"] for r in by[c]], 0)
    print(f"{c:<12} core " + " ".join(f"{v:.3f}" for v in cs))
    print(f"{'':<12} ring " + " ".join(f"{v:.3f}" for v in rsp))

Path("/kaggle/working/spectral_scan.jsonl").write_text("\n".join(json.dumps(r) for r in records))
Path("/kaggle/working/band_corr.json").write_text(json.dumps({"corr": Cm.tolist(), "chain_order": order}))
print("\nwrote spectral_scan.jsonl, band_corr.json")
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--slug", default="qwyi123/hod26-spectral-scan")
    ap.add_argument("--dataset", default="xishengfeng/hod26-planar")
    args = ap.parse_args()

    parts = [
        HEAD,
        strip("src/hod26/voc.py"),
        strip("src/hod26/cube.py"),
        function_source("src/hod26/spectral.py", "gaussian_srf_bank"),
        function_source("tools/error_analysis.py", "split_ids"),
        BODY,
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "spectral_scan.py").write_text("\n\n".join(parts))
    (args.out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": args.slug,
        "title": args.slug.split("/")[-1].replace("-", " ").title(),
        "code_file": "spectral_scan.py", "language": "python",
        "kernel_type": "script", "is_private": True,
        # CPU only: the whole point is that this costs no GPU quota.
        "enable_gpu": False, "enable_internet": False,
        "competition_sources": [], "dataset_sources": [args.dataset],
        "kernel_sources": [],
    }, indent=2))
    print(f"wrote {args.out_dir / 'spectral_scan.py'} (CPU only)")


if __name__ == "__main__":
    main()
