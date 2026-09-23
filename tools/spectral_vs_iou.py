#!/usr/bin/env python3
"""Read the spectral scan against the model's boxes, on CPU, locally.

Inputs are the two files the free kernels produce:
  spectral_scan.jsonl   from tools/build_spectral_scan.py (per-instance separability)
  val_predictions.json  from a predict kernel's score_val pass (held-out boxes)

Two questions:
  1. Is there ONE class-wide spectral-shape rule that separates an object from
     the background around it, across frames? Per-instance LDA is an oracle
     upper bound -- it fits a direction per object -- whereas a detector gets one
     rule per class. Grouped by frame so no frame is on both sides of a fold.
  2. Within the held-out split, do less separable instances get looser boxes?

    python3 tools/spectral_vs_iou.py --scan spectral_scan.jsonl --preds val_predictions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from hod26.voc import CLASSES  # noqa: E402

DEFICIT = ("stone_block", "people", "e-bike", "car")


def auc(p, n):
    x = np.r_[p, n]
    o = x.argsort()
    rk = np.empty(len(x))
    rk[o] = np.arange(1, len(x) + 1)
    return (rk[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


def grouped_lda_auc(X1, X0, groups, k=5):
    frames = np.unique(groups)
    np.random.default_rng(0).shuffle(frames)
    s1, s0 = np.zeros(len(X1)), np.zeros(len(X0))
    for fold in np.array_split(frames, k):
        te = np.isin(groups, fold)
        a, b = X1[~te], X0[~te]
        d = X1.shape[1]
        Sw = (np.cov(a.T, bias=True) * len(a) + np.cov(b.T, bias=True) * len(b)) / (len(a) + len(b))
        Sw += np.eye(d) * (1e-3 * np.trace(Sw) / d + 1e-12)
        w = np.linalg.solve(Sw, a.mean(0) - b.mean(0))
        s1[te], s0[te] = X1[te] @ w, X0[te] @ w
    return auc(s1, s0)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def spearman(a, b):
    r = lambda x: np.argsort(np.argsort(np.asarray(x, float)))  # noqa: E731
    return float(np.corrcoef(r(a), r(b))[0, 1]) if len(a) > 5 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, required=True)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    recs = [json.loads(l) for l in args.scan.read_text().splitlines() if l.strip()]
    by = defaultdict(list)
    for r in recs:
        by[r["cls"]].append(r)

    print("1. one class-wide spectral-shape rule, object vs its own ring, grouped 5-fold by frame")
    rows = []
    for c, rs in by.items():
        X1 = np.array([r["core_spec"] for r in rs])
        X0 = np.array([r["ring_spec"] for r in rs])
        rows.append((c, len(rs), grouped_lda_auc(X1, X0, np.array([r["id"] for r in rs]))))
    for c, n, a in sorted(rows, key=lambda t: t[2]):
        print(f"  {('*' if c in DEFICIT else ' ') + c:<16}{n:>5}  AUC {a:.3f}")

    preds = json.loads(args.preds.read_text())
    boxes = defaultdict(list)
    for img, c, conf, *box in preds:
        if conf >= args.conf:
            boxes[(img, c)].append(box)
    held = []
    for r in recs:
        if r["val"]:
            r["iou"] = max((iou(r["box"], b) for b in boxes.get((r["id"], CLASSES.index(r["cls"])), [])),
                           default=0.0)
            held.append(r)

    print(f"\n2. separability vs matched IoU, {len(held)} held-out instances (Spearman r)")
    for name, g in (("all", held), ("deficit 4", [r for r in held if r["cls"] in DEFICIT]),
                    ("other 14", [r for r in held if r["cls"] not in DEFICIT])):
        v = [(r["auc_raw"], r["iou"]) for r in g if r["auc_raw"] is not None]
        print(f"  {name:<10} n={len(v):>5}  r(auc_raw, IoU) = {spearman(*zip(*v)):+.2f}  "
              f"median IoU {np.median([r['iou'] for r in g]):.3f}")
    g = [r for r in held if r["cls"] in DEFICIT and r["auc_raw"] is not None]
    t = np.percentile([r["auc_raw"] for r in g], [33.3, 66.7])
    for lo, hi, lab in ((-1, t[0], "least separable"), (t[0], t[1], "middle"), (t[1], 2, "most separable")):
        s = [r for r in g if lo < r["auc_raw"] <= hi]
        print(f"  deficit, {lab:<16} n={len(s):>4}  tight(IoU>=.75) {np.mean([r['iou'] >= .75 for r in s]):.1%}")


if __name__ == "__main__":
    main()
