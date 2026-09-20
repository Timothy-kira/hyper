#!/usr/bin/env python3
"""Can a box's neighbourhood tell us how well it is localized?

The TTA sweep came back wholly negative, and its control arm said why: fusing
a single view with itself already costs 0.023, because RT-DETR is NMS-free and
the submission keeps all 300 queries per frame to fill the tail COCO
integrates over. Any clustering deletes rows, and deleted rows are pure loss.

But the loss came from *removing* boxes, not from *reordering* them, and
reordering is where the remaining theory points. COCO AP integrates a ranking
within each class, and detection confidence is known to correlate weakly with
box tightness -- the premise behind IoU-Net, GFL, VarifocalNet and Cascade-DETR
alike. A loose box ranked above a tight one costs AP even when both are found.
The usual fix is a trained IoU-prediction head, which cannot be bolted onto a
finished checkpoint; the substitute here is the same measurement Gidaris &
Komodakis's box voting uses, the agreement of a box with its own neighbourhood,
which needs no second forward pass at all.

So this keeps every row and every coordinate exactly as they are and changes
only the score. Nothing can be lost to deletion; the only question is whether
the ranking improves.

It runs on the saved predictions, so it costs no GPU. The first thing it
prints is the check the whole idea rests on: whether agreement correlates with
the true IoU against ground truth at all. If it does not, no sweep over the
exponent will rescue it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from hod26.coco_eval import evaluate  # noqa: E402
from hod26.voc import parse  # noqa: E402


def _iou_matrix(b):
    """Pairwise IoU within one (frame, class) group."""
    x1 = np.maximum(b[:, None, 0], b[None, :, 0])
    y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2])
    y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-9)


def agreement(rows, thr, weighted=True):
    """How much each box agrees with the neighbours that overlap it.

    A box is always its own neighbour, so one with nothing around it scores 1
    and is left alone rather than being penalised for being lonely. Weighting
    by the neighbours' confidence stops a cluster of junk boxes from
    corroborating each other.
    """
    groups = defaultdict(list)
    for n, (pid, cls, s, *_b) in enumerate(rows):
        groups[(pid, cls)].append(n)
    out = np.ones(len(rows))
    arr = np.asarray([r[3:] for r in rows], dtype=np.float64)
    sc = np.asarray([r[2] for r in rows], dtype=np.float64)
    for idx in groups.values():
        i = np.asarray(idx)
        m = _iou_matrix(arr[i])
        near = m >= thr
        w = (sc[i] if weighted else np.ones(len(i)))[None, :] * near
        out[i] = (w * m).sum(1) / np.maximum(w.sum(1), 1e-9)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="runs/predict_transformer_s2/val_predictions.json")
    ap.add_argument("--data", default="data/hod26_planar")
    ap.add_argument("--out", default="runs/rescore_val.json")
    args = ap.parse_args()

    rows = [tuple(r) for r in json.loads((REPO / args.preds).read_text())]
    ann_dir = REPO / args.data / "train" / "annotations"
    val_ids = sorted({int(r[0]) for r in rows})
    anns = [parse(ann_dir / f"{pid}.xml") for pid in val_ids]
    print(f"{len(rows)} predictions over {len(val_ids)} frames")

    base = evaluate(anns, rows, per_class=True)
    print(f"baseline mAP {base['mAP']:.4f}  mAP50 {base['mAP50']:.4f}")

    # The decisive check, before any sweep: does agreement track the truth?
    truth = defaultdict(list)
    for a in anns:
        for b in a.boxes:
            truth[(a.image_id, b.cls_id)].append((b.x1, b.y1, b.x2, b.y2))
    q = agreement(rows, 0.5, weighted=True)
    real, est, conf = [], [], []
    for n, (pid, cls, s, *b) in enumerate(rows):
        gts = truth.get((pid, cls))
        if not gts or s < 0.25:          # the tail is noise; judge on real detections
            continue
        best = 0.0
        for g in gts:
            ix = max(0.0, min(b[2], g[2]) - max(b[0], g[0]))
            iy = max(0.0, min(b[3], g[3]) - max(b[1], g[1]))
            inter = ix * iy
            u = (b[2]-b[0])*(b[3]-b[1]) + (g[2]-g[0])*(g[3]-g[1]) - inter
            best = max(best, inter / u if u > 0 else 0.0)
        real.append(best); est.append(q[n]); conf.append(s)
    real, est, conf = map(np.asarray, (real, est, conf))
    print(f"\non {len(real)} detections at conf>=0.25:")
    print(f"  corr(confidence, true IoU) = {np.corrcoef(conf, real)[0,1]:+.3f}"
          "   <- what the ranking uses today")
    print(f"  corr(agreement,  true IoU) = {np.corrcoef(est, real)[0,1]:+.3f}"
          "   <- what rescoring would add")
    print(f"  corr(agreement,  confidence) = {np.corrcoef(est, conf)[0,1]:+.3f}"
          "   <- how much of it is already in the confidence")

    out = {"baseline": base["mAP"],
           "corr_conf_iou": float(np.corrcoef(conf, real)[0, 1]),
           "corr_agree_iou": float(np.corrcoef(est, real)[0, 1]),
           "arms": {}}
    print(f"\n  {'arm':26s} {'mAP':>8} {'delta':>8}")
    for thr in (0.5, 0.7):
        for weighted in (True, False):
            q = agreement(rows, thr, weighted=weighted)
            for beta in (0.25, 0.5, 1.0, 2.0):
                scored = [(r[0], r[1], float(r[2] * q[n] ** beta), *r[3:])
                          for n, r in enumerate(rows)]
                e = evaluate(anns, scored)
                name = f"thr{thr}_{'w' if weighted else 'u'}_b{beta}"
                out["arms"][name] = {"mAP": e["mAP"], "mAP50": e["mAP50"],
                                     "delta": e["mAP"] - base["mAP"]}
                print(f"  {name:26s} {e['mAP']:>8.4f} {e['mAP']-base['mAP']:>+8.4f}")

    best = max(out["arms"], key=lambda k: out["arms"][k]["mAP"])
    d = out["arms"][best]["delta"]
    print(f"\n  best {best} at {out['arms'][best]['mAP']:.4f} ({d:+.4f})")
    print("  bar for using it: +0.005. Below that it is one split, not a result.")
    (REPO / args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
