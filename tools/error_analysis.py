#!/usr/bin/env python3
"""Where the held-out score is actually lost, box by box.

Aggregate AP says how much is missing, not what kind of mistake it is. This
splits every ground-truth box into the fate it met -- found and tight, found
but loose, found but called the wrong class, or not found at all -- because
those four have different fixes and the bottleneck ranking cannot tell them
apart.

The decomposition follows TIDE (Bolya et al., ECCV 2020): each error type is
scored by how much of the metric it costs, not by how often it happens.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from hod26.voc import CLASSES, parse  # noqa: E402

CACHE_SEED, VAL_FRACTION = 20260918, 0.2


def split_ids(all_ids):
    rng = np.random.RandomState(CACHE_SEED)
    ids = sorted(all_ids)
    perm = rng.permutation(len(ids))
    val = {ids[i] for i in perm[: int(round(len(ids) * VAL_FRACTION))]}
    return [i for i in ids if i not in val], [i for i in ids if i in val]


def iou_matrix(a, b):
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M)."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-9)


def analyse(anns, preds_by_img, conf_min=0.25):
    """Classify every GT box by the best prediction that reaches it."""
    fate = Counter()
    ious, by_class, conf_of_hit = [], defaultdict(Counter), []
    confusion = Counter()
    sizes = {"hit": [], "loose": [], "missed": []}

    for a in anns:
        gt = [(b.cls_id, (b.x1, b.y1, b.x2, b.y2)) for b in a.boxes]
        if not gt:
            continue
        p = [r for r in preds_by_img.get(a.image_id, []) if r[1] >= conf_min]
        pb = [r[2:] for r in p]
        pc = [r[0] for r in p]
        ps = [r[1] for r in p]
        M = iou_matrix([g[1] for g in gt], pb)
        for i, (cls, box) in enumerate(gt):
            side = float(np.sqrt((box[2] - box[0]) * (box[3] - box[1])))
            row = M[i]
            same = [j for j in range(len(pc)) if pc[j] == cls]
            best_same = max(same, key=lambda j: row[j], default=None)
            iou_same = row[best_same] if best_same is not None else 0.0
            best_any = int(np.argmax(row)) if len(row) else None
            iou_any = row[best_any] if best_any is not None else 0.0

            if iou_same >= 0.5:
                k = "hit_tight" if iou_same >= 0.75 else "hit_loose"
                fate[k] += 1
                ious.append(iou_same)
                by_class[CLASSES[cls]][k] += 1
                conf_of_hit.append(ps[best_same])
                sizes["hit" if k == "hit_tight" else "loose"].append(side)
            elif iou_same >= 0.1:
                fate["localization"] += 1
                ious.append(iou_same)
                by_class[CLASSES[cls]]["localization"] += 1
                sizes["loose"].append(side)
            elif iou_any >= 0.5:
                fate["classification"] += 1
                by_class[CLASSES[cls]]["classification"] += 1
                confusion[(CLASSES[cls], CLASSES[pc[best_any]])] += 1
                sizes["missed"].append(side)
            else:
                fate["missed"] += 1
                by_class[CLASSES[cls]]["missed"] += 1
                sizes["missed"].append(side)
    return fate, np.array(ious), by_class, confusion, sizes, np.array(conf_of_hit)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--submission", type=Path, default=None,
                    help="test-set predictions, to compare the two distributions")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    ann_dir = REPO / "data" / "hod26_planar" / "train" / "annotations"
    ids = sorted(int(p.stem) for p in ann_dir.glob("*.xml"))
    _, val_ids = split_ids(ids)
    anns = [parse(ann_dir / f"{i}.xml") for i in val_ids]

    rows = json.loads(args.preds.read_text())
    by_img = defaultdict(list)
    for image_id, cls, conf, x1, y1, x2, y2 in rows:
        by_img[image_id].append((cls, conf, x1, y1, x2, y2))

    fate, ious, by_class, confusion, sizes, hit_conf = analyse(anns, by_img, args.conf)
    n = sum(fate.values())
    print(f"{n} ground-truth boxes over {len(anns)} held-out frames, "
          f"predictions kept at conf >= {args.conf}\n")
    print("what happened to each ground-truth box")
    for k in ("hit_tight", "hit_loose", "localization", "classification", "missed"):
        label = {"hit_tight": "found, IoU >= 0.75",
                 "hit_loose": "found, IoU 0.50-0.75",
                 "localization": "found, IoU 0.10-0.50  (loose box)",
                 "classification": "boxed well, wrong class",
                 "missed": "not found at all"}[k]
        print(f"  {label:38s} {fate[k]:>6}  {fate[k] / n:6.1%}")

    if len(ious):
        print(f"\nIoU of matched boxes: median {np.median(ious):.3f}, "
              f"mean {ious.mean():.3f}, 25th pct {np.percentile(ious, 25):.3f}")
    print(f"size of found-tight boxes : median {np.median(sizes['hit']):.0f} px"
          if sizes["hit"] else "")
    print(f"size of loose boxes       : median {np.median(sizes['loose']):.0f} px"
          if sizes["loose"] else "")
    print(f"size of missed boxes      : median {np.median(sizes['missed']):.0f} px"
          if sizes["missed"] else "")

    print("\nworst classes by share of their boxes not found tight")
    rank = sorted(by_class.items(),
                  key=lambda kv: -(1 - kv[1]["hit_tight"] / max(1, sum(kv[1].values()))))
    for name, c in rank[:8]:
        tot = sum(c.values())
        print(f"  {name:16s} {tot:>5} boxes   tight {c['hit_tight'] / tot:5.1%}   "
              f"loose {(c['hit_loose'] + c['localization']) / tot:5.1%}   "
              f"wrong-class {c['classification'] / tot:5.1%}   "
              f"missed {c['missed'] / tot:5.1%}")

    if confusion:
        print("\nwrong-class confusions, most frequent first")
        for (a, b), k in confusion.most_common(8):
            print(f"  {a:16s} -> {b:16s} {k}")

    if args.submission and args.submission.exists():
        import csv
        tc, ts = Counter(), []
        with args.submission.open() as fh:
            for r in csv.DictReader(fh):
                tc[CLASSES[int(r["class_id"])]] += 1
                ts.append(float(r["confidence"]))
        vc = Counter(CLASSES[c] for c, _, *_ in
                     ((r[1], r[2]) + tuple(r[3:]) for r in rows))
        vs = np.array([r[2] for r in rows])
        ts = np.array(ts)
        print(f"\nheld-out vs test, predictions only (no test labels exist)")
        print(f"  confident predictions (conf >= 0.25): "
              f"held-out {(vs >= 0.25).sum() / len(vs) / 1:.4%} of rows, "
              f"test {(ts >= 0.25).sum() / len(ts):.4%}")
        print(f"  mean confidence of the top 10 per frame: "
              f"held-out {np.sort(vs)[-6000:].mean():.3f}, "
              f"test {np.sort(ts)[-10000:].mean():.3f}")
        vt, tt = sum(vc.values()), sum(tc.values())
        print("  class mix among confident rows (held-out vs test), largest gaps:")
        gaps = sorted(CLASSES, key=lambda c: -abs(vc[c] / vt - tc[c] / tt))[:6]
        for c in gaps:
            print(f"    {c:16s} {vc[c] / vt:6.2%}  vs  {tc[c] / tt:6.2%}")


if __name__ == "__main__":
    main()
