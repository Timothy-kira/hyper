#!/usr/bin/env python3
"""Assemble a CPU-only kernel that decomposes the held-out error.

The decomposition needs the ground-truth XML, which lives in the private
dataset and is 6 GB away from here. Running it on Kaggle instead costs no GPU
quota at all -- `enable_gpu: false` -- and takes about a minute, which makes it
the cheapest measurement in this repo: it says what kind of mistake the model
is making, and therefore which of the levers is worth a session.

Assembled from src/hod26/voc.py and tools/error_analysis.py rather than
rewritten, so the decomposition run on Kaggle cannot drift from the one that
runs locally.

    python3 tools/build_diagnose.py --out-dir /tmp/diag \
        --preds-from qwyi123/hod26-predict
    kaggle kernels push -p /tmp/diag
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DROP = re.compile(r"^\s*(from\s+\.|from\s+__future__\s+import|from hod26|"
                  r"sys\.path\.insert|import sys$)")

HEAD = '''"""Error decomposition on CPU, so it costs no GPU quota.

Assembled by tools/build_diagnose.py from src/hod26/voc.py and
tools/error_analysis.py verbatim -- do not edit directly.
"""
import json, statistics
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
'''

TAIL = '''

INPUT = Path("/kaggle/input")
ann_dir = next(p for p in INPUT.rglob("train/annotations") if p.is_dir())
preds_path = next(INPUT.rglob("val_predictions.json"))
print(f"annotations: {ann_dir}", flush=True)
print(f"predictions: {preds_path}", flush=True)

ids = sorted(int(p.stem) for p in ann_dir.glob("*.xml"))
_, val_ids = split_ids(ids)
anns = [parse(ann_dir / f"{i}.xml") for i in val_ids]
print(f"{len(ids)} annotated frames, {len(val_ids)} held out\\n", flush=True)

rows = json.loads(preds_path.read_text())
by_img = defaultdict(list)
for image_id, cls, conf, x1, y1, x2, y2 in rows:
    by_img[image_id].append((cls, conf, x1, y1, x2, y2))

fate, ious, by_class, confusion, sizes, hit_conf = analyse(anns, by_img, 0.25)
n = sum(fate.values())
print(f"=== {n} ground-truth boxes over {len(anns)} frames, conf >= 0.25 ===")
for k, label in (("hit_tight", "found, IoU >= 0.75"),
                 ("hit_loose", "found, IoU 0.50-0.75"),
                 ("localization", "found, IoU 0.10-0.50 (loose)"),
                 ("classification", "boxed well, wrong class"),
                 ("missed", "not found at all")):
    print(f"  {label:34s} {fate[k]:>6}  {fate[k]/n:6.1%}")
print(f"\\nmatched IoU: median {np.median(ious):.4f}  mean {ious.mean():.4f}  "
      f"p25 {np.percentile(ious,25):.4f}  p75 {np.percentile(ious,75):.4f}")
for k in ("hit", "loose", "missed"):
    if sizes[k]:
        print(f"  median size, {k:7s}: {np.median(sizes[k]):.0f} px")

print("\\n=== per class: share of boxes not found tight ===")
rank = sorted(by_class.items(),
              key=lambda kv: -(1 - kv[1]["hit_tight"] / max(1, sum(kv[1].values()))))
for name, d in rank:
    tot = sum(d.values())
    if not tot:
        continue
    print(f"  {name:16s} n={tot:>5}  tight {d['hit_tight']/tot:5.1%}  "
          f"loose {(d['hit_loose']+d['localization'])/tot:5.1%}  "
          f"missed {d['missed']/tot:5.1%}  wrongcls {d['classification']/tot:5.1%}")

counts, areas, ars, frames = Counter(), defaultdict(list), defaultdict(list), defaultdict(set)
for i in ids:
    a = parse(ann_dir / f"{i}.xml")
    for b in a.boxes:
        c = CLASSES[b.cls_id]
        counts[c] += 1
        areas[c].append(b.area)
        ars[c].append((b.x2 - b.x1) / max(1, (b.y2 - b.y1)))
        frames[c].add(i)
stats = {c: {"instances": counts[c], "frames": len(frames[c]),
             "median_area": statistics.median(areas[c]) if areas[c] else 0,
             "median_side": statistics.median(areas[c]) ** 0.5 if areas[c] else 0,
             "median_ar": statistics.median(ars[c]) if ars[c] else 0}
         for c in CLASSES}
print("\\n=== dataset statistics (all frames) ===")
print(f"{'class':<16}{'inst':>7}{'frames':>8}{'medSide':>9}{'medAR':>8}{'small?':>8}")
for c in sorted(stats, key=lambda k: stats[k]["frames"]):
    s = stats[c]
    print(f"  {c:<14}{s['instances']:>7}{s['frames']:>8}{s['median_side']:>9.1f}"
          f"{s['median_ar']:>8.2f}{'yes' if s['median_area'] < 1024 else '':>8}")

Path("/kaggle/working/diagnosis.json").write_text(json.dumps({
    "fate": dict(fate), "stats": stats,
    "iou": {"median": float(np.median(ious)), "mean": float(ious.mean()),
            "p25": float(np.percentile(ious, 25)), "p75": float(np.percentile(ious, 75))},
    "by_class": {k: dict(v) for k, v in by_class.items()},
    "confusion": {str(k): v for k, v in confusion.items()},
}, indent=2))
print("\\nwrote diagnosis.json")
'''


def strip(path: str) -> str:
    return "\n".join(ln for ln in (REPO / path).read_text().splitlines()
                     if not DROP.match(ln))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--preds-from", default="qwyi123/hod26-predict",
                    help="kernel whose output holds val_predictions.json")
    ap.add_argument("--slug", default="qwyi123/hod26-diagnose")
    ap.add_argument("--dataset", default="xishengfeng/hod26-planar")
    args = ap.parse_args()

    body = strip("tools/error_analysis.py")
    body = body[:body.index("def main()")].replace(
        'REPO = Path(__file__).resolve().parent.parent', 'REPO = Path(".")')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "diagnose_run.py").write_text(
        HEAD + "\n" + strip("src/hod26/voc.py") + "\n\n" + body + TAIL)
    (args.out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": args.slug,
        "title": args.slug.split("/")[-1].replace("-", " ").title(),
        "code_file": "diagnose_run.py", "language": "python",
        "kernel_type": "script", "is_private": True,
        # The whole point: no accelerator, so this measurement is free.
        "enable_gpu": False, "enable_internet": False,
        "competition_sources": [], "dataset_sources": [args.dataset],
        "kernel_sources": [args.preds_from],
    }, indent=2))
    print(f"wrote {args.out_dir / 'diagnose_run.py'} (CPU only)")


if __name__ == "__main__":
    main()
