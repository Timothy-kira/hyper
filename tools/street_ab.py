#!/usr/bin/env python3
"""Does an aspect-aware box loss tighten the boxes that are actually loose?

The error decomposition leaves one thing to fix. 96.6% of held-out ground truth
is found at IoU >= 0.5 and 0.1% is given the wrong class, so the score is the
distance from a median matched IoU of 0.864 up to the thresholds above it. And
across the eighteen classes, two things predict AP independently -- median
aspect ratio (r = -0.61, and -0.65 after controlling for frequency) and how
many frames hold the class (r = +0.61, +0.64 controlling for aspect).

CIoU is GIoU plus a centre-distance and an aspect-ratio consistency penalty, so
it addresses the first directly. This measures it where the failure lives: the
518 frames holding people, car, e-bike and stone_block, which share no frame
with the other fourteen classes. A 300-frame random proxy would give
stone_block four frames and settle nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dream_rsi.executor import KaggleRoundExecutor  # noqa: E402
from tools.phase_a import arm  # noqa: E402

STREET = ["people", "car", "e-bike", "stone_block"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--slug", default="xishengfeng/hod26-streetab")
    ap.add_argument("--arms", nargs="+", default=["giou", "ciou", "diou"])
    args = ap.parse_args()

    # The GIoU baseline is already measured -- mAP 0.4106, people 0.386,
    # car 0.562, e-bike 0.408, stone_block 0.286 -- so only the treatments run.
    all_arms = {
        "giou": {},
        "ciou": {"train.bbox_loss": "CIoU"},
        "diou": {"train.bbox_loss": "DIoU"},
    }
    cands = [(n, arm(**{"train.model": "rtdetr-l", **kw}))
             for n, kw in all_arms.items() if n in args.arms]
    cfg = {"round": "street-ab", "proxy_classes": STREET,
           "candidates": [{"node_id": n,
                           "candidate": {**c, "train": {**c["train"],
                                                        "epochs": args.epochs}}}
                          for n, c in cands]}
    for e in cfg["candidates"]:
        t = e["candidate"]["train"]
        print(f"  {e['node_id']:6s} {t['model']} {t['bbox_loss']:5s} "
              f"imgsz={t['imgsz']} ep={t['epochs']}")

    ex = KaggleRoundExecutor(args.slug, timeout_hours=4.0,
                             out_dir=REPO / "runs" / "streetab")
    ex.push(cfg)
    print(f"pushed {ex.slug}\n  {ex.wait()}")
    for r in ex.fetch().get("results", []):
        d = r.get("diagnostics") or {}
        pc = d.get("per_class", {})
        print(f"  {r['node_id']:6s} mAP={r.get('score')} mAP50={d.get('mAP50')} "
              f"{r.get('cost_seconds', 0) / 60:.0f} min")
        for c in STREET:
            if c in pc:
                print(f"      {c:14s} {pc[c]:.4f}")
        if r.get("error"):
            print(f"      ERROR {r['error'][-300:]}")


if __name__ == "__main__":
    main()
