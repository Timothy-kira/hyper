#!/usr/bin/env python3
"""Score an existing checkpoint on the held-out split through the prediction path.

Splits the calibration gap. The Phase A checkpoint scored 0.4556 under
ultralytics' validator during training and 0.40249 on the leaderboard. Those
differ in two ways at once -- scorer (ultralytics vs pycocotools) and inference
path (the val dataloader vs predict) on one side, and val frames vs test frames
on the other. Holding the frames fixed and changing only the scorer and path
says which one the 0.053 belongs to.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dream_rsi.executor import KaggleRoundExecutor  # noqa: E402
from tools.phase_a import arm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="xishengfeng/hod26-phasea-arch2")
    ap.add_argument("--weights", default="rtdetr-srf8_best.pt")
    ap.add_argument("--slug", default="xishengfeng/hod26-scoreval")
    args = ap.parse_args()

    cand = arm(**{"train.model": "rtdetr-l"})
    ex = KaggleRoundExecutor(args.slug, timeout_hours=2.0,
                             out_dir=REPO / "runs" / "scoreval",
                             kernel_sources=[args.source])
    ex.push({"round": "scoreval", "candidates": [],
             "submit": {"candidate": cand, "weights_from": args.weights,
                        "score_val": True}})
    print(f"pushed {ex.slug}")
    print(f"  {ex.wait()}")
    p = ex.fetch()
    print(f"\n  ultralytics validator, during training : 0.4556")
    print(f"  pycocotools + predict, same val frames : {p.get('mAP'):.4f}")
    print(f"  pycocotools + predict, test frames (LB): 0.40249")
    print(f"\n  {p.get('frames')} frames, {p.get('boxes')} boxes, mAP50 {p.get('mAP50'):.4f}")


if __name__ == "__main__":
    main()
