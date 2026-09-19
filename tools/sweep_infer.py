#!/usr/bin/env python3
"""Measure the submission path's inference settings against the scorer that counts.

The Phase A checkpoint reads 0.4556 under ultralytics' validator and 0.4076
through this path on the same frames with pycocotools. maxDets is not the
cause -- 100 and 300 give the identical number, because the detections past the
first hundred sit at conf 0.001 and match nothing. So either the two metrics
simply disagree, or this path boxes worse than the validator's does, and 0.048
is more than the whole distance between first place and fifteenth on this
leaderboard.

Each variant is one forward pass over frames staged once, so the whole sweep is
a few GPU-minutes, and whatever wins configures the submission that counts.
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

VARIANTS = {
    # What the submission does today: square letterbox at the checkpoint's imgsz.
    "default": {},
    # Is imgsz actually carrying over from the checkpoint, or silently 640?
    "imgsz1024": {"imgsz": 1024},
    "imgsz640": {"imgsz": 640},
    # The validator batches rectangularly: aspect preserved, almost no padding.
    # These frames are 493x241, so a square letterbox pads more than half the
    # canvas with grey.
    "rect": {"rect": True},
    "rect_imgsz1024": {"rect": True, "imgsz": 1024},
    # Organizers confirmed augmented inference is not an ensemble.
    "tta": {"augment": True},
}

# Round two. The first sweep found the path is not the problem -- every setting
# landed within 0.0002 of the others except imgsz 640, which cost 0.015. That
# leaves inference resolution as the one knob shown to move anything, and it was
# only tested downward. Localization is the largest bottleneck (+0.102 in macro
# AP) and the objects are 15-45 px, so more pixels at inference is the cheap
# thing worth ruling in or out. DETR-family models often lose here, because the
# query priors are tuned to the training scale -- which is exactly why it is
# measured rather than assumed.
UPSCALE = {
    "imgsz1024": {"imgsz": 1024},
    "imgsz1280": {"imgsz": 1280},
    "imgsz1536": {"imgsz": 1536},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="xishengfeng/hod26-phasea-arch2")
    ap.add_argument("--weights", default="rtdetr-srf8_best.pt")
    ap.add_argument("--slug", default="xishengfeng/hod26-sweepinfer")
    ap.add_argument("--variants", default="VARIANTS", choices=["VARIANTS", "UPSCALE"])
    args = ap.parse_args()
    variants = {"VARIANTS": VARIANTS, "UPSCALE": UPSCALE}[args.variants]

    cand = arm(**{"train.model": "rtdetr-l"})
    ex = KaggleRoundExecutor(args.slug, timeout_hours=3.0,
                             out_dir=REPO / "runs" / "sweepinfer",
                             kernel_sources=[args.source])
    ex.push({"round": "sweep", "candidates": [],
             "submit": {"candidate": cand, "weights_from": args.weights,
                        "sweep": variants, "score_val": False,
                        "predict_test": False}})
    print(f"pushed {ex.slug}")
    print(f"  {ex.wait()}")
    sweep = ex.fetch().get("sweep", {})
    print(f"\n  {'variant':22s} {'mAP':>8} {'mAP50':>8} {'boxes':>8} {'s':>6}")
    for name, r in sorted(sweep.items(), key=lambda kv: -kv[1]["mAP"]):
        print(f"  {name:22s} {r['mAP']:>8.4f} {r['mAP50']:>8.4f} "
              f"{r['boxes']:>8} {r['seconds']:>6.0f}")
    print("\n  for reference: ultralytics' own validator gave 0.4556 on these frames")
    (REPO / "runs" / "sweepinfer" / "sweep.json").write_text(json.dumps(sweep, indent=2))


if __name__ == "__main__":
    main()
