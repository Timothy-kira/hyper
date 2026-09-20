#!/usr/bin/env python3
"""Measure real test-time augmentation against the scorer that counts.

The first inference sweep recorded a "tta" arm at 0.4078 beside the default's
0.4076 and concluded TTA buys nothing here. That conclusion was measuring
nothing: RTDETRDetectionModel.predict (ultralytics/nn/tasks.py) takes an
`augment` argument and never reads it -- there is no _predict_augment branch
the way DetectionModel has one -- so both arms were the same forward pass.

TTA deserves a real measurement because it is aimed at the bottleneck. The
held-out error decomposition puts 16.6% of boxes in localization error against
1.4% missing or misclassified, and a uniform 3-pixel shift is enough to take
mAP from 1.00 to 0.249. Averaging a box over several views cancels the
independent part of that coordinate noise, and Weighted Boxes Fusion is the
merge that does it -- NMS keeps one member of a cluster and throws the rest
away, WBF replaces the cluster with its confidence-weighted mean.

Only scale-preserving views are offered. The upscale sweep already ruled the
other kind out (1024 -> 0.4076, 1280 -> 0.3847, 1536 -> 0.3362): a DETR
decoder's query priors are tuned to the training scale.

The views are predicted once and every fusion setting is then CPU work over
the cached rows, so the GPU cost is three forward passes no matter how many
arms are listed -- about ten minutes for the 600 held-out frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dream_rsi.executor import KaggleRoundExecutor  # noqa: E402
from tools.final_runs import full_candidate  # noqa: E402

# The fusion threshold matters more than it looks. These are one object seen
# twice, so the pair should overlap far above the 0.5-0.6 an NMS would use --
# the median matched IoU against ground truth is already 0.864. Set it too low
# and WBF merges genuinely distinct neighbours and collapses the low-confidence
# tail the metric integrates over: on synthetic 300-query frames, a single view
# fused at 0.5 loses two thirds of its rows. Hence the control arms, which fuse
# one view with itself and measure that cost alone.
ARMS = {
    "merge_only_80": {"views": ["id"], "iou": 0.80, "rescale": False},
    "merge_only_90": {"views": ["id"], "iou": 0.90, "rescale": False},

    # A horizontal flip. The model trains with fliplr=0.5, so it is already
    # equivariant to this and the second view is in distribution by
    # construction.
    "hflip_65": {"views": ["id", "hflip"], "iou": 0.65},
    "hflip_80": {"views": ["id", "hflip"], "iou": 0.80},
    "hflip_90": {"views": ["id", "hflip"], "iou": 0.90},
    "hflip_80_keepconf": {"views": ["id", "hflip"], "iou": 0.80, "rescale": False},
    "hflip_90_keepconf": {"views": ["id", "hflip"], "iou": 0.90, "rescale": False},

    # A whole-pixel translation instead: nothing is resized, objects simply sit
    # somewhere else on the feature grid. Three original pixels is about six on
    # the 1024 canvas, deliberately off the stride-8 lattice so the two views'
    # quantisation errors are not the same error twice.
    "shift3_90_keepconf": {"views": ["id", "shift3"], "iou": 0.90, "rescale": False},

    "all3_80_keepconf": {"views": ["id", "hflip", "shift3"], "iou": 0.80, "rescale": False},
    "all3_90_keepconf": {"views": ["id", "hflip", "shift3"], "iou": 0.90, "rescale": False},
    "all3_90": {"views": ["id", "hflip", "shift3"], "iou": 0.90},

    # Rescoring by how tightly the views agree. Averaging coordinates only
    # pays to the extent the views' errors are independent, and one checkpoint
    # seen from several angles is exactly the case where they are not -- so
    # the more valuable signal in the same forward passes may be the
    # disagreement, which is informative whether or not the errors are
    # independent. It estimates the thing confidence is known not to carry:
    # COCO AP integrates a ranking within each class, and a loose box ranked
    # above a tight one costs AP even when both are found. A trained
    # IoU-prediction head is the usual answer; this is the same measurement
    # taken with the model itself, the way Soft Teacher jitters a box and
    # reads the variance of the regressions. beta sets how hard it bites.
    "all3_90_agree05": {"views": ["id", "hflip", "shift3"], "iou": 0.90,
                        "rescale": False, "rescore": {"beta": 0.5, "singleton": 0.5}},
    "all3_90_agree10": {"views": ["id", "hflip", "shift3"], "iou": 0.90,
                        "rescale": False, "rescore": {"beta": 1.0, "singleton": 0.5}},
    "all3_90_agree20": {"views": ["id", "hflip", "shift3"], "iou": 0.90,
                        "rescale": False, "rescore": {"beta": 2.0, "singleton": 0.5}},
    # Same, but a singleton keeps its confidence instead of being discounted:
    # separates "agreement helps" from "penalising lone boxes helps".
    "all3_90_agree10_nofloor": {"views": ["id", "hflip", "shift3"], "iou": 0.90,
                                "rescale": False,
                                "rescore": {"beta": 1.0, "singleton": 1.0}},
    "hflip_90_agree10": {"views": ["id", "hflip"], "iou": 0.90, "rescale": False,
                         "rescore": {"beta": 1.0, "singleton": 0.5}},
    # Box voting on a single view's own queries, which is where this idea
    # started: Gidaris & Komodakis let each box in a neighbourhood vote for
    # the location with its score as the weight, one model, no augmentation.
    "vote_only_90_agree10": {"views": ["id"], "iou": 0.90, "rescale": False,
                             "rescore": {"beta": 1.0, "singleton": 1.0}},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="xishengfeng/hod26-final-transformer-s2")
    ap.add_argument("--weights", default="final_best.pt")
    ap.add_argument("--slug", default="xishengfeng/hod26-ttasweep")
    ap.add_argument("--epochs", type=int, default=34)
    args = ap.parse_args()

    out_dir = REPO / "runs" / args.slug.split("/")[-1]
    cand = full_candidate("transformer", args.epochs)
    ex = KaggleRoundExecutor(args.slug, timeout_hours=3.0, out_dir=out_dir,
                             kernel_sources=[args.source])
    ex.push({"round": "tta", "candidates": [],
             "submit": {"candidate": cand, "weights_from": args.weights,
                        "tta_sweep": {"arms": ARMS},
                        "score_val": False, "predict_test": False}})
    print(f"pushed {ex.slug} against {args.source}/{args.weights}")
    print(f"  {ex.wait()}")

    tta = ex.fetch().get("tta", {})
    if not tta:
        print("  no tta block in results.json")
        return
    best = tta.pop("_best", None)
    base = tta.get("base", {}).get("mAP")
    print(f"\n  {'arm':24s} {'mAP':>8} {'mAP50':>8} {'delta':>8} {'boxes':>9}")
    for name, r in sorted(tta.items(), key=lambda kv: -kv[1]["mAP"]):
        d = r.get("delta")
        print(f"  {name:24s} {r['mAP']:>8.4f} {r['mAP50']:>8.4f} "
              f"{('%+.4f' % d) if d is not None else '     ---':>8} {r['boxes']:>9}")
    print(f"\n  single-view reference {base:.4f}; best arm {best}")
    print("  a win under +0.005 is not a win: it is one seed on 600 frames")
    (out_dir / "tta.json").write_text(json.dumps(tta, indent=2))


if __name__ == "__main__":
    main()
