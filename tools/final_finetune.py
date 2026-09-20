#!/usr/bin/env python3
"""Fold the held-out 600 frames back in for a short final fine-tune.

The split reserved 600 of 3000 annotated frames to score against. That purchase
has stopped paying: the held-out number reads 0.06 above the leaderboard for the
same checkpoint, so it is not what selects anything any more -- the leaderboard
is, and it scores on 1000 frames the model has never seen. The 600 are
therefore 25% more training data being spent on a measurement we do not use.

This should have happened when there were eighteen GPU-hours rather than four,
and at this size it is a small intervention: two epochs of exposure at the tail
of the schedule, not a run. It is worth doing only because the downside is
bounded -- session 3's submission is already banked, so the worst case is that
this one scores lower and is discarded.

Augmented copies are off. At two epochs the 600 new frames want to be seen
twice rather than once, and the same budget buys either that or a single epoch
with copies; the standard practice of finishing on clean frames breaks the tie.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dream_rsi.budget import read_quota  # noqa: E402
from dream_rsi.executor import KaggleRoundExecutor  # noqa: E402
from tools.final_runs import full_candidate  # noqa: E402
from tools.predict_submit import predict_and_submit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-session", type=int, default=3)
    ap.add_argument("--reached", type=int, required=True,
                    help="epoch the previous session finished on")
    ap.add_argument("--epochs", type=int, default=2, help="extra epochs")
    ap.add_argument("--need", type=float, default=1.0,
                    help="GPU-h required; below this, do nothing and keep what "
                         "is already submitted")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    q = read_quota()
    have = q[0] if q else float("inf")
    total = args.reached + args.epochs
    if have < args.need:
        print(f"{have:.2f} GPU-h left, need {args.need}; standing down with "
              f"session {args.from_session} banked")
        return

    cand = full_candidate("transformer", total, {"train.repeat_threshold": 0.2})
    # A short warm restart: resuming at `reached` on a cosine shaped over
    # total + 4 reopens the rate enough for two epochs to move anything,
    # without the jump a fresh schedule would give.
    cand["train"]["schedule_epochs"] = total + 4
    cand["augment"]["copies"] = 0
    print(f"fine-tune: epochs {args.reached} -> {total} on all 3000 frames, "
          f"{have:.2f} GPU-h available")
    print(f"  {({k: cand['train'][k] for k in ('epochs', 'schedule_epochs', 'repeat_threshold', 'bbox_loss')})}"
          f" copies={cand['augment']['copies']}")
    if args.dry_run:
        return

    src = f"xishengfeng/hod26-final-transformer-s{args.from_session}"
    slug = "xishengfeng/hod26-final-transformer-s4"
    ex = KaggleRoundExecutor(slug, timeout_hours=3.0,
                             out_dir=REPO / "runs" / "final_transformer_s4",
                             kernel_sources=[src])
    ex.push({"round": "final-transformer-s4", "candidates": [],
             "submit": {"candidate": cand, "use_all_train": True,
                        "predict": False,
                        "session_hours": max(1.0, have - 0.4)}})
    print(f"  pushed {slug} (continues {src})")
    print(f"  {ex.wait()}")

    r = predict_and_submit(
        source=slug, cand=cand, slug="xishengfeng/hod26-predict-s4",
        message=f"transformer session 4: all 3000 frames, epoch {total}",
        out_dir=REPO / "runs" / "predict_transformer_s4")
    print(f"\nfine-tune leaderboard: {r.get('score')}")
    print("Keep it only if it clears session 3 by more than ~0.01: the final "
          "standing is decided on a private split, and a smaller margin on the "
          "public one is not a difference.")


if __name__ == "__main__":
    main()
