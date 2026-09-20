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
    ap.add_argument("--from-session", type=int, default=2,
                    help="which session's checkpoint to continue. Session 2 is "
                         "the best model this run produced (held-out 0.6873, "
                         "LB 0.62584); session 3 scored below it because the "
                         "resume bug made it a fresh 10-epoch run")
    ap.add_argument("--repeat-threshold", type=float, default=0.0,
                    help="repeat-factor sampling. Off by default: session 2 "
                         "trained without it, and folding 600 frames in is "
                         "already one change to the data at the tail of a "
                         "schedule. Session 3 carried it but ran from scratch, "
                         "so it was never actually measured")
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

    cand = full_candidate("transformer", total,
                          {"train.repeat_threshold": args.repeat_threshold})
    # The failure this run actually hit: find_checkpoint missed a nested mount,
    # returned None, and None means "first session", so the kernel restarted
    # from COCO and reported a healthy curve from epoch 1. A fine-tune that
    # silently does that is worse than useless -- two epochs from scratch --
    # and it would spend the last of the allowance doing it. Now it dies before
    # training instead.
    cand["require_resume"] = True
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
    print(f"  pushed {slug} (continues {src}, require_resume=True)")
    print(f"  {ex.wait()}")
    # Read the resume back out of the session's own report rather than trusting
    # it. require_resume guarantees a checkpoint was found; this confirms the
    # trainer actually started where it was supposed to.
    try:
        started = ex.fetch().get("last_epoch")
        print(f"  session reports last_epoch={started} (expected ~{total}); "
              f"a value near {args.epochs} would mean it restarted from scratch")
    except Exception as exc:                        # noqa: BLE001
        print(f"  could not read back the session's report: {exc}")

    r = predict_and_submit(
        source=slug, cand=cand, slug="xishengfeng/hod26-predict-s4",
        message=f"transformer session 4: all 3000 frames, epoch {total}",
        out_dir=REPO / "runs" / "predict_transformer_s4")
    print(f"\nfine-tune leaderboard: {r.get('score')}")
    print(f"Keep it only if it clears session {args.from_session} by more than ~0.01: the final "
          "standing is decided on a private split, and a smaller margin on the "
          "public one is not a difference.")


if __name__ == "__main__":
    main()
