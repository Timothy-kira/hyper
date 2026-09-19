#!/usr/bin/env python3
"""Evaluate and submit one training session's checkpoint.

Standalone so a session can be submitted whether or not the orchestrator that
produced it is still alive -- which, on this harness, is not something to
assume: detached processes have been reaped twice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.final_runs import full_candidate  # noqa: E402
from tools.predict_submit import predict_and_submit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--track", default="transformer")
    ap.add_argument("--epoch", type=int, required=True, help="epochs it finished")
    ap.add_argument("--total", type=int, default=27)
    ap.add_argument("--no-submit", action="store_true")
    args = ap.parse_args()

    src = f"xishengfeng/hod26-final-{args.track}-s{args.session}"
    r = predict_and_submit(
        source=src, cand=full_candidate(args.track, args.total),
        slug=f"xishengfeng/hod26-predict-s{args.session}",
        message=f"{args.track} session {args.session}: rtdetr-l + SRF adapter, "
                f"epoch {args.epoch}/{args.total}",
        out_dir=REPO / "runs" / f"predict_{args.track}_s{args.session}",
        submit=not args.no_submit)
    print(f"\nsession {args.session} @ epoch {args.epoch}: leaderboard {r.get('score')}")


if __name__ == "__main__":
    main()
