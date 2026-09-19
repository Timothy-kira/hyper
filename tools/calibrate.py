#!/usr/bin/env python3
"""Submit a checkpoint we already have, to learn what our own scores mean.

Every design decision so far has been made against ultralytics' validator on a
held-out 20% of the training frames. The competition scores with pycocotools on
1000 frames we have never seen. The two have never been compared -- the gap
between them is unknown in both size and sign -- so a local 0.4556 translates
to a leaderboard number only under an assumption nobody has tested.

Reusing the Phase A checkpoint turns that assumption into a measurement for a
few GPU-minutes: no training, just prediction from weights that already exist.
The score itself will be poor -- that model saw 300 images for 10 epochs -- but
a poor score at a *known* local value is exactly what calibrates the mapping.
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

COMPETITION = "hyperspectral-object-detection-challenge-2026"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="xishengfeng/hod26-phasea-arch2",
                    help="kernel whose output holds the checkpoint")
    ap.add_argument("--weights", default="rtdetr-srf8_best.pt")
    ap.add_argument("--local-score", type=float, default=0.4556,
                    help="what our own validator gave this checkpoint")
    ap.add_argument("--slug", default="xishengfeng/hod26-calibrate")
    ap.add_argument("--no-submit", action="store_true")
    args = ap.parse_args()

    # The same candidate the checkpoint was trained under: the channel spec has
    # to match or the frames it is shown will not be the frames it learned on.
    cand = arm(**{"train.model": "rtdetr-l"})
    ex = KaggleRoundExecutor(args.slug, timeout_hours=2.0,
                             out_dir=REPO / "runs" / "calibrate",
                             kernel_sources=[args.source])
    ex.push({"round": "calibrate", "candidates": [],
             "submit": {"candidate": cand, "weights_from": args.weights}})
    print(f"pushed {ex.slug} (reads {args.weights} from {args.source})")
    print(f"  {ex.wait()}")
    payload = ex.fetch()
    print(f"  {payload.get('rows')} rows from {payload.get('weights')}")

    sub = ex.out_dir / "submission.csv"
    if not sub.exists():
        raise SystemExit(f"no submission.csv in {ex.out_dir}")
    from tools.submit import check
    stats = check(sub)
    print(f"  validator: {stats}")
    if args.no_submit:
        return

    import subprocess
    msg = (f"calibration: Phase A checkpoint (rtdetr-l, band_stack + SRF adapter, "
           f"300 img / 10 ep), local val mAP {args.local_score}")
    r = subprocess.run(["kaggle", "competitions", "submit", "-c", COMPETITION,
                        "-f", str(sub), "-m", msg],
                       capture_output=True, text=True)
    print(r.stdout or r.stderr)
    print(f"\nlocal {args.local_score} -> leaderboard: read it back with\n"
          f"  kaggle competitions submissions {COMPETITION}")


if __name__ == "__main__":
    main()
