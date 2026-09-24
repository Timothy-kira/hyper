#!/usr/bin/env python3
"""Predict and submit from a finished training kernel's checkpoint.

Separate from the training session on purpose, and here for a second reason
as well. best.pt is only rewritten when an epoch beats the fitness recorded
in the checkpoint the run resumed from -- and when that figure was measured
on a validation split the previous session had trained on, it sits above
anything an honest run prints, so best.pt keeps the weights it arrived with
and the session predicts from the checkpoint it started at. Pointing this at
final_last.pt takes the weights the run actually produced.

It also scores the held-out 600 with pycocotools, the way the leaderboard
scores, which ultralytics' own validation reads about 0.05 above -- so it
says what a submission is worth before it is spent against the daily three.

    python3 tools/predict_from_run.py --source qwyi123/hod26-team \
        --weights final_last.pt --message "epoch 40, two cards" --no-submit
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

COMPETITION = "hyperspectral-object-detection-challenge-2026"


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def build(out: Path, source: str, slug: str, weights: str, epochs: int) -> None:
    from tools.final_runs import full_candidate
    out.mkdir(parents=True, exist_ok=True)
    cfg = out / "round.json"
    cfg.write_text(json.dumps({"round": "predict", "candidates": [], "submit": {
        "candidate": full_candidate("transformer", epochs),
        "weights_from": weights,
        "score_val": True,       # the leaderboard's own ruler, on the held-out 600
        "predict_test": True,
    }}, indent=2))
    # One card: inference is a few GPU-minutes and DDP buys nothing here.
    r = run([sys.executable, str(REPO / "tools" / "build_kernel.py"),
             "--round-config", str(cfg), "--out-dir", str(out), "--slug", slug,
             "--kernel-source", source, "--machine-shape", "NvidiaTeslaT4"])
    if r.returncode:
        raise SystemExit(f"build failed:\n{r.stdout}{r.stderr}")
    cfg.unlink()


def wait(slug: str, timeout_min: int = 60, log=print) -> str:
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        s = run(["kaggle", "kernels", "status", slug]).stdout
        for state in ("COMPLETE", "ERROR", "CANCEL"):
            if state in s:
                return state
        time.sleep(30)
    return "TIMEOUT"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="qwyi123/hod26-team",
                    help="the training kernel whose output holds the checkpoint")
    ap.add_argument("--weights", default="final_last.pt",
                    help="final_last.pt is the run's own weights; final_best.pt "
                         "may still be the checkpoint it resumed from")
    ap.add_argument("--slug", default="qwyi123/hod26-predict")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/tmp/hod26-predict"))
    ap.add_argument("--message", default="")
    ap.add_argument("--no-submit", action="store_true",
                    help="fetch and validate, but leave the daily three alone")
    args = ap.parse_args()

    build(args.out_dir / "kernel", args.source, args.slug,
          args.weights, args.epochs)
    push = run(["kaggle", "kernels", "push", "-p", str(args.out_dir / "kernel")])
    out = push.stdout + push.stderr
    if "successfully pushed" not in out.lower():
        raise SystemExit(f"push refused: {out.strip()[:300]}")
    print(f"pushed {args.slug}, reading {args.weights} from {args.source}")

    state = wait(args.slug)
    print(f"finished: {state}")
    if state != "COMPLETE":
        raise SystemExit(f"{args.slug} ended {state}; fetch its log before retrying")

    got = args.out_dir / "output"
    got.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "kernels", "output", args.slug, "-p", str(got)])

    results = got / "results.json"
    if results.exists():
        val = (json.loads(results.read_text()).get("val") or {})
        if val:
            print(f"held-out 600, scored as the leaderboard scores: "
                  f"mAP {val['mAP']:.4f}  mAP50 {val['mAP50']:.4f}")
            print("  (ultralytics' own validation reads about 0.05 above this)")

    sub = got / "submission.csv"
    if not sub.exists():
        raise SystemExit(f"no submission.csv in {got}")
    from tools.submit import check
    stats = check(sub)
    print(f"submission validated: {stats['rows']} rows over {stats['images']} frames")

    if args.no_submit:
        print("not submitting (--no-submit)")
        return 0
    msg = args.message or f"{args.weights} from {args.source}"
    r = run(["kaggle", "competitions", "submit", "-c", COMPETITION,
             "-f", str(sub), "-m", msg])
    said = (r.stdout or "") + (r.stderr or "")
    if "Successfully submitted" not in said:
        print(f"submission refused: {said.strip()[:300]}")
        return 1
    print(f"submitted: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
