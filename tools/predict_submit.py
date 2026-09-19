#!/usr/bin/env python3
"""Predict the test set from an existing checkpoint and submit it.

Split out of the training loop on purpose. A submission after every session is
worth more than an estimate after the last one: it turns each stage of the run
into a real leaderboard number, it says whether more epochs are still paying
before the quota is spent on them, and it means a failure late in the run
leaves something banked rather than nothing.

It costs about eight GPU-minutes -- the calibration run predicted 1000 frames
in four and a half -- and runs on the second GPU slot while the next training
session occupies the first, so it costs no wall clock at all.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMPETITION = "hyperspectral-object-detection-challenge-2026"


def predict_and_submit(source: str, cand: dict, slug: str, message: str,
                       out_dir: Path, weights: str = "final_best.pt",
                       submit: bool = True, timeout_hours: float = 2.0,
                       log=print) -> dict:
    """Run a predict-only kernel off ``source``'s checkpoint, then submit it."""
    from dream_rsi.executor import KaggleRoundExecutor
    from tools.submit import check

    ex = KaggleRoundExecutor(slug, timeout_hours=timeout_hours, out_dir=out_dir,
                             kernel_sources=[source])
    ex.push({"round": "predict", "candidates": [],
             "submit": {"candidate": cand, "weights_from": weights}})
    log(f"  predict kernel {slug} pushed (reads {weights} from {source})")
    state = ex.wait()
    payload = ex.fetch()
    sub = out_dir / "submission.csv"
    if not sub.exists():
        return {"error": f"{slug} finished {state} with no submission.csv"}

    stats = check(sub)
    log(f"  {stats['rows']} rows over {stats['images']} frames, validator ok")
    if not submit:
        return {"rows": stats["rows"], "images": stats["images"], "submitted": False}

    r = subprocess.run(["kaggle", "competitions", "submit", "-c", COMPETITION,
                        "-f", str(sub), "-m", message],
                       capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if "Successfully submitted" not in out:
        # A refused submission (the daily limit is 3) is not a reason to stop
        # training; the checkpoint is still on disk and can be submitted later.
        log(f"  submission refused: {out.strip()[:200]}")
        return {"rows": stats["rows"], "submitted": False, "reason": out.strip()[:200]}
    log(f"  submitted: {message}")
    return {"rows": stats["rows"], "submitted": True,
            "score": poll_score(message, log=log)}


def poll_score(message: str, tries: int = 40, wait: int = 30, log=print):
    """Wait for Kaggle to finish scoring the submission we just made."""
    for _ in range(tries):
        r = subprocess.run(["kaggle", "competitions", "submissions", COMPETITION],
                           capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if message[:40] not in line:
                continue
            if "COMPLETE" in line:
                score = line.split()[-1]
                log(f"  leaderboard: {score}")
                try:
                    return float(score)
                except ValueError:
                    return None
            break
        time.sleep(wait)
    log("  scoring did not finish in time; read it back later")
    return None
