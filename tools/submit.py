#!/usr/bin/env python3
"""Fit the chosen candidate at full fidelity on Kaggle and submit the result.

Reads the best candidate the Dream-RSI loop discovered, runs the submission
kernel, pulls submission.csv back, sanity-checks it against the declared schema,
and (unless --dry-run) submits it to the competition.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dream_rsi.executor import KaggleRoundExecutor  # noqa: E402
from dream_rsi.tree import load_pool  # noqa: E402
from src.hod26.submit import COLUMNS  # noqa: E402

COMPETITION = "hyperspectral-object-detection-challenge-2026"
N_TEST_IMAGES = 1000


def best_candidate(state_dir: Path) -> tuple[dict, float]:
    """The highest-scoring attempt across every recorded tree."""
    best = None
    for tree in load_pool(state_dir):
        n = tree.best()
        if n and (best is None or n.score > best.score):
            best = n
    if best is None:
        raise SystemExit(f"no scored attempt in {state_dir}; run the loop first")
    return best.candidate, best.score


def check(path: Path) -> dict:
    """Validate the submission before it is spent against the daily limit."""
    with path.open() as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise SystemExit("submission is empty")
    if rows[0] != COLUMNS:
        raise SystemExit(f"header {rows[0]} != required {COLUMNS}")

    body, images, problems = rows[1:], set(), []
    for i, r in enumerate(body):
        if int(r[0]) != i:
            problems.append(f"row {i}: id is {r[0]}, must be a 0-based counter")
            break
        cls = int(r[2])
        x1, y1, x2, y2 = (int(r[j]) for j in (4, 5, 6, 7))
        if not 0 <= cls <= 17:
            problems.append(f"row {i}: class_id {cls} outside 0..17")
        if x2 <= x1 or y2 <= y1:
            problems.append(f"row {i}: zero-area box")
        images.add(int(r[1]))
    if problems:
        raise SystemExit("submission rejected locally:\n  " + "\n  ".join(problems[:5]))
    return {"rows": len(body), "images": len(images)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "runs" / "rsi")
    ap.add_argument("--slug", default="xishengfeng/hod26-final")
    ap.add_argument("--message", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="build and validate, but do not spend a submission")
    args = ap.parse_args()

    cand, score = best_candidate(args.state_dir)
    print(f"best discovered candidate (proxy mAP {score:.4f}):")
    print(json.dumps(cand, indent=2))

    ex = KaggleRoundExecutor(args.slug, timeout_hours=11.0,
                             out_dir=args.state_dir / "final_output")
    cfg = {"round": "final", "candidates": [],
           "submit": {"candidate": {**cand, "fidelity": "full"}, "use_all_train": True}}
    ex.push(cfg)
    state = ex.wait()
    print(f"kernel finished: {state}")
    ex.fetch()

    sub = ex.out_dir / "submission.csv"
    if not sub.exists():
        raise SystemExit(f"kernel produced no submission.csv (state={state})")
    stats = check(sub)
    print(f"validated: {stats['rows']} rows over {stats['images']} images")
    if stats["images"] < N_TEST_IMAGES:
        print(f"  note: {N_TEST_IMAGES - stats['images']} test images have no "
              f"prediction at all; mAP only loses recall there, it is not an error")

    if args.dry_run:
        print(f"dry run - not submitting. File: {sub}")
        return
    msg = args.message or f"Dream-RSI discovered candidate, proxy mAP {score:.4f}"
    subprocess.run(["kaggle", "competitions", "submit", "-c", COMPETITION,
                    "-f", str(sub), "-m", msg], check=True)
    print("submitted.")


if __name__ == "__main__":
    main()
