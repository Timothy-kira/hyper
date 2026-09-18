#!/usr/bin/env python3
"""Fit one full-fidelity model per track and produce a submission for each.

Built on the trainable spectral adapter: a 1x1 band mixer initialised to the
offline discriminant in front of a convolution that stays exactly as
pretrained. All three spectral/spatial augmentations are on, expanding the
training set threefold.

The two runs hold out the same 20% the search used rather than fitting on
everything. Training on all of it would score each run against data it had
seen, which is the one number that cannot be compared between tracks -- and
comparing them is the entire point of running both.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dream_rsi.budget import read_quota  # noqa: E402
from dream_rsi.candidate import normalize, seed_candidate  # noqa: E402
from dream_rsi.executor import KaggleRoundExecutor  # noqa: E402
from dream_rsi.tree import load_pool  # noqa: E402

TRACK_MODEL = {"transformer": "rtdetr-l", "yolo26": "yolo26m"}
AUGMENT = {"sg_window": 7, "sg_polyorder": 2, "smote_alpha": 0.3,
           "cutmix_prob": 0.4, "cutmix_blocks": 24, "copies": 2}


def best_channel_mode(ablation: Path) -> str:
    """The channel construction that won the controlled ablation."""
    if not ablation.exists():
        return "lda3"
    rows = json.loads(ablation.read_text()).get("results", [])
    scored = [(r["score"], r["candidate"]["channels"]["mode"])
              for r in rows if r.get("score") is not None]
    if not scored:
        return "lda3"
    return max(scored)[1]


def best_train_params(state_dir: Path) -> dict:
    """Training parameters from the best attempt this track measured."""
    best = None
    for tree in load_pool(state_dir):
        n = tree.best()
        if n and (best is None or n.score > best.score):
            best = n
    return dict(best.candidate["train"]) if best else {}


def full_candidate(track: str, mode: str, state_dir: Path, epochs: int,
                   imgsz: int) -> dict:
    cand = seed_candidate(channels__mode=mode)
    learned = best_train_params(state_dir)
    for key in ("lr0", "mosaic", "scale", "fliplr", "hsv_s", "hsv_v"):
        if key in learned:
            cand["train"][key] = learned[key]
    cand["train"].update(model=TRACK_MODEL[track], epochs=epochs, imgsz=imgsz,
                         batch=8, spectral_stem="adapter")
    cand["augment"].update(AUGMENT)
    cand["fidelity"] = "full"
    return normalize(cand)


def wait_for_quota(need: float, poll: int = 600, log=print) -> float:
    """Block until the allowance can cover the run."""
    while True:
        read = read_quota()
        if read is None:
            log("cannot read the quota; proceeding without the check")
            return float("inf")
        remaining, refresh = read
        if remaining >= need:
            return remaining
        wait = f", refresh at {refresh.isoformat()}" if refresh else ""
        log(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {remaining:.2f} GPU-h "
            f"available, need {need:.1f}{wait}")
        time.sleep(poll)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="+", default=["transformer", "yolo26"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--need-hours", type=float, default=16.0)
    ap.add_argument("--ablation", type=Path,
                    default=REPO / "runs" / "ablation" / "results.json")
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    mode = best_channel_mode(args.ablation)
    print(f"channel mode from the ablation: {mode}")

    if not args.no_wait:
        have = wait_for_quota(args.need_hours)
        print(f"proceeding with {have:.2f} GPU-h available")

    results = {}
    for track in args.tracks:
        state = REPO / "runs" / f"rsi_{track}"
        cand = full_candidate(track, mode, state, args.epochs, args.imgsz)
        print(f"\n=== {track}: {cand['train']['model']} / {mode} / "
              f"{cand['train']['epochs']}ep / augment x{1 + cand['augment']['copies']} ===")

        ex = KaggleRoundExecutor(f"xishengfeng/hod26-final-{track}", timeout_hours=11.0,
                                 out_dir=REPO / "runs" / f"final_{track}")
        # Hold out the search's own validation split so the two tracks are
        # comparable; fitting on everything would score each on data it saw.
        ex.push({"round": f"final-{track}", "candidates": [],
                 "submit": {"candidate": cand, "use_all_train": False}})
        state_str = ex.wait()
        print(f"  kernel finished: {state_str}")
        try:
            payload = ex.fetch()
        except Exception as e:                       # noqa: BLE001
            print(f"  could not fetch results: {e}")
            continue
        sub = ex.out_dir / "submission.csv"
        results[track] = {"holdout": payload.get("holdout"), "rows": payload.get("rows"),
                          "submission": str(sub) if sub.exists() else None,
                          "candidate": cand}
        print(f"  holdout mAP={payload.get('holdout')} rows={payload.get('rows')}")

    out = REPO / "runs" / "final_comparison.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    ranked = sorted(((v.get("holdout") or -1, k) for k, v in results.items()), reverse=True)
    for score, track in ranked:
        print(f"  {track:12s} holdout mAP {score}")
    if ranked:
        print(f"\nbetter track: {ranked[0][1]}")


if __name__ == "__main__":
    main()
