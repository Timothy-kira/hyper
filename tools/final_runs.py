#!/usr/bin/env python3
"""Fit one full-fidelity model per track, chunked across Kaggle sessions.

The design is written down here rather than read back from the search. Every
number the search measured came from 300 images and 10 epochs -- about 1/100th
of the compute of the public 0.66 baseline -- and re-running the bottleneck
analysis on a healthy run against an undertrained one reordered the bottlenecks
outright. A proxy can say which choices are structural; it cannot pick the
final configuration, and carrying proxy-fitted hyperparameters into a
full-fidelity run is how a search talks itself into its own noise.

The run does not fit comfortably in one Kaggle session. Phase A measured
rtdetr-l at 130 s per epoch over 300 images at 1024, which puts 4800 training
images (2400 of the 80% split, doubled by augmentation) at about 35 minutes an
epoch -- around 10.4 GPU-hours for 18 epochs, plus rendering. A session killed
by the 12-hour limit loses its /kaggle/working entirely, so the run goes as N
sessions that each end on purpose, every one of them listing the last as a
kernel source and picking up its last.pt.
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

TRACK_MODEL = {"transformer": "rtdetr-l", "yolo26": "yolo26m"}

# The design, and why each part of it is here.
DESIGN = {
    # 16 bands in, so the front end exists at all. A 3-channel mode has already
    # collapsed the spectrum with a fixed rule and leaves nothing to adapt.
    "channels.mode": "band_stack",
    # Fixed non-negative SRF bank, then a trainable 8 -> 3 mix in front of an
    # untouched pretrained block. Averaging is what makes the frame readable to
    # a backbone trained on natural images; the trainable stage is what gets
    # the material discrimination back out of the averaged channels.
    "train.spectral_stem": "adapter",
    "train.srf_k": 8,
    "train.srf_width": 2.0,
    # Localization is the largest recoverable loss (mAP50 0.626 against
    # mAP50-95 0.429: objects found, boxes loose), objects are 15-45 px in a
    # 493x241 cube, and the feature stride is fixed. It is also the one thing
    # the public 0.66 notebooks do that the search never tried.
    "train.imgsz": 1024,
    # A T4 has 16 GB and RT-DETR trains without AMP; at 1024 that is what fits.
    # nbs is 64 either way, so the optimizer still steps on an effective 64.
    "train.batch": 4,
    # multi_scale is off, and this was measured rather than chosen: ultralytics
    # samples roughly 0.5x to 1.5x of imgsz, which at 1024 reaches 1984 px. The
    # first Phase A push OOM'd on a T4 at that size, ultralytics halved the
    # batch twice trying to recover, and BatchNorm then failed on a batch of
    # one -- both arms lost. The scale-drift argument for it is real but it
    # cannot be had at this resolution on this hardware.
    "train.multi_scale": False,
    # Large early gradients from a fresh head and mixer reach every pretrained
    # layer behind them. A longer ramp is the lever that does not also freeze.
    "train.warmup_epochs": 5.0,
    "train.cos_lr": True,
    "train.coco_prior": True,
    "fidelity": "full",
}

# All three augmentations on, doubling the training set. Three copies does not
# fit at 1024 within the remaining budget.
AUGMENT = {"sg_window": 7, "sg_polyorder": 2, "smote_alpha": 0.3,
           "cutmix_prob": 0.4, "cutmix_blocks": 24, "copies": 1}


def full_candidate(track: str, epochs: int) -> dict:
    cand = seed_candidate(**DESIGN)
    cand["train"].update(model=TRACK_MODEL[track], epochs=epochs)
    cand["augment"].update(AUGMENT)
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


def session_plan(total_epochs: int, chunks: int) -> list[int]:
    """Cumulative epoch target for each session.

    Each session trains *to* its target and stops there, so the target is what
    goes in the config; ultralytics reads the epochs already done out of the
    checkpoint.
    """
    return [round(total_epochs * (i + 1) / chunks) for i in range(chunks)]


def run_track(track: str, epochs: int, chunks: int, use_all_train: bool,
              timeout_hours: float) -> dict:
    cand_final = full_candidate(track, epochs)
    print(f"=== {track}: {cand_final['train']['model']} / "
          f"{cand_final['channels']['mode']} + srf{cand_final['train']['srf_k']} "
          f"adapter / {epochs}ep @ {cand_final['train']['imgsz']} / "
          f"augment x{1 + cand_final['augment']['copies']} in {chunks} session(s) ===")

    previous, payload = None, {}
    for i, target in enumerate(session_plan(epochs, chunks)):
        slug = f"xishengfeng/hod26-final-{track}-s{i + 1}"
        cand = full_candidate(track, target)
        ex = KaggleRoundExecutor(slug, timeout_hours=timeout_hours,
                                 out_dir=REPO / "runs" / f"final_{track}_s{i + 1}",
                                 kernel_sources=[previous] if previous else [])
        # Only the last session predicts and writes a submission; the earlier
        # ones exist to move the checkpoint forward.
        ex.push({"round": f"final-{track}-s{i + 1}", "candidates": [],
                 "submit": {"candidate": cand, "use_all_train": use_all_train,
                            "predict": i == chunks - 1}})
        print(f"  session {i + 1}/{chunks} -> {target} epochs, pushed {slug}"
              + (f" (continues {previous})" if previous else ""))
        state = ex.wait()
        print(f"  session {i + 1} finished: {state}")
        try:
            payload = ex.fetch()
        except Exception as e:                       # noqa: BLE001
            print(f"  could not fetch results: {e}")
            return {"error": str(e), "session": i + 1, "candidate": cand_final}
        print(f"  holdout mAP={payload.get('holdout')} rows={payload.get('rows')}")
        previous = slug

    sub = REPO / "runs" / f"final_{track}_s{chunks}" / "submission.csv"
    return {"holdout": payload.get("holdout"), "rows": payload.get("rows"),
            "submission": str(sub) if sub.exists() else None,
            "candidate": cand_final}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="+", default=["transformer"])
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--chunks", type=int, default=2,
                    help="Kaggle sessions to split the run across")
    ap.add_argument("--need-hours", type=float, default=14.0)
    ap.add_argument("--timeout-hours", type=float, default=11.5)
    ap.add_argument("--use-all-train", action="store_true",
                    help="refit on every frame; the holdout score then means nothing")
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    if not args.no_wait:
        print(f"proceeding with {wait_for_quota(args.need_hours):.2f} GPU-h available")

    results = {t: run_track(t, args.epochs, args.chunks, args.use_all_train,
                            args.timeout_hours) for t in args.tracks}
    out = REPO / "runs" / "final_comparison.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    ranked = sorted(((v.get("holdout") or -1, k) for k, v in results.items()), reverse=True)
    for score, track in ranked:
        print(f"  {track:12s} holdout mAP {score}")
    if len(ranked) > 1:
        print(f"\nbetter track: {ranked[0][1]}")


if __name__ == "__main__":
    main()
