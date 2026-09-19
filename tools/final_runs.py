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
from tools.predict_submit import predict_and_submit  # noqa: E402

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


CLOSE_MOSAIC = 5


def full_candidate(track: str, total: int) -> dict:
    """The candidate every session of the run declares.

    Each session asks for the whole run and stops itself on the clock, so the
    epoch count, the LR schedule and the mosaic-close epoch are all expressed
    in whole-run terms and none of them has to be shifted per session. Where
    the session boundary lands is then a measurement the kernel makes, not an
    estimate made here.
    """
    cand = seed_candidate(**DESIGN)
    cand["train"].update(model=TRACK_MODEL[track], epochs=total,
                         schedule_epochs=total, close_mosaic=CLOSE_MOSAIC)
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


def reached_epoch(payload: dict, default: int = 0) -> int:
    """How far the run has actually got, from a session's results.json.

    Sessions pushed before the clock guard existed do not report last_epoch;
    they ran to their target or died, so the target is the right fallback.
    """
    got = payload.get("last_epoch")
    if got is None:
        got = payload.get("epochs_to", default)
    return int(got or 0)


def run_track(track: str, total: int, use_all_train: bool, session_hours: float,
              timeout_hours: float, max_sessions: int,
              continue_from: str | None, no_submit: bool = False) -> dict:
    cand = full_candidate(track, total)
    print(f"=== {track}: {cand['train']['model']} / {cand['channels']['mode']} + "
          f"srf{cand['train']['srf_k']} adapter / {total}ep @ "
          f"{cand['train']['imgsz']} / augment x{1 + cand['augment']['copies']}, "
          f"{session_hours}h per session ===")

    previous, reached, payload, i = continue_from, 0, {}, 0
    scores: list[dict] = []
    if previous:
        ex = KaggleRoundExecutor(previous, timeout_hours=timeout_hours,
                                 out_dir=REPO / "runs" / f"final_{track}_s0")
        print(f"  waiting on {previous} (already pushed): {ex.wait()}")
        payload = ex.fetch()
        reached = reached_epoch(payload, total)
        i = int(previous.rsplit("-s", 1)[-1]) if "-s" in previous else 0
        print(f"  it reached epoch {reached}/{total}, holdout {payload.get('holdout')}")

    while reached < total and i < max_sessions:
        i += 1
        need = max(1.0, session_hours * 0.5)
        have = wait_for_quota(need)
        slug = f"xishengfeng/hod26-final-{track}-s{i}"
        ex = KaggleRoundExecutor(slug, timeout_hours=timeout_hours,
                                 out_dir=REPO / "runs" / f"final_{track}_s{i}",
                                 kernel_sources=[previous] if previous else [])
        # Training sessions never predict. Prediction is a separate kernel on
        # the other GPU slot afterwards, which keeps every minute of a
        # session's clock budget on training and makes the submission come
        # through one well-exercised path rather than two.
        ex.push({"round": f"final-{track}-s{i}", "candidates": [],
                 "submit": {"candidate": cand, "use_all_train": use_all_train,
                            "predict": False, "session_hours": session_hours}})
        print(f"  session {i}: from epoch {reached} toward {total}, {have:.1f} GPU-h "
              f"available, pushed {slug}"
              + (f" (continues {previous})" if previous else ""))
        state = ex.wait()
        print(f"  session {i} finished: {state}")
        try:
            payload = ex.fetch()
        except Exception as e:                       # noqa: BLE001
            print(f"  could not fetch results: {e}")
            return {"error": str(e), "session": i, "reached": reached,
                    "per_session": scores, "candidate": cand}
        got = reached_epoch(payload)
        print(f"  reached epoch {got}/{total}, holdout {payload.get('holdout')}, "
              f"predicted={payload.get('predicted')}")
        scores.append({"session": i, "epoch": got, "holdout": payload.get("holdout")})
        if not no_submit:
            r = predict_and_submit(
                source=slug, cand=cand,
                slug=f"xishengfeng/hod26-predict-s{i}",
                message=f"{track} session {i}: rtdetr-l + SRF adapter, epoch "
                        f"{got}/{total}, holdout {payload.get('holdout')}",
                out_dir=REPO / "runs" / f"predict_{track}_s{i}",
                log=print)
            scores[-1].update(lb=r.get("score"), submitted=r.get("submitted"))
            print(f"  session {i}: holdout {payload.get('holdout')} -> "
                  f"leaderboard {r.get('score')}")
        if got <= reached:
            # No forward progress means the next session would repeat this one.
            # Stopping here keeps the remaining quota for a deliberate retry.
            return {"error": f"session {i} ended at epoch {got}, no further than "
                             f"the {reached} it started from",
                    "reached": reached, "per_session": scores, "candidate": cand}
        previous, reached = slug, got

    return {"holdout": payload.get("holdout"), "reached": reached, "sessions": i,
            "per_session": scores, "candidate": cand}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="+", default=["transformer"])
    ap.add_argument("--epochs", type=int, default=27)
    ap.add_argument("--session-hours", type=float, default=10.5,
                    help="wall clock a session may spend before stopping itself; "
                         "Kaggle's cap is 12h and the kernel also has to render "
                         "and, on the last session, predict")
    ap.add_argument("--timeout-hours", type=float, default=11.9)
    ap.add_argument("--max-sessions", type=int, default=4)
    ap.add_argument("--continue-from", default=None,
                    help="a session already pushed; wait for it and carry on")
    ap.add_argument("--no-submit", action="store_true",
                    help="train only; skip the per-session predict and submit")
    ap.add_argument("--use-all-train", action="store_true",
                    help="refit on every frame; the holdout score then means nothing")
    args = ap.parse_args()

    results = {t: run_track(t, args.epochs, args.use_all_train, args.session_hours,
                            args.timeout_hours, args.max_sessions,
                            args.continue_from if t == args.tracks[0] else None,
                            args.no_submit)
               for t in args.tracks}
    out = REPO / "runs" / "final_comparison.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    for track, r in results.items():
        if r.get("error"):
            print(f"  {track:12s} stopped: {r['error']}")
        else:
            print(f"  {track:12s} epoch {r.get('reached')} over "
                  f"{r.get('sessions')} session(s)")
        for sc in r.get("per_session", []):
            print(f"    session {sc['session']}: epoch {sc['epoch']:>3}  "
                  f"holdout {sc.get('holdout')}  leaderboard {sc.get('lb')}")


if __name__ == "__main__":
    main()
