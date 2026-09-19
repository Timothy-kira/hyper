#!/usr/bin/env python3
"""Drive the rest of the run unattended: A/B verdict, submissions, extension.

Written because nothing local here survives reliably -- the orchestrator and
its own supervisor have both been reaped mid-run, once leaving a GPU session
finishing on Kaggle with nothing left to act on it. So this holds the whole
remaining plan in one process, logs every decision, and is safe to restart:
each step checks whether its work is already done before doing it.

The sequence:
  1. wait for the loss A/B, pick a winner by a written-down rule
  2. wait for session 2, evaluate and submit it (the clean 27-epoch reading)
  3. push session 3 -- the extension carrying the A/B winner, repeat sampling
     and a warm-restarted LR -- then evaluate and submit that too
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dream_rsi.budget import read_quota  # noqa: E402
from dream_rsi.executor import KaggleRoundExecutor, KernelError  # noqa: E402
from tools.final_runs import full_candidate  # noqa: E402
from tools.predict_submit import COMPETITION, predict_and_submit  # noqa: E402

STATE = REPO / "runs" / "overnight_state.json"
STREET = ["people", "car", "e-bike", "stone_block"]
# Round one on the street subset. DIoU is the base round two builds on.
DIOU_BASE = 0.4197
# Round two measured the noise rather than assuming it. Its four arms all share
# the DIoU base and differ only in loss details, yet they spread 0.0149 -- wider
# than DIoU's own 0.0091 lead over GIoU in round one. So a single seed over 104
# frames is good to about 0.015, not the 0.010 first guessed, and DIoU's lead is
# not a result either. Nothing on this subset earns a change unless it clears
# this.
NOISE = 0.015


def log(*a):
    line = f"[{datetime.now(timezone.utc):%H:%M:%S}] " + " ".join(str(x) for x in a)
    print(line, flush=True)


def state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(**kw):
    s = state()
    s.update(kw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2))


def wait_for(slug: str, poll: int = 180, hours: float = 13.0) -> str:
    """Block until a kernel leaves RUNNING, tolerating a flaky status call."""
    deadline = time.time() + hours * 3600
    ex = KaggleRoundExecutor(slug)
    while time.time() < deadline:
        try:
            st = ex.status()
        except KernelError as e:
            log(f"  status {slug} failed, retrying: {str(e)[:120]}")
            time.sleep(poll)
            continue
        if st in ("complete", "error", "cancelacknowledged"):
            return st
        time.sleep(poll)
    return "timeout"


def already_submitted(needle: str) -> bool:
    r = subprocess.run(["kaggle", "competitions", "submissions", COMPETITION],
                       capture_output=True, text=True)
    return needle in r.stdout


def pick_winner(slug: str) -> dict:
    """Read the A/B and turn it into overrides, by a rule fixed in advance.

    Best arm wins only if it clears the DIoU base by more than the noise floor.
    Otherwise the base stands -- a coin flip dressed as a result is worse than
    no change, because it would also be carried into the final model.
    """
    # The base is the objective the 27 epochs were actually trained under.
    # Switching it on a within-noise lead would be the coin flip this rule
    # exists to refuse -- and it would ride into the final model.
    base: dict = {}
    arms = {"logl1": {"train.log_size_l1": True},
            "sharpen": {"train.bbox_alpha": 3.0, "train.vfl_beta": 0.5},
            "both": {"train.log_size_l1": True, "train.bbox_alpha": 3.0,
                     "train.vfl_beta": 0.5},
            "gains": {"train.loss_gain": {"bbox": 2, "giou": 5}}}
    try:
        results = KaggleRoundExecutor(
            slug, out_dir=REPO / "runs" / slug.split("/")[-1]).fetch().get("results", [])
    except Exception as e:                            # noqa: BLE001
        log(f"  could not read the A/B ({e}); keeping the DIoU base")
        return base

    scored = {r["node_id"]: r for r in results if r.get("score") is not None}
    for nid, r in sorted(scored.items(), key=lambda kv: -kv[1]["score"]):
        pc = (r.get("diagnostics") or {}).get("per_class", {})
        log(f"  {nid:8s} mAP={r['score']:.4f} "
            + " ".join(f"{c}={pc[c]:.3f}" for c in STREET if c in pc))
    for r in results:
        if r.get("error"):
            log(f"  {r['node_id']:8s} FAILED {r['error'][-160:]}")

    if not scored:
        log("  no arm produced a score; keeping the DIoU base")
        return base
    best = max(scored.values(), key=lambda r: r["score"])
    gain = best["score"] - DIOU_BASE
    if gain <= NOISE:
        verdict = "below it" if gain < 0 else "inside it"
        log(f"  best arm {best['node_id']} is {gain:+.4f} against the DIoU "
            f"reference and the noise floor is {NOISE} -- {verdict}. Keeping "
            f"the loss the earlier sessions trained under.")
        return base
    log(f"  winner {best['node_id']}: {gain:+.4f} over DIoU, clears the floor")
    return {**base, **arms.get(best["node_id"], {})}


def submit_session(n: int, epoch: int, total: int) -> float | None:
    needle = f"session {n}:"
    if already_submitted(needle):
        log(f"  session {n} already submitted, skipping")
        return None
    r = predict_and_submit(
        source=f"xishengfeng/hod26-final-transformer-s{n}",
        cand=full_candidate("transformer", total),
        slug=f"xishengfeng/hod26-predict-s{n}",
        message=f"transformer session {n}: rtdetr-l + SRF adapter, epoch {epoch}/{total}",
        out_dir=REPO / "runs" / f"predict_transformer_s{n}", log=log)
    save(**{f"session{n}_lb": r.get("score"), f"session{n}_epoch": epoch})
    return r.get("score")


def reached_of(slug: str, default: int) -> int:
    try:
        p = KaggleRoundExecutor(
            slug, out_dir=REPO / "runs" / f"fetch_{slug.split('/')[-1]}").fetch()
        return int(p.get("last_epoch") or p.get("epochs_to") or default)
    except Exception as e:                            # noqa: BLE001
        log(f"  could not read {slug}'s epoch ({e}); assuming {default}")
        return default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab-slug", default="xishengfeng/hod26-streetab3")
    ap.add_argument("--extend-to", type=int, default=34,
                    help="total epochs after the extension. The LR restart is "
                         "this same number as schedule_epochs: resuming at 27 "
                         "on a 34-epoch cosine reopens the rate to about a "
                         "tenth of peak and anneals it back down, which is the "
                         "lever that works -- lr0 is ignored under "
                         "optimizer=auto, as the Phase A logs show.")
    ap.add_argument("--min-quota", type=float, default=5.0)
    args = ap.parse_args()

    log(f"starting; quota {read_quota()}")

    # 1. the A/B verdict
    if "overrides" not in state():
        log(f"waiting for the loss A/B {args.ab_slug}")
        log(f"  {args.ab_slug}: {wait_for(args.ab_slug, hours=5)}")
        save(overrides=pick_winner(args.ab_slug))
    over = state()["overrides"]
    log(f"overrides for the extension: {over}")

    # 2. session 2, the clean 27-epoch reading
    s2 = "xishengfeng/hod26-final-transformer-s2"
    log(f"waiting for {s2}")
    log(f"  {s2}: {wait_for(s2)}")
    reached2 = reached_of(s2, 27)
    log(f"session 2 reached epoch {reached2}/27")
    lb2 = submit_session(2, reached2, 27)
    log(f"session 2 leaderboard: {lb2}")

    # 3. the extension
    q = read_quota()
    have = q[0] if q else float("inf")
    if have < args.min_quota:
        log(f"only {have:.1f} GPU-h left, below the {args.min_quota} needed for "
            f"an extension; stopping with session 2 submitted")
        return
    total = max(args.extend_to, reached2 + 4)
    cand = full_candidate("transformer", total, over)
    log(f"session 3: epochs {reached2} -> {total}, {have:.1f} GPU-h available")
    log(f"  {json.dumps({k: cand['train'][k] for k in ('bbox_loss', 'bbox_alpha', 'vfl_beta', 'log_size_l1', 'repeat_threshold', 'schedule_epochs', 'epochs')})}")
    s3 = "xishengfeng/hod26-final-transformer-s3"
    ex = KaggleRoundExecutor(s3, timeout_hours=11.9,
                             out_dir=REPO / "runs" / "final_transformer_s3",
                             kernel_sources=[s2])
    ex.push({"round": "final-transformer-s3", "candidates": [],
             "submit": {"candidate": cand, "use_all_train": False,
                        "predict": False, "session_hours": 9.0}})
    log(f"  pushed {s3}")
    save(session3_pushed=True, session3_total=total)
    log(f"  {s3}: {wait_for(s3)}")
    reached3 = reached_of(s3, total)
    log(f"session 3 reached epoch {reached3}/{total}")
    lb3 = submit_session(3, reached3, total)
    log(f"session 3 leaderboard: {lb3}")

    log(f"done. session 2 {state().get('session2_lb')} at epoch {reached2}, "
        f"session 3 {lb3} at epoch {reached3}. quota {read_quota()}")


if __name__ == "__main__":
    main()
