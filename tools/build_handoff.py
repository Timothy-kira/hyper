#!/usr/bin/env python3
"""Package the rest of the run so a teammate can finish it on their own account.

Kaggle charges GPU time to whoever starts the session, so a second account's
allowance is only reachable by its owner pressing Run. Nothing here works
around that; it makes pressing Run the only thing they have to do.

The run does not start over. Session 4's checkpoint sits at epoch 21 and is
shared as a small private dataset, so the eleven hours this costs buy epochs
22 to 40 rather than 1 to 19 -- half the compute for a longer finish, and one
session instead of two. Everything below exists because that resume is also
the thing that silently failed three times: find_checkpoint searched one level
under /kaggle/input, missed a nested mount, returned None, and None means
"first session". require_resume turns that into an error in the first minute.

Session 4 rather than session 2 for the two epochs, not the score: the two are
0.00029 apart on the leaderboard against a run-to-run noise floor of about
0.001, so they are the same model, but 21 epochs is 21 epochs.

The held-out 600 go back to being held out even so. Session 4 trained on all
3000 frames, which is why its own validation number is not comparable -- but
what matters here is that the submission is predicted from best.pt, selected
on the validation split. Kept contaminated, that selection runs for nineteen
more epochs and picks whichever weights best memorise frames already seen.
Session 4 saw those 600 for two epochs; nineteen without washes that out well
enough for the selection to mean something again.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.final_runs import full_candidate  # noqa: E402

OUT = REPO / "handoff"
CKPT_DATASET = "xishengfeng/hod26-ckpt-s4"
RESUME_FROM = 21          # the epoch session 4's final_last.pt stopped on
TOTAL = 40                # 19 more epochs, about eleven hours
SESSION_HOURS = 11.0


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)

    cand = full_candidate("transformer", TOTAL)
    cand["require_resume"] = True

    cfg = OUT / "round-config.json"
    cfg.write_text(json.dumps({"round": "hod26-team", "candidates": [], "submit": {
        "candidate": cand,
        "use_all_train": False,
        # True rather than "if_complete": this is likely their only session, so
        # a submission has to come out of it wherever the clock stops it. The
        # kernel reserves thirty minutes for the prediction when predict is set.
        "predict": True,
        "session_hours": SESSION_HOURS,
    }}, indent=2))

    subprocess.run(
        [sys.executable, str(REPO / "tools" / "build_kernel.py"),
         "--round-config", str(cfg), "--out-dir", str(OUT),
         "--slug", "TEAMMATE/hod26-team", "--dataset-source", CKPT_DATASET],
        check=True, capture_output=True)
    cfg.unlink()

    meta = json.loads((OUT / "kernel-metadata.json").read_text())
    script = OUT / "hod26_round.py"
    print(f"  {script.relative_to(REPO)}  {script.stat().st_size / 1e3:.0f} kB")
    print(f"  datasets: {meta['dataset_sources']}")
    print(f"  epochs {RESUME_FROM} -> {TOTAL} in one {SESSION_HOURS}h session, "
          f"require_resume={cand['require_resume']}, held-out kept")


if __name__ == "__main__":
    main()
