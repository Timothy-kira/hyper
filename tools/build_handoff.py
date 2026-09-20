#!/usr/bin/env python3
"""Generate everything a teammate needs to run the long training on their own account.

Kaggle charges GPU time to whoever starts the session, so a second account's
allowance is only reachable by that account's owner pressing Run. Nothing here
is a workaround for that -- it is the opposite, a package complete enough that
pressing Run is genuinely all they have to do.

Two sessions, because 38 epochs is about 22 GPU-hours and Kaggle caps a session
at twelve. Both declare the whole 38-epoch run: the LR cosine is shaped over
the total, the clock guard stops each session cleanly when the next epoch will
not fit, and the second resumes from the first. That resume is the part that
silently failed for three sessions here, so the second file sets
require_resume, which fails in the first minute rather than restarting from
COCO and reporting a healthy curve from epoch 1.
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
TOTAL = 38
SESSION_HOURS = 11.0


def build(name: str, cand: dict, sources: list[str], predict) -> Path:
    cfg = {"round": name, "candidates": [], "submit": {
        "candidate": cand,
        "use_all_train": False,   # keep the 600 held-out: the curve is how we
                                  # tell whether session B is worth running
        "predict": predict,
        "session_hours": SESSION_HOURS,
    }}
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    cfg_path = d / "round-config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    subprocess.run(
        [sys.executable, str(REPO / "tools" / "build_kernel.py"),
         "--round-config", str(cfg_path), "--out-dir", str(d),
         "--slug", f"TEAMMATE/{name}",
         *sum((["--kernel-source", s] for s in sources), [])],
        check=True, capture_output=True)
    cfg_path.unlink()
    return d / "hod26_round.py"


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)

    a = full_candidate("transformer", TOTAL)
    pa = build("hod26-team-a", a, [], predict=False)

    b = full_candidate("transformer", TOTAL)
    # The whole point of the second file. Without it a missing checkpoint means
    # "first session" and the run starts over, which is what cost this project
    # three sessions and thirteen GPU-hours.
    b["require_resume"] = True
    pb = build("hod26-team-b", b, ["TEAMMATE/hod26-team-a"], predict="if_complete")

    for p in (pa, pb):
        print(f"  {p.relative_to(REPO)}  {p.stat().st_size / 1e3:.0f} kB")
    print(f"\n  target {TOTAL} epochs, {SESSION_HOURS}h per session, "
          f"held-out split kept")


if __name__ == "__main__":
    main()
