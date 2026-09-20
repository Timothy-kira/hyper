"""What a two-card run reports about itself.

DDP moves the validator, the trainer's epoch counter and the callback state
into a subprocess the parent never sees, so ultralytics hands back the best
checkpoint's *stored* train_metrics instead of a metrics object. Both
substitutions are silent and both produce a plausible number, which is what
makes them expensive: a resumed run that never beats the checkpoint it
inherited reports the previous session's score as its own, and the parent's
untouched epoch counter calls epoch 39 epoch 1.

The fixture below is the real metrics file from a two-card run of this kernel
(epochs 1-19 from one session, 20-21 from a second that had folded the
validation frames into training, 22 from the run under test). It reproduces
the exact shape that reported 0.7271 for a run whose own re-validation said
0.6985.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))

from test_datapath import load_kernel  # noqa: E402


# epoch, final_eval, mAP50-95 -- trimmed from the run, keeping both handovers
FIXTURE = [
    (19, False, 0.68731),
    (20, True, 0.687061),      # an earlier session signing off
    (20, False, 0.72578),      # the contaminated session's epochs
    (21, False, 0.72706),
    (22, True, 0.726977),      # ... and its sign-off: the number best.pt stores
    (22, False, 0.69177),      # this run's only epoch
    (23, True, 0.698469),      # this run re-validating best.pt on the real split
]


def _write(work: Path) -> None:
    lines = []
    for epoch, final, m in FIXTURE:
        lines.append(json.dumps({
            "epoch": epoch, "total_epochs": 40, "final_eval": final,
            "metrics": {"metrics/mAP50-95(B)": m, "metrics/mAP50(B)": 0.95},
            "per_class": {f"c{i}": 0.5 for i in range(18)},
        }))
    (work / "final_metrics.jsonl").write_text("\n".join(lines) + "\n")


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    work, data = tmp / "working", tmp / "input"
    work.mkdir()
    data.mkdir()
    m = load_kernel(work, data)
    _write(work)

    fails = []

    def expect(label: str, got, want) -> None:
        ok = got == want
        print(f"  {'ok  ' if ok else 'MISS'}  {label}")
        if not ok:
            fails.append(f"{label}: got {got!r}, wanted {want!r}")

    # What ultralytics actually returns under DDP: best.pt's stored metrics,
    # which here belong to the session before this one.
    stale = {"metrics/mAP50-95(B)": 0.726977, "metrics/mAP50(B)": 0.951}
    scores = m.scores_from_results(stale, "final")
    expect("holdout is this run's, not the inherited checkpoint's",
           round(scores["mAP"], 6), 0.698469)
    expect("per-class survives the dict handover", len(scores["per_class"]), 18)
    expect("last epoch is the run's, not the parent trainer's",
           m.metrics_tail("final").get("epoch"), 22)

    # A single-GPU run still reads the validator it has.
    class Box:
        map, map50, ap_class_index, maps = 0.61, 0.9, None, None

    class Results:
        box = Box()

    expect("single-GPU path untouched",
           round(m.scores_from_results(Results(), "final")["mAP"], 6), 0.61)

    for f in fails:
        print(f"\nFAIL {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
