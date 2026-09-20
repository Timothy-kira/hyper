"""The preflight must catch, in its first minute, what used to cost a session.

Kaggle's allowance does not come back, and near a deadline a wasted session can
be the whole remaining budget. The failures that cost money are never the ones
that raise at startup -- they are the ones that look healthy and only differ
later, or never surface at all. find_checkpoint() searching one level under
/kaggle/input is the precedent: it returned None, None means "first session",
and three runs restarted from COCO while reporting a perfectly normal curve
from epoch 1.

So each case below is a *silent* failure mode, and the test asserts on the
specific diagnostic rather than on the fact that something went wrong.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _build(tmp: Path) -> str:
    """Generate the kernel the way a real push does, then neuter its pip line."""
    import json
    sys.path.insert(0, str(REPO))
    from tools.final_runs import full_candidate

    cand = full_candidate("transformer", 38)
    cand["require_resume"] = True
    cfg = tmp / "round.json"
    cfg.write_text(json.dumps({"round": "pf", "candidates": [], "submit": {
        "candidate": cand, "use_all_train": False, "predict": False,
        "session_hours": 11.0}}))
    out = tmp / "build"
    subprocess.run([sys.executable, str(REPO / "tools" / "build_kernel.py"),
                    "--round-config", str(cfg), "--out-dir", str(out),
                    "--slug", "x/pf"], check=True, capture_output=True)
    src = (out / "hod26_round.py").read_text()
    assert "ultralytics==8.4.155" in src, \
        "the generated kernel must pin ultralytics; an unpinned install lets a " \
        "release made between build and run change behaviour mid-session"
    return src.replace(
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n"
        "                'ultralytics==8.4.155', 'pycocotools'], check=False)", "pass")


def _load(src: str, inp: Path, work: Path):
    s = src.replace('Path("/kaggle/input")', f'Path("{inp}")').replace(
        'Path("/kaggle/working")', f'Path("{work}")')
    m = types.ModuleType("pf")
    sys.modules["pf"] = m
    exec(compile(s, "kernel", "exec"), m.__dict__)
    lines: list[str] = []
    m.log = lines.append            # capture the diagnostics, not just the raise
    return m, lines


def _run(m, lines, cfg) -> str:
    lines.clear()
    try:
        m.preflight(cfg)
    except Exception:               # noqa: BLE001
        pass
    return "\n".join(lines)


def _dataset(inp: Path, n_train: int, n_test: int) -> None:
    for sub in ("train/annotations", "train/images", "test/images"):
        (inp / "planar" / sub).mkdir(parents=True, exist_ok=True)
    for i in range(n_train):
        (inp / "planar" / "train" / "annotations" / f"{i}.xml").write_text("")
        (inp / "planar" / "train" / "images" / f"{i}.png").write_bytes(b"")
    for i in range(n_test):
        (inp / "planar" / "test" / "images" / f"{i}.png").write_bytes(b"")


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    src = _build(tmp)
    inp, work = tmp / "input", tmp / "working"
    inp.mkdir()
    work.mkdir()
    cfg = {"submit": {"candidate": {"train": {"model": "rtdetr-l"},
                                    "require_resume": True}}}
    fails = []

    def expect(label: str, out: str, needle: str) -> None:
        ok = needle in out
        print(f"  {'ok  ' if ok else 'MISS'}  {label}")
        if not ok:
            fails.append(f"{label}: no diagnostic containing {needle!r}\n{out}")

    m, lines = _load(src, inp, work)
    out = _run(m, lines, cfg)
    expect("private dataset never mounted", out, "collaborator")

    # A truncated upload is the dangerous shape: the dataset is there, training
    # runs, and the only symptom is a slightly worse score nobody can explain.
    _dataset(inp, 2900, 1000)
    m, lines = _load(src, inp, work)
    out = _run(m, lines, cfg)
    expect("dataset present but short (2900 of 3000)", out, "got 2900/2900/1000")
    expect("require_resume with no checkpoint", out, "require_resume is set")

    import ultralytics
    real = ultralytics.__version__
    try:
        ultralytics.__version__ = "8.9.0"
        out = _run(m, lines, cfg)
        expect("ultralytics drifted off the pin", out, "8.9.0")
    finally:
        ultralytics.__version__ = real

    # The layout that actually broke three sessions: the checkpoint is real but
    # sits under input/notebooks/<user>/<slug>/, three levels down.
    _dataset(inp, 3000, 1000)
    ck = inp / "notebooks" / "someone" / "hod26-final-transformer-s1"
    ck.mkdir(parents=True)
    (ck / "final_last.pt").write_bytes(b"x")
    m, lines = _load(src, inp, work)
    out = _run(m, lines, cfg)
    expect("checkpoint nested three levels deep is found", out,
           "resume from")
    expect("full dataset accepted", out, "3000 train / 3000 xml / 1000 test")
    if "PREFLIGHT FAILED: require_resume" in out or "dataset not found" in out:
        fails.append(f"a healthy setup was rejected:\n{out}")

    print("\n".join(fails) if fails else "\npreflight catches every silent failure")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
