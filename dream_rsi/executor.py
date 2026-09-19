"""Runs one decision round on Kaggle GPU and returns what it measured.

The orchestrator has no GPU, so every attempt is executed as a Kaggle kernel:
push the generated script, wait for the session, read results.json back. A whole
batch travels in one kernel run, which is what makes the round the natural unit
of both the Dream-RSI decision interface and the GPU budget.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD_ROOT = REPO / "kernels" / "hod26_round" / "build"


class KernelError(RuntimeError):
    pass


def _kaggle(*args, timeout: int = 900) -> str:
    p = subprocess.run(["kaggle", *args], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise KernelError(f"kaggle {' '.join(args)} failed:\n{p.stdout}\n{p.stderr}")
    return p.stdout


class KaggleRoundExecutor:
    """Push one round, wait for it, and read back per-candidate scores."""

    def __init__(self, slug: str, poll_seconds: int = 60, timeout_hours: float = 6.0,
                 out_dir: Path | None = None, kernel_sources: list[str] | None = None):
        self.slug = slug
        # A run longer than Kaggle's 12-hour session limit is split across
        # sessions. Listing the previous kernel here mounts its /kaggle/working
        # under /kaggle/input, which is where the driver looks for last.pt.
        self.kernel_sources = list(kernel_sources or [])
        self.poll_seconds = poll_seconds
        self.timeout_hours = timeout_hours
        self.out_dir = Path(out_dir or REPO / "runs" / "kernel_output")
        # One build directory per kernel. Concurrent tracks previously generated
        # into a shared directory, so whichever built last decided what both
        # pushed -- one track could ship the other's source under the other's
        # slug, silently crossing two experiments.
        self.build_dir = BUILD_ROOT / slug.replace("/", "__")

    # -- lifecycle --------------------------------------------------------
    def push(self, round_cfg: dict) -> None:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = self.build_dir / "round_config.json"
        cfg_path.write_text(json.dumps(round_cfg, indent=2))
        subprocess.run(
            ["python3", str(REPO / "tools" / "build_kernel.py"),
             "--round-config", str(cfg_path), "--slug", self.slug,
             "--out-dir", str(self.build_dir),
             *sum((["--kernel-source", k] for k in self.kernel_sources), [])],
            check=True, capture_output=True, text=True,
        )
        _kaggle("kernels", "push", "-p", str(self.build_dir))

    def status(self) -> str:
        """Current session state, or "pending" if there is no session yet.

        A kernel that has never run has no session, and the status endpoint
        404s for it -- which is "not started", not a failure. Raising on it
        killed a track on its very first push.
        """
        try:
            out = _kaggle("kernels", "status", self.slug).lower()
        except KernelError as e:
            if "404" in str(e) or "not found" in str(e).lower():
                return "pending"
            raise
        for s in ("complete", "error", "cancelrequested", "cancelacknowledged",
                  "running", "queued"):
            if s in out:
                return s
        return "unknown"

    TERMINAL = ("complete", "error", "cancelacknowledged")

    def wait(self, start_timeout: int = 600) -> str:
        """Block until this run reaches a terminal state.

        Right after a push, Kaggle can still report the *previous* run's
        terminal status. Returning on that would fetch the previous round's
        results.json and write stale scores into the discovery tree, so the run
        must first be observed queued or running.
        """
        started, deadline = False, time.time() + start_timeout
        while not started and time.time() < deadline:
            s = self.status()
            if s != "pending" and s not in self.TERMINAL:
                started = True
                break
            time.sleep(min(self.poll_seconds, 15))

        deadline = time.time() + self.timeout_hours * 3600
        while time.time() < deadline:
            s = self.status()
            if s in self.TERMINAL:
                return s
            time.sleep(self.poll_seconds)
        raise KernelError(f"kernel {self.slug} still {self.status()} after "
                          f"{self.timeout_hours}h")

    def fetch(self) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        _kaggle("kernels", "output", self.slug, "-p", str(self.out_dir))
        results = self.out_dir / "results.json"
        if not results.exists():
            raise KernelError(f"no results.json in {self.out_dir}; the kernel "
                              f"likely died before the first candidate finished")
        return json.loads(results.read_text())

    def run_round(self, round_cfg: dict) -> list[dict]:
        """Push, wait, fetch. Returns one record per candidate.

        A kernel that errors after some candidates finished still yields their
        results, so a partial round feeds the tree instead of being discarded.
        """
        self.push(round_cfg)
        state = self.wait()
        try:
            payload = self.fetch()
        except KernelError:
            if state == "complete":
                raise
            return []

        results = payload.get("results", [])
        # Second guard against a stale fetch: the results must be about the
        # candidates this round actually scheduled.
        wanted = {c["node_id"] for c in round_cfg["candidates"]}
        got = {r.get("node_id") for r in results}
        if results and not (got & wanted):
            raise KernelError(
                f"fetched results for {sorted(got)} but this round scheduled "
                f"{sorted(wanted)}; refusing to score the tree from another run")
        return [r for r in results if r.get("node_id") in wanted]


class LocalMockExecutor:
    """Deterministic stand-in used to exercise the loop without spending quota.

    Scores a candidate from properties the real metric is known to reward --
    spectral channel diversity, resolution, training length -- so the plumbing
    can be tested end to end. It is not a performance predictor.
    """

    MODE_BONUS = {"pseudo_rgb": 0.00, "spread_rgb": 0.05, "pca3": 0.06,
                  "rgb_plus_ratio": 0.055, "band_stack": 0.08}

    def __init__(self, seed: int = 0, noise: float = 0.006):
        import random
        self._rng = random.Random(seed)
        self.noise = noise

    def run_round(self, round_cfg: dict) -> list[dict]:
        out = []
        for entry in round_cfg["candidates"]:
            c = entry["candidate"]
            s = (0.30
                 + self.MODE_BONUS.get(c["channels"]["mode"], 0.0)
                 + 0.00012 * (c["train"]["imgsz"] - 640)
                 + 0.0009 * min(c["train"]["epochs"], 45)
                 + (0.012 if c["infer"]["tta"] else 0.0)
                 + (0.004 if c["channels"]["stretch_hi"] < 100 else 0.0)
                 + self._rng.gauss(0, self.noise))
            out.append({
                "node_id": entry["node_id"], "candidate": c,
                "score": round(max(0.0, s), 5),
                "diagnostics": {"mock": True},
                "cost_seconds": 600.0,
            })
        return out
