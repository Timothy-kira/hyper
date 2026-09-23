"""Build the Kaggle kernel that smoke-tests S3T MAE pretraining on 2x T4.

The kernel carries the repo sources inline, writes them to disk (torchrun
workers import them by path, like the DDP fix in the round driver), then runs:

  A. 1 GPU,  compile on, ~60 steps  -- single-card throughput reference
  B. 2 GPUs, compile on, time-boxed -- the DDP run under test

and prints a verdict line per pass criterion.

    python3 tools/build_s3t_smoke.py --out-dir kernels/s3t_smoke/build
    kaggle kernels push -p kernels/s3t_smoke/build
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCES = [
    "src/hod26/cube.py",
    "src/hod26/s3t/__init__.py",
    "src/hod26/s3t/preprocess.py",
    "src/hod26/s3t/spectral.py",
    "src/hod26/s3t/mae.py",
    "tools/train_s3t_mae.py",
]

BODY = r'''
import json, os, subprocess, sys, time
from pathlib import Path

T0 = time.time()
CODE = Path("/kaggle/working/s3t_code")
OUT = Path("/kaggle/working/s3t_mae")


def say(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


for rel, text in SOURCES.items():
    dst = CODE / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text)
(CODE / "src" / "hod26" / "__init__.py").write_text("")

import torch
n = torch.cuda.device_count()
say(f"preflight: torch {torch.__version__}, {n} GPU(s): "
    f"{[torch.cuda.get_device_name(i) for i in range(n)]}")
if n < 2:
    say(f"PREFLIGHT FAILED: asked for 2 GPUs and got {n}; stopping before spending more")
    sys.exit(1)
say("preflight passed")


def run(tag, nproc, extra):
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}",
           "--standalone", str(CODE / "tools" / "train_s3t_mae.py"),
           "--out", str(OUT), "--tag", tag] + extra
    say(f"launch {tag}: {' '.join(cmd[3:])}")
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1",
               PYTORCH_ALLOC_CONF="expandable_segments:True")
    rc = subprocess.run(cmd, env=env).returncode
    rep = OUT / f"{tag}_report.json"
    return rc, (json.loads(rep.read_text()) if rep.exists() else None)


cpus = os.cpu_count() or 4
if MODE == "smoke":
    rc_a, a = run("single", 1, CONFIG + ["--max-steps", "60", "--minutes", "6",
                                         "--workers", str(max(1, cpus - 1))])
else:
    rc_a, a = 0, {}
TAG = "dual" if MODE == "smoke" else "pretrain"
rc_b, b = run(TAG, 2, CONFIG + ["--minutes", str(DUAL_MINUTES),
                                "--workers", str(max(1, cpus // 2 - 1))]
              + ([] if MODE == "smoke" else ["--schedule", "time", "--save-every-min", "10"]))

say("=" * 70)
verdict = []
def crit(name, ok, detail):
    verdict.append(ok)
    say(f"{'PASS' if ok else 'FAIL'}  {name:34s} {detail}")

if MODE == "smoke":
    crit("single-GPU run finished", rc_a == 0 and a is not None, f"rc={rc_a}")
crit("dual-GPU DDP run finished", rc_b == 0 and b is not None, f"rc={rc_b}")
if b:
    hist = b.get("history") or []
    losses = [h["loss"] for h in hist]
    finite = all(l == l and abs(l) != float("inf") for l in losses)
    scales = [h["scale"] for h in hist]
    crit("both ranks trained", b.get("world") == 2, f"world={b.get('world')}")
    crit("no NaN/Inf, GradScaler stable", finite and min(scales or [1]) >= 1,
         f"scale min {min(scales or [0])}, max {max(scales or [0])}")
    fell = bool(losses) and losses[-1] < 0.8 * losses[0]
    crit("loss falls", fell, f"{losses[0] if losses else None} -> {losses[-1] if losses else None}")
    crit("checkpoint round-trips", bool(b.get("checkpoint_roundtrip")), "")
    crit("recon picture written", (OUT / f"{TAG}_recon.png").exists(), "")
    say(f"accel: compile={b.get('compiled')} amp={b.get('amp')} sdpa={b.get('sdpa')}")
    say(f"peak mem per GPU (GB): {b.get('peak_mem_gb')}")
    say(f"data wait: {b.get('data_wait_frac')}")
    if a and a.get("crops_per_s") and b.get("crops_per_s"):
        say(f"throughput: 1 GPU {a['crops_per_s']:.1f} crops/s, 2 GPU {b['crops_per_s']:.1f} "
            f"crops/s -> speedup {b['crops_per_s'] / a['crops_per_s']:.2f}x")
say(f"{MODE.upper()} {'PASSED' if all(verdict) else 'FAILED'} in {(time.time() - T0) / 60:.1f} min")
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=REPO / "kernels" / "s3t_smoke" / "build")
    ap.add_argument("--slug", default="qwyi123/hod26-s3t-mae-smoke")
    ap.add_argument("--dataset", default="xishengfeng/hod26-planar")
    ap.add_argument("--dual-minutes", type=float, default=11)
    ap.add_argument("--mode", choices=["smoke", "pretrain"], default="smoke",
                    help="pretrain: the dual-GPU run only, time-budgeted cosine, periodic saves")
    ap.add_argument("--public", action="store_true",
                    help="publish the kernel (and so its output checkpoint) instead of private")
    ap.add_argument("--config", default="--batch 32 --crops-per-frame 8 --crop 128 "
                                         "--dim 64 --depth 4 --heads 4 --compile 1")
    args = ap.parse_args()
    sources = {rel: (REPO / rel).read_text() for rel in SOURCES}
    head = (f"SOURCES = {json.dumps(sources)}\n"
            f"CONFIG = {json.dumps(args.config.split())}\n"
            f"DUAL_MINUTES = {args.dual_minutes}\n"
            f"MODE = {args.mode!r}\n")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "s3t_smoke.py").write_text(head + BODY)
    (args.out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": args.slug, "title": args.slug.split("/")[-1].replace("-", " ").title(),
        "code_file": "s3t_smoke.py", "language": "python", "kernel_type": "script",
        "is_private": not args.public, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4x2",
        "enable_internet": True, "competition_sources": [],
        "dataset_sources": [args.dataset], "kernel_sources": [],
    }, indent=2))
    print(f"wrote {args.out_dir / 's3t_smoke.py'}  slug={args.slug}")


if __name__ == "__main__":
    main()
