#!/usr/bin/env python3
"""Build a short GPU kernel that measures the S3T-DETR training step before the
11-hour run is committed to it: peak memory, seconds per step, NaNs, and which
attention kernels actually run.

No dataset rendering: random 16-band 1024x1024 images with fake boxes go
through the real RT-DETR loss (denoising queries included), the real S3T front
(stem widening, AIFI context, chunked checkpoints) and the real accelerations
(AMP with fp32 loss, SDPA attention, fused optimizer). One T4 is enough: memory
is per card.

    python3 tools/build_s3t_detr_probe.py --slug <user>/hod26-s3t-detr-probe
    kaggle kernels push -p kernels/s3t_detr_probe/build
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from build_kernel import build  # noqa: E402
from tools.s3t_round import s3t_candidate  # noqa: E402

PROBE = r'''

# ============================ S3T-DETR probe ============================
import gc
import torch

P0 = time.time()


def say(msg):
    print(f"[{time.time() - P0:7.1f}s] {msg}", flush=True)


torch.backends.cudnn.benchmark = True
CUDA = torch.cuda.is_available()          # CPU only for a local dry run of this script
dev = torch.device("cuda:0" if CUDA else "cpu")
sync = torch.cuda.synchronize if CUDA else (lambda: None)
peak = (lambda: torch.cuda.max_memory_allocated(dev) / 2**30) if CUDA else (lambda: 0.0)
if CUDA:
    props = torch.cuda.get_device_properties(dev)
    TOTAL = props.total_memory / 2**30
    say(f"torch {torch.__version__}  {props.name}  {TOTAL:.1f} GB  sm{props.major}{props.minor}")
else:
    TOTAL = 1e9
    say("no GPU: local dry run")
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.downloads import attempt_download_asset
COCO = torch.load(attempt_download_asset("rtdetr-l.pt"), map_location="cpu", weights_only=False)["model"]
mae = find_mae_checkpoint(MAE_FILE)
say(f"MAE encoder: {mae}")


def build_net(chunks, compile_blocks, s3t=True, scale=0.5, fast=False, freeze=False):
    # Built the way RTDETRTrainer.get_model builds it: the 18-class graph from
    # the yaml, then every shape-compatible COCO tensor loaded into it.
    net = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=18, verbose=False)
    net.load(COCO, verbose=False)
    if s3t:
        install_spectral_adapter(net, 16, projection=LDA_16_TO_3, kind="s3t",
                                 mae_ckpt=str(mae) if mae else None, scale=scale, widen=True,
                                 context=True, ckpt_chunks=chunks, compile_blocks=compile_blocks,
                                 fast_kernels=fast, train_encoder=not freeze)
    done = enable_transformer_accel(net, fp32_loss=True, nc=18)
    return net.to(dev).train(), done


def fake_batch(b, g, ch=16):
    img = torch.rand(b, ch, IMGSZ, IMGSZ, device=dev, generator=g)  # [0,1], as ultralytics feeds u8/255
    n = 6
    xy = torch.rand(b * n, 2, device=dev, generator=g) * 0.7 + 0.15
    wh = torch.rand(b * n, 2, device=dev, generator=g) * 0.1 + 0.02
    return {"img": img,
            "batch_idx": torch.arange(b, device=dev).repeat_interleave(n).float(),
            "cls": torch.randint(0, 18, (b * n, 1), device=dev, generator=g).float(),
            "bboxes": torch.cat([xy, wh], 1)}


def attention_kernels(net, batch, amp):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA if CUDA else ProfilerActivity.CPU]) as prof:
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp and CUDA):
            loss, _ = net.loss(batch)
        loss.sum().backward()
        sync()
    names = {}
    for e in prof.key_averages():
        k = e.key.lower()
        if any(t in k for t in ("fmha", "attention", "efficient", "flash", "mem_eff", "cutlass")):
            names[e.key[:90]] = e.count
    return names


def top_ops(net, batch, amp, n=30):
    """Where one training step's GPU time goes: ops by self CUDA time, plus
    the S3T front's share measured with record_function ranges."""
    from torch.profiler import ProfilerActivity, profile, record_function
    front = net.model[0]
    orig = front.forward

    def timed(x):
        with record_function("S3T_FRONT_FWD"):
            return orig(x)

    front.forward = timed
    try:
        for _ in range(2):                                  # warm (compile, cudnn autotune)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp and CUDA):
                l, _ = net.loss(batch)
            l.sum().backward()
        sync()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp and CUDA):
                l, _ = net.loss(batch)
            l.sum().backward()
            sync()
    finally:
        front.forward = orig
    ka = prof.key_averages()
    attr = "self_device_time_total" if hasattr(ka[0], "self_device_time_total") else "self_cuda_time_total"
    tot = sum(getattr(e, attr) for e in ka) or 1
    rows = sorted(ka, key=lambda e: getattr(e, attr), reverse=True)[:n]
    out = [(e.key[:70], round(getattr(e, attr) / 1e3, 1), round(100 * getattr(e, attr) / tot, 1), e.count)
           for e in rows]
    say("   top GPU ops (self ms, % of step, calls):")
    for k, ms, pct, c in out:
        say(f"     {ms:8.1f} ms  {pct:5.1f}%  x{c:<5d} {k}")
    fr = [e for e in ka if e.key == "S3T_FRONT_FWD"]
    if fr:
        say(f"   S3T front forward range: {getattr(fr[0], 'device_time_total', 0) / 1e3:.1f} ms GPU")
    return out


def run(cfg):
    name = cfg["name"]
    if CUDA:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    rec = dict(cfg)
    net = opt = scaler = loss = batch = None
    try:
        s3t = cfg.get("s3t", True)
        ch = 16 if s3t else 3
        net, done = build_net(cfg["chunks"], cfg["compile"], s3t, cfg.get("scale", 0.5),
                              cfg.get("fast", False), cfg.get("freeze", False))
        rec["accel"] = done
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-5, fused=CUDA)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and CUDA)
        g = torch.Generator(device=dev).manual_seed(0)
        nan, times, scales = 0, [], []
        for step in range(cfg["steps"]):
            batch = fake_batch(cfg["batch"], g, ch)
            sync()
            t = time.time()
            with torch.autocast(dev.type, dtype=torch.float16, enabled=cfg["amp"] and CUDA):
                loss, _ = net.loss(batch)
            loss = loss.sum()
            if not torch.isfinite(loss):
                nan += 1
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sync()
            if step >= 3:
                times.append(time.time() - t)
            scales.append(scaler.get_scale() if scaler.is_enabled() else 1.0)
        rec.update(ok=True, peak_gb=peak(),
                   s_per_step=sum(times) / max(1, len(times)), nan_steps=nan,
                   final_scale=scales[-1], min_scale=min(scales))
        if cfg.get("profile"):
            rec["attention_kernels"] = attention_kernels(net, fake_batch(cfg["batch"], g, ch), cfg["amp"])
        if cfg.get("profile_ops"):
            rec["top_ops"] = top_ops(net, fake_batch(cfg["batch"], g, ch), cfg["amp"])
    except torch.OutOfMemoryError as e:
        rec.update(ok=False, error="OOM", peak_gb=peak())
    except Exception as e:                                        # noqa: BLE001
        rec.update(ok=False, error=f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")
    finally:
        net = opt = scaler = loss = batch = None
        gc.collect()
        if CUDA:
            torch.cuda.empty_cache()
    fits = rec.get("ok") and rec["peak_gb"] < 0.85 * TOTAL
    say(f"{name:34s} " + (f"peak {rec['peak_gb']:5.2f} GB  {rec['s_per_step']:5.2f} s/step  "
                          f"({cfg['batch'] / rec['s_per_step']:.2f} img/s)  NaN steps {rec['nan_steps']}  "
                          f"{'FITS' if fits else 'TOO BIG'}"
                          if rec.get("ok") else f"FAILED {rec.get('error')}  (peak {rec.get('peak_gb', 0):.2f} GB)"))
    if rec.get("attention_kernels") is not None:
        say(f"   attention kernels: {rec['attention_kernels'] or 'none matched'}")
    rec["fits"] = bool(fits)
    return rec


results = [run(c) for c in CONFIGS]
say("=" * 70)
best = [r for r in results if r.get("fits") and not r.get("nan_steps")]
if best:
    top = max(best, key=lambda r: r["batch"] / r["s_per_step"])
    say(f"fastest that fits: {top['name']}  ({top['batch'] / top['s_per_step']:.2f} img/s, "
        f"peak {top['peak_gb']:.2f} GB)")
Path("/kaggle/working/s3t_detr_probe.json").write_text(json.dumps(results, indent=1, default=str))
say("PROBE DONE")
'''

CONFIGS = [
    # the configuration that OOM'd, for reference
    {"name": "fp32  whole-ckpt  b1", "amp": False, "chunks": 0, "batch": 1, "compile": False, "steps": 5},
    {"name": "fp32  chunks8     b2", "amp": False, "chunks": 8, "batch": 2, "compile": False, "steps": 8},
    {"name": "AMP   chunks8     b2", "amp": True, "chunks": 8, "batch": 2, "compile": False, "steps": 25,
     "profile": True},
    {"name": "AMP   chunks8     b4", "amp": True, "chunks": 8, "batch": 4, "compile": False, "steps": 12},
    {"name": "AMP   chunks16    b4", "amp": True, "chunks": 16, "batch": 4, "compile": False, "steps": 8},
    {"name": "AMP   chunks8     b2  compiled", "amp": True, "chunks": 8, "batch": 2, "compile": True,
     "steps": 12},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--out-dir", type=Path, default=REPO / "kernels" / "s3t_detr_probe" / "build")
    ap.add_argument("--mae-kernel", default="qwyi123/hod26-s3t-mae-pretrain")
    ap.add_argument("--mae-file", default=None)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--configs", default=None,
                    help="JSON list replacing the default configurations (keys: name, amp, chunks, "
                         "batch, compile, steps, [s3t], [scale], [profile])")
    args = ap.parse_args()
    cand = s3t_candidate(total=2)
    src = build({"round": "probe", "candidates": [], "submit": {"candidate": cand}})
    tail = 'if __name__ == "__main__":\n    main()'
    if not src.rstrip().endswith(tail):
        raise RuntimeError("generated kernel no longer ends with the main() guard")
    src = src.rstrip()[: -len(tail)]
    head = (f"MAE_FILE = {args.mae_file!r}\nIMGSZ = {args.imgsz}\n"
            f"CONFIGS = {json.dumps(json.loads(args.configs) if args.configs else CONFIGS)}\n"
            .replace("true", "True").replace("false", "False"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "s3t_detr_probe.py").write_text(src + "\n" + head + PROBE)
    (args.out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": args.slug, "title": args.slug.split("/")[-1].replace("-", " ").title(),
        "code_file": "s3t_detr_probe.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True, "competition_sources": [], "dataset_sources": [],
        "kernel_sources": [args.mae_kernel],
    }, indent=2))
    print(f"wrote {args.out_dir / 's3t_detr_probe.py'}  slug={args.slug}")


if __name__ == "__main__":
    main()
