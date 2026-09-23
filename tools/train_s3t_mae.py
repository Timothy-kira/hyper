"""MAE pretraining of the S3T spectral encoder, single or multi GPU.

Launch with torchrun; one process per GPU:

    torchrun --nproc_per_node=2 tools/train_s3t_mae.py --data <root> --minutes 12

Every acceleration it turns on is logged as on or off with the reason, so a
smoke run says what actually happened rather than what was asked for:
DDP (NCCL), fp16 autocast + GradScaler, SDPA backend probe, visible-only
encoding, torch.compile (with eager fallback), fused AdamW, channels_last,
pinned persistent loader workers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
for cand in (HERE.parent / "src", HERE / "src"):
    if cand.is_dir():
        sys.path.insert(0, str(cand))

from hod26.cube import load_planar                      # noqa: E402
from hod26.s3t.mae import S3TMAE                        # noqa: E402
from hod26.s3t.preprocess import features               # noqa: E402
from hod26.s3t.spectral import SpectralEncoder          # noqa: E402

RANK = int(os.environ.get("RANK", 0))
LOCAL = int(os.environ.get("LOCAL_RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))
T0 = time.time()


def log(msg):
    if RANK == 0:
        print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def find_root(given):
    if given:
        return Path(given)
    stack = [(Path("/kaggle/input"), 0)]
    while stack:
        d, k = stack.pop()
        if (d / "train" / "images").is_dir() and (d / "test" / "images").is_dir():
            return d
        if k < 5 and d.is_dir():
            stack.extend((c, k + 1) for c in sorted(d.iterdir()) if c.is_dir())
    raise FileNotFoundError("no dataset with train/images and test/images under /kaggle/input")


class Crops(torch.utils.data.Dataset):
    """Unlabelled frames (train + test) -> k random crops of prepared features each."""

    def __init__(self, files, crop, k, seed, flip=False):
        self.files, self.crop, self.k, self.seed, self.flip = files, crop, k, seed, flip

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, int(time.time() * 1e3) & 0xFFFF))
        f = features(load_planar(self.files[i]))           # (3, 16, H, W)
        _, _, h, w = f.shape
        c = self.crop
        out = []
        for _ in range(self.k):
            y = int(rng.integers(0, h - c + 1))
            x = int(rng.integers(0, w - c + 1))
            p = f[:, :, y:y + c, x:x + c]
            if self.flip and rng.random() < 0.5:
                p = p[..., ::-1]
            if self.flip and rng.random() < 0.5:
                p = p[..., ::-1, :]
            out.append(np.ascontiguousarray(p))
        return torch.from_numpy(np.stack(out)).half()


def sdpa_probe(device, hd):
    """Which SDPA kernels run on this GPU for our shapes (fp16, seq 16)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q = torch.randn(64, 4, 16, hd, device=device, dtype=torch.float16)
    res = {}
    for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH"):
        be = getattr(SDPBackend, name, None)
        if be is None:
            res[name] = "not in this torch"
            continue
        try:
            with sdpa_kernel(be):
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
            res[name] = "ok"
        except Exception as e:                                   # noqa: BLE001
            res[name] = f"unavailable ({str(e).splitlines()[0][:90]})"
    return res


def compile_units(model):
    """The modules whose small kernels dominate a step: every Transformer block
    and spatial mix, in the encoder and the decoder."""
    from hod26.s3t.spectral import SpatialMix, SpectralBlock
    return [m for m in model.modules() if isinstance(m, (SpectralBlock, SpatialMix))]


def compile_blocks(model, mode):
    """Compile each block in place (nn.Module.compile).

    Block by block rather than torch.compile(DDP(model)): the model DDP wraps
    stays a plain module, so DDP's bucketing never meets a dynamo graph -- the
    two whole-model attempts on 2x T4 both died on an inductor stride guard
    there. Inside a block inductor still fuses LayerNorm, GELU, the residual
    adds and the LayerScale multiplies into a few Triton kernels, and with
    mode="reduce-overhead" each block also runs as a CUDA graph, removing the
    per-kernel launch cost that a 0.25M-parameter model is dominated by.
    """
    units = compile_units(model)
    for m in units:
        m.compile(mode=None if mode == "default" else mode, dynamic=False)
    return len(units)


def uncompile(model):
    for m in model.modules():
        if getattr(m, "_compiled_call_impl", None) is not None:
            m._compiled_call_impl = None


class GpuUtil:
    """Samples nvidia-smi in a thread; mean utilisation per GPU since start()."""

    def __init__(self, every=2.0):
        import threading
        self.every, self.samples, self.on = every, [], False
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import subprocess
        while True:
            if self.on:
                try:
                    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                                          "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=5).stdout
                    self.samples.append([float(v) for v in out.split()])
                except Exception:                                # noqa: BLE001
                    pass
            time.sleep(self.every)

    def start(self):
        self.samples, self.on = [], True
        if not self.t.is_alive():
            self.t.start()

    def mean(self):
        if not self.samples:
            return None
        n = min(len(s) for s in self.samples)
        return [round(sum(s[i] for s in self.samples) / len(self.samples), 1) for i in range(n)]


def pseudo_rgb(level, bands=(5, 8, 13)):
    x = level[list(bands)].float().cpu().numpy()
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return (np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="/kaggle/working/s3t_mae")
    ap.add_argument("--minutes", type=float, default=12)
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=32, help="crops per GPU per step")
    ap.add_argument("--crops-per-frame", type=int, default=8)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--compile-mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune-no-cudagraphs"],
                    help="reduce-overhead adds CUDA graphs on top of kernel fusion")
    ap.add_argument("--compile-scope", default="blocks", choices=["blocks", "whole"],
                    help="blocks: compile each block in place (DDP-safe); "
                         "whole: torch.compile(DDP(model)), the form that failed on 2x T4")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--limit-frames", type=int, default=0)
    ap.add_argument("--flip", type=int, default=0,
                    help="random flips of the crops (off: only official frames, only cropped)")
    ap.add_argument("--schedule", choices=["step", "time"], default="step",
                    help="cosine over --max-steps, or over the --minutes budget")
    ap.add_argument("--save-every-min", type=float, default=0,
                    help="also checkpoint every N minutes (0: only at the end)")
    args = ap.parse_args()

    cuda = torch.cuda.is_available()
    if WORLD > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
    device = torch.device(f"cuda:{LOCAL}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
    torch.manual_seed(1234 + RANK)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"tag": args.tag, "world": WORLD, "args": vars(args)}

    if cuda:
        cap = torch.cuda.get_device_capability(device)
        report["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        report["capability"] = f"sm{cap[0]}{cap[1]}"
        report["sdpa"] = sdpa_probe(device, args.dim // args.heads)
        log(f"torch {torch.__version__}  world {WORLD}  GPUs {report['gpu']}  {report['capability']}")
        for k, v in report["sdpa"].items():
            log(f"  SDPA {k:20s} {v}")
        if cap[0] < 8:
            log("  -> FlashAttention needs sm80+; on this GPU SDPA dispatches to the "
                "best available kernel above (memory-efficient on a T4)")
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() and cap[0] >= 8 else torch.float16
    else:
        amp_dtype = torch.bfloat16
    log(f"accel: DDP={'on (' + ('nccl' if cuda else 'gloo') + ')' if WORLD > 1 else 'off (1 process)'}  "
        f"autocast={amp_dtype}  GradScaler={'on' if amp_dtype == torch.float16 else 'off'}")

    root = find_root(args.data)
    files = sorted((root / "train" / "images").glob("*.png")) + sorted((root / "test" / "images").glob("*.png"))
    if args.limit_frames:
        files = files[:args.limit_frames]
    n_tr = len(list((root / "train" / "images").glob("*.png")))
    log(f"data: {root}  {len(files)} unlabelled frames = all official images "
        f"({n_tr} train + {len(files) - n_tr} test), no labels read, no generated images; "
        f"random {args.crop}px crops, flips {'on' if args.flip else 'off'}")
    ds = Crops(files, args.crop, args.crops_per_frame, seed=RANK, flip=bool(args.flip))
    sampler = torch.utils.data.DistributedSampler(ds, WORLD, RANK, shuffle=True, seed=0) if WORLD > 1 else None
    frames_per_step = max(1, args.batch // args.crops_per_frame)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=frames_per_step, sampler=sampler, shuffle=sampler is None,
        num_workers=args.workers, pin_memory=cuda, persistent_workers=args.workers > 0,
        drop_last=True, prefetch_factor=4 if args.workers > 0 else None)
    log(f"loader: {args.workers} workers/rank, pin_memory={cuda}, persistent, "
        f"{frames_per_step} frames x {args.crops_per_frame} crops = {frames_per_step * args.crops_per_frame} crops/GPU/step")

    enc = SpectralEncoder(dim=args.dim, depth=args.depth, heads=args.heads)
    model = S3TMAE(enc).to(device)
    # Only the conv weights: channels_last on a 4-D embedding Parameter would
    # just give DDP mismatched gradient strides.
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.to(memory_format=torch.channels_last)
    log("channels_last: on (conv weights)")
    n_par = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in enc.parameters())
    log(f"model: encoder {n_enc / 1e6:.2f}M params, MAE total {n_par / 1e6:.2f}M; "
        f"mask 0.75 spatial (units of 4x4 tokens) + 0.15 contiguous bands; "
        f"encoder computes visible positions only (25% of tokens)")
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05,
                                betas=(0.9, 0.95), fused=cuda)
        log(f"optimizer: AdamW fused={cuda}")
    except (TypeError, RuntimeError) as e:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
        log(f"optimizer: AdamW fused=off ({e})")
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and amp_dtype == torch.float16)

    if cuda:
        # Largest per-GPU batch that fits, halving from --batch on OOM, agreed
        # across ranks. A wrong guess used to kill the whole session at step 1.
        # Eager probe. It overestimates what the compiled step needs (the
        # compiled graph fuses elementwise chains), so the batch it picks is safe
        # for both -- and it is what the eager fallback below will actually use.
        # Probing through torch.compile was tried and tripped a guard when the
        # DDP-wrapped model was compiled a second time.
        import gc
        bs = args.batch
        while True:
            ok = True
            try:
                torch.cuda.reset_peak_memory_stats(device)
                xp = torch.randn(bs, 3, 16, args.crop, args.crop, device=device)
                with torch.autocast("cuda", dtype=amp_dtype):
                    lp, *_ = model(xp)
                lp.backward()
            except torch.OutOfMemoryError:
                ok = False
            # Outside the except block, so the traceback (and every tensor its
            # frames hold) is gone before the peak is read and the cache emptied.
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            model.zero_grad(set_to_none=True)
            xp = lp = None
            gc.collect()
            torch.cuda.empty_cache()
            fits = ok and peak < 0.75 * torch.cuda.get_device_properties(device).total_memory / 2**30
            log(f"memory probe (eager): {bs} crops/GPU -> peak {peak:.2f} GB "
                f"{'fits' if fits else 'too big'}")
            if fits or bs <= args.crops_per_frame:
                break
            bs = max(args.crops_per_frame, (bs // 2) // args.crops_per_frame * args.crops_per_frame)
        t = torch.tensor([bs], device=device)
        if WORLD > 1:
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
        bs = int(t.item())
        if bs != args.batch:
            frames_per_step = max(1, bs // args.crops_per_frame)
            dl = torch.utils.data.DataLoader(
                ds, batch_size=frames_per_step, sampler=sampler, shuffle=sampler is None,
                num_workers=args.workers, pin_memory=cuda, persistent_workers=args.workers > 0,
                drop_last=True, prefetch_factor=4 if args.workers > 0 else None)
            log(f"batch reduced to {frames_per_step * args.crops_per_frame} crops/GPU/step")
        report["batch_per_gpu"] = frames_per_step * args.crops_per_frame
        torch.cuda.reset_peak_memory_stats(device)

    net = model
    if WORLD > 1:
        net = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[LOCAL] if cuda else None, gradient_as_bucket_view=True,
            static_graph=True)
    run = net
    compiled = False
    if args.compile and (cuda or os.environ.get("S3T_COMPILE_CPU")):
        try:
            import torch._dynamo as dynamo   # "as": a bare import would make torch local here
            # DDPOptimizer splits graphs at DDP bucket boundaries; it is the
            # usual source of the stride-guard failures seen on 2x T4.
            dynamo.config.optimize_ddp = False
            if args.compile_scope == "blocks":
                n = compile_blocks(model, args.compile_mode)
                log(f"torch.compile: {n} blocks compiled in place, mode={args.compile_mode}, "
                    f"optimize_ddp=off")
            else:
                run = torch.compile(net, mode=None if args.compile_mode == "default" else args.compile_mode)
                log(f"torch.compile: whole model, mode={args.compile_mode}, optimize_ddp=off")
            compiled = True
        except Exception as e:                                   # noqa: BLE001
            uncompile(model)
            run = net
            log(f"torch.compile: off ({e})")
    report["compile_mode"] = args.compile_mode if compiled else "eager"
    report["compile_scope"] = args.compile_scope
    graphs = compiled and args.compile_mode == "reduce-overhead"
    util = GpuUtil() if cuda and RANK == 0 else None

    warm = 30
    cosine = args.max_steps < 100000 or args.schedule == "time"
    frac = 0.0            # fraction of the time budget spent, identical on every rank

    def lr_at(step):
        # Every rank must use the same lr, or their weights drift apart under
        # DDP: the step count is shared, and the time fraction is all-reduced.
        if step < warm:
            return args.lr * (step + 1) / warm
        if not cosine:
            return args.lr
        if args.schedule == "time":
            p = frac
        else:
            p = (step - warm) / max(1, args.max_steps - warm)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    step, epoch, hist = 0, 0, []
    seen = 0
    t_steady, seen_steady = None, 0
    stop = torch.zeros(2, device=device)
    t_train0, last_save = time.time(), time.time()
    deadline = T0 + args.minutes * 60
    model.train()
    done = False
    wait, t_prev = 0.0, time.time()
    while not done:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for xb in dl:
            if step >= 20:
                wait += time.time() - t_prev
            xb = xb.flatten(0, 1).to(device, non_blocking=True).float()
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            t_step = time.time()
            failed = None
            if graphs:
                torch.compiler.cudagraph_mark_step_begin()
            try:
                with torch.autocast(device.type, dtype=amp_dtype, enabled=cuda):
                    loss, parts, *_ = run(xb)
                scaler.scale(loss).backward()
            except Exception as e:                               # noqa: BLE001
                # The first steps are where compilation (and CUDA-graph
                # recording, which re-records on step 1-2) happens.
                if not (compiled and step < 3):
                    raise
                failed = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
            if failed:
                # Retried outside the except block: inside it the traceback
                # keeps the failed attempt's activations alive, and the eager
                # retry then ran out of memory on top of them.
                import gc
                log(f"torch.compile: failed at step {step}, falling back to eager ({failed})")
                uncompile(model)
                run, compiled, graphs = net, False, False
                report["compile_mode"] = f"eager (fallback: {failed[:120]})"
                loss = parts = None
                opt.zero_grad(set_to_none=True)
                gc.collect()
                if cuda:
                    torch.cuda.empty_cache()
                with torch.autocast(device.type, dtype=amp_dtype, enabled=cuda):
                    loss, parts, *_ = run(xb)
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if step == 0:
                if cuda:
                    torch.cuda.synchronize()
                log(f"torch.compile: {'on' if compiled else 'off'}; first step (incl. compile) "
                    f"{time.time() - t_step:.1f}s")
            step += 1
            seen += xb.shape[0] * WORLD
            if step == 20:
                if cuda:
                    torch.cuda.synchronize()
                t_steady, seen_steady = time.time(), seen
                if util:
                    util.start()
            lv = float(loss.detach())
            if not math.isfinite(lv):
                log(f"NON-FINITE loss at step {step}: {lv}")
            if step % 10 == 0 or step <= 3:
                rate = ""
                if t_steady is not None and step > 20:
                    if cuda:
                        torch.cuda.synchronize()
                    span = time.time() - t_steady
                    rate = (f"  {(seen - seen_steady) / span:7.1f} crops/s"
                            f"  data-wait {100 * wait / span:4.1f}%")
                    if util and util.mean():
                        rate += f"  gpu-util {util.mean()}%"
                mem = f"  mem {torch.cuda.max_memory_allocated(device) / 2**30:.2f}G" if cuda else ""
                log(f"step {step:5d}  loss {lv:.4f}  (norm {float(parts['norm']):.3f} "
                    f"l1 {float(parts['l1']):.3f} grad {float(parts['grad']):.4f})  "
                    f"gnorm {float(gn):.2f}  scale {scaler.get_scale() if scaler.is_enabled() else 1:.0f}  "
                    f"lr {lr_at(step):.2e}{rate}{mem}")
                hist.append({"step": step, "loss": lv, "t": time.time() - T0,
                             "scale": scaler.get_scale() if scaler.is_enabled() else 1.0})
            stop[0] = float(time.time() > deadline or step >= args.max_steps)
            stop[1] = (time.time() - t_train0) / max(1.0, deadline - t_train0)
            t_prev = time.time()
            if WORLD > 1:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            frac = float(stop[1])
            if args.save_every_min and RANK == 0 and time.time() - last_save > 60 * args.save_every_min:
                torch.save({"encoder": enc.state_dict(), "step": step,
                            "config": {"dim": args.dim, "depth": args.depth, "heads": args.heads}},
                           out / f"{args.tag}_encoder_latest.pt")
                last_save = time.time()
                log(f"saved {args.tag}_encoder_latest.pt at step {step}")
            if stop[0].item() > 0:
                done = True
                break
        epoch += 1

    if cuda:
        torch.cuda.synchronize()
    elapsed = time.time() - t_steady if t_steady else float("nan")
    report.update({
        "steps": step, "compiled": compiled, "amp": str(amp_dtype),
        "crops_per_s": (seen - seen_steady) / elapsed if t_steady and step > 20 else None,
        "data_wait_frac": wait / elapsed if t_steady and step > 20 else None,
        "gpu_util": util.mean() if util else None,
        "peak_mem_gb": [torch.cuda.max_memory_allocated(device) / 2**30] if cuda else None,
        "loss_first": hist[0]["loss"] if hist else None,
        "loss_last": hist[-1]["loss"] if hist else None, "history": hist,
    })
    if WORLD > 1 and cuda:
        mem = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**30], device=device)
        allm = [torch.zeros_like(mem) for _ in range(WORLD)]
        dist.all_gather(allm, mem)
        report["peak_mem_gb"] = [float(m) for m in allm]

    if RANK == 0:
        ck = out / f"{args.tag}_mae.pt"
        torch.save({"mae": model.state_dict(), "encoder": enc.state_dict(),
                    "config": {"dim": args.dim, "depth": args.depth, "heads": args.heads},
                    "step": step}, ck)
        back = torch.load(ck, map_location="cpu", weights_only=True)
        fresh = S3TMAE(SpectralEncoder(dim=args.dim, depth=args.depth, heads=args.heads))
        fresh.load_state_dict(back["mae"])
        same = all(torch.equal(a.cpu(), b) for a, b in zip(model.state_dict().values(),
                                                           fresh.state_dict().values()))
        report["checkpoint_roundtrip"] = same
        log(f"checkpoint {ck.name}: {ck.stat().st_size / 1e6:.1f} MB, reload identical={same}")

        # Reconstruction picture: input | masked input | reconstruction, bands 5/8/13.
        try:
            from PIL import Image
            model.eval()
            xb = ds[0][:1].to(device).float()
            with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype, enabled=cuda):
                _, _, pred, idx, bm = model(xb)
            s = enc.stride
            gh, gw = xb.shape[-2] // s, xb.shape[-1] // s
            keep = torch.zeros(gh * gw, device=device)
            keep[idx[0]] = 1
            kp = keep.view(gh, 1, gw, 1).expand(gh, s, gw, s).reshape(xb.shape[-2:])
            rec = pred[0].float().view(gh, gw, 16, s, s).permute(2, 0, 3, 1, 4).reshape(16, *xb.shape[-2:])
            lvl = xb[0, 0]
            panels = [pseudo_rgb(lvl), pseudo_rgb(lvl * kp), pseudo_rgb(rec * (1 - kp) + lvl * kp)]
            Image.fromarray(np.concatenate(panels, 1)).resize(
                (3 * 256, 256), Image.NEAREST).save(out / f"{args.tag}_recon.png")
            log(f"wrote {args.tag}_recon.png (input | masked | reconstruction, bands 5/8/13)")
        except Exception as e:                                   # noqa: BLE001
            log(f"recon picture skipped: {e}")
        (out / f"{args.tag}_report.json").write_text(json.dumps(report, indent=1, default=str))
        log(f"DONE {args.tag}: steps {step}, loss {report['loss_first']} -> {report['loss_last']}, "
            f"{report['crops_per_s']} crops/s, peak mem {report['peak_mem_gb']}")
    if WORLD > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
