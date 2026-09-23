#!/usr/bin/env python3
"""Build the S3T-DETR fine-tuning kernel: MAE-pretrained spectral Transformer
in front of a COCO-pretrained RT-DETR-L, trained on 2x T4 with augmentation.

    python3 tools/s3t_round.py --slug <account>/hod26-s3t-detr
    kaggle kernels push -p kernels/s3t_detr/build

Inputs the kernel needs attached:
  - dataset  xishengfeng/hod26-planar                  (the frames)
  - notebook zetaoxia/hod26-s3t-mae-pretrain3 output  (pretrain3_mae.pt, MAE v3)
The preflight refuses to start without either, before any GPU time is spent.

Augmentation, both kinds:
  offline, on the 16-band cube before rendering (one extra copy per frame):
    Savitzky-Golay along *wavelength* order, same-class spectral SMOTE,
    superpixel CutMix;
  online, ultralytics at load time: mosaic, horizontal flip, scale/translate
    (HSV is skipped on 16-channel input by design).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.final_runs import full_candidate  # noqa: E402

OUT = REPO / "kernels" / "s3t_detr" / "build"
MAE_KERNEL = "zetaoxia/hod26-s3t-mae-pretrain3"      # MAE v3, the S3T-X encoder
TOTAL = 45            # epochs: 0.508 s/step (probe v7) -> ~12.3 min/epoch on 2x T4, ~9.2 h,
                      # leaving an hour so the 3 closing no-mosaic epochs always run
SESSION_HOURS = 11.0

AUGMENT = {"sg_window": 7, "sg_polyorder": 2, "sg_chain": True,
           "smote_alpha": 0.3, "cutmix_prob": 0.4, "cutmix_blocks": 24, "copies": 1}


def s3t_candidate(total: int = TOTAL, batch: int = 2, mae_file: str | None = None,
                  compile_blocks: bool = False, arch: str = "tokens") -> dict:
    cand = full_candidate("transformer", total, {
        "channels.mode": "s3t_level",
        "train.spectral_stem": "s3t",
        # The encoder runs at native cube scale on top of an fp32 RT-DETR at
        # 1024; 2 per card keeps the pair inside a T4. nbs stays 64, so the
        # optimizer still steps on an effective batch of 64.
        "train.batch": batch,
    })
    cand["train"].update(s3t_scale=0.5, s3t_require_pretrain=True, s3t_widen=True, s3t_context=True,
                         # Memory and speed (see handoff/S3T.md, "OOM"): per-layer,
                         # 8-chunk checkpoints in the S3T front; AMP on the whole
                         # detector with the loss and Hungarian matching kept in fp32.
                         # normalize() turns AMP off for RT-DETR; this is set after it.
                         s3t_ckpt_chunks=8, s3t_compile=False, amp=True, amp_fp32_loss=True,
                         mosaic=1.0, fliplr=0.5, scale=0.5, close_mosaic=3)
    cand["train"]["s3t_compile"] = bool(compile_blocks)
    if arch == "xca":
        # S3T-X (handoff/S3T.md, "probe"): on one T4 at 1024^2, AMP, b2/card,
        # compiled blocks and no checkpoints are the fastest measured (0.50
        # s/step, 5.3 GB); the encoder is small enough to keep its activations.
        cand["train"].update(s3t_arch="xca", s3t_grad_ckpt=False, s3t_compile=True)
    # Head and losses (handoff/S3T.md, "head"): D-FINE's distribution refinement
    # on the pretrained decoder, DEIM's MAL for the classes, log-space w/h L1.
    # Mosaic already gives DEIM's dense one-to-one supervision.
    cand["train"].update(fdr=True, mal=True, log_size_l1=True)
    # Speed over bitwise reproducibility: deterministic=True would turn on
    # cudnn.deterministic and torch's deterministic algorithms, which limit
    # cudnn.benchmark's choices and swap in slower backward kernels (the
    # deformable attention's grid_sample among them). The probe's 0.50 s/step
    # was measured without them.
    cand["train"]["deterministic"] = False
    # ultralytics converts the whole model to channels_last on CUDA by default.
    # Measured on a T4 with this exact model (S3T-X + D-FINE, AMP, b2): slower
    # (0.586 vs 0.565 s/step) and 12.5 GB peak instead of 5.3 GB. Off.
    cand["train"]["channels_last"] = False
    if mae_file:
        cand["train"]["s3t_mae_file"] = mae_file
    cand["augment"].update(AUGMENT)
    return cand


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="TEAMMATE/hod26-s3t-detr")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--total", type=int, default=TOTAL)
    ap.add_argument("--mae-kernel", action="append", default=None,
                    help="pretraining notebook(s) to mount; default the v1 notebook")
    ap.add_argument("--mae-file", default=None,
                    help="which *_mae.pt to use when several are mounted, e.g. pretrain2_mae.pt")
    ap.add_argument("--batch", type=int, default=2, help="images per GPU")
    ap.add_argument("--compile-blocks", action="store_true",
                    help="torch.compile the S3T blocks (measure with the probe first)")
    ap.add_argument("--arch", choices=["xca", "tokens"], default="xca",
                    help="xca: S3T-X (MAE v3); tokens: the band-token encoder (MAE v1/v2)")
    ap.add_argument("--render-only", action="store_true",
                    help="build the CPU render notebook: materialise this candidate's dataset "
                         "into its output (no GPU, no training)")
    ap.add_argument("--smoke-only", action="store_true",
                    help="run every stage small (smoke train, val, best/last, fp16 eval, predict, "
                         "submission) and stop -- proves the pipeline on scarce quota")
    ap.add_argument("--render-kernel", default=None,
                    help="a render notebook to mount; its dataset is used instead of rendering")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = args.out_dir / "round-config.json"
    cfg.write_text(json.dumps({"round": "hod26-s3t-detr", "candidates": [], "submit": {
        "candidate": s3t_candidate(args.total, args.batch,
                                   args.mae_file or ("pretrain3_mae.pt" if args.arch == "xca" else None),
                                   args.compile_blocks, args.arch),
        "use_all_train": False,          # keep the 600 held out: they are the ruler
        "predict": True,                 # a submission comes out wherever the clock stops
        "session_hours": SESSION_HOURS,
        "require_gpus": 2,
        "render_only": bool(args.render_only),
        "smoke_only": bool(args.smoke_only),
    }}, indent=2))
    subprocess.run([sys.executable, str(REPO / "tools" / "build_kernel.py"),
                    "--round-config", str(cfg), "--out-dir", str(args.out_dir),
                    "--slug", args.slug,
                    *([] if args.render_only else
                      [a for k in (args.mae_kernel or [MAE_KERNEL]) for a in ("--kernel-source", k)]),
                    *(["--kernel-source", args.render_kernel] if args.render_kernel else []),
                    "--machine-shape", "cpu" if args.render_only else "NvidiaTeslaT4x2"], check=True)
    cfg.unlink()


if __name__ == "__main__":
    main()
