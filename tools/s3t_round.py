#!/usr/bin/env python3
"""Build the S3T-DETR fine-tuning kernel: MAE-pretrained spectral Transformer
in front of a COCO-pretrained RT-DETR-L, trained on 2x T4 with augmentation.

    python3 tools/s3t_round.py --slug <account>/hod26-s3t-detr
    kaggle kernels push -p kernels/s3t_detr/build

Inputs the kernel needs attached:
  - dataset  xishengfeng/hod26-planar                  (the frames)
  - notebook qwyi123/hod26-s3t-mae-pretrain output    (pretrain_mae.pt)
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
MAE_KERNEL = "qwyi123/hod26-s3t-mae-pretrain"
TOTAL = 16            # epochs the schedule is laid out over
SESSION_HOURS = 11.0

AUGMENT = {"sg_window": 7, "sg_polyorder": 2, "sg_chain": True,
           "smote_alpha": 0.3, "cutmix_prob": 0.4, "cutmix_blocks": 24, "copies": 1}


def s3t_candidate(total: int = TOTAL) -> dict:
    cand = full_candidate("transformer", total, {
        "channels.mode": "s3t_level",
        "train.spectral_stem": "s3t",
        # The encoder runs at native cube scale on top of an fp32 RT-DETR at
        # 1024; 2 per card keeps the pair inside a T4. nbs stays 64, so the
        # optimizer still steps on an effective batch of 64.
        "train.batch": 2,
    })
    cand["train"].update(s3t_scale=0.5, s3t_require_pretrain=True, s3t_widen=True, s3t_context=True,
                         mosaic=1.0, fliplr=0.5, scale=0.5, close_mosaic=3)
    cand["augment"].update(AUGMENT)
    return cand


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="TEAMMATE/hod26-s3t-detr")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--total", type=int, default=TOTAL)
    ap.add_argument("--mae-kernel", default=MAE_KERNEL)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = args.out_dir / "round-config.json"
    cfg.write_text(json.dumps({"round": "hod26-s3t-detr", "candidates": [], "submit": {
        "candidate": s3t_candidate(args.total),
        "use_all_train": False,          # keep the 600 held out: they are the ruler
        "predict": True,                 # a submission comes out wherever the clock stops
        "session_hours": SESSION_HOURS,
        "require_gpus": 2,
    }}, indent=2))
    subprocess.run([sys.executable, str(REPO / "tools" / "build_kernel.py"),
                    "--round-config", str(cfg), "--out-dir", str(args.out_dir),
                    "--slug", args.slug, "--kernel-source", args.mae_kernel,
                    "--machine-shape", "NvidiaTeslaT4x2"], check=True)
    cfg.unlink()


if __name__ == "__main__":
    main()
