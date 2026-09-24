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
TOTAL = 44            # epochs: the GPU smoke measures 0.550 s/it with the 512 loader (0.682
                      # at 1024). The last formal run's epoch took 1.29x its smoke's s/it x
                      # 1200 its (validation, loader); at that ratio ~850 s/epoch, 44 epochs
                      # ~10.4 h -- inside the 11 h session with the prediction reserve, so
                      # the cosine schedule and the 3 closing no-mosaic epochs always run
SESSION_HOURS = 11.0

AUGMENT = {"sg_window": 7, "sg_polyorder": 2, "sg_chain": True,
           "smote_alpha": 0.3, "cutmix_prob": 0.4, "cutmix_blocks": 24, "copies": 1}


# Staged unfreezing (handoff/S3T.md, "keeping the pretrained weights"), per
# part: [first epoch (0-based), ramp epochs, LR multiplier]; a part left out
# never trains. First the weights that say what and where (the heads) and the
# modules that start at zero; then the pretrained decoder, neck and MAE encoder,
# ramped in while the warmup is still raising the LR; then the COCO backbone at
# a tenth of the LR, as the official RT-DETR / D-FINE fine-tuning configs do.
# The stem and the backbone's BatchNorm stay frozen (freeze_norm there too).
UNFREEZE = {
    "head": [0, 0, 1.0], "new": [0, 0, 1.0], "mixer": [0, 0, 1.0],
    "decoder": [2, 2, 1.0], "neck": [2, 2, 1.0], "s3t_enc": [2, 2, 1.0],
    "backbone": [5, 3, 0.1],
}


# The four classes that score lowest (0.23-0.55 against 0.72-0.79), and why
# (handoff/DIAGNOSIS.md, "S3T-X"): grey -- their spectrum is the background's,
# uniformly darker -- which a 3-channel projection mostly discards; and the only
# classes that crowd and occlude one another.
WEAK = ["stone_block", "people", "e-bike", "car"]
# Fine-tune from the finished S3T-X model: everything trains from the first
# epoch, at a third of the from-COCO LR (backbone at a tenth of that); the
# stem's widened first conv (its 16 new band channels start at zero) at half.
FT_UNFREEZE = {**{pt: [0, 0, 0.33] for pt in ("head", "new", "mixer", "decoder", "neck", "s3t_enc")},
               "backbone": [0, 0, 0.033], "stem_in": [0, 0, 0.5]}


def finetune_candidate(cand: dict, init_from: str, epochs: int = 15) -> dict:
    """The S3T-X candidate as a fine-tune aimed at the four weak classes.

    S1  the stem reads all 16 bands beside the 3 projected channels;
    C1  box loss x2 on those classes' matched boxes;
    C2  RepGT repulsion off neighbouring ground truth;
    B2  crowd copy-paste of their instances into street frames.
    """
    tr = cand["train"]
    tr.update(epochs=epochs, schedule_epochs=epochs, init_from=init_from, s3t_stem_bands=True,
              box_cls_gain={c: 2.0 for c in WEAK}, rep_gain=0.5,
              unfreeze={k: list(v) for k, v in FT_UNFREEZE.items()},
              warmup_epochs=1.0, close_mosaic=3)
    cand["augment"].update(crowd_paste=3, crowd_paste_p=0.7, paste_classes=list(WEAK), paste_margin=4)
    return cand


def plain_finetune_candidate(cand: dict, init_from: str, epochs: int) -> dict:
    """The S3T-X candidate unchanged, warm-started from a finished model: only
    the data differs (the HOT2024 frames), so any change is the data's."""
    tr = cand["train"]
    tr.update(epochs=epochs, schedule_epochs=epochs, init_from=init_from,
              unfreeze={k: list(v) for k, v in FT_UNFREEZE.items() if k != "stem_in"},
              warmup_epochs=1.0, close_mosaic=2)
    return cand


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
        # The loader was the bottleneck (4 vCPUs; the GPUs sat at 53-61% in the
        # first formal epoch): a 16-channel mosaic at 1024 costs 145 ms/sample
        # on one core, at 512 41 ms. The cubes are 493x241, so a 512 loader
        # keeps every native pixel; the front upsamples 2x on the GPU and the
        # detector still sees 1024 (the encoder reads the 512 input directly).
        cand["train"].update(imgsz=512, s3t_upsample=2)
    # Head and losses (handoff/S3T.md, "head"): D-FINE's distribution refinement
    # on the pretrained decoder, DEIM's MAL for the classes, log-space w/h L1.
    # Mosaic already gives DEIM's dense one-to-one supervision.
    cand["train"].update(fdr=True, mal=True, log_size_l1=True)
    # Keep COCO and the MAE pretraining: staged unfreezing, and BatchNorm on
    # COCO's running statistics (2 images/card train-mode statistics are noise).
    cand["train"].update(unfreeze={k: list(v) for k, v in UNFREEZE.items()}, frozen_bn=True)
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


def _candidate(args) -> dict:
    cand = s3t_candidate(args.total, args.batch,
                         args.mae_file or ("pretrain3_mae.pt" if args.arch == "xca" else None),
                         args.compile_blocks, args.arch)
    if args.finetune_from and args.plain_finetune:
        cand = plain_finetune_candidate(cand, args.finetune_from, args.total)
    elif args.finetune_from:
        cand = finetune_candidate(cand, args.finetune_from, args.total)
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
    ap.add_argument("--no-smoke", action="store_true",
                    help="skip the ~4 min smoke at the start of the long run -- for code a "
                         "--smoke-only run has just proven")
    ap.add_argument("--render-kernel", default=None,
                    help="a render notebook to mount; its dataset is used instead of rendering")
    ap.add_argument("--finetune-from", default=None,
                    help="fine-tune for the weak classes from this checkpoint file (e.g. final_best.pt)")
    ap.add_argument("--finetune-kernel", default=None,
                    help="the kernel whose output holds --finetune-from")
    ap.add_argument("--plain-finetune", action="store_true",
                    help="with --finetune-from: the S3T-X recipe as it is (no S1/C1/C2/B2)")
    ap.add_argument("--extra-dataset", default=None,
                    help="Kaggle dataset with the HOT2024 frames (hot24_index.json), labelled "
                         "by the --finetune-from model and added to the training split")
    ap.add_argument("--extra-per-video", type=int, default=0,
                    help="frames per HOT video to use (0: all in the dataset)")
    args = ap.parse_args()
    if args.finetune_from and not (args.finetune_kernel or args.render_only):
        ap.error("--finetune-from needs --finetune-kernel (the kernel whose output holds it)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = args.out_dir / "round-config.json"
    cfg.write_text(json.dumps({"round": "hod26-s3t-detr", "candidates": [], "submit": {
        "candidate": _candidate(args),
        "use_all_train": False,          # keep the 600 held out: they are the ruler
        "predict": True,                 # a submission comes out wherever the clock stops
        "session_hours": SESSION_HOURS,
        "require_gpus": 2,
        "render_only": bool(args.render_only),
        "smoke_only": bool(args.smoke_only),
        "smoke": not args.no_smoke,
        **({"extra_data": {"per_video": args.extra_per_video, "hi": 0.6, "lo": 0.3,
                           **({"limit": 16} if args.smoke_only else {})}}
           if args.extra_dataset else {}),
    }}, indent=2))
    subprocess.run([sys.executable, str(REPO / "tools" / "build_kernel.py"),
                    "--round-config", str(cfg), "--out-dir", str(args.out_dir),
                    "--slug", args.slug,
                    *([] if args.render_only else
                      [a for k in (args.mae_kernel or [MAE_KERNEL]) for a in ("--kernel-source", k)]),
                    *(["--kernel-source", args.render_kernel] if args.render_kernel else []),
                    *(["--kernel-source", args.finetune_kernel]
                      if args.finetune_kernel and not args.render_only else []),
                    *(["--dataset-source", args.extra_dataset] if args.extra_dataset else []),
                    "--machine-shape", "cpu" if args.render_only else "NvidiaTeslaT4x2"], check=True)
    cfg.unlink()


if __name__ == "__main__":
    main()
