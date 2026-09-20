# The run — `qwyi123/hod26-team`

Live record of the handed-off session, kept so the last line of the log is not
the only thing that survives it.

## What is running

- **Started** 2026-09-20 16:06 UTC, account `qwyi123`, on **2x Tesla T4**.
- **Budget** `session_hours: 11.0` from script start, 30 min reserved for
  prediction — but at the measured pace the run finishes long before the cap
  binds.
- It replaced a single-card session (13:27–16:07, reached epoch ~26) that was
  cancelled to switch to both cards. Kaggle only saves `/kaggle/working` when a
  kernel exits cleanly, so that session left nothing behind; this one resumes
  from the same epoch-21 dataset checkpoint.

## Preflight — passed

```
preflight ok: 2x GPU Tesla T4
preflight ok: ultralytics 8.4.155
preflight ok: data .../hod26-planar: 3000 train / 3000 xml / 1000 test
preflight ok: rtdetr-l.pt 67 MB
preflight ok: resume from .../hod26-ckpt-s4/final_last.pt
preflight passed
submission fit: 2400 train / 600 val
```

```
[16:22:37]   2 GPU(s) visible; DDP across [0, 1], batch 8 (4/card)
[16:23:16]   training starts at epoch 22 of 40 (resume=True)
[16:23:17]   training starts at epoch 22 of 40 (resume=True)
```

Two `training starts` lines is one per rank and is the check that both cards
are live. One line means the run is on a single GPU.

## Pace

Measured over two full smoke runs of this kernel: **1041–1089 s per epoch on
two cards** against **2002 s on one** — **1.84–1.92x**. Per-card batch is
unchanged at 4, so each GPU does the work one card used to and the optimizer
still steps at `nbs=64`; only the wall clock moves.

Training opened at 16:23:16, so epochs 22 to 40 is about **5.2 hours** and the
run should reach 40 and predict by roughly **21:45 UTC** — against 00:32 and
epoch 38-39 for the single-card session it replaced.

## What the smoke runs caught

Both were the real kernel, cut to one epoch by the clock guard.

1. **`AttributeError: 'dict' object has no attribute 'box'`**, on the line
   after training. Under DDP the parent's validator never runs, so ultralytics
   returns the checkpoint's `train_metrics` — a flat dict — where a
   single-GPU run returns the metrics object. It lands *after* all the
   training, and a kernel that ends in a traceback does not keep
   `/kaggle/working`.
2. **The holdout number was the one it inherited.** The run signed off with
   `holdout mAP=0.7271` while its own re-validation, three lines earlier, said
   `0.6985`. 0.7271 is what `best.pt` stores, written by the session that had
   folded the validation frames into training.

Both are fixed and covered by `tests/test_ddp_scores.py`, which runs off the
metrics file the smoke run produced and needs no GPU.

## Watching it

```bash
export KAGGLE_API_TOKEN=...      # the account that started the run
python3 tools/watch_run.py qwyi123/hod26-team --seconds 120
```

## Epochs

| epoch | mAP50 | mAP50-95 | wall |
| --- | --- | --- | --- |
| 22 | pending | pending | |
