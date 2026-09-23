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

ultralytics' own validation on the held-out 600. The leaderboard's scorer
reads about 0.05 below this, which is why 0.69 here sits beside 0.626 there.

| epoch | mAP50 | mAP50-95 | wall |
| --- | --- | --- | --- |
| 22 | 0.9488 | 0.6938 | 1101 s |
| 23 | 0.9446 | 0.6891 | 1100 s |
| 24 | 0.9434 | 0.6879 | 1100 s |
| 25 | 0.9450 | 0.6915 | 1100 s |
| 26 | 0.9452 | 0.6891 | 1100 s |
| 27 | 0.9470 | 0.6918 | 1100 s |
| 28 | 0.9474 | 0.6940 | 1100 s |
| 29 | 0.9474 | 0.6941 | 1099 s |
| 30 | 0.9470 | 0.6947 | 1099 s |
| 31 | 0.9459 | 0.6939 | 1100 s |

Four points cannot separate "flat" from the +0.002/epoch the earlier sessions
ran at (epoch 10 → 19 went 0.665 → 0.687) against noise of about +/-0.003, so
they are consistent with the run still improving slowly. They are not evidence
that it has stopped.

## The checkpoint this run predicts from

`best.pt` is only rewritten when an epoch beats the fitness stored in the
checkpoint the run resumed from, and ultralytics restores that number on
resume. Ours was recorded at **0.727** by the session that had folded the 600
validation frames into its training set; this run holds them out and scores
about 0.69. The bar therefore sits above anything this run prints, `best.pt`
keeps the weights it arrived with, and the session's own `submission.csv`
would come from epoch 21 — the model already on the leaderboard at 0.62613.

The fix (clearing `best_fitness` on resume) is in the driver but **not in the
running session**: the push was refused with `Maximum batch GPU session count
of 2 reached` and restarting would have cost the epochs already done.

So the finish is a separate step, which costs about eight GPU-minutes and
needs no restart:

```bash
python3 tools/predict_from_run.py --source qwyi123/hod26-team \
    --weights final_last.pt --message "..." --no-submit
```

`final_last.pt` is the weights the run actually produced. The same kernel
scores the held-out 600 with pycocotools — the leaderboard's own ruler — so
the submission's worth is known before it is spent against the daily three.

## The leftover session

The v2 push at 16:06 started a new session without stopping the one from 13:27.
`kaggle kernels status` reports only a kernel's *latest* session, so the older one
is invisible to it, and two independent signals are what found it:

- a push at 19:42 refused with `Maximum batch GPU session count of 2 reached`,
  while only one kernel reported RUNNING;
- quota burning at ~2.0 GPU-h per wall hour with, nominally, one session running.

That second signal also settles how Kaggle bills: **each session costs 1x its own
wall clock, whatever its GPU count.** The 15:04-16:03 window decides it — a
dual-card kernel ran 38 minutes there alongside a single-card one, and 2x-for-dual
predicts 2.74 GPU-h against the 1.82 observed, where 1x predicts 1.86. Dual-card is
a 1.92x speedup at no extra cost; the doubled burn is the leftover, not the second
GPU.

It cannot be stopped from here: `cancel_kernel_session` needs a session id no public
endpoint returns, and guessing one could cancel the wrong session. Its own 11-hour
guard stops it around 00:32, and it reaches epoch ~40 on the way — a second,
independent epoch-40 model, bought at about 5 GPU-h.

## Finishing

Neither running session carries the `best_fitness` fix, so both write a
`submission.csv` predicted from the inherited epoch-21 `best.pt` — the model already
on the leaderboard at 0.62613. Neither is worth submitting.

The submission comes from `final_last.pt` instead, through
`tools/predict_from_run.py`, which also scores the held-out 600 with pycocotools so
the number is known before a submission is spent against the daily three.

## Outcome

`fit done at epoch 40/40; holdout mAP=0.6985` — and that number is `best.pt`'s,
which is the epoch-21 checkpoint the run resumed from. It was never rewritten: the
inherited `best_fitness` of 0.727 was measured on a validation split that session had
trained on, and no honest epoch of this run came near it. So the session's own
`submission.csv` was predicted from epoch-21 weights, exactly as predicted, and was
not submitted.

The submission came from `final_last.pt` through `tools/predict_from_run.py`:

| | held-out 600 (pycocotools) | leaderboard |
| --- | --- | --- |
| epoch 19, session 2 | 0.6873 (ultralytics) | 0.62584 |
| epoch 21, session 4 | — | 0.62613 |
| **epoch 40, session 5** | **0.6943** | **0.62954** |

Eighteen epochs of two-card training bought **+0.0034** on the leaderboard, against a
noise floor of 0.001. Real, but a twentieth of what the handoff's +0.0031/epoch
extrapolation promised.

The per-epoch curve says why. Epoch 22 scored 0.6938 and epoch 40 scored 0.6941 —
eighteen epochs, net 0.0003. Closing mosaic at 36, which usually steps the curve up,
did nothing here (0.6971 at 35, 0.6943 at 36). The best epoch of the whole run was
35, not 40.

Two calibrations worth keeping:

- pycocotools and ultralytics agree on this checkpoint (0.6943 vs 0.6941), and
  `maxDets` 100 and 300 also agree. The ~0.06 the README describes is not a
  difference between rulers — it is the held-out 600 against the test 1000.
- So held-out minus 0.065 is the leaderboard estimate: 0.6943 - 0.0648 = 0.62954,
  measured.

### Where the deficit is, at epoch 40

| class | AP (pycocotools) | at epoch 22 |
| --- | --- | --- |
| stone_block | 0.3221 | 0.315 |
| people | 0.4150 | 0.410 |
| e-bike | 0.4492 | 0.446 |
| car | 0.5895 | 0.581 |
| the other fourteen | 0.712-0.802 | 0.716-0.804 |

Eighteen epochs moved the four deficit classes by 0.003-0.009 each. Whatever is
holding them is not something more epochs of this recipe will fix.

## Session 6 — the four-lever fine-tune, and what it actually did

`qwyi123/hod26-combo`, pushed 23:42 UTC 2026-09-20, resumed session 5's
`final_last.pt` (epoch 40) for twelve more epochs (41 → 52) with four changes
at once: `log_size_l1=True`, `bbox_loss="DIoU"`, `loss_gain={"bbox":2,
"giou":5}`, `repeat_threshold=0.1`. `use_all_train=False` and `predict=False`,
so the held-out 600 stayed the ruler and nothing was spent on a submission
before it was scored. This run also carried the `best_fitness` fix
(`167db4f`) for the first time on a live session — the log said so directly:

```
best.pt selection restarts from this run's own scale (the checkpoint's
0.72706 was measured on a validation split it had trained on)
training starts at epoch 41 of 52 (resume=True)
```

`repeat sampling (t=0.1): +68 frames, stone_block x2.7, orange x1.4,
egg_wood x1.4, e-bike x1.3` — the rarity lever landed on the right class.

### Epoch trajectory (ultralytics' own validation)

| epoch | mAP50 | mAP50-95 |
| --- | --- | --- |
| 41 | 0.9477 | 0.6915 |
| 42 | 0.9492 | 0.6949 |
| 43 | 0.9487 | 0.6924 |
| 44 | 0.9465 | 0.6933 |
| 45 | 0.9450 | 0.6900 |
| 46 | 0.9473 | 0.6925 |
| 47 | 0.9486 | 0.6955 |
| 48 | 0.9481 | 0.6932 |
| 49 | 0.9489 | 0.6927 |
| 50 | 0.9492 | 0.6932 |
| 51 | 0.9460 | 0.6929 |
| 52 | 0.9440 | 0.6923 |

`fit done at epoch 52/52; holdout mAP=0.6953` — `best.pt`'s own number, this
time honestly this run's own best epoch rather than an inherited one.

### The submission

Scored separately with pycocotools (`tools/predict_from_run.py --weights
final_best.pt`, repo code, no TTA): **held-out mAP 0.69528**, submitted as
"hod26-combo final_best.pt epoch52 holdout0.69528" → **leaderboard 0.62994**,
a new team best.

| | held-out (pycocotools) | leaderboard |
| --- | --- | --- |
| session 5, epoch 40 | 0.6943 | 0.62954 |
| **session 6, epoch 52, four levers** | **0.69528** | **0.62994** |

Both moved up, both inside the noise floor (0.0017 held-out, ~0.001
leaderboard). So: not proven to help, and **did not lower the score**.

No clean per-class breakdown of this run exists. An earlier version of this
file had one, showing the four targeted classes moving net -0.0125 — that
table was wrong to include. It came from `kaggle kernels output
qwyi123/hod26-predict/2`, which silently returned the *latest* version (v3,
the supcon-branch TTA kernel) rather than v2; its log contains the TTA lines,
and v3's held-out is 0.6956, not v2's 0.69528. Those per-class numbers are the
supcon branch's pipeline, not this run's, and say nothing about the four
levers.

**Tooling note:** `kaggle kernels output <owner>/<slug>/<version>` does not
reliably fetch that version — check the downloaded log before trusting it.
`kaggle kernels pull` with a version suffix returns nothing at all.

The epoch-by-epoch numbers above come from this run's own training log, so
they are unaffected: 0.690-0.696 throughout, flat, no decline.

## What happened after session 6 (not in this repo, not committed)

Two more things were tried on the `qwyi123` account before quota ran out.
Neither is in this codebase, and neither should be repeated without more care.

**The 0.58168 submission.** `hod26-predict` v3 was built from the
`hod26-supcon` branch's driver — its source contains `bg_residual` (9
occurrences), `speccon` (23), `SupCon`, `smote_alpha_hard`; the repo's
generated kernel contains none of them — and added TTA (`id`, `hflip`, WBF
`iou=0.65`) to predict session 6's `final_best.pt`. It scored **0.58168**, a
0.048 drop from the same checkpoint's 0.62994.

The held-out 0.6956 in its log does not help locate the cause: it is scored
*before* the TTA step, without TTA, so it measures that branch's
preprocessing, not TTA. What is established is only that the drop came from
the supcon-branch prediction code; whether TTA/WBF, `bg_residual`, or another
change in that branch is responsible was not isolated. **Nothing from that
kernel is in this repo, and no committed change is implicated.**

**A spectral supervised-contrastive loss.** A from-scratch addition (kernel
`hod26-supcon`, never merged into this repo) attached an InfoNCE contrastive
loss to the adapter's ROI features for the three hardest classes
(`stone_block`, `people`, `e-bike`), alongside a new `bg_residual` channel
mode and hard-class-weighted SMOTE. It resumed from session 6's
`final_best.pt` with `ft_freeze="adapter_decoder"`. It trained for real
epochs, then crashed at its first validation pass:

```
File ".../ultralytics/engine/validator.py", line 259, in __call__
    self.loss[k] += v
KeyError: 'loss_giou'
```

The custom criterion's returned loss dict does not have the same keys the
validator's `loss_names` expects -- almost certainly because the added
contrastive term changed what `RTDETRDetectionLoss` (or the wrapper around
it) returns without updating the validator's key list to match, or a
mismatch between the training-time and eval-time key set for DETR's
per-layer auxiliary losses. It failed in DDP, both ranks, after real training
time -- burning close to the account's last GPU-hours. The idea (a
contrastive push on the adapter features for exactly the classes the
diagnosis names) is not dismissed, just unresolved; anyone picking it back up
should reproduce the crash on one card first, without DDP, to get a plain
traceback before touching either 16-band code or DDP's cloudpickle path.

## Final state, this account

- **Best submission: 0.62994** (session 6, epoch 52, held-out pycocotools
  0.6956). Banked, safe, already on the leaderboard.
- **Do not use** submission 56460799 (TTA, 0.58168) or anything from
  `hod26-supcon` (never finished).
- **GPU quota: 1.71 of 30 hours left**, refreshing 2026-09-26 -- after the
  competition deadline. This account is done; nothing further should be
  pushed from here.
- **Handoff:** `handoff/README.md` now points a fresh account back at the
  same epoch-21 checkpoint (`hod26-ckpt-s4`) this account started from, with
  every fix from sessions 5 and 6 (dual-GPU, `best_fitness` reset, honest
  holdout scoring, rank-0-only writers) baked into the script it downloads.
  It also carries the corrected expectation: this exact resume barely moves
  the score on its own (epoch 22 to 40 net +0.0003), so a fresh session
  should read `DIAGNOSIS.md` before deciding what, if anything, to change.
