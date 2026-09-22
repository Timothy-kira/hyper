# HOD26 — round two of the same run, on a fresh account

You are on team **Tims** in the [Hyperspectral Object Detection Challenge 2026](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026).
Deadline **2026-09-24 16:00 UTC** — about two days out as this is written.

One notebook, one Run, about eleven hours. You do not need to read the code,
and nobody needs your credentials — Kaggle charges GPU time to whoever starts
the session, so it only works with you pressing the button.

**This exact resume point (epoch 21 → 40) has already been run once**, on
another teammate's account, on two T4s. Read this section before starting —
it corrects a prediction the first version of this document made, and that
prediction turned out to be wrong.

## What this is worth, and what we now know

Our best submission is **0.62994**. Leaderboard first place is 0.67943.

The original plan for this handoff extrapolated **+0.0031/epoch** from three
early runs (9, 10, 19 epochs → 0.59680, 0.59783, 0.62584) and predicted that
epoch 22 → 40 would land near **0.69**. It did not. The run happened, on two
T4s, epoch 21 → 40, and the held-out mAP50-95 went **0.6938 at epoch 22 →
0.6941 at epoch 40 — a net of 0.0003 over eighteen epochs**, against a
measured run-to-run noise floor of about 0.0017. The curve had flattened well
before epoch 21; the early slope was not representative of the epochs this
handoff buys. **Do not expect this run to move the score much on its own.**

A follow-up twelve-epoch fine-tune from that same epoch-40 checkpoint, with
four loss/sampling changes aimed at a diagnosed bottleneck (see below), landed
at **0.62994** — +0.0004 over the plain epoch-40 result, inside the noise
floor. Worth stating plainly: **the four classes the changes targeted moved
net *negative*** (stone_block +0.001, people −0.009, e-bike +0.021, car
−0.025 — sum −0.013), while the macro number moved up by noise in classes
that were not the target. Twelve epochs at a low, near-finished learning rate
did not show the intended effect on the classes it was aimed at. That is a
real result, not a null one — it says this combination, at this length and
this point in the schedule, is not the fix, not that the diagnosis is wrong.

**The diagnostic behind that fine-tune is solid, even though the fine-tune's
result is not.** A CPU-only error decomposition (`handoff/DIAGNOSIS.md`, no
GPU quota, ~1 minute) on the held-out predictions found: 98.3% of ground truth
is found, 99.9% of what is found is named correctly. The entire remaining gap
is **box tightness** — median matched IoU 0.8697 — concentrated in four
classes (`stone_block`, `people`, `e-bike`, `car`) that are elongated and/or
rare, not small. More epochs of the plain recipe will not fix that; a
different loss shape or more of those classes' frames might. Read
`handoff/DIAGNOSIS.md` before trying anything past a plain resume.

**Two things were tried after the fine-tune and should not be repeated
blind.** A test-time-augmentation (hflip) + weighted-box-fusion pass raised
the *held-out* score (0.6956) but **collapsed the leaderboard score to
0.58168** — held-out and leaderboard disagreeing in opposite directions like
that means a bug in the TTA/merge path for the real test set, not a real
capability loss; it was not resolved before quota ran out. Separately, an
attempt to add a spectral contrastive loss for the three hardest classes
(never merged into this repo) crashed at its first validation with a loss-key
mismatch, after burning real training epochs. Neither is in the script below.

**Why this needs a fresh account.** The account that ran epoch 22 → 40 and
the follow-up fine-tune has 1.71 of its 30 GPU-hours left, and Kaggle's weekly
allowance does not refresh until 09-26 — after the deadline. That account is
done. This resumes from the **same epoch-21 checkpoint as the first handoff**,
not the further-trained epoch-40/52 checkpoint that produced 0.62994 — that
checkpoint is a private kernel output on the other account and was not
repackaged as a shared dataset. If you would rather resume from it than redo
epochs 22 → 40 (which, per above, does not move the score much by itself),
ask before starting; it is a five-minute repackage, not a retraining.

## Before you start

1. **Check your GPU allowance** — kaggle.com → avatar → Settings, or the gauge
   on any notebook's right panel. **You need about 12 hours.** Less than that,
   tell us before starting and we will resize the run.
2. **Confirm you can open both datasets.** xishengfeng is adding you as a
   collaborator on each; they are private, so a 404 means the invitation has
   not landed yet.
   - <https://www.kaggle.com/datasets/xishengfeng/hod26-planar> (6.1 GB, the frames)
   - <https://www.kaggle.com/datasets/xishengfeng/hod26-ckpt-s4> (330 MB, the checkpoint)

   **If either 404s, stop.** Nothing below works without both.

## Running it

1. kaggle.com → **Create → New Notebook**.
2. Get the script. It is 135 kB, so downloading beats pasting — a cell that
   large tends to lag the editor or truncate:

   <https://raw.githubusercontent.com/Timothy-kira/hyper/claude/kaggle-cli-setup-handoff-9w0v7x/handoff/hod26_round.py>

   (Not `-ppjny1` — that branch is frozen before the dual-GPU fixes below existed
   and will silently hand you a single-card, un-fixed script.)

   Save it, then **File → Import Notebook** and upload it. (If import gives
   you trouble: one code cell, paste the whole file in, nothing else.)
3. Right panel → **Input → Add Input → Datasets**, and add **exactly these two**:
   `hod26-planar` and `hod26-ckpt-s4`.

   If you see an older `hod26-ckpt-s2` listed, **do not add it**. Two attached
   checkpoints and the run would resume from whichever sorts first, which is
   the wrong one. It refuses to start rather than guess, but it is simpler not
   to attach it.
4. Right panel → **Session options → Accelerator → GPU T4 x2**. Both cards
   are now used, not just one: the run trains under DDP at 4 images per card.
   If the session comes up with a single T4 the preflight stops it in the
   first minute rather than spending the whole allowance at half speed, so a
   wrong accelerator costs nothing but a re-commit.
5. Same panel → **Internet → On**. Required: the COCO pretrained weights are
   fetched at startup, and without them the model trains from random
   initialisation and the whole session is wasted.
6. **Save Version → Save & Run All (Commit)**. Close the tab; it runs in the
   background.

### The first two minutes tell you everything

You want to see these two things:

```
preflight ok: 2x GPU Tesla T4
preflight passed
resuming from /kaggle/input/hod26-ckpt-s4/final_last.pt -> ...
2 GPU(s) visible; DDP across [0, 1], batch 8 (4/card)
training starts at epoch 22 of 40 (resume=True)
training starts at epoch 22 of 40 (resume=True)
```

`training starts` appearing **twice is correct** — one line per card. Seeing it
once means the run is on a single GPU.

- If it prints `PREFLIGHT FAILED` lines instead, **nothing has been spent**.
  Each line names exactly what is wrong; fix it and commit again.
- If it says **`epoch 1`** or **`resume=False`**, stop the session immediately
  and tell us. It would spend eleven hours retraining what we already have.

Then one line per epoch, roughly every 35 minutes:

```
epoch 24/40  loss 0.201/0.275/0.044  mAP50 0.9438  mAP50-95 0.6903  lr 1.65e-04  ...
```

`mAP50-95` is the number that matters.

**Expect the first one to read about 0.69, not the 0.727 in the checkpoint's
own history. That drop is correct and you should not report it as a problem.**
The previous session had folded the 600 validation frames into its training
set, which inflates the score by construction; this run holds them out again,
so the number goes back to measuring something real.

**From there, expect it to barely move.** The last time this exact resume ran,
epoch 22 read 0.6938 and epoch 40 read 0.6941 — see "What this is worth"
above. A number that sits flat in the high 0.69x range for most of the run is
the measured outcome, not a sign anything is broken.

### While it runs

- **Do not commit a second version.** A new commit restarts the run from the
  beginning and the hours do not come back.
- **Do not stop it early.** It stops itself at about eleven hours, prints
  `stopping cleanly at epoch N`, then predicts the test set and writes
  `submission.csv`. It reserves thirty minutes for that, so a submission comes
  out wherever the clock stops it.

## When it finishes

1. Right panel → **Share → Collaborators** → add **xishengfeng** (Can view is
   enough), or just tell him the notebook name.
2. He takes `submission.csv` from the Output tab and submits it. Submissions
   are shared across the team, so it does not matter who submits; we have 3 a
   day.

Send us the last line of the log either way — `fit done at epoch N/40;
holdout mAP=...` tells us where the run got to and what it is worth.

## If something goes wrong

| what you see | what it means |
| --- | --- |
| `PREFLIGHT FAILED: dataset not found` | `hod26-planar` not attached, or the invitation not accepted |
| `PREFLIGHT FAILED: require_resume` | `hod26-ckpt-s4` not attached — step 3 |
| `PREFLIGHT FAILED: more than one checkpoint` | both `-s2` and `-s4` attached; remove `-s2` |
| `PREFLIGHT FAILED: no GPU visible` | accelerator still off |
| `PREFLIGHT FAILED: asked for 2 GPUs and got 1` | accelerator is on but set to a single T4; switch it to **GPU T4 x2**. Nothing has been spent |
| `could not fetch rtdetr-l.pt` | Internet off |
| `expected 3000 train / 3000 xml / 1000 test` | the dataset mounted but is incomplete; tell us |
| killed at ~12 hours | Kaggle's hard cap; the guard should have stopped it at 11. The checkpoint is lost. Tell us rather than re-running |

Anything else: send the last 40 lines of the log. **Do not re-run a failed
session without asking** — each attempt is hours off an allowance that does
not refresh before the deadline.
