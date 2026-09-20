# HOD26 — one 11-hour run on your account

You are on team **Tims** in the [Hyperspectral Object Detection Challenge 2026](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026).
Deadline **2026-09-24 16:00 UTC**.

One notebook, one Run, about eleven hours. You do not need to read the code,
and nobody needs your credentials — Kaggle charges GPU time to whoever starts
the session, so it only works with you pressing the button.

## What this is worth

Our best submission is **0.62613**, rank 58 of 251. First place is 0.67943.

Every run so far restarted from scratch because of a bug in how sessions
handed checkpoints to each other, so our best model is a single 19-epoch run.
Three independent runs give the curve:

| epochs | leaderboard |
| --- | --- |
| 9 | 0.59680 |
| 10 | 0.59783 |
| 19 | 0.62584 |

**+0.0031 per epoch, and still not flattening at 19.** Reaching 38 extrapolates
to roughly **0.68**, the top of the leaderboard.

**You are not training from scratch.** Our epoch-19 checkpoint is attached, so
your eleven hours buy epochs **20 to 38** — half the compute for the same
finish. Our own allowance does not refresh until 09-26, two days after the
deadline, which is why this cannot be run on our side.

## Before you start

1. **Check your GPU allowance** — kaggle.com → avatar → Settings, or the gauge
   on any notebook's right panel. **You need about 12 hours.** Less than that,
   tell us before starting and we will resize the run.
2. **Confirm you can open both datasets.** xishengfeng is adding you as a
   collaborator on each; they are private, so a 404 means the invitation has
   not landed yet.
   - <https://www.kaggle.com/datasets/xishengfeng/hod26-planar> (6.1 GB, the frames)
   - <https://www.kaggle.com/datasets/xishengfeng/hod26-ckpt-s2> (330 MB, the checkpoint)

   **If either 404s, stop.** Nothing below works without both.

## Running it

1. kaggle.com → **Create → New Notebook**.
2. **File → Import Notebook** and upload `hod26_round.py`. (Or make one code
   cell and paste the whole file into it.)
3. Right panel → **Input → Add Input → Datasets**, and add **both**:
   `hod26-planar` and `hod26-ckpt-s2`. Missing the second one means it has
   nothing to resume from and it will refuse to start.
4. Right panel → **Session options → Accelerator → GPU T4 x2**.
5. Same panel → **Internet → On**. Required: the COCO pretrained weights are
   fetched at startup, and without them the model trains from random
   initialisation and the whole session is wasted.
6. **Save Version → Save & Run All (Commit)**. Close the tab; it runs in the
   background.

### The first two minutes tell you everything

You want to see these two things:

```
preflight passed
resuming from /kaggle/input/hod26-ckpt-s2/final_last.pt -> ...
training starts at epoch 20 of 38 (resume=True)
```

- If it prints `PREFLIGHT FAILED` lines instead, **nothing has been spent**.
  Each line names exactly what is wrong; fix it and commit again.
- If it says **`epoch 1`** or **`resume=False`**, stop the session immediately
  and tell us. It would spend eleven hours retraining what we already have.

Then one line per epoch, roughly every 35 minutes:

```
epoch 24/38  loss 0.201/0.275/0.044  mAP50 0.9438  mAP50-95 0.6840  lr 1.65e-04  ...
```

`mAP50-95` is the number that matters. It should be near **0.687** at epoch 20
and climbing.

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

Send us the last line of the log either way — `fit done at epoch N/38;
holdout mAP=...` tells us where the run got to and what it is worth.

## If something goes wrong

| what you see | what it means |
| --- | --- |
| `PREFLIGHT FAILED: dataset not found` | `hod26-planar` not attached, or the invitation not accepted |
| `PREFLIGHT FAILED: require_resume` | `hod26-ckpt-s2` not attached — step 3, the second dataset |
| `PREFLIGHT FAILED: no GPU visible` | accelerator still off |
| `could not fetch rtdetr-l.pt` | Internet off |
| `expected 3000 train / 3000 xml / 1000 test` | the dataset mounted but is incomplete; tell us |
| killed at ~12 hours | Kaggle's hard cap; the guard should have stopped it at 11. The checkpoint is lost. Tell us rather than re-running |

Anything else: send the last 40 lines of the log. **Do not re-run a failed
session without asking** — each attempt is hours off an allowance that does
not refresh before the deadline.
