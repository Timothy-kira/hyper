# HOD26 — running the long training on your account

You are on team **Tims** in the [Hyperspectral Object Detection Challenge 2026](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026).
Deadline **2026-09-24 16:00 UTC**.

Everything is prepared. You paste one file into a notebook and press Run,
twice, a few hours apart. You do not need to read or understand the code, and
you never need to hand anyone your credentials — Kaggle charges GPU time to
whoever starts the session, so this only works with you pressing the button,
and that is the whole reason this document exists.

## Why it is worth your GPU hours

Our best submission is **0.62613**, rank 58 of 251. First place is 0.67943.

Every run so far restarted from scratch because of a bug in how sessions
handed checkpoints to each other, so the best model we have is a single
19-epoch run. The measured curve, from three independent runs:

| epochs | leaderboard |
| --- | --- |
| 9 | 0.59680 |
| 10 | 0.59783 |
| 19 | 0.62584 |

That is **+0.0031 per epoch and still not flattening at 19**. A single
uninterrupted 38-epoch run extrapolates to roughly **0.68**, which is the top
of the leaderboard. The bug is fixed and verified; what is missing is only the
GPU hours, and ours do not refresh until 09-26, two days after the deadline.

## What you need first

1. **Check your GPU allowance.** kaggle.com → your avatar → Settings, or the
   gauge on any notebook's right-hand panel. **You need about 23 hours.** If
   you have less, say so before starting — a partial run is still useful, but
   we would size it differently.
2. **Accept the dataset invitation.** The 6.1 GB dataset `hod26-planar` is
   private on xishengfeng's account; he is adding you as a collaborator. Open
   <https://www.kaggle.com/datasets/xishengfeng/hod26-planar> and confirm you
   can see it. **If that page 404s, stop** — nothing below will work, and the
   notebook will burn a few minutes before telling you so.

## Session A — about 11 hours

1. kaggle.com → **Create → New Notebook**.
2. **File → Import Notebook**, and upload `hod26-team-a/hod26_round.py`.
   (Or: make one code cell and paste the whole file into it.)
3. Right-hand panel → **Input → Add Input → Datasets** → search
   `hod26-planar` → Add.
4. Right-hand panel → **Session options → Accelerator → GPU T4 x2**.
5. Same panel → **Internet → On**. Required: the COCO pretrained weights are
   downloaded at startup, and without them the model trains from random
   initialisation and the whole session is wasted.
6. **Save Version → Save & Run All (Commit)**. Close the tab; it runs in the
   background.

Name the notebook **`hod26-team-a`** exactly. Session B looks for it by name.

**Within the first minute** the log prints either `preflight passed` or a list
of `PREFLIGHT FAILED` lines naming exactly what is wrong. If it fails, nothing
has been spent — fix what it names and commit again. It checks the GPU is
really on, the library version, that the dataset is mounted *and* has the
right 3000/3000/1000 file counts, that the COCO weights downloaded, and the
free disk.

After that it prints one line per epoch, roughly every 35 minutes:

```
epoch 7/38  loss 0.299/0.390/0.071  mAP50 0.9395  mAP50-95 0.6690  lr 4.21e-04  ...
```

It stops itself cleanly at about 11 hours — you should see
`stopping cleanly at epoch N` — and saves its checkpoint. **Do not stop it
early**, and do not commit a second version while it is running: a new commit
restarts the run from the beginning and the hours do not come back.

Expect it to reach roughly **epoch 19**.

## Session B — about 11 hours, after A finishes

Same as above with three differences:

1. Use `hod26-team-b/hod26_round.py`.
2. Name it **`hod26-team-b`**.
3. **One extra input**: Add Input → **Notebook Output** → select your finished
   `hod26-team-a`. This is what lets it continue rather than start over.

Confirm in the log, in the first two minutes:

```
resuming from /kaggle/input/.../final_last.pt -> ...
training starts at epoch 20 of 38 (resume=True)
```

If it instead says `epoch 1` or `resume=False`, **stop the session
immediately** and tell us — it would otherwise spend eleven hours retraining
what session A already did. If session A's output was not attached at all, it
refuses to start and says so in the first minute.

When this one reaches epoch 38 it also predicts the test set and writes
`submission.csv` to its output.

## Handing it back

Once `hod26-team-b` is finished:

1. Right-hand panel → **Share → Collaborators** → add **xishengfeng** (Can
   view is enough), **or** just tell him the notebook name — teammates can read
   each other's work.
2. Tell him it is done. He downloads `submission.csv` from the notebook's
   Output tab and submits it from his account. Submissions are shared across
   the team, so it does not matter which of you submits; we have 3 per day.

If B stopped short of epoch 38 it will not have written a submission. That is
fine — its checkpoint is still the furthest the run has ever got, and he can
produce the submission from it on his own allowance in about ten minutes.

## If something goes wrong

- **`PREFLIGHT FAILED: dataset not found`** — the collaborator invitation has
  not been accepted, or the dataset was not attached in step 3.
- **`PREFLIGHT FAILED: no GPU visible`** — accelerator is still off.
- **`could not fetch rtdetr-l.pt`** — Internet is off.
- **`PREFLIGHT FAILED: require_resume`** (session B) — session A's output was
  not attached as an input.
- **Notebook killed around 12 hours** — Kaggle's hard cap. The clock guard
  should stop it at 11; if it was killed instead, the checkpoint is lost and
  that session's hours are gone. Tell us rather than re-running blind.

Anything else: send the last 40 lines of the log. Do not re-run a failed
session without asking — each attempt is hours off a budget that does not
refresh before the deadline.
