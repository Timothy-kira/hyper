# The run — `qwyi123/hod26-team`

Live record of the handed-off session, kept so the last line of the log is not
the only thing that survives it.

- **Started** 2026-09-20 13:27 UTC, account `qwyi123`, one commit (Save & Run All).
- **Budget** `session_hours: 11.0` measured from script start (~13:32), with
  30 min reserved for prediction, so training is cut off around **00:02 UTC**
  and the session ends around **00:32 UTC 09-21**.

## Preflight — passed

```
preflight ok: GPU Tesla T4
preflight ok: ultralytics 8.4.155
preflight ok: data /kaggle/input/datasets/xishengfeng/hod26-planar: 3000 train / 3000 xml / 1000 test
preflight ok: rtdetr-l.pt 67 MB
preflight ok: resume from /kaggle/input/datasets/xishengfeng/hod26-ckpt-s4/final_last.pt
preflight ok: scratch /kaggle/temp 1100.2 GB free
preflight passed
submission fit: 2400 train / 600 val
```

The two things the README says to check both read right: `resumed 943/943
tensors from the checkpoint`, and

```
[13:46:36]   training starts at epoch 22 of 40 (resume=True)
```

Internet was on (`rtdetr-l.pt` fetched), one checkpoint attached, the 600
validation frames are held out of the 3000 again — so the first `mAP50-95` is
expected near 0.69, not the checkpoint history's 0.727.

## Pace

Epoch 22 ran at **1.6 s/it over 1200 steps** — about 32 min of training plus
validation, matching the ~35 min/epoch the README predicts. Training opens at
13:46:36 and is cut off around 00:02, so roughly **17 epochs fit: 22 through
38 or 39 of 40**.

Ultralytics selected a single T4 (`CUDA:0`, 11.5G of 14.9G used) rather than
both. That is the same shape every earlier session ran in, and Kaggle bills
the session by wall clock either way, so it costs nothing here.

## Watching it

```bash
export KAGGLE_API_TOKEN=...      # the account that started the run
python3 tools/watch_run.py qwyi123/hod26-team --seconds 90
```

## Epochs

| epoch | mAP50 | mAP50-95 | wall |
| --- | --- | --- | --- |
| 22 | pending | pending | |
