# Where the remaining 0.05 is, measured on the epoch-40 model

TIDE-style decomposition of the 1930 held-out ground-truth boxes, run on CPU so
it cost no GPU quota (`tools/build_diagnose.py`, kernel `qwyi123/hod26-diagnose`).

## Nothing is missed and nothing is misnamed

| fate of a ground-truth box | count | share |
| --- | --- | --- |
| found, IoU >= 0.75 | 1642 | 85.1% |
| found, IoU 0.50-0.75 | 214 | 11.1% |
| found, IoU 0.10-0.50 | 42 | 2.2% |
| **boxed well, wrong class** | **2** | **0.1%** |
| **not found at all** | **30** | **1.6%** |

Detection and classification are finished work. **98.3% of ground truth is found
and 99.9% of what is found is named correctly.** The entire remaining score is the
distance from a median matched IoU of **0.8697** up to the thresholds above it —
mAP@[.5:.95] averages ten thresholds, and a box at 0.87 passes eight of them.

Forty epochs moved that median from 0.864 to 0.8697. It is not moving on its own.

## The deficit is four classes boxing loosely, not failing

| class | tight (>=0.75) | loose | missed | wrong class |
| --- | --- | --- | --- | --- |
| stone_block | 42.4% | **57.6%** | **0.0%** | 0.0% |
| people | 46.4% | **49.0%** | 4.2% | 0.4% |
| e-bike | 54.4% | 33.3% | 12.3% | 0.0% |
| car | 74.6% | 18.3% | 7.1% | 0.0% |
| the other fourteen | **90.6-100%** | 0-9.4% | ~0% | ~0% |

`stone_block` misses **nothing**. Every stone block in the held-out set is found;
57.6% of them are boxed too loosely to score. The same shape holds for `people`.

## Why those four, and not the small ones

| | r with AP |
| --- | --- |
| deviation from square, \|log(median aspect ratio)\| | **-0.46** |
| frames containing the class | **+0.54** |
| median object side in pixels | **-0.17** |

**Size is not the driver.** `table_tennis` has a 15.0 px median side — the
smallest in the dataset — and is boxed tight 92.5% of the time. `e-bike` is the
largest at 44.4 px and is tight 54.4%. The correlation with size is slightly
*negative*.

What predicts the deficit is elongation and rarity, and each dominates a
different class:

| class | frames | median side | median AR | dominated by |
| --- | --- | --- | --- | --- |
| stone_block | 42 | 20.8 | 0.73 | rarity (42 frames of 3000) |
| people | 352 | 29.2 | **0.47** | elongation (2:1 tall) |
| car | 193 | 28.9 | **1.68** | both |
| e-bike | 175 | 44.4 | 0.86 | rarity |
| table_tennis | 587 | 15.0 | 1.00 | — tight 92.5% |
| badminton | 608 | 23.7 | 0.81 | — tight 95.6% |

## The mechanism

Ultralytics regresses boxes in normalised coordinates and penalises size error
with a plain L1. A person is about 20 px wide in a ~493 px cube, so its
normalised width is ~0.04. An absolute L1 error of 0.005 — negligible for a wide
box — is **12.5% of a person's width**, and IoU falls off a cliff. The same error
on a square, frequent object like `table_tennis` costs proportionally less
because the two sides are balanced and the class has 587 frames of evidence to
average it out.

That is precisely what `log_size_l1` addresses: L1 on `log(w)` and `log(h)` makes
a relative error cost the same whatever the absolute size.

## Two frames' worth of context

The 18 classes split into two scene types that **share almost no frames**: of 596
held-out frames with confident detections, only **3** contain both a street class
and a tabletop class. Street frames are ~19.6% of the data and carry all four
deficit classes, at a median of 4 confident objects per frame against 3 for
tabletop, peaking at 16 against 10.

## What this rules out

- **Per-class classification weighting** — 0.1% of boxes are misnamed globally
  and 0.0% for three of the four deficit classes. There is nothing to win.
- **Pseudo-labels on the test frames** — self-training adds more examples of a
  detection problem that is already solved. It does not tighten a box.
- **A GAN** — same objection, plus it would have 42 frames to learn
  `stone_block` from.
- **More epochs of this recipe** — eighteen of them moved median IoU by 0.006.

## What it points at

Ranked by how directly it attacks a median IoU of 0.8697 on elongated, rare boxes.
All three knobs exist and are flags; none has ever run at full scale.

1. **`log_size_l1=True`** — relative rather than absolute size error. Aimed
   exactly at `people` (AR 0.47) and `car` (AR 1.68).
2. **`bbox_loss="DIoU"`** — measured at 0.4197 against GIoU's 0.4106 on the
   104-frame street proxy (`tools/street_ab.py`), nominally ahead but inside that
   proxy's 0.01 noise floor. Full scale can resolve it; CIoU gave the gain back,
   so the aspect term belongs in the L1, not in the overlap.
3. **`loss_gain={"bbox": 2, "giou": 5}`** — shift weight from coordinate L1 to
   overlap, the blunt version of the same idea.
4. **`repeat_threshold`** — the rarity half of the correlation, and the only
   lever for `stone_block`'s 42 frames. Measured at -0.002 at `t=0.2`, inside the
   noise; `[0.05, 0.1]` is in the search space and never ran.

If those four classes reached the 90% tight the other fourteen already manage,
their AP would go from 0.32-0.59 to roughly 0.75 — **about +0.045 macro**, which
is the distance to the top of the leaderboard.
