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

## Correction: the deficit is spectral after all — against the background

The sections above say the four deficit classes fail on box tightness and
that "the spectral front end is working". The second claim was only half
measured. It was true of the *material pairs* — apple vs apple_plastic and
the like, where the question is whether two classes' spectra differ. It was
never tested for the question that matters for a box edge: whether an
object's spectrum differs from the **background right around it**. A
teammate's observation was that for these four classes it does not. A
CPU-only scan (`tools/build_spectral_scan.py`, kernel
`qwyi123/hod26-spectral-scan`, 10063 annotated instances over all 3000
frames, no GPU quota) confirms it.

For each instance: a core (the box shrunk 15% per side) against a ring outside
the box with every annotated box removed from it. Validated first on synthetic
frames with known answers.

### No class-wide spectral signature separates them from their background

One spectral-shape rule per class, object mean vs its own ring mean, 5-fold
cross-validated with frames grouped so none sits on both sides
(`tools/spectral_vs_iou.py`). This is the question a detector actually faces —
it gets one rule per class, not one per object:

| class | AUC | | class | AUC |
| --- | --- | --- | --- | --- |
| **e-bike** | **0.570** | | badminton | 0.809 |
| **stone_block** | **0.588** | | table_tennis | 0.820 |
| **car** | **0.640** | | charger_head | 0.822 |
| **people** | **0.682** | | banana_plastic | 0.862 |
| car_toy | 0.703 | | rubik | 0.863 |
| egg_wood | 0.798 | | apple | 0.970 |
| banana | 0.803 | | apple_plastic / orange_plastic | 0.982 |
| | | | egg_plastic / egg / orange | 0.989-0.996 |

**The four deficit classes are exactly the bottom four of eighteen.** No
other class falls below 0.70.

### They are grey: the background's spectrum, scaled

Per-band contrast (core - ring) / ring, class median, is flat across all 16
bands for three of them — the object is the background's spectrum at a
different brightness:

| class | band 0 | band 7 | band 15 | log brightness ratio |
| --- | --- | --- | --- | --- |
| people | -0.34 | -0.35 | -0.27 | -0.39 (0.68x) |
| e-bike | -0.29 | -0.31 | -0.32 | -0.37 (0.69x) |
| stone_block | -0.13 | -0.13 | -0.06 | -0.11 (0.90x) |
| car | +0.19 | +0.13 | -0.10 | +0.11 |

Compare `orange_plastic`: -0.05, -0.69, +2.15 — a spectral signature, not a
brightness change. Spectral angle between object and ring against the
clutter within each: people 0.59, e-bike 0.60, car 0.88, stone_block 1.05;
most controls 1.2-8.6. For three of the four, the angle to the background is
*smaller than the spread inside the background itself*.

### And the boundary is nearly invisible

Across the box edge (2 px inside vs 2 px outside): spectral angle 0.76°
(stone_block), 1.17° (e-bike), 1.20° (people), 2.39° (car), against 1.3-9.3°
for the controls; brightness step 2-8% against up to 50%. With neither the
spectrum nor the brightness marking the edge, the regression head has only
spatial texture and shape to place it — which is why elongation predicted AP
(r = -0.46) in the section above: shape is the only cue these classes have.

### This explains part of the looseness, not all of it

On the 1929 held-out instances against the epoch-40 model's boxes:

- separability vs matched IoU, all instances: Spearman **r = +0.34**
- deficit classes by separability tertile: least separable third **45.5%**
  tight (IoU >= 0.75), middle 58.7%, most separable 58.1%
- within any one class the correlation is weak (+0.13 to +0.26)

Even the most separable third of the deficit instances is boxed tight only
58% of the time, against 85-100% for the other fourteen classes. And
`car_toy` and `badminton` separate from their backgrounds about as poorly per
pixel (raw-16 AUC 0.895, 0.854) yet box tight 95%+. So spectral
inseparability is a real part of the cause, and scene is the rest: street
frames, more objects per frame, occlusion, rarity.

### What it rules out, and what it does not

- **The front end is not throwing the difference away.** Raw 16 bands vs the
  8 SRF channels the network receives: 0.825 vs 0.798 (car), 0.829 vs 0.816
  (people), 0.851 vs 0.817 (e-bike), 0.892 vs 0.880 (stone_block). A loss of
  0.01-0.03. The information is not in the data to begin with.
- **Spectral-side fixes cannot help these four.** Spectral SMOTE,
  band selection, a better spectral adapter, a contrastive spectral loss
  (the `hod26-supcon` idea) all push on a signal these classes do not carry.
- **What remains is spatial and brightness.** They are darker than their
  background (0.68-0.90x), and `per_image_norm` keeps that. Levers aimed at
  shape and edges — resolution, the box-loss shape, more instances of these
  scenes — are the ones that address a boundary that is only visible in
  space.

### Two caveats

- **stone_block's ring may contain stone blocks.** The training set is known to
  leave some unannotated (README: sample 558). The ring removes annotated
  boxes only, so an unlabelled neighbour would sit in the "background" and make
  stone_block look less separable than it is.
- **Edge spectral angle correlates *negatively* with IoU within the deficit
  classes** (-0.32 to -0.34). Not understood; one guess is crowding — an edge
  that borders another person or vehicle shows a large angle and also a harder
  box. Unverified.

### Band order, found on the way

Band-to-band correlation of normalised spectra, 400k pixels: index-adjacent
pairs average 0.554, index-distant pairs -0.195, so index order is
approximately spectral order — with two exceptions. **Band 4** correlates at
about 0.1 with everything, and band 11 sits apart too. The correlation chain
puts them at the ends: [15, 13, 14, 12, 10, 8, 9, 7, 6, 1, 0, 2, 3, 5, 4, 11].
The Gaussian SRF bank (width 2 over *index*) averages band 4 with bands 2-6,
diluting it — and band 4 is where `orange` (+1.29 vs ~0 for its neighbours),
`orange_plastic` (+1.00 vs -0.6) and `banana_plastic` carry a distinctive
contrast. Those classes already score well, so this is not urgent, but the
SRF bank's premise that neighbouring indices are neighbouring wavelengths does
not hold for band 4.
