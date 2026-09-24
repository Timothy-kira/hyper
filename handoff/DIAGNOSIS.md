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


## Enhancing the difference: two CPU scans of fixed transforms

If the grey classes are the background at a different brightness, a transform
that exposes *local* contrast should separate them better than raw bands. Two
CPU scans (`tools/build_enhance_scan.py`, no GPU) measured class-wide pixel AUC
(core vs ring, grouped 5-fold by frame) and median edge d' across the box
boundary, for each transform:

| transform | AUC deficit 4 / other 14 | edge d' deficit / other |
| --- | --- | --- |
| raw16 (log bands, baseline) | 0.711 / 0.895 | 0.26 / 0.93 |
| lratio, 31 px plain window | 0.652 / 0.868 | 0.18 / 0.72 |
| lcn, 31 px plain window | 0.634 / 0.858 | 0.13 / 0.61 |
| local RX, 31 px | 0.695 / 0.718 | 0.23 / 0.46 |
| whiten16 | 0.713 / 0.895 | 0.27 / 0.92 |
| shape + raw | 0.718 / 0.902 | 0.28 / 0.98 |
| lratio, 31/63 annulus | 0.709 / 0.909 | 0.23 / 0.95 |
| **shape + lratio 31/63 annulus** | **0.728 / 0.922** | **0.28 / 1.02** |
| raw + lratio 63 + lratio 95 | 0.733 / 0.931 | 0.27 / 0.96 |

A plain 31 px window hurts: for 20–45 px objects it is mostly object, so the
ratio cancels the contrast it is meant to expose. A guard region (annulus)
fixes that. The best balanced transform, shape + 31/63 annular log-ratio, is
what S3T feeds its encoder (level, shape, contrast per band). The gains are
real but modest: a fixed per-pixel transform moves the deficit classes by
~0.02 AUC, which is why S3T learns the spectral-spatial mapping instead of
hand-picking one.


## S3T-X model (LB 0.63985): why stone_block / people / e-bike / car are still last (2026-09-24)

Held-out AP (pycocotools, `zetaoxia/hod26-s3tx-diag` re-scored the 600 frames
at 0.6834):

| class | AP |
| --- | --- |
| stone_block | 0.234 |
| people | 0.379 |
| e-bike | 0.395 |
| car | 0.551 |
| the other 14 | 0.719–0.794 |

Per-instance data: `spectral_scan.jsonl`, all 10063 annotated instances, with
core/ring spectra and the box of each.

### Not rare: crowded
Counted by instance, none of these classes is rare. people has 1095 instances,
the second most of any class. stone_block has 270, which is more than egg_wood
(258, AP 0.78). What is rare is **scenes**: stone_block appears in only 42
frames.

What sets these four apart is **density and occlusion**:

| class | instances / frame (max) | overlapping another box (IoU > 0.1) | > 30% covered |
| --- | --- | --- | --- |
| stone_block | 6.4 (19) | 10.0% | 5.9% |
| people | 3.1 (15) | 17.2% | 10.3% |
| e-bike | 2.6 (15) | **32.2%** | **23.7%** |
| car | 2.9 (11) | 12.7% | 7.4% |
| the other 14 | 1.0–2.3 (≤ 5) | 0–3.3% | 0–2.2% |

The pairs are mostly same-class (people×people 128, e-bike×e-bike 118,
car×car 64). There are also 60 e-bike×people pairs: riders, i.e. two different
objects that are meant to overlap.

Small size is not the cause: 60–65% of table_tennis and egg_wood boxes have a
side under 16 px, and those classes still reach AP 0.72–0.78.

### Spectrally: grey, dark, and a projection blind to it

| class | class-mean object-vs-ring spectral angle ÷ within-class spread | per-band contrast (core − ring)/ring | darker than surroundings |
| --- | --- | --- | --- |
| stone_block | 1.16 | −0.06 to −0.14, **flat over all 16 bands** | 80% |
| people | 0.58 | −0.27 to −0.36, flat | 80% |
| e-bike | **0.21** | −0.29 to −0.34, flat | 84% |
| car | 0.83; within-class spread 7.2° (paint colours) | +0.2 in bands 0–12, −0.1 in 13–15; sign agrees for only 55–67% of instances | 41% |
| orange_plastic (for contrast) | 9.71 | −0.85 to +2.15, 91–100% same sign | 14% |

Three of the four differ from the background only by a **uniform darkening**:
of the 16 bands, the only information they carry is brightness. All 14
tabletop classes are *brighter* than their background (86–100% of instances).

The 16 → 3 LDA projection that feeds the COCO stem has rows that **sum to
exactly 0**. That makes it orthogonal to a uniform change, so the pretrained
path received nothing from these classes at initialisation. After training,
the mixer had learned a small brightness component (cos 0.07–0.11 with the
uniform direction). A grey object's input signal is still about 1/8 of a
spectral object's.

### Better than LDA? The bottleneck is the three channels

Object-vs-background SNR is the Mahalanobis distance against the
between-instance covariance of the background rings. Projections were fitted
on train instances and evaluated on held-out ones.

| stem input | stone_block | people | e-bike | car | class separation (median / min) |
| --- | --- | --- | --- | --- | --- |
| LDA 3 (current) | 0.24 | 0.62 | 0.52 | 0.64 | 3.40 / 0.12 |
| contrast-LDA 3 | 0.28 | 0.70 | 0.55 | 1.55 | 1.73 / **0.04** |
| LDA 3 + mean level | 0.29 | 0.74 | 0.57 | 0.76 | 3.62 / 0.12 |
| LDA 3 + all 16 bands | **0.79** | **2.12** | **1.59** | **3.00** | **5.71 / 0.98** |

- Re-choosing the 3 channels trades one side against the other: a better SNR
  costs class separation.
- Keeping the 3 LDA channels and also giving the stem all 16 bands recovers
  3–4.7× the SNR.

### On the model (step 0)
Confident predictions (score ≥ 0.3):

| class | tight (IoU ≥ 0.75) | missed | tight when isolated | tight when overlapping > 0.1 |
| --- | --- | --- | --- | --- |
| stone_block | 36% | 24% | 30% | (n = 3) |
| people | 43% | 8% | **51%** | **15%** |
| e-bike | 47% | 9% | 52% | 37% (missed 21%) |
| car | 71% | 9% | 72% | 60% |
| badminton / rubik | 96–97% | 0% | | |

- **Brightness contrast predicts tightness.** From the lowest to the highest
  |log brightness| tertile, the tight share rises: people 33 → 53%, e-bike
  37 → 63%, car 62 → 90%. stone_block stays flat at about 36%.
- **Confident false positives** per 100 ground-truth boxes: 12.6–20.7 for these
  four classes, against 0–3.6 for the others.
- **Box bias:**
  - stone_block boxes are 8% too wide.
  - Correcting that post hoc does **not** survive cross-validation: fitted on
    one half of the frames and tested on the other, it gave −0.080 and +0.018.
    With 13–23 matched boxes the bias is not stable, so the correction is not
    used.

### What was done about it: fine-tune from the S3T-X best.pt
Kernel `zetaoxia/hod26-s3t-ft`, 15 epochs.

- **S1:** the stem's first conv reads the 3 LDA channels plus all 16 bands.
  The extra channels are zero-initialised, so step 0 is unchanged.
- **C1:** box loss ×2 on these four classes' matched pairs.
- **C2:** repulsion from same-class neighbouring ground truth, charging only
  the IoG *beyond* the truth's own overlap. Ground-truth boxes here overlap
  each other, so plain RepGT would charge the exact answer.
- **B2:** crowd copy-paste of real instances beside a same-class anchor in
  street frames. Each instance brings a feathered background margin and is
  scaled per band to the destination's surroundings.

Rejected:
- post-hoc box correction (above);
- rewriting e-bike/rider boxes: they are two different objects, labelled
  separately in the test set too;
- ignoring unmatched confident stone_block predictions as "missing labels":
  unverified, and it would teach the model to fire where annotators do not.

### Result of the weak-class fine-tune, and where its bottleneck was
`zetaoxia/hod26-s3t-ft`: 15 epochs, fine-tuned from the S3T-X best.pt.

| | held-out (pycocotools) | LB |
| --- | --- | --- |
| parent (S3T-X) | 0.6835 | **0.63985** |
| fine-tune | 0.6862 | 0.63083 |

Bootstrap over frames (40 resamples) of the held-out difference:

| | Δ | 95% interval |
| --- | --- | --- |
| overall | +0.0026 | [−0.0006, +0.0062] |
| stone_block | +0.027 | [+0.013, +0.055] |
| people | +0.007 | [−0.005, +0.022] |
| e-bike | −0.018 | [−0.049, +0.008] |
| car | +0.013 | [−0.003, +0.031] |
| car_toy | −0.012 | [−0.028, +0.003] |

Only stone_block's gain is outside the noise. The LB went down by 0.009.

1. **S1 barely engaged.** The 16 new stem channels ended at 2% of the RGB
   channels' weight norm (0.136 vs 6.93). The direction was right: 95% of
   what they learned lies along the uniform (brightness) direction, the cue
   the grey classes carry. The amount was not.
   - Zero init, at 0.5 × 4.55e-4 over about 1100 optimizer steps, is too
     little.
   - Accordingly the low-contrast tertile did not improve (tight share):
     - people 33 → 31%
     - e-bike 37 → 37%
     - car 62 → 60%
   - It would need a much higher LR for that part, or a non-zero init along
     the brightness direction.
2. **The ruler cannot resolve these classes.** The held-out set has 57
   e-bike, 58 stone_block and 41 orange instances. From epoch to epoch,
   e-bike AP swung 0.23–0.42 and stone_block 0.21–0.28. The
   per-class gate "all four weak classes up, no other class down more than
   0.005" is well inside that noise. The overall +0.0027 had an interval
   that includes 0. A 600-frame held-out cannot select per-class
   interventions, and the LB (1000 frames) disagreed.
3. **Crowd paste and repulsion helped some crowds and hurt e-bike rows.**
   - people touching another box: tight 38 → 44%, missed 18 → 11%.
   - car touching or overlapping: improved.
   - e-bike overlapping > 0.1: tight 37 → 32%, missed 21 → 26%, median
     IoU 0.70 → 0.61.
   - Pasting e-bikes into rows of e-bikes (and repelling them) may make
     parked rows less learnable. n = 19, so this is a hint, not a result.
4. **No ablation.** Four changes in one run means none of their effects can
   be separated. On LB evidence the combination is net negative, so the
   parent stays the submission.

## External data (2026-09-24): HOT2024, SAM3, HOD3K — and why the HOD3K fine-tune scored lower

**HOT2024 (same XIMEA VIS camera, tracking).** 80 videos fetched (tools/hot_fetch.py). Single-object
labels; the LB-best teacher recovers the tracked target at conf >= 0.6 for only people 0.19 / car 0.33,
so teacher-completed labels would leave most people/cars as background. Dropped.

**SAM3 (keras/sam3 on Kaggle) as a weak-class labeller, held-out 120 frames.** Text only: people
P 0.47 R 0.75 tightness 0.43 (teacher@0.6: 0.87/0.51/0.65), e-bike never found. Text + the teacher's
>= 0.6 boxes as exemplars (exemplars must be unpadded, int32 labels, or keras_hub silently drops them):
recall up (people .51->.72, car .71->.87, stone_block .37->.76) at a large precision/tightness cost;
useful only as ignore regions (GT taught as background: people .22->.15, stone .39->.15). Not used.

**HOD3K (S2ADet): xishengfeng/hsidataraw (train) + hsidata (val/test).** 3239 raw frames, same X2Cube
layout; classes 0,2 -> people (12144), 3 -> car (2188), 1 -> e-bike (817) — read off drawn boxes
(the id->name order is NOT the paper's count order). Seven crop sizes; six need a non-zero mosaic
phase (tools/build_hod3k_convert.py). No duplicates with the competition (max thumbnail cosine 0.928).
A global per-band gain brings people 20->7 deg to the competition but pushes car/e-bike further,
so none applied. Two-stage fine-tune from LB-best (4 ep comp+HOD3K, 2 ep comp only, band-gain 0.1):
held-out 0.6864 (LB-best 0.6828), **LB 0.62844 (LB-best 0.63985)**.

Why lower (boot_cmp over held-out, and test-set drift against LB-best's confident boxes):
- Held-out gain is not HOD3K's: people -0.010, e-bike -0.006, car 0.000 (all n.s.); the +0.0035
  overall ([-0.0024, +0.0073]) is stone_block +0.063 (58 boxes, absent from HOD3K) and noise.
- Test: of LB-best's confident car boxes only 77% survive (140 vs 177; weak-class FT kept 97%),
  while held-out car AP is unchanged -> HOD3K's cars (front/rear, far, dark) moved the car concept away
  from the test set's cars. People +13% boxes with fewer tight ones (IoU>=.9 vs LB-best 68%).
- Both fine-tunes from LB-best raised held-out and lowered LB (-0.009, -0.011): LB-best is epoch 30
  of 44; more training on the train distribution fits held-out (same distribution) tighter and the
  shifted test set (brighter material classes, see shift probe) worse. Held-out cannot rank models here.
