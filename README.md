# HOD26 — Hyperspectral Object Detection Challenge 2026

A solution pipeline for the [Hyperspectral Object Detection Challenge 2026](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026),
driven by a Dream-RSI recursive self-improvement loop ([arXiv:2609.14858](https://arxiv.org/abs/2609.14858)).

## The problem

3000 training and 1000 test frames, 18 classes, Pascal VOC boxes, scored as
COCO `mAP@[0.5:0.95]` (101-point interpolation, macro-averaged over classes
with ground truth; `mAP@0.5` secondary).

Each frame is a **16-bit greyscale PNG holding a 4×4 spectral mosaic**: a
(4H, 4W) image decodes to an (H, W, 16) cube, and that cube's resolution is the
coordinate system the boxes use. Two properties drive the whole design:

1. **The class list pairs objects that are only separable by spectrum.**
   `apple`/`apple_plastic`, `banana`/`banana_plastic`, `orange`/`orange_plastic`,
   `egg`/`egg_plastic`/`egg_wood`, `car`/`car_toy`. Shape cannot tell these apart.
2. **The organizers' demo composite throws that signal away.** Bands 0,1,2 are
   adjacent and nearly identical — measured `corr(ch0, ch2) = +0.997` on a sample
   frame, i.e. a greyscale image tripled. Wider band spacing (+0.254), PCA
   (+0.011) and a band-ratio channel (−0.903) all retain far more.

Objects are also small (a `stone_block` is ~11×20 px), and the metric punishes
imprecision hard: a uniform 3-pixel box shift drops `mAP` from 1.00 to 0.249
while `mAP@0.5` still reads 0.906.

## Layout

| Path | Role |
| --- | --- |
| `src/hod26/` | Object level: cube decoding, VOC parsing, the local COCO scorer, submission writing |
| `dream_rsi/` | Meta level: discovery tree, replay simulator, exploration policy, dreaming, the outer loop |
| `kernels/hod26_round/` | The Kaggle GPU kernel; `build/` is generated, not edited |
| `tools/` | `build_kernel.py` assembles the kernel; `submit.py` fits and submits |

`tools/build_kernel.py` generates the kernel by inlining the `src/hod26`
modules, so the scorer that ranks nodes in the discovery tree cannot drift from
the scorer used locally.

## The loop

One iteration:

1. **Online explore** — the current exploration policy drives real GPU attempts
   into a fresh discovery tree. One Kaggle kernel session runs one decision
   round, so decoding 3000 cubes is paid once per round rather than once per
   candidate. The binding resource is GPU quota (~35 h/week), not wall time.
2. **Construct the simulator** — the finished tree joins the replay pool.
3. **Dream** — policy versions are developed in sequence and replayed across the
   whole pool at zero execution cost, then the best is redeployed.

Only the exploration policy changes between iterations. The discovery agent, the
evaluator and the candidate space stay fixed, so an improvement is attributable
to exploration rather than to a moving target.

```bash
python3 -m dream_rsi.run --executor mock    # exercise the loop, no quota spent
python3 -m dream_rsi.run --executor kaggle --iterations 3 --workers 3
python3 tools/submit.py --dry-run           # fit at full fidelity, validate
python3 tools/submit.py                     # ... and submit
```

## Rules compliance

The organizers' clarifications in forum threads
[#727863](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/727863)
and [#729747](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/729747)
govern the points below.

**Declared pretrained weights.** The detector is **RT-DETR**, a transformer
detector (hybrid encoder, transformer decoder with IoU-aware query selection),
initialised from **Ultralytics RT-DETR COCO-pretrained weights**
(`rtdetr-l.pt`, `rtdetr-x.pt`, `rtdetr-resnet50/101`), from
[github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics),
licensed **AGPL-3.0**, pretrained on **COCO** (ImageNet for the ResNet
backbones). Organizers confirmed public ImageNet/COCO weights are "allowed and
encouraged" provided they are declared here. No other external dataset is used
for pretraining. The 16-band input does not change that: the detector itself is
built with three input channels, so every COCO tensor transfers including the
stem, and the sixteen bands are reduced to three in front of it by a fixed
non-negative Gaussian response bank followed by a trainable 1x1 mixer (24
parameters). Those 152 front-end parameters and the classifier rows for the six
classes with no COCO counterpart are the only weights that start from scratch.

Classifier head rows are inherited from COCO by class name, using ultralytics'
own remapping with an explicit HOD26 -> COCO name map (`COCO_PRIOR` in
`src/hod26/voc.py`) so that, for example, `people` inherits `person` and
`badminton` inherits `sports ball`. This transfers 12 of 18 rows rather than
the 4 that match by exact spelling. No external data or annotation is involved
-- it is the same COCO checkpoint, read for more of what it already contains.

**Single model.** The submission comes from one checkpoint. Test-time
augmentation and multi-scale inference are in the candidate space because
organizers confirmed they "do not count as an ensemble"; combining outputs of
different models does, and is not done anywhere in this pipeline. RT-DETR is
NMS-free -- top-k selection happens inside the decoder -- so the NMS IoU knob is
omitted for it rather than searched.

**Annotations.** Ground-truth XML files are used exactly as shipped. Nothing is
hand-corrected, added or removed. Note that the training set is known to be
incompletely annotated — sample 558 has visible stone blocks with no boxes while
440 annotates six — which is an open question in forum
[#737136](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/737136).

**Submission schema.** `id,image_id,class_id,confidence,x1,y1,x2,y2`, with `id`
a 0-based row counter, per the organizers' correction. The bundled
`sample_submission.csv` omits `id` and was explicitly disavowed as outdated.

## What was measured

Phase A, 300 images / 10 epochs / imgsz 1024, one seed each, scored on the same
held-out split by the trainer's own validation pass:

| model | front end | mAP@[.5:.95] | mAP@.5 | min |
| --- | --- | --- | --- | --- |
| rtdetr-l | band_stack, 16->8 SRF bank + trainable 8->3 | **0.4556** | 0.6603 | 23.1 |
| rtdetr-l | srf3 -- the same averaging, rendered offline | 0.4447 | 0.6501 | 23.6 |
| rtdetr-l | pseudo_rgb, bands 0/1/2 | 0.4325 | 0.6292 | 23.7 |
| yolo26m | band_stack, 16->8 SRF bank + trainable 8->3 | 0.4129 | 0.6376 | 13.7 |

The front-end ordering separates two claims that are usually made together.
Averaging all sixteen bands non-negatively beats picking three adjacent ones by
0.0122 -- that is the signal-to-noise argument, and it needs no trainable
parameters. The trainable mixer then adds 0.0109 on top of that averaging --
that is the adapter argument, and it is what recovers the material
discrimination the averaging costs (the three averaged channels correlate above
0.98 before the mixer sees them).

These are single runs at 1/100th of the final run's compute. They are used to
choose between structural alternatives, not to set hyperparameters.

### What our own numbers mean

The table above is ultralytics' validator. It reads high. Submitting the first
of those checkpoints and re-scoring it three ways puts a number on how high:

| same checkpoint, same 600 held-out frames unless noted | mAP |
| --- | --- |
| ultralytics' validator, during training | 0.4556 |
| pycocotools through the submission path | 0.4076 |
| pycocotools through the submission path, *test* frames (leaderboard) | 0.40249 |

So the held-out split is representative -- 0.005 from the test set -- and
essentially the whole 0.053 is the two metrics disagreeing rather than the test
set being harder. Every local number in this repository is therefore read down
by about 0.048 before it is compared to a leaderboard position.

The gap is not something to recover. maxDets is not the cause: capping at 100
and at 300 gives the identical 0.4076, because the detections past the first
hundred sit at conf 0.001 and match nothing. Nor is it the inference path --
sweeping it against pycocotools moves nothing that matters:

| inference setting | mAP |
| --- | --- |
| rectangular letterbox | 0.4078 |
| augmented inference (TTA) | 0.4078 |
| square letterbox at 1024 (what the submission does) | 0.4076 |
| square letterbox at 640 | 0.3928 |

which also confirms that imgsz carries over from the checkpoint correctly, that
inference resolution is worth 0.015 between 640 and 1024, and that TTA buys
nothing here.

## The full run

Four sessions, scored on the competition's own metric throughout. Every local
number here is pycocotools through the submission path, the same ruler the
leaderboard uses.

| session | epochs | held-out | leaderboard | gap |
| --- | --- | --- | --- | --- |
| 1 | 9 | 0.6689 | 0.59680 | 0.0721 |
| 3 | 10 | 0.6706 | 0.59783 | 0.0728 |
| 2 | 19 | 0.6869 | **0.62584** | 0.0611 |

Sessions 1 and 3 are two independent runs at nearly the same length, which is
not how they were meant to relate -- see below -- but it makes them the
reproducibility check this project never budgeted for. They land 0.0010 apart
on the leaderboard and 0.0017 apart on held-out, so **run-to-run noise at full
scale is about 0.001**, an order of magnitude below the 0.015 measured on the
104-frame street subset the loss A/B used. That is worth knowing: it means the
A/B's noise floor was a property of the subset, not of the pipeline.

It also settles, by accident, the one change session 3 was carrying. Session 3
had `repeat_threshold=0.2` and session 1 did not. The 10 -> 19 epoch slope is
+0.0031/epoch, so session 3's extra epoch alone should have put it at 0.5999;
it scored 0.59783. Repeat-factor sampling is worth about **-0.002** here --
inside the noise, but certainly not the gain it was included for.

The gap between held-out and the leaderboard does not widen with training:
0.072 at nine epochs, 0.073 at ten, 0.061 at nineteen. Longer training
generalises better rather than memorising the split.

### The resume that never resumed

The run was designed as one long training split across sessions, because a
Kaggle session is capped at twelve hours. It was not. `find_checkpoint()`
looked one level deep under `/kaggle/input`, while `find_weights()` -- the same
job, for the prediction kernels -- had a recursive fallback. Kaggle does not
mount a kernel's output at a stable depth; `data_root()` says so in its own
docstring and searches rather than guessing. So every prediction kernel found
its checkpoint and every training session failed to, and because `None` means
"no previous session", each one restarted from COCO and reported a healthy
curve from epoch 1:

```
[22:52:23]   training starts at epoch 1 of 34 (resume=False)
```

That line was printed, correctly, every time. The comment above
`find_checkpoint` had already named the failure mode -- "a resume that silently
restarts from epoch 0 looks like a slow run rather than a failure, which is why
the starting epoch is logged explicitly". The alarm was built and never read.

Sessions 1, 2 and 3 were therefore three runs of 9, 19 and 10 epochs rather
than one run of 38, and the best model is the longest single session rather
than their sum. At +0.0031/epoch, still undecayed at epoch 19, 38 epochs
extrapolates to roughly 0.68 -- against the 0.6643 that separates fourth place
from the baseline cluster. The bug is worth something like 0.03 to 0.05 mAP.

`find_checkpoint` now searches the way `find_weights` does, and a candidate may
set `require_resume`, which turns a missing checkpoint into an error before
training starts rather than a silent restart. A failed resume now costs
minutes.

### The inference side is closed

Three independent probes, all negative, and each failure has a cause rather
than a guess.

| probe | arms | best result |
| --- | --- | --- |
| inference settings (letterbox, imgsz, max_det) | 7 | nothing outside 0.0002 |
| TTA + Weighted Boxes Fusion | 11 | **-0.0196** |
| neighbourhood rescoring, no rows removed | 16 | +0.0002 |

Upscaling costs 0.02 to 0.07 (1024 -> 0.4076, 1280 -> 0.3847, 1536 -> 0.3362)
because a DETR decoder's query priors are tuned to the training scale.

Fusion costs about 0.023 before any augmentation helps: the control arm fuses a
single view with *itself* and still drops 180000 boxes to 166765. RT-DETR is
NMS-free and the submission keeps all 300 queries per frame to fill the tail
COCO integrates over, so any clustering deletes rows and deleted rows are pure
loss. The extra views do work -- three views beat the self-fusion control by
+0.0035, which is the real gain from averaging coordinates -- and it is nowhere
near enough to pay for the merge. Note that ultralytics' `augment=True` is a
no-op on this detector: `RTDETRDetectionModel.predict` accepts the flag and
never reads it, so an earlier sweep's "TTA" arm measured the identical forward
pass as its baseline.

Rescoring kept every row and every coordinate and changed only the ranking, so
nothing could be lost to deletion. The diagnostic explains the result before
the sweep does:

```
corr(confidence, true IoU)   = +0.808
corr(agreement,  true IoU)   = +0.429
corr(agreement,  confidence) = +0.574
```

IoU-Net, GFL, VarifocalNet and Cascade-DETR all rest on classification
confidence correlating *weakly* with box tightness. Here it correlates at
+0.808, because RT-DETR already performs IoU-aware query selection -- the thing
those papers add is inside the architecture. Neighbourhood agreement is a
strictly worse estimator of the same quantity and mostly redundant with it.

So the residual localization error -- 16.6% of held-out boxes, against 1.3%
missed and 0.1% misclassified -- is a limit of the trained model rather than of
how its output is decoded. It moves with training, and with nothing else tried
here.

## Credentials

`KAGGLE_API_TOKEN` is read from the environment. Nothing in this repository
contains or writes a credential.
