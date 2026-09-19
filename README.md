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

## Credentials

`KAGGLE_API_TOKEN` is read from the environment. Nothing in this repository
contains or writes a credential.
