"""HOD26 round driver — executes one Dream-RSI decision round on a Kaggle GPU.

One kernel session evaluates the whole batch of candidates the exploration
policy selected, so the cost of decoding 3000 spectral cubes is paid once per
round instead of once per candidate. Results land in /kaggle/working/results.json
for the orchestrator to fold back into the discovery tree.

The hod26.* helpers above this docstring are injected verbatim from the repo by
tools/build_kernel.py -- this file is the round-specific part only.
"""

import hashlib, json, os, shutil, time, traceback, zipfile
from pathlib import Path

import cv2
import numpy as np

# The competition cannot be attached as a kernel source (Kaggle drops
# competition_sources on push), so the frames arrive via a private dataset
# holding them band-planar and already de-mosaiced.
DATA = Path("/kaggle/input/hod26-planar")
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
# Rendered frames and run directories go to scratch, not to /kaggle/working.
# The full run materializes 3000 frames plus an augmented copy of each as
# 16-band TIFFs -- around 8 GB -- and everything under /kaggle/working is
# uploaded as the kernel's output and mounted by the session that continues it.
def _scratch():
    """A writable directory Kaggle will not save as this kernel's output.

    /kaggle/temp is the documented place for this and it does not exist in the
    image these kernels run on, so the original check fell through to
    /kaggle/working -- which meant the rendered frames were being uploaded as
    output and mounted by the next session. Only /kaggle/working is saved, so
    creating the directory is enough to keep it out; /tmp is the fallback and
    the output directory itself the last resort.
    """
    for c in (Path("/kaggle/temp"), Path("/tmp/hod26")):
        try:
            c.mkdir(parents=True, exist_ok=True)
            probe = c / ".w"
            probe.write_text("")
            probe.unlink()
            return c
        except OSError:
            continue
    return WORK


SCRATCH = _scratch()
RUNS = SCRATCH / "runs"
# When this kernel started. Kaggle's 12-hour cap is measured from here, not
# from the start of training, so the clock guard has to be too.
T0 = time.time()
VAL_FRACTION = 0.2
PREDICT_BATCH = 32
CACHE_SEED = 20260918


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def free_gb(path):
    """Free space where we are about to write several gigabytes of frames."""
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize / 1e9
    except OSError:
        return float("nan")


def _looks_like_dataset(p):
    return (p / "train" / "annotations").is_dir()


def data_root():
    """Locate the planar dataset wherever Kaggle decided to mount it.

    The mount layout is not stable between kernels: the same dataset has come up
    at /kaggle/input/<slug>/ on one kernel and nested under
    /kaggle/input/datasets/... on another, and an upload may also still be in the
    archives the CLI produced. Guessing the shape cost a GPU session, so this
    searches for the directory that actually holds the split instead, bounded in
    depth so it cannot wander a large input tree.
    """
    roots = [DATA, INPUT]

    def search(base, depth=5):
        if not base.exists():
            return None
        stack = [(base, 0)]
        while stack:
            d, k = stack.pop()
            if _looks_like_dataset(d):
                return d
            if k >= depth:
                continue
            try:
                stack.extend((c, k + 1) for c in sorted(d.iterdir()) if c.is_dir())
            except OSError:
                continue
        return None

    for base in roots:
        hit = search(base)
        if hit is not None:
            return hit

    # Nothing extracted: fall back to any archives found in the input tree.
    zips = sorted(Path("/kaggle/input").rglob("*.zip")) if Path("/kaggle/input").exists() else []
    if zips:
        out = WORK / "unpacked"
        out.mkdir(parents=True, exist_ok=True)
        for z in zips:
            log(f"  unpacking {z.name}")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out / z.stem)
        hit = search(out)
        if hit is not None:
            return hit

    listing = []
    if Path("/kaggle/input").exists():
        for d, _, files in os.walk("/kaggle/input"):
            rel = Path(d).relative_to("/kaggle/input")
            if len(rel.parts) <= 3:
                listing.append(f"{rel}({len(files)} files)")
            if len(listing) > 40:
                break
    raise FileNotFoundError(
        "planar dataset not attached; nothing under /kaggle/input holds "
        f"train/annotations. Tree: {listing}")


def require_ids(ids, where):
    """A wrong data path must fail here, not as a confusing downstream error."""
    if not ids:
        raise FileNotFoundError(f"no files matched under {where}")
    return ids


# ---------------------------------------------------------------- data ------
def split_ids(all_ids):
    """Deterministic train/val split, stable across every candidate and round."""
    rng = np.random.RandomState(CACHE_SEED)
    ids = sorted(all_ids)
    perm = rng.permutation(len(ids))
    n_val = int(round(len(ids) * VAL_FRACTION))
    val = {ids[i] for i in perm[:n_val]}
    return [i for i in ids if i not in val], [i for i in ids if i in val]


def _np_einsum(cube, bank):
    """(H, W, B) cube through a (C, B) response bank -> (H, W, C)."""
    return np.einsum("hwb,cb->hwc", cube.astype(np.float32),
                     np.asarray(bank, dtype=np.float32))


def build_channels(cube, spec):
    """Turn an (H, W, 16) cube into a 3-channel uint8 image per the candidate."""
    mode = spec["mode"]
    lo, hi = spec["stretch_lo"], spec["stretch_hi"]
    if mode == "pca3":
        h, w, b = cube.shape
        flat = cube.reshape(-1, b).astype(np.float32)
        flat -= flat.mean(0)
        # Top-3 right singular vectors; randomized subset keeps this cheap.
        idx = np.random.RandomState(0).choice(flat.shape[0], min(20000, flat.shape[0]), replace=False)
        _, _, vt = np.linalg.svd(flat[idx], full_matrices=False)
        proj = (flat @ vt[:3].T).reshape(h, w, 3)
        return np.dstack([stretch(proj[:, :, c], lo, hi) for c in range(3)])
    if mode == "rgb_plus_ratio":
        a = cube[:, :, 0].astype(np.float32)
        z = cube[:, :, 15].astype(np.float32)
        nd = (z - a) / (z + a + 1e-6)          # normalized difference: material cue
        return np.dstack([stretch(a, lo, hi), stretch(z, lo, hi), stretch(nd, lo, hi)])
    if mode == "lda3":
        # 16 -> 3 discriminant projection; the pretrained stem is untouched.
        flat = cube.reshape(-1, cube.shape[2]).astype(np.float32)
        tot = flat.sum(1, keepdims=True)
        flat = np.divide(flat, tot, out=np.zeros_like(flat), where=tot > 0)
        proj = (flat @ np.asarray(LDA_16_TO_3, np.float32).T).reshape(
            cube.shape[0], cube.shape[1], 3)
        return np.dstack([stretch(proj[:, :, c], lo, hi) for c in range(3)])
    if mode == "srf3":
        # Three non-negative Gaussian spectral response curves, as a real RGB
        # sensor has. One stretch shared across the three outputs, not one per
        # channel: the material cue is the *ratio* between them, and rescaling
        # each independently is exactly the operation that discards it.
        bank = gaussian_srf_bank(cube.shape[2], 3, float(spec.get("srf_width", 3.0)))
        y = _np_einsum(cube, bank)
        return stretch(y.reshape(y.shape[0], -1), lo, hi).reshape(y.shape)
    if mode == "bandgroup3":
        n = cube.shape[2]
        edges = [0, n // 3, 2 * n // 3, n]
        return np.dstack([stretch(cube[:, :, edges[i]:edges[i + 1]].mean(axis=2), lo, hi)
                          for i in range(3)])
    if mode == "bandsel":
        return np.dstack([stretch(cube[:, :, b], lo, hi) for b in BEST_BANDS])
    if mode == "band_stack":
        # Every band as its own input channel. Ultralytics reads this natively:
        # a multi-page TIFF is decoded with imdecodemulti and stacked on axis 2,
        # the model is built with ch=data["channels"], and the HSV augmentation
        # skips anything that is not 3-channel.
        #
        # One stretch shared across all bands, not one per band: the stem is
        # seeded from a projection fitted on relative band magnitudes, and
        # rescaling each band independently would destroy exactly the
        # relationship that seeding encodes.
        flat = stretch(cube.reshape(cube.shape[0], -1), lo, hi)
        return flat.reshape(cube.shape)
    bands = spec["bands"][:3]
    return np.dstack([stretch(cube[:, :, b], lo, hi) for b in bands])


def write_frame(path_stem, img):
    """Persist a rendered frame; >3 channels need a multi-page TIFF.

    cv2 cannot put 16 channels in a PNG, but ultralytics' reader decodes a
    multi-page TIFF with imdecodemulti and stacks the pages into (H, W, N).
    """
    if img.shape[2] > 3:
        path = path_stem.with_suffix(".tiff")
        ok = cv2.imwritemulti(str(path), [np.ascontiguousarray(img[:, :, c])
                                          for c in range(img.shape[2])])
    else:
        path = path_stem.with_suffix(".png")
        ok = cv2.imwrite(str(path), np.ascontiguousarray(img))
    if not ok:
        raise RuntimeError(f"failed to write {path}")
    return path


def channels_key(cand):
    """Identity of a rendered dataset.

    Keyed on channel construction *and* augmentation: candidates differing only
    in training or inference parameters consume byte-identical images, but a
    different augmentation setting produces a different dataset entirely.
    """
    spec = {"channels": cand["channels"], "augment": cand.get("augment", {})}
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]


def class_donors(index, anns, ids, limit=200):
    """One mean spectrum per class, for same-class spectral interpolation."""
    acc, n = {}, {}
    for pid in ids[:limit]:
        cube = load_planar(index[pid])
        for b in anns[pid].boxes:
            patch = cube[b.y1:b.y2, b.x1:b.x2, :]
            if patch.size == 0:
                continue
            m = patch.reshape(-1, cube.shape[2]).mean(0)
            acc[b.cls_id] = acc.get(b.cls_id, 0) + m
            n[b.cls_id] = n.get(b.cls_id, 0) + 1
    return {c: acc[c] / n[c] for c in acc}


def augment_cube(cube, boxes, aug, donors, pool, rng):
    """Apply the spectral and spatial operators a candidate asked for."""
    if aug.get("sg_window"):
        cube = savgol_spectral(cube, aug["sg_window"], aug["sg_polyorder"])
    if aug.get("smote_alpha"):
        cube = spectral_smote(cube, boxes, donors, aug["smote_alpha"], rng)
    if aug.get("cutmix_prob") and pool:
        other = load_planar(pool[int(rng.integers(0, len(pool)))])
        cube, boxes = superpixel_cutmix(cube, boxes, other, aug["cutmix_prob"],
                                        aug["cutmix_blocks"], rng)
    return cube, boxes


def repeat_factors(train_ids, anns, threshold: float):
    """How many times each training frame is written, by class rarity.

    Repeat-factor sampling (Gupta et al., LVIS 2019): a frame is repeated
    sqrt(t / f) times for the rarest class it contains, where f is the fraction
    of frames holding that class. The metric here macro-averages over eighteen
    classes, so a class contributes a full eighteenth however seldom it was
    photographed -- and the frame counts run from stone_block's 42 to
    badminton's 608. Two of the four worst-scoring classes are among the
    rarest, which is what this addresses; the other two are elongated rather
    than rare, which it does not.
    """
    import math
    if threshold <= 0:
        return {pid: 1 for pid in train_ids}
    n = max(1, len(train_ids))
    freq = {}
    for pid in train_ids:
        for name in {CLASSES[b.cls_id] for b in anns[pid].boxes}:
            freq[name] = freq.get(name, 0) + 1
    cls_rep = {c: max(1.0, math.sqrt(threshold / (k / n))) for c, k in freq.items()}
    out = {}
    for pid in train_ids:
        names = {CLASSES[b.cls_id] for b in anns[pid].boxes}
        out[pid] = int(round(max((cls_rep[c] for c in names), default=1.0)))
    extra = sum(out.values()) - len(train_ids)
    if extra:
        top = sorted(cls_rep.items(), key=lambda kv: -kv[1])[:4]
        log(f"  repeat sampling (t={threshold}): +{extra} frames, "
            + ", ".join(f"{c} x{r:.1f}" for c, r in top))
    return out


def materialize(cand, index, train_ids, val_ids, anns, root):  # noqa: C901
    """Write the YOLO dataset this candidate trains on, reusing it if rendered.

    Rendering is keyed on the channel spec alone, so a round that varies only
    imgsz, epochs or NMS settings encodes its images once instead of per
    candidate.
    """
    if (root / "data.yaml").exists():
        log(f"  reusing rendered dataset {root.name}")
        return root / "data.yaml"
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    aug = cand.get("augment", {})
    copies = int(aug.get("copies", 0))
    wants_aug = bool(aug.get("sg_window") or aug.get("smote_alpha") or aug.get("cutmix_prob"))
    donors = class_donors(index, anns, train_ids) if aug.get("smote_alpha") else {}
    pool = [index[p] for p in train_ids] if aug.get("cutmix_prob") else []
    rng = np.random.default_rng(0)

    def emit(root, split, stem, img, boxes, a):
        n = write_frame(root / "images" / split / stem, img)
        lines = []
        for b in boxes:
            cx = (b.x1 + b.x2) / 2 / a.width
            cy = (b.y1 + b.y2) / 2 / a.height
            bw = (b.x2 - b.x1) / a.width
            bh = (b.y2 - b.y1) / a.height
            lines.append(f"{b.cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        return n

    reps = repeat_factors(train_ids, anns, float(cand["train"].get("repeat_threshold", 0.0)))
    n_ch = 3
    for split, ids in (("train", train_ids), ("val", val_ids)):
        for pid in ids:
            cube = load_planar(index[pid])
            a = anns[pid]
            # The unaugmented frame is always written; validation is never
            # augmented, so the score keeps measuring the real distribution.
            img = build_channels(cube, cand["channels"])
            n_ch = img.shape[2]
            frame = emit(root, split, str(pid), img, a.boxes, a)

            if split == "train":
                # Augmented copies are re-rendered; repeats are file copies.
                # A repeat is not a wasted duplicate: ultralytics augments at
                # load time -- mosaic, flip, scale, HSV -- so the same frame
                # listed twice trains on two different images. Re-rendering it
                # would only add our own spectral augmentation on top, which is
                # what the copies setting is for and is separate from balance.
                for k in range(copies if wants_aug else 0):
                    c2, b2 = augment_cube(cube, list(a.boxes), aug, donors, pool, rng)
                    emit(root, split, f"{pid}_a{k}", build_channels(c2, cand["channels"]), b2, a)
                for k in range(reps[pid] - 1):
                    for src, dst in ((frame, frame.with_name(f"{pid}_r{k}{frame.suffix}")),
                                     (root / "labels" / split / f"{pid}.txt",
                                      root / "labels" / split / f"{pid}_r{k}.txt")):
                        shutil.copyfile(src, dst)

    for split, ids in (("train", train_ids), ("val", val_ids)):
        n = len(list((root / "images" / split).glob("*.png"))) + \
            len(list((root / "images" / split).glob("*.tiff")))
        expect = (sum(reps[p] + (copies if wants_aug else 0) for p in ids)
                  if split == "train" else len(ids))
        if n == 0 or n != expect:
            raise RuntimeError(f"{split}: wrote {n} images, expected {expect}")
    log(f"  rendered {cand['channels']['mode']}"
        + (f" +{copies} augmented copies/frame" if wants_aug and copies else ""))

    yaml = root / "data.yaml"
    yaml.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\n"
        f"channels: {n_ch}\n"
        f"nc: {len(CLASSES)}\nnames: {json.dumps(CLASSES)}\n"
    )
    return yaml


# ------------------------------------------------------------ evaluate ------
def predict_kwargs(cand):
    """Inference arguments, omitting NMS IoU for detectors that have no NMS."""
    inf = cand["infer"]
    kw = {"conf": inf["conf"], "max_det": inf["max_det"],
          "augment": inf["tta"], "verbose": False, "stream": True,
          "batch": PREDICT_BATCH}
    if inf.get("iou") is not None:
        kw["iou"] = inf["iou"]
    return kw


def _rows(pid, result):
    """Flatten one ultralytics Result into submission-shaped tuples."""
    b = result.boxes
    if b is None or len(b) == 0:
        return []
    xyxy = b.xyxy.cpu().numpy()
    return [(pid, int(c), float(s), float(x1), float(y1), float(x2), float(y2))
            for (x1, y1, x2, y2), c, s in
            zip(xyxy, b.cls.cpu().numpy(), b.conf.cpu().numpy())]


def frame_index(root, split, ids):
    """Map each id to its band-planar frame; no de-mosaicing needed at run time."""
    d = root / split / "images"
    return {pid: d / f"{pid}.png" for pid in ids}


def first_conv(net):
    """The stem convolution -- the only layer whose shape depends on band count."""
    import torch.nn as nn
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            return m
    return None


def pretrained_stem_weight(name):
    """The 3-channel stem kernel from the COCO checkpoint we start from."""
    import torch
    path = Path(f"{name}.pt")
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = ckpt.get("model") if isinstance(ckpt, dict) else ckpt
    if net is None:
        return None
    conv = first_conv(net)
    if conv is None or conv.in_channels != 3:
        return None
    return conv.weight.detach().float().clone()


# The wrapper is defined at module level, not built inside a factory: the
# trained model is pickled into every checkpoint, and pickle can only name a
# class that its module exposes. torch is guaranteed present wherever this runs,
# but the guard keeps the file importable for offline inspection.
try:
    import torch.nn as _nn
except ImportError:                                  # pragma: no cover
    _nn = None

if _nn is not None:
    class SpectralFront(_nn.Module):
        """The band reduction, wrapped *around* the pretrained first block.

        Replacing the block's inner Conv2d instead was the earlier attempt and
        it crashed at validation: ultralytics' fuse() reaches into every Conv
        block for m.conv.weight, and a Sequential has no .weight. Wrapping the
        whole block leaves its internals exactly as ultralytics expects and
        keeps the pretrained convolution reachable for fusion.
        """

        def __init__(self, front, block):
            super().__init__()
            self.front = front
            self.block = block

        def forward(self, x):
            return self.block(self.front(x))
else:                                                # pragma: no cover
    SpectralFront = None

try:
    import torch as _torch
    import torch.nn.functional as _F
    from ultralytics.models.utils.loss import RTDETRDetectionLoss as _RTDETRLoss
    from ultralytics.utils.metrics import bbox_iou as _bbox_iou
except ImportError:                                  # pragma: no cover
    _RTDETRLoss = None

if _RTDETRLoss is not None:
    class IoUKindDETRLoss(_RTDETRLoss):
        """RT-DETR's loss with the overlap term made selectable.

        Its box loss is L1 on the coordinates plus 1 - GIoU. The error
        decomposition says what is left of the score is box tightness on
        elongated objects -- median matched IoU 0.864, and AP correlates with
        a class's median aspect ratio at r = -0.65 net of frequency. CIoU is
        GIoU plus a centre-distance and an aspect-ratio consistency penalty,
        which is that failure written down; DIoU is the same without the
        aspect term, which is how to tell which half is doing the work.

        Defined at module level because the model is pickled into every
        checkpoint and pickle cannot name a class built inside a function.
        """

        iou_flag: dict = {"GIoU": True}

        def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=""):
            name_bbox, name_giou = f"loss_bbox{postfix}", f"loss_giou{postfix}"
            if not len(gt_bboxes):
                z = _torch.tensor(0.0, device=self.device)
                return {name_bbox: z, name_giou: z.clone()}
            n = len(gt_bboxes)
            iou = _bbox_iou(pred_bboxes, gt_bboxes, xywh=True, **self.iou_flag)
            return {
                name_bbox: (self.loss_gain["bbox"]
                            * _F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / n).squeeze(),
                name_giou: (self.loss_gain["giou"] * (1.0 - iou).sum() / n).squeeze(),
            }
else:                                                # pragma: no cover
    IoUKindDETRLoss = None

if SpectralFront is not None:
    # The trained model is pickled into every checkpoint, and pickle stores the
    # class by module *name*. This script is __main__ on Kaggle but not
    # necessarily anywhere else, and a checkpoint that names a module the
    # loading process does not have is unloadable -- which would strand a
    # chunked run at its first resume. Registering a stable module and pointing
    # the class at it makes the reference the same wherever the script runs.
    import sys as _sys
    import types as _types
    _mod = _sys.modules.setdefault("hod26_kernel", _types.ModuleType("hod26_kernel"))
    _mod.SpectralFront = SpectralFront
    SpectralFront.__module__ = "hod26_kernel"
    if IoUKindDETRLoss is not None:
        _mod.IoUKindDETRLoss = IoUKindDETRLoss
        IoUKindDETRLoss.__module__ = "hod26_kernel"


def install_spectral_adapter(net, n_bands, projection=None, ckpt_name=None,
                             srf_k=0, srf_width=2.0, stem_src=None):
    """Put a band mixer in front of an untouched pretrained first block.

    Two shapes, selected by ``srf_k``:

        srf_k == 0   16 --[trainable 1x1, init P]--> 3 --> pretrained block
        srf_k == k   16 --[fixed SRF bank]--> k --[trainable 1x1]--> 3 --> block

    The single-stage form starts from a signed discriminant, and that is what
    went wrong: every such projection is a weighted *difference* whose positive
    and negative halves cancel (|sum w| / sum |w| = 0.000), so it amplifies
    noise -- the rendered frame keeps only 0.537 of its gradient through a 3x3
    blur against pseudo_rgb's 0.744. Being trainable does not save it, because
    the first epochs still feed the backbone a frame it cannot read.

    The two-stage form fixes the first reduction to be an *average*: a
    non-negative, row-normalised Gaussian SRF bank, which has variance ~1/n
    rather than amplifying noise, and renders at 0.759. Only the second stage
    trains. Averaging also costs discrimination -- the three outputs correlate
    at 0.98 or above -- and recovering it is exactly what the trainable stage is
    for: a signed mix of already-denoised channels is what any CNN's first layer
    computes over R, G and B.

    The network behind it is built with three input channels, so every one of
    its pretrained tensors transfers, the stem included. That is the adapter
    shape the multispectral transfer literature converges on (UniRGB-IR
    2404.17360, SpectralX 2508.01731), and it matters because the measured gap
    between 12 and 202 bands under a pretrained backbone is small (TerraMind
    2603.06690) -- the pretrained spatial prior is worth more than the extra
    spectral resolution, so the prior is the thing to protect.
    """
    import numpy as _np
    import torch
    import torch.nn as nn

    block = net.model[0]
    if isinstance(block, SpectralFront):
        return False                      # already installed (resumed model)
    conv = first_conv(block)
    if conv is None or conv.in_channels != 3:
        log(f"  adapter skipped: first block reads "
            f"{None if conv is None else conv.in_channels} channels, not 3")
        return False

    if srf_k:
        bank = gaussian_srf_bank(n_bands, int(srf_k), float(srf_width))
        P = _np.asarray(srf_to_rgb_init(bank), dtype=_np.float32)
    else:
        bank = None
        P = _np.asarray(projection, dtype=_np.float32)
        if P.shape[1] != n_bands:
            log(f"  adapter skipped: projection maps {P.shape[1]} bands, not {n_bands}")
            return False
    out_ch, in_ch = P.shape[0], (bank.shape[0] if bank is not None else n_bands)
    if out_ch != 3:
        log(f"  adapter skipped: mixer produces {out_ch} channels, not 3")
        return False

    mixer = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
    with torch.no_grad():
        mixer.weight.copy_(torch.from_numpy(P).view(out_ch, in_ch, 1, 1))
    front = [mixer]
    if bank is not None:
        srf = nn.Conv2d(n_bands, in_ch, kernel_size=1, bias=False)
        with torch.no_grad():
            srf.weight.copy_(torch.from_numpy(
                _np.asarray(bank, dtype=_np.float32)).view(in_ch, n_bands, 1, 1))
        # Fixed on purpose. Its job is to guarantee an average; letting gradient
        # descent touch it lets the weights go negative, which is the failure
        # this stage exists to rule out.
        srf.weight.requires_grad_(False)
        front.insert(0, srf)

    wrapper = SpectralFront(nn.Sequential(*front), block).to(conv.weight.device)
    # parse_model tags every layer with its index and input sources, and the
    # forward loop reads them off the module it is holding.
    for attr in ("i", "f", "type", "np"):
        if hasattr(block, attr):
            setattr(wrapper, attr, getattr(block, attr))
    net.model[0] = wrapper
    # Keep the initial mixing so the run can prove the adapter actually trained.
    # Installing it in the wrong place leaves the optimizer without its
    # parameters, and it would then behave as a fixed projection while
    # reporting itself as the adapter -- a failure with no symptom.
    # Straight into __dict__: assigning a Module through nn.Module.__setattr__
    # would register the mixer a second time and put a duplicate key in every
    # state_dict, including the checkpoints a resume has to match exactly.
    net.__dict__["_hod26_mixer"] = mixer
    net.__dict__["_hod26_mixer_init"] = mixer.weight.detach().clone()
    if bank is not None:
        log(f"  spectral adapter: fixed {n_bands}->{in_ch} SRF bank (non-negative, "
            f"width {srf_width}) then trainable {in_ch}->3 1x1 in front of an "
            f"unchanged pretrained {type(block).__name__}")
    else:
        log(f"  spectral adapter: trainable {n_bands}->3 1x1 in front of an "
            f"unchanged pretrained {type(block).__name__}")
    return True


def adapter_drift(model):
    """How far the mixer moved from its initialisation. Zero means it never trained."""
    for net in (getattr(model, "model", None), model):
        mixer = getattr(net, "_hod26_mixer", None)
        init = getattr(net, "_hod26_mixer_init", None)
        if mixer is None or init is None:
            continue
        w = mixer.weight.detach().to(init.device)
        return {"l2": float((w - init).norm()),
                "rel": float((w - init).norm() / (init.norm() + 1e-12)),
                "init_norm": float(init.norm())}
    return None


def cls_head_params(net, nc):
    """The classifier-logit parameters of either head, one row per class.

    YOLO keeps them in the Detect head's cv3 branch, RT-DETR in the decoder's
    score_head and class_embed. Requiring nc rows excludes the box branch and
    RT-DETR's denoising embedding, which has nc + 1.
    """
    for name, prm in net.named_parameters():
        head = (".cv3." in name or "one2one_cv3" in name
                or "score_head" in name or "class_embed" in name)
        if head and prm.shape[0] == nc:
            yield name, prm


def perturb_derived_rows(net, names, scale=0.05):
    """Separate head rows that inherited the same COCO logits.

    apple and apple_plastic both start from COCO's apple, which is the point --
    they look identical and only the spectrum tells them apart. But identical
    rows produce identical logits, so the two classes start entangled and the
    loss has to pull them apart from exactly equal footing. A perturbation of a
    twentieth of the row's own scale costs nothing against the prior and gives
    the gradient a direction to work in from the first step.
    """
    import torch

    derived = [i for i, n in names.items() if n in COCO_PRIOR_DERIVED]
    if not derived:
        return 0
    g = torch.Generator(device="cpu").manual_seed(CACHE_SEED)
    touched = 0
    with torch.no_grad():
        for _, prm in cls_head_params(net, len(names)):
            ref = float(prm.detach().float().std()) or 1.0
            for i in derived:
                noise = torch.randn(prm[i].shape, generator=g).to(prm.device, prm.dtype)
                prm[i] += scale * ref * noise
            touched += 1
    log(f"  perturbed {len(derived)} derived class rows across {touched} head tensors")
    return touched


def install_bbox_loss(net, kind: str, nc: int, loss_gain: dict | None = None):
    """Swap the overlap term RT-DETR regresses against.

    ultralytics' bbox_iou already implements every variant, so the change is
    which flag it is passed. nc comes from the caller because the model does
    not carry it yet -- set_model_attributes attaches it after get_model runs.
    """
    kinds = {"GIoU": {"GIoU": True}, "DIoU": {"DIoU": True},
             "CIoU": {"CIoU": True}, "IoU": {}}
    if kind not in kinds:
        raise ValueError(f"unknown bbox loss {kind!r}; have {sorted(kinds)}")

    crit = IoUKindDETRLoss(nc=int(nc), use_vfl=True)
    crit.iou_flag = kinds[kind]
    if loss_gain:
        crit.loss_gain.update(loss_gain)
    net.criterion = crit
    log(f"  box loss: {kind}" + (f", gains {loss_gain}" if loss_gain else ""))
    return True


def restore_state(net, src):
    """Copy every shape-compatible tensor from a checkpoint module into ``net``.

    Used instead of ultralytics' own load() when resuming an adapted model.
    That path assumes the first layer is reachable as model.0.conv.weight and
    raises a KeyError on a wrapped stem; it is also a name-matched intersect,
    which would silently drop the front end rather than fail. Here the two
    modules have identical structure, so anything short of a full restore is a
    bug and says so.
    """
    import torch.nn as nn
    if not isinstance(src, nn.Module):
        raise RuntimeError("resume expected a checkpoint module to restore from")
    own, sd = net.state_dict(), src.float().state_dict()
    ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    net.load_state_dict(ok, strict=False)
    missing = sorted(set(own) - set(ok))
    if missing:
        raise RuntimeError(f"resume restored {len(ok)}/{len(own)} tensors; "
                           f"{len(missing)} missing, first: {missing[:5]}")
    log(f"  resumed {len(ok)}/{len(own)} tensors from the checkpoint")
    return len(ok)


def hod26_trainer(base_cls, adapter=None, coco_prior=True, schedule_epochs=0,
                  bbox_loss="GIoU", loss_gain=None, is_rtdetr=True):
    """A trainer that seeds the head from COCO by name and installs the adapter.

    Both have to happen inside get_model, and for the same reason: ultralytics
    builds the optimizer inside _setup_train, *before* on_pretrain_routine_end
    fires, so a module swapped in from that callback leaves the optimizer
    holding the old stem's parameters and the mixer's new ones never receive an
    update -- it would silently train as a fixed projection while reporting
    itself as the adapter. on_pretrain_routine_start is earlier still, and
    trainer.model does not exist yet there.

    The head prior rides the mechanism ultralytics already has. Its
    _remap_cls_by_names copies a pretrained classifier row into any target class
    whose *name* matches, and set_model_names_for_load is the hook that decides
    which names it sees. Renaming the targets to their COCO ancestors for the
    duration of the load lifts the inheritance from 4 of 18 classes to 12 --
    people alone is 1095 boxes that would otherwise start from noise, because
    COCO calls it person. The real names go back on immediately afterwards, so
    nothing downstream sees the substitution.
    """

    class HOD26Trainer(base_cls):
        def _setup_scheduler(self):
            """Shape the LR curve over the whole run, not over this session.

            A chunked run stops each session at its own epoch count, and
            ultralytics builds the schedule from that count -- so a first
            session of 9 out of 18 would decay all the way to lrf by its last
            epoch and the second would restart near half the peak. Building the
            curve over the full length instead makes the two halves join where
            a single uninterrupted run would have been.
            """
            real = self.epochs
            if schedule_epochs and schedule_epochs > real:
                self.epochs = schedule_epochs
            try:
                super()._setup_scheduler()
            finally:
                self.epochs = real

        def check_resume(self, overrides):
            super().check_resume(overrides)
            # check_resume replaces args wholesale with the checkpoint's, and
            # epochs is not on its override whitelist. A chunked run raises the
            # total each session, so without this the resume asserts that
            # training already finished.
            if self.resume and overrides.get("epochs"):
                self.args.epochs = overrides["epochs"]

        def set_model_names_for_load(self, model):
            parent = getattr(super(), "set_model_names_for_load", None)
            model = parent(model) if parent else model
            if (not coco_prior or getattr(self, "resume", False)
                    or not isinstance(getattr(model, "names", None), dict)):
                return model
            self._hod26_names = dict(model.names)
            model.names = {i: COCO_PRIOR.get(n, n) for i, n in model.names.items()}
            n_mapped = sum(1 for i, n in model.names.items()
                           if n != self._hod26_names[i])
            log(f"  head prior: renamed {n_mapped} classes to their COCO ancestors "
                f"for the weight load")
            return model

        def get_model(self, cfg=None, weights=None, verbose=True):
            ch = self.data.get("channels")
            resuming = bool(getattr(self, "resume", False)) and adapter
            if adapter:
                # Build and load a plain 3-channel network. A 16-channel build
                # cannot inherit the stem -- the shapes differ -- and then sits
                # behind 32.8M pretrained parameters tuned for what that stem
                # produced. The adapter is prepended afterwards, so the bands
                # arrive as 16 and the network still reads three.
                self.data["channels"] = 3
            try:
                # Resuming loads the checkpoint below, after the adapter is
                # back in place; letting ultralytics do it here would trip over
                # a stem it cannot address.
                net = super().get_model(cfg=cfg, verbose=verbose,
                                        weights=None if resuming else weights)
            finally:
                self.data["channels"] = ch
            real = getattr(self, "_hod26_names", None)
            if real is not None:
                net.names = real
                perturb_derived_rows(net, real)
            # RTDETRDetectionLoss only: the YOLO head computes its box loss
            # somewhere else entirely, so this override would silently miss.
            if is_rtdetr and (bbox_loss not in ("", "GIoU") or loss_gain):
                install_bbox_loss(net, bbox_loss or "GIoU",
                                  self.data["nc"], loss_gain)
            if adapter:
                install_spectral_adapter(net, **adapter)
                # Resuming rebuilds the model from its yaml, which has no
                # adapter in it. Only once the front end is back does the
                # checkpoint's every tensor have a key to land on.
                if resuming:
                    restore_state(net, weights)
            return net

    return HOD26Trainer


def replace_module(net, target, replacement):
    """Swap one module in place, wherever it sits in the tree."""
    for parent in net.modules():
        for attr, child in list(vars(parent).get("_modules", {}).items()):
            if child is target:
                parent._modules[attr] = replacement
                return True
    return False


def attach_spectral_stem_init(model, name, n_bands, projection):
    """Seed a multi-band stem from the pretrained RGB stem via the projection.

    A 16-channel stem cannot inherit COCO weights -- the shapes differ -- so it
    starts from noise, and that cost 0.043 mAP against a 3-channel run in the
    first GPU round. Composing the pretrained kernel with the discriminant
    projection fixes the initialization instead of the architecture:

        W16[o, b] = sum_c W3[o, c] * P[c, b]

    makes the stem's initial response identical to the pretrained stem reading
    P @ x, so training starts from a pretrained filter bank looking at the
    spectral discriminant rather than from scratch, and is then free to move
    beyond the three dimensions the projection can carry.

    Runs on on_pretrain_routine_start: by then ultralytics has built the model
    and transferred every tensor whose shape matched, so this fills the one that
    could not without being overwritten afterwards.
    """
    import numpy as _np
    import torch

    def hook(trainer):
        conv = first_conv(trainer.model)
        if conv is None or conv.in_channels != n_bands:
            return
        w3 = pretrained_stem_weight(name)
        if w3 is None or w3.shape[0] != conv.weight.shape[0]:
            log("  spectral stem init skipped: no usable pretrained stem")
            return
        P = torch.from_numpy(_np.asarray(projection, dtype=_np.float32))
        if P.shape != (w3.shape[1], n_bands):
            log(f"  spectral stem init skipped: projection {tuple(P.shape)} does not "
                f"map {n_bands} bands to {w3.shape[1]}")
            return
        w16 = torch.einsum("ocij,cb->obij", w3, P)
        # Preserve the pretrained layer's output scale: the projection's rows are
        # normalised for interpretability, not to keep activations in range.
        w16 *= w3.std() / (w16.std() + 1e-12)
        with torch.no_grad():
            conv.weight.copy_(w16.to(conv.weight.dtype).to(conv.weight.device))
        log(f"  spectral stem init: {tuple(conv.weight.shape)} seeded from the "
            f"pretrained {tuple(w3.shape)} stem via the {P.shape[0]}x{P.shape[1]} projection")

    model.add_callback("on_pretrain_routine_start", hook)


def base_trainer(name):
    """The trainer class ultralytics would have used for this model."""
    if name.startswith("rtdetr"):
        from ultralytics.models.rtdetr.train import RTDETRTrainer
        return RTDETRTrainer
    from ultralytics.models.yolo.detect import DetectionTrainer
    return DetectionTrainer


def build_model(name, weights=None):
    """Instantiate the detector. RT-DETR has its own model class in ultralytics."""
    from ultralytics import RTDETR, YOLO
    cls = RTDETR if name.startswith("rtdetr") else YOLO
    return cls(weights or f"{name}.pt")


# ------------------------------------------------- chunked runs & logging ---
# A Kaggle session is capped at 12 hours and its /kaggle/working is kept only
# when the kernel exits cleanly, so a run longer than that has to be split into
# sessions that each finish on purpose. Each session leaves last.pt, results.csv
# and metrics.jsonl as its output; the next lists it in kernel_sources, finds
# them under /kaggle/input and continues. A resume that silently restarts from
# epoch 0 looks like a slow run rather than a failure, which is why the starting
# epoch is logged explicitly.

def find_checkpoint(tag):
    """The previous session's last.pt, attached as another kernel's output."""
    for base in sorted(INPUT.glob("*")):
        if _looks_like_dataset(base):
            continue          # the planar frames, not a previous session
        for cand in (base / f"{tag}_last.pt", base / "last.pt"):
            if cand.exists():
                return cand
    return None


def stage_checkpoint(tag):
    """Put a previous session's state where ultralytics expects to resume from.

    ultralytics rebuilds save_dir from the checkpoint's own args, which is the
    same scratch runs/<tag> this session uses, and appends to the results.csv
    already there. Both files therefore have to be back in place before
    training starts.
    """
    src = find_checkpoint(tag)
    if src is None:
        return None
    run = RUNS / tag
    (run / "weights").mkdir(parents=True, exist_ok=True)
    last = run / "weights" / "last.pt"
    shutil.copy2(src, last)
    for name, dst in ((f"{tag}_best.pt", run / "weights" / "best.pt"),
                      (f"{tag}_results.csv", run / "results.csv"),
                      ("results.csv", run / "results.csv"),
                      (f"{tag}_metrics.jsonl", WORK / f"{tag}_metrics.jsonl"),
                      ("metrics.jsonl", WORK / f"{tag}_metrics.jsonl")):
        f = src.parent / name
        if f.exists() and not dst.exists():
            shutil.copy2(f, dst)
    # best.pt matters as much as last.pt here. best_fitness is restored from the
    # checkpoint, so a session whose epochs never beat the previous session's
    # best writes no best.pt at all -- and then the session that submits has
    # none to submit from.
    log(f"resuming from {src} -> {last}")
    return str(last)


def snapshot_for_resume(tag, run):
    """Copy last.pt out while it still carries optimizer state.

    ultralytics strips the optimizer, the EMA and the epoch number from last.pt
    once training ends, which is exactly what a resume needs -- a stripped
    checkpoint restarts from epoch 0 with a fresh schedule and reports itself as
    a normal run. So the copy is taken per epoch, from on_model_save, before the
    strip can reach it.
    """
    for src, dst in ((run / "weights" / "last.pt", WORK / f"{tag}_last.pt"),
                     (run / "results.csv", WORK / f"{tag}_results.csv")):
        if src.exists():
            shutil.copy2(src, dst)


def keep_for_resume(tag):
    """Copy the remaining outputs into this kernel's output root."""
    run = RUNS / tag
    src = run / "weights" / "best.pt"
    if src.exists():
        shutil.copy2(src, WORK / f"{tag}_best.pt")
    for f in (WORK / f"{tag}_last.pt", WORK / f"{tag}_best.pt",
              WORK / f"{tag}_results.csv", WORK / f"{tag}_metrics.jsonl"):
        if f.exists():
            log(f"kept {f.name} ({f.stat().st_size / 1e6:.1f} MB)")


def attach_epoch_log(model, tag, budget_seconds=0, reserve_seconds=300):
    """One flushed stdout line and one JSON record per epoch, and the clock.

    The clock guard is the other half of chunking. A session that overruns
    Kaggle's 12-hour cap is killed, and a killed kernel's /kaggle/working is not
    saved -- so an overrun costs both the GPU hours and the checkpoint they
    bought. Rather than sizing sessions from an estimate made in advance, each
    one stops itself when the epoch it is about to start will not fit, which
    makes the boundary a measurement instead of a guess.

    Setting trainer.stop is the clean way out: ultralytics checks it immediately
    after this callback (engine/trainer.py, "if self.stop: break"), so the run
    leaves through the same path a completed one does -- final_eval, save,
    return. The unstripped last.pt is already copied out from on_model_save.


    A pushed kernel is a batch job: its log cannot be fetched through the API
    until it finishes, so the only thing that makes a long run watchable is what
    it prints to the console its web page streams. The JSON file is the one to
    plot from afterwards, and it is written incrementally so it survives a
    session that is killed rather than ending.
    """
    path = WORK / f"{tag}_metrics.jsonl"
    state = {}

    def record(trainer):
        try:
            _record(trainer)
        except Exception:
            # A logging fault must never end a nine-hour run.
            log(f"  epoch log failed: {traceback.format_exc(limit=2)}")

    def _record(trainer):
        def _f(x):
            try:
                return round(float(x), 6)
            except (TypeError, ValueError):
                return None

        m = {k: _f(v) for k, v in (trainer.metrics or {}).items()}
        rec = {
            "tag": tag,
            "epoch": int(trainer.epoch) + 1,
            "total_epochs": int(trainer.epochs),
            "seconds": _f(getattr(trainer, "epoch_time", None)),
            "lr": {k: _f(v) for k, v in (getattr(trainer, "lr", None) or {}).items()},
            "loss": {k: _f(v) for k, v in (getattr(trainer, "tloss", None) or {}).items()},
            "metrics": {k: v for k, v in m.items() if v is not None},
            "fitness": _f(getattr(trainer, "fitness", None)),
        }
        # final_eval re-fires this callback once for the best checkpoint, one
        # epoch past the end; marking it keeps the plotted curve honest. On a
        # completed run that record sits past total_epochs, but a run the clock
        # cut short ends below the total and its final_eval record looks like an
        # ordinary epoch -- so the guard's own flag is what identifies it there.
        rec["final_eval"] = (rec["epoch"] > rec["total_epochs"]
                             or bool(state.get("stopped")))
        try:
            box = trainer.validator.metrics.box
            rec["per_class"] = {trainer.data["names"][int(c)]: _f(a)
                                for c, a in zip(box.ap_class_index,
                                                box.maps[box.ap_class_index])}
        except Exception:
            rec["per_class"] = {}
        drift = adapter_drift(trainer)
        if drift:
            rec["adapter_drift"] = round(drift["rel"], 6)
            # The model handed back after training is the *best* checkpoint,
            # which may predate the last epoch, so its drift is not this run's.
            state["drift"] = drift
        with path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        loss = "/".join(f"{v:.3f}" for v in rec["loss"].values() if v is not None)
        log(f"  epoch {rec['epoch']}/{rec['total_epochs']}  loss {loss}  "
            f"mAP50 {m.get('metrics/mAP50(B)', float('nan')):.4f}  "
            f"mAP50-95 {m.get('metrics/mAP50-95(B)', float('nan')):.4f}  "
            f"lr {next(iter(rec['lr'].values()), float('nan')):.2e}  "
            f"{rec['seconds']:.0f}s"
            + (f"  drift {drift['rel']:.4%}" if drift else ""))

        if rec["final_eval"]:
            return
        state["last_epoch"] = rec["epoch"]
        if not budget_seconds:
            return
        elapsed = time.time() - T0
        # 15% headroom: epochs are not identical, and the one that overruns is
        # the one that costs the whole session.
        need = (rec["seconds"] or 0) * 1.15
        if elapsed + need + reserve_seconds > budget_seconds:
            trainer.stop = True
            state["stopped"] = True
            log(f"  stopping cleanly at epoch {rec['epoch']}: {elapsed / 3600:.2f}h "
                f"used of {budget_seconds / 3600:.2f}h, next epoch needs "
                f"~{need / 60:.0f} min and {reserve_seconds / 60:.0f} min is "
                f"reserved for finishing up. The next session resumes here.")

    def announce(trainer):
        log(f"  training starts at epoch {trainer.start_epoch + 1} of "
            f"{trainer.epochs} (resume={bool(trainer.resume)})")

    model.add_callback("on_fit_epoch_end", record)
    model.add_callback("on_train_start", announce)
    model.add_callback("on_model_save", lambda t: snapshot_for_resume(tag, Path(t.save_dir)))
    return state


def run_candidate(cand, index, train_ids, val_ids, anns, tag, budget_seconds=0,
                  reserve_seconds=300):

    root = SCRATCH / f"ds_{channels_key(cand)}"
    log(f"  scratch {SCRATCH} ({free_gb(SCRATCH):.1f} GB free), "
        f"output {WORK} ({free_gb(WORK):.1f} GB free)")
    yaml = materialize(cand, index, train_ids, val_ids, anns, root)
    tr, inf = cand["train"], cand["infer"]

    # Defensive clamp: a candidate that reached here without normalization must
    # not burn a GPU session on an argument ultralytics will reject.
    close_mosaic = min(tr.get("close_mosaic", 5), max(0, tr["epochs"] - 1))

    resume_from = stage_checkpoint(tag)
    model = build_model(tr["model"], resume_from)
    adapter = None
    if tr.get("in_channels", 3) > 3:
        # Remember which checkpoint this started from; the stem strategies need
        # to read its 3-channel kernel back after ultralytics rebuilds the model.
        try:
            model.model._hod26_ckpt = tr["model"]
        except AttributeError:
            pass
        # Which starting projection the adapter gets. Smoother starts trade
        # measured separability for robustness to a spectral shift; the adapter
        # is trainable, so this is a starting point, not a commitment. It is
        # ignored entirely when srf_k selects the two-stage front end.
        proj = PDA_PROJECTIONS.get(str(tr.get("adapter_penalty", "0")), LDA_16_TO_3)
        if tr.get("spectral_stem", "adapter") == "adapter":
            adapter = {"n_bands": tr["in_channels"], "projection": proj,
                       "ckpt_name": tr["model"], "srf_k": tr.get("srf_k", 0),
                       "srf_width": tr.get("srf_width", 2.0)}
        else:
            attach_spectral_stem_init(model, tr["model"], tr["in_channels"], proj)
    trainer_cls = hod26_trainer(base_trainer(tr["model"]), adapter=adapter,
                                coco_prior=tr.get("coco_prior", True),
                                schedule_epochs=int(tr.get("schedule_epochs", 0)),
                                bbox_loss=tr.get("bbox_loss", "GIoU"),
                                loss_gain=tr.get("loss_gain") or None,
                                is_rtdetr=tr["model"].startswith("rtdetr"))
    log_state = attach_epoch_log(model, tag, budget_seconds, reserve_seconds)
    results = model.train(
        data=str(yaml), epochs=tr["epochs"], imgsz=tr["imgsz"], batch=tr["batch"],
        lr0=tr["lr0"], mosaic=tr["mosaic"], close_mosaic=close_mosaic,
        hsv_h=tr["hsv_h"], hsv_s=tr["hsv_s"], hsv_v=tr["hsv_v"],
        fliplr=tr["fliplr"], scale=tr["scale"], cos_lr=tr.get("cos_lr", True),
        multi_scale=tr.get("multi_scale", False),
        warmup_epochs=tr.get("warmup_epochs", 3.0),
        project=str(RUNS), name=tag, exist_ok=True,
        verbose=False, plots=False, val=True, seed=0,
        amp=tr.get("amp", True), deterministic=tr.get("deterministic", True),
        resume=bool(resume_from), trainer=trainer_cls,
    )
    keep_for_resume(tag)

    # Score from the trainer's own validation pass rather than a second
    # inference pass of our own. Three reasons, in order of weight:
    #  - it is the only scorer that works at every channel count: ultralytics'
    #    predict() loader hands a 3-channel array to a 16-channel model, so a
    #    predict-based score cannot rank spectral candidates at all;
    #  - it is free, the pass already ran as part of training;
    #  - one scorer for every node keeps tree scores comparable, which is what
    #    the replay simulator depends on.
    # The pycocotools scorer still guards the final submission, where the
    # official protocol matters and the candidate is known to be 3-channel.
    drift = log_state.get("drift") or adapter_drift(model)
    if drift is not None:
        if drift["l2"] == 0.0:
            log("  !! adapter did NOT train: its weights are unchanged, so it ran "
                "as a fixed projection")
        else:
            log(f"  adapter trained: moved {drift['rel']:.4%} from its initialisation "
                f"(L2 {drift['l2']:.4f} of {drift['init_norm']:.4f})")

    box = results.box
    scores = {
        "mAP": float(box.map),          # mAP@[.5:.95], the competition's primary
        "mAP50": float(box.map50),
        "per_class": {CLASSES[int(c)]: float(a)
                      for c, a in zip(results.box.ap_class_index, box.maps[box.ap_class_index])}
        if getattr(box, "ap_class_index", None) is not None else {},
    }
    scores["adapter"] = drift
    # The epoch the run actually reached, which the clock guard can cut short.
    # Read from the trainer rather than counted from the log: the log's last
    # record is ultralytics re-validating the best checkpoint, which is not an
    # epoch, and getting this one too high would let the orchestrator call an
    # unfinished run done and never produce a submission.
    reached = getattr(getattr(model, "trainer", None), "epoch", None)
    scores["last_epoch"] = (int(reached) + 1 if reached is not None
                            else log_state.get("last_epoch", tr["epochs"]))
    weights = RUNS / tag / "weights" / "best.pt"
    return scores, [], (str(weights) if weights.exists() else None)


def predict_test(model, cand, test_dir, png_ids):
    """Run the trained model over the test set at cube resolution.

    The frames are staged into a directory and the *directory* is handed to
    predict(). Passing a list of paths instead looks equivalent and is not:
    ultralytics' check_source runs autocast_list over a list, which opens each
    file with PIL and returns a 3-channel RGB image -- so a 16-band model would
    be fed three bands, and nothing in the output would say so. A directory
    keeps the multi-page TIFF reader in the path.
    """
    sizes, of_path = {}, {}
    staging = SCRATCH / "test_images"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    for n, pid in enumerate(png_ids):
        img = build_channels(load_planar(test_dir / f"{pid}.png"), cand["channels"])
        sizes[pid] = (img.shape[1], img.shape[0])
        of_path[str(write_frame(staging / str(pid), img))] = pid
        if n % 250 == 0:
            log(f"  staged {n}/{len(png_ids)}")

    preds, seen = [], 0
    for r in model.predict(str(staging), **predict_kwargs(cand)):
        pid = of_path.get(str(Path(r.path).resolve()), of_path.get(str(r.path)))
        if pid is None:
            raise RuntimeError(f"prediction for an unexpected frame: {r.path}")
        preds.extend(_rows(pid, r))
        seen += 1
        if seen % 250 == 0:
            log(f"  predicted {seen}/{len(png_ids)}")
    if seen != len(png_ids):
        raise RuntimeError(f"predicted {seen} frames but staged {len(png_ids)}")
    return preds, sizes


def find_weights(name):
    """A checkpoint left behind by another kernel, mounted under /kaggle/input."""
    for base in sorted(INPUT.glob("*")):
        if _looks_like_dataset(base):
            continue
        direct = base / name
        if direct.exists():
            return direct
        for c in sorted(base.glob(f"**/{name}")):
            return c
    return None


def score_val_split(model, cand, root):
    """Score the held-out split with pycocotools, per class.

    This is the honest diagnostic. Training already validates every epoch, but
    with ultralytics' own metric, which measures 0.048 above the competition's
    on the very same frames -- and the bottleneck ranking is built from
    *per-class* AP, so a ruler that disagrees on the total can reorder what
    looks worst. Scoring the same way the leaderboard does keeps the thing we
    steer by and the thing we are scored on in the same units.
    """
    ann_dir = root / "train" / "annotations"
    ids = require_ids(sorted(int(p.stem) for p in ann_dir.glob("*.xml")), ann_dir)
    _, val_ids = split_ids(ids)
    # parse() takes image_id from the filename, the same id predict_test tags
    # its rows with.
    anns = [parse(ann_dir / f"{pid}.xml") for pid in val_ids]

    preds, _ = predict_test(model, cand, root / "train" / "images", val_ids)
    shutil.rmtree(SCRATCH / "test_images", ignore_errors=True)
    scored = evaluate(anns, preds, per_class=True)
    wide = evaluate(anns, preds, max_dets=300)
    log(f"held-out split, {len(val_ids)} frames, {len(preds)} boxes:")
    log(f"  maxDets=100 (what COCOeval reports): mAP={scored['mAP']:.4f} "
        f"mAP50={scored['mAP50']:.4f}")
    log(f"  maxDets=300 (what ultralytics counts): mAP={wide['mAP']:.4f} "
        f"mAP50={wide['mAP50']:.4f}")
    for name, ap in sorted(scored["per_class"].items(), key=lambda kv: kv[1]):
        log(f"    {name:16s} {ap:.4f}")
    # Keep the raw predictions: every question asked of them afterwards is then
    # a CPU question, not another GPU session.
    (WORK / "val_predictions.json").write_text(json.dumps(
        [[int(r[0]), int(r[1]), round(float(r[2]), 5)] + [round(float(x), 2) for x in r[3:]]
         for r in preds]))
    return {"frames": len(val_ids), "boxes": len(preds), **scored,
            "mAP_maxdet300": wide["mAP"], "mAP50_maxdet300": wide["mAP50"]}


def sweep_inference(model, cand, root, variants):
    """Measure the submission path itself, one inference setting at a time.

    The held-out score through this path (0.4076) sits 0.048 below what
    ultralytics' validator reported for the same checkpoint on the same frames
    (0.4556). That is either two metrics disagreeing, in which case nothing is
    lost, or this path producing worse boxes than the validator's, in which
    case 0.048 is being left on the table at every submission -- more than the
    whole distance between first place and fifteenth. Rather than reason about
    which, each setting is measured against the scorer that counts.

    The frames are staged once and reused, so each extra variant costs only its
    own forward pass.
    """
    ann_dir = root / "train" / "annotations"
    ids = require_ids(sorted(int(p.stem) for p in ann_dir.glob("*.xml")), ann_dir)
    _, val_ids = split_ids(ids)
    anns = [parse(ann_dir / f"{pid}.xml") for pid in val_ids]

    staging = SCRATCH / "test_images"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    of_path = {}
    for n, pid in enumerate(val_ids):
        img = build_channels(load_planar(root / "train" / "images" / f"{pid}.png"),
                             cand["channels"])
        of_path[str(write_frame(staging / str(pid), img))] = pid
        if n % 250 == 0:
            log(f"  staged {n}/{len(val_ids)}")

    out = {}
    for name, extra in variants.items():
        kw = {**predict_kwargs(cand), **extra}
        t0 = time.time()
        preds = []
        for r in model.predict(str(staging), **kw):
            pid = of_path.get(str(Path(r.path).resolve()), of_path.get(str(r.path)))
            preds.extend(_rows(pid, r))
        scored = evaluate(anns, preds, per_class=True)
        out[name] = {**scored, "boxes": len(preds),
                     "seconds": round(time.time() - t0, 1), "kwargs": {
                         k: v for k, v in extra.items()}}
        log(f"  {name:22s} mAP={scored['mAP']:.4f} mAP50={scored['mAP50']:.4f} "
            f"({len(preds)} boxes, {out[name]['seconds']:.0f}s)")
    shutil.rmtree(staging, ignore_errors=True)
    return out


def predict_test_set(model, cand, root):
    """Predict the competition's test frames and write submission.csv."""
    test_dir = root / "test" / "images"
    test_ids = require_ids(sorted(int(p.stem) for p in test_dir.glob("*.png")), test_dir)
    log(f"predicting {len(test_ids)} test images")
    preds, sizes = predict_test(model, cand, test_dir, test_ids)
    shutil.rmtree(SCRATCH / "test_images", ignore_errors=True)
    n = write(WORK / "submission.csv", preds, clip_to=sizes)
    log(f"wrote submission.csv: {n} rows over {len({p[0] for p in preds})} images")
    return {"rows": n, "images": len({p[0] for p in preds})}


def run_from_checkpoint(round_cfg):
    """Evaluate and/or submit a checkpoint another kernel already trained.

    Kept out of the training sessions on purpose. A training session that also
    predicted would spend part of its clock budget on inference and would only
    ever report its own metric; running this afterwards on the second GPU slot
    costs no wall clock and gives both numbers that matter -- the leaderboard's,
    and a per-class breakdown measured the same way -- after every session
    rather than once at the end.
    """
    sub = round_cfg["submit"]
    cand = sub["candidate"]
    w = find_weights(sub["weights_from"])
    if w is None:
        raise RuntimeError(f"no {sub['weights_from']} under {INPUT}; attached: "
                           f"{[p.name for p in sorted(INPUT.glob('*'))]}")
    log(f"from checkpoint {w} ({w.stat().st_size / 1e6:.0f} MB)")

    root = data_root()
    model = build_model(cand["train"]["model"], str(w))
    out = {"mode": "checkpoint", "weights": str(w), "candidate": cand}
    if sub.get("sweep"):
        out["sweep"] = sweep_inference(model, cand, root, sub["sweep"])
    if sub.get("score_val", True):
        out["val"] = score_val_split(model, cand, root)
    if sub.get("predict_test", True):
        out.update(predict_test_set(model, cand, root))
        out["predicted"] = True
    (WORK / "results.json").write_text(json.dumps(out, indent=2))


def run_submission(round_cfg):
    """Train one candidate at full fidelity and write submission.csv."""
    cand = round_cfg["submit"]["candidate"]
    root = data_root()
    ann_dir = root / "train" / "annotations"
    test_dir = root / "test" / "images"

    ids = require_ids(sorted(int(p.stem) for p in ann_dir.glob("*.xml")), ann_dir)
    train_ids, val_ids = split_ids(ids)
    if round_cfg["submit"].get("use_all_train", True):
        # Config was already selected on val; refit on everything for the final run.
        train_ids, val_ids = ids, val_ids[:60]   # a token val set keeps YOLO happy
    log(f"submission fit: {len(train_ids)} train / {len(val_ids)} val")

    anns = {pid: parse(ann_dir / f"{pid}.xml") for pid in set(train_ids) | set(val_ids)}
    index = frame_index(root, "train", sorted(set(train_ids) | set(val_ids)))

    # A chunked run does not know in advance which session will be the last one
    # -- the clock decides -- so "if_complete" lets the session that reaches the
    # target epoch be the one that predicts, and reserves the time to do it.
    target = int(cand["train"]["epochs"])
    want = round_cfg["submit"].get("predict", True)
    budget = float(round_cfg["submit"].get("session_hours", 0) or 0) * 3600
    reserve = 1800 if want else 300

    scores, _, weights = run_candidate(cand, index, train_ids, val_ids, anns, "final",
                                       budget_seconds=budget, reserve_seconds=reserve)
    note = " (optimistic: seen in training)" if round_cfg["submit"].get("use_all_train", True) else ""
    reached = int(scores.get("last_epoch") or 0)
    log(f"fit done at epoch {reached}/{target}; holdout mAP={scores['mAP']:.4f}{note}")

    predict = want is True or (want == "if_complete" and reached >= target)
    if not predict:
        # An unfinished session of a chunked run. Its whole job is to advance
        # the checkpoint; predicting 1000 frames here would cost GPU time and
        # produce a submission from a half-trained model.
        (WORK / "results.json").write_text(json.dumps({
            "mode": "submit", "rows": 0, "holdout": scores["mAP"],
            "last_epoch": reached, "epochs_to": target, "candidate": cand,
            "weights": weights, "predicted": False,
        }, indent=2))
        log(f"session ended at epoch {reached} of {target}: checkpoint saved, "
            f"prediction deferred")
        for d in SCRATCH.glob("ds_*"):
            shutil.rmtree(d, ignore_errors=True)
        return

    model = build_model(cand["train"]["model"], weights)
    test_ids = require_ids(sorted(int(p.stem) for p in test_dir.glob("*.png")), test_dir)
    log(f"predicting {len(test_ids)} test images")
    preds, sizes = predict_test(model, cand, test_dir, test_ids)

    shutil.rmtree(SCRATCH / "test_images", ignore_errors=True)
    n = write(WORK / "submission.csv", preds, clip_to=sizes)
    log(f"wrote submission.csv: {n} rows over {len({p[0] for p in preds})} images")
    (WORK / "results.json").write_text(json.dumps({
        "mode": "submit", "rows": n, "holdout": scores["mAP"],
        "last_epoch": reached, "epochs_to": target,
        "candidate": cand, "weights": weights, "predicted": True,
    }, indent=2))


def main():
    round_cfg = json.loads(Path(__file__).with_name("round.json").read_text()) \
        if Path(__file__).with_name("round.json").exists() else ROUND_CONFIG

    if round_cfg.get("submit"):
        if round_cfg["submit"].get("weights_from"):
            return run_from_checkpoint(round_cfg)
        return run_submission(round_cfg)

    root = data_root()
    ann_dir = root / "train" / "annotations"
    ids = require_ids(sorted(int(p.stem) for p in ann_dir.glob("*.xml")), ann_dir)
    train_ids, val_ids = split_ids(ids)
    anns = {pid: parse(ann_dir / f"{pid}.xml") for pid in ids}

    keep = round_cfg.get("proxy_classes")
    if keep:
        # Restrict the proxy to the frames holding a chosen set of classes.
        # The four weak classes live in 518 frames that share no frame with the
        # other fourteen, so a 300-frame random proxy gives stone_block about
        # four frames and can measure nothing about it. Selecting the regime
        # under test gives those classes enough support to rank a treatment.
        keep = set(keep)
        sel = {pid for pid, a in anns.items()
               if {CLASSES[b.cls_id] for b in a.boxes} & keep}
        train_ids = [i for i in train_ids if i in sel]
        val_ids = [i for i in val_ids if i in sel]
        log(f"restricted to frames containing {sorted(keep)}: "
            f"{len(train_ids)} train / {len(val_ids)} val")

    limit = round_cfg.get("proxy_train_images")
    if limit:
        train_ids = train_ids[:limit]
        val_ids = val_ids[:round_cfg.get("proxy_val_images", len(val_ids))]
    log(f"{len(train_ids)} train / {len(val_ids)} val images")
    index = frame_index(root, "train", train_ids + val_ids)

    results = []
    for cand_entry in round_cfg["candidates"]:
        node_id, cand = cand_entry["node_id"], cand_entry["candidate"]
        t0 = time.time()
        log(f"=== candidate {node_id}: {cand['channels']['mode']} / "
            f"{cand['train']['model']} / imgsz={cand['train']['imgsz']} ===")
        rec = {"node_id": node_id, "candidate": cand}
        try:
            scores, preds, weights = run_candidate(cand, index, train_ids, val_ids, anns, node_id)
            rec.update(score=scores["mAP"], diagnostics={
                "mAP50": scores["mAP50"], "per_class": scores["per_class"],
                "weights": weights, "adapter": scores.get("adapter"),
            })
            log(f"  -> mAP={scores['mAP']:.4f}  mAP50={scores['mAP50']:.4f}")
        except Exception:
            rec.update(score=None, error=traceback.format_exc()[-2000:])
            log(f"  !! failed:\n{rec['error']}")
        rec["cost_seconds"] = time.time() - t0
        results.append(rec)
        (WORK / "results.json").write_text(json.dumps(
            {"round": round_cfg.get("round"), "results": results}, indent=2))

    for d in SCRATCH.glob("ds_*"):
        shutil.rmtree(d, ignore_errors=True)
    log(f"round complete: {sum(r['score'] is not None for r in results)}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
