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
WORK = Path("/kaggle/working")
VAL_FRACTION = 0.2
PREDICT_BATCH = 32
CACHE_SEED = 20260918


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


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
    roots = [DATA, Path("/kaggle/input")]

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

    n_ch = 3
    for split, ids in (("train", train_ids), ("val", val_ids)):
        for pid in ids:
            cube = load_planar(index[pid])
            a = anns[pid]
            # The unaugmented frame is always written; validation is never
            # augmented, so the score keeps measuring the real distribution.
            img = build_channels(cube, cand["channels"])
            n_ch = img.shape[2]
            emit(root, split, str(pid), img, a.boxes, a)

            if split == "train" and wants_aug:
                for k in range(copies):
                    c2, b2 = augment_cube(cube, list(a.boxes), aug, donors, pool, rng)
                    emit(root, split, f"{pid}_a{k}", build_channels(c2, cand["channels"]), b2, a)

    for split, ids in (("train", train_ids), ("val", val_ids)):
        n = len(list((root / "images" / split).glob("*.png"))) + \
            len(list((root / "images" / split).glob("*.tiff")))
        expect = len(ids) * (1 + copies if split == "train" and wants_aug else 1)
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
          "augment": inf["tta"], "verbose": False, "stream": False}
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


def install_spectral_adapter(net, n_bands, projection, ckpt_name):
    """Put a trainable 1x1 band mixer in front of an untouched pretrained stem.

        W_mix[c, b] = P[c, b]      (48 parameters, initialised to the discriminant)

    The alternatives each give something up: a fixed 3-channel projection cannot
    adapt, and reparameterising the stem lets all 4608 of its weights drift from
    what COCO learned. This keeps the pretrained convolution exactly as trained
    and learns only the mixing, which is the adapter shape the multispectral
    transfer literature converges on (UniRGB-IR 2404.17360, SpectralX
    2508.01731). It matters because the measured gap between 12 and 202 bands
    under a pretrained backbone is small (TerraMind 2603.06690) -- the
    pretrained spatial prior is worth more than the extra spectral resolution,
    so the prior is the thing to protect.
    """
    import numpy as _np
    import torch
    import torch.nn as nn

    conv = first_conv(net)
    if conv is None or conv.in_channels != n_bands:
        return False
    P = _np.asarray(projection, dtype=_np.float32)
    if P.shape[1] != n_bands:
        log(f"  adapter skipped: projection maps {P.shape[1]} bands, not {n_bands}")
        return False
    out_ch = P.shape[0]

    pre = pretrained_stem_weight(ckpt_name)
    if pre is None or pre.shape[1] != out_ch or pre.shape[0] != conv.out_channels:
        log(f"  adapter skipped: pretrained stem {None if pre is None else tuple(pre.shape)} "
            f"does not fit a {out_ch}-channel mixer into {conv.out_channels} filters")
        return False

    mixer = nn.Conv2d(n_bands, out_ch, kernel_size=1, bias=False)
    stem = nn.Conv2d(out_ch, conv.out_channels, conv.kernel_size, conv.stride,
                     conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        mixer.weight.copy_(torch.from_numpy(P).view(out_ch, n_bands, 1, 1))
        stem.weight.copy_(pre)
        if conv.bias is not None:
            stem.bias.copy_(conv.bias)
    dev = conv.weight.device
    if not replace_module(net, conv, nn.Sequential(mixer, stem).to(dev)):
        log("  adapter skipped: could not locate the stem in the module tree")
        return False
    log(f"  spectral adapter: trainable {n_bands}->{out_ch} 1x1 in front of an "
        f"unchanged {tuple(pre.shape)} pretrained stem")
    return True


def adapter_trainer(base_cls, n_bands, projection, ckpt_name):
    """A trainer whose get_model returns a model that already has the adapter.

    The adapter cannot be installed from a callback. ultralytics builds the
    optimizer inside _setup_train, *before* on_pretrain_routine_end fires, so a
    module swapped in from that callback leaves the optimizer holding the old
    stem's parameters and the mixer's 48 new ones never receive an update -- it
    would silently train as a fixed projection while reporting itself as the
    adapter. on_pretrain_routine_start is earlier still, and trainer.model does
    not exist yet there. Overriding get_model puts the adapter in place before
    the optimizer is ever constructed.
    """

    class AdapterTrainer(base_cls):
        def get_model(self, cfg=None, weights=None, verbose=True):
            net = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            install_spectral_adapter(net, n_bands, projection, ckpt_name)
            return net

    return AdapterTrainer


def pretrained_stem_weight_from(net, want_in):
    """The COCO stem kernel, read back from the checkpoint the run started from."""
    name = getattr(net, "_hod26_ckpt", None)
    w = pretrained_stem_weight(name) if name else None
    if w is not None and w.shape[1] == want_in:
        return w
    return None


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


def run_candidate(cand, index, train_ids, val_ids, anns, tag):

    root = WORK / f"ds_{channels_key(cand)}"
    yaml = materialize(cand, index, train_ids, val_ids, anns, root)
    tr, inf = cand["train"], cand["infer"]

    # Defensive clamp: a candidate that reached here without normalization must
    # not burn a GPU session on an argument ultralytics will reject.
    close_mosaic = min(tr.get("close_mosaic", 5), max(0, tr["epochs"] - 1))

    model = build_model(tr["model"])
    trainer_cls = None
    if tr.get("in_channels", 3) > 3:
        # Remember which checkpoint this started from; the stem strategies need
        # to read its 3-channel kernel back after ultralytics rebuilds the model.
        try:
            model.model._hod26_ckpt = tr["model"]
        except AttributeError:
            pass
        # Which starting projection the adapter gets. Smoother starts trade
        # measured separability for robustness to a spectral shift; the adapter
        # is trainable, so this is a starting point, not a commitment.
        proj = PDA_PROJECTIONS.get(str(tr.get("adapter_penalty", "0")), LDA_16_TO_3)
        if tr.get("spectral_stem", "adapter") == "adapter":
            trainer_cls = adapter_trainer(base_trainer(tr["model"]), tr["in_channels"],
                                          proj, tr["model"])
        else:
            attach_spectral_stem_init(model, tr["model"], tr["in_channels"], proj)
    results = model.train(
        data=str(yaml), epochs=tr["epochs"], imgsz=tr["imgsz"], batch=tr["batch"],
        lr0=tr["lr0"], mosaic=tr["mosaic"], close_mosaic=close_mosaic,
        hsv_h=tr["hsv_h"], hsv_s=tr["hsv_s"], hsv_v=tr["hsv_v"],
        fliplr=tr["fliplr"], scale=tr["scale"], cos_lr=tr.get("cos_lr", True),
        project=str(WORK / "runs"), name=tag, exist_ok=True,
        verbose=False, plots=False, val=True, seed=0,
        amp=tr.get("amp", True), deterministic=tr.get("deterministic", True),
        **({"trainer": trainer_cls} if trainer_cls else {}),
    )

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
    box = results.box
    scores = {
        "mAP": float(box.map),          # mAP@[.5:.95], the competition's primary
        "mAP50": float(box.map50),
        "per_class": {CLASSES[int(c)]: float(a)
                      for c, a in zip(results.box.ap_class_index, box.maps[box.ap_class_index])}
        if getattr(box, "ap_class_index", None) is not None else {},
    }
    weights = WORK / "runs" / tag / "weights" / "best.pt"
    return scores, [], (str(weights) if weights.exists() else None)


def predict_test(model, cand, test_dir, png_ids):
    """Run the trained model over the test set at cube resolution."""
    inf = cand["infer"]
    preds, sizes, staged = [], {}, {}
    staging = WORK / "test_images"
    staging.mkdir(parents=True, exist_ok=True)
    for n, pid in enumerate(png_ids):
        img = build_channels(load_planar(test_dir / f"{pid}.png"), cand["channels"])
        sizes[pid] = (img.shape[1], img.shape[0])
        staged[pid] = write_frame(staging / str(pid), img)
        if n % 250 == 0:
            log(f"  staged {n}/{len(png_ids)}")

    for lo in range(0, len(png_ids), PREDICT_BATCH):
        ids = png_ids[lo:lo + PREDICT_BATCH]
        chunk = [str(staged[pid]) for pid in ids]
        for pid, r in zip(ids, model.predict(chunk, **predict_kwargs(cand))):
            preds.extend(_rows(pid, r))
        log(f"  predicted {min(lo + PREDICT_BATCH, len(png_ids))}/{len(png_ids)}")
    return preds, sizes


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
    scores, _, weights = run_candidate(cand, index, train_ids, val_ids, anns, "final")
    log(f"fit done; holdout mAP={scores['mAP']:.4f} (optimistic: seen in training)")

    model = build_model(cand["train"]["model"], weights)
    test_ids = require_ids(sorted(int(p.stem) for p in test_dir.glob("*.png")), test_dir)
    log(f"predicting {len(test_ids)} test images")
    preds, sizes = predict_test(model, cand, test_dir, test_ids)

    n = write(WORK / "submission.csv", preds, clip_to=sizes)
    log(f"wrote submission.csv: {n} rows over {len({p[0] for p in preds})} images")
    (WORK / "results.json").write_text(json.dumps({
        "mode": "submit", "rows": n, "holdout": scores["mAP"],
        "candidate": cand, "weights": weights,
    }, indent=2))


def main():
    round_cfg = json.loads(Path(__file__).with_name("round.json").read_text()) \
        if Path(__file__).with_name("round.json").exists() else ROUND_CONFIG

    if round_cfg.get("submit"):
        return run_submission(round_cfg)

    root = data_root()
    ann_dir = root / "train" / "annotations"
    ids = require_ids(sorted(int(p.stem) for p in ann_dir.glob("*.xml")), ann_dir)
    train_ids, val_ids = split_ids(ids)

    limit = round_cfg.get("proxy_train_images")
    if limit:
        train_ids = train_ids[:limit]
        val_ids = val_ids[:round_cfg.get("proxy_val_images", len(val_ids))]
    log(f"{len(train_ids)} train / {len(val_ids)} val images")

    anns = {pid: parse(ann_dir / f"{pid}.xml") for pid in train_ids + val_ids}
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
                "weights": weights,
            })
            log(f"  -> mAP={scores['mAP']:.4f}  mAP50={scores['mAP50']:.4f}")
        except Exception:
            rec.update(score=None, error=traceback.format_exc()[-2000:])
            log(f"  !! failed:\n{rec['error']}")
        rec["cost_seconds"] = time.time() - t0
        results.append(rec)
        (WORK / "results.json").write_text(json.dumps(
            {"round": round_cfg.get("round"), "results": results}, indent=2))

    for d in WORK.glob("ds_*"):
        shutil.rmtree(d, ignore_errors=True)
    log(f"round complete: {sum(r['score'] is not None for r in results)}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
