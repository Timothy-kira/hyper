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
    if mode == "s3t_level":
        # S3T's input: the 16 bands, log radiance, per-frame scaled and
        # sub-pixel aligned, quantised on a fixed map the front end inverts.
        return level_u8(cube)
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


def paste_pool(index, anns, ids, names):
    """{cls_id: [(frame path, box)]} of the named classes' instances, for crowd_paste."""
    want = {CLASSES.index(n) for n in names}
    out = {c: [] for c in want}
    for pid in ids:
        for b in anns[pid].boxes:
            if b.cls_id in want:
                out[b.cls_id].append((str(index[pid]), (b.x1, b.y1, b.x2, b.y2)))
    return out


def augment_cube(cube, boxes, aug, donors, pool, rng, paste=None):
    """Apply the spectral and spatial operators a candidate asked for."""
    if aug.get("sg_window"):
        # Smoothing is along wavelength, which is not mosaic order (bands 4 and
        # 11 are out of place); sg_chain smooths along the measured chain.
        cube = savgol_spectral(cube, aug["sg_window"], aug["sg_polyorder"],
                               order=BAND_CHAIN if aug.get("sg_chain") else None)
    if aug.get("smote_alpha"):
        cube = spectral_smote(cube, boxes, donors, aug["smote_alpha"], rng)
    if aug.get("cutmix_prob") and pool:
        other = load_planar(pool[int(rng.integers(0, len(pool)))])
        cube, boxes = superpixel_cutmix(cube, boxes, other, aug["cutmix_prob"],
                                        aug["cutmix_blocks"], rng)
    if paste and aug.get("crowd_paste") and rng.random() < float(aug.get("crowd_paste_p", 1.0)):
        anchors = {CLASSES.index(n) for n in aug["paste_classes"]}
        cube, boxes = crowd_paste(cube, boxes, paste, rng, anchors,
                                  lambda path: load_planar(Path(path)),
                                  n_max=int(aug["crowd_paste"]),
                                  margin=int(aug.get("paste_margin", 4)))
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


_MAT: dict = {}


def _emit(root, split, stem, img, boxes, a):
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


def _render_frame(job):
    """Render one frame (and, for train, its augmented copies and repeats)."""
    split, pid = job
    cv2.setNumThreads(1)          # one process per core already; no nested pools after fork
    m = _MAT
    root, a = m["root"], m["anns"][pid]
    cube = load_planar(m["index"][pid])
    # The unaugmented frame is always written; validation is never augmented,
    # so the score keeps measuring the real distribution.
    img = build_channels(cube, m["channels"])
    frame = _emit(root, split, str(pid), img, a.boxes, a)
    if split == "train":
        # Augmented copies are re-rendered; repeats are file copies. A repeat is
        # not a wasted duplicate: ultralytics augments at load time -- mosaic,
        # flip, scale -- so the same frame listed twice trains on two different
        # images. Re-rendering it would only add our own spectral augmentation
        # on top, which is what the copies setting is for.
        rng = np.random.default_rng((0, int(pid)))
        for k in range(m["copies"]):
            c2, b2 = augment_cube(cube, list(a.boxes), m["aug"], m["donors"], m["pool"], rng,
                                  paste=m.get("paste"))
            _emit(root, split, f"{pid}_a{k}", build_channels(c2, m["channels"]), b2, a)
        for k in range(m["reps"][pid] - 1):
            for src, dst in ((frame, frame.with_name(f"{pid}_r{k}{frame.suffix}")),
                             (root / "labels" / split / f"{pid}.txt",
                              root / "labels" / split / f"{pid}_r{k}.txt")):
                shutil.copyfile(src, dst)
    return img.shape[2]


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
    wants_aug = bool(aug.get("sg_window") or aug.get("smote_alpha") or aug.get("cutmix_prob")
                     or aug.get("crowd_paste"))
    donors = class_donors(index, anns, train_ids) if aug.get("smote_alpha") else {}
    pool = [index[p] for p in train_ids] if aug.get("cutmix_prob") else []
    reps = repeat_factors(train_ids, anns, float(cand["train"].get("repeat_threshold", 0.0)))

    # One process per CPU, forked so they inherit everything below without
    # pickling it. Each frame's augmentation draws from its own generator,
    # seeded by the frame id, so the result does not depend on scheduling.
    _MAT.clear()
    paste = (paste_pool(index, anns, train_ids, aug["paste_classes"])
             if aug.get("crowd_paste") else None)
    if paste:
        log(f"  crowd paste: up to {aug['crowd_paste']} per augmented street frame from "
            + ", ".join(f"{CLASSES[c]} {len(v)}" for c, v in sorted(paste.items())))
    _MAT.update(index=index, anns=anns, channels=cand["channels"], aug=aug, donors=donors,
                pool=pool, copies=copies if wants_aug else 0, reps=reps, root=root, paste=paste)
    jobs = [("train", p) for p in train_ids] + [("val", p) for p in val_ids]
    workers = max(1, min(os.cpu_count() or 1, 8))
    t_r = time.time()
    if workers > 1:
        import multiprocessing as _mp
        with _mp.get_context("fork").Pool(workers) as mp_pool:
            chans = mp_pool.map(_render_frame, jobs, chunksize=8)
    else:
        chans = [_render_frame(j) for j in jobs]
    _MAT.clear()
    n_ch = chans[0] if chans else 3
    log(f"  rendering: {len(jobs)} frames on {workers} processes in {time.time() - t_r:.0f}s")

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


def render_fingerprint(cand, train_ids, val_ids):
    """Everything a rendered dataset depends on: channels, augmentation, repeat
    factors and the exact split. channels_key names the directory; this is
    what a prerendered one must match to be used."""
    spec = {"channels": cand["channels"], "augment": cand.get("augment", {}),
            "repeat_threshold": float(cand["train"].get("repeat_threshold", 0.0)),
            "train": sorted(int(i) for i in train_ids), "val": sorted(int(i) for i in val_ids)}
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def find_prerendered(cand, train_ids, val_ids):
    """A dataset rendered by the render notebook (render_only), mounted under INPUT.

    None when none is attached (render here, as before). A mounted one for this
    channel spec whose fingerprint differs is an error: silently training on a
    render made for another configuration is the failure this guards.
    """
    key = channels_key(cand)
    want = render_fingerprint(cand, train_ids, val_ids)
    for man in sorted(INPUT.rglob("render_manifest.json")) if INPUT.exists() else []:
        try:
            m = json.loads(man.read_text())
        except Exception:                                        # noqa: BLE001
            continue
        if m.get("key") != key:
            continue
        if m.get("fingerprint") != want:
            raise RuntimeError(
                f"prerendered dataset {man.parent} is for a different configuration "
                f"(fingerprint {m.get('fingerprint', '?')[:12]} != {want[:12]}): its "
                f"channels/augment/split do not match this run. Re-run the render "
                f"notebook with this candidate, or detach it to render here.")
        return man.parent
    return None


def adopt_prerendered(src, root):
    """Use a mounted render: images linked (read-only input), labels copied so
    ultralytics can write its label cache beside them."""
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    (root / "images").mkdir()
    for split in ("train", "val"):
        (root / "images" / split).symlink_to(src / "images" / split, target_is_directory=True)
    shutil.copytree(src / "labels", root / "labels")
    for c in (root / "labels").glob("*.cache"):
        c.unlink()
    text = (src / "data.yaml").read_text().splitlines()
    text = [f"path: {root}" if ln.startswith("path:") else ln for ln in text]
    (root / "data.yaml").write_text("\n".join(text) + "\n")
    m = json.loads((src / "render_manifest.json").read_text())
    log(f"  using prerendered dataset {src} ({m.get('counts')}, rendered in "
        f"{m.get('seconds', 0):.0f}s by the render notebook): no rendering here")
    return root / "data.yaml"


def submission_split(root, sub):
    """(train_ids, val_ids) for a submission run -- the render notebook, the
    preflight and the training all derive it here, so their fingerprints agree."""
    ann_dir = root / "train" / "annotations"
    ids = require_ids(sorted(int(p.stem) for p in ann_dir.glob("*.xml")), ann_dir)
    train_ids, val_ids = split_ids(ids)
    if sub.get("use_all_train", True):
        train_ids, val_ids = ids, val_ids[:60]
    return train_ids, val_ids


def run_render(round_cfg):
    """render_only: materialise the submission candidate's dataset into the
    output, with a manifest, for the training kernel to mount."""
    sub = round_cfg["submit"]
    cand = sub["candidate"]
    root = data_root()
    train_ids, val_ids = submission_split(root, sub)
    ann_dir = root / "train" / "annotations"
    anns = {pid: parse(ann_dir / f"{pid}.xml") for pid in set(train_ids) | set(val_ids)}
    index = frame_index(root, "train", sorted(set(train_ids) | set(val_ids)))
    out = WORK / f"ds_{channels_key(cand)}"
    t0 = time.time()
    materialize(cand, index, train_ids, val_ids, anns, out)
    counts = {sp: len(list((out / "images" / sp).glob("*.tiff"))) + len(list((out / "images" / sp).glob("*.png")))
              for sp in ("train", "val")}
    (out / "render_manifest.json").write_text(json.dumps({
        "key": channels_key(cand), "fingerprint": render_fingerprint(cand, train_ids, val_ids),
        "counts": counts, "seconds": time.time() - t0, "channels": cand["channels"],
        "augment": cand.get("augment", {})}, indent=1))
    for c in (out / "labels").glob("*.cache"):
        c.unlink()
    log(f"RENDER DONE: {out.name} {counts} in {time.time() - t0:.0f}s")


# ------------------------------------------------------------ evaluate ------
def predict_kwargs(cand):
    """Inference arguments, omitting NMS IoU for detectors that have no NMS."""
    inf = cand["infer"]
    kw = {"conf": inf["conf"], "max_det": inf["max_det"],
          "augment": inf["tta"], "verbose": False, "stream": True,
          "batch": PREDICT_BATCH}
    if inf.get("iou") is not None:
        kw["iou"] = inf["iou"]
    if int(cand["train"].get("s3t_upsample", 1)) > 1:
        # The model upsamples inside; predict must feed it what the loader fed
        # it in training, not the detector's own resolution.
        kw["imgsz"] = cand["train"]["imgsz"]
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
            y = self.block(self.front(x))
            # S3T-X joins the stem's output (both at stride 4); the older
            # fronts have already joined at its input.
            fuse = getattr(self.front, "fuse_stem", None)
            return y if fuse is None else fuse(y)

    class SDPAMultiheadAttention(_nn.MultiheadAttention):
        """nn.MultiheadAttention that never asks for the attention weights.

        ultralytics calls AIFI's and the decoder's attention without
        need_weights=False, so PyTorch materialises the full weight matrix
        instead of dispatching to scaled_dot_product_attention's fused kernels.
        Same parameters, same output; the class is swapped in place.
        """

        def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                    attn_mask=None, average_attn_weights=True, is_causal=False):
            return super().forward(query, key, value, key_padding_mask=key_padding_mask,
                                   need_weights=False, attn_mask=attn_mask,
                                   average_attn_weights=average_attn_weights, is_causal=is_causal)

    class FP32Criterion(_nn.Module):
        """The detector's loss computed in fp32 under an AMP forward.

        ultralytics warns that RT-DETR's bipartite matching can produce NaN in
        fp16, and HungarianMatcher does not cast back itself. Everything before
        the loss runs in fp16; the matching and the loss do not. A Module,
        because the model registers its criterion as a child module.
        """

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        @staticmethod
        def _f(x):
            if isinstance(x, (tuple, list)):
                return type(x)(FP32Criterion._f(v) for v in x)
            return x.float() if hasattr(x, "is_floating_point") and x.is_floating_point() else x

        def forward(self, preds, targets, **kw):
            import torch as _t
            dev = "cuda" if _t.cuda.is_available() else "cpu"
            with _t.autocast(dev, enabled=False):
                kw = {k: self._f(v) for k, v in kw.items()}
                return self.inner(self._f(preds), targets, **kw)

    class FrozenBatchNorm2d(_nn.BatchNorm2d):
        """BatchNorm that always normalises with its running statistics.

        At 2 images per card and no SyncBN, train-mode BatchNorm normalises
        each layer with the statistics of two mosaics and folds them into the
        COCO running estimates it was pretrained with (momentum 0.03 a step),
        so the pretrained statistics are gone within a few hundred steps and
        training normalises differently from evaluation. Eval mode throughout
        keeps COCO's statistics and makes the layer a fixed affine map whose
        scale and shift still train (where their part's LR allows). The class
        is swapped in place, so state_dict, fuse() and pickling are unchanged.
        """

        def train(self, mode=True):
            return super().train(False)
else:                                                # pragma: no cover
    SpectralFront = FrozenBatchNorm2d = None

try:
    import torch as _torch
    import torch.nn.functional as _F
    from ultralytics.utils.torch_utils import autocast as _autocast
    from ultralytics.models.utils.loss import RTDETRDetectionLoss as _RTDETRLoss
    from ultralytics.utils.loss import VarifocalLoss as _VFL
    from ultralytics.utils.metrics import bbox_iou as _bbox_iou
    from ultralytics.nn.modules.transformer import MLP as _MLP
    from ultralytics.nn.modules.transformer import DeformableTransformerDecoder as _DTD
    from ultralytics.nn.modules.utils import inverse_sigmoid as _inv_sig
except ImportError:                                  # pragma: no cover
    _RTDETRLoss = None

if _RTDETRLoss is not None:
    class IoUKindVFL(_VFL):
        """Varifocal loss with a floor under the weight it gives a positive.

        VFL weights a matched query by its own IoU with the ground truth
        (models/utils/loss.py sets gt_score to exactly that), so a pair at IoU
        0.6 receives two thirds the classification gradient of one at 0.9. That
        is deliberate -- it is how the predicted score comes to mean
        localization quality -- but it means training leans away from the
        near-misses, and within the four classes carrying our deficit, half the
        boxes sit in that discounted band.

        beta lifts the floor: 0 is VFL unchanged, 1 weights every positive
        alike, 0.5 halves the discount without flattening the quality signal.
        """

        beta: float = 0.0
        # MAL (DEIM, CVPR 2025) is beta=1 with the target raised to a power:
        # every matched query gets full weight, and its target is IoU^1.5, so a
        # low-IoU match is taught a low score instead of being down-weighted
        # out of the loss.
        target_pow: float = 1.0

        def forward(self, pred_score, gt_score, label):
            if not self.beta:
                return super().forward(pred_score, gt_score, label)
            if self.target_pow != 1.0:
                gt_score = gt_score.clamp_min(0).pow(self.target_pow)
            pos = gt_score + self.beta * (1.0 - gt_score)
            weight = (self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label)
                      + pos * label)
            with _autocast(enabled=False, device=pred_score.device.type):
                return (_F.binary_cross_entropy_with_logits(
                    pred_score.float(), gt_score.float(), reduction="none")
                    * weight).mean(1).sum()

    class IoUKindDETRLoss(_RTDETRLoss):
        """RT-DETR's box and class losses, conditioned on how good a match is.

        The error decomposition leaves one thing to fix: 96.6% of held-out
        ground truth is found at IoU >= 0.5 and 0.1% is given the wrong class,
        so the score is the distance from a median matched IoU of 0.864 up to
        the thresholds above it. Across the eighteen classes, AP tracks median
        aspect ratio at r = -0.65 net of frequency. Three knobs, each defaulting
        to the stock loss exactly:

        iou_flag   which overlap term. CIoU is GIoU plus a centre-distance and
                   an aspect-ratio penalty, which is that failure written down;
                   DIoU is the same without the aspect term, which separates
                   which half does the work.
        alpha      1 - IoU becomes 1 - IoU^alpha, so d/dIoU = -alpha*IoU^(a-1)
                   grows as a box closes on its target: found, now sharpen it
                   (alpha-IoU, He et al. 2021). The power form keeps gradient
                   everywhere, unlike a clamped interval, so the loose tail is
                   not abandoned.
        log_size   L1 on width and height in log space. DETR scales boxes to
                   [0,1] and takes an absolute L1, so an error of 0.01 costs
                   the same on a side of 0.041 as on one of 0.174 -- 24%
                   against 5.7% in the terms IoU actually charges. The
                   gradient between the two sides of one box is misallocated
                   about fourfold, and worse the flatter the box. Log space
                   makes it relative (arXiv 2410.22638: +2.2 AP, +2.9 small).

        Defined at module level because the model is pickled into every
        checkpoint and pickle cannot name a class built inside a function.
        """

        iou_flag: dict = {"GIoU": True}
        alpha_iou: float = 1.0
        log_size: bool = False
        EPS = 1e-4
        # D-FINE (ICLR 2025) on top of the decoder (FDRDecoder): FGL trains
        # each layer's edge distributions on the matched boxes, DDF distils
        # the last layer's box into the earlier layers' distributions (GO-LSD).
        # Gains are D-FINE's.
        fdr: bool = False
        fgl_gain: float = 0.15
        ddf_gain: float = 1.5
        # Per-class weight on a matched pair's L1 and overlap terms ({cls_id:
        # gain}; others 1), and a repulsion from neighbouring ground truth
        # (after RepGT, Repulsion Loss, Wang et al. CVPR 2018), at rep_gain x
        # the overlap term's gain. Both aimed at the crowded, mutually
        # occluding classes -- stone_block / people / e-bike / car are the only
        # ones whose boxes overlap other boxes at all (10-32% vs 0-3%).
        #
        # Plain RepGT would be wrong here: it charges any overlap with another
        # box, and these ground-truth boxes overlap *each other* (a rider's
        # box and the e-bike's, neighbours in a crowd), so the exact answer
        # would be charged and pushed off its own target. Only the overlap
        # beyond the truth's is charged: max(0, IoG(pred, g') - IoG(gt, g')).
        # The true box costs nothing; a box that swallows part of a neighbour
        # (two people in one box, a drift toward the next car) is pushed back.
        # Same-class neighbours only: the measured crowding is people x people,
        # e-bike x e-bike, car x car; a rider and the e-bike under them are two
        # different objects whose boxes are meant to overlap.
        box_cls_gain: dict = {}
        rep_gain: float = 0.0
        rep_sigma: float = 0.5

        def __getstate__(self):
            state = self.__dict__.copy()
            state.pop("_fdr_calls", None)
            state.pop("_box_ctx", None)
            return state

        def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=None,
                      gt_mask=None, postfix="", match_indices=None):
            # Same matching, recorded, so the distribution losses reuse it.
            if match_indices is None:
                match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                             masks=masks, gt_mask=gt_mask)
            calls = self.__dict__.get("_fdr_calls")
            if calls is not None:
                calls.append((match_indices, pred_bboxes, pred_scores))
            # What the per-pair terms need and _get_loss_bbox is not given: the
            # matched pairs' classes and images, and every box of the batch.
            self.__dict__["_box_ctx"] = (match_indices, gt_bboxes, gt_cls, gt_groups)
            try:
                return super()._get_loss(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                         masks=masks, gt_mask=gt_mask, postfix=postfix,
                                         match_indices=match_indices)
            finally:
                self.__dict__.pop("_box_ctx", None)

        def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
            dec = self.__dict__.get("_fdr_decoder")
            stash = dec.__dict__.get("_fdr") if (self.fdr and dec is not None) else None
            if stash is None:
                return super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
            dec.__dict__["_fdr"] = None                     # one loss per forward
            self.__dict__["_fdr_calls"] = calls = []
            try:
                loss = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
            finally:
                self.__dict__.pop("_fdr_calls", None)
            loss.update(fdr_losses(self, dec, stash, calls, batch, dn_meta))
            return loss

        def _l1(self, pred_bboxes, gt_bboxes):
            """Per matched pair (summed over the four coordinates)."""
            if not self.log_size:
                return _F.l1_loss(pred_bboxes, gt_bboxes, reduction="none").sum(-1)
            # Centres stay linear; only the sizes move to log space. Sizes come
            # out of a sigmoid, so a width near zero would send log to -inf --
            # the clamp is what keeps that from surfacing as a NaN epochs later.
            ctr = _F.l1_loss(pred_bboxes[..., :2], gt_bboxes[..., :2], reduction="none").sum(-1)
            wh = _F.l1_loss(pred_bboxes[..., 2:].clamp_min(self.EPS).log(),
                            gt_bboxes[..., 2:].clamp_min(self.EPS).log(),
                            reduction="none").sum(-1)
            return ctr + wh

        def _pair_ctx(self, n):
            """(image of each matched pair, gt index of each, all gt boxes, classes, groups) or None."""
            ctx = self.__dict__.get("_box_ctx")
            if ctx is None:
                return None
            match_indices, gt_all, gt_cls, groups = ctx
            bidx = _torch.cat([_torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
            gidx = _torch.cat([dst for (_, dst) in match_indices])
            if len(gidx) != n:
                return None
            return bidx.to(gt_all.device), gidx.to(gt_all.device), gt_all, gt_cls, groups

        def _pair_weights(self, n, ctx, device):
            if not self.box_cls_gain or ctx is None:
                return None
            cls = ctx[3][ctx[1]].view(-1).long()
            w = _torch.ones(n, device=device)
            for c, g in self.box_cls_gain.items():
                w = _torch.where(cls == int(c), _torch.full_like(w, float(g)), w)
            return w

        def _rep_gt(self, pred_bboxes, ctx):
            """Sum over matched pairs of smooth-ln(excess IoG) against the worst other gt.

            Excess over the pair's own ground truth: zero, with zero gradient,
            for a prediction equal to its target however much the targets overlap.
            """
            bidx, gidx, gt_all, gt_cls, groups = ctx
            if gt_all.shape[0] < 2:
                return pred_bboxes.sum() * 0.0
            img = _torch.repeat_interleave(_torch.arange(len(groups), device=gt_all.device),
                                           _torch.as_tensor(groups, device=gt_all.device))
            p = _torch.cat([pred_bboxes[:, :2] - pred_bboxes[:, 2:] / 2,
                            pred_bboxes[:, :2] + pred_bboxes[:, 2:] / 2], -1).float()
            g = _torch.cat([gt_all[:, :2] - gt_all[:, 2:] / 2,
                            gt_all[:, :2] + gt_all[:, 2:] / 2], -1).float()
            lt = _torch.maximum(p[:, None, :2], g[None, :, :2])
            rb = _torch.minimum(p[:, None, 2:], g[None, :, 2:])
            inter = (rb - lt).clamp_min(0).prod(-1)                       # (n, m)
            area_g = (g[:, 2:] - g[:, :2]).clamp_min(1e-9).prod(-1)
            area_p = (p[:, 2:] - p[:, :2]).clamp_min(1e-9).prod(-1)
            cls = gt_cls.view(-1).to(gt_all.device)
            other = (bidx[:, None] == img[None, :]) & (cls[gidx][:, None] == cls[None, :])
            other[_torch.arange(len(gidx), device=other.device), gidx] = False
            # how much of each other gt the prediction covers, beyond what its
            # own ground truth covers
            t = g[gidx]
            lt_t = _torch.maximum(t[:, None, :2], g[None, :, :2])
            rb_t = _torch.minimum(t[:, None, 2:], g[None, :, 2:])
            true_iog = (rb_t - lt_t).clamp_min(0).prod(-1) / area_g[None, :]
            excess = inter / area_g[None, :] - true_iog
            excess = _torch.where(other, excess, _torch.full_like(excess, -1.0))
            best = excess.detach().argmax(1)
            ex = excess.gather(1, best[:, None])[:, 0]
            valid = ex.detach() > 0
            if not bool(valid.any()):
                return pred_bboxes.sum() * 0.0
            iog = ex.clamp(0.0, 1.0 - 1e-4)
            sig = self.rep_sigma
            import math as _m
            smooth = _torch.where(iog <= sig, -_torch.log1p(-iog),
                                  (iog - sig) / (1.0 - sig) - _m.log(1.0 - sig))
            return (smooth * valid).sum()

        def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=""):
            name_bbox, name_giou = f"loss_bbox{postfix}", f"loss_giou{postfix}"
            if not len(gt_bboxes):
                z = _torch.tensor(0.0, device=self.device)
                return {name_bbox: z, name_giou: z.clone()}
            n = len(gt_bboxes)
            ctx = self._pair_ctx(n) if (self.box_cls_gain or self.rep_gain) else None
            w = self._pair_weights(n, ctx, pred_bboxes.device)
            variant = _bbox_iou(pred_bboxes, gt_bboxes, xywh=True, **self.iou_flag)
            if self.alpha_iou == 1.0:
                overlap = 1.0 - variant
            else:
                # Power the IoU, keep the variant's geometric penalty linear --
                # alpha-IoU's form. The penalty is (iou - variant) by
                # construction, whatever the variant.
                plain = _bbox_iou(pred_bboxes, gt_bboxes, xywh=True)
                overlap = (1.0 - plain.clamp_min(0).pow(self.alpha_iou)) + (plain - variant)
            l1 = self._l1(pred_bboxes, gt_bboxes)
            overlap = overlap.view(-1)
            if w is not None:
                l1, overlap = l1 * w, overlap * w
            giou = self.loss_gain["giou"] * overlap.sum()
            if self.rep_gain and ctx is not None:
                # Folded into the overlap term: ultralytics' aux-layer sum keeps
                # only the three stock keys, so a separate key would be dropped
                # for every layer but the last.
                giou = giou + self.rep_gain * self.loss_gain["giou"] * self._rep_gt(pred_bboxes, ctx)
            return {
                name_bbox: (self.loss_gain["bbox"] * l1.sum() / n).squeeze(),
                name_giou: (giou / n).squeeze(),
            }

    def fdr_weighting(reg_max=32, up=0.5, reg_scale=4.0):
        """D-FINE's W(n): reg_max + 1 edge offsets, dense near 0, +-2*up*reg_scale at the ends."""
        ub1 = abs(up) * abs(reg_scale)
        step = (ub1 + 1) ** (2 / (reg_max - 2))
        left = [-(step ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right = [step ** i - 1 for i in range(1, reg_max // 2)]
        return _torch.tensor([-2 * ub1] + left + [0.0] + right + [2 * ub1], dtype=_torch.float32)

    def fdr_apply(box, corners, project, reg_scale):
        """cxcywh box, corner logits (..., 4 * (R + 1)) -> the box with its edges moved.

        Each edge moves by the expectation of W under its distribution, in units
        of the box's side / reg_scale (D-FINE's distance2bbox). Uniform logits
        give an expectation of 0 (W is odd), so zero-initialised heads leave the
        box exactly where the pretrained head put it.
        """
        p = project.float()
        d = (corners.float().unflatten(-1, (4, p.numel())).softmax(-1) * p).sum(-1)
        cx, cy, w, h = box.float().unbind(-1)
        half = 0.5 * reg_scale
        x1 = cx - (half + d[..., 0]) * w / reg_scale
        y1 = cy - (half + d[..., 1]) * h / reg_scale
        x2 = cx + (half + d[..., 2]) * w / reg_scale
        y2 = cy + (half + d[..., 3]) * h / reg_scale
        c = _torch.stack([(x1 + x2) / 2, (y1 + y2) / 2], -1).clamp(0.0, 1.0)
        wh = _torch.stack([x2 - x1, y2 - y1], -1).clamp(1e-4, 1.0)
        return _torch.cat([c, wh], -1).to(box.dtype)

    def fdr_targets(ref, gt, project, reg_scale):
        """Where gt's edges sit relative to ref, as two adjacent bins of W and their weights.

        D-FINE's bbox2distance + translate_gt: an edge offset between W[k] and
        W[k+1] is split linearly between them, so the expectation reproduces it
        exactly; beyond either end it goes wholly to the end bin.
        """
        p = project.float()
        rmax = p.numel() - 1
        ref, gt = ref.float(), gt.float()
        sw = ref[..., 2] / reg_scale + 1e-16
        sh = ref[..., 3] / reg_scale + 1e-16
        g1, g2 = gt[..., :2] - gt[..., 2:] / 2, gt[..., :2] + gt[..., 2:] / 2
        d = _torch.stack([(ref[..., 0] - g1[..., 0]) / sw, (ref[..., 1] - g1[..., 1]) / sh,
                          (g2[..., 0] - ref[..., 0]) / sw, (g2[..., 1] - ref[..., 1]) / sh],
                         -1).reshape(-1) - 0.5 * reg_scale
        k = (p[None, :] <= d[:, None]).sum(1) - 1
        lo = k.clamp(0, rmax - 1)
        wr = ((d - p[lo]) / (p[lo + 1] - p[lo])).clamp(0.0, 1.0)
        wr = _torch.where(k < 0, _torch.zeros_like(wr), wr)
        wr = _torch.where(k >= rmax, _torch.ones_like(wr), wr)
        return lo, 1.0 - wr, wr

    def _two_bin_kl(logits, lo, wl, wr):
        """KL(two-bin target || softmax(logits)) per edge; its gradient is D-FINE's FGL/DDF's."""
        lp = logits.float().log_softmax(-1)
        ce = -(lp.gather(1, lo[:, None])[:, 0] * wl + lp.gather(1, (lo + 1)[:, None])[:, 0] * wr)
        ent = -(wl * wl.clamp_min(1e-12).log() + wr * wr.clamp_min(1e-12).log())
        return ce - ent

    def fdr_losses(crit, dec, stash, calls, batch, dn_meta):
        """FGL on every decoder layer's matched boxes; DDF from the last layer to the rest.

        calls are the loss's own matchings, in DETRLoss's order: the main
        queries' last layer, then its aux layers (encoder, decoder 0..L-2), then
        the same for the denoising queries (no encoder layer there).

        Batched across layers -- one target encoding, one IoU, one KL per part
        and loss -- and free of host syncs: the per-layer loop it replaces cost
        ~0.09 s of a 0.42 s step on a T4 in small kernels alone.
        """
        proj, rs = dec.fdr_project.float(), float(dec.fdr_reg_scale)
        corners, refs = stash["corners"].float(), stash["refs"].float()
        L, R1 = corners.shape[0], proj.numel()
        if dn_meta is not None:
            dn_c, mc = corners.split(dn_meta["dn_num_split"], dim=2)
            dn_r, mr = refs.split(dn_meta["dn_num_split"], dim=2)
            parts = [(mc, mr, [L - 1, None] + list(range(L - 1))),
                     (dn_c, dn_r, [L - 1] + list(range(L - 1)))]
        else:
            parts = [(corners, refs, [L - 1, None] + list(range(L - 1)))]
        if sum(len(pl) for _, _, pl in parts) != len(calls):
            raise RuntimeError(f"FDR loss: {len(calls)} matchings for layers "
                               f"{[pl for _, _, pl in parts]} -- DETRLoss changed its call order")
        gt = batch["bboxes"].float()
        zero = corners.sum() * 0.0
        fgl = ddf = zero
        k = 0
        for C, Rf, plan in parts:
            t_idx, t_box, t_score = calls[k]                     # this part's last layer: the teacher
            # FGL: every layer's matched (corners, reference, gt, box), each
            # row weighted by its IoU / the layer's match count.
            pcs, rfs, gts, pbs, inv = [], [], [], [], []
            for layer in plan:
                mi, pb, _ = calls[k]
                k += 1
                if layer is None:
                    continue
                idx, gt_idx = crit._get_index(mi)
                n = len(gt_idx)
                if n:
                    pcs.append(C[layer][idx])
                    rfs.append(Rf[layer][idx])
                    gts.append(gt[gt_idx])
                    pbs.append(pb[idx])
                    inv.append(_torch.full((n,), 1.0 / n, device=gt.device))
            if pcs:
                g_all = _torch.cat(gts)
                lo, wl, wr = fdr_targets(_torch.cat(rfs), g_all, proj, rs)
                iou = _bbox_iou(_torch.cat(pbs).detach().float(), g_all, xywh=True).view(-1).clamp_min(0)
                w = (iou * _torch.cat(inv)).repeat_interleave(4)
                fgl = fgl + (_two_bin_kl(_torch.cat(pcs).reshape(-1, R1), lo, wl, wr) * w).sum()
            if not crit.ddf_gain or L < 2:
                continue
            # DDF: layers 0..L-2 against the teacher's box, all at once. The
            # weights (teacher IoU where matched, its confidence elsewhere) and
            # the pos/neg balance are the same for every layer.
            t_box = t_box.detach().float()
            b = t_box.shape[0]
            wt = t_score.detach().float().sigmoid().max(-1).values
            pos = _torch.zeros_like(wt, dtype=_torch.bool)
            tidx, tgt_idx = crit._get_index(t_idx)
            if len(tgt_idx):
                pos[tidx] = True
                wt = wt.index_put(tidx, _bbox_iou(t_box[tidx], gt[tgt_idx], xywh=True).view(-1).clamp_min(0))
            m = L - 1
            lo, wl, wr = fdr_targets(Rf[:m].reshape(-1, 4), t_box.expand(m, *t_box.shape).reshape(-1, 4),
                                     proj, rs)
            kl = _two_bin_kl(C[:m].reshape(-1, R1), lo, wl, wr)
            wt4 = wt.expand(m, *wt.shape).reshape(-1).repeat_interleave(4)
            pos4 = pos.expand(m, *pos.shape).reshape(-1).repeat_interleave(4).float()
            kl = kl * wt4
            scale = 8.0 / b                                       # D-FINE: independent of batch per GPU
            n_pos, n_neg = (pos4.sum() / m * scale) ** 0.5, ((1 - pos4).sum() / m * scale) ** 0.5
            l_pos = (kl * pos4).sum() / pos4.sum().clamp_min(1.0)  # mean over layers' pos edges
            l_neg = (kl * (1 - pos4)).sum() / (1 - pos4).sum().clamp_min(1.0)
            # sum over layers of each layer's balanced mean == m x the pooled one
            ddf = ddf + m * (l_pos * n_pos + l_neg * n_neg) / (n_pos + n_neg).clamp_min(1e-6)
        return {"loss_fgl": crit.fgl_gain * fgl, "loss_ddf": crit.ddf_gain * ddf}

    class FDRDecoder(_DTD):
        """RT-DETR's pretrained decoder with D-FINE's distribution refinement added.

        D-FINE replaces the box head: each layer predicts, per edge, a
        distribution over reg_max + 1 offsets and the box is its expectation.
        Replacing it here would throw away the COCO-pretrained box heads, so
        the distributions are added *on top*: layer i's box is the pretrained
        head's box with each edge moved by the expectation of its distribution
        (fdr_apply), and the heads producing the logits start at zero -- a
        uniform distribution, an offset of exactly 0, the pretrained decoder at
        step 0. The refined box is also the next layer's reference.

        Swapped in by class (install_fdr), like SDPAMultiheadAttention: the
        layers, heads and weights are the pretrained ones.
        """

        def __getstate__(self):
            state = self.__dict__.copy()
            state.pop("_fdr", None)
            return state

        def forward(self, embed, refer_bbox, feats, shapes, bbox_head, score_head, pos_mlp,
                    attn_mask=None, padding_mask=None):
            output = embed
            boxes, logits, corners, refs = [], [], [], []
            last = None
            proj, rs = self.fdr_project, self.fdr_reg_scale
            refer_bbox = refer_bbox.sigmoid()
            for i, layer in enumerate(self.layers):
                output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask,
                               pos_mlp(refer_bbox))
                delta = bbox_head[i](output)
                c = self.fdr[i](output)
                refined = fdr_apply(_torch.sigmoid(delta + _inv_sig(refer_bbox)), c, proj, rs)
                if self.training:
                    logits.append(score_head[i](output))
                    # ultralytics' look-forward-twice: from layer 1 on, the box
                    # the loss sees is built on the previous layer's undetached box.
                    coarse = (_torch.sigmoid(delta + _inv_sig(refer_bbox)) if i == 0
                              else _torch.sigmoid(delta + _inv_sig(last)))
                    boxes.append(refined if i == 0 else fdr_apply(coarse, c, proj, rs))
                    corners.append(c)
                    refs.append(coarse.detach())
                elif i == self.eval_idx:
                    logits.append(score_head[i](output))
                    boxes.append(refined)
                    break
                last = refined
                refer_bbox = refined.detach() if self.training else refined
            self.__dict__["_fdr"] = ({"corners": _torch.stack(corners), "refs": _torch.stack(refs)}
                                     if self.training else None)
            return _torch.stack(boxes), _torch.stack(logits)

else:                                                # pragma: no cover
    IoUKindDETRLoss = IoUKindVFL = FDRDecoder = None

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
        for _c in (IoUKindDETRLoss, IoUKindVFL, FDRDecoder):
            setattr(_mod, _c.__name__, _c)
            _c.__module__ = "hod26_kernel"


def materialise_kernel_module():
    """Give the DDP workers a real hod26_kernel.py to import.

    The registration above pins SpectralFront and the loss overrides to a
    synthetic module so a checkpoint names the same class wherever it is
    loaded. cloudpickle -- which ultralytics uses to hand the trainer, the
    model and the callbacks to its DDP workers -- resolves that name exactly
    as pickle does: the module answers in sys.modules here, so the classes go
    across *by reference*, and the worker, a fresh interpreter that never ran
    this script, dies on ModuleNotFoundError before the first batch. Measured,
    not assumed: pickling the same shape and loading it in a clean process
    reproduces it every time.

    Writing this file to disk under that name and putting it on sys.path makes
    the reference resolvable in the worker, and the worker's sys.path is this
    process's -- ultralytics bakes it into the file it generates. The import is
    cheap because every expensive step in this script sits behind main(),
    which only __main__ runs.

    Returns False when the source cannot be located, which is the caller's cue
    to stay on one GPU. A second card is worth a few hours; it is not worth
    failing the session outright.
    """
    import sys as _s
    src = None
    for cand in (globals().get("__file__"), "/kaggle/src/script.py"):
        try:
            if cand and Path(cand).is_file():
                src = Path(cand)
                break
        except OSError:
            continue
    if src is None:
        return False
    try:
        target = SCRATCH / "hod26_kernel.py"
        target.write_text(src.read_text())
        if str(SCRATCH) not in _s.path:
            _s.path.insert(0, str(SCRATCH))
        return True
    except OSError:
        return False


def visible_gpus():
    """How many CUDA devices this session actually got, never more than asked."""
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:                                           # noqa: BLE001
        return 0


def find_mae_checkpoint(name=None):
    """The S3T encoder pretrained by MAE, wherever its kernel output is mounted.

    Exactly one, or an error: with two pretraining notebooks attached (v1's
    pretrain_mae.pt and v2's pretrain2_mae.pt) picking by sort order would
    silently train the detector on whichever path sorts first. name (the
    candidate's train.s3t_mae_file) selects one by file name.
    """
    hits = sorted(INPUT.rglob("*_mae.pt")) if INPUT.exists() else []
    if name:
        hits = [h for h in hits if h.name == name]
    if len(hits) > 1:
        raise RuntimeError(f"{len(hits)} MAE checkpoints attached ({[str(h) for h in hits]}); "
                           f"set train.s3t_mae_file to the one to use")
    return hits[0] if hits else None


def install_s3t_front(net, n_bands=16, projection=None, mae_ckpt=None, scale=0.5,
                      inject=(19, 14, 10), dim=64, depth=4, heads=4, widen=True,
                      context=True, ctx_layer=11, ckpt_chunks=8, compile_blocks=False,
                      fast_kernels=False, train_encoder=True, arch="tokens", grad_ckpt=True,
                      windows=(16, 16, None, None), upsample=1, stem_bands=False, **_):
    """S3T encoder in front of the pretrained first block, plus side injections.

    The encoder is loaded from the MAE checkpoint when one is given; its config
    decides the architecture (arch "xca": S3T-X, fused at the stem's output;
    "tokens": the band-token encoder, joined at the stem's widened input).
    inject names the hybrid encoder's input projections (P3, P4, P5 in
    rtdetr-l); each gets the spectral features through a zero-initialised 1x1.
    upsample (xca only) lets the loader run at 1/upsample of the detector's
    resolution: the front reads that input directly and hands the detector a
    bilinear upsample made on the GPU (see S3TXFront).
    """
    import torch

    block = net.model[0]
    if isinstance(block, SpectralFront):
        return False                      # already installed (resumed model)
    cfg = {"arch": arch, "dim": dim, "depth": depth, "heads": heads, "windows": list(windows)}
    state = None
    if mae_ckpt:
        ck = torch.load(mae_ckpt, map_location="cpu", weights_only=True)
        cfg.update(ck.get("config") or {})
        if cfg.get("arch", "tokens") != arch:
            raise RuntimeError(f"MAE checkpoint {mae_ckpt} is arch={cfg.get('arch', 'tokens')!r}, "
                               f"the run asks for arch={arch!r}")
        state = ck["encoder"]
    enc = build_encoder(cfg)
    if state is not None:
        enc.load_state_dict(state, strict=True)
        log(f"  S3T encoder: MAE weights from {mae_ckpt} (step {ck.get('step')})")
    else:
        log("  S3T encoder: NO pretrained weights -- random initialisation")
    xca = cfg.get("arch") == "xca"
    if xca:
        stem_ch = [m for m in block.modules() if isinstance(m, torch.nn.Conv2d)][-1].out_channels
        front = S3TXFront(enc, projection=projection, scale=scale, grad_ckpt=grad_ckpt,
                          stem_ch=stem_ch, train_encoder=train_encoder, upsample=upsample,
                          stem_bands=stem_bands)
        widen = False
        if stem_bands:
            # 3 projected channels + all 16 bands into the stem's first conv;
            # the 16 new input channels start at zero (identical output at
            # step 0). Marked so staged unfreezing can train it (part stem_in)
            # while the rest of the stem stays frozen.
            conv = widen_first_conv(block, n_bands)
            conv.__dict__["_hod26_widened"] = True
            log(f"  stem input: 3 projected + {n_bands} band channels "
                f"(first conv {conv.in_channels} in, the {n_bands} new ones zero-initialised)")
    else:
        front = S3TFront(enc, projection=projection, scale=scale, widen=widen,
                         ckpt_chunks=ckpt_chunks, fast_kernels=fast_kernels,
                         train_encoder=train_encoder)
    if compile_blocks:
        # Fuse each block's LayerNorm/GELU/residual chains (nn.Module.compile,
        # in place, so DDP and pickling see ordinary modules). No fallback: a
        # failure surfaces as an error at the first step.
        from torch import nn as _tnn
        n_c = 0
        for m in enc.modules():
            if type(m).__name__ in ("SpectralBlock", "SpatialMix", "SpectralPool", "XCABlock"):
                m.compile()
                n_c += 1
        log(f"  S3T blocks compiled in place: {n_c}")
    dev = next(block.parameters()).device
    if widen:
        # 3 -> 3 + D input channels on the pretrained stem's first conv, the
        # new ones zero: no 3-channel bottleneck, identical output at step 0.
        conv = widen_first_conv(block, enc.dim)
        log(f"  stem widened: first conv now reads {conv.in_channels} channels "
            f"(3 pretrained + {enc.dim} S3T, zero-initialised)")
    wrapper = SpectralFront(front, block).to(dev)
    for attr in ("i", "f", "type", "np"):
        if hasattr(block, attr):
            setattr(wrapper, attr, getattr(block, attr))
    net.model[0] = wrapper
    ctx_ch = None
    if context:
        # AIFI (layer 11 in rtdetr-l) runs before the P4/P3 input projections
        # (14, 19), so its global context is ready when those injections run.
        aifi = net.model[ctx_layer]
        ctx_ch = getattr(getattr(aifi, "ma", None), "embed_dim", 256)
        net.model[ctx_layer] = Tap(aifi, front).to(dev)
    for i in inject:
        layer = net.model[i]
        out_ch = [m for m in layer.modules() if isinstance(m, torch.nn.Conv2d)][-1].out_channels
        if context and i > ctx_layer:
            net.model[i] = ContextInject(layer, front, out_ch, ctx_ch=ctx_ch).to(dev)
        else:
            net.model[i] = Inject(layer, front, out_ch).to(dev)
    net.__dict__["_hod26_mixer"] = front.base
    net.__dict__["_hod26_mixer_init"] = front.base.weight.detach().clone()
    n_enc = sum(p.numel() for p in enc.parameters())
    if xca:
        log(f"  S3T-X: cross-covariance attention, windows {enc.windows}, channels_last; "
            f"{'per-block checkpoints' if grad_ckpt else 'no checkpoints'}; encoder "
            f"{'trained' if train_encoder else 'FROZEN (feature extractor)'}")
        if upsample > 1:
            log(f"  S3T-X input: loader at 1/{upsample} of the detector's resolution; the "
                f"encoder reads it directly (scale {scale * upsample:g}), the detector gets a "
                f"{upsample}x bilinear upsample on the GPU")
        log(f"  S3T-X front: {n_enc / 1e6:.2f}M-param encoder at {scale}x input scale, "
            f"stride-4 output joined to {type(block).__name__}'s {front.fuse.out_channels}-ch output "
            f"(zero-init 1x1); learned pyramid -> zero-init injections at layers {list(inject)}"
            + (f"; P3/P4 read AIFI's global context (layer {ctx_layer}, {ctx_ch}-d)" if context else ""))
        return True
    log(f"  S3T memory: per-layer checkpoints, per-position layers in {ckpt_chunks} chunks"
        if ckpt_chunks else "  S3T memory: one checkpoint around the whole encoder")
    log(f"  S3T kernels: {'16-token attention as batched matmul, depthwise conv channels_last' if fast_kernels else 'SDPA attention, NCHW depthwise conv'}; "
        f"encoder {'trained' if train_encoder else 'FROZEN (feature extractor)'}")
    log(f"  S3T front: {n_enc / 1e6:.2f}M-param spectral Transformer at {scale}x input "
        f"scale, {'3+' + str(enc.dim) if widen else '16->3'} channels into the pretrained "
        f"{type(block).__name__}, zero-init side injections at layers {list(inject)}"
        + (f"; P3/P4 read AIFI's global context (layer {ctx_layer}, {ctx_ch}-d)" if context else ""))
    return True


def enable_transformer_accel(net, fp32_loss=True, nc=None):
    """Put every attention in the detector on SDPA, and its loss on fp32.

    Returns what it did, for the acceleration table the run prints.
    """
    import torch
    n_mha = 0
    for m in net.modules():
        if type(m) is torch.nn.MultiheadAttention:
            m.__class__ = SDPAMultiheadAttention
            n_mha += 1
    wrapped = False
    if fp32_loss:
        if getattr(net, "criterion", None) is None:
            # get_model runs before ultralytics sets net.nc; the criterion needs it.
            if nc is not None and not hasattr(net, "nc"):
                net.nc = int(nc)
            net.criterion = net.init_criterion()
        if not isinstance(net.criterion, FP32Criterion):
            net.criterion = FP32Criterion(net.criterion)
        wrapped = True
    return {"mha_to_sdpa": n_mha, "fp32_loss": wrapped}


S3T_COMPILE_UNITS = ("SpectralBlock", "SpatialMix", "SpectralPool", "XCABlock")


def _install_mosaic_canvas_reuse():
    """Reuse one mosaic canvas per loader process instead of a fresh np.full.

    A 4-image mosaic at imgsz 1024 with 16 channels allocates a 2048x2048x16
    uint8 canvas (67 MB) per sample; the page faults of the fresh allocation
    were 40% of a sample's loading time (67 of 165 ms on one core), and a T4
    pair is fed by 4 vCPUs. The canvas lives only until RandomPerspective,
    which always warps it into a new array (the mosaic border is never zero),
    so a per-process buffer refilled with 114 gives byte-identical samples
    (tested) at 1.6-1.7x the loader throughput. Installed at import, so the
    DDP workers (which import this module) and their forked loader workers
    all have it.
    """
    try:
        import numpy as _np
        from ultralytics.data.augment import Mosaic
    except Exception:                                           # noqa: BLE001
        return False
    if getattr(Mosaic.apply_image, "_hod26_canvas", False):
        return True
    orig = Mosaic.apply_image

    def apply_image(self, labels, params=None):
        if self.n != 4 or params is None or "layout" not in params:
            return orig(self, labels, params)
        shape = (self.imgsz * 2, self.imgsz * 2, labels["img"].shape[2])
        buf = self.__dict__.get("_hod26_buf")
        if buf is None or buf.shape != shape:
            buf = self.__dict__["_hod26_buf"] = _np.empty(shape, dtype=_np.uint8)
        buf.fill(114)
        for item in params["layout"]:
            img = item["labels_patch"]["img"]
            buf[item["y1a"]:item["y2a"], item["x1a"]:item["x2a"]] = \
                img[item["y1b"]:item["y2b"], item["x1b"]:item["x2b"]]
        labels["img"] = buf
        return labels

    apply_image._hod26_canvas = True
    apply_image._hod26_orig = orig
    Mosaic.apply_image = apply_image
    return True


def _install_amp_check_override():
    """ultralytics' check_amp, overridable for runs that require AMP.

    check_amp compares fp32 and fp16 inference of a *different* model (YOLO26n)
    on one image and fails on any difference in the number of boxes. On a T4
    with cudnn.benchmark that flips between processes: the S3T-X smoke passed it
    and the long run, two minutes later in the same session, failed it -- and
    would have trained in fp32 at half the speed had the run not refused. AMP
    on *this* model is measured instead (probes and smokes: finite losses,
    stable GradScaler), and its loss and Hungarian matching run in fp32. So when
    HOD26_REQUIRE_AMP=1 (set by run_candidate for such runs, inherited by the
    DDP workers) a failed check is logged and overridden; otherwise untouched.
    """
    try:
        import ultralytics.engine.trainer as _tr
    except Exception:                                           # noqa: BLE001
        return False
    if getattr(_tr.check_amp, "_hod26", False):
        return True
    orig = _tr.check_amp

    def check_amp(model):
        ok = orig(model)
        if not ok and os.environ.get("HOD26_REQUIRE_AMP") == "1":
            os.environ["HOD26_AMP_OVERRIDDEN"] = "1"
            print("AMP: ultralytics check_amp failed (YOLO26n fp32 vs fp16 box count); "
                  "overridden -- this run requires AMP, measured stable on its own model, "
                  "loss and matching in fp32", flush=True)
            return True
        return ok

    check_amp._hod26 = True
    _tr.check_amp = check_amp
    return True


_install_amp_check_override()


def mosaic_canvas_patched():
    try:
        from ultralytics.data.augment import Mosaic
        return bool(getattr(Mosaic.apply_image, "_hod26_canvas", False))
    except Exception:                                           # noqa: BLE001
        return False


_install_mosaic_canvas_reuse()


def ensure_accel(net, nc=None, fp32_loss=True, compile_blocks=False):
    """(Re)apply every acceleration and report what is *actually* in effect.

    Idempotent, and called where the training process sets its model up
    (HOD26Trainer.setup_model), because under DDP that is not where the model
    was built: ultralytics builds it in the parent with get_model and hands each
    worker a cloudpickled copy. The SDPA class swap and the fp32 criterion
    travel with the model; cudnn.benchmark is a per-process flag and
    nn.Module.compile's compiled call is not pickled. The first S3T-X session
    trained that way -- S3T blocks eager, cudnn not autotuning -- while its
    table, which read a flag set in the parent, said nothing was wrong.
    """
    import torch
    # build_optimizer runs after ultralytics has wrapped the model in DDP; the
    # first S3T-X session with this function died on the wrapper's missing
    # init_criterion. Everything below acts on the model itself.
    net = getattr(net, "module", net)
    torch.backends.cudnn.benchmark = True
    enable_transformer_accel(net, fp32_loss=fp32_loss, nc=nc)
    front = getattr(getattr(net, "model", [None])[0], "front", None)
    enc = getattr(front, "enc", None)
    blocks = [m for m in enc.modules() if type(m).__name__ in S3T_COMPILE_UNITS] if enc is not None else []
    if compile_blocks:
        for m in blocks:
            if m.__dict__.get("_compiled_call_impl") is None:
                m.compile()
    mods = list(net.modules())
    return {
        "sdpa_mha": sum(isinstance(m, SDPAMultiheadAttention) for m in mods),
        "plain_mha": sum(type(m) is torch.nn.MultiheadAttention for m in mods),
        "fp32_loss": isinstance(getattr(net, "criterion", None), FP32Criterion),
        "compiled": sum(m.__dict__.get("_compiled_call_impl") is not None for m in blocks),
        "blocks": len(blocks), "compile_wanted": bool(compile_blocks),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "mosaic_canvas_reuse": mosaic_canvas_patched(),
    }


def install_spectral_adapter(net, n_bands, projection=None, ckpt_name=None,
                             srf_k=0, srf_width=2.0, stem_src=None, kind="mixer", **s3t):
    if kind == "s3t":
        return install_s3t_front(net, n_bands=n_bands, projection=projection, **s3t)
    return _install_mixer(net, n_bands, projection, ckpt_name, srf_k, srf_width, stem_src)


def _install_mixer(net, n_bands, projection=None, ckpt_name=None,
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


IOU_KINDS = {"GIoU": {"GIoU": True}, "DIoU": {"DIoU": True},
             "CIoU": {"CIoU": True}, "IoU": {}}


def install_fdr(net, reg_max: int = 32, reg_scale: float = 4.0, up: float = 0.5):
    """D-FINE's distribution refinement on RT-DETR's decoder (see FDRDecoder).

    One zero-initialised 3-layer MLP per decoder layer, hidden -> 4 x (reg_max
    + 1) edge logits, as D-FINE initialises its heads. Returns the decoder.
    """
    import torch
    head = net.model[-1]
    dec = head.decoder
    if isinstance(dec, FDRDecoder):
        return dec
    hd = int(getattr(head, "hidden_dim", 256))
    dev = next(dec.parameters()).device
    heads = torch.nn.ModuleList(_MLP(hd, hd, 4 * (reg_max + 1), 3) for _ in range(len(dec.layers)))
    for m in heads:
        torch.nn.init.zeros_(m.layers[-1].weight)
        torch.nn.init.zeros_(m.layers[-1].bias)
    dec.__class__ = FDRDecoder
    dec.fdr = heads.to(dev)
    dec.register_buffer("fdr_project", fdr_weighting(reg_max, up, reg_scale).to(dev), persistent=False)
    dec.fdr_reg_scale = float(reg_scale)
    return dec


def install_bbox_loss(net, nc: int, kind: str = "GIoU", alpha: float = 1.0,
                      beta: float = 0.0, log_size: bool = False,
                      loss_gain: dict | None = None, fdr: bool = False, mal: bool = False,
                      box_cls_gain: dict | None = None, rep_gain: float = 0.0):
    """Condition RT-DETR's box and class losses on how good each match is.

    Every default reproduces the stock loss exactly, which is what makes an A/B
    over these clean. nc comes from the caller because the model does not carry
    it yet -- set_model_attributes attaches it after get_model runs.
    """
    if kind not in IOU_KINDS:
        raise ValueError(f"unknown bbox loss {kind!r}; have {sorted(IOU_KINDS)}")

    crit = IoUKindDETRLoss(nc=int(nc), use_vfl=True)
    crit.iou_flag = IOU_KINDS[kind]
    crit.alpha_iou = float(alpha)
    crit.log_size = bool(log_size)
    if mal:
        # MAL: every matched query at full weight, target IoU^1.5 (DEIM's
        # gamma); negatives keep RT-DETR's alpha * p^gamma focal weight.
        vfl = IoUKindVFL(gamma=crit.vfl.gamma, alpha=crit.vfl.alpha)
        vfl.beta, vfl.target_pow = 1.0, 1.5
        crit.vfl = vfl
    elif beta:
        vfl = IoUKindVFL()
        vfl.beta = float(beta)
        crit.vfl = vfl
    if fdr:
        crit.fdr = True
        crit.__dict__["_fdr_decoder"] = install_fdr(net)
    if loss_gain:
        crit.loss_gain.update(loss_gain)
    if box_cls_gain:
        # names -> ids, so the candidate can say what it means
        crit.box_cls_gain = {int(CLASSES.index(c) if isinstance(c, str) else c): float(g)
                             for c, g in box_cls_gain.items()}
    if rep_gain:
        crit.rep_gain = float(rep_gain)
    net.criterion = crit
    log(f"  box loss: {kind}"
        + (f", alpha={alpha}" if alpha != 1.0 else "")
        + (", MAL (target IoU^1.5, positives at full weight)" if mal else "")
        + (f", vfl_beta={beta}" if beta and not mal else "")
        + (", D-FINE FDR (+FGL 0.15, GO-LSD/DDF 1.5) on the pretrained decoder" if fdr else "")
        + (", log-space wh" if log_size else "")
        + (f", gains {loss_gain}" if loss_gain else "")
        + (f", box loss x{sorted(set(box_cls_gain.values()))} for {sorted(box_cls_gain)}" if box_cls_gain else "")
        + (f", repulsion {rep_gain} (same-class neighbours, smooth-ln of IoG beyond the truth's own overlap)" if rep_gain else ""))
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


def warm_start(net, path):
    """Load a trained checkpoint's weights into a freshly built model, schedule reset.

    Not a resume: the optimizer, EMA and epoch counter start over; only the
    weights come across. Every tensor of the new model must be found, with one
    allowance -- a conv the new model widened (more input channels, e.g. the
    stem reading the 16 bands too) takes the checkpoint's weights on its first
    channels and keeps its zero-initialised extra ones. Anything else missing
    or mis-shaped is an error, not a silent partial load.
    """
    import torch
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    src = (ck.get("ema") or ck.get("model")) if isinstance(ck, dict) else ck
    if src is None:
        raise RuntimeError(f"warm start: no model in {path}")
    sd, own = src.float().state_dict(), net.state_dict()
    new, widened, bad = {}, [], []
    for k, v in own.items():
        t = sd.get(k)
        if t is None:
            bad.append(k)
        elif t.shape == v.shape:
            new[k] = t
        elif (t.dim() == 4 and v.dim() == 4 and t.shape[0] == v.shape[0]
              and t.shape[2:] == v.shape[2:] and t.shape[1] < v.shape[1]):
            w = torch.zeros_like(v, dtype=t.dtype)
            w[:, :t.shape[1]] = t
            new[k] = w
            widened.append(f"{k} {tuple(t.shape)}->{tuple(v.shape)}")
        else:
            bad.append(f"{k} {tuple(t.shape)} vs {tuple(v.shape)}")
    if bad:
        raise RuntimeError(f"warm start from {path}: {len(bad)} tensors not found or "
                           f"mis-shaped, first: {bad[:5]}")
    net.load_state_dict(new, strict=True)
    log(f"  warm start: {len(new)}/{len(own)} tensors from {path}"
        + (f"; widened: {widened}" if widened else ""))
    return len(new)


def freeze_batchnorm(net):
    """Every BatchNorm2d in the model to FrozenBatchNorm2d (eval mode for good)."""
    n = 0
    for m in net.modules():
        if type(m) is _nn.BatchNorm2d:
            m.__class__ = FrozenBatchNorm2d
            m.eval()
            n += 1
    return n


# What each part of the S3T-X detector is, for staged unfreezing:
#   head        the classification and box heads (decoder and encoder-query
#               selection) and the denoising class embedding -- the weights that
#               say what and where;
#   new         everything zero- or randomly initialised for this model: the S3T
#               fusion, pyramid and injections, and D-FINE's distribution heads;
#   mixer       the 16 -> 3 band projection in front of the pretrained stem;
#   decoder     the pretrained decoder layers, input projections, query
#               positional head and encoder output projection;
#   neck        the pretrained hybrid encoder (AIFI + CCFM);
#   s3t_enc     the MAE-pretrained S3T-X encoder;
#   backbone    HGNetv2 stages 1-4 (COCO);
#   stem        HGNetv2's stem (COCO);
#   stem_in     the stem's first conv when widened to also read all 16 bands
#               (s3t_stem_bands): its new input channels start at zero;
#   frozen_norm BatchNorm scale/shift in the stem and backbone.
UNFREEZE_HEAD_KEYS = ("dec_score_head", "dec_bbox_head", "enc_score_head",
                      "enc_bbox_head", "denoising_class_embed")
UNFREEZE_PARTS = ("head", "new", "mixer", "decoder", "neck", "s3t_enc",
                  "backbone", "stem", "stem_in", "frozen_norm")


def param_parts(net):
    """{id(param): (part, name)} for every parameter of an (unwrapped) detector."""
    net = getattr(net, "module", net)
    n_bb = len((getattr(net, "yaml", None) or {}).get("backbone") or []) or 10
    last = len(net.model) - 1
    norms = {id(p) for m in net.model[:n_bb] for mm in m.modules()
             if isinstance(mm, _nn.BatchNorm2d) for p in mm.parameters(recurse=False)}
    widened = {id(p) for mm in net.model[0].modules() if mm.__dict__.get("_hod26_widened")
               for p in mm.parameters(recurse=False)}
    out = {}
    for name, p in net.named_parameters():
        bits = name.split(".")
        if bits[0] != "model":
            raise RuntimeError(f"unfreeze: parameter outside net.model: {name}")
        i, rest = int(bits[1]), ".".join(bits[2:])
        if id(p) in norms:
            part = "frozen_norm"
        elif id(p) in widened:
            part = "stem_in"
        elif i == 0:
            part = ("s3t_enc" if rest.startswith("front.enc.") else
                    "mixer" if rest.startswith("front.base.") else
                    "new" if rest.startswith("front.") else "stem")
        elif i < n_bb:
            part = "backbone"
        elif i < last:
            wrapped = type(net.model[i]).__name__ in ("Inject", "ContextInject", "Tap")
            part = "new" if wrapped and not rest.startswith("layer.") else "neck"
        elif rest.startswith("decoder.fdr."):
            part = "new"
        else:
            part = "head" if bits[2] in UNFREEZE_HEAD_KEYS else "decoder"
        out[id(p)] = (part, name)
    return out


def unfreeze_mult(schedule, part, epoch):
    """LR multiplier of a part at an epoch: 0 until its first epoch, then a
    linear ramp over `ramp` epochs to its target. No entry: frozen for good."""
    s = schedule.get(part)
    if not s:
        return 0.0
    start, ramp, target = s
    if epoch < start:
        return 0.0
    return float(target) * (1.0 if ramp <= 0 else min(1.0, (epoch - start + 1) / ramp))


def split_groups_by_part(groups, net, schedule):
    """ultralytics' three groups (weight / bn / bias), each split by part.

    Every group keeps its own hyperparameters (and the param_group key the
    warmup reads); the part rides along as an extra key.
    """
    parts = param_parts(net)
    unknown = sorted({pt for pt, _ in parts.values()} - set(UNFREEZE_PARTS))
    if unknown:
        raise RuntimeError(f"unfreeze: unclassified parts {unknown}")
    extra = sorted(set(schedule) - set(UNFREEZE_PARTS))
    if extra:
        raise RuntimeError(f"unfreeze: schedule names unknown parts {extra}")
    out = []
    for g in groups:
        by = {}
        for p in g["params"]:
            if id(p) not in parts:
                raise RuntimeError("unfreeze: an optimizer parameter is not in the model")
            by.setdefault(parts[id(p)][0], []).append(p)
        for part in UNFREEZE_PARTS:
            if by.get(part):
                out.append({**{k: v for k, v in g.items() if k != "params"},
                            "params": by[part], "part": part})
    n_opt = sum(len(g["params"]) for g in out)
    if n_opt != sum(len(g["params"]) for g in groups):
        raise RuntimeError("unfreeze: the split lost parameters")
    return out


def attach_unfreeze(opt, trainer, schedule):
    """Scale each group's LR by its part's multiplier for the step, then put it back.

    Multipliers rather than requires_grad: DDP registers its gradient hooks
    once, on the parameters that require grad when it wraps the model, and
    ultralytics turns requires_grad back on for any float parameter it finds
    frozen. A multiplier of 0 is an exact freeze under AdamW (the update and
    the decoupled decay both scale with the LR), and the LR the warmup and the
    scheduler write is left alone -- only the step sees the product.
    """
    state = {"epoch": None, "saved": None}

    def pre(o, args, kwargs):
        e = int(getattr(trainer, "epoch", 0) or 0)
        if e != state["epoch"]:
            state["epoch"] = e
            if is_main_rank():
                log(f"  unfreeze epoch {e + 1}: " + ", ".join(
                    f"{pt} {unfreeze_mult(schedule, pt, e):g}" for pt in UNFREEZE_PARTS))
        state["saved"] = [g["lr"] for g in o.param_groups]
        for g in o.param_groups:
            g["lr"] = g["lr"] * unfreeze_mult(schedule, g.get("part"), e)

    def post(o, args, kwargs):
        for g, lr in zip(o.param_groups, state["saved"]):
            g["lr"] = lr

    opt.register_step_pre_hook(pre)
    opt.register_step_post_hook(post)
    opt.__dict__["_hod26_unfreeze"] = schedule
    return opt


def hod26_trainer(base_cls, adapter=None, coco_prior=True, schedule_epochs=0,
                  bbox_loss="GIoU", loss_gain=None, is_rtdetr=True,
                  bbox_alpha=1.0, vfl_beta=0.0, log_size_l1=False,
                  reset_best_fitness=True, accel=None, fdr=False, mal=False,
                  unfreeze=None, frozen_bn=False, box_cls_gain=None, rep_gain=0.0,
                  init_from=None):
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
        def setup_model(self):
            # Runs in the process that trains -- a DDP worker included, where
            # get_model never ran (see ensure_accel).
            ckpt = super().setup_model()
            if accel:
                self.__dict__["_hod26_accel"] = ensure_accel(
                    self.model, nc=self.data["nc"], fp32_loss=accel.get("fp32_loss", True),
                    compile_blocks=bool(adapter and adapter.get("compile_blocks")))
            return ckpt

        def build_optimizer(self, *args, **kwargs):
            opt = super().build_optimizer(*args, **kwargs)
            groups = opt.param_groups
            if unfreeze:
                groups = split_groups_by_part(groups, self.model, unfreeze)
                if is_main_rank():
                    counts = {}
                    for g in groups:
                        counts[g["part"]] = counts.get(g["part"], 0) + sum(p.numel() for p in g["params"])
                    log("  staged unfreezing (part: first epoch, ramp epochs, LR x; params): " + "; ".join(
                        f"{pt} {'frozen' if not unfreeze.get(pt) else tuple(unfreeze[pt])} "
                        f"{counts.get(pt, 0) / 1e6:.2f}M" for pt in UNFREEZE_PARTS))
            if not accel or not accel.get("fused_optimizer"):
                if unfreeze:
                    opt = attach_unfreeze(type(opt)(groups, lr=opt.defaults["lr"]), self, unfreeze)
                return opt
            import torch
            cls = type(opt)
            # Rebuilt from the same groups (each carries its own lr, momentum /
            # betas and weight decay), with the fused CUDA kernel. No fallback.
            # The flag has to go on each group: the groups already carry
            # fused=None from the first build, and a group's own value wins
            # over the constructor's default -- passing fused=True alone
            # silently kept the foreach kernels.
            for g in groups:
                g["fused"], g["foreach"] = True, None
            fused = cls(groups, lr=opt.defaults["lr"], fused=True)
            if unfreeze:
                attach_unfreeze(fused, self, unfreeze)
            is_fused = all(g.get("fused") is True for g in fused.param_groups)
            # What this process's model actually has (ensure_accel), not a flag.
            done = ensure_accel(self.model, nc=self.data["nc"], fp32_loss=accel.get("fp32_loss", True),
                                compile_blocks=bool(adapter and adapter.get("compile_blocks")))
            sdpa_ok = done["sdpa_mha"] > 0 and done["plain_mha"] == 0
            comp_ok = done["compiled"] == done["blocks"] > 0 if done["compile_wanted"] else None
            overridden = os.environ.get("HOD26_AMP_OVERRIDDEN") == "1"
            table = [
                ("AMP fp16 (+GradScaler)", bool(getattr(self, "amp", False)),
                 "ultralytics check_amp result" if not getattr(self, "amp", False) else
                 ("ultralytics check_amp failed; overridden (required, measured stable)"
                  if overridden else "")),
                ("loss + Hungarian matching in fp32", done["fp32_loss"], ""),
                ("RT-DETR attention via SDPA", sdpa_ok,
                 f"{done['sdpa_mha']} SDPA, {done['plain_mha']} plain nn.MultiheadAttention"),
                ("S3T blocks torch.compile", bool(comp_ok),
                 f"{done['compiled']}/{done['blocks']} blocks" if comp_ok is not None else "not requested"),
                (f"fused {cls.__name__}", is_fused, "" if is_fused else "a group is not fused"),
                ("cudnn.benchmark", done["cudnn_benchmark"], ""),
                ("mosaic canvas reuse (data loader)", done["mosaic_canvas_reuse"],
                 "" if done["mosaic_canvas_reuse"] else "patch not installed"),
                ("non-deterministic kernels (fastest cudnn / grid_sample)",
                 not bool(torch.backends.cudnn.deterministic),
                 "deterministic=True in the candidate" if torch.backends.cudnn.deterministic else ""),
                ("FlashAttention / TF32 / bf16", False, "not supported on T4 (sm75)"),
            ]
            self.__dict__["_hod26_accel_table"] = [(n, bool(on), why) for n, on, why in table]
            if is_main_rank():
                log("  acceleration table:")
                for name, on, why in table:
                    log(f"    {'ON ' if on else 'off'}  {name}" + (f"  ({why})" if why else ""))
            if not is_fused:
                raise RuntimeError("fused optimizer requested but a parameter group is not fused")
            missing = [n for n, ok in (("fp32 loss", done["fp32_loss"] or not accel.get("fp32_loss", True)),
                                       ("SDPA attention", sdpa_ok), ("cudnn.benchmark", done["cudnn_benchmark"]),
                                       ("S3T torch.compile", comp_ok is not False),
                                       ("mosaic canvas reuse", done["mosaic_canvas_reuse"])) if not ok]
            if missing:
                raise RuntimeError(f"accelerations requested but not in effect in this process: {missing}")
            if accel.get("require_amp") and not getattr(self, "amp", False):
                raise RuntimeError("AMP was requested but ultralytics turned it off "
                                   "(check_amp failed); stopping instead of training in fp32")
            return fused

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

        def resume_training(self, ckpt):
            """Resume the weights, but not the previous session's yardstick.

            best.pt is only rewritten when an epoch beats self.best_fitness,
            and _load_checkpoint_state restores that number from the
            checkpoint. The session this run continues had folded the 600
            validation frames into its training set, so the figure it recorded
            -- 0.727 -- measures memorisation, while this run holds those
            frames out and honestly scores about 0.69. Comparing the two picks
            nothing: the bar sits above anything this run can print, best.pt
            keeps the weights it arrived with, and eighteen epochs of training
            end up predicting from the checkpoint they started from.

            Clearing it restarts selection on this run's own scale, which is
            the only one its epochs are measured against. last.pt is
            unaffected, so a resume still continues from the right weights.
            """
            super().resume_training(ckpt)
            if self.resume and reset_best_fitness:
                previous = self.best_fitness
                self.best_fitness = None
                log(f"  best.pt selection restarts from this run's own scale "
                    f"(the checkpoint's {previous} was measured on a "
                    f"validation split it had trained on)")

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
            if is_rtdetr and any((bbox_loss not in ("", "GIoU"), loss_gain,
                                  bbox_alpha != 1.0, vfl_beta, log_size_l1, fdr, mal,
                                  box_cls_gain, rep_gain)):
                install_bbox_loss(net, self.data["nc"], bbox_loss or "GIoU",
                                  bbox_alpha, vfl_beta, log_size_l1, loss_gain, fdr=fdr, mal=mal,
                                  box_cls_gain=box_cls_gain, rep_gain=rep_gain)
            if adapter:
                install_spectral_adapter(net, **adapter)
            if frozen_bn:
                n_bn = freeze_batchnorm(net)
                log(f"  BatchNorm: {n_bn} layers frozen to COCO's running statistics "
                    f"(eval mode throughout; 2 images/card, no SyncBN)")
            if init_from and not resuming:
                # A fine-tune: the finished model's weights into this freshly
                # built one (widened layers keep their zero extra channels).
                warm_start(net, init_from)
            if accel:
                done = enable_transformer_accel(net, fp32_loss=accel.get("fp32_loss", True),
                                                nc=self.data["nc"])
                self.__dict__["_hod26_accel"] = done
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
    """The previous session's last.pt, attached as another kernel's output.

    The recursive fallback is not decoration. Kaggle does not mount a kernel's
    output at a stable depth -- the same artifact has come up at
    /kaggle/input/<slug>/ on one kernel and nested a level or two down on
    another, which is why data_root() searches for the dataset rather than
    guessing its path. find_weights() has had this fallback all along, which is
    why every prediction kernel found its checkpoint; this function did not,
    and so the chunked resume it exists to serve silently started from scratch
    every time. Sessions 1, 2 and 3 were three independent runs of 9, 19 and 10
    epochs rather than one run of 38, and the best model came out of the
    longest of the three rather than the sum.

    Returning None has to keep meaning "no previous session" for the first
    session of a run, so the loud failure belongs in the caller, which knows
    whether a checkpoint was expected. See require_resume in run_candidate.
    """
    for base in sorted(INPUT.glob("*")):
        if _looks_like_dataset(base):
            continue          # the planar frames, not a previous session
        for name in (f"{tag}_last.pt", "last.pt"):
            direct = base / name
            if direct.exists():
                return direct
            for c in sorted(base.glob(f"**/{name}")):
                return c
    return None


def checkpoint_sources(tag):
    """Every mount that could satisfy find_checkpoint, not just the one it picks.

    find_checkpoint returns the first match in sorted order, so two attached
    checkpoints resolve to whichever sorts first with nothing in the log to
    say a choice was made. "Silently resumed from the wrong session" is the
    same class of failure as "silently did not resume at all", which is what
    cost sessions 1 to 3; preflight refuses instead of picking.
    """
    found = []
    for base in sorted(INPUT.glob("*")):
        if _looks_like_dataset(base):
            continue
        for name in (f"{tag}_last.pt", "last.pt"):
            if (base / name).exists() or next(base.glob(f"**/{name}"), None):
                found.append(base)
                break
    return found


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


def is_main_rank():
    """True in a single-GPU run, and in rank 0 of a DDP one.

    torch.distributed.run sets RANK in every worker it spawns; nothing sets it
    otherwise. The callbacks below run inside those workers, so without this
    both ranks append to the same metrics file and copy the same 264 MB
    checkpoint to the same path at the same time. Only rank 0 validates, so
    rank 1's numbers are NaN anyway -- the guard drops a corruption risk and a
    stream of misleading log lines together.
    """
    try:
        return int(os.environ.get("RANK", -1)) in (-1, 0)
    except ValueError:
        return True


def metrics_records(tag):
    """Every epoch record rank 0 wrote, oldest first.

    Under DDP the validator, the trainer's epoch counter and the callback
    state all live in a subprocess the parent never sees, so this file is the
    only place the parent can read what the run actually did.
    """
    out = []
    try:
        for line in (WORK / f"{tag}_metrics.jsonl").read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    return out


def metrics_tail(tag):
    """The last record that is an epoch, or {}.

    A cut-short run ends with ultralytics re-validating the best checkpoint,
    which is not an epoch -- final_eval marks it, and it is skipped here for
    the same reason the trainer's own counter is preferred when there is one.
    """
    return next((r for r in reversed(metrics_records(tag))
                 if not r.get("final_eval") and r.get("per_class")), {})


def scores_from_results(results, tag):
    """mAP for the run, whichever shape ultralytics hands back.

    On one GPU this is the validator's metrics object. Under DDP the parent's
    validator never ran, so ultralytics falls back to the checkpoint's
    train_metrics: a flat dict of the same numbers with no per-class
    breakdown attached. Rank 0 wrote that breakdown to the metrics file epoch
    by epoch, so multi-GPU runs keep their per-class scores instead of
    silently reporting none.
    """
    box = getattr(results, "box", None)
    if box is not None:
        return {
            "mAP": float(box.map),          # mAP@[.5:.95], the competition's primary
            "mAP50": float(box.map50),
            "per_class": {CLASSES[int(c)]: float(a)
                          for c, a in zip(box.ap_class_index, box.maps[box.ap_class_index])}
            if getattr(box, "ap_class_index", None) is not None else {},
        }
    # The dict ultralytics falls back to is best.pt's *stored* train_metrics,
    # written by whichever session saved that checkpoint. A resumed run that
    # never beats the checkpoint it inherited keeps the inherited best, and so
    # reports the previous session's number as this run's holdout -- here,
    # 0.7271 from a session that had folded the validation frames into
    # training, for a run whose own re-validation said 0.6985. The single-GPU
    # path reads the validator after its final pass over *this* run's split,
    # and the final_eval record is that same pass, so preferring it keeps both
    # paths reporting the same measurement.
    final = next((r for r in reversed(metrics_records(tag))
                  if r.get("final_eval")), {})
    flat = final.get("metrics") or (results if isinstance(results, dict) else {})
    nan = float("nan")
    return {
        "mAP": float(flat.get("metrics/mAP50-95(B)", nan)),
        "mAP50": float(flat.get("metrics/mAP50(B)", nan)),
        "per_class": final.get("per_class") or metrics_tail(tag).get("per_class", {}),
    }


def snapshot_for_resume(tag, run):
    """Copy last.pt out while it still carries optimizer state.

    ultralytics strips the optimizer, the EMA and the epoch number from last.pt
    once training ends, which is exactly what a resume needs -- a stripped
    checkpoint restarts from epoch 0 with a fresh schedule and reports itself as
    a normal run. So the copy is taken per epoch, from on_model_save, before the
    strip can reach it.
    """
    if not is_main_rank():
        return
    for src, dst in ((run / "weights" / "last.pt", WORK / f"{tag}_last.pt"),
                     (run / "results.csv", WORK / f"{tag}_results.csv")):
        if src.exists():
            shutil.copy2(src, dst)
    # best.pt too, whenever an epoch rewrote it: the runs directory is scratch
    # and is not saved, so a session that ends in an exception would otherwise
    # keep last.pt and lose the best weights it had already found.
    best, kept = run / "weights" / "best.pt", WORK / f"{tag}_best.pt"
    if best.exists() and (not kept.exists() or best.stat().st_mtime > kept.stat().st_mtime):
        shutil.copy2(best, kept)


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
    # Bound here, not read from the global at call time. Under DDP this
    # callback is cloudpickled into a worker that imported this module minutes
    # after the session began, so its own T0 is not the session's -- and a
    # guard measuring from the wrong zero would sail past Kaggle's 12-hour cap
    # and lose the checkpoint it exists to protect. A closure cell travels with
    # the callback; a module global does not.
    t0 = T0

    def record(trainer):
        if not is_main_rank():
            return
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
        # Mean GPU utilisation over the epoch, per card: high means the GPU
        # is the limit, low means the data loader (4 vCPUs for 2 ranks) is.
        g = state.get("_gpu") or []
        if g:
            n = min(len(x) for x in g)
            rec["gpu_util"] = [round(sum(x[i] for x in g) / len(g)) for i in range(n)]
            g.clear()
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
            + (f"  gpu {rec['gpu_util']}%" if rec.get("gpu_util") else "")
            + (f"  drift {drift['rel']:.4%}" if drift else ""))

        if rec["final_eval"]:
            return
        state["last_epoch"] = rec["epoch"]
        if not budget_seconds:
            return
        elapsed = time.time() - t0
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
        if is_main_rank() and "_gpu" not in state:
            # Started here, in the process that trains: the callbacks are
            # pickled to the DDP workers before training, and a thread is not.
            import subprocess
            import threading
            samples = state.setdefault("_gpu", [])

            def sample():
                while True:
                    try:
                        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                                              "--format=csv,noheader,nounits"],
                                             capture_output=True, text=True, timeout=5).stdout
                        vals = [float(v) for v in out.split()]
                        if vals:
                            samples.append(vals)
                    except Exception:                           # noqa: BLE001
                        pass
                    time.sleep(5)

            threading.Thread(target=sample, daemon=True).start()

    model.add_callback("on_fit_epoch_end", record)
    model.add_callback("on_train_start", announce)
    model.add_callback("on_model_save", lambda t: snapshot_for_resume(tag, Path(t.save_dir)))
    return state


def attach_speed_probe(model, path, skip=4):
    """The training process writes its own seconds/iteration, peak memory and
    acceleration table to `path` at the end of the first epoch (rank 0).
    Registered as callbacks, so it travels to the DDP workers with them."""
    stamps = []

    def on_batch(trainer):
        if is_main_rank():
            stamps.append(time.time())

    def on_epoch(trainer):
        if not is_main_rank() or Path(path).exists():
            return
        import statistics
        import torch
        d = [b - a for a, b in zip(stamps[skip:], stamps[skip + 1:])]
        rec = {"iterations": len(stamps), "s_per_it": statistics.median(d) if d else None,
               "peak_gb": (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0,
               "table": trainer.__dict__.get("_hod26_accel_table") or [],
               "world": int(os.environ.get("WORLD_SIZE", 1))}
        Path(path).write_text(json.dumps(rec, default=str))

    model.add_callback("on_train_batch_end", on_batch)
    model.add_callback("on_train_epoch_end", on_epoch)


def run_smoke(cand, index, train_ids, val_ids, anns, sub, keep=False):
    """A few minutes on the real GPUs through the real training path, first.

    Both GPU-only failures of the S3T-X run surfaced only after the full render
    and model build (a crash in the DDP worker; accelerations silently absent
    in it). This trains the same candidate -- same trainer, DDP, compile, AMP,
    D-FINE -- on 40 frames for one epoch and checks, before the long run:
    the worker's acceleration table (build_optimizer already refuses a missing
    one), seconds per iteration after the compile warm-up against the probe's
    0.51 s/step, and that saving and validation work. No fallback: a failure
    stops the session with the numbers, after ~3 minutes instead of hours.
    """
    import copy
    sc = copy.deepcopy(cand)
    sc["train"]["epochs"] = 1
    sc["require_resume"] = False
    speed = WORK / "smoke_speed.json"
    speed.unlink(missing_ok=True)
    limit = float(sub.get("smoke_max_s_per_it", 0.82))
    t0 = time.time()
    log(f"SMOKE: 1 epoch on {min(40, len(train_ids))} frames through the real training path")
    try:
        run_candidate(sc, index, train_ids[:40], val_ids[:8], anns, "smoke",
                      root=SCRATCH / "ds_smoke", prerendered=False, speed_file=speed)
        rec = json.loads(speed.read_text()) if speed.exists() else None
        if rec is None:
            raise RuntimeError("the training process wrote no speed record")
        off = [n for n, on, why in rec["table"]
               if not on and why != "not requested"
               and not n.startswith("FlashAttention")
               and not (n.startswith("AMP") and not sc["train"].get("amp"))]
        spi = rec.get("s_per_it")
        if off:
            raise RuntimeError(f"accelerations off in the training process: {off}")
        if spi is None:
            raise RuntimeError(f"too few iterations to time ({rec['iterations']})")
        if spi > limit:
            raise RuntimeError(f"{spi:.3f} s/it after warm-up, over the {limit:.2f} s/it limit "
                               f"(probe: 0.51 s/step on one T4) -- not starting the long run")
        log(f"SMOKE ok: {spi:.3f} s/it after warm-up ({rec['iterations']} its, {rec['world']} GPU), "
            f"accel all ON, peak {rec['peak_gb']:.2f} GB/card, {time.time() - t0:.0f}s")
        kept = {n: (WORK / f"smoke_{n}.pt").exists() for n in ("best", "last")}
        if not all(kept.values()):
            raise RuntimeError(f"best/last not kept in the output: {kept}")
        log("SMOKE ok: smoke_best.pt and smoke_last.pt kept in the output")
        if keep:
            return RUNS / "smoke" / "weights" / "best.pt"
    except Exception as exc:
        log(f"SMOKE FAILED: {type(exc).__name__}: {str(exc)[:600]}")
        raise
    finally:
        if not keep:
            clean_smoke()


def clean_smoke():
    shutil.rmtree(SCRATCH / "ds_smoke", ignore_errors=True)
    shutil.rmtree(RUNS / "smoke", ignore_errors=True)
    for f in WORK.glob("smoke_*"):
        f.unlink(missing_ok=True)


def run_smoke_only(round_cfg, cand, index, train_ids, val_ids, anns, test_dir):
    """Every stage of a submission session, small: smoke training (DDP, every
    acceleration), validation, best/last kept, the fp16 final evaluation, then
    best.pt reloaded, a slice of the test set predicted and a submission written
    and checked. For proving a pipeline before a long run on scarce quota."""
    t0 = time.time()
    try:
        weights = run_smoke(cand, index, train_ids, val_ids, anns, round_cfg["submit"], keep=True)
        model = build_model(cand["train"]["model"], str(weights))
        test_ids = require_ids(sorted(int(p.stem) for p in test_dir.glob("*.png")), test_dir)[:24]
        preds, sizes = predict_test(model, cand, test_dir, test_ids)
        out = WORK / "smoke_submission.csv"
        n = write(out, preds, clip_to=sizes)
        header = out.read_text().splitlines()[0] if out.exists() else ""
        if n <= 0 or not header:
            raise RuntimeError(f"submission empty ({n} rows)")
        imgs = len({p[0] for p in preds})
        log(f"SMOKE ok: predicted {len(test_ids)} test frames with best.pt -> {n} rows over {imgs} "
            f"images; header {header!r}")
        log(f"SMOKE ALL OK in {time.time() - t0:.0f}s: train (DDP, accel) / val / best+last kept / "
            f"fp16 final eval / reload / predict / submission")
        (WORK / "results.json").write_text(json.dumps({"mode": "smoke_only", "ok": True,
                                                       "rows": n, "images": imgs}, indent=2))
    except Exception as exc:
        log(f"SMOKE FAILED: {type(exc).__name__}: {str(exc)[:600]}")
        raise
    finally:
        clean_smoke()
        shutil.rmtree(SCRATCH / "test_images", ignore_errors=True)


def run_candidate(cand, index, train_ids, val_ids, anns, tag, budget_seconds=0,
                  reserve_seconds=300, root=None, prerendered=True, speed_file=None):

    root = root or SCRATCH / f"ds_{channels_key(cand)}"
    log(f"  scratch {SCRATCH} ({free_gb(SCRATCH):.1f} GB free), "
        f"output {WORK} ({free_gb(WORK):.1f} GB free)")
    pre = find_prerendered(cand, train_ids, val_ids) if prerendered else None
    yaml = (adopt_prerendered(pre, root) if pre is not None
            else materialize(cand, index, train_ids, val_ids, anns, root))
    tr, inf = cand["train"], cand["infer"]

    # Defensive clamp: a candidate that reached here without normalization must
    # not burn a GPU session on an argument ultralytics will reject.
    close_mosaic = min(tr.get("close_mosaic", 5), max(0, tr["epochs"] - 1))

    resume_from = stage_checkpoint(tag)
    if cand.get("require_resume") and resume_from is None:
        # The expensive failure mode this run actually hit: no checkpoint found
        # looks exactly like a first session, so the run restarts from COCO and
        # reports a perfectly healthy curve from epoch 1. Three sessions and
        # thirteen GPU-hours went that way. A session that is meant to continue
        # one says so, and dies here instead -- a failed resume then costs the
        # minute it takes to mount the inputs rather than the whole allowance.
        listing = {b.name: sorted(q.name for q in b.glob("**/*.pt"))[:6]
                   for b in sorted(INPUT.glob("*"))}
        raise RuntimeError(
            f"require_resume is set but no {tag}_last.pt or last.pt was found "
            f"under {INPUT}; attached inputs and their checkpoints: {listing}")
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
        if tr.get("spectral_stem") == "s3t":
            mae = find_mae_checkpoint(tr.get("s3t_mae_file"))
            if mae is None and tr.get("s3t_require_pretrain", True):
                raise RuntimeError("spectral_stem=s3t needs the MAE checkpoint "
                                   f"(*_mae.pt) attached under {INPUT}")
            adapter = {"kind": "s3t", "n_bands": tr["in_channels"], "projection": proj,
                       "mae_ckpt": str(mae) if mae else None,
                       "scale": float(tr.get("s3t_scale", 0.5)),
                       "widen": bool(tr.get("s3t_widen", True)),
                       "context": bool(tr.get("s3t_context", True)),
                       "ckpt_chunks": int(tr.get("s3t_ckpt_chunks", 8)),
                       "compile_blocks": bool(tr.get("s3t_compile", False)),
                       "fast_kernels": bool(tr.get("s3t_fast_kernels", False)),
                       "train_encoder": bool(tr.get("s3t_train_encoder", True)),
                       "arch": tr.get("s3t_arch", "tokens"),
                       "grad_ckpt": bool(tr.get("s3t_grad_ckpt", True)),
                       "upsample": int(tr.get("s3t_upsample", 1)),
                       "stem_bands": bool(tr.get("s3t_stem_bands", False))}
        elif tr.get("spectral_stem", "adapter") == "adapter":
            adapter = {"n_bands": tr["in_channels"], "projection": proj,
                       "ckpt_name": tr["model"], "srf_k": tr.get("srf_k", 0),
                       "srf_width": tr.get("srf_width", 2.0)}
        else:
            attach_spectral_stem_init(model, tr["model"], tr["in_channels"], proj)
    init_path = None
    if tr.get("init_from") and not resume_from:
        init_path = find_weights(tr["init_from"])
        if init_path is None:
            raise RuntimeError(f"train.init_from={tr['init_from']} not found under {INPUT}")
        init_path = str(init_path)
    trainer_cls = hod26_trainer(base_trainer(tr["model"]), adapter=adapter,
                                coco_prior=tr.get("coco_prior", True),
                                schedule_epochs=int(tr.get("schedule_epochs", 0)),
                                bbox_loss=tr.get("bbox_loss", "GIoU"),
                                loss_gain=tr.get("loss_gain") or None,
                                is_rtdetr=tr["model"].startswith("rtdetr"),
                                bbox_alpha=float(tr.get("bbox_alpha", 1.0)),
                                vfl_beta=float(tr.get("vfl_beta", 0.0)),
                                log_size_l1=bool(tr.get("log_size_l1", False)),
                                fdr=bool(tr.get("fdr", False)), mal=bool(tr.get("mal", False)),
                                unfreeze=tr.get("unfreeze") or None,
                                frozen_bn=bool(tr.get("frozen_bn", False)),
                                box_cls_gain=tr.get("box_cls_gain") or None,
                                rep_gain=float(tr.get("rep_gain", 0.0)),
                                init_from=init_path,
                                accel=({"fp32_loss": bool(tr.get("amp_fp32_loss", True)),
                                        "fused_optimizer": True,
                                        "require_amp": bool(tr.get("amp", False))}
                                       if tr.get("spectral_stem") == "s3t" else None))
    if tr.get("spectral_stem") == "s3t":
        import torch
        torch.backends.cudnn.benchmark = True
        if tr.get("amp"):
            os.environ["HOD26_REQUIRE_AMP"] = "1"
    # One card or both, decided by what the session actually has rather than by
    # what the metadata asked for: a request for two that lands on one must not
    # take the run down with it. Per-card batch stays at tr["batch"], so each
    # GPU does exactly the work it did on a single-card run and the optimizer
    # still steps at nbs=64 -- the wall clock changes, the schedule does not.
    gpus = visible_gpus()
    use_ddp = gpus > 1 and materialise_kernel_module()
    if gpus > 1 and not use_ddp:
        log("  WARNING: two GPUs are visible but this script's source could not "
            "be found on disk, so the DDP workers could not import it. "
            "Training on one card.")
    ddp_args = {"device": list(range(gpus))} if use_ddp else {}
    batch = tr["batch"] * (gpus if use_ddp else 1)
    log(f"  {gpus} GPU(s) visible; "
        + (f"DDP across {list(range(gpus))}" if use_ddp else "single card")
        + f", batch {batch} ({tr['batch']}/card)")
    log_state = attach_epoch_log(model, tag, budget_seconds, reserve_seconds)
    if speed_file is not None:
        attach_speed_probe(model, speed_file)
    results = model.train(
        data=str(yaml), epochs=tr["epochs"], imgsz=tr["imgsz"], batch=batch,
        lr0=tr["lr0"], mosaic=tr["mosaic"], close_mosaic=close_mosaic,
        hsv_h=tr["hsv_h"], hsv_s=tr["hsv_s"], hsv_v=tr["hsv_v"],
        fliplr=tr["fliplr"], scale=tr["scale"], cos_lr=tr.get("cos_lr", True),
        multi_scale=tr.get("multi_scale", False),
        warmup_epochs=tr.get("warmup_epochs", 3.0), nbs=int(tr.get("nbs") or 64),
        # None: ultralytics' own default, which on CUDA converts the whole
        # model to channels_last; the S3T probe measures both.
        channels_last=tr.get("channels_last"),
        project=str(RUNS), name=tag, exist_ok=True,
        verbose=False, plots=False, val=True, seed=0,
        amp=tr.get("amp", True), deterministic=tr.get("deterministic", True),
        resume=bool(resume_from), trainer=trainer_cls, **ddp_args,
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

    scores = scores_from_results(results, tag)
    scores["adapter"] = drift
    # The epoch the run actually reached, which the clock guard can cut short.
    # Read from the trainer rather than counted from the log: the log's last
    # record is ultralytics re-validating the best checkpoint, which is not an
    # epoch, and getting this one too high would let the orchestrator call an
    # unfinished run done and never produce a submission.
    # Under DDP the parent's trainer never ran an epoch, so its counter reads
    # the start of training rather than the end of it -- claiming epoch 1 of a
    # run that reached 39. The metrics file is rank 0's own record and is the
    # only honest source there; log_state is empty for the same reason.
    reached = getattr(getattr(model, "trainer", None), "epoch", None)
    if use_ddp:
        scores["last_epoch"] = metrics_tail(tag).get("epoch", tr["epochs"])
    else:
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


# --- Test-time augmentation -------------------------------------------------
#
# ultralytics' `augment=True` is a no-op on this detector. RTDETRDetectionModel
# .predict (nn/tasks.py) accepts the flag and never reads it -- there is no
# _predict_augment branch the way DetectionModel has one -- so the "tta" arm of
# the first inference sweep measured the identical forward pass as "default"
# and came back 0.0002 apart, which is what an untested knob looks like.
#
# TTA is worth testing properly here because localization, not detection, is
# where the score goes: 82% of held-out boxes are already matched at IoU >= 0.75
# and only 1.4% of the error is missing or misclassified objects, while a
# uniform 3-pixel shift is enough to take mAP from 1.00 to 0.249. Averaging the
# same box across views cancels the independent part of that coordinate noise.
# Weighted Boxes Fusion is the right merge for it: unlike NMS, which keeps one
# member of a cluster and discards the rest, WBF replaces the cluster with its
# confidence-weighted mean, so every view contributes to the coordinates.
#
# Scale views are deliberately excluded. The upscale sweep already measured
# them -- imgsz 1024 -> 0.4076, 1280 -> 0.3847, 1536 -> 0.3362 -- because a
# DETR decoder's query priors are tuned to the training scale. Only
# scale-preserving views are offered: a horizontal flip, which the model is
# already equivariant to because it trains with fliplr=0.5, and a whole-pixel
# translation that moves objects relative to the feature grid without resizing
# anything.

VIEWS = {
    "id": (),
    "hflip": ("hflip",),
    "shift3": ("shift3",),
    "shift5": ("shift5",),
    "hflip_shift3": ("hflip", "shift3"),
}


def _apply_view(img, ops):
    """Transform a rendered frame for one TTA view. Shape is preserved."""
    for op in ops:
        if op == "hflip":
            img = img[:, ::-1]
        elif op.startswith("shift"):
            n = int(op[5:])
            # Pad top-left by n and crop back to size: the content moves down
            # and right by exactly n pixels and nothing is rescaled, so the
            # decoder still sees objects at the scale it was trained on.
            img = np.pad(img, ((n, 0), (n, 0), (0, 0)), mode="edge")[
                : img.shape[0], : img.shape[1]]
        else:
            raise ValueError(f"unknown view op {op!r}")
    return np.ascontiguousarray(img)


def _unapply_view(xyxy, ops, w, h):
    """Map boxes from one view's coordinates back to the frame's."""
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4).copy()
    for op in reversed(ops):
        if op == "hflip":
            x1 = w - xyxy[:, 2]
            xyxy[:, 2] = w - xyxy[:, 0]
            xyxy[:, 0] = x1
        elif op.startswith("shift"):
            xyxy[:, [0, 2]] -= int(op[5:])
            xyxy[:, [1, 3]] -= int(op[5:])
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h)
    return xyxy


def _wbf_cluster(boxes, scores, n_views, iou_thr, rescale, rescore=None):
    """Weighted Boxes Fusion over one frame's detections of one class.

    Boxes are taken in descending confidence. Each either joins the cluster it
    overlaps most (above `iou_thr`), whose fused box is then the running
    confidence-weighted mean of its members, or starts a new one. A cluster's
    score is the mean of its members', optionally scaled by how many of the
    views found it -- a box only one view saw is, on the evidence, less certain
    than one all of them saw.

    `rescore` turns the cluster's internal agreement into a second signal.
    Averaging coordinates only pays off to the extent the views' errors are
    independent, and for one checkpoint seen from several angles they are not;
    but how tightly a cluster's members agree is informative whether or not
    their errors are independent, and it estimates the one thing the
    confidence does not carry. Detection confidence is known to correlate
    weakly with localization quality, which costs AP directly because the
    metric integrates a *ranking* over each class -- a loose box ranked above
    a tight one is a real loss even when both are found. The usual fix is a
    trained IoU-prediction head; this is the same measurement taken with the
    model itself, the way Soft Teacher measures pseudo-box reliability by
    jittering a box and looking at the variance of the regressions.

    Agreement is the mean IoU between each joining member and the cluster as
    it stood, which needs no second pass over the members.
    """
    order = np.argsort(-scores)
    boxes, scores = boxes[order], scores[order]
    fused, fscore, weight, count, agree = [], [], [], [], []
    for b, s in zip(boxes, scores):
        j, best_iou = -1, 0.0
        if fused:
            fa = np.asarray(fused)
            xx1 = np.maximum(fa[:, 0], b[0])
            yy1 = np.maximum(fa[:, 1], b[1])
            xx2 = np.minimum(fa[:, 2], b[2])
            yy2 = np.minimum(fa[:, 3], b[3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            union = ((fa[:, 2] - fa[:, 0]) * (fa[:, 3] - fa[:, 1])
                     + (b[2] - b[0]) * (b[3] - b[1]) - inter)
            iou = inter / np.maximum(union, 1e-9)
            k = int(np.argmax(iou))
            if iou[k] >= iou_thr:
                j, best_iou = k, float(iou[k])
        if j < 0:
            fused.append(b.astype(np.float64))
            fscore.append(float(s))
            weight.append(float(s))
            count.append(1)
            agree.append(0.0)
            continue
        w = weight[j] + s
        fused[j] = (fused[j] * weight[j] + b * s) / max(w, 1e-9)
        count[j] += 1
        fscore[j] += (s - fscore[j]) / count[j]
        agree[j] += best_iou
        weight[j] = w
    out = np.asarray(fscore, dtype=np.float64)
    n = np.asarray(count)
    if rescale:
        out = out * np.minimum(n, n_views) / n_views
    if rescore:
        # A singleton has nothing to agree with. Scoring it 0 would delete the
        # tail of the metric, so it takes a floor and is judged on confidence.
        q = np.where(n > 1, np.asarray(agree) / np.maximum(n - 1, 1),
                     float(rescore.get("singleton", 0.5)))
        out = out * np.clip(q, 0.0, 1.0) ** float(rescore.get("beta", 1.0))
    return np.asarray(fused, dtype=np.float64), out


def wbf(per_view, n_views, iou_thr=0.65, rescale=True, max_det=300,
        rescore=None):
    """Fuse several views' rows into one set, per frame and per class.

    `per_view` is a list of row lists, each row (pid, cls, score, x1, y1, x2, y2)
    already mapped back to frame coordinates.
    """
    grouped = {}
    for rows in per_view:
        for pid, cls, score, x1, y1, x2, y2 in rows:
            grouped.setdefault((pid, cls), ([], []))
            grouped[(pid, cls)][0].append((x1, y1, x2, y2))
            grouped[(pid, cls)][1].append(score)
    by_frame = {}
    for (pid, cls), (boxes, scores) in grouped.items():
        fb, fs = _wbf_cluster(np.asarray(boxes, dtype=np.float64),
                              np.asarray(scores, dtype=np.float64),
                              n_views, iou_thr, rescale, rescore)
        by_frame.setdefault(pid, []).extend(
            (pid, cls, float(s), float(b[0]), float(b[1]), float(b[2]),
             float(b[3])) for b, s in zip(fb, fs))
    out = []
    for pid, rows in by_frame.items():
        rows.sort(key=lambda r: -r[2])
        out.extend(rows[:max_det])
    return out


def _read_frame(path):
    """Read back a staged frame the same way ultralytics' loader will."""
    if path.suffix == ".tiff":
        ok, pages = cv2.imreadmulti(str(path), flags=cv2.IMREAD_UNCHANGED)
        if not ok or not pages:
            raise RuntimeError(f"could not read {path}")
        return np.stack(pages, axis=2)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"could not read {path}")
    return img if img.ndim == 3 else img[:, :, None]


def predict_views(model, cand, src_dir, png_ids, views):
    """Run one forward pass per view and return each view's rows, unwarped.

    The mosaic decode is paid once: the identity view is rendered from the
    cubes and every other view is built by reading those staged frames back
    and transforming them, which is a file read rather than a de-mosaic. The
    frames are not held in memory -- at sixteen bands, a thousand of them is
    close to two gigabytes, and this runs beside a training session.
    """
    if "id" not in views:
        raise ValueError("the identity view is the reference; include it")
    kw = predict_kwargs(cand)
    base = SCRATCH / "tta_id"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    staged, sizes = {}, {}
    for n, pid in enumerate(png_ids):
        img = build_channels(load_planar(src_dir / f"{pid}.png"), cand["channels"])
        sizes[pid] = (img.shape[1], img.shape[0])
        staged[pid] = write_frame(base / str(pid), img)
        if n % 250 == 0:
            log(f"  decoded {n}/{len(png_ids)}")

    per_view = []
    for view in views:
        ops = VIEWS[view]
        if ops:
            staging = SCRATCH / "tta_view"
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True, exist_ok=True)
            of_path = {str(write_frame(staging / str(pid),
                                       _apply_view(_read_frame(staged[pid]), ops))): pid
                       for pid in png_ids}
        else:
            staging = base
            of_path = {str(staged[pid]): pid for pid in png_ids}

        t0, rows, seen = time.time(), [], 0
        for r in model.predict(str(staging), **kw):
            pid = of_path.get(str(Path(r.path).resolve()), of_path.get(str(r.path)))
            if pid is None:
                raise RuntimeError(f"prediction for an unexpected frame: {r.path}")
            raw = _rows(pid, r)
            if raw:
                w, h = sizes[pid]
                backed = _unapply_view([q[3:] for q in raw], ops, w, h)
                rows.extend((pid, q[1], q[2], *map(float, bb))
                            for q, bb in zip(raw, backed))
            seen += 1
        if seen != len(png_ids):
            raise RuntimeError(f"view {view}: predicted {seen} of {len(png_ids)}")
        log(f"  view {view:14s} {len(rows)} boxes in {time.time() - t0:.0f}s")
        per_view.append(rows)
        if ops:
            shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(base, ignore_errors=True)
    return per_view, sizes


def tta_sweep(model, cand, root, spec):
    """Measure TTA + WBF against the scorer that counts, on the held-out split.

    Every arm that shares a view set shares its forward passes: the views are
    predicted once and each fusion setting is then pure CPU over the cached
    rows. So the GPU cost is the number of distinct views, not the number of
    arms, and the fusion knobs are effectively free to sweep.
    """
    ann_dir = root / "train" / "annotations"
    ids = require_ids(sorted(int(p.stem) for p in ann_dir.glob("*.xml")), ann_dir)
    _, val_ids = split_ids(ids)
    anns = [parse(ann_dir / f"{pid}.xml") for pid in val_ids]

    arms = spec["arms"]
    needed = sorted({v for a in arms.values() for v in a["views"]})
    log(f"TTA sweep: {len(arms)} arms over views {needed}")
    cached, _ = predict_views(model, cand, root / "train" / "images", val_ids, needed)
    rows_by_view = dict(zip(needed, cached))

    # Keep the raw per-view boxes. Every fusion question asked afterwards --
    # another threshold, another rescoring exponent, a view combination nobody
    # thought of -- is then CPU work on a laptop instead of another GPU
    # session, and the allowance does not refresh before the deadline. As
    # float32 this is a few tens of megabytes; as JSON it would be gigabytes.
    np.savez_compressed(
        WORK / "tta_views.npz",
        views=np.array(needed),
        **{f"rows_{v}": np.asarray(rows_by_view[v], dtype=np.float64)
           for v in needed})
    log(f"  wrote tta_views.npz ({sum(len(r) for r in cached)} raw boxes "
        f"over {len(needed)} views)")

    out = {}
    base = evaluate(anns, rows_by_view["id"], per_class=True)
    out["base"] = {**base, "boxes": len(rows_by_view["id"]),
                   "note": "single view, no fusion -- the current submission path"}
    log(f"  {'base':28s} mAP={base['mAP']:.4f} mAP50={base['mAP50']:.4f}")
    for name, a in arms.items():
        t0 = time.time()
        fused = wbf([rows_by_view[v] for v in a["views"]], len(a["views"]),
                    iou_thr=a.get("iou", 0.65), rescale=a.get("rescale", True),
                    max_det=cand["infer"]["max_det"], rescore=a.get("rescore"))
        scored = evaluate(anns, fused, per_class=True)
        out[name] = {**scored, "boxes": len(fused), "arm": a,
                     "delta": round(scored["mAP"] - base["mAP"], 5),
                     "seconds": round(time.time() - t0, 1)}
        log(f"  {name:28s} mAP={scored['mAP']:.4f} mAP50={scored['mAP50']:.4f} "
            f"d={out[name]['delta']:+.4f} ({len(fused)} boxes)")
    best = max(out, key=lambda k: out[k]["mAP"])
    log(f"  best arm: {best} at mAP {out[best]['mAP']:.4f} "
        f"({out[best].get('delta', 0.0):+.4f} vs the current path)")
    out["_best"] = best
    return out


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
    """Predict the competition's test frames and write submission.csv.

    With `infer.tta_views` set, the frames go through the multi-view path and
    the views are fused with WBF instead of a single forward pass being taken
    as the answer. Organizers confirmed test-time augmentation does not count
    as an ensemble; this is one checkpoint looked at from several angles, not
    several checkpoints.
    """
    test_dir = root / "test" / "images"
    test_ids = require_ids(sorted(int(p.stem) for p in test_dir.glob("*.png")), test_dir)
    log(f"predicting {len(test_ids)} test images")
    views = cand["infer"].get("tta_views") or []
    if views:
        log(f"  TTA views {views}, WBF iou={cand['infer'].get('wbf_iou', 0.65)} "
            f"rescale={cand['infer'].get('wbf_rescale', True)}")
        per_view, sizes = predict_views(model, cand, test_dir, test_ids, views)
        preds = wbf(per_view, len(views),
                    iou_thr=cand["infer"].get("wbf_iou", 0.65),
                    rescale=cand["infer"].get("wbf_rescale", True),
                    max_det=cand["infer"]["max_det"],
                    rescore=cand["infer"].get("wbf_rescore"))
    else:
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
    if sub.get("tta_sweep"):
        out["tta"] = tta_sweep(model, cand, root, sub["tta_sweep"])
    if sub.get("score_val", True):
        out["val"] = score_val_split(model, cand, root)
    if sub.get("predict_test", True):
        out.update(predict_test_set(model, cand, root))
        out["predicted"] = True
    (WORK / "results.json").write_text(json.dumps(out, indent=2))


# Test frame ids are offset into their own range so they cannot collide with a
# training frame of the same number, which both sets have.
PSEUDO_OFFSET = 1_000_000


def pseudo_labels(root, conf: float, source: str = "submission.csv"):
    """Turn a previous session's test predictions into training annotations.

    Semi-supervised training on the competition's own unlabelled test frames.
    The labels are this pipeline's own predictions, never anything read from
    the organizers, so the leaderboard reading stays honest: a model that only
    memorizes its own pseudo-labels reproduces its own score, and any gain has
    to come from having adapted to frames it had not seen.

    Measured on the held-out split where truth is available, predictions kept
    at conf 0.6 are 91.7% precise, 93.2% complete, and -- the number that makes
    this worth doing -- 99.7% correct about which class. What the model is
    unsure of is whether an object is there and where its edges are, not what
    it is.
    """
    src = None
    for base in sorted(INPUT.glob("*")):
        if _looks_like_dataset(base):
            continue
        c = base / source
        if c.exists():
            src = c
            break
    if src is None:
        log(f"  no {source} among the attached kernels; training without pseudo-labels")
        return {}, {}

    import csv as _csv
    kept = {}
    with src.open() as fh:
        for r in _csv.DictReader(fh):
            if float(r["confidence"]) < conf:
                continue
            pid = int(r["image_id"])
            kept.setdefault(pid, []).append(Box(
                int(r["class_id"]), int(float(r["x1"])), int(float(r["y1"])),
                int(float(r["x2"])), int(float(r["y2"]))))

    test_dir = root / "test" / "images"
    anns, index = {}, {}
    for pid, boxes in kept.items():
        f = test_dir / f"{pid}.png"
        if not f.exists():
            continue
        h, w = load_planar(f).shape[:2]
        boxes = [b for b in boxes if 0 <= b.x1 < b.x2 <= w and 0 <= b.y1 < b.y2 <= h]
        if not boxes:
            continue
        key = PSEUDO_OFFSET + pid
        anns[key] = Annotation(key, w, h, 16, tuple(boxes))
        index[key] = f
    n = sum(len(a.boxes) for a in anns.values())
    log(f"  pseudo-labels from {src}: {len(anns)} test frames, {n} boxes "
        f"at conf >= {conf} ({n / max(1, len(anns)):.1f} per frame)")
    return anns, index


def run_submission(round_cfg):
    """Train one candidate at full fidelity and write submission.csv."""
    cand = round_cfg["submit"]["candidate"]
    root = data_root()
    ann_dir = root / "train" / "annotations"
    test_dir = root / "test" / "images"

    # use_all_train: config was already selected on val; refit on everything,
    # with a token val set that keeps YOLO happy.
    train_ids, val_ids = submission_split(root, round_cfg["submit"])
    log(f"submission fit: {len(train_ids)} train / {len(val_ids)} val")

    anns = {pid: parse(ann_dir / f"{pid}.xml") for pid in set(train_ids) | set(val_ids)}
    index = frame_index(root, "train", sorted(set(train_ids) | set(val_ids)))

    conf = float(round_cfg["submit"].get("pseudo_conf", 0) or 0)
    if conf:
        p_anns, p_index = pseudo_labels(root, conf)
        anns.update(p_anns)
        index.update(p_index)
        train_ids = list(train_ids) + sorted(p_anns)
        log(f"  training set is now {len(train_ids)} frames "
            f"({len(train_ids) - len(p_anns)} labelled + {len(p_anns)} pseudo)")

    # A chunked run does not know in advance which session will be the last one
    # -- the clock decides -- so "if_complete" lets the session that reaches the
    # target epoch be the one that predicts, and reserves the time to do it.
    target = int(cand["train"]["epochs"])
    want = round_cfg["submit"].get("predict", True)
    budget = float(round_cfg["submit"].get("session_hours", 0) or 0) * 3600
    reserve = 1800 if want else 300

    if round_cfg["submit"].get("smoke_only"):
        return run_smoke_only(round_cfg, cand, index, train_ids, val_ids, anns, test_dir)
    if round_cfg["submit"].get("smoke", True) and visible_gpus() > 0:
        run_smoke(cand, index, train_ids, val_ids, anns, round_cfg["submit"])
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


ULTRALYTICS_PIN = "8.4.155"


def preflight(round_cfg):
    """Check every import this session depends on, before it costs anything.

    A long session fails in one of two ways. It raises in the first minute and
    costs nothing, or it runs for hours on a wrong assumption and costs the
    allowance -- which on Kaggle does not come back and, near a deadline, can
    be the whole remaining budget. Everything checked here is of the second
    kind: it looks fine at startup and only shows up later, or does not show up
    at all.

    The precedent is find_checkpoint(), which searched one level under
    /kaggle/input, found nothing, returned None, and so restarted from COCO
    while reporting a healthy curve from epoch 1. Three sessions and thirteen
    GPU-hours went that way. Every item below is a thing that could do the same.
    """
    import shutil as _sh
    bad, note = [], []

    render_only = bool((round_cfg.get("submit") or {}).get("render_only"))
    try:
        import torch
        if render_only:
            note.append("render only: no GPU needed")
        elif not torch.cuda.is_available():
            bad.append("no GPU visible: the notebook's accelerator is off. "
                       "Settings -> Accelerator -> GPU T4 x2 before running.")
        else:
            n = torch.cuda.device_count()
            note.append(f"{n}x GPU {torch.cuda.get_device_name(0)}")
            # An accelerator that silently comes back smaller than the one
            # asked for is the expensive failure here: the run works, so
            # nothing raises, and the session spends its whole allowance at
            # half speed. Cheaper to refuse in the first minute.
            want = int((round_cfg.get("submit") or {}).get("require_gpus", 1))
            if n < want:
                bad.append(
                    f"asked for {want} GPUs and got {n}. Set the kernel's "
                    "machine_shape to NvidiaTeslaT4x2 (or the notebook's "
                    "Accelerator to GPU T4 x2) and run again -- nothing has "
                    "been spent.")
    except Exception as exc:                                    # noqa: BLE001
        bad.append(f"torch unavailable: {exc}")

    # The adapter, the loss overrides and the clock guard all reach into
    # ultralytics internals -- the module layout of the first block, the
    # classifier remap hook, trainer.stop, check_resume. A newer release can
    # move any of them, and the failure mode that costs money is the one where
    # nothing raises and the behaviour quietly differs.
    try:
        import ultralytics
        if ultralytics.__version__ != ULTRALYTICS_PIN:
            bad.append(f"ultralytics {ultralytics.__version__} but this kernel "
                       f"is written against {ULTRALYTICS_PIN}; the pinned "
                       f"install did not take")
        else:
            note.append(f"ultralytics {ultralytics.__version__}")
    except Exception as exc:                                    # noqa: BLE001
        bad.append(f"ultralytics unavailable: {exc}")

    try:
        import pycocotools  # noqa: F401
    except Exception as exc:                                    # noqa: BLE001
        bad.append(f"pycocotools unavailable: {exc}")

    # The dataset. A private dataset attached by someone who cannot see it
    # simply is not mounted, and data_root() then raises much later.
    try:
        root = data_root()
        ann = len(list((root / "train" / "annotations").glob("*.xml")))
        tr = len(list((root / "train" / "images").glob("*.png")))
        te = len(list((root / "test" / "images").glob("*.png")))
        note.append(f"data {root}: {tr} train / {ann} xml / {te} test")
        if (ann, tr, te) != (3000, 3000, 1000):
            bad.append(f"expected 3000 train / 3000 xml / 1000 test, "
                       f"got {tr}/{ann}/{te}")
    except Exception as exc:                                    # noqa: BLE001
        bad.append(f"dataset not found ({exc}). Attached inputs: "
                   f"{[q.name for q in sorted(INPUT.glob('*'))]}. If the "
                   f"planar dataset is private, its owner must add this "
                   f"account as a collaborator before it can be mounted.")

    sub = round_cfg.get("submit") or {}
    cand = sub.get("candidate") or {}

    # COCO weights. ultralytics fetches these from GitHub on first use, so a
    # notebook with internet off trains a randomly initialised detector for
    # hours instead of failing.
    name = (cand.get("train") or {}).get("model", "rtdetr-l")
    try:
        from ultralytics.utils.downloads import attempt_download_asset
        got = attempt_download_asset(f"{name}.pt")
        if not Path(got).exists():
            raise RuntimeError(f"{got} missing after download")
        note.append(f"{name}.pt {Path(got).stat().st_size / 1e6:.0f} MB")
    except Exception as exc:                                    # noqa: BLE001
        bad.append(f"could not fetch {name}.pt ({exc}). Settings -> Internet "
                   f"must be on, or the COCO weights never arrive and the "
                   f"detector trains from random initialisation.")

    # The resume, checked here rather than after the dataset is materialised.
    if cand.get("require_resume"):
        ck = find_checkpoint("final")
        if ck is None:
            bad.append("require_resume is set but no final_last.pt/last.pt is "
                       "reachable. Attached: " + str({
                           q.name: sorted(x.name for x in q.glob("**/*.pt"))[:6]
                           for q in sorted(INPUT.glob("*"))}))
        else:
            srcs = checkpoint_sources("final")
            if len(srcs) > 1:
                bad.append(
                    "more than one checkpoint is attached and the resume takes "
                    "whichever sorts first, which would be "
                    f"{srcs[0].name}. Detach all but the one to resume from. "
                    "Attached: " + ", ".join(q.name for q in srcs))
            else:
                note.append(f"resume from {ck}")
    if cand and not render_only and not sub.get("weights_from"):
        # A mounted render must be the one this run would make -- checked now,
        # not after the MAE, the model and the DDP spawn.
        try:
            pre = find_prerendered(cand, *submission_split(data_root(), sub))
            note.append(f"prerendered dataset {pre}" if pre else
                        "no prerendered dataset attached: rendering here")
        except Exception as exc:                                 # noqa: BLE001
            bad.append(str(exc))
    if (cand.get("train") or {}).get("spectral_stem") == "s3t" and not render_only:
        try:
            mae = find_mae_checkpoint((cand.get("train") or {}).get("s3t_mae_file"))
        except RuntimeError as exc:
            bad.append(str(exc))
            mae = "ambiguous"
        if mae is None and (cand.get("train") or {}).get("s3t_require_pretrain", True):
            bad.append("spectral_stem=s3t but no MAE checkpoint (*_mae.pt) is "
                       "attached. Add the pretraining notebook's output "
                       "(zetaoxia/hod26-s3t-mae-pretrain3 for S3T-X) as an input. Attached: "
                       + str([q.name for q in sorted(INPUT.glob('*'))]))
        elif mae is not None and mae != "ambiguous":
            want = (cand.get("train") or {}).get("s3t_arch", "tokens")
            try:
                import torch
                got = (torch.load(mae, map_location="cpu", weights_only=True).get("config")
                       or {}).get("arch", "tokens")
            except Exception as exc:                            # noqa: BLE001
                got = f"unreadable ({exc})"
            if got != want:
                bad.append(f"MAE checkpoint {mae} is arch={got!r}; the run asks for "
                           f"s3t_arch={want!r}")
            else:
                note.append(f"S3T MAE encoder {mae} (arch {got})")
    if sub.get("weights_from"):
        if find_weights(sub["weights_from"]) is None:
            bad.append(f"weights_from={sub['weights_from']} not found under "
                       f"{INPUT}")
    init = ((cand or {}).get("train") or {}).get("init_from")
    if init and not sub.get("weights_from") and not render_only:
        if find_weights(init) is None:
            bad.append(f"train.init_from={init} (the checkpoint to fine-tune from) not found "
                       f"under {INPUT}; attach the kernel that produced it")
        else:
            note.append(f"fine-tune from {find_weights(init)}")

    # Scratch. A full disk surfaces as a cryptic write error deep in training.
    try:
        free = free_gb(SCRATCH)
        note.append(f"scratch {SCRATCH} {free:.1f} GB free")
        if free < 20:
            bad.append(f"only {free:.1f} GB free at {SCRATCH}; rendering 3000 "
                       f"frames plus augmented copies needs about 20")
    except Exception as exc:                                    # noqa: BLE001
        bad.append(f"scratch unusable: {exc}")

    for line in note:
        log(f"  preflight ok: {line}")
    if bad:
        for line in bad:
            log(f"  PREFLIGHT FAILED: {line}")
        raise RuntimeError(f"{len(bad)} preflight check(s) failed; refusing to "
                           f"spend a GPU session on a run that cannot finish")
    log("preflight passed")


def main():
    round_cfg = json.loads(Path(__file__).with_name("round.json").read_text()) \
        if Path(__file__).with_name("round.json").exists() else ROUND_CONFIG
    preflight(round_cfg)

    if round_cfg.get("submit"):
        if round_cfg["submit"].get("render_only"):
            return run_render(round_cfg)
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


def _register_kernel_classes():
    """Every class this script defines, under the stable module name hod26_kernel.

    A checkpoint names its classes by module; this script is __main__ here and
    hod26_kernel in the DDP workers (materialise_kernel_module). cloudpickle
    sends a __main__ class to the workers *by value*, and the worker's own
    torch.save (plain pickle) then cannot name it -- the first S3T-X smoke died
    exactly so, on S3TXFront, at the first checkpoint. Registering all of them
    (the inlined S3T modules included, not a hand-kept list) makes every class
    travel by reference to a module the worker can import.
    """
    import sys as _sys
    import types as _types
    mod = _sys.modules.setdefault("hod26_kernel", _types.ModuleType("hod26_kernel"))
    here = __name__
    n = 0
    for name, obj in list(globals().items()):
        if isinstance(obj, type) and obj.__module__ == here and not name.startswith("__"):
            if here != "hod26_kernel":
                obj.__module__ = "hod26_kernel"
            setattr(mod, name, obj)
            n += 1
    return n


_register_kernel_classes()


if __name__ == "__main__":
    main()
