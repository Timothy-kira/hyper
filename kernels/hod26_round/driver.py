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
    """Locate the planar dataset.

    Kaggle may serve an upload extracted, nested one level down, or still as the
    archives the CLI produced -- accept all three rather than spend a kernel
    session discovering which one happened.
    """
    candidates = [DATA]
    if Path("/kaggle/input").exists():
        candidates += sorted(Path("/kaggle/input").glob("*"))

    for cand in candidates:
        if not cand.exists():
            continue
        for probe in (cand, cand / "hod26_planar"):
            if _looks_like_dataset(probe):
                return probe

    for cand in candidates:                       # fall back to archives
        zips = sorted(cand.glob("*.zip")) if cand.is_dir() else []
        if not zips:
            continue
        out = WORK / "unpacked"
        out.mkdir(parents=True, exist_ok=True)
        for z in zips:
            log(f"  unpacking {z.name}")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out / z.stem)
        for extra in cand.glob("*.json"):
            shutil.copy(extra, out / extra.name)
        # Whether the CLI kept a "train/" prefix inside each archive is not
        # knowable from here, so find the directory that actually holds the
        # split rather than guessing the nesting.
        for d in [out, *(p for p in out.rglob("*") if p.is_dir())]:
            if _looks_like_dataset(d):
                return d

    listing = sorted(p.name for p in Path("/kaggle/input").iterdir()) \
        if Path("/kaggle/input").exists() else "/kaggle/input does not exist"
    raise FileNotFoundError(f"planar dataset not attached. /kaggle/input holds: {listing}")


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
    if mode == "bandsel":
        return np.dstack([stretch(cube[:, :, b], lo, hi) for b in BEST_BANDS])
    if mode == "band_stack":
        # Every band as its own input channel. Ultralytics reads this natively:
        # a multi-page TIFF is decoded with imdecodemulti and stacked on axis 2,
        # the model is built with ch=data["channels"], and the HSV augmentation
        # skips anything that is not 3-channel.
        return np.dstack([stretch(cube[:, :, b], lo, hi) for b in range(cube.shape[2])])
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


def channels_key(spec):
    """Identity of a rendered dataset: candidates differing only in training or
    inference parameters consume byte-identical images."""
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]


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

    for split, ids in (("train", train_ids), ("val", val_ids)):
        for pid in ids:
            img = build_channels(load_planar(index[pid]), cand["channels"])
            # No BGR flip: these are spectral bands, not colour. What matters is
            # that training and inference write and read them the same way.
            # cv2.imwrite returns False (it does not raise) on a bad buffer, and
            # a non-contiguous view is one -- hence the explicit check.
            n_ch = img.shape[2]
            write_frame(root / "images" / split / str(pid), img)
            a = anns[pid]
            lines = []
            for b in a.boxes:
                cx = (b.x1 + b.x2) / 2 / a.width
                cy = (b.y1 + b.y2) / 2 / a.height
                bw = (b.x2 - b.x1) / a.width
                bh = (b.y2 - b.y1) / a.height
                lines.append(f"{b.cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (root / "labels" / split / f"{pid}.txt").write_text("\n".join(lines))

    for split, ids in (("train", train_ids), ("val", val_ids)):
        n = len(list((root / "images" / split).glob("*.png"))) + \
            len(list((root / "images" / split).glob("*.tiff")))
        if n == 0 or n != len(ids):
            raise RuntimeError(f"{split}: wrote {n} images, expected {len(ids)}")

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


def build_model(name, weights=None):
    """Instantiate the detector. RT-DETR has its own model class in ultralytics."""
    from ultralytics import RTDETR, YOLO
    cls = RTDETR if name.startswith("rtdetr") else YOLO
    return cls(weights or f"{name}.pt")


def run_candidate(cand, index, train_ids, val_ids, anns, tag):

    root = WORK / f"ds_{channels_key(cand['channels'])}"
    yaml = materialize(cand, index, train_ids, val_ids, anns, root)
    tr, inf = cand["train"], cand["infer"]

    # Defensive clamp: a candidate that reached here without normalization must
    # not burn a GPU session on an argument ultralytics will reject.
    close_mosaic = min(tr.get("close_mosaic", 5), max(0, tr["epochs"] - 1))

    model = build_model(tr["model"])
    results = model.train(
        data=str(yaml), epochs=tr["epochs"], imgsz=tr["imgsz"], batch=tr["batch"],
        lr0=tr["lr0"], mosaic=tr["mosaic"], close_mosaic=close_mosaic,
        hsv_h=tr["hsv_h"], hsv_s=tr["hsv_s"], hsv_v=tr["hsv_v"],
        fliplr=tr["fliplr"], scale=tr["scale"], cos_lr=tr.get("cos_lr", True),
        project=str(WORK / "runs"), name=tag, exist_ok=True,
        verbose=False, plots=False, val=True, seed=0,
        amp=tr.get("amp", True), deterministic=tr.get("deterministic", True),
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
    from ultralytics import YOLO

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
