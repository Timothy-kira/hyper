"""HOD26 round driver — executes one Dream-RSI decision round on a Kaggle GPU.

One kernel session evaluates the whole batch of candidates the exploration
policy selected, so the cost of decoding 3000 spectral cubes is paid once per
round instead of once per candidate. Results land in /kaggle/working/results.json
for the orchestrator to fold back into the discovery tree.

The hod26.* helpers above this docstring are injected verbatim from the repo by
tools/build_kernel.py -- this file is the round-specific part only.
"""

import json, os, shutil, time, traceback
from pathlib import Path

import numpy as np

COMP = "/kaggle/input/hyperspectral-object-detection-challenge-2026"
WORK = Path("/kaggle/working")
CACHE = WORK / "cache"
VAL_FRACTION = 0.2
CACHE_SEED = 20260918


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- data ------
def split_ids(all_ids):
    """Deterministic train/val split, stable across every candidate and round."""
    rng = np.random.RandomState(CACHE_SEED)
    ids = sorted(all_ids)
    perm = rng.permutation(len(ids))
    n_val = int(round(len(ids) * VAL_FRACTION))
    val = {ids[i] for i in perm[:n_val]}
    return [i for i in ids if i not in val], [i for i in ids if i in val]


def decode_all(png_dir, ids, out_dir):
    """Decode every mosaic PNG to a uint16 cube memmap, once per session."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for n, pid in enumerate(ids):
        dst = out_dir / f"{pid}.npy"
        if not dst.exists():
            np.save(dst, load_cube(png_dir / f"{pid}.png"))
        index[pid] = dst
        if n % 250 == 0:
            log(f"  decoded {n}/{len(ids)}")
    return index


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
    bands = spec["bands"][:3]
    return np.dstack([stretch(cube[:, :, b], lo, hi) for b in bands])


def materialize(cand, index, train_ids, val_ids, anns, root):
    """Write the YOLO dataset this candidate trains on."""
    import cv2
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, ids in (("train", train_ids), ("val", val_ids)):
        for pid in ids:
            img = build_channels(np.load(index[pid]), cand["channels"])
            # No BGR flip: these are spectral bands, not colour. What matters is
            # that training and inference write and read them the same way.
            # cv2.imwrite returns False (it does not raise) on a bad buffer, and
            # a non-contiguous view is one -- hence the explicit check.
            dst = root / "images" / split / f"{pid}.png"
            if not cv2.imwrite(str(dst), np.ascontiguousarray(img)):
                raise RuntimeError(f"cv2.imwrite failed for {dst}")
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
        n = len(list((root / "images" / split).glob("*.png")))
        if n != len(ids):
            raise RuntimeError(f"{split}: wrote {n} images, expected {len(ids)}")

    yaml = root / "data.yaml"
    yaml.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(CLASSES)}\nnames: {json.dumps(CLASSES)}\n"
    )
    return yaml


# ------------------------------------------------------------ evaluate ------
def run_candidate(cand, index, train_ids, val_ids, anns, tag):
    from ultralytics import YOLO

    root = WORK / f"ds_{tag}"
    yaml = materialize(cand, index, train_ids, val_ids, anns, root)
    tr, inf = cand["train"], cand["infer"]

    model = YOLO(f"{tr['model']}.pt")
    model.train(
        data=str(yaml), epochs=tr["epochs"], imgsz=tr["imgsz"], batch=tr["batch"],
        lr0=tr["lr0"], mosaic=tr["mosaic"], close_mosaic=tr.get("close_mosaic", 5),
        hsv_h=tr["hsv_h"], hsv_s=tr["hsv_s"], hsv_v=tr["hsv_v"],
        fliplr=tr["fliplr"], scale=tr["scale"], cos_lr=tr.get("cos_lr", True),
        project=str(WORK / "runs"), name=tag, exist_ok=True,
        verbose=False, plots=False, val=False, seed=0,
    )

    preds = []
    for pid in val_ids:
        r = model.predict(str(root / "images" / "val" / f"{pid}.png"),
                          conf=inf["conf"], iou=inf["iou"], max_det=inf["max_det"],
                          augment=inf["tta"], verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            continue
        xyxy = b.xyxy.cpu().numpy()
        for (x1, y1, x2, y2), c, s in zip(xyxy, b.cls.cpu().numpy(), b.conf.cpu().numpy()):
            preds.append((pid, int(c), float(s), float(x1), float(y1), float(x2), float(y2)))

    scores = evaluate([anns[p] for p in val_ids], preds, per_class=True)
    weights = WORK / "runs" / tag / "weights" / "best.pt"
    return scores, preds, (str(weights) if weights.exists() else None)


def predict_test(model, cand, test_dir, png_ids):
    """Run the trained model over the test set at cube resolution."""
    import cv2
    inf = cand["infer"]
    preds, sizes = [], {}
    staging = WORK / "test_images"
    staging.mkdir(parents=True, exist_ok=True)
    for n, pid in enumerate(png_ids):
        img = build_channels(load_cube(test_dir / f"{pid}.png"), cand["channels"])
        sizes[pid] = (img.shape[1], img.shape[0])
        path = staging / f"{pid}.png"
        if not cv2.imwrite(str(path), np.ascontiguousarray(img)):
            raise RuntimeError(f"cv2.imwrite failed for {path}")
        r = model.predict(str(path), conf=inf["conf"], iou=inf["iou"],
                          max_det=inf["max_det"], augment=inf["tta"], verbose=False)[0]
        path.unlink()
        b = r.boxes
        if b is not None and len(b):
            xyxy = b.xyxy.cpu().numpy()
            for (x1, y1, x2, y2), c, sc in zip(xyxy, b.cls.cpu().numpy(), b.conf.cpu().numpy()):
                preds.append((pid, int(c), float(sc), float(x1), float(y1), float(x2), float(y2)))
        if n % 200 == 0:
            log(f"  predicted {n}/{len(png_ids)}")
    return preds, sizes


def run_submission(round_cfg):
    """Train one candidate at full fidelity and write submission.csv."""
    from ultralytics import YOLO

    cand = round_cfg["submit"]["candidate"]
    ann_dir = Path(COMP) / "data_train/data_train/Annotations/VIS"
    png_dir = Path(COMP) / "data_train/data_train/VIS"
    test_dir = Path(COMP) / "data_test/data_test/VIS"

    ids = sorted(int(p.stem) for p in ann_dir.glob("*.xml"))
    train_ids, val_ids = split_ids(ids)
    if round_cfg["submit"].get("use_all_train", True):
        # Config was already selected on val; refit on everything for the final run.
        train_ids, val_ids = ids, val_ids[:60]   # a token val set keeps YOLO happy
    log(f"submission fit: {len(train_ids)} train / {len(val_ids)} val")

    anns = {pid: parse(ann_dir / f"{pid}.xml") for pid in set(train_ids) | set(val_ids)}
    index = decode_all(png_dir, sorted(set(train_ids) | set(val_ids)), CACHE)
    scores, _, weights = run_candidate(cand, index, train_ids, val_ids, anns, "final")
    log(f"fit done; holdout mAP={scores['mAP']:.4f} (optimistic: seen in training)")

    model = YOLO(weights)
    test_ids = sorted(int(p.stem) for p in test_dir.glob("*.png"))
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

    ann_dir = Path(COMP) / "data_train/data_train/Annotations/VIS"
    png_dir = Path(COMP) / "data_train/data_train/VIS"
    ids = sorted(int(p.stem) for p in ann_dir.glob("*.xml"))
    train_ids, val_ids = split_ids(ids)

    limit = round_cfg.get("proxy_train_images")
    if limit:
        train_ids = train_ids[:limit]
        val_ids = val_ids[:round_cfg.get("proxy_val_images", len(val_ids))]
    log(f"{len(train_ids)} train / {len(val_ids)} val images")

    anns = {pid: parse(ann_dir / f"{pid}.xml") for pid in train_ids + val_ids}
    log("decoding cubes (once for the whole round)")
    index = decode_all(png_dir, train_ids + val_ids, CACHE)

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
                "n_preds": len(preds), "weights": weights,
            })
            log(f"  -> mAP={scores['mAP']:.4f}  mAP50={scores['mAP50']:.4f}")
        except Exception:
            rec.update(score=None, error=traceback.format_exc()[-2000:])
            log(f"  !! failed:\n{rec['error']}")
        rec["cost_seconds"] = time.time() - t0
        results.append(rec)
        (WORK / "results.json").write_text(json.dumps(
            {"round": round_cfg.get("round"), "results": results}, indent=2))
        shutil.rmtree(WORK / f"ds_{node_id}", ignore_errors=True)

    log(f"round complete: {sum(r['score'] is not None for r in results)}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
