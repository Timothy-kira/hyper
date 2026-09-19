"""End-to-end check of the kernel's data path, without a GPU.

Everything the round kernel does before it hands a dataset to ultralytics is
exercised here against a miniature dataset built from the sample frames. The
two smoke failures on Kaggle were both in this stretch of code, and each cost a
push-and-wait cycle to discover; this catches that class of bug locally.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from hod26.cube import load_cube, to_planar  # noqa: E402
from hod26.voc import CLASSES, parse  # noqa: E402

SAMPLES = [10, 440, 558]


def build_fake_dataset(dest: Path, n_train: int = 12, n_test: int = 4) -> Path:
    """A planar dataset shaped like the real one, from the checked-in samples."""
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "train" / "images").mkdir(parents=True)
    (dest / "train" / "annotations").mkdir(parents=True)
    (dest / "test" / "images").mkdir(parents=True)

    for i in range(n_train):
        base = SAMPLES[i % len(SAMPLES)]
        pid = 1000 + i
        Image.fromarray(to_planar(load_cube(REPO / f"sample/img/{base}.png"))).save(
            dest / "train" / "images" / f"{pid}.png", compress_level=1)
        xml = (REPO / f"sample/ann/{base}.xml").read_text()
        (dest / "train" / "annotations" / f"{pid}.xml").write_text(
            xml.replace(f"<filename>{base}.png", f"<filename>{pid}.png"))

    for i in range(n_test):
        base = SAMPLES[i % len(SAMPLES)]
        Image.fromarray(to_planar(load_cube(REPO / f"sample/img/{base}.png"))).save(
            dest / "test" / "images" / f"{2000 + i}.png", compress_level=1)
    return dest


def load_kernel(work: Path, data: Path):
    """Import the generated kernel with its pip bootstrap and paths neutralized.

    Builds it first so the test always exercises the current driver rather than
    whatever a previous run happened to leave behind.
    """
    out = REPO / "kernels" / "hod26_round" / "build" / "_test"
    build = out / "hod26_round.py"
    cfg = out / "round_config.json"
    out.mkdir(parents=True, exist_ok=True)
    cfg.write_text('{"round": "test", "candidates": []}')
    subprocess.run(
        [sys.executable, str(REPO / "tools" / "build_kernel.py"),
         "--round-config", str(cfg), "--out-dir", str(out)],
        check=True, capture_output=True, text=True,
    )
    spec = importlib.util.spec_from_file_location("hod26_kernel_under_test", build)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    real_run, subprocess.run = subprocess.run, lambda *a, **k: None
    try:
        spec.loader.exec_module(mod)
    finally:
        subprocess.run = real_run
    mod.DATA, mod.WORK = data, work
    # Scratch is a separate volume on Kaggle; here it just has to not be the
    # output directory, so the test sees the same split the kernel does.
    mod.SCRATCH = work.parent / "scratch"
    mod.RUNS = mod.SCRATCH / "runs"
    work.mkdir(parents=True, exist_ok=True)
    mod.SCRATCH.mkdir(parents=True, exist_ok=True)
    return mod


def test_datapath(tmp_root: Path) -> None:
    from dream_rsi.candidate import CHANNEL_MODES, seed_candidate

    data = build_fake_dataset(tmp_root / "input" / "hod26-planar")
    k = load_kernel(tmp_root / "work", data)

    root = k.data_root()
    assert root == data, root

    ann_dir = root / "train" / "annotations"
    ids = k.require_ids(sorted(int(p.stem) for p in ann_dir.glob("*.xml")), ann_dir)
    train_ids, val_ids = k.split_ids(ids)
    assert train_ids and val_ids and not set(train_ids) & set(val_ids)
    # The split must not drift between candidates or rounds, or scores in the
    # discovery tree stop being comparable.
    assert (train_ids, val_ids) == k.split_ids(list(reversed(ids)))

    anns = {p: k.parse(ann_dir / f"{p}.xml") for p in ids}
    index = k.frame_index(root, "train", ids)

    import cv2

    for mode in CHANNEL_MODES:
        cand = seed_candidate(channels__mode=mode)
        n_ch = cand["train"]["in_channels"]
        ext = ".tiff" if n_ch > 3 else ".png"
        ds = k.WORK / f"ds_{k.channels_key(cand)}"
        yaml = k.materialize(cand, index, train_ids, val_ids, anns, ds)
        assert yaml.exists()
        assert f"channels: {n_ch}" in yaml.read_text(), (mode, yaml.read_text())

        for split, split_ids in (("train", train_ids), ("val", val_ids)):
            imgs = sorted((ds / "images" / split).glob(f"*{ext}"))
            assert len(imgs) == len(split_ids), (mode, split, len(imgs))
            # Read it exactly the way ultralytics.utils.patches.imread does, so
            # the test fails here rather than on a GPU if that path changes.
            buf = np.fromfile(imgs[0], np.uint8)
            if ext == ".tiff":
                ok, frames = cv2.imdecodemulti(buf, cv2.IMREAD_UNCHANGED)
                assert ok and len(frames) == n_ch, (mode, len(frames) if ok else "decode failed")
                arr = np.stack(frames, axis=2)
            else:
                arr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            assert arr.shape[2] == n_ch and arr.dtype == np.uint8, (mode, arr.shape, arr.dtype)
        # rendering twice must reuse, not rebuild
        assert k.materialize(cand, index, train_ids, val_ids, anns, ds) == yaml

    # labels must be YOLO-normalized and agree with the source XML
    pid = train_ids[0]
    ann = anns[pid]
    ds = k.WORK / f"ds_{k.channels_key(seed_candidate())}"
    rows = (ds / "labels" / "train" / f"{pid}.txt").read_text().strip().splitlines()
    assert len(rows) == len(ann.boxes)
    for row, box in zip(rows, ann.boxes):
        cls, cx, cy, bw, bh = row.split()
        assert int(cls) == box.cls_id
        assert abs(float(cx) - (box.x1 + box.x2) / 2 / ann.width) < 1e-6
        assert abs(float(bh) - (box.y2 - box.y1) / ann.height) < 1e-6
        assert 0 <= float(cx) <= 1 and 0 <= float(cy) <= 1

    # Augmentation must add copies to train only, leave val at the real
    # distribution, and keep every label describing what is under it.
    aug = seed_candidate()
    aug["augment"].update(sg_window=5, sg_polyorder=2, smote_alpha=0.3,
                          cutmix_prob=0.4, cutmix_blocks=16, copies=2)
    from dream_rsi.candidate import normalize
    aug = normalize(aug)
    assert aug["augment"]["copies"] == 2, aug["augment"]
    ds_aug = k.WORK / f"ds_{k.channels_key(aug)}"
    assert ds_aug != ds, "augmentation must not reuse the unaugmented render"
    k.materialize(aug, index, train_ids, val_ids, anns, ds_aug)
    n_train = len(list((ds_aug / "images" / "train").glob("*.png")))
    n_val = len(list((ds_aug / "images" / "val").glob("*.png")))
    assert n_train == len(train_ids) * 3, (n_train, len(train_ids))
    assert n_val == len(val_ids), (n_val, len(val_ids))
    for lbl in (ds_aug / "labels" / "train").glob("*_a*.txt"):
        for row in lbl.read_text().split("\n"):
            if not row:
                continue
            cls, cx, cy, bw, bh = row.split()
            assert 0 <= int(cls) <= 17 and all(0 <= float(v) <= 1 for v in (cx, cy, bw, bh)), row

    # a missing dataset must name what /kaggle/input actually holds
    k.DATA = tmp_root / "nope"
    try:
        load_kernel(tmp_root / "work2", tmp_root / "nope").data_root()
    except FileNotFoundError as e:
        assert "/kaggle/input" in str(e)
    else:
        raise AssertionError("data_root() accepted a missing dataset")

    print(f"OK: {len(ids)} ids, {len(train_ids)}/{len(val_ids)} split, "
          f"{len(CHANNEL_MODES)} channel modes (incl. 16-band TIFF), "
          f"labels verified against XML; augmented train x3, val untouched")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_datapath(Path(td))
