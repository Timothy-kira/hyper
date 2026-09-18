#!/usr/bin/env python3
"""Convert the downloaded competition archive into the band-planar dataset.

Reads members straight out of the zip so the 15.65 GB archive is never expanded
on disk -- only the ~5.5 GB planar output is written. Each worker opens its own
ZipFile handle, since one handle cannot be shared across processes.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from hod26.cube import to_planar, x2cube  # noqa: E402

COMPRESS_LEVEL = 1
TRAIN_IMG = "data_train/data_train/VIS/"
TRAIN_ANN = "data_train/data_train/Annotations/VIS/"
TEST_IMG = "data_test/data_test/VIS/"

_zip: zipfile.ZipFile | None = None
_zip_path: str | None = None


def _handle(path: str) -> zipfile.ZipFile:
    global _zip, _zip_path
    if _zip is None or _zip_path != path:
        _zip, _zip_path = zipfile.ZipFile(path), path
    return _zip


def convert(args) -> tuple[str, int, int, int]:
    """Decode one mosaic member and write it back out band-planar."""
    zip_path, member, dst = args
    dst = Path(dst)
    if dst.exists():
        return member, dst.stat().st_size, 0, 0
    raw = _handle(zip_path).read(member)
    cube = x2cube(np.array(Image.open(io.BytesIO(raw))))
    buf = io.BytesIO()
    Image.fromarray(to_planar(cube)).save(buf, format="PNG", compress_level=COMPRESS_LEVEL)
    dst.write_bytes(buf.getvalue())
    return member, buf.tell(), cube.shape[0], cube.shape[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "hod26_planar")
    ap.add_argument("--slug", default="xishengfeng/hod26-planar")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip)
    names = zf.namelist()
    train = sorted(n for n in names if n.startswith(TRAIN_IMG) and n.endswith(".png"))
    test = sorted(n for n in names if n.startswith(TEST_IMG) and n.endswith(".png"))
    anns = sorted(n for n in names if n.startswith(TRAIN_ANN) and n.endswith(".xml"))
    print(f"archive holds {len(train)} train / {len(test)} test frames, {len(anns)} annotations")

    for split in ("train", "test"):
        (args.out / split / "images").mkdir(parents=True, exist_ok=True)
    ann_dir = args.out / "train" / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    for n in anns:                       # tiny; no point parallelizing
        dst = ann_dir / Path(n).name
        if not dst.exists():
            dst.write_bytes(zf.read(n))
    if "class.txt" in names:
        (args.out / "class.txt").write_bytes(zf.read("class.txt"))
    print(f"annotations: {len(list(ann_dir.glob('*.xml')))}")

    shapes: dict[str, dict] = {"train": {}, "test": {}}
    for split, members in (("train", train), ("test", test)):
        jobs = [(str(args.zip), m, str(args.out / split / "images" / Path(m).name))
                for m in members]
        done, written, t0 = 0, 0, time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for fut in as_completed([pool.submit(convert, j) for j in jobs]):
                member, nbytes, h, w = fut.result()
                if h:
                    shapes[split][Path(member).stem] = [h, w]
                done += 1
                written += nbytes
                if done % 250 == 0 or done == len(jobs):
                    el = time.time() - t0
                    print(f"  {split}: {done}/{len(jobs)}  {written/1e9:.2f} GB  "
                          f"{done/el:.1f} img/s  eta {(len(jobs)-done)/max(done/el,1e-9)/60:.1f} min",
                          flush=True)

    (args.out / "shapes.json").write_text(json.dumps(shapes))
    (args.out / "dataset-metadata.json").write_text(json.dumps({
        "title": "HOD26 Planar Cubes",
        "id": args.slug,
        "licenses": [{"name": "other"}],
    }, indent=2))
    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"\ndone: {size/1e9:.2f} GB in {args.out}")


if __name__ == "__main__":
    main()
