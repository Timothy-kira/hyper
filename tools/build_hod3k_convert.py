#!/usr/bin/env python3
"""A CPU kernel turning HOD3K's raw mosaics into the competition's planar layout.

HOD3K (S2ADet, TGRS 2023) is shot with the same XIMEA 4x4 VIS camera. Its raw
16-band mosaics come in two Kaggle datasets: xishengfeng/hsidataraw (train,
flat) and xishengfeng/hsidata (HSI/val, HSI/test); the YOLO labels are under
hsidata/hsidetection/sa_information/labels/<split>/ (the se_information copy is
identical). Per frame the kernel

  - replaces the handful of sensor-flag pixels above the 10-bit range (a few
    dozen per frame) by the median of their 3x3 same-band neighbours,
  - de-mosaics with X2Cube at phase (0, 0) (the scan found every other phase
    10+ degrees further from the competition's spectra), stores band-planar PNG,
  - maps the classes: 0 and 2 -> people (11937 + 207 = the paper's 12144),
    1 -> car, 3 -> e-bike, and converts YOLO centre/size to cube-pixel corners.

Output: /kaggle/working/hod3k/images/<n>.png and hod3k_index.json, mounted by
the training kernel as a kernel source (train.extra_data index hod3k_index.json).

    python3 tools/build_hod3k_convert.py --out-dir /tmp/conv --slug zetaoxia/hod26-hod3k-convert
    kaggle kernels push -p /tmp/conv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CODE = r'''"""HOD3K raw mosaics -> competition planar layout + hod3k_index.json. CPU only."""
import json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from PIL import Image
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
OUT = Path("/kaggle/working/hod3k")
MAP = {0: "people", 2: "people", 1: "car", 3: "e-bike"}

def x2cube(img, cell=4):
    m, n = img.shape
    t = img.reshape(m // cell, cell, n // cell, cell)
    return np.ascontiguousarray(t.transpose(0, 2, 1, 3)).reshape(m // cell, n // cell, cell * cell)

def to_planar(cube):
    h, w, b = cube.shape
    return np.ascontiguousarray(cube.transpose(2, 0, 1).reshape(b * h, w))

def fix_hot(a):
    bad = np.argwhere(a > 4095)
    if len(bad) == 0:
        return a
    a = a.copy()
    H, W = a.shape
    for y, x in bad:
        ys = [y + d for d in (-4, 0, 4) if 0 <= y + d < H]
        xs = [x + d for d in (-4, 0, 4) if 0 <= x + d < W]
        nb = a[np.ix_(ys, xs)].ravel()
        nb = nb[nb <= 4095]
        a[y, x] = int(np.median(nb)) if len(nb) else 0
    return a

def job(args):
    n, split, stem, src, lab = args
    try:
        a = np.array(Image.open(src))
        if a.ndim != 2 or a.dtype != np.uint16 or a.shape[0] % 4 or a.shape[1] % 4:
            return None, f"{split}/{stem}: shape {a.shape} {a.dtype}"
        nh = int((a > 4095).sum())
        if nh > 0.001 * a.size:
            return None, f"{split}/{stem}: {nh} pixels above 10 bit"
        cube = x2cube(fix_hot(a))
        H, W = cube.shape[:2]
        Image.fromarray(to_planar(cube)).save(OUT / "images" / f"{n}.png")
        boxes = []
        for line in lab.read_text().split("\n"):
            v = line.split()
            if len(v) < 5:
                continue
            c, cx, cy, w, h = int(v[0]), *map(float, v[1:5])
            if c not in MAP:
                continue
            boxes.append([MAP[c], (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        return {"id": n, "split": split, "stem": stem, "w": W, "h": H, "hot": nh, "boxes": boxes}, None
    except Exception as e:                                   # noqa: BLE001
        return None, f"{split}/{stem}: {e!r}"

def main():
    inp = Path("/kaggle/input")
    H = next(p for p in inp.rglob("HSI") if p.is_dir())
    L = next(p for p in inp.rglob("sa_information") if p.is_dir()) / "labels"
    tr = next(p.parent for p in sorted(inp.rglob("0001.png")) if "HSI" not in str(p) and "hsidetection" not in str(p))
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    jobs, n, missing = [], 0, []
    for split, d in (("train", tr), ("val", H / "val"), ("test", H / "test")):
        for f in sorted(d.glob("*.png")):
            lab = L / split / f"{f.stem}.txt"
            if not lab.exists():
                missing.append(f"{split}/{f.stem}")
                continue
            jobs.append((n, split, f.stem, f, lab)); n += 1
    log(f"{len(jobs)} frames with labels; {len(missing)} raw frames without a label file: {missing[:10]}")
    frames, errs = [], []
    with Pool(4) as pool:
        for k, (fr, err) in enumerate(pool.imap(job, jobs, chunksize=8)):
            (frames.append(fr) if fr else errs.append(err))
            if k % 500 == 0:
                log(f"  {k}/{len(jobs)}")
    frames.sort(key=lambda f: f["id"])
    (OUT / "hod3k_index.json").write_text(json.dumps({"frames": frames}))
    from collections import Counter
    cnt = Counter(b[0] for f in frames for b in f["boxes"])
    log(f"wrote {len(frames)} frames ({Counter(f['split'] for f in frames)}), boxes {dict(cnt)}, "
        f"frames with flag pixels {sum(f['hot'] > 0 for f in frames)}; errors {len(errs)}: {errs[:10]}")
    log("HOD3K CONVERT DONE")

main()
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--slug", default="zetaoxia/hod26-hod3k-convert")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "hod3k_convert.py").write_text(CODE)
    (args.out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": args.slug, "title": args.slug.split("/")[-1].replace("-", " ").title(),
        "code_file": "hod3k_convert.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_internet": False,
        "competition_sources": [], "dataset_sources": ["xishengfeng/hsidata", "xishengfeng/hsidataraw"],
        "kernel_sources": [],
    }, indent=2))
    print(f"wrote {args.out_dir / 'hod3k_convert.py'} (CPU only)")


if __name__ == "__main__":
    main()
