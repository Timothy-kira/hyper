#!/usr/bin/env python3
"""Mirror the competition data into a private Kaggle Dataset the kernels can attach.

Kaggle silently drops ``competition_sources`` on push for this competition --
a kernel comes up with an empty /kaggle/input -- while ``dataset_sources``
attaches normally. So the data has to arrive as a dataset.

Frames are stored band-planar rather than as the shipped 4x4 mosaic: stacking
the 16 decoded bands vertically puts spatially adjacent pixels next to each
other, which PNG compresses to ~35% of the mosaic's size, losslessly. That
shrinks the upload threefold and removes the de-mosaic step from every kernel
run. Reshaping (16H, W) back to (H, W, 16) recovers the exact original cube.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from hod26.cube import N_BANDS, x2cube  # noqa: E402

COMPETITION = "hyperspectral-object-detection-challenge-2026"
BASE = f"https://www.kaggle.com/api/v1/competitions/data/download/{COMPETITION}/"
COMPRESS_LEVEL = 1   # 35% of mosaic size at 0.10s; level 9 buys 4pp for 34x the time

_print_lock = threading.Lock()


def token() -> str:
    t = os.environ.get("KAGGLE_API_TOKEN")
    if t:
        return t
    p = Path.home() / ".kaggle" / "access_token"
    if p.exists():
        return p.read_text().strip()
    raise SystemExit("no KAGGLE_API_TOKEN in the environment and no ~/.kaggle/access_token")


def fetch(path: str, tok: str, retries: int = 5) -> bytes:
    """GET one competition file, backing off on throttling."""
    req = urllib.request.Request(
        BASE + path.replace("/", "%2F"), headers={"Authorization": f"Bearer {tok}"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError(f"unreachable: {path}")


def to_planar(raw: bytes) -> tuple[bytes, tuple[int, int]]:
    """Decode a mosaic PNG and re-encode it band-planar. Returns (png, (H, W))."""
    cube = x2cube(np.array(Image.open(io.BytesIO(raw))))
    h, w, b = cube.shape
    planar = np.ascontiguousarray(cube.transpose(2, 0, 1).reshape(b * h, w))
    buf = io.BytesIO()
    Image.fromarray(planar).save(buf, format="PNG", compress_level=COMPRESS_LEVEL)
    return buf.getvalue(), (h, w)


def one_image(pid: int, remote: str, dst: Path, tok: str) -> tuple[int, tuple[int, int], int]:
    if dst.exists():                       # resumable: re-running skips finished work
        h16, w = Image.open(dst).size[1], Image.open(dst).size[0]
        return pid, (h16 // N_BANDS, w), dst.stat().st_size
    png, hw = to_planar(fetch(remote, tok))
    dst.write_bytes(png)
    return pid, hw, len(png)


def mirror(split: str, remote_dir: str, ids: list[int], out: Path, tok: str,
           workers: int) -> dict:
    img_dir = out / split / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    shapes, done, total_bytes, t0 = {}, 0, 0, time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(one_image, pid, f"{remote_dir}/{pid}.png",
                        img_dir / f"{pid}.png", tok): pid
            for pid in ids
        }
        for fut in as_completed(futures):
            pid, hw, nbytes = fut.result()
            shapes[str(pid)] = hw
            done += 1
            total_bytes += nbytes
            if done % 100 == 0 or done == len(ids):
                el = time.time() - t0
                with _print_lock:
                    print(f"  {split}: {done}/{len(ids)}  {total_bytes/1e9:.2f} GB written  "
                          f"{done/el:.1f} img/s  eta {(len(ids)-done)/max(done/el,1e-6)/60:.0f} min",
                          flush=True)
    return shapes


def mirror_annotations(ids: list[int], out: Path, tok: str, workers: int) -> None:
    ann = out / "train" / "annotations"
    ann.mkdir(parents=True, exist_ok=True)

    def grab(pid: int) -> None:
        dst = ann / f"{pid}.xml"
        if not dst.exists():
            dst.write_bytes(fetch(f"data_train/data_train/Annotations/VIS/{pid}.xml", tok))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(as_completed([pool.submit(grab, p) for p in ids]), 1):
            if i % 500 == 0 or i == len(ids):
                print(f"  annotations: {i}/{len(ids)}", flush=True)


def competition_ids(tok: str) -> tuple[list[int], list[int]]:
    """List train and test ids straight from the competition file index."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    train, test, page = set(), set(), None
    while True:
        r = api.competition_list_files(COMPETITION, page_size=200, page_token=page)
        for f in (r.files or []):
            if f.name.startswith("data_train/data_train/Annotations/VIS/"):
                train.add(int(Path(f.name).stem))
            elif f.name.startswith("data_test/data_test/VIS/"):
                test.add(int(Path(f.name).stem))
        page = r.next_page_token
        if not page:
            break
    return sorted(train), sorted(test)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "hod26_planar")
    ap.add_argument("--slug", default="xishengfeng/hod26-planar")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N ids per split")
    args = ap.parse_args()

    tok = token()
    args.out.mkdir(parents=True, exist_ok=True)

    print("listing competition files ...", flush=True)
    train_ids, test_ids = competition_ids(tok)
    if args.limit:
        train_ids, test_ids = train_ids[:args.limit], test_ids[:args.limit]
    print(f"  {len(train_ids)} train / {len(test_ids)} test", flush=True)

    (args.out / "class.txt").write_bytes(fetch("class.txt", tok))
    print("mirroring annotations ...", flush=True)
    mirror_annotations(train_ids, args.out, tok, args.workers)

    print("mirroring train frames ...", flush=True)
    shapes = {"train": mirror("train", "data_train/data_train/VIS", train_ids,
                              args.out, tok, args.workers)}
    print("mirroring test frames ...", flush=True)
    shapes["test"] = mirror("test", "data_test/data_test/VIS", test_ids,
                            args.out, tok, args.workers)

    (args.out / "shapes.json").write_text(json.dumps(shapes))
    (args.out / "dataset-metadata.json").write_text(json.dumps({
        "title": "HOD26 Planar Cubes",
        "id": args.slug,
        "licenses": [{"name": "other"}],
    }, indent=2))

    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"\ndone: {size/1e9:.2f} GB in {args.out}")
    print(f"upload with:\n  kaggle datasets create -p {args.out} --dir-mode zip")


if __name__ == "__main__":
    main()
