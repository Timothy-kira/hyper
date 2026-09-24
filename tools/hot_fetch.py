#!/usr/bin/env python3
"""Fetch HOT2024 VIS videos (same XIMEA 4x4 VIS camera) into the planar layout.

HOT is single-object tracking: every video tracks one target and its name says
what that target is. HOT_CLASS_MAP merges the video names onto our 18 classes;
names mapping to "rider" (person on a two-wheeler) are split into people +
e-bike by the teacher later, "check" names are only previewed, and names not
in the map are never downloaded.

Per video: download the zip from the public Google Drive folder, keep up to
--per-video frames evenly spaced, de-mosaic with the organisers' X2Cube, store
band-planar PNGs (hod26.cube.to_planar) and one VOC XML holding the tracked box.

    python3 tools/hot_fetch.py --out /path/hot24 --probe          # which are downloadable
    python3 tools/hot_fetch.py --out /path/hot24 --per-video 25   # fetch + convert
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

TRAIN_VIS = "18NStLV4MNznADFKGV9OY1bR7WzreZZWu"   # HOT2024 training/HSI-VIS
VAL_VIS = "1pIqIkUQS5IDwp2FHjfaonQTKDMc-hiS3"     # HOT2024 validation/HSI-VIS
DL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"


def _rng(prefix, lo, hi):
    return [f"{prefix}{i}" for i in range(lo, hi + 1)]


HOT_CLASS_MAP = {
    **{n: "people" for n in [
        "people", *_rng("people", 2, 4), "pedestrian", "pedestrain", *_rng("pedestrian", 2, 5),
        *_rng("high_person", 1, 4), "L_person", *_rng("L_person", 7, 9), "L_runner",
        "L_basketball_person", "S_person2", "S_runner1", "S_runner2", "S_walker", "S_warmup",
        "S_jump2", "student", "worker", "player", "surround_person"]},
    **{n: "car" for n in [
        "car", *_rng("car", 1, 12), "automobile", *_rng("automobile", 2, 14), *_rng("high_car", 1, 3),
        "L_car2", "L_car3", "surround_car", "taxi"]},
    **{n: "rider" for n in [*_rng("rider", 1, 6), *_rng("high_rider", 1, 4)]},
    **{n: "rubik" for n in ["rubik", *_rng("rubik", 2, 7)]},
    **{n: "table_tennis" for n in ["pingpong2", "pingpong4", "snow_table_tennis3", "snow_table_tennis4"]},
    "oranges1": "orange", "oranges5": "orange", "apple": "apple", "badminton": "badminton",
    "mirror_egg": "egg?",
    **{n: "check" for n in ["toy", "toy1", "toy2", "toy3", "fruit", "ball",
                            "ball&mirror7", "ball&mirror9", "ball&mirror10"]},
}


def list_folder(fid):
    html = urllib.request.urlopen(f"https://drive.google.com/embeddedfolderview?id={fid}", timeout=60).read().decode()
    ids = re.findall(r'/file/d/([\w-]+)', html)
    names = [n.replace("&amp;", "&").removesuffix(".zip") for n in re.findall(r'flip-entry-title">([^<]+)<', html)]
    assert len(ids) == len(names), (len(ids), len(names))
    return dict(zip(names, ids))


def probe(fid):
    req = urllib.request.Request(DL.format(fid), headers={"Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            if "text/html" in r.headers.get("Content-Type", ""):
                return None
            m = re.search(r"/(\d+)$", r.headers.get("Content-Range", ""))
            return int(m.group(1)) if m else -1
    except Exception:                                   # noqa: BLE001
        return None


def fetch(fid, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(DL.format(fid), timeout=600) as r:
                if "text/html" in r.headers.get("Content-Type", ""):
                    return None
                return r.read()
        except Exception:                               # noqa: BLE001
            time.sleep(2 ** (k + 1))
    return None


def voc(stem, w, h, objs):
    body = "".join(
        f"<object><name>{c}</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>"
        f"<bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"
        for c, x1, y1, x2, y2 in objs)
    return (f"<annotation><folder>hot24</folder><filename>{stem}.png</filename>"
            f"<size><width>{w}</width><height>{h}</height><depth>16</depth></size>{body}</annotation>")


def convert(name, cls, blob, out, per_video):
    import numpy as np
    from PIL import Image

    from hod26.cube import to_planar, x2cube
    z = zipfile.ZipFile(io.BytesIO(blob))
    pngs = sorted(n for n in z.namelist() if n.lower().endswith(".png"))
    gt_name = next((n for n in z.namelist() if n.endswith("groundtruth_rect.txt")), None)
    rects = []
    if gt_name:
        for line in z.read(gt_name).decode().splitlines():
            v = [float(t) for t in re.split(r"[\s,]+", line.strip()) if t]
            rects.append(v[:4] if len(v) >= 4 else None)
    idx = np.unique(np.linspace(0, len(pngs) - 1, min(per_video, len(pngs))).round().astype(int))
    safe = re.sub(r"[^\w]", "_", name)
    rows = []
    for i in idx:
        raw = np.array(Image.open(io.BytesIO(z.read(pngs[i]))))
        h4, w4 = (raw.shape[0] // 4) * 4, (raw.shape[1] // 4) * 4
        cube = x2cube(raw[:h4, :w4])
        H, W, _ = cube.shape
        stem = f"hot_{safe}_{i:04d}"
        Image.fromarray(to_planar(cube)).save(out / "images" / f"{stem}.png")
        r = rects[i] if i < len(rects) else None
        objs = []
        if r is not None and r[2] > 1 and r[3] > 1:
            x1, y1 = max(0, int(round(r[0]))), max(0, int(round(r[1])))
            x2, y2 = min(W, int(round(r[0] + r[2]))), min(H, int(round(r[1] + r[3])))
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                objs.append((cls, x1, y1, x2, y2))
        (out / "annotations" / f"{stem}.xml").write_text(voc(stem, W, H, objs))
        rows.append({"stem": stem, "frame": int(i), "has_box": bool(objs)})
    return {"video": name, "hot_class": cls, "n_frames": len(pngs), "n_gt": len(rects),
            "raw_shape": list(raw.shape), "raw_max": int(raw.max()), "cube": [H, W], "kept": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-video", type=int, default=25)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--shard", default="0/1", help="i/n: this process fetches every n-th mapped video")
    args = ap.parse_args()
    out = args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    # Many training zips are owner-restricted; the validation folder carries
    # downloadable copies of several of them, so fall back to it by name.
    train, val = list_folder(TRAIN_VIS), list_folder(VAL_VIS)
    listing = {n: [fid for fid in (train.get(n), val.get(n)) if fid] for n in {*train, *val}}
    todo = {n: c for n, c in HOT_CLASS_MAP.items() if n in listing and (args.only is None or n in args.only)}
    missing = sorted(set(HOT_CLASS_MAP) - set(listing))
    print(f"{len(train)} + {len(val)} videos listed; {len(todo)} mapped; not in listing: {missing}", flush=True)
    if args.probe:
        res = {n: next((s for s in map(probe, listing[n]) if s), None) for n in todo}
        (out / "probe.json").write_text(json.dumps(res, indent=1))
        ok = {n: s for n, s in res.items() if s}
        print(f"downloadable {len(ok)}/{len(res)}, {sum(ok.values()) / 1e9:.1f} GB")
        for c in sorted(set(todo.values())):
            names = [n for n in todo if todo[n] == c]
            print(f"  {c:13s} ok {sum(bool(res[n]) for n in names)}/{len(names)}  blocked: "
                  + " ".join(n for n in names if not res[n]))
        return
    si, sn = map(int, args.shard.split("/"))
    todo = {n: c for k, (n, c) in enumerate(sorted(todo.items())) if k % sn == si}
    man_path = out / f"manifest_{si}of{sn}.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    done = set()
    for other in out.glob("manifest*.json"):
        done |= set(json.loads(other.read_text()))
    for n, c in todo.items():
        if n in manifest or n in done:
            continue
        t = time.time()
        blob = next((b for b in (fetch(fid) for fid in listing[n]) if b is not None), None)
        if blob is None:
            manifest[n] = {"video": n, "hot_class": c, "error": "not downloadable"}
            print(f"  {n}: not downloadable", flush=True)
        else:
            manifest[n] = convert(n, c, blob, out, args.per_video)
            print(f"  {n} -> {c}: {manifest[n]['n_frames']} frames, kept {len(manifest[n]['kept'])}, "
                  f"{len(blob) / 1e6:.0f} MB, {time.time() - t:.0f}s", flush=True)
        man_path.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
