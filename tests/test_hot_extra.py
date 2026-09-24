"""HOT2024 external frames: merging the single tracked box with the teacher.

merge_hot_labels  the tracked box keeps its merged class and replaces the
                  teacher's box on the same object; other confident detections
                  become labels, uncertain ones ignore regions; a rider becomes
                  people + e-bike from the teacher, or an ignore region.
blank_regions     fills only the ignore box, never a labelled box's pixels.
hot_extra         end to end with a stub teacher: the frames land in the
                  extra id range with the merged labels, blanked copies are
                  what the index points to, per_video subsamples, empty frames
                  are dropped.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from test_s3t_detr import build_kernel  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    from PIL import Image

    from hod26.cube import load_planar, to_planar
    from tools.s3t_round import plain_finetune_candidate, s3t_candidate

    cand = plain_finetune_candidate(s3t_candidate(total=2, arch="xca"), "final_best.pt", 2)
    check("plain fine-tune: no S1/C1/C2/B2, warm start, everything trains",
          not cand["train"].get("s3t_stem_bands") and not cand["train"].get("box_cls_gain")
          and not cand["train"].get("rep_gain") and not cand["augment"].get("crowd_paste")
          and cand["train"]["init_from"] == "final_best.pt"
          and "stem_in" not in cand["train"]["unfreeze"]
          and all(v[0] == 0 for v in cand["train"]["unfreeze"].values()))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        m = build_kernel(td, cand)
        C = m.CLASSES.index
        P, E, CAR = C("people"), C("e-bike"), C("car")

        # ---- merge
        preds = [(P, 0.9, 10, 10, 30, 60),     # the tracked person, teacher's box
                 (P, 0.8, 100, 10, 120, 60),   # another person: label
                 (P, 0.45, 200, 10, 220, 60),  # unsure: ignore
                 (CAR, 0.2, 300, 10, 360, 40)]  # below lo: nothing
        boxes, ign = m.merge_hot_labels("people", (11, 11, 31, 61), preds)
        check("tracked box kept under its merged class, teacher's duplicate dropped",
              boxes[0] == (P, 11, 11, 31, 61) and not any(b[1:] == (10, 10, 30, 60) for b in boxes))
        check("other confident detection added as a label", (P, 100, 10, 120, 60) in boxes)
        check("0.3-0.6 detection becomes an ignore region, < 0.3 nothing",
              ign == [(200, 10, 220, 60)] and len(boxes) == 2, f"{boxes} {ign}")
        boxes, ign = m.merge_hot_labels("people", None, preds)
        check("target out of view: teacher labels only", len(boxes) == 2 and len(ign) == 1)

        rider = [(P, 0.7, 50, 5, 70, 45), (E, 0.4, 45, 25, 80, 60), (CAR, 0.9, 200, 0, 260, 40)]
        boxes, ign = m.merge_hot_labels("rider", (45, 5, 80, 60), rider)
        check("rider -> the teacher's people and e-bike boxes on it (>= lo), no ignore",
              (P, 50, 5, 70, 45) in boxes and (E, 45, 25, 80, 60) in boxes
              and (CAR, 200, 0, 260, 40) in boxes and not ign, f"{boxes} {ign}")
        boxes, ign = m.merge_hot_labels("rider", (45, 5, 80, 60), [(CAR, 0.9, 200, 0, 260, 40)])
        check("rider with nothing found on it -> the tracked box is ignored",
              ign == [(45, 5, 80, 60)] and boxes == [(CAR, 200, 0, 260, 40)], f"{boxes} {ign}")

        # ---- blank
        cube = np.full((40, 60, 16), 100, np.uint16)
        cube[10:20, 10:20] = 900          # ignore region
        cube[15:25, 15:25] = 500          # labelled box overlapping it
        out = m.blank_regions(cube, [(10, 10, 20, 20)], [(15, 15, 25, 25)])
        check("ignore box filled with its surroundings",
              (out[10:15, 10:20] == 100).all() and (out[15:20, 10:15] == 100).all())
        check("labelled pixels untouched, outside untouched",
              (out[15:25, 15:25] == cube[15:25, 15:25]).all() and (out[30:, :] == 100).all())

        # ---- end to end with a stub teacher
        root = td / "input" / "hod26-hot24"
        (root / "images").mkdir(parents=True)
        frames = []
        rng = np.random.default_rng(0)
        for k in range(6):
            Image.fromarray(to_planar(rng.integers(0, 300, (32, 64, 16)).astype(np.uint16))).save(
                root / "images" / f"{k}.png")
            frames.append({"id": k, "video": "vA" if k < 4 else "vB", "frame": k * 10,
                           "cls": "people" if k < 4 else "car",
                           "box": None if k in (3, 5) else [2, 2, 12, 20]})
        (root / "hot24_index.json").write_text(json.dumps({"frames": frames}))
        m.INPUT = td / "input"
        m.SCRATCH = td / "scratch"
        m.SCRATCH.mkdir()
        m.find_weights = lambda name: Path("teacher.pt")
        m.build_model = lambda name, w: "teacher"
        stub = {0: [(P, 0.45, 30, 2, 40, 20)], 1: [], 2: [], 3: [], 4: [], 5: []}
        m.predict_test = lambda model, cand, d, ids: (
            [(i, *r) for i in ids for r in stub[i]], {i: (64, 32) for i in ids})
        anns, index = m.hot_extra({"extra_data": {"per_video": 3}}, cand)
        keys = sorted(anns)
        check("per_video 3: vA's frames 0/1(or 2)/3 and vB's 4/5 considered; box-less dropped",
              all(k >= m.EXTRA_OFFSET for k in keys)
              and set(k - m.EXTRA_OFFSET for k in keys) <= {0, 1, 2, 4}
              and m.EXTRA_OFFSET + 3 not in anns and m.EXTRA_OFFSET + 5 not in anns, str(keys))
        a0 = anns[m.EXTRA_OFFSET]
        check("merged label in the annotation",
              [(b.cls_id, b.x1, b.y1, b.x2, b.y2) for b in a0.boxes] == [(P, 2, 2, 12, 20)]
              and a0.width == 64 and a0.height == 32)
        p0 = index[m.EXTRA_OFFSET]
        orig = load_planar(root / "images" / "0.png")
        got = load_planar(p0)
        check("frame with an ignore region: index points to the blanked copy",
              p0.parent == m.SCRATCH / "extra_frames" and got.shape == orig.shape
              and (got[2:20, 30:40] == got[2, 30]).all() and (got[:, :28] == orig[:, :28]).all())
        check("frame without one: the original file",
              index[m.EXTRA_OFFSET + 4] == root / "images" / "4.png"
              and anns[m.EXTRA_OFFSET + 4].boxes[0].cls_id == CAR)

        # ---- fully labelled external frames (HOD3K): no teacher
        lab = td / "input" / "hod3k"
        (lab / "images").mkdir(parents=True)
        cube = rng.integers(0, 300, (32, 64, 16)).astype(np.uint16)
        Image.fromarray(to_planar(cube)).save(lab / "images" / "7.png")
        Image.fromarray(to_planar(cube)).save(lab / "images" / "8.png")
        (lab / "hod3k_index.json").write_text(json.dumps({"frames": [
            {"id": 7, "w": 64, "h": 32, "boxes": [["people", 2, 2, 12, 20], ["e-bike", 20, 5, 40, 30]],
             "ignore": [[50, 2, 60, 12]]},
            {"id": 8, "w": 64, "h": 32, "boxes": []}]}))
        m.build_model = lambda name, w: (_ for _ in ()).throw(AssertionError("no teacher for labelled data"))
        anns, index = m.hot_extra({"extra_data": {"index": "hod3k_index.json"}}, cand)
        k7 = m.EXTRA_OFFSET + 7
        check("labelled: boxes taken as given, no teacher, empty frame dropped",
              sorted(anns) == [k7] and [(b.cls_id, b.x1, b.y1, b.x2, b.y2) for b in anns[k7].boxes]
              == [(P, 2, 2, 12, 20), (E, 20, 5, 40, 30)])
        got = load_planar(index[k7])
        check("labelled: ignore region blanked, labelled pixels intact",
              (got[2:12, 50:60] == got[2, 50]).all() and (got[2:20, 2:12] == cube[2:20, 2:12]).all())

        # ---- band gain jitter
        rng2 = np.random.default_rng(1)
        c0 = np.full((8, 8, 16), 1000, np.uint16)
        c1, _ = m.augment_cube(c0, [], {"band_gain": 0.1}, {}, [], rng2)
        ratio = c1[0, 0].astype(float) / 1000
        check("band_gain: one gain per band within +-10%, spatially uniform, dtype kept",
              c1.dtype == np.uint16 and (np.abs(ratio - 1) <= 0.1 + 1e-3).all() and ratio.std() > 0.01
              and (c1 == c1[0, 0]).all())

        # ---- stage-2 list: competition frames only, from the same render
        root = td / "ds"
        for sub_ in ("images/train", "labels/train", "images/val"):
            (root / sub_).mkdir(parents=True)
        names = ["12.png", "12_a0.png", "12_r0.png", f"{m.EXTRA_OFFSET + 3}.png", f"{m.EXTRA_OFFSET + 4}.png"]
        for nme in names:
            (root / "images/train" / nme).write_bytes(b"")
        (root / "data.yaml").write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnc: 18\n")
        y2, n2 = m.comp_only_yaml(root / "data.yaml")
        listed = [Path(q).name for q in (root / "train_comp.txt").read_text().split()]
        check("stage 2 yaml lists the competition's frames, copies and repeats, no external frame",
              n2 == 3 and sorted(listed) == ["12.png", "12_a0.png", "12_r0.png"]
              and f"train: {root / 'train_comp.txt'}" in y2.read_text() and "val: images/val" in y2.read_text())

    print(f"\n{len(fails)} failure(s)" if fails else "\nall HOT extra-data checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
