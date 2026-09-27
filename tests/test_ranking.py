"""Phase 2: the ranking set's raw mosaics are staged to planar and predicted into the
same submission.csv as the test set.

find_ranking_dir finds data_ranking under the input tree; stage_ranking writes each
mosaic as the planar PNG load_planar reads back to exactly load_cube's cube;
predict_test_set(ranking=True) writes test + ranking rows in one CSV and refuses
overlapping ids.
"""

from __future__ import annotations

import csv
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

    from hod26.cube import load_cube, load_planar, to_planar
    from tools.s3t_round import s3t_candidate

    cand = s3t_candidate(total=2, arch="xca")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        m = build_kernel(td, cand)
        inp = td / "input"
        rdir = inp / "competitions" / "hod" / "data_ranking" / "data_ranking" / "VIS"
        rdir.mkdir(parents=True)
        rng = np.random.default_rng(0)
        for i in (5001, 5002, 5003):
            Image.fromarray(rng.integers(0, 1023, (64, 96)).astype(np.uint16)).save(rdir / f"{i}.png")
        root = inp / "planar"
        (root / "test" / "images").mkdir(parents=True)
        for i in (1, 2):
            Image.fromarray(to_planar(rng.integers(0, 300, (16, 24, 16)).astype(np.uint16))).save(
                root / "test" / "images" / f"{i}.png")
        m.INPUT, m.SCRATCH, m.WORK = inp, td / "scratch", td / "work"
        m.SCRATCH.mkdir(); m.WORK.mkdir()

        check("find_ranking_dir finds data_ranking", m.find_ranking_dir() == rdir)
        out, ids = m.stage_ranking(rdir)
        check("stage_ranking: every mosaic, ids from the file names", ids == [5001, 5002, 5003])
        check("staged planar reads back to exactly load_cube's cube",
              all(np.array_equal(load_planar(out / f"{i}.png"), load_cube(rdir / f"{i}.png")) for i in ids))

        def fake_predict(model, cand, d, pids):
            rows = [(p, 3, 0.9, 1.0, 1.0, 5.0, 5.0) for p in pids]
            sizes = {p: load_planar(Path(d) / f"{p}.png").shape[1::-1] for p in pids}
            return rows, sizes
        m.predict_test = fake_predict
        res = m.predict_test_set(None, cand, root, ranking=True)
        rows = list(csv.DictReader(open(m.WORK / "submission.csv")))
        got = sorted({int(r["image_id"]) for r in rows})
        check("one CSV: test (1, 2) and ranking (5001-5003) rows", got == [1, 2, 5001, 5002, 5003]
              and res["images"] == 5, str(got))
        check("header unchanged", list(rows[0].keys()) == ["id", "image_id", "class_id", "confidence",
                                                           "x1", "y1", "x2", "y2"])
        Image.fromarray(rng.integers(0, 1023, (64, 96)).astype(np.uint16)).save(rdir / "2.png")
        try:
            m.predict_test_set(None, cand, root, ranking=True)
            ok = False
        except RuntimeError as e:
            ok = "overlap" in str(e)
        check("overlapping ids refused", ok)

    print(f"\n{len(fails)} failure(s)" if fails else "\nall ranking checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
