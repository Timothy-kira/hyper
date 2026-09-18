"""Exercise the submission path without a GPU.

The submission is the one deliverable that has to work first time on the day:
it runs once, at the end, against a deadline. This drives run_submission()'s
logic with a stub detector so that everything around the model -- test-set
discovery, staging, prediction bookkeeping, the submission schema and the local
validator -- is known good before any of it is trusted with the real thing.
"""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from hod26.submit import COLUMNS  # noqa: E402
from tests.test_datapath import build_fake_dataset, load_kernel  # noqa: E402


class StubBoxes:
    """Two detections per frame, in the shape ultralytics Results.boxes has."""

    def __init__(self, w: int, h: int):
        self.xyxy = _T(np.array([[4, 5, 24, 30], [w - 30, h - 24, w - 5, h - 3]], np.float32))
        self.cls = _T(np.array([14, 2], np.float32))
        self.conf = _T(np.array([0.9, 0.42], np.float32))

    def __len__(self) -> int:
        return 2


class _T:
    def __init__(self, a): self._a = a
    def cpu(self): return self
    def numpy(self): return self._a


class StubResult:
    def __init__(self, path):
        import cv2
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        h, w = im.shape[:2]
        self.boxes = StubBoxes(w, h)


class StubModel:
    """Stands in for YOLO/RTDETR: records what it was asked to predict."""

    def __init__(self, *a, **kw):
        self.seen: list[str] = []

    def train(self, **kw):
        m = types.SimpleNamespace()
        m.box = types.SimpleNamespace(map=0.33, map50=0.51, maps=np.zeros(18),
                                      ap_class_index=np.array([], int))
        return m

    def predict(self, paths, **kw):
        paths = [paths] if isinstance(paths, str) else list(paths)
        self.seen.extend(paths)
        return [StubResult(p) for p in paths]


def test_submission(tmp_root: Path) -> None:
    from dream_rsi.candidate import seed_candidate

    data = build_fake_dataset(tmp_root / "input" / "hod26-planar", n_train=12, n_test=7)
    k = load_kernel(tmp_root / "work", data)

    stub = StubModel()
    k.build_model = lambda *a, **kw: stub
    # Training is covered elsewhere; here the point is everything around it.
    k.run_candidate = lambda *a, **kw: (
        {"mAP": 0.33, "mAP50": 0.51, "per_class": {}}, [],
        str(tmp_root / "work" / "weights.pt"),
    )

    cand = seed_candidate()
    k.run_submission({"round": "final",
                      "submit": {"candidate": cand, "use_all_train": True}})

    sub = k.WORK / "submission.csv"
    assert sub.exists(), "no submission.csv produced"
    rows = list(csv.reader(sub.open()))
    assert rows[0] == COLUMNS, rows[0]
    body = rows[1:]

    test_ids = sorted(int(p.stem) for p in (data / "test" / "images").glob("*.png"))
    assert len(test_ids) == 7

    # Every test frame must be predicted -- a frame silently skipped is pure
    # lost recall, and the metric has no way to tell us that happened.
    predicted = {int(r[1]) for r in body}
    assert predicted == set(test_ids), (sorted(predicted), test_ids)
    assert len(stub.seen) == len(test_ids), (len(stub.seen), len(test_ids))

    # id must be a dense 0-based counter, since the grader drops it by position.
    assert [int(r[0]) for r in body] == list(range(len(body)))

    sizes = {}
    for pid in test_ids:
        import cv2
        im = cv2.imread(str(data / "test" / "images" / f"{pid}.png"), cv2.IMREAD_UNCHANGED)
        sizes[pid] = (im.shape[1] // 1, im.shape[0] // 16)   # planar: 16 bands stacked

    for r in body:
        image_id, cls_id = int(r[1]), int(r[2])
        conf = float(r[3])
        x1, y1, x2, y2 = (int(r[j]) for j in (4, 5, 6, 7))
        assert 0 <= cls_id <= 17, r
        assert 0.0 <= conf <= 1.0, r
        assert x2 > x1 and y2 > y1, f"degenerate box survived: {r}"
        w, h = sizes[image_id]
        assert 0 <= x1 <= w and 0 <= x2 <= w, (r, w)
        assert 0 <= y1 <= h and 0 <= y2 <= h, (r, h)

    # And the shipped validator must accept what we just wrote.
    sys.argv = ["submit", "--dry-run"]
    from tools.submit import check
    stats = check(sub)
    assert stats["rows"] == len(body) and stats["images"] == len(test_ids), stats

    print(f"OK: {len(body)} rows over {stats['images']}/{len(test_ids)} test frames, "
          f"schema and bounds verified, validator accepts")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_submission(Path(td))
