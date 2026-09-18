"""Submission writer for HOD26.

Schema is the one the organizers declared authoritative (forum #729747):
``id,image_id,class_id,confidence,x1,y1,x2,y2`` with ``id`` a unique
0-based row counter. The bundled sample_submission.csv omits ``id`` and
the organizers said to disregard it.
"""

from __future__ import annotations

import csv
from pathlib import Path

COLUMNS = ["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]


def write(path, preds, clip_to: dict[int, tuple[int, int]] | None = None) -> int:
    """Write predictions, dropping degenerate boxes. Returns rows written.

    ``preds`` rows are ``(image_id, class_id, confidence, x1, y1, x2, y2)``.
    ``clip_to`` optionally maps image_id -> (width, height) for clamping.
    Coordinates are emitted as 0-indexed integer pixels, zero-area rows removed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for image_id, cls_id, conf, x1, y1, x2, y2 in preds:
            x1, y1, x2, y2 = (int(round(float(v))) for v in (x1, y1, x2, y2))
            if clip_to and int(image_id) in clip_to:
                W, H = clip_to[int(image_id)]
                x1, x2 = max(0, min(W, x1)), max(0, min(W, x2))
                y1, y2 = max(0, min(H, y1)), max(0, min(H, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            w.writerow([n, int(image_id), int(cls_id), f"{float(conf):.6f}", x1, y1, x2, y2])
            n += 1
    return n
