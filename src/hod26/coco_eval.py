"""Local scorer mirroring the official HOD26 metric.

Organizers confirmed (forum #729747): pycocotools COCOeval in bbox mode,
IoU 0.50:0.05:0.95, 101-point interpolation, macro-averaged over classes
that have ground truth. mAP@0.5 is the same procedure at IoU 0.50.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .voc import CLASSES


def _gt_dict(anns) -> dict:
    images, annotations = [], []
    for a in anns:
        images.append({"id": a.image_id, "width": a.width, "height": a.height})
        for b in a.boxes:
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": a.image_id,
                "category_id": b.cls_id,
                "bbox": [b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1],  # COCO xywh
                "area": float(b.area),
                "iscrowd": 0,
            })
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)],
    }


def evaluate(anns, preds, per_class: bool = False) -> dict:
    """Score predictions against annotations.

    ``preds`` rows are ``(image_id, class_id, confidence, x1, y1, x2, y2)``.
    Returns mAP@[.5:.95] (primary), mAP@0.5 (secondary), and optionally per-class AP.
    """
    gt = COCO()
    gt.dataset = _gt_dict(anns)
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()

    if not len(preds):
        return {"mAP": 0.0, "mAP50": 0.0, **({"per_class": {}} if per_class else {})}

    dets = [{
        "image_id": int(r[0]), "category_id": int(r[1]), "score": float(r[2]),
        "bbox": [float(r[3]), float(r[4]), float(r[5]) - float(r[3]), float(r[6]) - float(r[4])],
    } for r in preds]

    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        e = COCOeval(gt, dt, "bbox")
        e.evaluate(); e.accumulate(); e.summarize()

    out = {"mAP": float(e.stats[0]), "mAP50": float(e.stats[1])}
    if per_class:
        # precision: [TxRxKxAxM]; take all-area, maxDet=100, average over IoU+recall
        prec = e.eval["precision"][:, :, :, 0, -1]
        gt_classes = {a["category_id"] for a in gt.dataset["annotations"]}
        out["per_class"] = {}
        for k, name in enumerate(CLASSES):
            if k not in gt_classes:
                continue
            p = prec[:, :, k]
            p = p[p > -1]
            out["per_class"][name] = float(np.mean(p)) if p.size else 0.0
    return out
