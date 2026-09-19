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


def _mean_precision(e, iou_idx: int | None = None) -> float:
    """Mean precision over IoU thresholds, recall points and classes with GT.

    ``precision`` is [IoU x recall x class x area x maxDets]; area 0 is "all"
    and the last maxDets slot is the cap the caller asked for. -1 marks classes
    with no ground truth, which the macro average must skip rather than score.
    """
    p = e.eval["precision"][:, :, :, 0, -1]
    if iou_idx is not None:
        p = p[iou_idx:iou_idx + 1]
    p = p[p > -1]
    return float(p.mean()) if p.size else 0.0


def evaluate(anns, preds, per_class: bool = False, max_dets: int = 100) -> dict:
    """Score predictions against annotations.

    ``preds`` rows are ``(image_id, class_id, confidence, x1, y1, x2, y2)``.
    Returns mAP@[.5:.95] (primary), mAP@0.5 (secondary), and optionally per-class AP.

    ``max_dets`` is pycocotools' cap on detections counted per image, and the
    default of 100 is the one COCOeval applies to the AP it reports. It matters
    here because RT-DETR emits 300 boxes per frame and ultralytics scores all of
    them: the same predictions read at 100 and at 300 are not the same number,
    which is most of why our training-time score sat above the leaderboard.
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
        e.params.maxDets = [1, 10, int(max_dets)]
        e.evaluate(); e.accumulate(); e.summarize()

    # Read the averages off the precision array rather than e.stats. COCOeval's
    # summarize() hardcodes maxDets=100 in the signature of the helper that
    # produces stats[0], so with any other cap it finds no matching slot and
    # reports -1. The array itself is indexed by the cap we asked for.
    out = {"mAP": _mean_precision(e), "mAP50": _mean_precision(e, iou_idx=0)}
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
