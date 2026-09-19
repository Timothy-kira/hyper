#!/usr/bin/env python3
"""Phase A: the two questions a proxy can still answer, at full resolution.

Two kernels, pushed together because Kaggle allows two concurrent GPU sessions:

  arch      band_stack + the two-stage SRF adapter, rtdetr-l against yolo26m
  frontend  rtdetr-l, pseudo_rgb (the 0.429 reference) against srf3, the same
            averaging rendered offline into three channels

The adapter arm sits in the first kernel rather than the second, so the
front-end comparison is rtdetr-srf8 against rtdetr-pseudorgb across the two --
they run at identical fidelity, on the same split, scored by the same pass.
Augmentation is off in both: this phase is about the architecture and the front
end, and leaving it out keeps the rendering cheap.

Everything except the question under test is held at the full run's settings --
imgsz above all, because resolution is the one design choice a 640 proxy would
rank for a run that trains at 1024.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dream_rsi.candidate import seed_candidate  # noqa: E402
from dream_rsi.executor import KaggleRoundExecutor  # noqa: E402
from tools.final_runs import DESIGN  # noqa: E402


def arm(**over) -> dict:
    cfg = dict(DESIGN)
    cfg["fidelity"] = "proxy"
    cfg.update(over)
    return seed_candidate(**cfg)


KERNELS = {
    "arch": [
        ("rtdetr-srf8", arm(**{"train.model": "rtdetr-l"})),
        ("yolo26m-srf8", arm(**{"train.model": "yolo26m"})),
    ],
    # The first push lost this arm to the multi_scale OOM, and it is the one the
    # track decision turns on, so it goes again on its own at the same batch as
    # the yolo26m arm it is being compared with.
    "arch2": [
        ("rtdetr-srf8", arm(**{"train.model": "rtdetr-l"})),
    ],
    "frontend": [
        ("rtdetr-pseudorgb", arm(**{"train.model": "rtdetr-l",
                                    "channels.mode": "pseudo_rgb"})),
        ("rtdetr-srf3", arm(**{"train.model": "rtdetr-l",
                               "channels.mode": "srf3"})),
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--images", type=int, default=300)
    ap.add_argument("--kernels", nargs="+", default=list(KERNELS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pushed = {}
    for name in args.kernels:
        cands = KERNELS[name]
        cfg = {
            "round": f"phaseA-{name}",
            "proxy_train_images": args.images,
            "proxy_val_images": 120,
            "candidates": [{"node_id": nid,
                            "candidate": {**c, "train": {**c["train"],
                                                         "epochs": args.epochs}}}
                           for nid, c in cands],
        }
        for e in cfg["candidates"]:
            t = e["candidate"]["train"]
            print(f"  {e['node_id']:18s} {t['model']:10s} "
                  f"{e['candidate']['channels']['mode']:10s} srf_k={t['srf_k']} "
                  f"imgsz={t['imgsz']} batch={t['batch']} ep={t['epochs']}")
        if args.dry_run:
            (REPO / "runs" / f"phaseA_{name}.json").write_text(json.dumps(cfg, indent=2))
            continue
        ex = KaggleRoundExecutor(f"xishengfeng/hod26-phasea-{name}", timeout_hours=6.0,
                                 out_dir=REPO / "runs" / f"phaseA_{name}")
        ex.push(cfg)
        print(f"pushed {ex.slug}")
        pushed[name] = ex

    for name, ex in pushed.items():
        print(f"{name}: {ex.wait()}")
        try:
            payload = ex.fetch()
        except Exception as e:                       # noqa: BLE001
            print(f"  fetch failed: {e}")
            continue
        for r in payload.get("results", []):
            d = r.get("diagnostics") or {}
            drift = (d.get("adapter") or {}).get("rel")
            print(f"  {r['node_id']:18s} mAP={r.get('score')} "
                  f"mAP50={d.get('mAP50')} drift={drift} "
                  f"{r.get('cost_seconds', 0) / 60:.1f} min"
                  + (f"\n    ERROR {r['error'][-400:]}" if r.get("error") else ""))


if __name__ == "__main__":
    main()
