#!/usr/bin/env python3
"""A CPU kernel asking whether the test frames are spectrally shifted from train.

Held-out mAP has read 0.04-0.06 above the leaderboard for three different
models, and on test (not held-out) the model hesitates between material
classes: 8-9% of egg / egg_plastic / egg_wood / orange / table_tennis
detections have a rival class above half the top score, against 0-1% on
held-out. That is what a spectral / illumination shift would do.

Measured, all on the normalised log level the network sees
(normalise_frame + align_bands):

  frame level   each frame's background spectrum shape (median over pixels of
                level minus its band mean) and brightness spread; a frame-
                grouped logistic regression train-vs-test (adversarial
                validation) -- AUC 0.5 means no shift; its weights say which
                bands move.
  object level  per class, the core spectrum shape of train ground-truth boxes
                against confident test detections (score >= 0.6 in the S3T-X
                submission): mean shift per band, in units of the train
                within-class spread, and the spectral angle of the shift.

Pure numpy + torch, CPU only:

    python3 tools/build_shift_probe.py --out-dir /tmp/shift --slug zetaoxia/hod26-shift-probe
    kaggle kernels push -p /tmp/shift
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from tools.build_edge_probe import function_source, strip  # noqa: E402

HEAD = '''"""Is the test set spectrally shifted from train? CPU only.

Assembled by tools/build_shift_probe.py -- do not edit directly.
"""
import csv, json, os, time
from collections import defaultdict
from pathlib import Path
import numpy as np
'''

BODY = r'''

T0 = time.time()
INPUT = Path(os.environ.get("SHIFT_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("SHIFT_WORK", "/kaggle/working"))
MIN_SCORE = 0.6


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def frame_stats(path, boxes):
    """(frame shape spectrum 16, brightness p10/p50/p90, per-box core shape spectra)."""
    L = normalise_frame(align_bands(load_planar(path).astype(np.float32)))
    S = L - L.mean(-1, keepdims=True)
    Y = L.mean(-1)
    bg = np.ones(Y.shape, bool)
    for (_, x1, y1, x2, y2) in boxes:
        bg[max(0, y1):y2, max(0, x1):x2] = False
    frame = np.median(S[bg], 0) if bg.sum() > 100 else np.median(S.reshape(-1, 16), 0)
    bright = np.percentile(Y, [10, 50, 90])
    cores = []
    for (c, x1, y1, x2, y2) in boxes:
        w, h = x2 - x1, y2 - y1
        if w < 4 or h < 4:
            continue
        mx, my = int(w * 0.15), int(h * 0.15)
        core = S[y1 + my:y2 - my, x1 + mx:x2 - mx].reshape(-1, 16)
        if len(core):
            cores.append((c, core.mean(0), float(Y[y1 + my:y2 - my, x1 + mx:x2 - mx].mean())))
    return frame, bright, cores


def fit_logreg(X, y, lam=1e-2):
    import torch
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    Xs = (X - mu) / sd
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xs @ w + b, y) + lam * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    w, b = w.detach(), b.detach()
    return (lambda Z: torch.sigmoid(((torch.as_tensor(Z, dtype=torch.float32) - mu) / sd) @ w + b).numpy()), (w / sd).numpy()


def auc(p, n):
    x = np.concatenate([p, n])
    r = np.empty(len(x))
    r[x.argsort(kind="mergesort")] = np.arange(1, len(x) + 1)
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def _job(job):
    split, pid, path, boxes = job
    try:
        return split, pid, frame_stats(path, boxes)
    except Exception as exc:                           # noqa: BLE001
        return split, pid, repr(exc)


def main():
    ann_dir = next(p for p in INPUT.rglob("train/annotations") if p.is_dir())
    root = ann_dir.parent.parent
    tr_img, te_img = root / "train" / "images", root / "test" / "images"
    sub = next(iter(sorted(INPUT.rglob("submission.csv"))))
    log(f"data {root}; test detections from {sub}")
    jobs = []
    for p in sorted(ann_dir.glob("*.xml")):
        a = parse(p)
        jobs.append(("train", a.image_id, tr_img / f"{p.stem}.png",
                     [(b.cls_id, b.x1, b.y1, b.x2, b.y2) for b in a.boxes]))
    det = defaultdict(list)
    with open(sub) as fh:
        for r in list(csv.reader(fh))[1:]:
            if float(r[3]) >= MIN_SCORE:
                det[r[1]].append((int(r[2]), int(round(float(r[4]))), int(round(float(r[5]))),
                                  int(round(float(r[6]))), int(round(float(r[7])))))
    for p in sorted(te_img.glob("*.png")):
        jobs.append(("test", p.stem, p, det.get(p.stem, [])))
    log(f"{sum(j[0] == 'train' for j in jobs)} train + {sum(j[0] == 'test' for j in jobs)} test frames")
    from multiprocessing import Pool
    out = []
    with Pool(max(1, os.cpu_count() or 1)) as pool:
        for n, r in enumerate(pool.imap(_job, jobs, chunksize=8)):
            out.append(r)
            if n % 500 == 0:
                log(f"  {n}/{len(jobs)}")
    bad = [r for r in out if isinstance(r[2], str)]
    if bad:
        log(f"{len(bad)} frames failed, first: {bad[0]}")
    out = [r for r in out if not isinstance(r[2], str)]
    report = {}

    # ---- frame level: adversarial validation train vs test
    Xf = np.array([np.concatenate([r[2][0], r[2][1]]) for r in out], np.float32)
    yf = np.array([r[0] == "test" for r in out], np.float32)
    fold = np.arange(len(out)) % 5
    prob = np.zeros(len(out), np.float32)
    for k in range(5):
        clf, _ = fit_logreg(Xf[fold != k], yf[fold != k])
        prob[fold == k] = clf(Xf[fold == k])
    a_all = auc(prob[yf == 1], prob[yf == 0])
    _, wts = fit_logreg(Xf, yf)
    shape_only = Xf[:, :16]
    prob2 = np.zeros(len(out), np.float32)
    for k in range(5):
        clf, _ = fit_logreg(shape_only[fold != k], yf[fold != k])
        prob2[fold == k] = clf(shape_only[fold == k])
    a_shape = auc(prob2[yf == 1], prob2[yf == 0])
    d = Xf[yf == 1].mean(0) - Xf[yf == 0].mean(0)
    sd = Xf[yf == 0].std(0) + 1e-6
    log(f"FRAME adversarial validation train vs test: AUC {a_all:.3f} (background spectrum shape + brightness), "
        f"{a_shape:.3f} (shape only)")
    log("FRAME background shape shift test-train per band, in train SDs: "
        + " ".join(f"{v:+.2f}" for v in d[:16] / sd[:16]))
    log("FRAME brightness p10/p50/p90 shift in train SDs: " + " ".join(f"{v:+.2f}" for v in d[16:] / sd[16:]))
    report["frame"] = {"auc": a_all, "auc_shape": a_shape, "shift_sd": (d / sd).tolist(), "weights": wts.tolist()}

    # ---- object level, per class
    per = defaultdict(lambda: {"train": [], "test": [], "ytr": [], "yte": []})
    for split, pid, (frame, bright, cores) in out:
        for c, spec, y in cores:
            per[CLASSES[c]][split].append(spec)
            per[CLASSES[c]]["ytr" if split == "train" else "yte"].append(y)
    report["objects"] = {}
    log("OBJECT core-spectrum shift, test detections vs train ground truth:")
    for c in CLASSES:
        tr, te = np.array(per[c]["train"]), np.array(per[c]["test"])
        if len(tr) < 20 or len(te) < 10:
            continue
        m_tr, m_te = tr.mean(0), te.mean(0)
        spread = tr.std(0) + 1e-6
        z = (m_te - m_tr) / spread
        ang = np.degrees(np.arccos(np.clip(
            (m_tr @ m_te) / (np.linalg.norm(m_tr) * np.linalg.norm(m_te) + 1e-9), -1, 1)))
        within = np.degrees(np.arccos(np.clip((tr @ m_tr) / (np.linalg.norm(tr, axis=1) * np.linalg.norm(m_tr) + 1e-9), -1, 1)))
        ybr = float(np.mean(per[c]["yte"]) - np.mean(per[c]["ytr"])) / (np.std(per[c]["ytr"]) + 1e-6)
        lab = np.r_[np.zeros(len(tr)), np.ones(len(te))].astype(np.float32)
        X = np.vstack([tr, te]).astype(np.float32)
        f = np.arange(len(X)) % 5
        pr = np.zeros(len(X), np.float32)
        for k in range(5):
            clf, _ = fit_logreg(X[f != k], lab[f != k])
            pr[f == k] = clf(X[f == k])
        a = auc(pr[lab == 1], pr[lab == 0])
        report["objects"][c] = {"n_train": len(tr), "n_test": len(te), "auc_train_vs_test": a,
                                "angle_deg": float(ang), "within_deg_median": float(np.median(within)),
                                "max_band_shift_sd": float(np.abs(z).max()), "band_shift_sd": z.tolist(),
                                "brightness_shift_sd": ybr}
        log(f"    {c:15s} n {len(tr):4d}/{len(te):4d}  train-vs-test AUC {a:.3f}  mean-shape angle "
            f"{ang:5.2f} deg (within-class median {np.median(within):5.2f})  max band shift {np.abs(z).max():.2f} SD  "
            f"brightness {ybr:+.2f} SD")
    (WORK / "shift_probe.json").write_text(json.dumps(report, indent=1))
    log("SHIFT PROBE DONE")


if __name__ == "__main__":
    main()
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--slug", default="zetaoxia/hod26-shift-probe")
    ap.add_argument("--dataset", default="xishengfeng/hod26-planar")
    ap.add_argument("--detections", default="zetaoxia/hod26-s3t-detr",
                    help="kernel whose output holds the submission.csv to locate test objects")
    args = ap.parse_args()
    parts = [
        HEAD,
        strip("src/hod26/voc.py"),
        strip("src/hod26/cube.py"),
        *(function_source("src/hod26/s3t/preprocess.py", f)
          for f in ("band_offsets", "_shift_axis", "align_bands", "normalise_frame")),
        "CELL = 4",
        BODY,
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "shift_probe.py").write_text("\n\n".join(parts))
    (args.out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": args.slug, "title": args.slug.split("/")[-1].replace("-", " ").title(),
        "code_file": "shift_probe.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_internet": False,
        "competition_sources": [], "dataset_sources": [args.dataset],
        "kernel_sources": [args.detections],
    }, indent=2))
    print(f"wrote {args.out_dir / 'shift_probe.py'} (CPU only)")


if __name__ == "__main__":
    main()
