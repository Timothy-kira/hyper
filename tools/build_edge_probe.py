#!/usr/bin/env python3
"""A CPU kernel asking where the weak classes' box edges can be read from.

stone_block / people / e-bike / car have almost no spectral edge (object vs
ring spectral angle across the box edge 0.76-2.4 deg, brightness step 2-8%;
the other classes 1.3-9.3 deg, up to 50%), and they are the only classes that
occlude each other. The score is lost at IoU 0.75-0.95, so the question is
which signal, if any, can place their box edges to within 1-2 px:

  F0 spectral16  the 16 normalised log bands at the pixel
  F1 brightness  their mean
  F2 texture     gradient magnitude and local variance of the brightness
  F3 context     brightness minus its mean over annuli 7/15 .. 95/127 px
  F4 shape       a 13 x 13 brightness patch (sampled every 2 px)
  F5 shape+spec  F4 + the same patch of the 3 LDA channels + F0
  F6 all         F5 + F2 + F3

For every instance, 3 positions along each of its 4 edges, a 16-pixel profile
across the edge (8 inside, 8 outside). Per class and feature group, a
logistic regression trained inside vs outside (frame-grouped 5-fold, outside
pixels covered by another box excluded) reports:
  AUC            inside vs outside, 2 px either side of the edge
  edge error     the step that best splits the profile's probabilities,
                 against the true edge: share within 0, 1, 2 px
split by side, and by whether the edge is free or shared with another box.

Pure numpy + torch (no sklearn), CPU only:

    python3 tools/build_edge_probe.py --out-dir /tmp/edge --slug zetaoxia/hod26-edge-probe
    kaggle kernels push -p /tmp/edge
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
DROP = re.compile(r"^\s*(from\s+\.|from\s+__future__\s+import|from hod26|"
                  r"sys\.path\.insert|import sys$)")


def strip(path: str) -> str:
    return "\n".join(ln for ln in (REPO / path).read_text().splitlines()
                     if not DROP.match(ln))


def function_source(path: str, name: str) -> str:
    src = (REPO / path).read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise KeyError(f"{name} not found in {path}")


HEAD = '''"""Where can the weak classes' box edges be read from? CPU only.

Assembled by tools/build_edge_probe.py -- do not edit directly.
"""
import json, math, os, time
from collections import defaultdict
from pathlib import Path
import numpy as np
'''

BODY = r'''

CELL = 4
WEAK = ("stone_block", "people", "e-bike", "car")
CONTROLS = ("badminton", "rubik", "car_toy", "table_tennis")
N_CONTROL = int(os.environ.get("EDGE_N_CONTROL", 300))
POSITIONS = 3                 # along each edge
HALF = 8                      # profile: t = -7..8, inside t <= 0
PATCH_R, PATCH_S = 6, 2       # 13 x 13 support, every 2 px -> 7 x 7
RINGS = ((7, 15), (15, 31), (31, 63), (63, 95), (95, 127))
PAD = HALF + PATCH_R + 2
T0 = time.time()
INPUT = Path(os.environ.get("EDGE_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("EDGE_WORK", "/kaggle/working"))


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def box_sum2(x, k):
    r = k // 2
    p = np.pad(x, r, mode="reflect")
    s = np.pad(np.cumsum(np.cumsum(p, 0, dtype=np.float64), 1), ((1, 0), (1, 0)))
    h, w = x.shape
    return (s[k:k + h, k:k + w] - s[:h, k:k + w] - s[k:k + h, :w] + s[:h, :w]).astype(np.float32)


def maps(cube):
    """Per-pixel feature maps of one frame, padded by PAD."""
    L = normalise_frame(align_bands(cube.astype(np.float32)))        # (H, W, 16)
    Y = L.mean(-1)
    P = L @ LDA.T                                                     # (H, W, 3)
    gy, gx = np.gradient(box_sum2(Y, 3) / 9.0)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    var = [box_sum2(Y * Y, k) / k ** 2 - (box_sum2(Y, k) / k ** 2) ** 2 for k in (5, 9)]
    ctx = [Y - (box_sum2(Y, o) - box_sum2(Y, i)) / float(o * o - i * i) for i, o in RINGS]
    pix = np.concatenate([L, Y[..., None], np.stack([grad] + var, -1), np.stack(ctx, -1)], -1)
    pad = lambda a: np.pad(a, ((PAD, PAD), (PAD, PAD)) + ((0, 0),) * (a.ndim - 2), mode="reflect")
    return pad(pix).astype(np.float32), pad(Y).astype(np.float32), pad(P).astype(np.float32)


OFF = np.arange(-PATCH_R, PATCH_R + 1, PATCH_S)


def feat_at(pix, Y, P, ys, xs):
    """(n, D) features at padded coordinates."""
    dy, dx = np.meshgrid(OFF, OFF, indexing="ij")
    py, px = ys[:, None] + dy.reshape(-1)[None], xs[:, None] + dx.reshape(-1)[None]
    return np.concatenate([pix[ys, xs], Y[py, px], P[py, px].reshape(len(ys), -1)], 1)


# column layout of feat_at
C_SPEC = list(range(0, 16))
C_BRIGHT = [16]
C_TEX = [17, 18, 19]
C_CTX = [20, 21, 22, 23, 24]
N_PIX = 25
C_SHAPE = list(range(N_PIX, N_PIX + len(OFF) ** 2))
C_LDA = list(range(N_PIX + len(OFF) ** 2, N_PIX + 4 * len(OFF) ** 2))
GROUPS = {
    "F0 spectral16": C_SPEC,
    "F1 brightness": C_BRIGHT,
    "F2 texture": C_BRIGHT + C_TEX,
    "F3 context": C_BRIGHT + C_CTX,
    "F4 shape": C_SHAPE,
    "F5 shape+spectral": C_SHAPE + C_LDA + C_SPEC,
    "F6 all": C_SHAPE + C_LDA + C_SPEC + C_BRIGHT + C_TEX + C_CTX,
}


def edges(b):
    """(side, fixed coordinate of the last inside pixel, outward sign, along-axis span)."""
    return [("top", b.y1, -1, (b.x1, b.x2)), ("bottom", b.y2 - 1, +1, (b.x1, b.x2)),
            ("left", b.x1, -1, (b.y1, b.y2)), ("right", b.x2 - 1, +1, (b.y1, b.y2))]


def frame_samples(pid, boxes, chosen):
    cube = load_planar(IMG / f"{pid}.png").astype(np.float32)
    H, W = cube.shape[:2]
    pix, Y, P = maps(cube)
    t = np.arange(-HALF + 1, HALF + 1)           # 16 steps; inside t <= 0
    out = []
    for k, b in enumerate(boxes):
        if k not in chosen or b.x2 - b.x1 < 4 or b.y2 - b.y1 < 4:
            continue
        others = [o for j, o in enumerate(boxes) if j != k]
        for side, a, s, (lo, hi) in edges(b):
            span = hi - lo
            for q in range(POSITIONS):
                along = int(lo + span * (0.2 + 0.6 * q / max(1, POSITIONS - 1)))
                steps = a + s * t
                if side in ("top", "bottom"):
                    ys, xs = steps, np.full_like(steps, along)
                else:
                    ys, xs = np.full_like(steps, along), steps
                if ys.min() < 0 or xs.min() < 0 or ys.max() >= H or xs.max() >= W:
                    continue
                covered = np.array([any(o.x1 <= x < o.x2 and o.y1 <= y < o.y2 for o in others)
                                    for y, x in zip(ys, xs)])
                # shared edge: the first outside pixels belong to another box
                shared = bool(covered[HALF:HALF + 2].any())
                f = feat_at(pix, Y, P, ys + PAD, xs + PAD).astype(np.float16)
                out.append((CLASSES[b.cls_id], (f, covered, side, shared)))
    return out


def fit_logreg(X, y, lam=1e-2, iters=60):
    import torch
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    Xs = (X - mu) / sd
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xs @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) + lam * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    w, b = w.detach(), b.detach()
    return lambda Z: torch.sigmoid(((torch.as_tensor(Z, dtype=torch.float32) - mu) / sd) @ w + b).numpy()


def auc(p, n):
    x = np.concatenate([p, n])
    r = np.empty(len(x))
    r[x.argsort(kind="mergesort")] = np.arange(1, len(x) + 1)
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def edge_error(prob):
    """Profiles (n, 16), inside first -> |k - HALF|: the split maximising mean(in) - mean(out)."""
    best = np.zeros(len(prob), int)
    score = np.full(len(prob), -9.0)
    for k in range(3, 2 * HALF - 2):
        s = prob[:, :k].mean(1) - prob[:, k:].mean(1)
        better = s > score
        best[better], score[better] = k, s[better]
    return np.abs(best - HALF)


def evaluate(rows, groups):
    """rows: list of (frame, feats (16, D), covered (16,), side, shared)."""
    frames = np.array([r[0] for r in rows])
    fold = np.array([hash(int(f)) % 5 for f in frames])
    F = np.stack([r[1] for r in rows]).astype(np.float32)            # (n, 16, D)
    cov = np.stack([r[2] for r in rows])
    side = np.array([r[3] for r in rows])
    shared = np.array([r[4] for r in rows])
    train_t = np.r_[HALF - 3:HALF + 3]                                 # t = -2..3
    res = {}
    for g, cols in groups.items():
        prob = np.zeros((len(rows), 2 * HALF), np.float32)
        pos_s, neg_s = [], []
        for k in range(5):
            tr, te = fold != k, fold == k
            if not te.any() or not tr.any():
                continue
            Xi = F[tr][:, train_t][:, :, cols]
            lab = np.broadcast_to((np.arange(2 * HALF)[train_t] < HALF)[None], Xi.shape[:2])
            keep = ~cov[tr][:, train_t] | lab                        # no covered outside pixel
            clf = fit_logreg(Xi[keep], lab[keep].astype(np.float32))
            prob[te] = clf(F[te][:, :, cols].reshape(-1, len(cols))).reshape(-1, 2 * HALF)
            pt = prob[te][:, train_t]
            lt = np.broadcast_to((np.arange(2 * HALF)[train_t] < HALF)[None], pt.shape)
            kt = ~cov[te][:, train_t] | lt
            pos_s.append(pt[kt & lt]); neg_s.append(pt[kt & ~lt])
        err = edge_error(prob)
        r = {"auc": auc(np.concatenate(pos_s), np.concatenate(neg_s)),
             "n_edges": int(len(rows)),
             "within0": float((err == 0).mean()), "within1": float((err <= 1).mean()),
             "within2": float((err <= 2).mean()), "median_err": float(np.median(err))}
        for name, m in (("free", ~shared), ("shared", shared)):
            if m.sum() >= 10:
                r[f"within2_{name}"] = float((err[m] <= 2).mean())
                r[f"n_{name}"] = int(m.sum())
        for sd in ("top", "bottom", "left", "right"):
            m = side == sd
            if m.sum() >= 10:
                r[f"within2_{sd}"] = float((err[m] <= 2).mean())
        res[g] = r
    # each context ring alone (univariate, sign-free)
    ring = {}
    lab = np.arange(2 * HALF) < HALF
    for j, (i_, o_) in enumerate(RINGS):
        v = F[:, train_t][:, :, C_CTX[j]]
        l = np.broadcast_to(lab[train_t][None], v.shape)
        keep = ~cov[:, train_t] | l
        a = auc(v[keep & l], v[keep & ~l])
        ring[f"{i_}/{o_}"] = max(a, 1 - a)
    res["_rings"] = ring
    return res


def main():
    global IMG
    ann_dir = next(p for p in INPUT.rglob("train/annotations") if p.is_dir())
    IMG = ann_dir.parent / "images"
    ids = sorted(int(p.stem) for p in ann_dir.glob("*.xml"))
    anns = {pid: parse(ann_dir / f"{pid}.xml") for pid in ids}
    rng = np.random.default_rng(0)
    want = {}                                   # frame -> set of box indices
    by_cls = defaultdict(list)
    for pid, a in anns.items():
        for k, b in enumerate(a.boxes):
            by_cls[CLASSES[b.cls_id]].append((pid, k))
    for c in WEAK:
        for pid, k in by_cls[c]:
            want.setdefault(pid, set()).add(k)
    for c in CONTROLS:
        pick = by_cls[c]
        for j in rng.permutation(len(pick))[:N_CONTROL]:
            pid, k = pick[j]
            want.setdefault(pid, set()).add(k)
    log(f"{len(want)} frames, " + ", ".join(f"{c} {len(by_cls[c])}" for c in WEAK)
        + f"; controls {N_CONTROL} each")
    from multiprocessing import Pool
    jobs = [(pid, list(anns[pid].boxes), want[pid]) for pid in sorted(want)]
    rows = defaultdict(list)
    with Pool(max(1, os.cpu_count() or 1)) as pool:
        for n, (pid, res) in enumerate(pool.imap(_job, jobs, chunksize=4)):
            for cls, item in res:
                rows[cls].append((pid,) + item)
            if n % 200 == 0:
                log(f"  frames {n}/{len(jobs)}")
    report = {}
    for c in WEAK + CONTROLS:
        if len(rows[c]) < 50:
            log(f"{c}: only {len(rows[c])} edge profiles, skipped")
            continue
        report[c] = evaluate(rows[c], GROUPS)
        log(f"{c}: {len(rows[c])} edge profiles")
        for g, r in report[c].items():
            if g.startswith("_"):
                continue
            log(f"    {g:18s} AUC {r['auc']:.3f}  edge within 0/1/2 px {r['within0']:.2f}/{r['within1']:.2f}/"
                f"{r['within2']:.2f}  free {r.get('within2_free', float('nan')):.2f} "
                f"shared {r.get('within2_shared', float('nan')):.2f}")
        log("    rings " + ", ".join(f"{k} {v:.3f}" for k, v in report[c]["_rings"].items()))
    (WORK / "edge_probe.json").write_text(json.dumps(report, indent=1))
    log("EDGE PROBE DONE")


def _job(job):
    pid, boxes, chosen = job
    return pid, frame_samples(pid, boxes, chosen)


if __name__ == "__main__":
    main()
'''


def main() -> None:
    from hod26.spectral import LDA_16_TO_3
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--slug", default="zetaoxia/hod26-edge-probe")
    ap.add_argument("--dataset", default="xishengfeng/hod26-planar")
    args = ap.parse_args()
    parts = [
        HEAD,
        strip("src/hod26/voc.py"),
        strip("src/hod26/cube.py"),
        *(function_source("src/hod26/s3t/preprocess.py", f)
          for f in ("band_offsets", "_shift_axis", "align_bands", "normalise_frame")),
        "LDA = np.array(" + repr(LDA_16_TO_3) + ", dtype=np.float32)",
        BODY,
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "edge_probe.py").write_text("\n\n".join(parts))
    (args.out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": args.slug, "title": args.slug.split("/")[-1].replace("-", " ").title(),
        "code_file": "edge_probe.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_internet": False,
        "competition_sources": [], "dataset_sources": [args.dataset], "kernel_sources": [],
    }, indent=2))
    print(f"wrote {args.out_dir / 'edge_probe.py'} (CPU only)")


if __name__ == "__main__":
    main()
