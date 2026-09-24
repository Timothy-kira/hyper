#!/usr/bin/env python3
"""A CPU kernel that measures how much the S3T branch contributes to the detector.

Two questions, asked of the finished S3T-X model (and of the weak-class
fine-tune) on held-out frames, at no GPU cost:

1. Magnitude: at each place the spectral branch joins the detector -- the
   stem's stride-4 output (fuse) and the P5/P4/P3 input projections (the
   injections) -- the RMS of what it adds relative to the RMS of what the
   pretrained path produced there. Whole map, and inside the boxes of the
   weak classes (stone_block / people / e-bike / car) against inside the boxes
   of the others.
2. Effect: per-class AP with every S3T contribution zeroed (fuse and injection
   projections) against the intact model, on the street frames that hold the
   weak classes plus a sample of tabletop frames. This is the number that says
   whether the branch matters, not just whether it is large.

For the fine-tune it also reports how large the stem's 16 extra band channels
(S1) are at the first conv's output against the 3 projected ones.

The kernel is the round driver itself with its entry point replaced, so
checkpoints unpickle against the same classes and frames are rendered and
scored exactly as in training:

    python3 tools/build_fusion_probe.py --out-dir /tmp/probe --slug zetaoxia/hod26-s3tx-fusion-probe
    kaggle kernels push -p /tmp/probe
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PROBE = r'''

# ---------------------------------------------------------------- probe ----
PROBE_WEAK = ["stone_block", "people", "e-bike", "car"]
PROBE_TABLE = 40
PROBE_SIZE = 512


def _probe_frames(root):
    ann_dir = root / "train" / "annotations"
    ids = require_ids(sorted(int(p.stem) for p in ann_dir.glob("*.xml")), ann_dir)
    _, val_ids = split_ids(ids)
    anns = {pid: parse(ann_dir / f"{pid}.xml") for pid in val_ids}
    weak = {CLASSES.index(c) for c in PROBE_WEAK}
    street = [p for p in val_ids if any(b.cls_id in weak for b in anns[p].boxes)]
    s = set(street)
    other = [p for p in val_ids if p not in s]
    rng = np.random.default_rng(0)
    table = sorted(int(v) for v in rng.choice(other, min(PROBE_TABLE, len(other)), replace=False))
    return street, table, anns


def _probe_input(root, pid, channels):
    import torch
    img = build_channels(load_planar(root / "train" / "images" / f"{pid}.png"), channels)
    H, W = img.shape[:2]
    # RT-DETR's predictor stretches to the square (LetterBox scale_fill).
    r = cv2.resize(img, (PROBE_SIZE, PROBE_SIZE), interpolation=cv2.INTER_LINEAR)
    if r.ndim == 2:
        r = r[:, :, None]
    x = torch.from_numpy(np.ascontiguousarray(r.transpose(2, 0, 1))).float()[None] / 255.0
    return x, W, H


def _probe_load(tag):
    import torch
    hits = sorted(set(INPUT.glob(f"**/{tag}/final_best.pt")) | set(INPUT.glob(f"**/{tag}/**/final_best.pt")))
    if not hits:
        raise RuntimeError(f"no final_best.pt of {tag} under {INPUT}")
    ck = torch.load(hits[0], map_location="cpu", weights_only=False)
    net = (ck.get("ema") or ck.get("model")) if isinstance(ck, dict) else ck
    return net.float().eval(), hits[0]


def _probe_hooks(net, rec):
    front = net.model[0].front
    orig = front.fuse_stem

    def fuse_stem(y):
        out = orig(y)
        rec["stem"] = (y.detach(), (out - y).detach())
        return out
    front.__dict__["fuse_stem"] = fuse_stem
    for i in (10, 14, 19):
        m = net.model[i]
        if not hasattr(m, "layer"):
            continue
        m.layer.register_forward_hook(lambda mod, a, out, i=i: rec.__setitem__(f"b{i}", out.detach()))
        m.register_forward_hook(
            lambda mod, a, out, i=i: rec.__setitem__(f"P{ {10: 5, 14: 4, 19: 3}[i] }",
                                                     (rec[f"b{i}"], (out - rec[f"b{i}"]).detach())))


def _probe_mask(boxes, W, H, h, w):
    import torch
    m = torch.zeros(h, w, dtype=torch.bool)
    for b in boxes:
        x1, x2 = int(b.x1 * w / W), max(int(b.x1 * w / W) + 1, int(np.ceil(b.x2 * w / W)))
        y1, y2 = int(b.y1 * h / H), max(int(b.y1 * h / H) + 1, int(np.ceil(b.y2 * h / H)))
        m[y1:y2, x1:x2] = True
    return m


def _probe_rows(y, pid, W, H):
    y = y[0].float()
    boxes, scores = y[:, :4], y[:, 4:]
    s, c = scores.max(-1)
    cx, cy, bw, bh = boxes.unbind(-1)
    out = []
    for k in range(len(s)):
        out.append((pid, int(c[k]), float(s[k]), float((cx[k] - bw[k] / 2) * W), float((cy[k] - bh[k] / 2) * H),
                    float((cx[k] + bw[k] / 2) * W), float((cy[k] + bh[k] / 2) * H)))
    return out


def _probe_zero_s3t(net):
    import copy as _copy
    import torch
    z = _copy.deepcopy(net)
    front = z.model[0].front
    with torch.no_grad():
        front.fuse.weight.zero_()
        front.fuse.bias.zero_()
        for i in (10, 14, 19):
            if hasattr(z.model[i], "proj"):
                z.model[i].proj.weight.zero_()
                z.model[i].proj.bias.zero_()
    # the deepcopy keeps each injection's front pointing at the copy's front
    return z


def probe():
    import os
    import torch
    import torch.nn.functional as TF
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    root = data_root()
    street, table, anns = _probe_frames(root)
    frames = street + table
    weak = {CLASSES.index(c) for c in PROBE_WEAK}
    log(f"PROBE: {len(street)} held-out street frames (weak classes) + {len(table)} tabletop frames, "
        f"{torch.get_num_threads()} CPU threads")
    channels = PROBE_CANDIDATE["channels"]
    report = {"frames": {"street": len(street), "table": len(table)}}
    for tag in ("hod26-s3t-detr", "hod26-s3t-ft"):
        try:
            net, path = _probe_load(tag)
        except RuntimeError as exc:
            log(f"  {tag}: {exc}")
            continue
        log(f"PROBE {tag}: {path}")
        # the zeroed copy first: a copy taken after the hooks would carry the
        # recording fuse_stem, which calls the intact model's fusion
        off = _probe_zero_s3t(net)
        rec = {}
        _probe_hooks(net, rec)
        sums = {}          # (site, region) -> [sum sq delta, sum sq base, n]

        def acc(site, region, base, delta, mask=None):
            if mask is not None:
                if not bool(mask.any()):
                    return
                base, delta = base[..., mask], delta[..., mask]
            s = sums.setdefault((site, region), [0.0, 0.0])
            s[0] += float((delta.float() ** 2).mean())
            s[1] += float((base.float() ** 2).mean())
        s1 = {}
        preds, preds_off = [], []
        t0 = time.time()
        for n, pid in enumerate(frames):
            x, W, H = _probe_input(root, pid, channels)
            with torch.no_grad():
                out = net(x)
            preds += _probe_rows(out[0] if isinstance(out, (list, tuple)) else out, pid, W, H)
            bw = [b for b in anns[pid].boxes if b.cls_id in weak]
            bo = [b for b in anns[pid].boxes if b.cls_id not in weak]
            for site in ("stem", "P3", "P4", "P5"):
                if site not in rec:
                    continue
                base, delta = rec[site]
                h, w = base.shape[-2:]
                acc(site, "all", base, delta)
                acc(site, "weak boxes", base, delta, _probe_mask(bw, W, H, h, w))
                acc(site, "other boxes", base, delta, _probe_mask(bo, W, H, h, w))
            front = net.model[0].front
            if getattr(front, "stem_bands", False):
                conv = next(m for m in net.model[0].block.modules() if isinstance(m, torch.nn.Conv2d))
                with torch.no_grad():
                    xin = front(x)
                    a3 = TF.conv2d(xin[:, :3], conv.weight[:, :3], None, conv.stride, conv.padding)
                    a16 = TF.conv2d(xin[:, 3:], conv.weight[:, 3:], None, conv.stride, conv.padding)
                h, w = a3.shape[-2:]
                for region, mask in (("all", None), ("weak boxes", _probe_mask(bw, W, H, h, w))):
                    b3, b16 = (a3, a16) if mask is None else (a3[..., mask], a16[..., mask])
                    if mask is not None and not bool(mask.any()):
                        continue
                    s = s1.setdefault(region, [0.0, 0.0])
                    s[0] += float((b16 ** 2).mean())
                    s[1] += float((b3 ** 2).mean())
            with torch.no_grad():
                out_off = off(x)
            preds_off += _probe_rows(out_off[0] if isinstance(out_off, (list, tuple)) else out_off, pid, W, H)
            if n % 20 == 0:
                log(f"  {tag}: {n + 1}/{len(frames)} frames, {time.time() - t0:.0f}s")
        ratios = {f"{site} / {region}": (d / b) ** 0.5 if b > 0 else None
                  for (site, region), (d, b) in sums.items()}
        log(f"PROBE {tag}: RMS(added by S3T) / RMS(pretrained path) at each join:")
        for k, v in sorted(ratios.items()):
            log(f"    {k:22s} {v:.4f}")
        if s1:
            s1r = {k: (a / b) ** 0.5 for k, (a, b) in s1.items()}
            log(f"PROBE {tag}: S1 stem1 RMS(16 band channels) / RMS(3 projected) = "
                + ", ".join(f"{k} {v:.4f}" for k, v in s1r.items()))
        else:
            s1r = None
        sub = [anns[p] for p in frames]
        on = evaluate(sub, preds, per_class=True)
        offs = evaluate(sub, preds_off, per_class=True)
        log(f"PROBE {tag}: AP with the S3T branch vs zeroed (these {len(frames)} frames): "
            f"mAP {on['mAP']:.4f} vs {offs['mAP']:.4f}")
        for c in sorted(on["per_class"], key=lambda c: on["per_class"][c]):
            log(f"    {c:16s} {on['per_class'][c]:.4f} -> {offs['per_class'][c]:.4f}  "
                f"({offs['per_class'][c] - on['per_class'][c]:+.4f} without S3T)")
        report[tag] = {"ratios": ratios, "s1": s1r, "ap_on": on, "ap_off": offs}
    (WORK / "fusion_probe.json").write_text(json.dumps(report, indent=1, default=float))
    log("PROBE DONE")


if __name__ == "__main__":
    probe()
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--slug", default="zetaoxia/hod26-s3tx-fusion-probe")
    ap.add_argument("--candidate", type=Path, required=True,
                    help="results.json of the run whose candidate (channel spec) the frames are rendered with")
    ap.add_argument("--kernel-source", action="append", default=["zetaoxia/hod26-s3t-detr", "zetaoxia/hod26-s3t-ft"])
    args = ap.parse_args()
    cand = json.loads(args.candidate.read_text())["candidate"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = args.out_dir / "round.json"
    cfg.write_text(json.dumps({"round": "fusion-probe", "candidates": []}))
    subprocess.run([sys.executable, str(REPO / "tools" / "build_kernel.py"), "--round-config", str(cfg),
                    "--out-dir", str(args.out_dir), "--slug", args.slug, "--machine-shape", "cpu",
                    *[a for k in args.kernel_source for a in ("--kernel-source", k)]], check=True)
    cfg.unlink()
    py = args.out_dir / "hod26_round.py"
    src = py.read_text()
    guard = '\nif __name__ == "__main__":\n    main()'
    if not src.rstrip().endswith(guard.strip()):
        raise SystemExit("unexpected end of the generated kernel; the probe cannot replace its entry point")
    src = src.rstrip()[: -len(guard.strip())].rstrip() + "\n\nPROBE_CANDIDATE = json.loads(" + repr(json.dumps(cand)) + ")\n" + PROBE
    py.write_text(src)
    print(f"wrote {py} ({len(src)} bytes)")


if __name__ == "__main__":
    main()
