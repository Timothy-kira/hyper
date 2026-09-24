"""The fine-tune aimed at stone_block / people / e-bike / car.

S1  the stem also reads the 16 bands: identical output at step 0, gradient
    reaches the new channels, staged unfreezing sees them as stem_in;
    warm start carries a finished 3-channel model into the widened one exactly.
C1  per-class box-loss gain: gain 1 is the stock loss; gain 2 scales only the
    named class's pairs.
C2  repulsion: zero (and zero gradient) for predictions equal to their
    targets even when the targets overlap; positive for a box pressing
    further into a neighbour, and its gradient moves the box off it.
B2  crowd_paste: pastes only where an anchor of the crowded classes is,
    boxes stay inside the frame, covered boxes are dropped, deterministic.
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from test_dfine import fake_batch  # noqa: E402
from test_s3t_detr import build_kernel  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        fails.append(name)


def detector(m, torch, stem_bands, seed=0):
    from ultralytics.nn.tasks import RTDETRDetectionModel
    torch.manual_seed(seed)
    net = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=18, verbose=False)
    m.install_bbox_loss(net, 18, "GIoU", log_size=True, fdr=True, mal=True)
    m.install_spectral_adapter(net, 16, projection=m.LDA_16_TO_3, kind="s3t", scale=0.5,
                               arch="xca", upsample=2, stem_bands=stem_bands)
    m.freeze_batchnorm(net)
    return net


def main() -> int:
    import torch

    from hod26.augment import crowd_paste
    from hod26.voc import Box
    from tools.s3t_round import WEAK, finetune_candidate, s3t_candidate

    cand = finetune_candidate(s3t_candidate(total=2, arch="xca"), "final_best.pt", 2)
    with tempfile.TemporaryDirectory() as td:
        m = build_kernel(Path(td), cand)

        # ---- S1 + warm start
        src = detector(m, torch, stem_bands=False, seed=0)
        ck = Path(td) / "final_best.pt"
        torch.save({"model": copy.deepcopy(src).half(), "ema": None}, ck)
        net = detector(m, torch, stem_bands=True, seed=1)
        conv = next(x for x in net.model[0].block.modules() if isinstance(x, torch.nn.Conv2d))
        check("stem's first conv reads 3 + 16 channels", conv.in_channels == 19, str(conv.in_channels))
        m.warm_start(net, ck)
        check("warm start: widened conv = checkpoint's on the first 3, zero on the 16 new",
              torch.allclose(conv.weight[:, :3], next(
                  x for x in src.model[0].block.modules() if isinstance(x, torch.nn.Conv2d)).weight.half().float(),
                  atol=1e-6) and float(conv.weight[:, 3:].abs().sum()) == 0.0)
        x = torch.rand(1, 16, 128, 160)
        src_h = copy.deepcopy(src).half().float().eval()
        net.eval()
        with torch.no_grad():
            a, b = net(x), src_h(x)
        check("warm-started widened model == the checkpoint's model at step 0 (decoder boxes, scores)",
              torch.allclose(a[1][0], b[1][0], atol=1e-4) and torch.allclose(a[1][1], b[1][1], atol=1e-4),
              f"{float((a[1][0] - b[1][0]).abs().max()):.2e}")
        net.train()
        loss, _ = net.loss(fake_batch(torch, ch=16))
        net.zero_grad()
        loss.backward()
        g = conv.weight.grad
        check("gradient reaches the 16 new stem channels", g is not None and float(g[:, 3:].abs().sum()) > 0)
        parts = m.param_parts(net)
        stem_in = [n for _, (pt, n) in parts.items() if pt == "stem_in"]
        check("staged unfreezing: the widened conv is part stem_in, the rest of the stem is not",
              stem_in == ["model.0.block.stem1.conv.weight"], str(stem_in))
        dark = torch.full((1, 16, 1, 1), -0.2)
        w3 = net.model[0].front.base.weight.float()
        check("the LDA projection itself is blind to a uniform darkening (why S1 exists)",
              float((w3 * dark).sum(1).abs().max()) < 2e-3)   # fp16-rounded rows
        bad = torch.nn.Sequential(detector(m, torch, stem_bands=True, seed=2))
        try:
            m.warm_start(bad, ck)
            refused = False
        except RuntimeError:
            refused = True
        check("warm start refuses a model it cannot fully fill", refused)

        # ---- a fine-tune never resumes the parent run it mounts
        real, real_find = m.stage_checkpoint, m.find_checkpoint
        m.stage_checkpoint = lambda tag: Path("/kaggle/input/parent/final_last.pt")
        m.find_checkpoint = lambda tag: Path("/kaggle/input/parent/final_last.pt")
        try:
            plain = s3t_candidate(total=2, arch="xca")
            ft_resume = copy.deepcopy(cand)
            ft_resume["require_resume"] = True
            check("fine-tune: the parent's final_last.pt is not taken for a resume",
                  m.resume_checkpoint(cand, "final") is None
                  and m.resume_checkpoint(plain, "final") is not None
                  and m.resume_checkpoint(ft_resume, "final") is not None)
        finally:
            m.stage_checkpoint, m.find_checkpoint = real, real_find

        # ---- C1 / C2
        crit = net.criterion
        check("fine-tune candidate installs the weak-class gain and RepGT",
              crit.box_cls_gain == {} and crit.rep_gain == 0.0)   # this net was built without them
        m.install_bbox_loss(net, 18, "GIoU", log_size=True, fdr=True, mal=True,
                            box_cls_gain={c: 2.0 for c in WEAK}, rep_gain=0.5)
        crit = net.criterion
        check("  (names resolved to class ids)",
              sorted(crit.box_cls_gain) == sorted(m.CLASSES.index(c) for c in WEAK))

        def bbox_terms(gain, rep, cls_of_pairs):
            crit.box_cls_gain, crit.rep_gain = gain, rep
            gt = torch.tensor([[0.30, 0.50, 0.10, 0.20], [0.36, 0.50, 0.10, 0.20], [0.80, 0.20, 0.05, 0.05]])
            pred = (gt + torch.tensor([[0.01, 0.0, 0.02, 0.0], [-0.01, 0.01, 0.0, 0.01],
                                       [0.0, 0.0, 0.01, 0.01]])).requires_grad_(True)
            gt_cls = torch.tensor(cls_of_pairs)
            mi = [(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))]
            crit.__dict__["_box_ctx"] = (mi, gt, gt_cls, [3])
            try:
                out = crit._get_loss_bbox(pred, gt)
            finally:
                crit.__dict__.pop("_box_ctx", None)
            return out, pred

        people, apple = m.CLASSES.index("people"), m.CLASSES.index("apple")
        base, _ = bbox_terms({}, 0.0, [apple, apple, apple])
        same, _ = bbox_terms({people: 2.0}, 0.0, [apple, apple, apple])
        up, _ = bbox_terms({people: 2.0}, 0.0, [people, apple, apple])
        check("C1: no weak class among the pairs -> the stock loss exactly",
              torch.allclose(base["loss_bbox"], same["loss_bbox"]) and torch.allclose(base["loss_giou"], same["loss_giou"]))
        check("C1: a people pair at gain 2 raises the box terms", float(up["loss_bbox"]) > float(base["loss_bbox"]))
        rep, pred = bbox_terms({}, 0.5, [people, people, apple])
        check("C2: two overlapping neighbours -> RepGT adds to the overlap term",
              float(rep["loss_giou"]) > float(base["loss_giou"]))
        norep, pred0 = bbox_terms({}, 0.0, [people, people, apple])
        g = (torch.autograd.grad(rep["loss_giou"], pred)[0]
             - torch.autograd.grad(norep["loss_giou"], pred0)[0])      # the repulsion's own gradient
        check("C2: the repulsion pushes box 0 left (away from box 1) and box 1 right",
              float(g[0, 0]) > 0 and float(g[1, 0]) < 0, str(g[:2, 0].tolist()))
        check("C2: the isolated box feels no repulsion", float(g[2].abs().sum()) == 0.0)

        # the objection that shaped C2: ground truths overlap each other, and
        # the exact answer must cost nothing
        crit.box_cls_gain, crit.rep_gain = {}, 0.5
        gt = torch.tensor([[0.30, 0.50, 0.10, 0.20], [0.36, 0.50, 0.10, 0.20]])
        exact = gt.clone().requires_grad_(True)
        mi = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        ctx = (torch.tensor([0, 0]), torch.tensor([0, 1]), gt, torch.tensor([people, people]), [2])
        r = crit._rep_gt(exact, ctx)
        gr = torch.autograd.grad(r, exact, allow_unused=True)[0]
        check("C2: overlapping ground truths predicted exactly -> no repulsion, no gradient",
              float(r) == 0.0 and (gr is None or float(gr.abs().sum()) == 0.0), f"{float(r)}")
        ebike = m.CLASSES.index("e-bike")
        drift = (gt + torch.tensor([[0.02, 0.0, 0.03, 0.0], [-0.02, 0.0, 0.03, 0.0]])).requires_grad_(True)
        r_same = crit._rep_gt(drift, (torch.tensor([0, 0]), torch.tensor([0, 1]), gt,
                                      torch.tensor([people, people]), [2]))
        r_cross = crit._rep_gt(drift, (torch.tensor([0, 0]), torch.tensor([0, 1]), gt,
                                       torch.tensor([people, ebike]), [2]))
        check("C2: same-class neighbours only -- a rider over an e-bike is never repelled",
              float(r_same) > 0 and float(r_cross) == 0.0, f"{float(r_same)} {float(r_cross)}")

        # ---- B2 crowd paste
        rng_src = np.random.default_rng(0)
        src_cube = (rng_src.random((60, 80, 16)) * 1000 + 500).astype(np.uint16)
        src_cube[20:40, 30:40] //= 2
        dst_cube = (rng_src.random((60, 80, 16)) * 1000 + 500).astype(np.uint16)
        pid = m.CLASSES.index("people")
        pool = {pid: [("s", (30, 20, 40, 40))]}
        anchor = Box(pid, 40, 20, 50, 40)
        victim = Box(m.CLASSES.index("people"), 31, 22, 38, 38)
        c1, b1 = crowd_paste(dst_cube, [anchor, victim], pool, np.random.default_rng(3), {pid},
                             lambda k: src_cube, n_max=1, margin=4)
        c2, b2 = crowd_paste(dst_cube, [anchor, victim], pool, np.random.default_rng(3), {pid},
                             lambda k: src_cube, n_max=1, margin=4)
        new = [b for b in b1 if b not in (anchor, victim)]
        check("B2: one instance pasted beside the anchor, inside the frame",
              len(new) == 1 and new[0].cls_id == pid and 0 <= new[0].x1 < new[0].x2 <= 80
              and 0 <= new[0].y1 < new[0].y2 <= 60 and (new[0].x2 - new[0].x1, new[0].y2 - new[0].y1) == (10, 20),
              str(b1))
        check("B2: deterministic for a seed", np.array_equal(c1, c2) and b1 == b2)
        check("B2: dtype kept, pixels changed only around the paste",
              c1.dtype == dst_cube.dtype and not np.array_equal(c1, dst_cube)
              and np.array_equal(c1[:, :new[0].x1 - 4 if new[0].x1 >= 4 else 0], dst_cube[:, :new[0].x1 - 4 if new[0].x1 >= 4 else 0]))
        c3, b3 = crowd_paste(dst_cube, [Box(m.CLASSES.index("apple"), 5, 5, 15, 15)], pool,
                             np.random.default_rng(3), {pid}, lambda k: src_cube)
        check("B2: a frame without a crowded-class anchor is untouched (tabletop scenes)",
              np.array_equal(c3, dst_cube) and len(b3) == 1)
        car_anchor = Box(m.CLASSES.index("car"), 40, 20, 60, 35)
        c4, b4 = crowd_paste(dst_cube, [car_anchor], pool, np.random.default_rng(3),
                             {pid, m.CLASSES.index("car")}, lambda k: src_cube, n_max=3)
        check("B2: same class only -- a person is never pasted beside a car",
              np.array_equal(c4, dst_cube) and b4 == [car_anchor])
        covered = [b for b in b1 if b == victim]
        overlap = (max(0, min(victim.x2, new[0].x2) - max(victim.x1, new[0].x1))
                   * max(0, min(victim.y2, new[0].y2) - max(victim.y1, new[0].y1)))
        frac = overlap / ((victim.x2 - victim.x1) * (victim.y2 - victim.y1))
        check("B2: a box the paste covers by > 70% is dropped, others kept",
              (not covered) == (frac > 0.7) and anchor in b1, f"covered {frac:.2f}")

    print(f"\n{len(fails)} failure(s)" if fails else "\nall weak-class checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
