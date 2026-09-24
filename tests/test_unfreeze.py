"""Staged unfreezing, frozen BatchNorm and the 512 loader with the GPU upsample.

Every parameter of the S3T-X detector must belong to exactly one part; a step
must move exactly the parts whose multiplier is above zero at that epoch (and
the frozen ones by exactly nothing), by no more than their share of the LR;
the LR that the warmup and the scheduler write must survive the step; and the
optimizer must still work under torch's LR scheduler. BatchNorm must keep
COCO's running statistics through train-mode forwards, and the model must
still pickle. With upsample 2 the detector's input is the 2x of what the
loader gives, and the encoder reads the loader's input unresized.
"""

from __future__ import annotations

import copy
import pickle
import sys
import tempfile
import types
from pathlib import Path

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


def main() -> int:
    import torch
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.nn.tasks import RTDETRDetectionModel

    from tools.s3t_round import UNFREEZE, s3t_candidate

    cand = s3t_candidate(total=2, arch="xca")
    tr = cand["train"]
    check("S3T-X candidate: loader at 512, 2x upsample on the GPU",
          tr["imgsz"] == 512 and tr["s3t_upsample"] == 2)
    check("S3T-X candidate: staged unfreezing and frozen BatchNorm",
          tr["unfreeze"] == {k: list(v) for k, v in UNFREEZE.items()} and tr["frozen_bn"] is True)

    with tempfile.TemporaryDirectory() as td:
        m = build_kernel(Path(td), cand)
        torch.manual_seed(0)
        net = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=18, verbose=False)
        m.install_bbox_loss(net, 18, "GIoU", log_size=True, fdr=True, mal=True)
        m.install_spectral_adapter(net, 16, projection=m.LDA_16_TO_3, kind="s3t", scale=0.5,
                                   arch="xca", upsample=2)

        # ---- upsample: detector input is 2x the loader's, encoder unresized
        front = net.model[0].front
        x = torch.rand(1, 16, 128, 160)
        with torch.no_grad():
            y = front(x)
        check("front hands the stem a 2x upsample of the projection",
              tuple(y.shape[-2:]) == (256, 320)
              and torch.allclose(y, torch.nn.functional.interpolate(
                  front.base(x), scale_factor=2, mode="bilinear", align_corners=False), atol=1e-6))
        f = front.__dict__["_side"]
        check("encoder output at the stem's stride-4 grid of the 2x image",
              tuple(f.shape[-2:]) == (64, 80), str(tuple(f.shape)))

        # ---- parts
        parts = m.param_parts(net)
        names = {pt: [n for _, (p_, n) in parts.items() if p_ == pt] for pt in m.UNFREEZE_PARTS}
        check("every parameter has a part", len(parts) == len(list(net.parameters())))
        check("every part is non-empty", all(names[pt] for pt in m.UNFREEZE_PARTS),
              str({k: len(v) for k, v in names.items()}))
        check("heads: score/bbox heads and the denoising class embedding",
              all(any(k in n for n in names["head"]) for k in m.UNFREEZE_HEAD_KEYS)
              and not any("decoder.layers" in n for n in names["head"]))
        check("new: S3T fusion/pyramid, injections, D-FINE heads (no pretrained layer)",
              any(n.startswith("model.0.front.fuse") for n in names["new"])
              and any(n.startswith("model.0.front.pyramid") for n in names["new"])
              and any(n.startswith("model.19.q.") for n in names["new"])
              and any(".decoder.fdr." in n for n in names["new"])
              and not any(".layer." in n for n in names["new"]))
        check("stem is model.0.block's convs, its BatchNorm under frozen_norm",
              all(n.startswith("model.0.block.") and ".bn." not in n for n in names["stem"])
              and any(n.startswith("model.0.block.") for n in names["frozen_norm"]))
        check("backbone is layers 1-9, neck 10-27 incl. the wrapped layers",
              all(1 <= int(n.split(".")[1]) <= 9 for n in names["backbone"])
              and all(10 <= int(n.split(".")[1]) <= 27 for n in names["neck"])
              and any(n.startswith("model.19.layer.") for n in names["neck"]))
        check("S3T encoder and mixer", all(n.startswith("model.0.front.enc.") for n in names["s3t_enc"])
              and names["mixer"] == ["model.0.front.base.weight"])

        # ---- frozen BatchNorm
        n_bn = m.freeze_batchnorm(net)
        bns = [x for x in net.modules() if isinstance(x, torch.nn.BatchNorm2d)]
        check("every BatchNorm2d frozen", n_bn == len(bns) > 0
              and all(type(b).__name__ == "FrozenBatchNorm2d" for b in bns))
        net.train()
        check("net.train() leaves BatchNorm in eval mode", not any(b.training for b in bns)
              and net.model[28].training)
        stats = [b.running_mean.clone() for b in bns]
        net.loss(fake_batch(torch, ch=16))
        check("a train-mode forward keeps COCO's running statistics",
              all(torch.equal(a, b.running_mean) for a, b in zip(stats, bns)))

        # ---- the optimizer: ultralytics' groups, split, stepped per stage
        dummy = types.SimpleNamespace(args=types.SimpleNamespace(warmup_bias_lr=0.0), data={"nc": 18})
        opt0 = BaseTrainer.build_optimizer(dummy, net, name="AdamW", lr=1e-3, momentum=0.9, decay=1e-4)
        groups = m.split_groups_by_part(opt0.param_groups, net, cand["train"]["unfreeze"])
        check("split keeps ultralytics' group keys and adds the part",
              all("param_group" in g and g["part"] in m.UNFREEZE_PARTS for g in groups)
              and {g["param_group"] for g in groups} == {"weight", "bn", "bias"})
        trainer = types.SimpleNamespace(epoch=0)
        opt = m.attach_unfreeze(type(opt0)(groups, lr=1e-3), trainer, cand["train"]["unfreeze"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0)
        base_lr = [g["lr"] for g in opt.param_groups]
        batch = fake_batch(torch, ch=16)
        lr = 1e-3

        def step_at(epoch):
            trainer.epoch = epoch
            before = {n: p.detach().clone() for n, p in net.named_parameters()}
            torch.manual_seed(3)
            loss, _ = net.loss(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            moved = {}
            for n, p in net.named_parameters():
                pt = parts[id(p)][0]
                moved[pt] = max(moved.get(pt, 0.0), float((p.detach() - before[n]).abs().max()))
            return moved

        sched_uf = cand["train"]["unfreeze"]
        for epoch in (0, 2, 5, 7):
            moved = step_at(epoch)
            want = {pt: m.unfreeze_mult(sched_uf, pt, epoch) for pt in m.UNFREEZE_PARTS}
            frozen_ok = all(moved[pt] == 0.0 for pt in m.UNFREEZE_PARTS if want[pt] == 0)
            live_ok = all(moved[pt] > 0.0 for pt in m.UNFREEZE_PARTS if want[pt] > 0)
            # Adam moves a parameter by at most ~lr x (1 - b1) / sqrt(1 - b2) ~ 3.2 lr per step.
            bound_ok = all(moved[pt] <= 3.3 * lr * want[pt] + 1e-9 for pt in m.UNFREEZE_PARTS)
            check(f"epoch {epoch + 1}: exactly the live parts move, frozen ones by 0.0",
                  frozen_ok and live_ok, str({k: f"{v:.1e}" for k, v in moved.items()}))
            check(f"epoch {epoch + 1}: each part within its share of the LR", bound_ok,
                  str({k: (f"{v:.1e}", want[k]) for k, v in moved.items()}))
        check("the stem and the backbone BatchNorm never move",
              step_at(30)["stem"] == 0.0 and step_at(31)["frozen_norm"] == 0.0)
        check("the LR the warmup/scheduler wrote is restored after each step",
              [g["lr"] for g in opt.param_groups] == base_lr)
        sched.step()
        check("torch's LR scheduler still steps the wrapped optimizer",
              [g["lr"] for g in opt.param_groups] == base_lr)
        check("ramp: decoder 0.5 on its first epoch, 1.0 after; backbone 0.1/3 .. 0.1",
              m.unfreeze_mult(sched_uf, "decoder", 2) == 0.5
              and m.unfreeze_mult(sched_uf, "decoder", 3) == 1.0
              and abs(m.unfreeze_mult(sched_uf, "backbone", 5) - 0.1 / 3) < 1e-12
              and m.unfreeze_mult(sched_uf, "backbone", 9) == 0.1
              and m.unfreeze_mult(sched_uf, "stem", 30) == 0.0)
        sd = opt.state_dict()
        opt2 = m.attach_unfreeze(type(opt0)(m.split_groups_by_part(opt0.param_groups, net, sched_uf),
                                            lr=1e-3), trainer, sched_uf)
        try:
            opt2.load_state_dict(sd)
            ok = all(g["part"] == h["part"] for g, h in zip(opt2.param_groups, opt.param_groups))
        except Exception as e:  # noqa: BLE001
            ok = str(e)[:200]
        check("optimizer state_dict round-trips (resume), parts kept", ok is True, str(ok))

        # ---- EMA / checkpoint paths
        cp = copy.deepcopy(net)
        back = pickle.loads(pickle.dumps(cp))
        back.train()
        check("deepcopy + pickle keep FrozenBatchNorm2d (still eval under train())",
              all(type(b).__name__ == "FrozenBatchNorm2d" and not b.training
                  for b in back.modules() if isinstance(b, torch.nn.BatchNorm2d)))
        back.eval()
        try:
            back.fuse(verbose=False)
            with torch.no_grad():
                back(torch.rand(1, 16, 128, 160))
            ok = True
        except Exception as e:  # noqa: BLE001
            ok = str(e)[:200]
        check("fuse() + eval forward with frozen BatchNorm and the upsample", ok is True, str(ok))

    print(f"\n{len(fails)} failure(s)" if fails else "\nall unfreeze checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
