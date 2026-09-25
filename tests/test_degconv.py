"""DEGConv (DEGGate) around the RT-DETR decoder inputs, and its curriculum fine-tune.

- the module is the identity at initialisation (gamma = 0), bit for bit, and
  gamma receives a gradient; once gamma moves, the branch does too;
- its orientation histogram peaks at 0 / pi for a vertical step edge and at
  pi/2 for a horizontal one; bf16 autocast gives no NaN;
- on the S3T-X detector: install_degconv wraps exactly the decoder's input
  levels, param_parts puts DEGConv under "deg" and the wrapped layers under
  "neck"; warm start from a checkpoint without DEGConv loads every tensor and
  the wrapped detector reproduces the plain one's output;
- DEG_UNFREEZE: in the first epoch only "deg" moves.
"""

from __future__ import annotations

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
from test_weak_classes import detector  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    import math

    import torch
    from ultralytics.engine.trainer import BaseTrainer

    from hod26.s3t.front import DEGGate
    from tools.s3t_round import DEG_UNFREEZE, plain_finetune_candidate, s3t_candidate

    # ---- the module alone
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(8, 32, 3, padding=1)
    g = DEGGate(conv, 32)
    x = torch.randn(2, 8, 20, 28)
    with torch.no_grad():
        check("identity at init (gamma 0), bit for bit", torch.equal(g(x), conv(x)))
    y = g(x)
    y.square().sum().backward()
    check("gamma receives a gradient at init", g.gamma.grad is not None and g.gamma.grad.abs().sum() > 0)
    g.zero_grad()
    with torch.no_grad():
        g.gamma.fill_(0.1)
    g(x).square().sum().backward()
    check("once gamma moves, edge / gate / embedding train",
          all(p.grad is not None and p.grad.abs().sum() > 0
              for n, p in g.named_parameters() if not n.startswith("layer.")))

    img = torch.zeros(1, 32, 16, 16)
    img[..., :, 8:] = 1.0                       # vertical edge: gradient along x -> theta 0
    hv = g.direction_hist(img)[0].sum((1, 2))
    img2 = torch.zeros(1, 32, 16, 16)
    img2[..., 8:, :] = 1.0                      # horizontal edge: gradient along y -> pi/2
    hh = g.direction_hist(img2)[0].sum((1, 2))
    centres = ((torch.arange(18) + 0.5) * math.pi / 18).tolist()
    check("vertical edge peaks at 0 / pi", int(hv.argmax()) in (0, 17), f"{hv.argmax()} {hv.tolist()}")
    check("horizontal edge peaks at pi/2", abs(centres[int(hh.argmax())] - math.pi / 2) < math.pi / 18,
          f"{hh.argmax()}")
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        g.gamma.fill_(0.1)
        yb = g(x)
    check("bf16 autocast: finite", torch.isfinite(yb.float()).all())
    import copy
    gd = copy.deepcopy(g).double()
    try:
        with torch.no_grad():
            ok = torch.isfinite(gd(x.double())).all() and gd(x.double()).dtype == torch.float64
    except Exception as e:  # noqa: BLE001
        ok = False
        print(e)
    check("model cast whole (.double(), like ultralytics' .half() final eval) runs", bool(ok))

    # ---- on the S3T-X detector
    cand = plain_finetune_candidate(s3t_candidate(total=2, arch="xca"), "final_best.pt", 12)
    cand["train"].update(degconv=True, unfreeze={k: list(v) for k, v in DEG_UNFREEZE.items()})
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        m = build_kernel(td, cand)
        plain = detector(m, torch, stem_bands=False, seed=1)
        ck = td / "plain.pt"
        torch.save({"model": plain}, ck)
        net = detector(m, torch, stem_bands=False, seed=2)
        levels = m.install_degconv(net)
        check("DEGConv wraps exactly the decoder's inputs",
              list(levels) == list(net.model[-1].f)
              and all(type(net.model[i]).__name__ == "DEGGate" for i in levels))
        parts = m.param_parts(net)
        deg = [n for _, (pt, n) in parts.items() if pt == "deg"]
        check("param_parts: DEGConv -> deg, the wrapped layers -> neck",
              deg and all(".layer." not in n for n in deg)
              and all(parts[id(p)][0] == "neck" for i in levels
                      for n, p in net.model[i].named_parameters() if n.startswith("layer.")))
        m.warm_start(net, ck)
        check("warm start: gamma still zero", all(float(net.model[i].gamma.abs().sum()) == 0 for i in levels))
        plain.eval(); net.eval()
        xin = torch.rand(1, 16, 128, 160)
        with torch.no_grad():
            a, b = plain(xin), net(xin)
        a = a[0] if isinstance(a, (list, tuple)) else a
        b = b[0] if isinstance(b, (list, tuple)) else b
        check("wrapped detector reproduces the plain one after warm start",
              torch.allclose(a, b, atol=1e-5), f"{(a - b).abs().max()}")

        # ---- curriculum: epoch 1, only deg moves
        net.train()
        dummy = types.SimpleNamespace(args=types.SimpleNamespace(warmup_bias_lr=0.0), data={"nc": 18})
        opt0 = BaseTrainer.build_optimizer(dummy, net, name="AdamW", lr=1e-3, momentum=0.9, decay=1e-4)
        groups = m.split_groups_by_part(opt0.param_groups, net, DEG_UNFREEZE)
        trainer = types.SimpleNamespace(epoch=0)
        opt = m.attach_unfreeze(type(opt0)(groups, lr=1e-3), trainer, DEG_UNFREEZE)
        before = {n: p.detach().clone() for n, p in net.named_parameters()}
        loss, _ = net.loss(fake_batch(torch, ch=16))
        opt.zero_grad(); loss.backward(); opt.step()
        moved = {}
        for n, p in net.named_parameters():
            pt = parts[id(p)][0]
            moved[pt] = max(moved.get(pt, 0.0), float((p.detach() - before[n]).abs().max()))
        check("DEG_UNFREEZE epoch 1: deg moves, every other part by 0.0",
              moved.get("deg", 0) > 0 and all(v == 0.0 for k, v in moved.items() if k != "deg"),
              str({k: f"{v:.1e}" for k, v in moved.items()}))

    print(f"\n{len(fails)} failure(s)" if fails else "\nall DEGConv checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
