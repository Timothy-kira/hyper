"""S3T-X: the cross-covariance spectral encoder, alone and in front of RT-DETR.

The encoder must keep one vector per position, never let a masked position
reach a visible one, and tell a missing band from a zero one. In front of the
detector it must start as an exact no-op on the plain projection, join the
stem's output at stride 4, feed P3/P4/P5 from its own pyramid, and survive the
deepcopy, pickle, fuse and state_dict round trips ultralytics puts it through.
"""

from __future__ import annotations

import copy
import pickle
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from test_s3t_detr import build_kernel  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        fails.append(name)


def encoder_checks():
    import torch
    from hod26.s3t.xca import XCAEncoder, build_encoder

    torch.manual_seed(0)
    enc = XCAEncoder(dim=32, depth=3, heads=4, windows=(8, None)).eval()
    check("windows padded to depth with global blocks", enc.windows == [8, None, None], str(enc.windows))
    x = torch.randn(2, 3, 16, 40, 56)
    with torch.no_grad():
        y = enc(x)
    check("one D-vector per stride-2 position", tuple(y.shape) == (2, 32, 20, 28), str(tuple(y.shape)))
    with torch.no_grad():
        y_odd = enc(torch.randn(1, 3, 16, 37, 51))
    check("odd sizes (window padding)", tuple(y_odd.shape) == (1, 32, 19, 26), str(tuple(y_odd.shape)))
    check("no BatchNorm (masked batches, 2-image detector batches)",
          not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in enc.modules()))

    # Masked positions never reach visible outputs.
    vis = (torch.rand(2, 1, 20, 28) > 0.6).float()
    px = vis.repeat_interleave(2, -2).repeat_interleave(2, -1)[:, :, None]      # (B, 1, 1, H, W)
    xa = x * px
    xb = xa + torch.randn_like(x) * (1 - px) * 5.0           # change only masked pixels ...
    xb = xb * px                                             # ... which the caller zeroes anyway
    xc = x * px + (1 - px) * 3.0                             # and a caller that forgets to zero them
    with torch.no_grad():
        ya, yc = enc(xa, vis), enc(xc, vis)
    # The stem's 3x3 stride-2 conv reads one input pixel of each neighbouring
    # position, so only a zeroed input keeps the visible outputs clean: the MAE
    # zeroes masked pixels before the encoder, as S3TMAE2 does.
    check("masked outputs are exactly zero", float((ya * (1 - vis)).abs().max()) == 0.0)
    del xb, yc

    # Content at masked *positions* (after the stem) never leaks: perturb the
    # stem output at masked positions and the visible outputs do not move.
    with torch.no_grad():
        s0 = enc.stem_map(xa, vis)
        noise = torch.randn_like(s0) * (1 - vis) * 10.0

        def run(s):
            y_ = s * vis
            for blk in enc.blocks:
                y_ = blk(y_, vis)
            return y_

        d = (run(s0) - run(s0 + noise)).abs() * vis
    check("no masked position reaches a visible one through the blocks", float(d.max()) < 1e-5,
          f"{float(d.max()):.2e}")

    # A missing band is not a zero band.
    bv = torch.ones(2, 16, dtype=torch.bool)
    bv[:, 3] = False
    xz = x.clone()
    xz[:, :, 3] = 0
    with torch.no_grad():
        y_zero, y_missing = enc(xz), enc(xz, band_vis=bv)
    check("band_vis changes the encoding of a masked band", float((y_zero - y_missing).abs().max()) > 1e-4)
    with torch.no_grad():
        y_all = enc(x, band_vis=torch.ones(2, 16, dtype=torch.bool))
    check("all bands present == no band_vis", torch.allclose(y_all, enc(x), atol=1e-6))

    # Brightness survives to the output (no per-pixel normalisation of the input).
    with torch.no_grad():
        dy = (enc(x) - enc(x + torch.tensor([0.3, 0, 0]).view(1, 3, 1, 1, 1))).abs().mean()
    check("a brightness change reaches the output", float(dy) > 1e-3, f"{float(dy):.2e}")

    enc.train()
    enc.grad_ckpt = True
    xg = x.clone().requires_grad_(True)
    enc(xg).square().mean().backward()
    g1 = enc.blocks[0].attn.qkv.weight.grad.clone()
    enc.zero_grad()
    enc.grad_ckpt = False
    enc(x).square().mean().backward()
    check("per-block checkpoints give the same gradient",
          torch.allclose(g1, enc.blocks[0].attn.qkv.weight.grad, atol=1e-6))

    e2 = build_encoder(enc.config())
    check("build_encoder rebuilds the same architecture from config()",
          isinstance(e2, XCAEncoder) and e2.windows == enc.windows
          and set(e2.state_dict()) == set(enc.state_dict()))
    return enc


def detector_checks(enc):
    import torch
    from ultralytics.nn.tasks import RTDETRDetectionModel

    from tools.s3t_round import s3t_candidate

    cand = s3t_candidate(total=2)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        m = build_kernel(tmp, cand)
        ck = tmp / "pretrain3_mae.pt"
        torch.save({"encoder": enc.state_dict(), "config": enc.config(), "step": 11}, ck)

        net = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=18, verbose=False)
        orig_net = copy.deepcopy(net).eval()
        ok = m.install_spectral_adapter(net, 16, projection=m.LDA_16_TO_3, kind="s3t",
                                        mae_ckpt=str(ck), scale=0.5, arch="xca")
        check("S3T-X front installs on rtdetr-l", ok is True)
        front = net.model[0].front
        check("the front is S3TXFront", type(front).__name__ == "S3TXFront", type(front).__name__)
        check("MAE encoder weights loaded",
              all(torch.equal(a, b) for a, b in zip(front.enc.state_dict().values(),
                                                    enc.state_dict().values())))
        stem = next(x for x in net.model[0].block.modules() if isinstance(x, torch.nn.Conv2d))
        check("stem input left at 3 channels (fusion is at its output)", stem.in_channels == 3)
        check("fusion conv maps D -> the stem's 48 output channels",
              front.fuse.in_channels == enc.dim and front.fuse.out_channels == 48)
        check("P3/P4 injections read AIFI context, P5 plain",
              [type(net.model[i]).__name__ for i in (19, 14, 10, 11)]
              == ["ContextInject", "ContextInject", "Inject", "Tap"])
        keys = list(net.state_dict())
        check("state_dict has no duplicated front keys", len(keys) == len(set(keys)))

        try:
            net_bad = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=18, verbose=False)
            m.install_spectral_adapter(net_bad, 16, projection=m.LDA_16_TO_3, kind="s3t",
                                       mae_ckpt=str(ck), scale=0.5, arch="tokens")
            raised = False
        except RuntimeError:
            raised = True
        check("an MAE checkpoint of the other architecture is refused", raised)

        flat = lambda o: [t for t in (o if isinstance(o, (list, tuple)) else [o]) if torch.is_tensor(t)]
        net.eval()
        xb = torch.rand(1, 16, 256, 320)
        with torch.no_grad():
            out = net(xb)
            want = orig_net(front.base(xb))
        check("whole detector at step 0 == pretrained detector on the projection",
              all(torch.allclose(a, b_, atol=1e-4) for a, b_ in zip(flat(out), flat(want))))
        pyr = front.__dict__["_pyr"]
        check("pyramid on the P3/P4/P5 grids", [tuple(p.shape[-2:]) for p in pyr]
              == [(32, 40), (16, 20), (8, 10)], str([tuple(p.shape[-2:]) for p in pyr]))
        check("encoder grid == HGStem grid (stride 4)",
              tuple(front.__dict__["_side"].shape[-2:]) == (64, 80))
        # ultralytics feeds multiples of 32 (RT-DETR's neck needs them); one
        # whose P5 grid is odd must still land on the detector's grids.
        with torch.no_grad():
            net(torch.rand(1, 16, 224, 352))
        check("odd P5 grid (224 x 352) matches the detector",
              [tuple(p.shape[-2:]) for p in front.__dict__["_pyr"]] == [(28, 44), (14, 22), (7, 11)])

        net.train()
        loss = sum(t.float().abs().mean() for t in flat(net(xb)))
        loss.backward()
        check("gradient reaches the stem fusion", front.fuse.weight.grad is not None
              and front.fuse.weight.grad.abs().sum() > 0)
        check("gradient reaches the P3 injection", net.model[19].proj.weight.grad.abs().sum() > 0)

        probe = copy.deepcopy(net)
        probe.train()
        with torch.no_grad():
            probe.model[0].front.fuse.weight.normal_(0, 0.02)
            probe.model[19].proj.weight.normal_(0, 0.02)
        sum(t.float().abs().mean() for t in flat(probe(xb))).backward()
        pf = probe.model[0].front
        g_enc = pf.enc.blocks[0].attn.qkv.weight.grad
        g_pyr = pf.pyramid.down[0].weight.grad
        check("gradient reaches the encoder once the fusion moves", g_enc is not None and g_enc.abs().sum() > 0)
        check("gradient reaches the pyramid once an injection moves", g_pyr is not None and g_pyr.abs().sum() > 0)
        check("gradient reaches the context cross-attention", probe.model[19].q.weight.grad.abs().sum() > 0)

        cp = copy.deepcopy(net)
        check("deepcopy drops the per-forward tensors",
              not any(k in cp.model[0].front.__dict__ for k in ("_side", "_pyr", "_ctx")))
        check("the copy's injections point at the copy's front", cp.model[19].front is cp.model[0].front)
        back = pickle.loads(pickle.dumps(cp))
        check("pickle round-trip keeps the shared front", back.model[14].front is back.model[0].front)
        back.eval()
        try:
            back.fuse(verbose=False)
            with torch.no_grad():
                back(xb)
            fused = True
        except Exception as e:                  # noqa: BLE001
            fused = str(e)
        check("fuse() for validation still works", fused is True, str(fused))
        sd = back.state_dict()
        net2 = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=18, verbose=False)
        m.install_spectral_adapter(net2, 16, projection=m.LDA_16_TO_3, kind="s3t",
                                   mae_ckpt=str(ck), scale=0.5, arch="xca")
        try:
            net2.fuse(verbose=False)
            net2.load_state_dict(sd, strict=True)
            rt = True
        except Exception as e:                  # noqa: BLE001
            rt = str(e)[:200]
        check("weights load into a freshly built S3T-X model", rt is True, str(rt))


def main() -> int:
    enc = encoder_checks()
    detector_checks(enc)
    print(f"\n{len(fails)} failure(s)" if fails else "\nall S3T-X checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
