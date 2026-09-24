"""D-FINE's distribution refinement and DEIM's MAL on the pretrained RT-DETR.

The decoder must start as exactly the pretrained one (zero-initialised
distribution heads), the edge-offset encoding must round-trip, the loss must
gain FGL and DDF terms computed on the loss's own matchings (main and
denoising queries), gradient must reach the new heads, and the model must
survive the deepcopy / pickle / state_dict paths of EMA, checkpoints and resume.
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


def fake_batch(torch, b=2, n=5, size=256, ch=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(b * n, 2, generator=g) * 0.6 + 0.2
    wh = torch.rand(b * n, 2, generator=g) * 0.15 + 0.03
    return {"img": torch.rand(b, ch, size, size, generator=g),
            "batch_idx": torch.arange(b).repeat_interleave(n).float(),
            "cls": torch.randint(0, 18, (b * n, 1), generator=g).float(),
            "bboxes": torch.cat([xy, wh], 1)}


def main() -> int:
    import torch
    from ultralytics.nn.tasks import RTDETRDetectionModel

    from tools.s3t_round import s3t_candidate

    cand = s3t_candidate(total=2)
    with tempfile.TemporaryDirectory() as td:
        m = build_kernel(Path(td), cand)

        W = m.fdr_weighting(32, 0.5, 4.0)
        check("W has reg_max + 1 values, odd, ends at +-4",
              W.numel() == 33 and torch.allclose(W, -W.flip(0)) and float(W[-1]) == 4.0
              and bool((W.diff() > 0).all()))
        check("W is densest near zero", float(W[17] - W[16]) < float(W[-1] - W[-2]) / 5)

        box = torch.tensor([[0.4, 0.5, 0.2, 0.1], [0.7, 0.3, 0.05, 0.3]])
        zero = torch.zeros(2, 4 * 33)
        check("uniform logits leave the box unchanged",
              torch.allclose(m.fdr_apply(box, zero, W, 4.0), box, atol=1e-6))
        gt = torch.tensor([[0.41, 0.49, 0.23, 0.12], [0.69, 0.31, 0.06, 0.27]])
        lo, wl, wr = m.fdr_targets(box, gt, W, 4.0)
        logits = torch.full((8, 33), -1e4)
        logits[torch.arange(8), lo] = wl.clamp_min(1e-12).log()
        logits[torch.arange(8), lo + 1] = wr.clamp_min(1e-12).log()
        back = m.fdr_apply(box, logits.reshape(2, 4 * 33), W, 4.0)
        check("target encoding round-trips (expectation of the two bins == gt)",
              torch.allclose(back, gt, atol=1e-4), f"{back} vs {gt}")
        far = torch.tensor([[0.4, 0.5, 0.9, 0.9]])
        lo2, wl2, wr2 = m.fdr_targets(box[:1], far, W, 4.0)
        check("offsets beyond W's range go wholly to the end bin",
              bool(((lo2 == 31) & (wr2 == 1)).any() or ((lo2 == 0) & (wl2 == 1)).any()))

        # ---- decoder: identity at step 0
        torch.manual_seed(0)
        net = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=18, verbose=False)
        ref = copy.deepcopy(net).eval()
        m.install_bbox_loss(net, 18, "GIoU", log_size=True, fdr=True, mal=True)
        dec = net.model[-1].decoder
        check("decoder swapped to FDRDecoder, one head per layer",
              type(dec).__name__ == "FDRDecoder" and len(dec.fdr) == len(dec.layers))
        check("criterion linked to the decoder", net.criterion.__dict__.get("_fdr_decoder") is dec)
        check("MAL installed (full-weight positives, target IoU^1.5)",
              net.criterion.vfl.beta == 1.0 and net.criterion.vfl.target_pow == 1.5)
        x = torch.rand(1, 3, 256, 320)
        net.eval()
        with torch.no_grad():
            a, b_ = net(x), ref(x)
        # Compare the decoder's own boxes and scores: the final (y) output is
        # sorted by score, and a random-init net's near-tied scores reorder on
        # a 1e-7 difference.
        check("eval output at step 0 == pretrained decoder (boxes, scores)",
              torch.allclose(a[1][0], b_[1][0], atol=1e-5) and torch.allclose(a[1][1], b_[1][1], atol=1e-5),
              f"{float((a[1][0] - b_[1][0]).abs().max()):.2e}")

        # ---- loss: FGL and DDF present, finite, gradients reach the heads
        net.train()
        batch = fake_batch(torch)
        total, items = net.loss(batch)
        crit = net.criterion
        # the loss dict itself, for the new terms
        torch.manual_seed(1)
        preds = net.predict(batch["img"], batch=_targets(torch, batch))
        check("training forward leaves the distribution stash for the loss",
              dec.__dict__.get("_fdr") is not None)
        ld = _loss_dict(net, torch, batch, preds)
        check("loss has loss_fgl and loss_ddf", "loss_fgl" in ld and "loss_ddf" in ld, str(sorted(ld)))
        check("  both finite and positive", all(torch.isfinite(ld[k]) and ld[k] > 0
                                               for k in ("loss_fgl", "loss_ddf")),
              f"{float(ld.get('loss_fgl', torch.tensor(-1.0)).detach()):.4f}")
        check("the stash is consumed by one loss", dec.__dict__.get("_fdr") is None)
        net.zero_grad()
        total.backward()
        g = dec.fdr[-1].layers[-1].weight.grad
        check("total loss is finite", bool(torch.isfinite(total)))
        check("gradient reaches the zero-initialised distribution heads", g is not None and g.abs().sum() > 0)
        check("every parameter gets a gradient (DDP)",
              all(p.grad is not None for p in net.parameters() if p.requires_grad),
              str([n for n, p in net.named_parameters() if p.requires_grad and p.grad is None][:5]))

        # FGL's gradient points downhill: one small step on the distribution
        # heads alone, same denoising noise both times, lowers it.
        heads = [p for n, p in net.named_parameters() if ".fdr." in n]

        def fgl_now():
            torch.manual_seed(5)
            return _loss_dict(net, torch, batch, net.predict(batch["img"], batch=_targets(torch, batch)))["loss_fgl"]

        v0 = fgl_now()
        net.zero_grad()
        v0.backward()
        with torch.no_grad():
            for p in heads:
                p -= 0.05 * p.grad
        v1 = fgl_now()
        check("FGL falls along its own gradient", float(v1) < float(v0), f"{float(v0):.4f} -> {float(v1):.4f}")

        # ---- EMA / checkpoint / resume paths
        cp = copy.deepcopy(net)
        check("deepcopy: stash dropped, criterion follows the copy's decoder",
              "_fdr" not in cp.model[-1].decoder.__dict__
              and cp.criterion.__dict__.get("_fdr_decoder") is cp.model[-1].decoder)
        back = pickle.loads(pickle.dumps(cp))
        check("pickle round trip keeps the class and the link",
              type(back.model[-1].decoder).__name__ == "FDRDecoder"
              and back.criterion.__dict__.get("_fdr_decoder") is back.model[-1].decoder)
        back.eval()
        try:
            back.fuse(verbose=False)
            with torch.no_grad():
                back(x)
            ok = True
        except Exception as e:  # noqa: BLE001
            ok = str(e)
        check("fuse() + eval forward", ok is True, str(ok))
        net2 = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=18, verbose=False)
        m.install_bbox_loss(net2, 18, "GIoU", log_size=True, fdr=True, mal=True)
        try:
            net2.load_state_dict(cp.state_dict(), strict=True)
            rt = True
        except Exception as e:  # noqa: BLE001
            rt = str(e)[:200]
        check("state_dict loads into a freshly installed model (resume)", rt is True, str(rt))
        check("W is not in the state_dict (rebuilt on install)",
              not any("fdr_project" in k for k in net.state_dict()))

        # ---- eval loss path (validator): no stash, stock loss, no error
        net.eval()
        with torch.no_grad():
            p_eval = net(batch["img"])
            try:
                net.loss(batch, p_eval)
                ok = True
            except Exception as e:  # noqa: BLE001
                ok = str(e)[:200]
        check("validation loss works without the stash", ok is True, str(ok))

        cand_ok = cand["train"].get("fdr") and cand["train"].get("mal") and cand["train"].get("log_size_l1")
        check("S3T candidate turns on D-FINE FDR, MAL and log-size L1", bool(cand_ok),
              str({k: cand["train"].get(k) for k in ("fdr", "mal", "log_size_l1")}))

    print(f"\n{len(fails)} failure(s)" if fails else "\nall D-FINE checks passed")
    return 1 if fails else 0


def _targets(torch, batch):
    bs = len(batch["img"])
    bi = batch["batch_idx"]
    gt_groups = [(bi == i).sum().item() for i in range(bs)]
    return {"cls": batch["cls"].long().view(-1), "bboxes": batch["bboxes"],
            "batch_idx": bi.long().view(-1), "gt_groups": gt_groups}


def _loss_dict(net, torch, batch, preds):
    targets = _targets(torch, batch)
    dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds
    if dn_meta is None:
        dn_bboxes = dn_scores = None
    else:
        dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
        dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)
    dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])
    dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])
    return net.criterion((dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes,
                         dn_scores=dn_scores, dn_meta=dn_meta)


if __name__ == "__main__":
    raise SystemExit(main())
