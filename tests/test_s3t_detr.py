"""S3T in front of RT-DETR, exercised through the generated kernel.

Everything that would otherwise only surface hours into a GPU session:
the front and the injections must install on a real rtdetr-l graph, start as a
no-op on top of the plain projection, receive gradient, survive the deepcopy
ultralytics makes for its EMA and checkpoints, fuse for validation, and put no
duplicate keys in the state_dict a resume has to match. The fine-tuning
features must also match what MAE pretraining saw.
"""

from __future__ import annotations

import copy
import json
import pickle
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

fails = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        fails.append(name)


def build_kernel(tmp: Path, cand: dict) -> types.ModuleType:
    cfg = tmp / "round.json"
    cfg.write_text(json.dumps({"round": "s3t", "candidates": [], "submit": {
        "candidate": cand, "use_all_train": False, "predict": False, "session_hours": 1.0}}))
    out = tmp / "build"
    subprocess.run([sys.executable, str(REPO / "tools" / "build_kernel.py"),
                    "--round-config", str(cfg), "--out-dir", str(out), "--slug", "x/s3t"],
                   check=True, capture_output=True)
    src = (out / "hod26_round.py").read_text()
    m = types.ModuleType("s3tk")
    sys.modules["s3tk"] = m
    exec(compile(src, "kernel", "exec"), m.__dict__)
    m.log = lambda *a: None
    return m


def main() -> int:
    import torch
    from hod26.s3t import preprocess as P
    from hod26.s3t.front import level_features
    from hod26.s3t.mae import S3TMAE
    from hod26.s3t.spectral import SpectralEncoder
    from tools.s3t_round import s3t_candidate

    # Fine-tuning features == pretraining features (up to the uint8 quantum).
    rng = np.random.default_rng(0)
    cube = rng.integers(200, 4000, (60, 90, 16)).astype(np.uint16)
    ref = P.features(cube)                                          # (3, 16, H, W)
    x = torch.from_numpy(P.level_u8(cube)).permute(2, 0, 1)[None].float() / 255
    got = level_features(x * P.LEVEL_SPAN + P.LEVEL_LO)[0].numpy()
    err = np.abs(got - ref).max(axis=(1, 2, 3))
    check("detection features match MAE features", bool((err < 0.01).all()), f"max err {err}")

    cand = s3t_candidate(total=2)
    check("candidate: s3t stem, 16 channels, level input",
          cand["train"]["spectral_stem"] == "s3t" and cand["train"]["in_channels"] == 16
          and cand["channels"]["mode"] == "s3t_level", json.dumps(cand["train"])[:200])
    check("candidate: spectral augmentation on, S-G along wavelength",
          cand["augment"]["copies"] >= 1 and cand["augment"]["sg_chain"]
          and cand["augment"]["smote_alpha"] > 0 and cand["augment"]["cutmix_prob"] > 0,
          json.dumps(cand["augment"]))
    check("candidate: ultralytics spatial augmentation on",
          cand["train"]["mosaic"] > 0 and cand["train"]["fliplr"] > 0 and cand["train"]["scale"] > 0,
          json.dumps({k: cand["train"][k] for k in ("mosaic", "fliplr", "scale")}))

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        m = build_kernel(tmp, cand)
        img = m.build_channels(cube, cand["channels"])
        check("renderer emits the aligned uint8 level", img.dtype == np.uint8
              and np.array_equal(img, P.level_u8(cube)))

        # A small MAE checkpoint, as the pretraining kernel writes it.
        enc = SpectralEncoder(dim=32, depth=2, heads=4)
        torch.manual_seed(0)
        for p_ in enc.parameters():
            p_.data.normal_(0, 0.02)
        ck = tmp / "pretrain_mae.pt"
        torch.save({"encoder": enc.state_dict(), "mae": S3TMAE(enc).state_dict(),
                    "config": {"dim": 32, "depth": 2, "heads": 4}, "step": 7}, ck)

        from ultralytics.nn.tasks import RTDETRDetectionModel
        net = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=18, verbose=False)
        ok = m.install_spectral_adapter(net, 16, projection=m.LDA_16_TO_3, kind="s3t",
                                        mae_ckpt=str(ck), scale=0.5)
        check("S3T front installs on rtdetr-l", ok is True)
        front = net.model[0].front
        check("MAE encoder weights loaded",
              all(torch.equal(a, b) for a, b in zip(front.enc.state_dict().values(),
                                                    enc.state_dict().values())))
        check("side injections wrap P3/P4/P5 input projections",
              all(type(net.model[i]).__name__ == "Inject" for i in (19, 14, 10)))
        keys = list(net.state_dict())
        check("state_dict has no duplicated front keys",
              len(keys) == len(set(keys)) and not any(".front.enc" in k and "model.19" in k for k in keys))

        net.eval()
        xb = torch.rand(1, 16, 160, 224)
        with torch.no_grad():
            out = net(xb)
            base = net.model[0].block(front.base(xb))
            plain = net.model[0](xb)
        check("front starts as the plain projection (zero-init head)", torch.allclose(base, plain, atol=1e-6))
        check("detector forward runs on 16-band input", out is not None)

        net.train()
        y = net.predict(xb) if False else net(xb)
        loss = sum(t.float().abs().mean() for t in (y if isinstance(y, (list, tuple)) else [y])
                   if torch.is_tensor(t))
        loss.backward()
        g_head = front.head.weight.grad
        g_inj = net.model[19].proj.weight.grad
        g_enc = front.enc.stem[0].weight.grad
        check("gradient reaches the zero-init head", g_head is not None and g_head.abs().sum() > 0)
        check("gradient reaches the side injection", g_inj is not None and g_inj.abs().sum() > 0)
        check("encoder is in the graph (grad after the head moves)", g_enc is not None)

        cp = copy.deepcopy(net)                 # ultralytics EMA / checkpoint path
        check("deepcopy after a forward (EMA, checkpoints)", "_side" not in cp.model[0].front.__dict__)
        check("the copy's injections point at the copy's front",
              cp.model[19].front is cp.model[0].front)
        blob = pickle.dumps(cp)
        back = pickle.loads(blob)
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

        m.INPUT = tmp
        check("preflight finds the MAE checkpoint", m.find_mae_checkpoint() == ck)

    print("\n".join(fails) if fails else "\nS3T-DETR checks pass")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
