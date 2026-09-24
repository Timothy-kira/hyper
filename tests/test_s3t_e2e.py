"""The whole S3T-X detection session on CPU, on a miniature dataset.

run_submission end to end, the way the Kaggle kernel runs it: render the
16-band frames with augmentation, build RT-DETR-L with the S3T-X front (MAE v3
checkpoint mounted under INPUT), D-FINE's distribution refinement and MAL,
train one epoch through ultralytics' trainer (EMA, validation, checkpoint
saving), keep best.pt and last.pt in the output, reload best.pt and write a
submission. Every stage that only ran on a GPU before now runs here first.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from test_datapath import load_kernel  # noqa: E402


def synth_dataset(dest: Path, n_train: int = 12, n_test: int = 3, h: int = 241, w: int = 493) -> Path:
    """Band-planar frames with a few spectrally distinct boxes, and their VOC XML."""
    import numpy as np
    from PIL import Image

    from hod26.cube import to_planar
    from hod26.voc import CLASSES

    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("test", n_test)):
        (dest / split / "images").mkdir(parents=True, exist_ok=True)
        if split == "train":
            (dest / split / "annotations").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            pid = (1000 if split == "train" else 2000) + i
            cube = rng.integers(300, 900, (h, w, 16)).astype(np.uint16)
            objs = []
            for _ in range(int(rng.integers(2, 5))):
                bw, bh = int(rng.integers(12, 60)), int(rng.integers(12, 60))
                x1, y1 = int(rng.integers(0, w - bw)), int(rng.integers(0, h - bh))
                c = int(rng.integers(0, len(CLASSES)))
                cube[y1:y1 + bh, x1:x1 + bw] = (rng.integers(1500, 3500, 16) * (1 + c / 18)).astype(np.uint16)
                objs.append((CLASSES[c], x1, y1, x1 + bw, y1 + bh))
            Image.fromarray(to_planar(cube)).save(dest / split / "images" / f"{pid}.png", compress_level=1)
            if split == "train":
                body = "".join(
                    f"<object><name>{nm}</name><difficult>0</difficult><bndbox><xmin>{a}</xmin>"
                    f"<ymin>{b}</ymin><xmax>{c_}</xmax><ymax>{d}</ymax></bndbox></object>"
                    for nm, a, b, c_, d in objs)
                (dest / split / "annotations" / f"{pid}.xml").write_text(
                    f"<annotation><filename>{pid}.png</filename><size><width>{w}</width>"
                    f"<height>{h}</height><depth>16</depth></size>{body}</annotation>")
    return dest


class _Emulated(Exception):
    pass


def ddp_worker_emulation(check):
    """The training process of a 2-GPU run, on CPU, exactly as ultralytics 8.4 makes it.

    ultralytics builds the model in the parent (Model.train -> trainer.get_model),
    cloudpickles {trainer class, args, model, callbacks} to each DDP worker, and
    the worker constructs a fresh trainer, assigns the unpickled model, runs
    setup_model, wraps it in DistributedDataParallel and only then calls
    build_optimizer. Both GPU-only failures so far lived on that path (lost
    compilation and cudnn.benchmark; a DDP wrapper handed to ensure_accel), and a
    single-process run never takes it. Here the production run_candidate builds
    the trainer class and the parent side for real; trainer.train() is replaced
    by the worker side, with a real single-process gloo DDP.
    """
    import os
    import torch
    import torch.distributed as dist
    import cloudpickle
    from ultralytics.utils import DEFAULT_CFG_DICT

    from hod26.s3t.xca import XCAEncoder
    from tools.s3t_round import s3t_candidate

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        data = synth_dataset(tmp / "input" / "hod26-planar")
        k = load_kernel(tmp / "work", data)
        k.INPUT = tmp / "input"
        mae_dir = k.INPUT / "hod26-s3t-mae-pretrain3" / "s3t_mae"
        mae_dir.mkdir(parents=True)
        enc = XCAEncoder()
        torch.save({"encoder": enc.state_dict(), "config": enc.config(), "step": 1},
                   mae_dir / "pretrain3_mae.pt")
        cand = s3t_candidate(total=1, batch=2, mae_file="pretrain3_mae.pt", arch="xca")
        # the production flags, compile included; CPU only drops AMP
        cand["train"].update(epochs=1, imgsz=128, amp=False, workers=0)
        seen = {}
        factory = k.hod26_trainer

        def emulating_factory(*a, **kw):
            cls = factory(*a, **kw)

            class Parent(cls):
                def train(self):                           # the parent stops here and spawns
                    blob = cloudpickle.dumps({"trainer": cls, "args": vars(self.args),
                                              "model": self.model, "callbacks": self.callbacks})
                    # The worker is another interpreter: it can import this
                    # script only as hod26_kernel, unpickles with cloudpickle and
                    # saves checkpoints with plain pickle (ultralytics save_model).
                    import subprocess
                    import sys as _sys
                    blobf = tmp / "ddp_state.pt"
                    blobf.write_bytes(blob)
                    kdir = tmp / "kmod"
                    kdir.mkdir(exist_ok=True)
                    (kdir / "hod26_kernel.py").write_text(Path(k.__file__).read_text())
                    code = (
                        "import sys, io, copy\n"
                        f"sys.path[:0] = [{str(kdir)!r}] + {_sys.path!r}\n"
                        "import torch, cloudpickle, types, importlib\n"
                        # In production the kernel is __main__ and cloudpickle sends
                        # its functions by value; here it is a named module, so give
                        # the worker its functions under that name -- but not its
                        # classes, which must resolve through hod26_kernel.
                        "hk = importlib.import_module('hod26_kernel')\n"
                        f"shim = types.ModuleType({k.__name__!r})\n"
                        "[setattr(shim, n, v) for n, v in vars(hk).items()"
                        " if callable(v) and not isinstance(v, type)]\n"
                        f"sys.modules[{k.__name__!r}] = shim\n"
                        f"st = cloudpickle.loads(open({str(blobf)!r}, 'rb').read())\n"
                        "m = st['model']\n"
                        "buf = io.BytesIO(); torch.save({'ema': copy.deepcopy(m).half()}, buf); buf.seek(0)\n"
                        "back = torch.load(buf, map_location='cpu', weights_only=False)['ema']\n"
                        "print('WORKER_SAVE_OK', type(back.model[0].front).__name__,"
                        " type(back.model[-1].decoder).__name__)\n")
                    r = subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True,
                                       timeout=900)
                    seen["worker_save"] = r.stdout + r.stderr[-1500:]
                    state = cloudpickle.loads(blob)
                    cfg = DEFAULT_CFG_DICT.copy()
                    cfg.update(save_dir="")
                    w = state["trainer"](cfg=cfg, overrides=state["args"], _callbacks=state["callbacks"])
                    w.model = state["model"]
                    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
                    os.environ.setdefault("MASTER_PORT", "29533")
                    dist.init_process_group("gloo", rank=0, world_size=1)
                    try:
                        w.setup_model()
                        w.model = torch.nn.parallel.DistributedDataParallel(w.model, find_unused_parameters=True)
                        w.build_optimizer(model=w.model, name="auto", lr=1e-4, momentum=0.9,
                                          decay=1e-5, iterations=1000)
                        seen["table"] = w.__dict__.get("_hod26_accel_table") or []
                        g = torch.Generator().manual_seed(0)
                        n = 3
                        batch = {"img": torch.rand(2, 16, 128, 128, generator=g),
                                 "batch_idx": torch.arange(2).repeat_interleave(n).float(),
                                 "cls": torch.randint(0, 18, (2 * n, 1), generator=g).float(),
                                 "bboxes": torch.cat([torch.rand(2 * n, 2, generator=g) * 0.6 + 0.2,
                                                      torch.rand(2 * n, 2, generator=g) * 0.2 + 0.05], 1)}
                        loss, _ = w.model(batch)
                        loss.sum().backward()
                        seen["loss"] = float(loss.sum())
                    finally:
                        dist.destroy_process_group()
                    raise _Emulated()

            return Parent

        k.hod26_trainer = emulating_factory
        said = []
        base_log = k.log
        k.log = lambda msg: (said.append(str(msg)), base_log(msg))
        root = k.data_root()
        train_ids, val_ids = k.submission_split(root, {"use_all_train": False})
        ann = root / "train" / "annotations"
        anns = {p: k.parse(ann / f"{p}.xml") for p in train_ids + val_ids}
        try:
            k.run_candidate(cand, k.frame_index(root, "train", train_ids + val_ids),
                            train_ids, val_ids, anns, "ddpemu")
            err = "trainer.train() was never reached"
        except _Emulated:
            err = None
        except Exception as e:                                  # noqa: BLE001
            import traceback
            err = "".join(traceback.format_exception(e))[-1500:]
        check("DDP worker path (pickled model, DDP wrapper): setup_model + build_optimizer "
              "+ a training step", err is None, str(err))
        # Read from the worker's own trainer: its class travelled by value, so
        # a log hook set here would only see the parent's copy.
        table = seen.get("table") or []
        want = ("loss + Hungarian", "RT-DETR attention via SDPA", "S3T blocks torch.compile",
                "fused AdamW", "cudnn.benchmark", "mosaic canvas reuse")
        off = [w_ for w_ in want if not any(on and w_ in n for n, on, _ in table)]
        check("DDP worker: every requested acceleration ON in the worker's table", not off, str(off))
        check("DDP worker (separate process): unpickles the model and saves a checkpoint "
              "with plain pickle", "WORKER_SAVE_OK S3TXFront FDRDecoder" in seen.get("worker_save", ""),
              seen.get("worker_save", "")[-1500:])
        check("DDP worker: finite loss through the DDP-wrapped model",
              seen.get("loss") is not None and seen["loss"] == seen["loss"], str(seen.get("loss")))


def smoke_checks(check):
    """run_smoke: passes and cleans up under its limit, stops the session over it."""
    import torch

    from hod26.s3t.xca import XCAEncoder
    from tools.s3t_round import s3t_candidate

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        data = synth_dataset(tmp / "input" / "hod26-planar")
        k = load_kernel(tmp / "work", data)
        k.INPUT = tmp / "input"
        mae_dir = k.INPUT / "hod26-s3t-mae-pretrain3" / "s3t_mae"
        mae_dir.mkdir(parents=True)
        enc = XCAEncoder()
        torch.save({"encoder": enc.state_dict(), "config": enc.config(), "step": 1},
                   mae_dir / "pretrain3_mae.pt")
        cand = s3t_candidate(total=5, batch=2, mae_file="pretrain3_mae.pt", arch="xca")
        cand["train"].update(imgsz=128, amp=False, s3t_compile=False, workers=0)
        root = k.data_root()
        tr, va = k.submission_split(root, {"use_all_train": False})
        ann = root / "train" / "annotations"
        anns = {p: k.parse(ann / f"{p}.xml") for p in tr + va}
        index = k.frame_index(root, "train", tr + va)
        said = []
        base_log = k.log
        k.log = lambda msg: (said.append(str(msg)), base_log(msg))
        try:
            k.run_smoke(cand, index, tr, va, anns, {"smoke_max_s_per_it": 1e9})
            ok = True
        except Exception as e:                                  # noqa: BLE001
            ok = str(e)[:300]
        check("smoke passes under its limit", ok is True and any(m.startswith("SMOKE ok") for m in said), str(ok))
        check("  and leaves nothing behind (dataset, run, output files)",
              not (k.SCRATCH / "ds_smoke").exists() and not (k.RUNS / "smoke").exists()
              and not list(k.WORK.glob("smoke_*")))
        check("  and does not touch the real run's epochs", cand["train"]["epochs"] == 5)
        try:
            k.run_smoke(cand, index, tr, va, anns, {"smoke_max_s_per_it": 1e-6})
            stopped = False
        except RuntimeError as e:
            stopped = "limit" in str(e)
        check("smoke over its s/it limit stops the session (no fallback)",
              stopped and any(m.startswith("SMOKE FAILED") for m in said))

        # smoke_only: every stage small, then stop
        said.clear()
        sub = {"candidate": cand, "use_all_train": False, "predict": True, "session_hours": 1.0,
               "smoke_only": True, "smoke_max_s_per_it": 1e9}
        try:
            k.run_submission({"round": "smoke", "candidates": [], "submit": sub})
            ok = True
        except Exception as e:                                  # noqa: BLE001
            ok = str(e)[:300]
        check("smoke_only: train / val / best+last / final eval / reload / predict / submission",
              ok is True and any(m.startswith("SMOKE ALL OK") for m in said), str(ok))
        check("  and it stops there: no long run, nothing of the smoke left behind",
              not (k.WORK / "final_last.pt").exists() and not list(k.WORK.glob("smoke_*"))
              and json.loads((k.WORK / "results.json").read_text()).get("mode") == "smoke_only")


def mosaic_check(check):
    """The kernel's mosaic canvas reuse: installed, and byte-identical to ultralytics'."""
    import copy

    import numpy as np
    from ultralytics.cfg import get_cfg
    from ultralytics.data.augment import Mosaic
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models.rtdetr.train import RTDETRDataset

    from tools.s3t_round import s3t_candidate

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        data = synth_dataset(tmp / "input" / "hod26-planar", n_train=16)
        k = load_kernel(tmp / "work", data)
        check("mosaic canvas reuse installed at import", k.mosaic_canvas_patched())
        cand = s3t_candidate(total=1, arch="xca")
        root = k.data_root()
        tr, va = k.submission_split(root, {"use_all_train": False})
        ann = root / "train" / "annotations"
        anns = {p: k.parse(ann / f"{p}.xml") for p in tr + va}
        yaml = k.materialize(cand, k.frame_index(root, "train", tr + va), tr, va, anns, tmp / "ds")
        d = check_det_dataset(str(yaml))
        args = get_cfg(overrides=dict(imgsz=256, mosaic=1.0, hsv_h=0, hsv_s=0, hsv_v=0))
        ds = RTDETRDataset(img_path=d["train"], imgsz=256, batch_size=2, augment=True, hyp=args,
                           rect=False, cache=False, single_cls=False, prefix="", classes=None,
                           data=d, fraction=1.0)
        patched = Mosaic.apply_image
        orig = patched._hod26_orig
        cap = []

        def capture(self, labels, params=None):
            cap.append((self, copy.deepcopy(labels), copy.deepcopy(params)))
            return orig(self, labels, params)

        Mosaic.apply_image = capture
        try:
            for i in range(5):
                ds[i]
        finally:
            Mosaic.apply_image = patched
        same = []
        for self_, lab, par in cap:
            a = orig(self_, copy.deepcopy(lab), par)["img"].copy()
            b = patched(self_, copy.deepcopy(lab), par)["img"].copy()
            c = patched(self_, copy.deepcopy(lab), par)["img"]        # the buffer reused
            same.append(np.array_equal(a, b) and np.array_equal(a, c))
        check("mosaic canvas reuse: byte-identical to ultralytics' mosaic (buffer reused twice)",
              bool(cap) and all(same), f"{sum(same)}/{len(cap)}")


def main() -> int:
    import torch

    from hod26.s3t.xca import XCAEncoder
    from tools.s3t_round import s3t_candidate

    fails = []

    def check(name, ok, detail=""):
        print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))
        if not ok:
            fails.append(name)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        data = synth_dataset(tmp / "input" / "hod26-planar")
        k = load_kernel(tmp / "work", data)
        k.INPUT = tmp / "input"
        mae_dir = k.INPUT / "hod26-s3t-mae-pretrain3" / "s3t_mae"
        mae_dir.mkdir(parents=True)
        enc = XCAEncoder()
        torch.save({"encoder": enc.state_dict(), "config": enc.config(), "step": 1},
                   mae_dir / "pretrain3_mae.pt")

        cand = s3t_candidate(total=1, batch=2, mae_file="pretrain3_mae.pt", arch="xca")
        tr = cand["train"]
        # CPU: no AMP, no compile; everything else as the real run.
        # nbs = batch: an optimizer step every batch. At the real nbs of 64 a
        # run this short makes one step, at warmup step 0 where every lr is 0.
        tr.update(epochs=2, imgsz=128, amp=False, s3t_compile=False, workers=0, nbs=2)
        # The real schedule, compressed into two epochs: heads and the new
        # modules from the first, everything but the stem from the second.
        tr["unfreeze"] = {**{pt: [0, 0, 1.0] for pt in ("head", "new", "mixer")},
                          **{pt: [1, 1, 1.0] for pt in ("decoder", "neck", "s3t_enc")},
                          "backbone": [1, 1, 0.1]}
        round_cfg = {"round": "e2e", "candidates": [], "submit": {
            "candidate": cand, "use_all_train": False, "predict": True, "session_hours": 1.0}}
        said_run = []
        base_log = k.log
        k.log = lambda msg: (said_run.append(str(msg)), base_log(msg))
        k.run_submission(round_cfg)

        work = k.WORK
        check("submission.csv written", (work / "submission.csv").exists())
        res = json.loads((work / "results.json").read_text())
        check("results.json says predicted", res.get("predicted") is True, str(res)[:200])
        check("final_last.pt kept in the output", (work / "final_last.pt").exists())
        check("final_best.pt kept in the output", (work / "final_best.pt").exists())
        ck = torch.load(work / "final_best.pt", map_location="cpu", weights_only=False)
        net = ck.get("ema") or ck.get("model")
        dec = net.model[-1].decoder
        front = net.model[0].front
        check("best.pt carries the S3T-X front", type(front).__name__ == "S3TXFront",
              type(front).__name__)
        check("best.pt carries the D-FINE decoder", type(dec).__name__ == "FDRDecoder",
              type(dec).__name__)
        moved = float(front.fuse.weight.abs().sum())
        check("the zero-initialised stem fusion trained", moved > 0, f"{moved}")
        fdr_moved = float(sum(m.layers[-1].weight.abs().sum() for m in dec.fdr))
        check("the zero-initialised distribution heads trained", fdr_moved > 0, f"{fdr_moved}")
        inj = float(net.model[19].proj.weight.abs().sum())
        check("the zero-initialised P3 injection trained", inj > 0, f"{inj}")
        enc_w = torch.load(k.INPUT / "hod26-s3t-mae-pretrain3" / "s3t_mae" / "pretrain3_mae.pt",
                           weights_only=True)["encoder"]["blocks.0.attn.qkv.weight"]
        d_enc = float((front.enc.blocks[0].attn.qkv.weight.float() - enc_w).abs().max())
        check("the MAE encoder is fine-tuned once its stage comes", d_enc > 0, f"{d_enc}")
        check("BatchNorm frozen in the saved model",
              all(type(b).__name__ == "FrozenBatchNorm2d"
                  for b in net.modules() if isinstance(b, torch.nn.BatchNorm2d)))
        check("staged unfreezing announced, per-epoch multipliers logged",
              any("staged unfreezing" in m for m in said_run)
              and any(m.startswith("  unfreeze epoch 2:") and "backbone 0.1" in m for m in said_run),
              str([m for m in said_run if "unfreeze" in m][:4]))
        check("the loader runs at imgsz, the model upsamples on the GPU",
              any("S3T-X input: loader at 1/2" in m for m in said_run))
        opt = torch.load(work / "final_last.pt", map_location="cpu", weights_only=False)["optimizer"]
        check("the optimizer is really fused (every group)",
              all(g.get("fused") is True for g in opt["param_groups"]),
              str([g.get("fused") for g in opt["param_groups"]]))
        lines = [ln for ln in (work / "final_metrics.jsonl").read_text().splitlines() if ln.strip()]
        check("per-epoch metrics logged", len(lines) >= 1)

    # ---- the render notebook, then a training session that mounts its output
    with tempfile.TemporaryDirectory() as td:
        import os
        import shutil
        import stat
        tmp = Path(td)
        data = synth_dataset(tmp / "input" / "hod26-planar")
        cand = s3t_candidate(total=1, batch=2, mae_file="pretrain3_mae.pt", arch="xca")
        cand["train"].update(epochs=2, imgsz=128, amp=False, s3t_compile=False, workers=0, nbs=2)
        sub = {"candidate": cand, "use_all_train": False, "predict": True, "session_hours": 1.0}

        r = load_kernel(tmp / "render_work", data)
        r.run_render({"round": "render", "candidates": [], "submit": dict(sub, render_only=True)})
        ds = next((tmp / "render_work").glob("ds_*"))
        check("render_only writes the dataset and its manifest", (ds / "render_manifest.json").exists()
              and (ds / "data.yaml").exists())
        mount = tmp / "input" / "hod26-s3t-render" / ds.name
        mount.parent.mkdir(parents=True)
        shutil.copytree(ds, mount)
        for dirpath, dirnames, filenames in os.walk(mount):     # read-only, like /kaggle/input
            for f in filenames:
                os.chmod(Path(dirpath) / f, stat.S_IRUSR | stat.S_IRGRP)
            os.chmod(dirpath, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)

        k = load_kernel(tmp / "work", data)
        k.INPUT = tmp / "input"
        mae_dir = k.INPUT / "hod26-s3t-mae-pretrain3" / "s3t_mae"
        mae_dir.mkdir(parents=True)
        enc = XCAEncoder()
        torch.save({"encoder": enc.state_dict(), "config": enc.config(), "step": 1},
                   mae_dir / "pretrain3_mae.pt")
        said = []
        base_log = k.log
        k.log = lambda msg: (said.append(str(msg)), base_log(msg))
        pre = k.find_prerendered(cand, *k.submission_split(k.data_root(), sub))
        check("the mounted render is found and matches", pre == mount, str(pre))
        k.run_submission({"round": "e2e", "candidates": [], "submit": sub})
        check("training used the mounted render (no rendering in the session)",
              any("using prerendered dataset" in m for m in said)
              and not any("rendering:" in m for m in said))
        check("  and still trained, kept best/last and predicted",
              (k.WORK / "final_best.pt").exists() and (k.WORK / "final_last.pt").exists()
              and (k.WORK / "submission.csv").exists())

        other = dict(cand, augment=dict(cand["augment"], copies=2))
        try:
            k.find_prerendered(other, *k.submission_split(k.data_root(), sub))
            refused = False
        except RuntimeError:
            refused = False
        # a different augment is a different channels_key: not this render at all
        check("a different configuration does not pick this render up",
              k.find_prerendered(other, *k.submission_split(k.data_root(), sub)) is None)
        man = json.loads((mount / "render_manifest.json").read_text())
        try:
            k.find_prerendered(cand, *k.submission_split(k.data_root(), dict(sub, use_all_train=True)))
            refused = False
        except RuntimeError:
            refused = True
        check("same channels but another split: refused, not silently used", refused, man["key"])
        for dirpath, dirnames, filenames in os.walk(mount):
            os.chmod(dirpath, 0o755)

    ddp_worker_emulation(check)
    smoke_checks(check)
    mosaic_check(check)

    print(f"\n{len(fails)} failure(s)" if fails else "\nS3T-X end-to-end session passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
