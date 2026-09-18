"""The candidate space a discovery node occupies, and the agent that proposes one.

A candidate is a complete, runnable description of one HOD26 solution: how the
16-band cube becomes model input, what is trained, and how predictions are
formed. The exploration policy decides *where* to spend an attempt; the agent
here decides *what* that attempt tries.
"""

from __future__ import annotations

import random
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from hod26.spectral import BEST_BANDS  # noqa: E402

# Channel builders turn an (H, W, 16) cube into model input. "pseudo_rgb" is the
# organizers' demo and the obvious default; the rest exist because the class list
# pairs visually identical objects by material (apple/apple_plastic,
# egg/egg_plastic/egg_wood, car/car_toy), which only the spectrum separates.
CHANNEL_MODES = [
    "pseudo_rgb",      # bands 0,1,2 -- the demo baseline
    "spread_rgb",      # bands 0,7,15 -- widest spectral spacing in 3 channels
    "pca3",            # first 3 principal components over bands
    "band_stack",      # all 16 bands as input channels (multi-page TIFF)
    "rgb_plus_ratio",  # 3 bands + normalized band-ratio channels
    "bandsel",         # the 3 bands most separable across material pairs
    "lda3",            # 16 -> 3 discriminant projection, pretrained stem intact
    "bandgroup3",      # 3 contiguous band groups averaged: reduction that also
                       # raises SNR, unlike picking single bands
]

# Transformer detectors. RT-DETR is a hybrid-encoder DETR with IoU-aware query
# selection; ultralytics builds it with ch=data["channels"], so the 16-band
# input path works here exactly as it does for the CNN detectors.
TRANSFORMER_MODELS = ["rtdetr-l", "rtdetr-x", "rtdetr-resnet50", "rtdetr-resnet101"]

# Control track. yolo26-p2 carries an extra high-resolution head, which is the
# variant that matters here: 69.8% of the annotated objects are COCO-small.
YOLO_MODELS = ["yolo26s", "yolo26m", "yolo26-p2", "yolo26l"]

TRACKS = {"transformer": TRANSFORMER_MODELS, "yolo26": YOLO_MODELS}


def is_transformer(model: str) -> bool:
    return model.startswith("rtdetr")


DEFAULT: dict = {
    "channels": {
        "mode": "pseudo_rgb",
        "bands": [0, 1, 2],
        "stretch_lo": 0.0,       # percentile clip; 0/100 = the demo's min-max
        "stretch_hi": 100.0,
        "per_image_norm": True,  # False = dataset-wide statistics
    },
    "train": {
        "model": "rtdetr-l",
        "imgsz": 640,
        "epochs": 10,
        "batch": 16,
        "lr0": 0.01,
        "mosaic": 1.0,
        "close_mosaic": 5,
        "hsv_h": 0.0,            # hue jitter is meaningless on spectral channels
        "hsv_s": 0.4,
        "hsv_v": 0.4,
        "fliplr": 0.5,
        "scale": 0.5,
        "cos_lr": True,
        # Only meaningful when the input is wider than the pretrained stem.
        # "adapter" learns a 48-parameter 16->3 mixer and leaves the pretrained
        # convolution untouched; "seed" reparameterises the stem itself, which
        # starts in the same place but lets all 4608 of its weights drift.
        "spectral_stem": "adapter",
    },
    "infer": {
        "conf": 0.001,           # mAP rewards deep recall, not a clean top-1
        "iou": 0.7,              # NMS IoU; inert for RT-DETR, which is NMS-free
        "max_det": 300,
        "tta": False,            # allowed: single checkpoint, merged augmentations
        "multi_scale": [],
    },
    # The pipeline is searched alongside the model: these operate on the cube,
    # before projection, because the measured bottleneck is spectral and an
    # augmentation applied after projection can only perturb what survived it.
    "augment": {
        "sg_window": 0,          # 0 = off; Savitzky-Golay window along the bands
        "sg_polyorder": 2,
        "smote_alpha": 0.0,      # same-class spectral interpolation strength
        "cutmix_prob": 0.0,      # per-superpixel paste probability
        "cutmix_blocks": 24,
        "copies": 0,             # extra augmented copies per training frame
    },
    "fidelity": "proxy",         # proxy = subsampled/short; full = the real run
}

# Moves the agent may make. Localization dominates this metric (a uniform 3px
# shift costs ~0.75 mAP), so resolution and box-quality knobs are weighted in.
_MOVES: dict[str, list] = {
    "channels.mode": CHANNEL_MODES,
    "channels.stretch_lo": [0.0, 0.5, 1.0, 2.0],
    "channels.stretch_hi": [98.0, 99.0, 99.5, 100.0],
    "channels.per_image_norm": [True, False],
    "train.model": TRANSFORMER_MODELS + YOLO_MODELS,
    "train.imgsz": [640, 768, 896, 1024],
    "train.epochs": [6, 10, 14, 20],
    "train.lr0": [0.003, 0.005, 0.01, 0.02],
    "train.mosaic": [0.0, 0.5, 1.0],
    "train.scale": [0.3, 0.5, 0.7],
    "infer.conf": [0.0005, 0.001, 0.005],
    "infer.iou": [0.6, 0.7, 0.8],
    "infer.tta": [True, False],
    "infer.multi_scale": [[], [0.8, 1.0, 1.25]],
    "augment.sg_window": [0, 5, 7, 9],
    "augment.sg_polyorder": [2, 3],
    "augment.smote_alpha": [0.0, 0.3, 0.5],
    "augment.cutmix_prob": [0.0, 0.3, 0.5],
    "augment.cutmix_blocks": [16, 24, 40],
    "augment.copies": [0, 1, 2],
    "train.spectral_stem": ["adapter", "seed"],
}

_BANDS_FOR_MODE = {
    "pseudo_rgb": [0, 1, 2],
    "spread_rgb": [0, 7, 15],
    "pca3": list(range(16)),
    "band_stack": list(range(16)),
    "rgb_plus_ratio": [0, 7, 15],
    "bandsel": list(BEST_BANDS),
    "lda3": list(range(16)),
    "bandgroup3": list(range(16)),
}


def _get(cfg: dict, path: str):
    node = cfg
    for k in path.split("."):
        node = node[k]
    return node


def _set(cfg: dict, path: str, value) -> None:
    keys = path.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def normalize(cfg: dict) -> dict:
    """Repair a candidate into a runnable state after mutation."""
    cfg = deepcopy(cfg)
    aug = cfg.setdefault("augment", deepcopy(DEFAULT["augment"]))
    # Savitzky-Golay needs an odd window strictly greater than the polynomial
    # order, or the least-squares fit is not determined.
    if aug["sg_window"]:
        aug["sg_window"] = max(3, aug["sg_window"] | 1)
        # Order must stay below the window, but an order near it fits the window
        # exactly and smooths nothing -- cap it where the filter still filters.
        aug["sg_polyorder"] = max(1, min(aug["sg_polyorder"], aug["sg_window"] - 2, 4))
    # Copies only pay for themselves if something actually varies between them.
    if not (aug["sg_window"] or aug["smote_alpha"] or aug["cutmix_prob"]):
        aug["copies"] = 0
    cfg["channels"]["bands"] = _BANDS_FOR_MODE[cfg["channels"]["mode"]]
    if cfg["channels"]["stretch_hi"] <= cfg["channels"]["stretch_lo"]:
        cfg["channels"]["stretch_hi"] = 100.0
    # Ultralytics builds the model with ch=data["channels"], so 16-band input
    # needs no patching -- but COCO weights cannot transfer into a 16-channel
    # stem, so that one layer trains from scratch while the rest is pretrained.
    cfg["train"]["in_channels"] = 16 if cfg["channels"]["mode"] == "band_stack" else 3
    if is_transformer(cfg["train"]["model"]):
        # Ultralytics warns that AMP can produce NaNs during RT-DETR's bipartite
        # matching, and that grid_sample rejects deterministic mode.
        cfg["train"]["amp"] = False
        cfg["train"]["deterministic"] = False
        # Top-k selection happens inside the decoder, so there is no NMS to tune.
        cfg["infer"]["iou"] = None
    else:
        cfg["train"]["amp"] = True
        cfg["train"]["deterministic"] = True
    if cfg["train"]["in_channels"] != 3:
        # Ultralytics skips HSV on non-3-channel input anyway; make it explicit.
        cfg["train"]["hsv_h"] = cfg["train"]["hsv_s"] = cfg["train"]["hsv_v"] = 0.0
    if cfg["fidelity"] == "proxy":
        cfg["train"]["epochs"] = min(cfg["train"]["epochs"], 12)
        cfg["train"]["imgsz"] = min(cfg["train"]["imgsz"], 768)
    # Ultralytics disables mosaic for the last close_mosaic epochs; that is
    # meaningless once it exceeds the run length.
    cfg["train"]["close_mosaic"] = min(cfg["train"]["close_mosaic"],
                                       max(0, cfg["train"]["epochs"] - 1))
    return cfg


def seed_candidate(**overrides) -> dict:
    cfg = deepcopy(DEFAULT)
    for path, val in overrides.items():
        _set(cfg, path.replace("__", "."), val)
    return normalize(cfg)


class DiscoveryAgent:
    """Proposes the next candidate from a parent and what its siblings measured.

    Mirrors the paper's discovery agent: it resumes a parent's workspace and
    uses the accumulated observations as context. Sibling results are read so a
    branch does not re-try a move a neighbour already measured as bad.
    """

    def __init__(self, seed: int = 0, track: str | None = None):
        self._rng = random.Random(seed)
        # A track pins the model family so the two can be compared as tracks
        # rather than letting one crowd the other out of a shared search.
        self._models = TRACKS.get(track) if track else None

    def _pin_track(self, cfg: dict) -> dict:
        """Keep a candidate inside this agent's model family."""
        if self._models and cfg["train"]["model"] not in self._models:
            cfg["train"]["model"] = self._models[0]
        return normalize(cfg)

    def propose(self, parent_candidate: dict | None, history: list[dict],
                n_moves: int = 1) -> tuple[dict, str]:
        """Return ``(candidate, rationale)``.

        ``history`` holds ``{"candidate": ..., "score": ...}`` for attempts
        already measured anywhere in the tree.
        """
        tried = {self._sig(h["candidate"]) for h in history}

        if parent_candidate is None:
            # Expanding the root opens an *independent* workspace, so it must
            # land somewhere new. Only the very first attempt is the untouched
            # baseline -- that is the reading the tree measures every gain
            # against.
            if not history:
                return (self._pin_track(seed_candidate()),
                        "root: organizers' pseudo-RGB demo baseline")

            # Channel construction is the axis the bottleneck analysis points at:
            # the largest recoverable loss is material discrimination, which no
            # amount of training-parameter tuning can address. One move in
            # fourteen would reach a given mode about one time in a hundred, so
            # a new branch covers an untried mode first and only then wanders.
            seen_modes = {h["candidate"]["channels"]["mode"] for h in history}
            untried = [m for m in CHANNEL_MODES if m not in seen_modes]
            if untried:
                mode = self._rng.choice(untried)
                cfg = self._pin_track(seed_candidate(channels__mode=mode))
                if self._sig(cfg) not in tried:
                    return cfg, (f"root: channel mode {mode!r}, untried "
                                 f"({len(untried) - 1} modes still unexplored)")

            parent_candidate = self._pin_track(seed_candidate())
            n_moves = max(n_moves, 2)
        best = max((h for h in history if h.get("score") is not None),
                   key=lambda h: h["score"], default=None)

        # Covering channel modes cannot depend on the policy opening new
        # branches: with the default two, a rollout measures two modes and then
        # refines them forever, which structurally locks out the mode the
        # bottleneck analysis points at. So an untried mode can also arrive as a
        # child -- same training recipe, different channels. That is a
        # controlled ablation, and a better comparison than pairing each mode
        # with its own random hyperparameters.
        seen_modes = {h["candidate"]["channels"]["mode"] for h in history}
        untried = [m for m in CHANNEL_MODES if m not in seen_modes]
        if untried and self._rng.random() < 0.6:
            cfg = deepcopy(parent_candidate)
            mode = self._rng.choice(untried)
            _set(cfg, "channels.mode", mode)
            cfg = self._pin_track(cfg)
            if self._sig(cfg) not in tried:
                why = f"channels.mode={mode!r} (untried; rest held from parent)"
                if best is not None:
                    why += f" (best measured so far {best['score']:.4f})"
                return cfg, why

        for _ in range(192):
            cfg = deepcopy(parent_candidate)
            picked = []
            for _ in range(n_moves):
                usable = [k for k in _MOVES
                          if not (k == "infer.iou" and is_transformer(cfg["train"]["model"]))]
                path = self._rng.choice(usable)
                choices = self._models if (path == "train.model" and self._models) \
                    else _MOVES[path]
                options = [o for o in choices if o != _get(cfg, path)]
                if not options:
                    continue
                val = self._rng.choice(options)
                _set(cfg, path, val)
                picked.append(f"{path}={val!r}")
            cfg = self._pin_track(cfg)
            if not picked or self._sig(cfg) in tried:
                continue
            why = "; ".join(picked)
            if best is not None:
                why += f" (best measured so far {best['score']:.4f})"
            return cfg, why

        return self._pin_track(deepcopy(parent_candidate)), "exhausted: repeating parent"

    @staticmethod
    def _sig(cfg: dict) -> tuple:
        flat = []
        for section in ("channels", "train", "infer", "augment"):
            for k in sorted(cfg.get(section, {})):
                v = cfg[section][k]
                flat.append((f"{section}.{k}", tuple(v) if isinstance(v, list) else v))
        return tuple(flat)
