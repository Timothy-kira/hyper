"""The candidate space a discovery node occupies, and the agent that proposes one.

A candidate is a complete, runnable description of one HOD26 solution: how the
16-band cube becomes model input, what is trained, and how predictions are
formed. The exploration policy decides *where* to spend an attempt; the agent
here decides *what* that attempt tries.
"""

from __future__ import annotations

import random
from copy import deepcopy

# Channel builders turn an (H, W, 16) cube into model input. "pseudo_rgb" is the
# organizers' demo and the obvious default; the rest exist because the class list
# pairs visually identical objects by material (apple/apple_plastic,
# egg/egg_plastic/egg_wood, car/car_toy), which only the spectrum separates.
CHANNEL_MODES = [
    "pseudo_rgb",      # bands 0,1,2 -- the demo baseline
    "spread_rgb",      # bands 0,7,15 -- widest spectral spacing in 3 channels
    "pca3",            # first 3 principal components over bands
    "band_stack",      # all 16 bands as input channels
    "rgb_plus_ratio",  # 3 bands + normalized band-ratio channels
]

DEFAULT: dict = {
    "channels": {
        "mode": "pseudo_rgb",
        "bands": [0, 1, 2],
        "stretch_lo": 0.0,       # percentile clip; 0/100 = the demo's min-max
        "stretch_hi": 100.0,
        "per_image_norm": True,  # False = dataset-wide statistics
    },
    "train": {
        "model": "yolo11s",
        "imgsz": 640,
        "epochs": 30,
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
    },
    "infer": {
        "conf": 0.001,           # mAP rewards deep recall, not a clean top-1
        "iou": 0.7,
        "max_det": 300,
        "tta": False,            # allowed: single checkpoint, merged augmentations
        "multi_scale": [],
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
    "train.model": ["yolo11s", "yolo11m", "yolo11l"],
    "train.imgsz": [640, 768, 896, 1024],
    "train.epochs": [20, 30, 45, 60],
    "train.lr0": [0.003, 0.005, 0.01, 0.02],
    "train.mosaic": [0.0, 0.5, 1.0],
    "train.scale": [0.3, 0.5, 0.7],
    "infer.conf": [0.0005, 0.001, 0.005],
    "infer.iou": [0.6, 0.7, 0.8],
    "infer.tta": [True, False],
    "infer.multi_scale": [[], [0.8, 1.0, 1.25]],
}

_BANDS_FOR_MODE = {
    "pseudo_rgb": [0, 1, 2],
    "spread_rgb": [0, 7, 15],
    "pca3": list(range(16)),
    "band_stack": list(range(16)),
    "rgb_plus_ratio": [0, 7, 15],
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
    cfg["channels"]["bands"] = _BANDS_FOR_MODE[cfg["channels"]["mode"]]
    if cfg["channels"]["stretch_hi"] <= cfg["channels"]["stretch_lo"]:
        cfg["channels"]["stretch_hi"] = 100.0
    # band_stack needs 16 input channels, which rules out COCO-pretrained RGB
    # weights unless the stem is re-inflated; flag it so the kernel handles it.
    cfg["train"]["in_channels"] = 16 if cfg["channels"]["mode"] == "band_stack" else 3
    if cfg["fidelity"] == "proxy":
        cfg["train"]["epochs"] = min(cfg["train"]["epochs"], 18)
        cfg["train"]["imgsz"] = min(cfg["train"]["imgsz"], 768)
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

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def propose(self, parent_candidate: dict | None, history: list[dict],
                n_moves: int = 1) -> tuple[dict, str]:
        """Return ``(candidate, rationale)``.

        ``history`` holds ``{"candidate": ..., "score": ...}`` for attempts
        already measured anywhere in the tree.
        """
        if parent_candidate is None:
            return seed_candidate(), "root: organizers' pseudo-RGB demo baseline"

        tried = {self._sig(h["candidate"]) for h in history}
        best = max((h for h in history if h.get("score") is not None),
                   key=lambda h: h["score"], default=None)

        for _ in range(96):
            cfg = deepcopy(parent_candidate)
            picked = []
            for _ in range(n_moves):
                path = self._rng.choice(list(_MOVES))
                options = [o for o in _MOVES[path] if o != _get(cfg, path)]
                if not options:
                    continue
                val = self._rng.choice(options)
                _set(cfg, path, val)
                picked.append(f"{path}={val!r}")
            cfg = normalize(cfg)
            if not picked or self._sig(cfg) in tried:
                continue
            why = "; ".join(picked)
            if best is not None:
                why += f" (best measured so far {best['score']:.4f})"
            return cfg, why

        return normalize(deepcopy(parent_candidate)), "exhausted: repeating parent"

    @staticmethod
    def _sig(cfg: dict) -> tuple:
        flat = []
        for section in ("channels", "train", "infer"):
            for k in sorted(cfg.get(section, {})):
                v = cfg[section][k]
                flat.append((f"{section}.{k}", tuple(v) if isinstance(v, list) else v))
        return tuple(flat)
