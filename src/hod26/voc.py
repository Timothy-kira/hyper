"""Pascal VOC annotation handling for HOD26.

Boxes live in cube coordinates (the XML ``size`` is the cube's W/H/16),
so they need no rescaling when the model consumes decoded cubes.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

# Order is authoritative: it is the competition's class.txt, and its index
# is the class_id expected in the submission.
CLASSES = [
    "apple", "apple_plastic", "badminton", "banana", "banana_plastic",
    "car", "car_toy", "charger_head", "e-bike", "egg",
    "egg_plastic", "egg_wood", "orange", "orange_plastic", "people",
    "rubik", "stone_block", "table_tennis",
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# Classes that a pseudo-RGB composite cannot separate from its partner by
# shape alone — the pairs that make this a *hyperspectral* problem.
MATERIAL_GROUPS = [
    ("apple", "apple_plastic"),
    ("banana", "banana_plastic"),
    ("orange", "orange_plastic"),
    ("egg", "egg_plastic", "egg_wood"),
    ("car", "car_toy"),
]


@dataclass(frozen=True)
class Box:
    cls_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    difficult: bool = False

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


@dataclass(frozen=True)
class Annotation:
    image_id: int
    width: int
    height: int
    depth: int
    boxes: tuple[Box, ...]


def parse(path) -> Annotation:
    """Parse one VOC XML. Unknown class names raise rather than silently drop."""
    path = Path(path)
    root = ET.parse(path).getroot()
    size = root.find("size")
    w, h = int(size.findtext("width")), int(size.findtext("height"))
    depth = int(size.findtext("depth") or 16)

    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in CLASS_TO_ID:
            raise KeyError(f"{path}: unknown class {name!r}")
        bb = obj.find("bndbox")
        # VOC is nominally 1-indexed and inclusive; this dataset is already
        # 0-indexed pixel coordinates, so take them as-is and only clamp.
        x1 = max(0, min(w, int(float(bb.findtext("xmin")))))
        y1 = max(0, min(h, int(float(bb.findtext("ymin")))))
        x2 = max(0, min(w, int(float(bb.findtext("xmax")))))
        y2 = max(0, min(h, int(float(bb.findtext("ymax")))))
        if x2 <= x1 or y2 <= y1:  # zero-area boxes would poison the metric
            continue
        boxes.append(
            Box(CLASS_TO_ID[name], x1, y1, x2, y2,
                bool(int(obj.findtext("difficult") or 0)))
        )
    return Annotation(int(path.stem), w, h, depth, tuple(boxes))
