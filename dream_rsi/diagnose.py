"""Where the score is actually being lost, and what to read about it.

Ranking candidates tells you which is better; it does not tell you what is
holding all of them back. This reads a discovery tree's per-class results
against the dataset's own statistics and names the bottleneck, so the search --
and the literature it draws on -- can be aimed rather than broadcast.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from hod26.voc import CLASSES, MATERIAL_GROUPS, parse  # noqa: E402

COCO_SMALL = 32 ** 2      # area thresholds the metric itself is stratified by
COCO_LARGE = 96 ** 2


@dataclass
class ClassStat:
    name: str
    count: int
    median_area: float
    ap: float | None = None

    @property
    def median_side(self) -> float:
        return float(np.sqrt(self.median_area))

    @property
    def is_small(self) -> bool:
        return self.median_area < COCO_SMALL


@dataclass
class Finding:
    """One bottleneck, with the evidence that identified it."""

    name: str
    severity: float                  # macro-AP points recoverable, roughly
    evidence: str
    search_terms: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"[{self.severity:+.3f}] {self.name}\n    {self.evidence}"


def dataset_stats(ann_dir) -> dict[str, ClassStat]:
    """Per-class counts and median box area, straight from the annotations."""
    counts: dict[str, int] = defaultdict(int)
    areas: dict[str, list[float]] = defaultdict(list)
    for path in sorted(Path(ann_dir).glob("*.xml")):
        for b in parse(path).boxes:
            name = CLASSES[b.cls_id]
            counts[name] += 1
            areas[name].append(b.area)
    return {
        c: ClassStat(c, counts[c], statistics.median(areas[c]) if areas[c] else 0.0)
        for c in CLASSES
    }


def analyse(per_class_ap: dict[str, float], stats: dict[str, ClassStat],
            m_ap: float, m_ap50: float) -> list[Finding]:
    """Rank what is costing the macro average, worst first.

    Severity is expressed in macro-AP points, because that is the quantity the
    competition scores: a class contributes 1/18th of the total however rare it
    is, so a dead rare class costs exactly as much as a dead common one.
    """
    for name, ap in per_class_ap.items():
        if name in stats:
            stats[name].ap = ap

    scored = [s for s in stats.values() if s.ap is not None]
    if not scored:
        return []
    n = len(scored)
    findings: list[Finding] = []

    # 1. Localization vs detection. The metric averages IoU 0.50:0.95, so a
    #    model that finds objects but boxes them loosely loses most of its score
    #    between those thresholds.
    if m_ap50 > 0:
        loc_gap = m_ap50 - m_ap
        findings.append(Finding(
            "localization precision",
            severity=loc_gap * 0.5,
            evidence=(f"mAP@0.5={m_ap50:.3f} but mAP@[.5:.95]={m_ap:.3f}: "
                      f"{loc_gap / m_ap50:.0%} of the score is lost to box "
                      f"tightness, not to finding the objects"),
            search_terms=["DETR small object localization refinement",
                          "fine-grained bounding box regression distribution",
                          "IoU-aware query selection detection transformer"],
        ))

    # 2. Material pairs: same shape, same size, different label. Only the
    #    spectrum separates them, so a gap here is a spectral failure.
    pair_losses = []
    for group in MATERIAL_GROUPS:
        aps = [(c, stats[c].ap) for c in group if c in stats and stats[c].ap is not None]
        if len(aps) < 2:
            continue
        best, worst = max(aps, key=lambda t: t[1]), min(aps, key=lambda t: t[1])
        if best[1] - worst[1] > 0.1:
            pair_losses.append((best, worst))
    if pair_losses:
        detail = "; ".join(f"{b[0]} {b[1]:.3f} vs {w[0]} {w[1]:.3f}"
                           for b, w in pair_losses)
        findings.append(Finding(
            "material discrimination",
            severity=sum(b[1] - w[1] for b, w in pair_losses) / n,
            evidence=(f"pairs that are the same shape and size split hard: {detail}. "
                      f"Shape cannot separate these -- only the spectrum can"),
            search_terms=["hyperspectral material classification small objects",
                          "spectral spatial feature fusion object detection",
                          "band selection hyperspectral detection deep learning"],
        ))

    # 3. Classes at or near zero AP. Each is a full 1/18th of the macro average.
    dead = sorted([s for s in scored if s.ap < 0.05], key=lambda s: s.ap)
    if dead:
        small_dead = [s for s in dead if s.is_small]
        findings.append(Finding(
            "near-zero classes",
            severity=sum(max(0.0, m_ap - s.ap) for s in dead) / n,
            evidence=(f"{len(dead)}/{n} classes below 0.05 AP "
                      f"({', '.join(f'{s.name} {s.ap:.3f} @{s.median_side:.0f}px' for s in dead[:6])}"
                      f"{'...' if len(dead) > 6 else ''}); "
                      f"{len(small_dead)} of them are COCO-small. Each dead class "
                      f"costs a full 1/{n} of the macro average"),
            search_terms=["small object detection transformer high resolution",
                          "long tail object detection class balanced loss",
                          "copy paste augmentation small objects"],
        ))

    # 4. Does AP track object size? If it does, resolution is the lever.
    sized = [(s.median_side, s.ap) for s in scored if s.count >= 50]
    if len(sized) >= 6:
        sides = np.array([t[0] for t in sized])
        aps = np.array([t[1] for t in sized])
        if sides.std() > 0 and aps.std() > 0:
            r = float(np.corrcoef(sides, aps)[0, 1])
            if r > 0.3:
                small = [s for s in scored if s.is_small]
                findings.append(Finding(
                    "object scale",
                    severity=r * m_ap * 0.5,
                    evidence=(f"AP correlates +{r:.2f} with median object side; "
                              f"{len(small)}/{n} classes are COCO-small. "
                              f"Resolution and feature stride are the lever"),
                    search_terms=["high resolution detection transformer small objects",
                                  "multi scale deformable attention small object",
                                  "slicing aided hyper inference SAHI"],
                ))

    # 5. Rarity, measured separately from size so the two are not confused.
    common = [s for s in scored if s.count >= 500]
    rare = [s for s in scored if s.count < 350]
    if common and rare:
        gap = statistics.mean(s.ap for s in common) - statistics.mean(s.ap for s in rare)
        if gap > 0.05:
            findings.append(Finding(
                "class imbalance",
                severity=gap * len(rare) / n,
                evidence=(f"common classes average {statistics.mean(s.ap for s in common):.3f} AP, "
                          f"rare ones {statistics.mean(s.ap for s in rare):.3f} "
                          f"({len(rare)} classes under 350 instances). The metric "
                          f"macro-averages, so rare classes are not discounted"),
                search_terms=["long tailed object detection decoupled training",
                              "repeat factor sampling detection",
                              "class balanced focal loss detection"],
            ))

    return sorted(findings, key=lambda f: -f.severity)


def report(findings: list[Finding]) -> str:
    if not findings:
        return "no scored classes yet"
    out = ["bottlenecks, worst first (severity in macro-AP points):"]
    for f in findings:
        out.append(str(f))
    return "\n".join(out)
