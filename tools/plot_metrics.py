#!/usr/bin/env python3
"""Turn a run's metrics.jsonl into curves.

The kernel appends one JSON object per epoch while it trains, so this works on
a partial file from a run still in progress as well as on a finished one. The
final_eval record -- ultralytics re-validating the best checkpoint one epoch
past the end -- is dropped, since plotting it puts a phantom epoch on the axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = {
    "mAP50-95": "metrics/mAP50-95(B)",
    "mAP50": "metrics/mAP50(B)",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
}


def read(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue          # a torn last line from a killed session
        if not r.get("final_eval"):
            rows.append(r)
    # A resumed run's file holds both sessions; the later record wins per epoch.
    by_epoch = {r["epoch"]: r for r in rows}
    return [by_epoch[e] for e in sorted(by_epoch)]


def table(rows: list[dict]) -> str:
    head = f"{'ep':>3}  {'mAP50-95':>9} {'mAP50':>7} {'losses':>26} {'lr':>9} {'drift':>8} {'s':>6}"
    out = [head, "-" * len(head)]
    for r in rows:
        m = r.get("metrics", {})
        losses = " ".join(f"{v:.3f}" for v in r.get("loss", {}).values() if v is not None)
        lr = next(iter(r.get("lr", {}).values()), None)
        out.append(
            f"{r['epoch']:>3}  {m.get(KEYS['mAP50-95'], float('nan')):>9.4f} "
            f"{m.get(KEYS['mAP50'], float('nan')):>7.4f} {losses:>26} "
            f"{(lr if lr is not None else float('nan')):>9.2e} "
            f"{r.get('adapter_drift', float('nan')):>8.5f} "
            f"{(r.get('seconds') or 0):>6.0f}")
    return "\n".join(out)


def svg(rows: list[dict], out: Path, w: int = 760, h: int = 360) -> None:
    """A dependency-free line plot: matplotlib is not worth an install here."""
    series = {name: [(r["epoch"], r["metrics"].get(key)) for r in rows
                     if r.get("metrics", {}).get(key) is not None]
              for name, key in KEYS.items()}
    series = {k: v for k, v in series.items() if v}
    if not series:
        raise SystemExit("no validation metrics in the log yet")
    xs = [e for v in series.values() for e, _ in v]
    x0, x1 = min(xs), max(xs)
    pad, top = 48, 1.0
    colours = {"mAP50-95": "#c0392b", "mAP50": "#2980b9",
               "precision": "#27ae60", "recall": "#8e44ad"}

    def px(e):
        return pad + (e - x0) / max(1, x1 - x0) * (w - 2 * pad)

    def py(y):
        return h - pad - (y / top) * (h - 2 * pad)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
             f'font-family="system-ui,sans-serif" font-size="11">',
             f'<rect width="{w}" height="{h}" fill="white"/>']
    for g in range(6):
        y = py(g / 5)
        parts.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{w - pad}" y2="{y:.1f}" '
                     f'stroke="#eee"/><text x="{pad - 6}" y="{y + 3:.1f}" '
                     f'text-anchor="end" fill="#888">{g / 5:.1f}</text>')
    for i, (name, pts) in enumerate(series.items()):
        d = " ".join(f"{'M' if j == 0 else 'L'}{px(e):.1f},{py(y):.1f}"
                     for j, (e, y) in enumerate(pts))
        c = colours.get(name, "#333")
        parts.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2"/>')
        parts.append(f'<text x="{w - pad - 90}" y="{pad + 14 * i}" fill="{c}">{name} '
                     f'{pts[-1][1]:.4f}</text>')
    parts.append(f'<text x="{w / 2}" y="{h - 12}" text-anchor="middle" fill="#666">epoch '
                 f'{x0}..{x1}</text></svg>')
    out.write_text("\n".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", type=Path, nargs="+")
    ap.add_argument("--svg", type=Path, default=None)
    args = ap.parse_args()

    rows: list[dict] = []
    for f in args.metrics:
        rows.extend(read(f))
    by_epoch = {r["epoch"]: r for r in rows}
    rows = [by_epoch[e] for e in sorted(by_epoch)]
    print(table(rows))
    if args.svg:
        svg(rows, args.svg)
        print(f"\nwrote {args.svg}")


if __name__ == "__main__":
    main()
