#!/usr/bin/env python3
"""Digest the live log of a running Kaggle kernel.

A pushed kernel is a batch job: `kaggle kernels logs` returns nothing until it
finishes, and `-f` streams the whole backlog every time -- 150 kB of
carriage-return progress bars to find the two lines that matter. This follows
the stream for a bounded window, throws the redraws away, and prints the
script's own timestamped lines plus where the current epoch has got to.

    python3 tools/watch_run.py qwyi123/hod26-team --seconds 45
"""
import argparse
import re
import subprocess
import sys

# The kernel script prefixes every line it writes itself with a clock.
SCRIPT_LINE = re.compile(r"^\[\d\d:\d\d:\d\d\]")
# Ultralytics' per-batch redraw: "  22/40  11.5G  ... 56% ... 673/1200 1.6s/it 17:54<14:10"
PROGRESS = re.compile(
    r"(?P<epoch>\d+)/(?P<total>\d+)\s+\S+G\s+.*?"
    r"(?P<done>\d+)/(?P<steps>\d+)\s+(?P<rate>[\d.]+s/it)\s+"
    r"(?P<spent>[\d:]+)<(?P<left>[\d:]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\[K")


def clean(text):
    # Progress bars overwrite themselves with \r; only the last draw survives.
    return ANSI.sub("", text.replace("\r", "\n"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", help="<owner>/<kernel-name>")
    ap.add_argument("--seconds", type=int, default=45,
                    help="how long to follow the stream")
    ap.add_argument("--tail", type=int, default=25,
                    help="how many script lines to print")
    args = ap.parse_args()

    status = subprocess.run(["kaggle", "kernels", "status", args.slug],
                            capture_output=True, text=True)
    print(status.stdout.strip() or status.stderr.strip())

    # subprocess' own timeout raises on a stream that never ends and throws away
    # what it read, so the window is enforced with `timeout(1)` and the output kept.
    proc = subprocess.run(
        ["timeout", str(args.seconds), "kaggle", "kernels", "logs", args.slug, "-f"],
        capture_output=True, text=True, errors="replace", check=False)
    lines = clean(proc.stdout).splitlines()
    if not lines:
        print("no log yet (session may still be installing)", file=sys.stderr)
        return 1

    script = [ln for ln in lines if SCRIPT_LINE.match(ln)]
    for ln in script[-args.tail:]:
        print(ln)

    last = None
    for ln in lines:
        m = PROGRESS.search(ln)
        if m:
            last = m
    if last:
        g = last.groupdict()
        print(f"  -> epoch {g['epoch']}/{g['total']}  step {g['done']}/{g['steps']}"
              f"  {g['rate']}  {g['spent']} in, {g['left']} left")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
