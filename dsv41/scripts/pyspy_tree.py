#!/usr/bin/env python3
"""Top-down tree of a py-spy raw (collapsed) profile, rooted at a frame, for host-overhead attribution.

Keeps only stacks that contain ROOT (function name, e.g. event_loop_overlap), strips frames above it, and prints
inclusive sample share per call path down to DEPTH levels, hiding nodes below MIN_PCT of the ROOT total.
With 1000 Hz sampling, samples / seconds sampled / steps per second = ms per step.
Usage: pyspy_tree.py RAW.txt [--root event_loop_overlap] [--depth 6] [--min-pct 1.0] [--focus run_batch]
  --focus F: re-root at the first frame named F below ROOT (percentages stay relative to ROOT).
"""
import argparse, re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("raw"); ap.add_argument("--root", default="event_loop_overlap")
ap.add_argument("--depth", type=int, default=6); ap.add_argument("--min-pct", type=float, default=1.0)
ap.add_argument("--focus", default="")
a = ap.parse_args()

fname = lambda fr: fr.split(" (")[0]
tree, total = defaultdict(int), 0
for line in open(a.raw):
    m = re.match(r"^(.*) (\d+)$", line.rstrip("\n"))
    if not m:
        continue
    frames, n = m.group(1).split(";"), int(m.group(2))
    idx = next((i for i, f in enumerate(frames) if fname(f) == a.root), None)
    if idx is None:
        continue
    total += n
    frames = frames[idx + 1:]
    if a.focus:
        j = next((i for i, f in enumerate(frames) if fname(f) == a.focus), None)
        if j is None:
            continue
        frames = frames[j:]
    for d in range(1, min(a.depth, len(frames)) + 1):
        tree[tuple(frames[:d])] += n

print(f"root={a.root} samples={total}")
for path in sorted(tree):
    pct = 100.0 * tree[path] / max(total, 1)
    if pct >= a.min_pct:
        f = path[-1]
        short = re.sub(r"\(.*/", "(", f)
        print(f"{'  ' * (len(path) - 1)}{pct:5.1f}% {tree[path]:6d}  {short[:110]}")
