#!/usr/bin/env python3
"""Rank host time in a py-spy raw (collapsed) trace recorded with --threads [--idle].
    pyspy_rank.py <pyspy_raw.txt> [top=35] [focus_substring]
Per thread: total samples; for the busiest thread (or the one whose stacks contain focus_substring): top inclusive
functions (each counted once per stack) and top leaf (self) frames, as % of that thread's samples."""
import collections, re, sys

path = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 35
focus = sys.argv[3] if len(sys.argv) > 3 else None
threads = collections.defaultdict(list)
for line in open(path):
    line = line.rstrip("\n")
    m = re.match(r"^(.*) (\d+)$", line)
    if not m:
        continue
    frames, n = m.group(1).split(";"), int(m.group(2))
    tid = next((f for f in frames if f.startswith("thread ")), "thread ?")
    body = [f for f in frames if not f.startswith(("thread ", "process "))]
    threads[tid].append((body, n))


def short(f):
    m = re.match(r"^(.*) \((.*?):(\d+)\)$", f)
    if not m:
        return f[:90]
    fn, file = m.group(1), m.group(2).split("/")[-1]
    return f"{fn} ({file})"


tot = {t: sum(n for _, n in s) for t, s in threads.items()}
for t, n in sorted(tot.items(), key=lambda x: -x[1]):
    leafs = collections.Counter()
    for b, k in threads[t]:
        leafs[short(b[-1]) if b else "?"] += k
    print(f"{t[:40]:40s} samples {n:7d}  top leaf: {leafs.most_common(1)[0][0][:70]}")
if focus:
    cand = [t for t in threads if any(focus in f for b, _ in threads[t] for f in b)]
    t = max(cand, key=lambda x: tot[x])
else:
    t = max(tot, key=tot.get)
N = tot[t]
inc, leaf = collections.Counter(), collections.Counter()
for b, k in threads[t]:
    for f in set(short(x) for x in b):
        inc[f] += k
    leaf[short(b[-1]) if b else "?"] += k
print(f"\n== {t} ({N} samples) inclusive top {top}")
for f, k in inc.most_common(top):
    print(f"{100 * k / N:6.1f}%  {f[:150]}")
print(f"\n== leaf top {top // 2}")
for f, k in leaf.most_common(top // 2):
    print(f"{100 * k / N:6.1f}%  {f[:150]}")
