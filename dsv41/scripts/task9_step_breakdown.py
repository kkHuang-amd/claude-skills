#!/usr/bin/env python3
"""Per-step kernel breakdown of a GPU decode trace (steps segmented at idle gaps > 1 ms like atomport_step_spans.py).
Prints median per-step: span, kernel sum, idle inside the span, and per kernel name (shortened) median count + us,
sorted by time; plus the first/last kernel names of the step (to spot draft vs verify order).
Usage: task9_step_breakdown.py TRACE [--top 40] [--width 90]"""
import argparse, collections, gzip, json, statistics as st

ap = argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("--top", type=int, default=40)
ap.add_argument("--width", type=int, default=90); ap.add_argument("--phase-top", type=int, default=12); a = ap.parse_args()
ev = json.load(gzip.open(a.trace, "rt"))["traceEvents"]
k = sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"], key=lambda e: e["ts"])
segs, cur, end = [], [k[0]], k[0]["ts"] + k[0]["dur"]
for e in k[1:]:
    if e["ts"] - end > 1000:
        segs.append(cur); cur = []
    cur.append(e); end = max(end, e["ts"] + e["dur"])
segs.append(cur); full = segs[1:-1]
spans = [max(e["ts"] + e["dur"] for e in s) - s[0]["ts"] for s in full]
sums = [sum(e["dur"] for e in s) for s in full]
print(f"steps={len(full)} span_med={st.median(spans):.0f}us ksum_med={st.median(sums):.0f}us "
      f"idle_in_span_med={st.median(sp - su for sp, su in zip(spans, sums)):.0f}us")
per_t, per_c = collections.defaultdict(list), collections.defaultdict(list)
for s in full:
    t, c = collections.Counter(), collections.Counter()
    for e in s:
        n = e["name"][: a.width]; t[n] += e["dur"]; c[n] += 1
    for n in t:
        per_t[n].append(t[n]); per_c[n].append(c[n])
rows = sorted(((st.median(v), st.median(per_c[n]), n) for n, v in per_t.items()), reverse=True)
for us, cnt, n in rows[: a.top]:
    print(f"{us:8.1f} us x{cnt:4.0f}  {n}")
MONO = "dsv41_mono_"; PT = a.phase_top
ph = collections.defaultdict(list); ph_names = {p: collections.defaultdict(list) for p in ("pre", "between", "post")}
for s in full:
    idx = [i for i, e in enumerate(s) if MONO in e["name"]]
    if not idx:
        continue
    i0, i1 = idx[0], idx[-1]
    parts = {"pre": s[:i0], "mono": [e for e in s[i0:i1 + 1] if MONO in e["name"]],
             "between": [e for e in s[i0:i1 + 1] if MONO not in e["name"]], "post": s[i1 + 1:]}
    for p, es in parts.items():
        ph[p + "_us"].append(sum(e["dur"] for e in es)); ph[p + "_n"].append(len(es))
        if p in ph_names:
            c = collections.Counter()
            for e in es:
                c[e["name"][:60]] += e["dur"]
            for n, d in c.items():
                ph_names[p][n].append(d)
    ph["pre_wall"].append(s[i0]["ts"] - s[0]["ts"])
    ph["post_wall"].append(max(e["ts"] + e["dur"] for e in s) - s[i1]["ts"] - s[i1]["dur"])
print("phases (median us / kernels): " + "  ".join(f"{p}={st.median(ph[p + '_us']):.0f}/{st.median(ph[p + '_n']):.0f}"
      for p in ("pre", "mono", "between", "post")) + f"  pre_wall={st.median(ph['pre_wall']):.0f} post_wall={st.median(ph['post_wall']):.0f}")
for p, d in ph_names.items():
    top = sorted(((sum(v) / len(full), n) for n, v in d.items()), reverse=True)[:PT]
    print(f"[{p}] " + "; ".join(f"{n} {us:.0f}" for us, n in top))
mid = full[len(full) // 2]
print("first:", " | ".join(e["name"][:40] for e in mid[:6]))
print("last:", " | ".join(e["name"][:40] for e in mid[-6:]))
