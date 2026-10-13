#!/usr/bin/env python3
"""Does decode ITL in an AgentX run come from overlapping other requests' prefills?
    agentx_overlap.py <run_dir> [<run_dir> ...]     (dirs holding aiperf_artifacts/profile_export.jsonl)
Per request (profiling phase, OSL >= 16): decode window = [start + TTFT, end]; prefill window of every other request
= [start, start + TTFT]. Reports concurrency (requests in flight), and ITL split by whether the decode window overlaps
any other request's prefill, plus the slow-10% composition."""
import json, statistics as st, sys


def load(d):
    rows = []
    for l in open(f"{d}/aiperf_artifacts/profile_export.jsonl"):
        r = json.loads(l); m, md = r["metrics"], r["metadata"]
        if md.get("benchmark_phase") != "profiling":
            continue
        g = lambda k: (m.get(k) or {}).get("value")
        osl, ttft, lat = g("output_sequence_length"), g("time_to_first_token"), g("request_latency")
        if not osl or osl < 16 or ttft is None or lat is None:
            continue
        s = md["request_start_ns"] / 1e6
        rows.append(dict(s=s, f=s + ttft, e=s + lat, osl=osl, isl=g("input_sequence_length") or 0,
                         itl=(lat - ttft) / (osl - 1), depth=md.get("agent_depth"),
                         cached=g("usage_prompt_cache_read_tokens") or 0))
    return rows


for d in sys.argv[1:]:
    rows = load(d)
    for r in rows:
        ov = 0.0
        for o in rows:
            if o is r:
                continue
            ov += max(0.0, min(r["e"], o["f"]) - max(r["f"], o["s"]))
        r["ov"] = ov / max(r["e"] - r["f"], 1e-9)
        r["inflight"] = sum(1 for o in rows if o is not r and o["s"] < r["e"] and o["e"] > r["f"])
    yes = [r["itl"] for r in rows if r["ov"] > 0]
    no = [r["itl"] for r in rows if r["ov"] == 0]
    slow = sorted(rows, key=lambda r: r["itl"])[int(len(rows) * 0.9):]
    p90 = st.quantiles([r["itl"] for r in rows], n=10)[8]
    print(f"{d[-44:]:44s} n={len(rows)} ITL p50 {st.median(r['itl'] for r in rows):.3f} p90 {p90:.3f} | "
          f"decode overlaps another prefill: {len(yes)} reqs med ITL {st.median(yes) if yes else 0:.3f}; "
          f"no overlap: {len(no)} med {st.median(no) if no else 0:.3f}")
    print(f"    slow10%: {sum(1 for r in slow if r['ov'] > 0)}/{len(slow)} overlap a prefill, med overlap frac "
          f"{st.median(r['ov'] for r in slow):.2f}, med OSL {st.median(r['osl'] for r in slow):.0f}, "
          f"med inflight {st.median(r['inflight'] for r in slow):.0f}; all: med inflight "
          f"{st.median(r['inflight'] for r in rows):.0f}, depth>0 {sum(1 for r in rows if r['depth'])}")
