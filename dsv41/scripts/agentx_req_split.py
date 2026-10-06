#!/usr/bin/env python3
"""Split an AgentX (aiperf) run into prefill (TTFT) vs decode (TPOT) from per-request records.

    python3 -I agentx_req_split.py <label>=<profile_export.jsonl> [...]

Only benchmark_phase == profiling, not cancelled, OSL >= 2. TPOT = decode_duration / (OSL - 1), the same definition as
bmk_agentic request_metrics.latency.tpot; intvty P90 = 1000 / TPOT p90 (the "P90" column in results/agentx.md).
"""
import json
import statistics as st
import sys

ISL_EDGES = [65536, 131072, 262144]  # buckets: <64k, 64-128k, 128-256k, >=256k


def q(a, p):
    if not a:
        return float("nan")
    a = sorted(a)
    return a[min(len(a) - 1, int(p / 100 * len(a)))]


def val(m, k):
    v = m.get(k)
    return v.get("value") if isinstance(v, dict) else v


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            md, m = d.get("metadata", {}), d.get("metrics", {})
            if md.get("benchmark_phase") != "profiling" or str(md.get("was_cancelled")) == "True":
                continue
            osl, ttft, dec = val(m, "output_sequence_length"), val(m, "time_to_first_token"), val(m, "decode_duration")
            if not osl or osl < 2 or ttft is None or dec is None:
                continue
            rows.append(dict(isl=val(m, "input_sequence_length") or 0, osl=osl, ttft=ttft, dec=dec,
                             tpot=dec / (osl - 1), turn=int(md.get("turn_index", 0))))
    return rows


def bucket(isl):
    for i, e in enumerate(ISL_EDGES):
        if isl < e:
            return i
    return len(ISL_EDGES)


def main():
    names = ["<64k", "64-128k", "128-256k", ">=256k"]
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        r = load(path)
        ttft, tpot = [x["ttft"] for x in r], [x["tpot"] for x in r]
        tot = sum(x["ttft"] + x["dec"] for x in r)
        print(f"== {label}: n={len(r)} ISL mean {st.mean(x['isl'] for x in r):,.0f} OSL mean "
              f"{st.mean(x['osl'] for x in r):,.0f}")
        print(f"  TTFT ms mean/p50/p90 {st.mean(ttft):.0f}/{q(ttft, 50):.0f}/{q(ttft, 90):.0f}  "
              f"TPOT ms mean/p50/p90 {st.mean(tpot):.3f}/{q(tpot, 50):.3f}/{q(tpot, 90):.3f}  "
              f"intvty P90 {1000 / q(tpot, 90):.1f}  TTFT share of req time {sum(ttft) / tot:.1%}")
        cold = [x["ttft"] for x in r if x["turn"] == 0]
        warm = [x["ttft"] for x in r if x["turn"] > 0]
        print(f"  TTFT turn0 n={len(cold)} p50/p90 {q(cold, 50):.0f}/{q(cold, 90):.0f}  "
              f"turn>0 n={len(warm)} p50/p90 {q(warm, 50):.0f}/{q(warm, 90):.0f}")
        for b, nm in enumerate(names):
            s = [x for x in r if bucket(x["isl"]) == b]
            if s:
                print(f"  ISL {nm:>8}: n={len(s):4d} TTFT p50/p90 {q([x['ttft'] for x in s], 50):6.0f}/"
                      f"{q([x['ttft'] for x in s], 90):6.0f}  TPOT p50/p90 {q([x['tpot'] for x in s], 50):.3f}/"
                      f"{q([x['tpot'] for x in s], 90):.3f}")


if __name__ == "__main__":
    main()
