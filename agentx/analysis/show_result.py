#!/usr/bin/env python3
"""Print the headline AgentX metrics from one or more agg_*.json files.

Usage:
    python3 show_result.py <agg json> [<agg json> ...]

With two or more files the second and later are compared against the first,
which is the usual "am I at parity with CI?" question. Prints percentages, not
the whole json, so it stays cheap to read.
"""
import json
import sys


def pick(node, *names):
    """Latency nodes are dicts of percentiles; throughput nodes are scalars."""
    if node is None:
        return None
    if isinstance(node, (int, float)):
        return node
    for n in names:
        if n in node:
            v = node[n]
            return v if isinstance(v, (int, float)) else None
    return None


def dig(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def extract(path):
    with open(path) as f:
        agg = json.load(f)
    rm = agg.get("request_metrics", {})
    lat = rm.get("latency", {})
    return {
        "out_tput_per_gpu": pick(dig(rm, "throughput", "per_gpu", "output_tput_tps")),
        "total_tput_per_gpu": pick(dig(rm, "throughput", "per_gpu", "total_tput_tps")),
        "duration_s": pick(dig(rm, "throughput", "duration_seconds")),
        "ttft_p50": pick(lat.get("ttft"), "p50", "median", "avg"),
        "itl_p90": pick(lat.get("itl"), "p90"),
        "intvty_p90": pick(lat.get("intvty"), "p90"),
        "cache_hit": pick(dig(rm, "cache", "theoretical_cache_hit_rate")),
        # ISL must match before a throughput delta means anything.
        "isl_mean": pick(dig(rm, "tokens", "input"), "mean"),
        "osl_actual_mean": pick(dig(rm, "tokens", "output_actual"), "mean"),
        "osl_expected_mean": pick(dig(rm, "tokens", "output_expected"), "mean"),
        "gpu_cache_hit": pick(dig(agg, "server_metrics", "cache", "gpu_cache_hit_rate")),
        "host_cache_hit": pick(dig(agg, "server_metrics", "cache", "cpu_cache_hit_rate")),
        "profiled": pick(dig(agg, "request_accounting", "records_profiled")),
        "errors": pick(dig(agg, "request_accounting", "records_error_dropped")),
        "gpu_power_w": pick(agg.get("avg_total_gpu_power_w")),
        "j_per_out_token": pick(agg.get("joules_per_output_token")),
    }


META_KEYS = ("hw", "image", "recipe_fingerprint", "model", "conc", "tp", "ep",
             "dp_attention", "kv_offloading", "allocated_cpu_dram_gb", "spec_decoding")


def meta(path):
    with open(path) as f:
        agg = json.load(f)
    return {k: agg.get(k) for k in META_KEYS}


def fmt(v):
    return "n/a" if v is None else f"{v:,.4g}"


def main(paths):
    rows = [(p, extract(p)) for p in paths]
    base_path, base = rows[0]
    print(f"base: {base_path}")
    m = meta(base_path)
    print("  " + "  ".join(f"{k}={m[k] if m[k] not in (None, '') else '<empty>'}"
                           for k in ("hw", "image", "recipe_fingerprint")))
    for k, v in base.items():
        print(f"  {k:20s} {fmt(v)}")
    for p, cur in rows[1:]:
        print(f"\nvs base: {p}")
        for k, v in cur.items():
            b = base.get(k)
            if isinstance(v, (int, float)) and isinstance(b, (int, float)) and b:
                print(f"  {k:20s} {fmt(v):>12s}  {100.0 * (v - b) / b:+.2f}%")
            else:
                print(f"  {k:20s} {fmt(v):>12s}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
