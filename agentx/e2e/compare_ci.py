#!/usr/bin/env python3
"""Compare AgentX result JSONs metric by metric.

usage: compare_ci.py <baseline.json> <candidate.json> [<candidate2.json> ...]

baseline is usually a CI point from ci_baseline_<run>/; candidates are local
$RESULT_DIR/$RESULT_FILENAME.json files. Sanity gates print first: a candidate
that fails them did a different workload and its deltas mean nothing.
"""
import json
import sys

# The two that decide a regression; everything else is diagnostic.
PRIMARY = [  # (label, dotted path, higher_is_better, replicate noise %)
    ("tok/s/GPU total", "request_metrics.throughput.per_gpu.total_tput_tps", True, 5.67),
    ("intvty p90", "request_metrics.latency.intvty.p90", True, 6.6),
]
METRICS = [  # (label, dotted path, higher_is_better)
    ("tok/s/GPU output", "request_metrics.throughput.per_gpu.output_tput_tps", True),
    ("TTFT mean s", "request_metrics.latency.ttft.mean", False),
    ("TTFT p50 s", "request_metrics.latency.ttft.p50", False),
    ("ITL mean ms", "request_metrics.latency.itl.mean", False),
    ("ITL p90 ms", "request_metrics.latency.itl.p90", False),
    ("intvty p50", "request_metrics.latency.intvty.p50", True),
    ("e2el mean s", "request_metrics.latency.e2el.mean", False),
    ("overall cache hit", "server_metrics.cache.overall_cache_hit_rate", True),
]
GATES = [
    ("successful", "num_requests_successful"),
    ("error_dropped", "request_accounting.records_error_dropped"),
    ("duration_s", "request_metrics.throughput.duration_seconds"),
    ("ISL mean", "request_metrics.tokens.input.mean"),
    ("OSL mean", "request_metrics.tokens.output_actual.mean"),
]
MS = {"ITL mean ms", "ITL p90 ms"}


def get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def fmt(label, v):
    if v is None:
        return "-"
    if label in MS:
        v *= 1000
    return f"{v:,.3f}" if abs(v) < 100 else f"{v:,.1f}"


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    runs = [json.load(open(p)) for p in sys.argv[1:]]
    base = runs[0]
    names = ["baseline"] + [f"cand{i}" for i in range(1, len(runs))]
    w = 20
    print(f"{'':<{w}}" + "".join(f"{n:>14}" for n in names))
    for k in ("image", "conc", "model"):
        print(f"{k:<{w}}" + "".join(f"{str(r.get(k))[-13:]:>14}" for r in runs))

    print("-- gates")
    for label, path in GATES:
        print(f"{label:<{w}}" + "".join(f"{fmt(label, get(r, path)):>14}" for r in runs))
    bisl = get(base, "request_metrics.tokens.input.mean")
    for n, r in zip(names[1:], runs[1:]):
        warn = []
        if get(r, "request_accounting.records_error_dropped"):
            warn.append("errors>0")
        dur = get(r, "request_metrics.throughput.duration_seconds") or 0
        if dur < 0.95 * (get(base, "request_metrics.throughput.duration_seconds") or 0):
            warn.append(f"short window {dur:.0f}s")
        isl = get(r, "request_metrics.tokens.input.mean")
        if bisl and isl and abs(isl / bisl - 1) > 0.03:
            warn.append(f"ISL moved {100 * (isl / bisl - 1):+.1f}%")
        print(f"  {n}: {'GATES OK' if not warn else 'WARN ' + ', '.join(warn)}")

    print("-- PRIMARY (ok < replicate noise [tok 5.67%, intvty 6.6%]; rerun up to 10%; REGR >= 10%)")
    for label, path, hib, noise in PRIMARY:
        b = get(base, path)
        row = f"{label:<{w}}{fmt(label, b):>14}"
        for r in runs[1:]:
            v = get(r, path)
            if v is None or not b:
                row += f"{fmt(label, v):>14}"
                continue
            d = 100 * (v / b - 1)
            worse = (d < 0) if hib else (d > 0)
            tag = "ok" if abs(d) < noise else ("REGR" if worse and abs(d) >= 10 else
                  ("rerun" if worse else "better"))
            row += f"{fmt(label, v):>9} {d:+4.0f}% {tag}"
        print(row)

    print("-- diagnostic (delta vs baseline; '!' = worse)")
    for label, path, hib in METRICS:
        b = get(base, path)
        row = f"{label:<{w}}{fmt(label, b):>14}"
        for r in runs[1:]:
            v = get(r, path)
            if v is None or not b:
                row += f"{fmt(label, v):>14}"
                continue
            d = 100 * (v / b - 1)
            worse = (d < 0) if hib else (d > 0)
            row += f"{fmt(label, v):>9} {d:+4.0f}%{'!' if worse and abs(d) >= 5 else ' '}"
        print(row)


if __name__ == "__main__":
    main()
