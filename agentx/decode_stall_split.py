#!/usr/bin/env python3
"""Separate "decode step got slower" from "decode got stalled by prefill".

Why this is needed: the benchmark's `ITL p90` is aiperf's
`(request_latency - ttft) / (osl - 1)` averaged per request, i.e. TPOT. That
formula cannot distinguish a decode step that is intrinsically slower (bigger
batch) from a decode step that waited while the scheduler ran someone else's
prefill. Both land in the same number.

Server-side logs can separate them. SGLang logs one "Decode batch" line every
`decode_log_interval` (default 40) decode iterations, carrying
`gen throughput (token/s)` measured over that window. So per window:

    tokens_emitted  ~= running_req * accept_len * 40
    window_seconds  =  tokens_emitted / gen_throughput
    per_step_ms     =  window_seconds / 40 * 1000

Classify each window by whether any "Prefill batch" line for the same DP rank
falls inside it:

    prefill-free windows -> intrinsic decode step cost
    windows with prefill -> intrinsic + stall

The gap between the two medians is the stall, and it is measured, not inferred.

Caveats kept in view:
  - `gen throughput` is SGLang's own averaging over the window; per-step numbers
    derived from it are means, so this bounds the stall's average size, not its
    peak.
  - `accept_len` is the MTP acceptance of that window; tokens per iteration is
    running_req * accept_len only to the extent acceptance is steady.
  - Windows are matched to prefills by log timestamp at 1 s resolution, so a
    prefill landing on a window boundary can be attributed to either side.

Usage: decode_stall_split.py <arm> [<arm> ...]
"""
import re, statistics, sys

INTERVAL = 40           # --decode-log-interval default
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?:\.\d+)? (DP\d+) TP\d+\]")


def parse(path):
    """-> {rank: [(t, kind, fields)]} with t as epoch-ish seconds."""
    import datetime
    per = {}
    for ln in open(path, errors="ignore"):
        m = TS.match(ln)
        if not m:
            continue
        if "Decode batch" in ln:
            kind = "d"
            f = {}
            for k, pat in (("rr", r"#running-req: (\d+)"),
                           ("al", r"accept len: ([\d.]+)"),
                           ("gt", r"gen throughput \(token/s\): ([\d.]+)")):
                mm = re.search(pat, ln)
                if not mm:
                    break
                f[k] = float(mm.group(1))
            else:
                pass
            if len(f) != 3:
                continue
        elif "Prefill batch" in ln:
            kind = "p"
            f = {}
            for k, pat in (("new", r"#new-token: (\d+)"),
                           ("cached", r"#cached-token: (\d+)")):
                mm = re.search(pat, ln)
                if mm:
                    f[k] = float(mm.group(1))
        else:
            continue
        t = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
        per.setdefault(m.group(2), []).append((t, kind, f))
    return per


def analyse(arm):
    path = "/workspace/results/%s/server.log" % arm
    per = parse(path)
    if not per:
        print("%s: no parsable batch lines" % arm)
        return

    # Under DP attention the ranks run in lockstep -- prepare_mlp_sync_batch
    # all-gathers a common batch type -- so a prefill on ANY rank stalls every
    # rank's decode. Classifying a window by its own rank's prefills only would
    # count cross-rank stalls as "intrinsic" and inflate the clean baseline.
    # Build one global prefill timeline and classify every window against it.
    prefill_ts = sorted(t for evs in per.values() for t, k, _ in evs if k == "p")
    cached = [f["cached"] for evs in per.values() for _, k, f in evs
              if k == "p" and "cached" in f]

    import bisect

    def prefill_in(t0, t1):
        i = bisect.bisect_right(prefill_ts, t0)
        return i < len(prefill_ts) and prefill_ts[i] < t1

    def prefill_count(t0, t1):
        return (bisect.bisect_left(prefill_ts, t1)
                - bisect.bisect_right(prefill_ts, t0))

    # Dose-response is the load-bearing analysis. A binary clean/contaminated
    # split is nearly meaningless here: --prefill-decode-interval is 10 while a
    # log window is decode_log_interval=40 iterations, so 3-4 prefills fit in
    # every window by construction and "% contaminated" is an artefact of those
    # two numbers, not evidence about the mechanism. Worse, the few genuinely
    # prefill-free windows are the moments with no prefill demand at all, i.e.
    # light load, so using them as the baseline biases the "intrinsic" cost low.
    # Bucketing by how many prefills landed in the window keeps every window in
    # the sample and asks whether more prefill work means a slower decode step.
    dose = {}
    clean, stalled, own_clean = [], [], []
    for rank, evs in per.items():
        evs.sort(key=lambda e: e[0])
        prev_d = None
        own_pend = 0
        for t, kind, f in evs:
            if kind == "p":
                own_pend += 1
                continue
            if prev_d is not None:
                tok = f["rr"] * f["al"] * INTERVAL
                if f["gt"] > 0 and tok > 0:
                    step_ms = (tok / f["gt"]) / INTERVAL * 1000.0
                    # Guard against log-rotation / restart artefacts.
                    if 0.1 < step_ms < 5000:
                        n_pf = prefill_count(prev_d, t)
                        key = n_pf if n_pf <= 3 else 4
                        dose.setdefault(key, []).append(step_ms)
                        if n_pf:
                            stalled.append(step_ms)
                        else:
                            clean.append(step_ms)
                        if not own_pend:
                            own_clean.append(step_ms)
            prev_d = t
            own_pend = 0

    def q(v, p):
        return statistics.quantiles(v, n=100)[p - 1] if len(v) > 2 else float("nan")

    print("== %s" % arm)
    print("   decode windows: %d globally prefill-free, %d with a prefill on some rank"
          " (%.0f %% contaminated)"
          % (len(clean), len(stalled),
             100.0 * len(stalled) / max(1, len(clean) + len(stalled))))
    if own_clean:
        print("   [own-rank-only classification would have called %d windows clean --"
              " that is the biased number]" % len(own_clean))
    if clean:
        print("   intrinsic decode step : median %.2f ms  p90 %.2f ms"
              % (statistics.median(clean), q(clean, 90)))
    if stalled:
        print("   with prefill inside   : median %.2f ms  p90 %.2f ms"
              % (statistics.median(stalled), q(stalled, 90)))
    if clean and stalled:
        a, b = statistics.median(clean), statistics.median(stalled)
        print("   => stall adds %.2f ms per decode step (%.0f %% of intrinsic)"
              % (b - a, 100.0 * (b - a) / a))
    if dose:
        print("   dose-response (prefills landing in the window -> per-step cost):")
        base = None
        for k in sorted(dose):
            v = dose[k]
            if len(v) < 20:
                continue
            med = statistics.median(v)
            if base is None:
                base = med
            label = "%d" % k if k <= 3 else ">=4"
            print("     %-4s prefills  n=%-5d median %7.2f ms   %+.0f %% vs 0"
                  % (label, len(v), med, 100.0 * (med - base) / base))
    if cached:
        print("   prefill cached-tokens : median %s  p90 %s  max %s  (n=%d)"
              % ("{:,.0f}".format(statistics.median(cached)),
                 "{:,.0f}".format(q(cached, 90)),
                 "{:,.0f}".format(max(cached)), len(cached)))


if __name__ == "__main__":
    for a in sys.argv[1:] or ["fp4-dptbo-c160"]:
        analyse(a)
