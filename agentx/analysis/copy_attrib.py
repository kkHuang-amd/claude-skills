#!/usr/bin/env python3
"""Attribute the unfused `copy` kernels to their call sites.

    python3 copy_attrib.py <trace dir> [--role copy]

3.85 ms/step of MI355X's decode wall is `copy`-role kernels that have no B200
counterpart at all (B200 spends 0.03 ms). Knowing the kernel names is not
actionable; knowing which python/aten operation launches each one is.

Method, and the two constraints that shape it:

1. **Attribution only works in EXTEND.** Inside a cuda-graph-replayed window the
   launching CPU op no longer runs, so nothing correlates. `TARGET_VERIFY` is
   replayed, `EXTEND` is eager. The result is assumed to carry over to the same
   kernels in the replayed decode steps -- same code path, different launch
   mechanism, so verify a fix against the DECODE step wall, not this table.

2. **`External id` is not enough.** A kernel event carries `correlation`, not
   `External id`; the `External id` lives on the `cuda_runtime` launch event.
   And only aten's `hipLaunchKernel` carries one -- Triton's
   `hipModuleLaunchKernel` does not, which is exactly why the two largest
   offenders here (`_fill_padded_rows_kernel`, `_swa_scatter_kernel`, both
   Triton) resolve to nothing by that route.

So this walks `kernel.correlation -> cuda_runtime launch event -> its
timestamp`, then finds the innermost `cpu_op` / `user_annotation` whose span
CONTAINS that timestamp. Timestamp containment works for every launcher
regardless of framework, and it yields the enclosing frame rather than a generic
`aten::copy_` that names the operation but not the caller.
"""
import collections
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import GPU_CATS, STEP_RE, classify, norm_name, rank_of  # noqa: E402


def load_raw(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)["traceEvents"]


def eager_windows(ev):
    """-> [(lo, hi)] for outermost non-replayed step annotations."""
    raw = []
    for e in ev:
        if (e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation"
                and e.get("dur") and str(e.get("name", "")).startswith("step[")):
            m = STEP_RE.match(e["name"])
            if m and m.group("type") == "EXTEND":
                raw.append((e["ts"], e["ts"] + e["dur"]))
    raw.sort()
    out = []
    for lo, hi in raw:
        if out and lo >= out[-1][0] and hi <= out[-1][1]:
            continue
        out.append((lo, hi))
    return out


def main(d, role_want):
    per_kernel = collections.defaultdict(lambda: [0.0, 0])
    sites = collections.defaultdict(lambda: [0.0, 0])
    unresolved = collections.defaultdict(lambda: [0.0, 0])
    grids = {}
    nfiles = 0

    for f in sorted(Path(d).glob("*.trace.json.gz"), key=rank_of):
        ev = load_raw(f)
        wins = eager_windows(ev)
        if not wins:
            continue
        nfiles += 1

        # correlation -> the launch event's timestamp on the CPU timeline.
        launch_ts = {}
        for e in ev:
            if e.get("cat") == "cuda_runtime" and "Launch" in str(e.get("name", "")):
                a = e.get("args") or {}
                if "correlation" in a:
                    launch_ts[a["correlation"]] = e["ts"]

        # Containers, for innermost-enclosing-frame lookup by timestamp.
        cont = [(e["ts"], e["ts"] + e["dur"], e.get("name", "?"))
                for e in ev
                if e.get("cat") in ("cpu_op", "user_annotation") and e.get("dur")]

        queries = []
        for e in ev:
            if e.get("ph") != "X" or e.get("cat") not in GPU_CATS or not e.get("dur"):
                continue
            if not any(lo <= e["ts"] <= hi for lo, hi in wins):
                continue
            name = e.get("name", "?")
            if classify(name) != role_want:
                continue
            k = norm_name(name)
            per_kernel[k][0] += e["dur"] / 1000.0
            per_kernel[k][1] += 1
            a = e.get("args") or {}
            if a.get("grid") and k not in grids:
                grids[k] = (a["grid"], a.get("block"))
            ts = launch_ts.get(a.get("correlation"))
            if ts is None:
                unresolved[k][0] += e["dur"] / 1000.0
                unresolved[k][1] += 1
                continue
            queries.append((ts, k, e["dur"] / 1000.0))

        # Sweep: one pass over containers and queries in time order, keeping the
        # set of open frames. Nesting means the shortest open frame is innermost.
        events = [(lo, 0, hi, nm) for lo, hi, nm in cont]
        events += [(ts, 1, k, ms) for ts, k, ms in queries]
        events.sort(key=lambda x: (x[0], x[1]))
        open_frames = []
        for t, kind, a, b in events:
            if kind == 0:
                open_frames.append((a, b))          # (hi, name)
            else:
                open_frames = [x for x in open_frames if x[0] >= t]
                if not open_frames:
                    unresolved[a][0] += b
                    unresolved[a][1] += 1
                    continue
                inner = min(open_frames, key=lambda x: x[0] - t)[1]
                outer = max(open_frames, key=lambda x: x[0] - t)[1]
                sites[(a, inner, outer)][0] += b
                sites[(a, inner, outer)][1] += 1

    if not per_kernel:
        print(f"no '{role_want}' kernels inside any EXTEND window in {d}")
        return

    tot = sum(v[0] for v in per_kernel.values())
    unres = sum(v[0] for v in unresolved.values())
    print(f"=== {d}\n{nfiles} ranks with EXTEND windows; role='{role_want}'")
    print(f"{tot:.2f} ms total in EXTEND, {100 * (1 - unres / tot):.0f} % resolved "
          f"to a call site\n")

    print("per kernel (EXTEND only -- NOT the per-step decode cost):")
    print(f"{'ms':>8s} {'calls':>7s} {'grid x block':>18s}  kernel")
    for k, (ms, n) in sorted(per_kernel.items(), key=lambda x: -x[1][0]):
        g = grids.get(k)
        gs = f"{g[0][0]}x{g[1][0]}" if g else "-"
        u = unresolved.get(k, [0, 0])
        tag = f"   [{u[1]} unresolved]" if u[1] else ""
        print(f"{ms:8.3f} {n:7d} {gs:>18s}  {k[:60]}{tag}")

    print("\ncall sites, by time:")
    for (k, inner, outer), (ms, n) in sorted(sites.items(), key=lambda x: -x[1][0]):
        if ms < 0.02:
            continue
        print(f"{ms:8.3f} {n:7d}  {k[:52]}")
        print(f"{'':18s}innermost: {inner[:58]}")
        if outer != inner:
            print(f"{'':18s}outermost: {outer[:58]}")

    print("\nVerify any fix against the DECODE step wall, not against this "
          "table:\nEXTEND is eager, decode is graph-replayed.")


if __name__ == "__main__":
    args = sys.argv[1:]
    role = "copy"
    if "--role" in args:
        i = args.index("--role")
        role = args[i + 1]
        args = args[:i] + args[i + 2:]
    main(args[0] if args else "/shared_nfs/kk/pr35619/trace_c128_pdi24_steady", role)
