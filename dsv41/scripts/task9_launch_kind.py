#!/usr/bin/env python3
"""Count, per kernel-name group, which runtime call launched the kernels (hipGraphLaunch vs eager launches), per step.
Groups: mono (dsv41_mono_), nccl, other. Usage: task9_launch_kind.py TRACE [--steps 40]"""
import argparse, collections, gzip, json

ap = argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("--steps", type=int, default=40); a = ap.parse_args()
ev = [e for e in json.load(gzip.open(a.trace, "rt"))["traceEvents"] if e.get("ph") == "X"]
rt = {e["args"]["correlation"]: e["name"] for e in ev if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})}
c, t = collections.Counter(), collections.Counter()
for e in ev:
    if e.get("cat") != "kernel":
        continue
    g = "mono" if "dsv41_mono_" in e["name"] else "nccl" if "nccl" in e["name"] else "other"
    r = rt.get(e.get("args", {}).get("correlation"), "?")
    c[(g, r)] += 1; t[(g, r)] += e["dur"]
for (g, r), n in sorted(c.items()):
    print(f"{g:6s} {r:28s} {n / a.steps:7.1f}/step {t[(g, r)] / a.steps:8.0f} us/step")
