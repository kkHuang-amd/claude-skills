#!/usr/bin/env python3
"""Summarize rocprofv3 --pmc CSVs (one or more pass dirs) per kernel: mean duration, resources, counters, derived ratios.
Usage: rocprof_counter_summary.py <pass_dir> [<pass_dir> ...] [--match REGEX] [--name-width N]
Derived ratios (units as rocprofv3 documents them; WMMA counters are {emulated}, treat as indicative):
  wait%      = SQ_WAIT_ANY / SQ_WAVE_CYCLES          (wave time spent waiting on anything)
  instwait%  = SQ_WAIT_INST_ANY / SQ_WAVE_CYCLES     (wave time waiting for instruction issue)
  hbm_TB/s   = FETCH_SIZE (KB) * 1024 / duration
  l2hit%     = GL2C_HIT / (GL2C_HIT + GL2C_MISS)
  wmma/vmem  = SQ_INST_CYCLES_VALU_WMMA / SQ_INST_CYCLES_VMEM_LOAD
"""
import argparse, csv, glob, os, re, statistics
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--match", default=".")
ap.add_argument("--name-width", type=int, default=60)
a = ap.parse_args()

dur = defaultdict(list)      # kernel -> [ns] (one per dispatch, de-duplicated across counter rows)
res = {}                     # kernel -> (vgpr, sgpr, scratch, lds)
cnt = defaultdict(lambda: defaultdict(list))
seen = set()
for d in a.dirs:
    for f in glob.glob(os.path.join(d, "**", "*counter_collection.csv"), recursive=True):
        for r in csv.DictReader(open(f)):
            k = r["Kernel_Name"]
            if not re.search(a.match, k):
                continue
            k = k[: a.name_width]
            cnt[k][r["Counter_Name"]].append(float(r["Counter_Value"]))
            key = (f, r["Dispatch_Id"])
            if key not in seen:
                seen.add(key)
                dur[k].append(int(r["End_Timestamp"]) - int(r["Start_Timestamp"]))
                res[k] = (r["VGPR_Count"], r["SGPR_Count"], r["Scratch_Size"], r["LDS_Block_Size"])

for k in sorted(dur, key=lambda x: -statistics.mean(dur[x])):
    c = {n: statistics.mean(v) for n, v in cnt[k].items()}
    t_ns = statistics.mean(dur[k])
    v, s, sc, lds = res[k]
    print(f"{k}\n  n={len(dur[k])} dur={t_ns / 1e3:.1f} us  VGPR={v} SGPR={s} scratch={sc} LDS={lds}")
    out = []
    if "SQ_WAVE_CYCLES" in c and c["SQ_WAVE_CYCLES"]:
        for n, lab in (("SQ_WAIT_ANY", "wait%"), ("SQ_WAIT_INST_ANY", "instwait%")):
            if n in c:
                out.append(f"{lab}={100 * c[n] / c['SQ_WAVE_CYCLES']:.1f}")
    if "FETCH_SIZE" in c:
        out.append(f"hbm_TB/s={c['FETCH_SIZE'] * 1024 / t_ns / 1e3:.2f}")
    if c.get("GL2C_HIT", 0) + c.get("GL2C_MISS", 0):
        out.append(f"l2hit%={100 * c['GL2C_HIT'] / (c['GL2C_HIT'] + c['GL2C_MISS']):.1f}")
    if c.get("SQ_INST_CYCLES_VMEM_LOAD"):
        out.append(f"wmma/vmem={c.get('SQ_INST_CYCLES_VALU_WMMA', 0) / c['SQ_INST_CYCLES_VMEM_LOAD']:.2f}")
    print("  " + " ".join(out))
    print("  " + " ".join(f"{n}={v:.4g}" for n, v in sorted(c.items())))
