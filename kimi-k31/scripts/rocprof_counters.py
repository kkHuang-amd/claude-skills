#!/usr/bin/env python3
"""rocprofv3 counter groups for one torch-free binary; prints the last dispatch of the
kernel matching <pattern> as ratios. Usage: rocprof_counters.py <binary> [pattern] [args...]"""
import csv
import glob
import os
import shutil
import subprocess
import sys

binary = sys.argv[1]
pat = sys.argv[2] if len(sys.argv) > 2 else ""
args = sys.argv[3:] or ["3"]
groups = [
    "SQ_WAVES SQ_WAVE_CYCLES SQ_BUSY_CYCLES SQ_WAIT_ANY SQ_WAIT_INST_ANY SQ_ACTIVE_INST_ANY",
    "SQ_ACTIVE_INST_VALU SQ_ACTIVE_INST_SCA SQ_ACTIVE_INST_LDS SQ_ACTIVE_INST_VMEM SQ_ACTIVE_INST_MISC SQ_WAIT_INST_LDS",
    "SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS SQ_INSTS_VMEM_RD SQ_INSTS_MFMA SQ_INSTS_BRANCH",
    "SQ_LDS_BANK_CONFLICT SQ_LDS_ADDR_CONFLICT SQ_LDS_IDX_ACTIVE SQ_LDS_UNALIGNED_STALL SQ_VALU_MFMA_BUSY_CYCLES SQ_BUSY_CU_CYCLES",
    "SQ_INST_LEVEL_VMEM SQ_INST_CYCLES_VMEM_RD SQ_INST_LEVEL_LDS TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_sum",
]
vals = {}
for gi, grp in enumerate(groups):
    out = f"/tmp/rocprof_cnt/g{gi}"
    shutil.rmtree(out, ignore_errors=True)
    r = subprocess.run(["rocprofv3", "--pmc", *grp.split(), "-d", out, "-o", "c", "--output-format", "csv", "--", binary, *args],
                       capture_output=True, text=True, timeout=600)
    files = glob.glob(f"{out}/**/*counter_collection.csv", recursive=True)
    if not files:
        print(f"group {gi} failed: {(r.stderr or r.stdout)[-300:]}")
        continue
    rows = [x for x in csv.DictReader(open(files[0])) if pat in x["Kernel_Name"]]
    last = max(int(x["Dispatch_Id"]) for x in rows)
    for x in rows:
        if int(x["Dispatch_Id"]) == last:
            vals[x["Counter_Name"]] = float(x["Counter_Value"])
wc = vals.get("SQ_WAVE_CYCLES", 1.0)
print(f"waves {vals.get('SQ_WAVES', 0):.0f}  wave-cycles {wc:.3e}")
for k in ("SQ_WAIT_ANY", "SQ_WAIT_INST_ANY", "SQ_ACTIVE_INST_ANY", "SQ_ACTIVE_INST_VALU", "SQ_ACTIVE_INST_SCA",
          "SQ_ACTIVE_INST_LDS", "SQ_ACTIVE_INST_VMEM", "SQ_ACTIVE_INST_MISC", "SQ_WAIT_INST_LDS"):
    if k in vals:
        print(f"  {k:24s} {vals[k] / wc:6.3f} of wave-cycles")
w = vals.get("SQ_WAVES", 1.0)
print("  per wave: " + ", ".join(f"{k[9:]} {vals[k] / w:.0f}" for k in
      ("SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_LDS", "SQ_INSTS_VMEM_RD", "SQ_INSTS_MFMA", "SQ_INSTS_BRANCH") if k in vals))
if "SQ_INSTS_LDS" in vals:
    print(f"  LDS bank conflict cycles / LDS instr {vals.get('SQ_LDS_BANK_CONFLICT', 0) / vals['SQ_INSTS_LDS']:.2f}, "
          f"addr conflict {vals.get('SQ_LDS_ADDR_CONFLICT', 0) / vals['SQ_INSTS_LDS']:.2f}, "
          f"unaligned stall {vals.get('SQ_LDS_UNALIGNED_STALL', 0) / vals['SQ_INSTS_LDS']:.2f}")
if "SQ_VALU_MFMA_BUSY_CYCLES" in vals and "SQ_BUSY_CU_CYCLES" in vals:
    print(f"  MFMA busy / CU busy {vals['SQ_VALU_MFMA_BUSY_CYCLES'] / vals['SQ_BUSY_CU_CYCLES']:.3f}")
if "SQ_INST_LEVEL_VMEM" in vals and "SQ_INSTS_VMEM_RD" in vals:
    print(f"  avg VMEM latency (level/insts) {vals['SQ_INST_LEVEL_VMEM'] / vals['SQ_INSTS_VMEM_RD']:.0f} cycles")
if "TCC_HIT_sum" in vals:
    h, m = vals["TCC_HIT_sum"], vals["TCC_MISS_sum"]
    print(f"  L2 hit {h / max(h + m, 1):.3f}  EA rd req {vals.get('TCC_EA0_RDREQ_sum', 0):.3e}")
