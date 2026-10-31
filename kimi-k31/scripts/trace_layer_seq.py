#!/usr/bin/env python3
"""Print the kernel sequence of decoder layers inside one step.
Layers are delimited by the MoE stage-2 kernel (one per layer).
Usage: trace_layer_seq.py <trace.json.gz> [STEP=VERIFY] [step_idx=10] [layers=4,5]"""
import gzip, json, re, sys

ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
ST = sys.argv[2] if len(sys.argv) > 2 else "VERIFY"
si = int(sys.argv[3]) if len(sys.argv) > 3 else 10
want = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "4,5").split(",")]
steps = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev
               if e.get("cat") == "gpu_user_annotation" and e["name"].startswith(f"step[{ST}"))
a, b = steps[si]
ks = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
             and a <= e["ts"] <= b), key=lambda e: e["ts"])
DELIM = re.compile(r"opus_moe_stage2|moe_stage2|down_ws_kernel")
layers, cur = [], []
for e in ks:
    cur.append(e)
    if DELIM.search(e["name"]):
        layers.append(cur); cur = []
layers.append(cur)
print(f"{ST} step {si}: {len(ks)} kernels, {len(layers)} segments (segment i ends with layer i's MoE stage2)")
for li in want:
    L = layers[li]
    print(f"\n-- segment {li}: {len(L)} kernels, {sum(e['dur'] for e in L):.1f} us")
    for e in L:
        print(f"  {e['dur']:7.1f}  {re.sub(r'<.*', '', e['name'])[:95]}")
