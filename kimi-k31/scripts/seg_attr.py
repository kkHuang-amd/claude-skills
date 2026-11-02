#!/usr/bin/env python3
"""Per-segment attribution of a wide VERIFY step.
Usage: seg_attr.py <TP0 trace.json.gz> [show_seg,...]
Segment i ends with layer i's MoE stage2. Kinds: L = has a k31 layer launch, D = no launch.
Buckets per segment: launch, glue (at::native gather/index/cat), moe (topk..stage2), rest."""
import gzip, json, re, sys
from collections import defaultdict

ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
show = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 and sys.argv[2] else []
KIND = sys.argv[3] if len(sys.argv) > 3 else None
steps = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev
               if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("step[VERIFY"))
ks_all = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
                key=lambda e: e["ts"])
DELIM = re.compile(r"opus_moe_stage2|moe_stage2|down_ws_kernel")
LAUNCH = re.compile(r"k31_kda_layer")
GLUE = re.compile(r"at::native::.*(gather|index_elementwise|CatArrayBatchedCopy)")
MOE = re.compile(r"topk_reg|moe_sorting|fused_mx_quant_moe_sort|mfma_moe1|moe_stage2")

def segs(a, b):
    out, cur = [], []
    for e in ks_all:
        if a <= e["ts"] <= b:
            cur.append(e)
            if DELIM.search(e["name"]):
                out.append(cur); cur = []
    out.append(cur)
    return out

use = steps[5:-2]
S = [segs(a, b) for a, b in use]
n = min(len(s) for s in S)
kind = list(KIND)[:n] if KIND else ["L" if any(LAUNCH.search(e["name"]) for e in S[0][i]) else "D" for i in range(n)]
tot = defaultdict(lambda: defaultdict(float))
rest_names = defaultdict(lambda: defaultdict(float))
for s in S:
    for i in range(n):
        prev = kind[i - 1] if i else "-"
        k = kind[i] + ("first" if kind[i] == "L" and prev != "L" else "after_run" if kind[i] == "D" and prev == "L" else "")
        for e in s[i]:
            nm = e["name"]
            b = "launch" if LAUNCH.search(nm) else "glue" if GLUE.search(nm) else "moe" if MOE.search(nm) else "rest"
            tot[k][b] += e["dur"] / len(S)
            tot[k]["_n"] += 0
            if b == "rest":
                rest_names[k][re.sub(r"<.*|\(.*", "", nm)[:70]] += e["dur"] / len(S)
cnt = defaultdict(int)
for i in range(n):
    prev = kind[i - 1] if i else "-"
    k = kind[i] + ("first" if kind[i] == "L" and prev != "L" else "after_run" if kind[i] == "D" and prev == "L" else "")
    cnt[k] += 1
print(f"steps used {len(S)}, segments {n}, kinds: {''.join(kind)}")
json.dump({k: dict(v) for k, v in tot.items()}, open(sys.argv[1] + ".seg.json", "w"))
print(f"{'kind':12s} {'n':>3s} {'launch':>8s} {'glue':>8s} {'moe':>8s} {'rest':>8s}   ms/step (sum over segments)")
for k in sorted(tot):
    t = tot[k]
    print(f"{k:12s} {cnt[k]:3d} {t['launch']/1e3:8.3f} {t['glue']/1e3:8.3f} {t['moe']/1e3:8.3f} {t['rest']/1e3:8.3f}")
for k in sorted(rest_names):
    top = sorted(rest_names[k].items(), key=lambda x: -x[1])[:8]
    print(f"\nrest in {k} (us/step, summed over {cnt[k]} segments):")
    for nm, us in top:
        print(f"  {us:8.1f}  {nm}")
for i in show:
    print(f"\n-- segment {i} ({kind[i]}), step {len(S)//2}:")
    for e in S[len(S) // 2][i]:
        print(f"  {e['dur']:7.1f}  {re.sub(r'<.*|\(.*', '', e['name'])[:80]}")
