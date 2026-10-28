# Split the hot-loop stall+duration of a gfx1250 ATT trace by exact opcode (s_wait_* kinds, barrier, wmma, ...).
# Usage: att_wait_breakdown.py <ui_output_dir> [lo hi]   (lo/hi = instruction idx range; default: the loop att_hot_loop picks)
import collections, glob, json, os, sys
D = sys.argv[1]
code = json.load(open(os.path.join(D, "code.json")))["code"]
asm = {r[2]: r[0].strip() for r in code}
if len(sys.argv) > 3:
    lo, hi = int(sys.argv[2]), int(sys.argv[3])
else:
    lo, hi = min(asm), max(asm)
st = collections.Counter(); du = collections.Counter()
for f in glob.glob(os.path.join(D, "se*_wv*.json")):
    for t, _, s, d, i in json.load(open(f))["wave"]["instructions"]:
        if lo <= i <= hi:
            op = asm[i].split()[0]
            st[op] += s; du[op] += d
tot = sum(st.values()) + sum(du.values())
for op in sorted(set(st) | set(du), key=lambda o: -(st[o] + du[o]))[:14]:
    print(f"{op:28s} {100 * (st[op] + du[op]) / tot:5.1f}%  (stall {100 * st[op] / tot:4.1f}%)")
