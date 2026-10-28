# gfx1250 ATT hot-loop breakdown: pick the backward branch whose loop body holds the most v_wmma, then sum per-instruction-class
# stall and duration over that PC range across all traced waves. Usage: att_hot_loop.py <ui_output_dir> [top_pcs]
import collections, glob, json, os, re, sys

D = sys.argv[1]; top = int(sys.argv[2]) if len(sys.argv) > 2 else 12
code = json.load(open(os.path.join(D, "code.json")))["code"]
by_idx = {r[2]: r for r in code}
order = sorted(by_idx)
asm = {i: by_idx[i][0].strip() for i in order}
addr = {i: by_idx[i][5] for i in order}
idx_of_addr = {addr[i]: i for i in order}
loops = []
for i in order:
    if asm[i].startswith("s_cbranch") or asm[i].startswith("s_branch"):
        try:
            off = int(asm[i].split()[-1])
        except ValueError:
            continue
        off = off - 65536 if off >= 32768 else off
        if off >= 0:
            continue
        tgt = addr[i] + 4 + 4 * off
        if tgt in idx_of_addr:
            lo = idx_of_addr[tgt]
            n = sum(1 for j in order if lo <= j <= i and asm[j].startswith("v_wmma"))
            loops.append((n, lo, i))
hits = collections.Counter()
for f in glob.glob(os.path.join(D, "se*_wv*.json")):
    for e in json.load(open(f))["wave"]["instructions"]:
        hits[e[4]] += 1
loops = [(sum(hits[j] for j in range(lo, hi + 1)), n, lo, hi) for n, lo, hi in loops if n > 0]
loops.sort(reverse=True)
print("loops by executed instrs (events, n_wmma, lo, hi):", loops[:4])
if loops:
    _, n_wmma, lo, hi = loops[0]
else:  # fully unrolled kernel (e.g. K=512): whole program
    lo, hi = order[0], order[-1]; n_wmma = sum(1 for j in order if asm[j].startswith("v_wmma"))
    print("no WMMA loop: whole-kernel breakdown")


def klass(s):
    op = s.split()[0]
    for k, p in (("wmma", "v_wmma"), ("tensor/tdm", "tensor_"), ("s_wait_tensorcnt", "s_wait_tensorcnt"),
                 ("s_wait_dscnt", "s_wait_dscnt"), ("s_wait_other", "s_wait"), ("barrier", "s_barrier"),
                 ("ds_load", "ds_load"), ("ds_store", "ds_store"), ("ds_other", "ds_"), ("buffer/global", ("buffer_", "global_")),
                 ("valu", "v_"), ("salu", "s_")):
        if op.startswith(p):
            return k
    return "other"


cls_stall = collections.Counter(); cls_dur = collections.Counter(); cls_n = collections.Counter(); pc = collections.Counter()
win = 0; nwave = 0
for f in glob.glob(os.path.join(D, "se*_wv*.json")):
    ins = json.load(open(f))["wave"]["instructions"]
    ev = [e for e in ins if lo <= e[4] <= hi]
    if not ev:
        continue
    nwave += 1
    win += ev[-1][0] - ev[0][0]
    for t, _, st, du, i in ev:
        k = klass(asm[i]); cls_stall[k] += st; cls_dur[k] += du; cls_n[k] += 1; pc[i] += st
tot = sum(cls_stall.values()) + sum(cls_dur.values())
print(f"hot loop idx {lo}..{hi}: {hi - lo + 1} instrs, {n_wmma} v_wmma; waves {nwave}, mean loop window {win / max(nwave, 1):.0f} cyc")
print(f"{'class':18s} {'count':>8s} {'stall%':>7s} {'dur%':>6s}  (share of stall+dur over the loop, all waves)")
for k in sorted(set(cls_stall) | set(cls_dur), key=lambda k: -(cls_stall[k] + cls_dur[k])):
    print(f"{k:18s} {cls_n[k]:8d} {100 * cls_stall[k] / tot:6.1f}% {100 * cls_dur[k] / tot:5.1f}%")
print("top stall PCs:")
for i, s in pc.most_common(top):
    print(f"  {100 * s / tot:5.1f}%  idx {i}: {asm[i][:90]}")
