"""Per-instruction-class time (stall + duration) over the innermost loop containing the kernel's last MFMA,
the loop = from the target of the back-edge after the last MFMA to that branch. Usage: <ui_dir>..."""
import glob, json, os, re, sys, collections

for D in sys.argv[1:]:
    code = json.load(open(os.path.join(D, "code.json")))["code"]
    by_idx = {r[2]: r for r in code}
    order = sorted(by_idx)
    asm = {i: by_idx[i][0].strip() for i in order}
    mf = [i for i in order if asm[i].startswith("v_mfma")]
    last = mf[-1]
    br = next(i for i in order if i > last and asm[i].startswith("s_cbranch"))
    off = int(asm[br].split()[-1]); off = off - 65536 if off >= 32768 else off  # simm16 dwords
    tgt = by_idx[br][5] + 4 + 4 * off
    lab = next(i for i in order if by_idx[i][5] == tgt)
    lo, hi = lab, br
    cls = collections.Counter(); stall = collections.Counter(); per_pc = collections.Counter(); win = 0; nwave = 0
    for f in glob.glob(os.path.join(D, "se*_wv*.json")):
        ins = json.load(open(f))["wave"]["instructions"]
        ev = [e for e in ins if lo <= e[4] <= hi]
        if not ev:
            continue
        nwave += 1
        win += ev[-1][0] - ev[0][0]
        for t, _, st, du, i in ev:
            a = asm[i]
            k = ("s_waitcnt vmcnt" if a.startswith("s_waitcnt") and "vmcnt" in a else
                 "s_waitcnt lgkm" if a.startswith("s_waitcnt") else
                 "s_barrier" if a.startswith("s_barrier") else
                 "buffer_load" if a.startswith("buffer_load") else
                 "v_mfma" if a.startswith("v_mfma") else
                 "ds_" if a.startswith("ds_") else
                 "salu" if a.startswith("s_") else "valu/other")
            cls[k] += st + du; stall[k] += st; per_pc[i] += st
    tot = sum(cls.values())
    print(f"== {os.path.basename(D)}: DOWN loop idx {lo}..{hi}, waves {nwave}, mean loop window {win / max(nwave,1):.0f} cyc, "
          f"sum(stall+dur)/wave {tot / max(nwave,1):.0f}")
    for k, v in cls.most_common():
        print(f"   {k:16s} {v / nwave:9.0f} cyc/wave ({v / tot * 100:4.1f} %)  stall {stall[k] / nwave:9.0f}")
    print("   top stall PCs:", "; ".join(f"{asm[i][:40]} {v / nwave:.0f}" for i, v in per_pc.most_common(6)))
