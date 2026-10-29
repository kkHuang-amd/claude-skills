# Per-GPU clock/power per concurrency window. Windows = seconds where server log "Decode batch ... #running-req: N" has N in band.
import csv, re, sys, statistics as st
from datetime import datetime
d = sys.argv[1]
run = {}
for l in open(f"{d}/server.log", errors="ignore"):
    m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\]]*\] Decode batch.*#running-req: (\d+)", l)
    if m: run[int(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp())] = int(m.group(2))
bands = {"c2": (2, 2), "c16": (15, 16), "c64": (60, 64)}
rows = [r for r in csv.DictReader(l for l in open(f"{d}/clk.csv") if not l.startswith("#") and not l.startswith("t,gpu") or l.startswith("t,gpu,power") and not hasattr(sys, "_hdr") and not setattr(sys, "_hdr", 1))]
f = lambda x: float(x) if x not in ("N/A", "None", "") else None
for name, (lo, hi) in bands.items():
    print(f"== {name}  (seconds with running-req in [{lo},{hi}]: {sum(lo <= v <= hi for v in run.values())})")
    print("gpu  clk_mean  clk_p10  clk_p90   power_W")
    for g in range(8):
        cl, pw = [], []
        for r in rows:
            if r["gpu"] != str(g): continue
            v = run.get(int(float(r["t"])))
            if v is None or not lo <= v <= hi: continue
            if f(r["clk"]) is not None: cl.append(f(r["clk"]))
            if f(r["power"]) is not None: pw.append(f(r["power"]))
        if cl:
            q = st.quantiles(cl, n=10) if len(cl) > 1 else [cl[0]] * 9
            print(f"{g:>3} {st.mean(cl):9.0f} {q[0]:8.0f} {q[-1]:8.0f}  {st.mean(pw) if pw else float('nan'):8.0f}")
# PPT throttle share per window: sum of per-second deltas of ppt_accumulated / accumulation_counter while in band.
print("== PPT throttle share per window (even GPU = socket)")
for name, (lo, hi) in bands.items():
    out = []
    for g in range(0, 8, 2):
        rr = [r for r in rows if r["gpu"] == str(g) and f(r["cnt"]) is not None and f(r["ppt"]) is not None]
        num = den = 0
        for a, b in zip(rr, rr[1:]):
            v = run.get(int(float(b["t"])))
            if v is None or not lo <= v <= hi: continue
            num += f(b["ppt"]) - f(a["ppt"]); den += f(b["cnt"]) - f(a["cnt"])
        out.append(f"GPU{g} {100*num/den:.0f}%" if den else f"GPU{g} n/a")
    print(name, " ".join(out))
