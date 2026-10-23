"""Sample socket power and mean GFX clock of every GPU once a second (amd-smi), or summarize.

Usage: python3 gpu_power_sample.py sample2 <out.csv>         (runs until killed; v2 adds throttle)
       python3 gpu_power_sample.py summarize2 <out.csv> <server.log> [--min-running 60]
       python3 gpu_power_sample.py sample <out.csv>          (v1, power + clock only)
       python3 gpu_power_sample.py summarize <out.csv> <server.log> [--min-running 60]
summarize keeps samples inside the decode window: first..last 'Decode batch' log line whose
'#running-req' >= --min-running, and prints per-GPU median power / clock plus the all-GPU median.
"""
import json, re, statistics as st, subprocess, sys, time
from datetime import datetime

def sample(out):
    with open(out, "a") as f:
        while True:
            t = time.time()
            try:
                d = json.loads(subprocess.run(["amd-smi", "metric", "-p", "-c", "--json"],
                                              capture_output=True, text=True, timeout=10).stdout)
                rows = d["gpu_data"] if isinstance(d, dict) else d
                for g in rows:
                    sp = g["power"]["socket_power"]
                    if not isinstance(sp, dict):  # "N/A" on some GPUs/samples
                        f.write(f"# {t:.1f} gpu {g.get('gpu')} socket_power {sp}\n"); continue
                    p = sp["value"]
                    clks = [v["clk"]["value"] for k, v in g["clock"].items()
                            if k.startswith("gfx_") and isinstance(v, dict) and isinstance(v.get("clk"), dict)
                            and isinstance(v["clk"].get("value"), (int, float))]
                    f.write(f"{t:.1f},{g['gpu']},{p},{st.mean(clks) if clks else 'nan'}\n")
                f.flush()
            except Exception as e:  # keep sampling through transient amd-smi errors
                f.write(f"# {t:.1f} error {e}\n"); f.flush()
            time.sleep(max(0, 1 - (time.time() - t)))

def summarize(csv, log, min_running=60):
    ts = []
    for line in open(log, errors="ignore"):
        m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\]]*\] Decode batch.*#running-req: (\d+)", line)
        if m and int(m.group(2)) >= min_running:
            ts.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp())
    if not ts:
        print("no decode window found"); return
    lo, hi = min(ts), max(ts)
    per = {}
    for line in open(csv):
        if line.startswith("#"): continue
        t, g, p, c = line.strip().split(",")
        if lo <= float(t) <= hi:
            per.setdefault(int(g), []).append((float(p), float(c)))
    print(f"decode window {hi - lo:.0f}s, {sum(len(v) for v in per.values())} samples")
    allp, allc = [], []
    for g in sorted(per):
        ps, cs = [x[0] for x in per[g]], [x[1] for x in per[g]]
        allp += ps; allc += cs
        print(f"  GPU{g}: power med {st.median(ps):.0f} W (max {max(ps):.0f})  gfx clk med {st.median(cs):.0f} MHz (min {min(cs):.0f})")
    print(f"ALL: power med {st.median(allp):.0f} W  gfx clk med {st.median(allc):.0f} MHz")

THR = ["ppt", "socket_thermal", "vr_thermal", "hbm_thermal", "prochot"]

def _v(x):
    return x.get("value") if isinstance(x, dict) else x

def sample2(out):
    """Per GPU per second: power, mean gfx clk, gfx activity, hotspot, accumulation_counter and
    *_accumulated throttle counters. On MI455 each socket is 2 GPUs (partitions); socket-level
    power/throttle/temperature are only reported on the even GPU, the odd one gives N/A."""
    with open(out, "a") as f:
        f.write("t,gpu,power,clk,gfx_act,hotspot,cnt," + ",".join(THR) + "\n")
        while True:
            t = time.time()
            try:
                d = json.loads(subprocess.run(["amd-smi", "metric", "-p", "-c", "-t", "-u", "--throttle", "--json"],
                                              capture_output=True, text=True, timeout=10).stdout)
                for g in d["gpu_data"]:
                    get = lambda k: g.get(k) if isinstance(g.get(k), dict) else {}
                    th, ck = get("throttle"), get("clock")
                    clks = [v["clk"]["value"] for k, v in ck.items() if k.startswith("gfx_") and isinstance(v, dict)
                            and isinstance(v.get("clk"), dict) and isinstance(v["clk"].get("value"), (int, float))]
                    row = [f"{t:.1f}", g["gpu"], _v(get("power").get("socket_power")), f"{st.mean(clks):.0f}" if clks else "N/A",
                           _v(get("usage").get("gfx_activity")), _v(get("temperature").get("hotspot")), th.get("accumulation_counter")]
                    row += [th.get(f"{k}_accumulated") for k in THR]
                    f.write(",".join(str(x) for x in row) + "\n")
                f.flush()
            except Exception as e:
                f.write(f"# {t:.1f} error {e}\n"); f.flush()
            time.sleep(max(0, 1 - (time.time() - t)))

def summarize2(csv, log, min_running=60):
    ts = [datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
          for m in (re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\]]*\] Decode batch.*#running-req: (\d+)", l)
                    for l in open(log, errors="ignore")) if m and int(m.group(2)) >= min_running]
    if not ts:
        print("no decode window found"); return
    lo, hi = min(ts), max(ts)
    rows = {}
    for line in open(csv):
        if line.startswith(("#", "t,")): continue
        c = line.strip().split(",")
        if lo <= float(c[0]) <= hi:
            rows.setdefault(int(c[1]), []).append(c)
    num = lambda x: float(x) if x not in ("N/A", "None", "") else None
    print(f"decode window {hi - lo:.0f}s")
    for g in sorted(rows):
        r = rows[g]
        clk = [num(c[3]) for c in r if num(c[3]) is not None]
        act = [num(c[4]) for c in r if num(c[4]) is not None]
        line = f"  GPU{g}: clk med {st.median(clk):.0f} MHz" if clk else f"  GPU{g}: clk N/A"
        line += f", gfx_act med {st.median(act):.0f}%" if act else ""
        pw = [num(c[2]) for c in r if num(c[2]) is not None]
        if pw:
            hs = [num(c[5]) for c in r if num(c[5]) is not None]
            first, last = r[0], r[-1]
            dc = num(last[6]) - num(first[6])
            fr = {k: (num(last[7 + i]) - num(first[7 + i])) / dc for i, k in enumerate(THR) if dc and num(last[7 + i]) is not None}
            line += f" | SOCKET power med {st.median(pw):.0f} W (max {max(pw):.0f}), hotspot med {st.median(hs):.0f} C"
            line += " | throttled: " + " ".join(f"{k}={100 * v:.0f}%" for k, v in fr.items())
        print(line)

if __name__ == "__main__":
    if sys.argv[1] == "sample":
        sample(sys.argv[2])
    elif sys.argv[1] == "sample2":
        sample2(sys.argv[2])
    elif sys.argv[1] == "summarize2":
        mr = int(sys.argv[sys.argv.index("--min-running") + 1]) if "--min-running" in sys.argv else 60
        summarize2(sys.argv[2], sys.argv[3], mr)
    else:
        mr = int(sys.argv[sys.argv.index("--min-running") + 1]) if "--min-running" in sys.argv else 60
        summarize(sys.argv[2], sys.argv[3], mr)
