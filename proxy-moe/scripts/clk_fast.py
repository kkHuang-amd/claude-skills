# Fast per-socket GFXCLK sampler: sysfs freq1_input (only the first partition of each socket exposes it) every PERIOD s.
# Usage: clk_fast.py <out.csv> <seconds> [period_s=0.001]
import glob, sys, time
out, dur = sys.argv[1], float(sys.argv[2]); per = float(sys.argv[3]) if len(sys.argv) > 3 else 0.001
fs = sorted(glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*/freq1_input"), key=lambda f: int(f.split("/card")[1].split("/")[0]))
with open(out, "w") as o:
    o.write("t," + ",".join(f.split("/")[4] for f in fs) + "\n")
    end = time.time() + dur
    while (t := time.time()) < end:
        o.write(f"{t:.6f}," + ",".join(str(int(open(f).read()) // 1000000) for f in fs) + "\n")
        time.sleep(max(0, per - (time.time() - t)))
