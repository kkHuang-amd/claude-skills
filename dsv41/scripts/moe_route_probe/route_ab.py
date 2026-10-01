#!/usr/bin/env python3
"""Distinct routed experts per MoE call vs batch size, using moe_route_probe stats (eager servers).
For each (temperature, bs): snapshot the probe file, run decode_bs_sweep.py (one rep), wait for the probe flush,
diff the cumulative stats and print mean distinct experts per call for every (E, num_tokens) key seen in the window
(target MoE = E 384/385, DSpark draft = E 128/129; decode verify keys have small num_tokens).
  python3 route_ab.py --engine sglang --port 8888 --probe '/shared_nfs/kk/dsv41/atomport/moe_probe/sgl.*'
"""
import argparse, glob, json, subprocess, sys, time, os

ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True)
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--probe", required=True, help="glob of probe files; the lowest pid is used")
ap.add_argument("--bs", default="1,4,6")
ap.add_argument("--temps", default="0,default")
ap.add_argument("--ctx", type=int, default=8192)
ap.add_argument("--osl", type=int, default=512)
ap.add_argument("--max-tokens-key", type=int, default=64, help="only print keys with num_tokens <= this")
a = ap.parse_args()
SWEEP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "decode_bs_sweep.py")


def snap():
    f = sorted(glob.glob(a.probe), key=lambda p: int(p.rsplit(".", 1)[1]))[0]
    return json.load(open(f))["stats"]


for temp in a.temps.split(","):
    for bs in [int(x) for x in a.bs.split(",")]:
        before = snap()
        cmd = [sys.executable, SWEEP, "--engine", a.engine, "--port", str(a.port), "--ctx", str(a.ctx),
               "--bs", str(bs), "--osl", str(a.osl), "--repeat", "1", "--tag", f"route_{a.engine}",
               "--out", "/tmp/route_ab.tsv"]
        if temp != "default":
            cmd += ["--temperature", temp]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        time.sleep(5)
        after = snap()
        rows = []
        for k, v in after.items():
            E, n = map(int, k.split(":"))
            b = before.get(k, [0, 0, 0, 0])
            dc, ds = v[0] - b[0], v[1] - b[1]
            if dc > 0 and n <= a.max_tokens_key:
                rows.append((E, n, dc, ds / dc))
        rows.sort()
        txt = "  ".join(f"E{E}/T{n}: {m:.1f} (n={c})" for E, n, c, m in rows if c >= 20)
        print(f"{a.engine} temp={temp} bs={bs}  {txt}", flush=True)
