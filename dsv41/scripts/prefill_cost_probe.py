"""Prefill cost vs cached prefix length, for calibrating SGLANG_PREFILL_DECODE_* (cost-scaled PDI).
Warms a random prefix of L tokens into the radix cache, then times prefix + NEW fresh tokens (max_new_tokens=1).
  python3 prefill_cost_probe.py --port 8888 [--new 16384] [--prefix-k 0,32,128,256,512] [--reps 3]
Prints one line per L: median ms, ms per 1k new tokens, ratio to L=0 (-> PREFIX_KTOK_SCALE = L_k / (ratio - 1))."""
import argparse, random, statistics, time
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8888)
ap.add_argument("--new", type=int, default=16384)
ap.add_argument("--prefix-k", default="0,32,128,256,512")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--vocab", type=int, default=120000)
a = ap.parse_args()
url = f"http://localhost:{a.port}/generate"
rng = random.Random(0)


def gen(ids):
    t = time.perf_counter()
    r = requests.post(url, json={"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0}},
                      timeout=3600)
    r.raise_for_status()
    return (time.perf_counter() - t) * 1e3, r.json()["meta_info"].get("cached_tokens")


rand = lambda n: [rng.randrange(1000, a.vocab) for _ in range(n)]
base = None
for lk in [int(x) for x in a.prefix_k.split(",")]:
    prefix = rand(lk * 1024)
    if prefix:
        gen(prefix)
    ms, cached = [], []
    for _ in range(a.reps):
        t, c = gen(prefix + rand(a.new))
        ms.append(t); cached.append(c)
    m = statistics.median(ms)
    base = base or m
    print(f"prefix {lk:4d}k  new {a.new}  median {m:8.1f} ms  ({m / (a.new / 1000):6.1f} ms/ktok)  "
          f"ratio {m / base:5.2f}  cached {cached}  all {[round(x) for x in ms]}", flush=True)
