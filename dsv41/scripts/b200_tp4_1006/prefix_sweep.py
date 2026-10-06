"""P1a: cached-prefix TTFT sweep (conc 1). usage: prefix_sweep.py [L,L,..] [N,N,..]"""
import json, random, re, statistics, sys, time, urllib.request

URL = "http://127.0.0.1:8000"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
REPS = 3
LO, HI = 1000, 120000  # vocab_size 129280; stay clear of special/added tokens


def metrics():
    txt = urllib.request.urlopen(URL + "/metrics", timeout=30).read().decode()
    out = {}
    for k in ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"):
        m = re.findall(r"^" + re.escape(k) + r"(?:\{[^}]*\})? ([0-9.e+]+)$", txt, re.M)
        out[k] = sum(float(v) for v in m)
    return out


def ttft(ids):
    body = json.dumps({"model": MODEL, "prompt": ids, "max_tokens": 1, "stream": True,
                       "temperature": 0, "ignore_eos": True}).encode()
    req = urllib.request.Request(URL + "/v1/completions", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            if first is None and d.get("choices") and d["choices"][0].get("text") is not None:
                first = time.perf_counter()
    return ((first or time.perf_counter()) - t0) * 1e3


def toks(rng, n):
    return [rng.randrange(LO, HI) for _ in range(n)]


def main():
    LS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "32768,65536,131072,262144").split(",")]
    NS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "512,4096").split(",")]
    print("L,N,rep,ttft_ms,hit_tokens,query_tokens", flush=True)
    res = {}
    for li, L in enumerate(LS):
        for N in NS:
            rng = random.Random(1000 * li + N)
            P = toks(rng, L)
            ttft(P)  # warm the prefix cache
            vals = []
            for rep in range(REPS):
                m0 = metrics()
                t = ttft(P + toks(rng, N))
                m1 = metrics()
                h = m1["vllm:prefix_cache_hits_total"] - m0["vllm:prefix_cache_hits_total"]
                q = m1["vllm:prefix_cache_queries_total"] - m0["vllm:prefix_cache_queries_total"]
                print(f"{L},{N},{rep},{t:.1f},{h:.0f},{q:.0f}", flush=True)
                vals.append(t)
            res[(L, N)] = statistics.median(vals)
    print("SUMMARY L,N,median_ttft_ms")
    for (L, N), v in res.items():
        print(f"SUMMARY {L},{N},{v:.1f}")
    if len(LS) > 1:
        for N in NS:
            xs = [L / 1024 for L in LS]; ys = [res[(L, N)] for L in LS]
            mx, my = statistics.mean(xs), statistics.mean(ys)
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
            print(f"FIT N={N}: TTFT = {my - b * mx:.1f} ms + {b:.3f} ms per 1k prefix")


if __name__ == "__main__":
    main()
