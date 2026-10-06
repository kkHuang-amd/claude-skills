#!/usr/bin/env python3
"""Cached-prefix TTFT sweep (conc 1) -- the MI355X side of B200_REQUEST_1006.md Deliverable 1b.

    python3 -I prefix_ttft_sweep.py --port 8888 [--api generate|completions] [--prefix 32768,65536,131072,262144]
                                    [--new 512,4096] [--reps 3] [--timeout 300]

For each prefix length L: send P_L (random ids, 1 output token) to warm the prefix cache, then P_L + N fresh random ids
`reps` times and record TTFT (time to the first streamed token). Prints an L x N table of median TTFT, the reported
cached-token count, and the fit TTFT = a + b x L per N. Every request has a timeout so a server hang fails fast.
"""
import argparse
import json
import random
import statistics as st
import sys
import time

import requests


def ttft(a, ids):
    if a.api == "chat":  # TEXT prompt -> the server tokenizes it, like AgentX
        url, body = f"{a.base}/v1/chat/completions", {
            "model": "default", "messages": [{"role": "user", "content": a.tok.decode(ids)}], "max_tokens": 1,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}}
    elif a.api == "generate":
        url, body = f"{a.base}/generate", {"input_ids": ids, "stream": True,
                                          "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}}
    else:
        url, body = f"{a.base}/v1/completions", {"model": "default", "prompt": ids, "max_tokens": 1,
                                                 "temperature": 0.0, "stream": True,
                                                 "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); first = None; cached = None
    with requests.post(url, json=body, stream=True, timeout=a.timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            ch = (d.get("choices") or [{}])[0]
            got = d.get("text") or ch.get("text") or (ch.get("delta") or {}).get("content") or ch.get("finish_reason")
            if first is None and got:
                first = time.perf_counter()
            mi = d.get("meta_info") or {}
            usage = d.get("usage") or {}
            cached = mi.get("cached_tokens", cached)
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", cached)
    if first is None:
        raise RuntimeError("no token streamed")
    return (first - t0) * 1e3, cached


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8888)
    p.add_argument("--api", choices=["generate", "completions", "chat"], default="generate")
    p.add_argument("--model-path", default="/shared_nfs/models/deepseek-ai/DeepSeek-V4.1-Flash")
    p.add_argument("--prefix", default="32768,65536,131072,262144")
    p.add_argument("--new", default="512,4096")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.base = f"http://127.0.0.1:{a.port}"
    Ls = [int(x) for x in a.prefix.split(",")]; Ns = [int(x) for x in a.new.split(",")]
    rng = random.Random(a.seed)
    rid = lambda n: [rng.randrange(1000, 100000) for _ in range(n)]
    if a.api == "chat":  # real text (sglang sources) so decode -> server re-tokenize stays ~L tokens
        import glob
        from transformers import AutoTokenizer
        a.tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True)
        src = "".join(open(f, errors="ignore").read() for f in
                      sorted(glob.glob("/sgl-workspace/sglang/python/sglang/srt/**/*.py", recursive=True))[:400])
        pool = a.tok(src, add_special_tokens=False)["input_ids"]
        need = max(Ls) + len(Ns) * a.reps * max(Ns)
        pool = (pool * (need // len(pool) + 1))[:need]
        off = [max(Ls)]  # suffixes come from a region the prefixes never use

        def rid(n, _src=pool):
            if n in Ls:
                return _src[:n]
            s = _src[off[0]:off[0] + n]; off[0] += n
            return s
    res = {}
    for L in Ls:
        P = rid(L)
        w, wc = ttft(a, P)  # cold prefill of the prefix; warms the cache
        print(f"L={L:>7} warm (cold prefill) {w:8.0f} ms cached={wc}", flush=True)
        for N in Ns:
            ts, cs = [], []
            for _ in range(a.reps):
                t, c = ttft(a, P + rid(N)); ts.append(t); cs.append(c)
            res[(L, N)] = st.median(ts)
            print(f"L={L:>7} N={N:>5} TTFT ms median {st.median(ts):7.0f} (all {', '.join(f'{x:.0f}' for x in ts)}) "
                  f"cached={cs}", flush=True)
    print(f"\n| prefix L | " + " | ".join(f"N={N} TTFT ms" for N in Ns) + " |")
    print("|---|" + "---|" * len(Ns))
    for L in Ls:
        print(f"| {L} | " + " | ".join(f"{res[(L, N)]:.0f}" for N in Ns) + " |")
    for N in Ns:
        xs = [L / 1000 for L in Ls]; ys = [res[(L, N)] for L in Ls]
        mx, my = st.mean(xs), st.mean(ys)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        print(f"N={N}: TTFT = {my - b * mx:.0f} ms + {b:.2f} ms per 1k prefix")


if __name__ == "__main__":
    sys.exit(main())
