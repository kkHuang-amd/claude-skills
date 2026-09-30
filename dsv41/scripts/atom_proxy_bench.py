"""ATOM counterpart of atomport_proxy_bench.py: c1 random-id prompt via /v1/completions (prompt_token_ids),
streamed; decode tok/s = (completion_tokens - first-chunk tokens) / (t_last - t_first).
  python3 atom_proxy_bench.py --port 8000 --ctx 2048 --osl 1024 --repeat 3 --temperature 1.0 --tag x"""
import argparse, json, random, time, requests

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--osl", type=int, default=1024)
ap.add_argument("--repeat", type=int, default=3)
ap.add_argument("--temperature", type=float, default=None)  # None = omit (server default, like AgentX)
ap.add_argument("--model", default="/shared_nfs/deepseek-ai/DeepSeek-V4.1-Flash")
ap.add_argument("--tag", default="x")
ap.add_argument("--out", default="/shared_nfs/kk/dsv41/atomport/atom_proxy.tsv")
a = ap.parse_args()

for rep in range(a.repeat):
    ids = [random.Random(rep).randint(1000, 100000) for _ in range(a.ctx)]
    body = {"model": a.model, "prompt_token_ids": ids, "max_tokens": a.osl, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    if a.temperature is not None:
        body["temperature"] = a.temperature
    t0 = time.perf_counter(); stamps = []; usage = None
    with requests.post(f"http://127.0.0.1:{a.port}/v1/completions", json=body, stream=True, timeout=3600) as r:
        for line in r.iter_lines():
            if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and d["choices"][0].get("text"):
                stamps.append(time.perf_counter())
    n = (usage or {}).get("completion_tokens", 0)
    tps = (n - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else float("nan")
    line = (f"{a.tag}\tctx={a.ctx}\ttemp={a.temperature}\trep={rep}\tdecode_tps_per_req={tps:.1f}"
            f"\tchunks={len(stamps)}\ttokens={n}\tttft_s={stamps[0]-t0 if stamps else float('nan'):.2f}")
    print(line, flush=True)
    with open(a.out, "a") as f:
        f.write(time.strftime("%F %T") + "\t" + line + "\n")
