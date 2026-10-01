#!/usr/bin/env python3
"""Compare generated token streams (SGLang /generate output_ids vs ATOM /v1/completions return_token_ids) for the
same random-id prompts: unique-token ratio and top repeated ids. Detects degenerate (repeating) decode streams,
which collapse MoE routing to a few experts.
  python3 token_stream_check.py --ctx 8192 --osl 200 --seeds 0,1,2,3 [--temperature 0]
"""
import argparse, collections, random, requests

ap = argparse.ArgumentParser()
ap.add_argument("--ctx", type=int, default=8192)
ap.add_argument("--osl", type=int, default=200)
ap.add_argument("--seeds", default="0,1,2,3")
ap.add_argument("--temperature", type=float, default=None)
ap.add_argument("--sgl-port", type=int, default=8888)
ap.add_argument("--atom-port", type=int, default=8000)
ap.add_argument("--model", default="/shared_nfs/deepseek-ai/DeepSeek-V4.1-Flash")
ap.add_argument("--atom-token-log", default="/shared_nfs/kk/atom_run/moe_probe/atom.tokens.jsonl",
                help="written by moe_route_probe's delivered_text hook (ATOM API text is synthetic)")
a = ap.parse_args()


def sgl(ids):
    sp = {"max_new_tokens": a.osl, "ignore_eos": True}
    if a.temperature is not None:
        sp["temperature"] = a.temperature
    r = requests.post(f"http://127.0.0.1:{a.sgl_port}/generate", json={"input_ids": ids, "sampling_params": sp},
                      timeout=3600).json()
    return r["output_ids"]


def atom(ids):
    body = {"model": a.model, "prompt_token_ids": ids, "max_tokens": a.osl, "ignore_eos": True,
            "return_token_ids": True}
    if a.temperature is not None:
        body["temperature"] = a.temperature
    import json, os
    log = a.atom_token_log
    n0 = sum(1 for _ in open(log)) if os.path.exists(log) else 0
    requests.post(f"http://127.0.0.1:{a.atom_port}/v1/completions", json=body, timeout=3600).raise_for_status()
    lines = open(log).read().splitlines()[n0:] if os.path.exists(log) else []
    return json.loads(lines[-1]) if lines else None


for s in [int(x) for x in a.seeds.split(",")]:
    rng = random.Random(s)
    ids = [rng.randint(1000, 100000) for _ in range(a.ctx)]
    for name, fn in (("sglang", sgl), ("atom", atom)):
        try:
            out = fn(ids)
        except Exception as e:
            print(f"seed={s} {name}: ERROR {e!r}"[:200]); continue
        if not out:
            print(f"seed={s} {name}: no token ids in response"); continue
        top = collections.Counter(out).most_common(3)
        print(f"seed={s} {name:6s} n={len(out)} unique={len(set(out))} ({len(set(out)) / len(out):.2f}) "
              f"top={top} head={out[:12]}")
