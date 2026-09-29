#!/usr/bin/env python3
"""Dump greedy generations for a fixed prompt set from a running server, or compare two dumps.

Used for A/B parity of changes that must not alter numerics (e.g. metadata moved into a CUDA graph).
  dump:    greedy_parity_dump.py dump --port 8888 --out A.json [--n 16 --max-tokens 256]
  compare: greedy_parity_dump.py cmp A.json B.json
Prompts: the first N GSM8K test questions if available (sglang.utils download cache), else fixed synthetic text.
Output: per prompt, output_ids; cmp prints how many prompts are identical and the first diverging token index.
"""
import argparse, json, sys, requests

ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("files", nargs="*")
ap.add_argument("--port", type=int, default=8888); ap.add_argument("--out", default="")
ap.add_argument("--n", type=int, default=16); ap.add_argument("--max-tokens", type=int, default=256)
a = ap.parse_args()

if a.mode == "cmp":
    x, y = (json.load(open(f)) for f in a.files)
    same = 0
    for i, (p, q) in enumerate(zip(x, y)):
        if p == q:
            same += 1
            continue
        d = next((k for k, (s, t) in enumerate(zip(p, q)) if s != t), min(len(p), len(q)))
        print(f"prompt {i}: diverges at token {d} (len {len(p)} vs {len(q)})")
    print(f"identical {same}/{len(x)}")
    sys.exit(0)

prompts = [f"Question {i}: A farmer has {3 + i} fields with {12 + 5 * i} rows of {7 + i} plants each. "
           f"He sells {i + 2} rows per field. Explain step by step how many plants remain." for i in range(a.n)]
outs = []
for p in prompts:
    r = requests.post(f"http://127.0.0.1:{a.port}/generate", json={
        "text": p, "sampling_params": {"temperature": 0, "max_new_tokens": a.max_tokens, "ignore_eos": True}},
        timeout=600).json()
    outs.append(r["output_ids"] if "output_ids" in r else r["text"])
json.dump(outs, open(a.out, "w"))
print(f"dumped {len(outs)} to {a.out}")
