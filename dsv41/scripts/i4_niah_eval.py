#!/usr/bin/env python3
"""I4 long-context accuracy probe (RULER-like multi-key needle in a haystack) against an OpenAI-compatible server.
  python3 -I i4_niah_eval.py --port 8888 [--lengths 32000,64000,120000] [--n 20] [--keys 64] [--conc 4]
Each prompt hides KEYS "The magic number for <word> is <7 digits>." lines at random depths of seeded filler text,
then asks for one key's number. Score = exact match of that number in the answer. Deterministic per seed, so the
same questions go to both sides of an A/B. Prints one line per length plus a total; per-item rows to --out (jsonl).
"""
import argparse
import concurrent.futures as cf
import json
import random
import re
import urllib.request

WORDS = ("amber basil cedar delta ember fjord grove harbor island jasper kettle lantern meadow nectar orchard pebble "
         "quarry raven saffron timber umber valley willow yarrow zephyr anchor bramble canyon dune eagle falcon garnet "
         "heron iris juniper kelp lotus maple nimbus onyx prairie quill reef sierra thistle tundra violet walnut").split()
FILLER = ("The committee reviewed the quarterly logistics report and noted steady progress on the regional depots. "
          "Several teams adjusted their schedules after the weather delayed the inbound shipments by two days. "
          "A long discussion followed about inventory accounting, supplier contracts and the maintenance backlog. "
          "Nobody disputed the figures, but the minutes record a request for a clearer summary next time. ")


def build(seed, target_chars, keys):
    rng = random.Random(seed)
    names = rng.sample([f"{a}-{b}" for a in WORDS for b in WORDS if a != b], keys)
    values = [str(rng.randrange(1_000_000, 10_000_000)) for _ in names]
    n_fill = max(1, target_chars // len(FILLER))
    chunks = [FILLER] * n_fill
    for name, val in zip(names, values):
        chunks.insert(rng.randrange(len(chunks) + 1), f"The magic number for {name} is {val}. ")
    q = rng.randrange(keys)
    prompt = ("Below is a long document. Some lines state magic numbers for named items.\n\n" + "".join(chunks)
              + f"\n\nWhat is the magic number for {names[q]}? Answer with only the 7-digit number.")
    return prompt, values[q]


def ask(port, prompt, max_tokens, model):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": max_tokens, "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    return (m.get("content") or ""), d.get("usage", {}).get("prompt_tokens", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--lengths", default="32000,64000,120000", help="target prompt tokens (approx, 4 chars/token)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--keys", type=int, default=64)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--model", default="default")
    ap.add_argument("--out", default="/tmp/i4_niah.jsonl")
    a = ap.parse_args()
    total = hit = 0
    with open(a.out, "w") as f:
        for L in (int(x) for x in a.lengths.split(",")):
            items = [build(1000 * L + i, L * 4, a.keys) for i in range(a.n)]
            with cf.ThreadPoolExecutor(a.conc) as ex:
                res = list(ex.map(lambda it: ask(a.port, it[0], a.max_tokens, a.model), items))
            ok = 0
            ptoks = []
            for (prompt, want), (ans, pt) in zip(items, res):
                got = re.findall(r"\d{7}", ans)
                good = bool(got) and got[-1] == want
                ok += good
                ptoks.append(pt)
                f.write(json.dumps({"len": L, "want": want, "answer": ans[:200], "ok": good, "prompt_tokens": pt}) + "\n")
            total += len(items)
            hit += ok
            print(f"len~{L}: {ok}/{len(items)} = {ok / len(items):.3f} (prompt tokens {min(ptoks)}-{max(ptoks)})", flush=True)
    print(f"total: {hit}/{total} = {hit / total:.3f}")


if __name__ == "__main__":
    main()
