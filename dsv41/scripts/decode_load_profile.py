#!/usr/bin/env python3
"""Fixed-shape decode load against an SGLang server, optionally profiling steady-state decode.

    python3 -I decode_load_profile.py --port 8888 --conc 8 --isl 65536 --osl 1024 [--profile-dir D --profile-steps 40]

Sends `conc` streaming /generate requests (random input ids, ignore_eos), measures TTFT and TPOT per request
(TPOT = (t_last - t_first) / (tokens - 1), the aiperf definition). With --profile-dir, POSTs /start_profile once
every request has its first token (all prefills done -> pure decode), for --profile-steps forward steps.
Mirrors B200_REQUEST_1006.md Deliverable 1 (D64) / 2 so both sides profile the same shape.
"""
import argparse
import json
import random
import statistics as st
import threading
import time

import requests


def one(url, ids, osl, res, i, first_evt):
    body = {"input_ids": ids, "stream": True,
            "sampling_params": {"max_new_tokens": osl, "ignore_eos": True, "temperature": 0.0}}
    t0 = time.perf_counter(); t_first = t_last = None; ntok = 0
    with requests.post(url, json=body, stream=True, timeout=3600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            n = d.get("meta_info", {}).get("completion_tokens", ntok)
            if n > ntok:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now; first_evt[i].set()
                t_last, ntok = now, n
    res[i] = dict(ttft=(t_first - t0) * 1e3, tpot=(t_last - t_first) * 1e3 / max(ntok - 1, 1), ntok=ntok)


def run(a, label, profile):
    base = f"http://127.0.0.1:{a.port}"
    rng = random.Random(a.seed)
    res = [None] * a.conc; first = [threading.Event() for _ in range(a.conc)]
    ths = [threading.Thread(target=one, args=(f"{base}/generate", [rng.randrange(1000, 100000) for _ in range(a.isl)],
                                              a.osl, res, i, first)) for i in range(a.conc)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    if profile:
        for e in first:
            e.wait()
        time.sleep(0.5)  # let the batch settle into steady decode
        r = requests.post(f"{base}/start_profile", json={"output_dir": a.profile_dir, "num_steps": a.profile_steps,
                          "activities": ["GPU", "CPU"], "record_shapes": True, "with_stack": False}, timeout=600)
        print(f"[{label}] start_profile at {time.perf_counter() - t0:.1f}s -> {r.status_code} {r.text[:120]}")
    for t in ths:
        t.join()
    ok = [x for x in res if x]
    tt, tp = [x["ttft"] for x in ok], [x["tpot"] for x in ok]
    print(f"[{label}] conc {a.conc} ISL {a.isl} OSL {a.osl}: ok {len(ok)}/{a.conc}, tokens {sum(x['ntok'] for x in ok)}, "
          f"TTFT ms mean {st.mean(tt):.0f} max {max(tt):.0f}, TPOT ms mean {st.mean(tp):.3f} min {min(tp):.3f} "
          f"max {max(tp):.3f}, wall {time.perf_counter() - t0:.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8888)
    p.add_argument("--conc", type=int, default=8)
    p.add_argument("--isl", type=int, default=65536)
    p.add_argument("--osl", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--profile-dir")
    p.add_argument("--profile-steps", type=int, default=40)
    a = p.parse_args()
    run(a, "warmup", False)              # JIT / graph warmup at this shape
    a.seed += 1; run(a, "clean", False)  # numbers without profiler overhead
    if a.profile_dir:
        a.seed += 1; run(a, "profile", True)


if __name__ == "__main__":
    main()
