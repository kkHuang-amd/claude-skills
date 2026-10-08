#!/usr/bin/env python3
"""Proxy decode benchmark for the ATOM-port A/B loop (faster than a 3600 s AgentX run).

Sends CONC concurrent streaming /generate requests of CTX random input ids + OSL output tokens (ignore_eos),
REPEAT times, and prints per-request decode tok/s (= (OSL-1) / (t_last - t_first)) and P90 ITL-derived
interactivity. Random ids avoid tokenizer cost; DSpark AL is fixed by the server's SGLANG_SIMULATE_ACC_LEN.
Optional PROFILE_DIR: after the first ~2 s of decode in repeat 0, call /start_profile for PROFILE_STEPS steps.

Env/args: --port 8888 --ctx 2048 --osl 1024 --conc 1 --repeat 3 --tag x [--profile-dir D --profile-steps 40]
Output: one summary line per repeat to stdout; append a row to --out (default /shared_nfs/kk/results/DeepSeek-V4.1-Flash/atomport/proxy.tsv).
"""
import argparse, json, random, statistics, threading, time, os
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8888)
ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--osl", type=int, default=1024)
ap.add_argument("--conc", type=int, default=1)
ap.add_argument("--repeat", type=int, default=3)
ap.add_argument("--tag", default="x")
ap.add_argument("--temperature", type=float, default=0.0)  # AgentX sends none -> server default 1.0
ap.add_argument("--profile-dir", default="")
ap.add_argument("--profile-steps", type=int, default=40)
ap.add_argument("--out", default="/shared_nfs/kk/results/DeepSeek-V4.1-Flash/atomport/proxy.tsv")
a = ap.parse_args()
URL = f"http://127.0.0.1:{a.port}"


def one(seed, res):
    rng = random.Random(seed)
    ids = [rng.randint(1000, 100000) for _ in range(a.ctx)]
    body = {"input_ids": ids, "stream": True,
            "sampling_params": {"max_new_tokens": a.osl, "ignore_eos": True, "temperature": a.temperature}}
    t0 = time.perf_counter(); stamps = []; ntok = []
    with requests.post(URL + "/generate", json=body, stream=True, timeout=3600) as r:
        for line in r.iter_lines():
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            stamps.append(time.perf_counter()); ntok.append(d["meta_info"]["completion_tokens"])
    res.append((t0, stamps, ntok))


def profile_later():
    time.sleep(2.0)
    body = {"output_dir": a.profile_dir, "num_steps": a.profile_steps, "activities": ["GPU"],
            "with_stack": False, "record_shapes": False}
    r = requests.post(URL + "/start_profile", json=body, timeout=600)
    print("start_profile:", r.status_code, r.text[:120], flush=True)


os.makedirs(os.path.dirname(a.out), exist_ok=True)
for rep in range(a.repeat):
    res = []
    ths = [threading.Thread(target=one, args=(rep * 1000 + i, res)) for i in range(a.conc)]
    if rep == 0 and a.profile_dir:
        ths.append(threading.Thread(target=profile_later))
    for t in ths: t.start()
    for t in ths: t.join()
    tps, itl_all, ttft = [], [], []
    for t0, st, nt in res:
        ttft.append(st[0] - t0)
        tps.append((nt[-1] - nt[0]) / (st[-1] - st[0]))
        for i in range(1, len(st)):
            dn = nt[i] - nt[i - 1]
            if dn > 0:
                itl_all += [(st[i] - st[i - 1]) / dn] * dn
    itl_all.sort()
    p90 = itl_all[int(0.9 * (len(itl_all) - 1))]
    line = (f"{a.tag}\tctx={a.ctx}\tconc={a.conc}\ttemp={a.temperature}\trep={rep}\tdecode_tps_per_req={statistics.mean(tps):.1f}"
            f"\tp90_interactivity={1/p90:.1f}\tttft_s={statistics.mean(ttft):.2f}")
    print(line, flush=True)
    with open(a.out, "a") as f:
        f.write(time.strftime("%F %T") + "\t" + line + "\n")
