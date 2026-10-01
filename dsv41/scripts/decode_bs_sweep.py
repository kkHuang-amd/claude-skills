#!/usr/bin/env python3
"""Decode-only step-cost sweep vs batch size, SGLang or ATOM (same simulated DSpark acceptance on both servers).

Per (ctx, bs): warm the prefix cache with each prompt (max_tokens=1, sequential), then fire bs concurrent streaming
requests of the same prompts (ignore_eos, OSL tokens) so all decode together without prefill interference.
decode tok/s per request = (n_last - n_first) / (t_last - t_first); also reports the decode-start spread (should be
small vs decode time, else the point is contaminated by residual prefill) and a P90-interactivity proxy.
  python3 decode_bs_sweep.py --engine sglang --port 8888 --ctx 8192,65536 --bs 1,2,4,6,8 --osl 2048 --tag sgl
  python3 decode_bs_sweep.py --engine atom   --port 8000 ...
Temperature omitted by default (server default, like AgentX). Rows appended to --out.
"""
import argparse, json, random, statistics, threading, time
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--engine", choices=["sglang", "atom"], required=True)
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--ctx", default="8192,65536")
ap.add_argument("--bs", default="1,2,4,6,8")
ap.add_argument("--osl", type=int, default=2048)
ap.add_argument("--repeat", type=int, default=2)
ap.add_argument("--temperature", type=float, default=None)
ap.add_argument("--model", default="/shared_nfs/deepseek-ai/DeepSeek-V4.1-Flash")
ap.add_argument("--tag", default="x")
ap.add_argument("--out", default="/shared_nfs/kk/dsv41/atomport/decode_bs_sweep.tsv")
ap.add_argument("--profile-dir", default="", help="profile repeat 0 of each point once all streams decode")
ap.add_argument("--profile-delay", type=float, default=2.0)
ap.add_argument("--profile-steps", type=int, default=40, help="SGLang num_steps")
ap.add_argument("--profile-secs", type=float, default=0.5, help="ATOM start->stop window")
a = ap.parse_args()
URL = f"http://127.0.0.1:{a.port}"
started = []


def request(ids, max_tokens, stream):
    if a.engine == "sglang":
        sp = {"max_new_tokens": max_tokens, "ignore_eos": True}
        if a.temperature is not None:
            sp["temperature"] = a.temperature
        return URL + "/generate", {"input_ids": ids, "stream": stream, "sampling_params": sp}
    body = {"model": a.model, "prompt_token_ids": ids, "max_tokens": max_tokens, "ignore_eos": True,
            "stream": stream}
    if stream:
        body["stream_options"] = {"include_usage": True}
    if a.temperature is not None:
        body["temperature"] = a.temperature
    return URL + "/v1/completions", body


def one(ids, res):
    url, body = request(ids, a.osl, True)
    stamps, ntok, n = [], [], 0
    with requests.post(url, json=body, stream=True, timeout=7200) as r:
        for line in r.iter_lines():
            if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            if not stamps:
                started.append(1)
            if a.engine == "sglang":
                stamps.append(time.perf_counter()); ntok.append(d["meta_info"]["completion_tokens"])
            else:
                if d.get("usage"):
                    n = d["usage"].get("completion_tokens", n)
                if d.get("choices") and d["choices"][0].get("text"):
                    stamps.append(time.perf_counter()); ntok.append(len(ntok) + 1)
    if a.engine == "atom" and n and ntok:
        # ATOM chunks may carry several tokens; rescale chunk index to the reported token count.
        ntok = [round(i * n / len(ntok)) for i in ntok]
    res.append((stamps, ntok))


def profile_when_decoding(bs, sub):
    while len(started) < bs:
        time.sleep(0.05)
    time.sleep(a.profile_delay)
    if a.engine == "sglang":
        body = {"output_dir": f"{a.profile_dir}/{sub}", "num_steps": a.profile_steps, "activities": ["GPU"],
                "with_stack": False, "record_shapes": False}
        r = requests.post(URL + "/start_profile", json=body, timeout=600)
    else:
        requests.post(URL + "/start_profile", timeout=600).raise_for_status()
        time.sleep(a.profile_secs)
        r = requests.post(URL + "/stop_profile", timeout=1800)
    print(f"profile {sub}: {r.status_code} {r.text[:160]}", flush=True)


for ctx in [int(x) for x in a.ctx.split(",")]:
    for bs in [int(x) for x in a.bs.split(",")]:
        for rep in range(a.repeat):
            prompts = []
            for i in range(bs):
                rng = random.Random((ctx, bs, rep, i).__hash__())
                prompts.append([rng.randint(1000, 100000) for _ in range(ctx)])
            for p in prompts:
                url, body = request(p, 1, False)
                requests.post(url, json=body, timeout=7200).raise_for_status()
            res = []
            started.clear()
            ths = [threading.Thread(target=one, args=(p, res)) for p in prompts]
            if a.profile_dir and rep == 0:
                ths.append(threading.Thread(target=profile_when_decoding, args=(bs, f"{a.tag}_ctx{ctx}_bs{bs}")))
            for t in ths: t.start()
            for t in ths: t.join()
            tps, itl = [], []
            for st, nt in res:
                tps.append((nt[-1] - nt[0]) / (st[-1] - st[0]))
                for i in range(1, len(st)):
                    dn = nt[i] - nt[i - 1]
                    if dn > 0:
                        itl += [(st[i] - st[i - 1]) / dn] * dn
            itl.sort()
            p90 = itl[int(0.9 * (len(itl) - 1))]
            starts = [st[0] for st, _ in res]
            dec = statistics.mean(st[-1] - st[0] for st, _ in res)
            line = (f"{a.tag}\t{a.engine}\tctx={ctx}\tbs={bs}\trep={rep}\ttps_per_req={statistics.mean(tps):.1f}"
                    f"\ttps_min={min(tps):.1f}\tagg_tps={sum(tps):.0f}\tp90_intvty={1 / p90:.1f}"
                    f"\tstart_spread_s={max(starts) - min(starts):.2f}\tdecode_s={dec:.2f}")
            print(line, flush=True)
            with open(a.out, "a") as f:
                f.write(time.strftime("%F %T") + "\t" + line + "\n")
