"""EP1 decode MoE down (stage2, K=512) tile sweep on the server path (h13-17 1029).

Each candidate = one CSV (EP1 v3 row for that token count with stage2 fields replaced), loaded via
AITER_CONFIG_GROUPED_FMOE exactly like the server; op test --scenario kernel --no-bias times gemm1/gemm2
alone. Candidates are dealt round-robin over --gpus, one serial worker per GPU; the v3 baseline runs at the start
and end of every worker. Results: <out>/results.jsonl (one line per run).

  python3 down_decode_sweep_1029.py --out DIR --tokens 2 16 64 --gpus 0 1 2 3 4 5 6 7            # pass 1
  python3 down_decode_sweep_1029.py --out DIR --tokens 2 --gpus 7 --limit 1                      # smoke
  python3 down_decode_sweep_1029.py --out DIR --report
Pass 1 grid: tile_m2 {16,32,64} x tile_n2 {128,256,512} x tile_k2 {128,256,512} x nb2 {2,3,4} (nb2 <= 512/tile_k2,
min 2 only when 2 k-tiles exist), n_warp2 / persistent_workers2 from v3. tile_m2 < 64 is run twice: stage1 tile_m 64
(expert rows still aligned to 64) and stage1 tile_m = tile_m2 (real alignment; gemm1 changes too -> compare g1+g2).
--grid2 "TAG:m2,n2,k2,nb2" ... runs pass 2: n_warp2 {1,2,4} x pw2 {0,96,192,384} on the listed stage2 tiles.
--stage1: stage1 tile_n {128,256,512} x tile_k {128,256,512} x nb {2,3,4} x n_warp {1,2,4} on the v4 rows (baseline v4).
"""

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import threading

V3 = "/workspace/proxy_work/ep1_grouped_fmoe_gfx1250_v3.csv"
V4 = "/workspace/proxy_work/ep1_grouped_fmoe_gfx1250_v4.csv"
AITER = "/sgl-workspace/aiter"
CMD = ("python3 op_tests/flydsl_tests/test_flydsl_grouped_gemm.py --no-check-aot-cache --scenario kernel "
       "--data-init uniform --scale-init auto --data-format a8w4 --experts 512 --topk 16 --model-dim 16384 "
       "--inter-dim 512 --act silu --no-bias --tokens {T}")


def v3_rows(path=V3):
    with open(path, newline="") as f:
        r = list(csv.DictReader(f))
    return r[0].keys(), {int(x["token"]): x for x in r}


def pass1(T, base):
    out = []
    for m2, n2, k2, nb in itertools.product((16, 32, 64), (128, 256, 512), (128, 256, 512), (2, 3, 4)):
        kt = 512 // k2
        if nb > max(kt, 2) or (kt == 1 and nb != 2):
            continue
        for m1 in sorted({64, m2}):
            c = dict(base, tile_m=m1, tile_m2=m2, tile_n2=n2, tile_k2=k2, num_buffer_stage2=nb)
            out.append((f"t{T}_s1m{m1}_m{m2}_n{n2}_k{k2}_b{nb}", c))
    return out


def pass2(T, base, grid2):
    out = []
    for g in grid2:
        tag, vals = g.split(":")
        m2, n2, k2, nb = map(int, vals.split(","))
        for nw, pw in itertools.product((1, 2, 4), (0, 96, 192, 384)):
            c = dict(base, tile_m=max(64, m2) if m2 == 64 else m2, tile_m2=m2, tile_n2=n2, tile_k2=k2,
                     num_buffer_stage2=nb, n_warp2=nw, persistent_workers2=pw)
            out.append((f"t{T}_{tag}_w{nw}_pw{pw}", c))
    return out


def pass_s1(T, base):
    """Stage1 (gate-up, K=16384) tiles at tile_m 16 on top of a v4 row; fused quant needs (tile_n / n_warp) % 64 == 0."""
    out = []
    for n, k, nb, nw in itertools.product((128, 256, 512), (128, 256, 512), (2, 3, 4), (1, 2, 4)):
        if (n // nw) % 64:
            continue
        c = dict(base, tile_n=n, tile_k=k, num_buffers=nb, n_warp=nw)
        out.append((f"t{T}_s1_n{n}_k{k}_b{nb}_w{nw}", c))
    return out


def run_one(out, gpu, tag, T, cfg, keys, lock):
    path = f"{out}/csv/{tag}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerow(cfg)
    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu), AITER_CONFIG_GROUPED_FMOE=path)
    for k in list(env):
        if k.startswith("AITER_TDM_TILE") or k.startswith("AITER_TDM_NUM_BUFFERS") or k.startswith("AITER_TDM_N_WARP"):
            env.pop(k)
    log = f"{out}/logs/{tag}.gpu{gpu}.log"
    try:
        p = subprocess.run(CMD.format(T=T), shell=True, cwd=AITER, env=env, capture_output=True, text=True, timeout=900)
        txt = p.stdout + p.stderr
        rc = p.returncode
    except subprocess.TimeoutExpired:
        txt, rc = "TIMEOUT", -9
    open(log, "w").write(txt)
    g = dict(re.findall(r"\] (gemm[12]): us = ([0-9.]+)", txt))
    miss = "no grouped CSV config matched" in txt
    rec = {"tag": tag, "T": T, "gpu": gpu, "rc": rc, "g1": float(g["gemm1"]) if "gemm1" in g else None,
           "g2": float(g["gemm2"]) if "gemm2" in g else None, "csv_miss": miss,
           "err": None if g else (re.findall(r"(\w*Error[^\n]{0,160})", txt) or [txt[-160:]])[-1]}
    with lock:
        with open(f"{out}/results.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec)[:220], flush=True)


def report(out):
    rs = [json.loads(l) for l in open(f"{out}/results.jsonl")]
    by = {}
    for r in rs:
        by.setdefault((r["T"], r["tag"].split(".")[0]), []).append(r)
    for T in sorted({r["T"] for r in rs}):
        base = [r for r in rs if r["T"] == T and r["tag"].startswith(f"t{T}_base") and r["g2"]]
        b1 = sorted(r["g1"] for r in base)
        b2 = sorted(r["g2"] for r in base)
        print(f"T={T} baseline n={len(base)} g1 med {b1[len(b1) // 2] if b1 else 'NA'} g2 med {b2[len(b2) // 2] if b2 else 'NA'}")
        ok = [r for r in rs if r["T"] == T and r["g2"] and not r["tag"].startswith(f"t{T}_base")]
        for r in sorted(ok, key=lambda r: r["g1"] + r["g2"])[:10]:
            print(f"  g1+g2 {r['g1'] + r['g2']:8.1f}  g1 {r['g1']:7.1f}  g2 {r['g2']:7.1f}  {r['tag']}")
        bad = [r for r in rs if r["T"] == T and not r["g2"]]
        print(f"  failed {len(bad)}" + (f" e.g. {bad[0]['tag']}: {str(bad[0]['err'])[:120]}" if bad else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, nargs="+", default=[2, 16, 64])
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--grid2", nargs="*", default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--stage1", action="store_true", help="sweep stage1 tiles on the v4 rows (baseline = v4)")
    a = ap.parse_args()
    if a.report:
        return report(a.out)
    os.makedirs(f"{a.out}/csv", exist_ok=True)
    os.makedirs(f"{a.out}/logs", exist_ok=True)
    keys, rows = v3_rows(V4 if a.stage1 else V3)
    done = set()
    if os.path.exists(f"{a.out}/results.jsonl"):
        done = {json.loads(l)["tag"] for l in open(f"{a.out}/results.jsonl")}
    jobs = []
    for T in a.tokens:
        base = rows[T]
        cands = pass_s1(T, base) if a.stage1 else pass2(T, base, a.grid2) if a.grid2 else pass1(T, base)
        jobs += [(t, T, c) for t, c in cands if t not in done]
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{len(jobs)} candidates on GPUs {a.gpus}", flush=True)
    lock = threading.Lock()

    def worker(i, gpu):
        mine = jobs[i:: len(a.gpus)]
        Ts = sorted({T for _, T, _ in mine})
        for T in Ts:
            run_one(a.out, gpu, f"t{T}_base.g{gpu}.start", T, rows[T], keys, lock)
        for tag, T, c in mine:
            run_one(a.out, gpu, tag, T, c, keys, lock)
        for T in Ts:
            run_one(a.out, gpu, f"t{T}_base.g{gpu}.end", T, rows[T], keys, lock)

    th = [threading.Thread(target=worker, args=(i, g)) for i, g in enumerate(a.gpus)]
    for t in th:
        t.start()
    for t in th:
        t.join()


if __name__ == "__main__":
    sys.exit(main())
