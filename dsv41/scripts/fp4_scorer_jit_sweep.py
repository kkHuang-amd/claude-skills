#!/usr/bin/env python3
"""P0.4 (FP4_INDEX_PLANE_PORT.md): JIT compiles + first-call cost across shapes, SGLang prefill scorer (A) vs row-group (B).

One process = one MODE (A | B). Sweeps key widths (rows fixed) then row counts (width fixed), each case in order;
per case prints first-call ms (sync), steady ms, and new entries in the FlyDSL / Triton caches since the previous case.
A also times its prefill workspace build (Triton schedule kernels) separately.
Run it in a fresh process with empty FLYDSL_RUNTIME_CACHE_DIR / TRITON_CACHE_DIR for "cold node", then again with the
same dirs for "server restart, disk cache warm" (fp4_scorer_jit_sweep.sh does both).
  MODE=A|B PYTHONPATH=/sgl-workspace/aiter-6145 FLYDSL_RUNTIME_CACHE_DIR=... TRITON_CACHE_DIR=... python3 fp4_scorer_jit_sweep.py
"""
import importlib.util
import os
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bench", os.path.join(HERE, "fp4_scorer_vs_sglang_bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
S = bench.S
MODE = os.environ["MODE"]


def n_entries(d):
    n = 0
    for _, dirs, files in os.walk(d):
        n += len(files)
    return n


caches = [os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", ""), os.environ.get("TRITON_CACHE_DIR", "")]


def counts():
    return tuple(n_entries(c) if c and os.path.isdir(c) else 0 for c in caches)


def wall(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1e3


def run(name, prefix, new, ratio=1):
    c = bench.make_case([(prefix, new)], ratio)
    T = c["T"]
    before = counts()
    ws_first = float("nan")
    if MODE == "A":
        holder = {}
        ws_first = wall(lambda: holder.setdefault("ws", S.prepare_fp4_prefill_workspace(
            c["row_table"], c["row_ends"], page_table_bucket=bench.BUCKET)))
        ws = holder["ws"]

        def call():
            S.aiter_fp4_paged_mqa_logits(
                q_fp4=c["qp"], q_scale=c["qs"].view(T, 1, 4, 16, 4), k_payload=c["kv_cache"],
                k_scale=c["kv_scale"], weights=c["w"], page_table=c["row_table"],
                c4_seq_lens=c["row_ends"], weight_scale=1.0, is_decode=False,
                page_table_bucket=bench.BUCKET, prefill_workspace=ws,
            )
    else:
        width = -(-int(c["row_ends"].max()) // (bench.PAGE * bench.BUCKET)) * bench.PAGE * bench.BUCKET
        out = torch.empty((T, width), dtype=torch.float32, device="cuda")

        def call():
            plan = bench.make_fp4_mqa_plan(
                num_seqs=1, max_qlen=T, num_rows=T, heads=bench.H, page_size=bench.PAGE, max_seq_len=width
            )
            bench.flydsl_pa_mqa_logits_fp4_rowgroup(
                plan, c["qp"], c["qs"].reshape(T, -1), c["kv_cache"], c["kv_scale"], c["req_table"],
                c["w"], c["qsl"], c["row_ends"], weight_scale=1.0, out=out,
            )
    first = wall(call)
    steady = min(wall(call) for _ in range(3))
    after = counts()
    print(f"{MODE} {name:<22} rows={T:>6} keys<={int(c['row_ends'].max()):>6} | first {first:9.1f} ms "
          f"steady {steady:7.2f} ms | ws_first {ws_first:8.1f} ms | new flydsl {after[0] - before[0]:>3} "
          f"triton {after[1] - before[1]:>3}", flush=True)
    del c
    torch.cuda.empty_cache()


if __name__ == "__main__":
    rows = 2048
    for keys in (4096, 8192, 16384, 20480, 32768, 49152, 65536, 98304, 131072, 196608, 262144):
        run(f"width {keys // 1024}k", prefix=keys - rows, new=rows)
    for r in (512, 1024, 4096, 8192, 16384):
        run(f"rows {r}", prefix=32768 - r if r < 32768 else 0, new=r)
