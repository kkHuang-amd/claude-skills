#!/usr/bin/env python3
"""P0.3 (FP4_INDEX_PLANE_PORT.md): SGLang's current FP4 index scorer vs aiter #6145 row-group, V4.1 shapes.

A = SGLang path, `aiter_fp4_paged_mqa_logits` (sglang/kernels/ops/attention/dsv4/fp4_indexer_hip.py):
    prefill: per-token rows, per-row page table, `flydsl_pa_mqa_logits_fp4_prefill` + prepared workspace
             (page_table_bucket=LOW_RATIO_PAGE_TABLE_BUCKET=64);
    decode : `flydsl_pa_mqa_logits_fp4` + prepared decode workspace.
    "A sched" = the workspace build (prefill: once per step per ratio, outside graph; decode: in graph).
B = `flydsl_pa_mqa_logits_fp4_rowgroup`, request-level ragged rows off the REQUEST page table, plan prebuilt.
Same page-64 preshuffle plane, same q/weights. Check: max |A-B| over each row's valid columns.

Run (aiter >= 5b2f7d1d1 first on the path; sglang importable):
  PYTHONPATH=/sgl-workspace/aiter-6145 HIP_VISIBLE_DEVICES=0 python3 fp4_scorer_vs_sglang_bench.py
Env: AITER_DIR (op_tests helpers, default /sgl-workspace/aiter-6145), ITERS (default 20), ONLY=prefill|decode.
Output: one line per case on stdout.
"""
import importlib.util
import os

import torch

AITER_DIR = os.environ.get("AITER_DIR", "/sgl-workspace/aiter-6145")
spec = importlib.util.spec_from_file_location(
    "t", os.path.join(AITER_DIR, "op_tests/test_flydsl_pa_mqa_logits_fp4.py")
)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_rowgroup import (
    flydsl_pa_mqa_logits_fp4_rowgroup,
    make_fp4_mqa_plan,
)

from sglang.kernels.ops.attention.dsv4 import fp4_indexer_hip as S

dev, PAGE, D, H = "cuda", 64, 128, 32  # V4.1: index_n_heads=32, index_head_dim=128
BUCKET = 64  # LOW_RATIO_PAGE_TABLE_BUCKET
ITERS = int(os.environ.get("ITERS", "20"))
COLD = os.environ.get("COLD", "0") == "1"  # aiter #6145 method: read 2 GiB before each timed call, kernel only
_FLUSH = None


def timeit_cold(fn, iters=ITERS):
    """Per-call events after a 2 GiB read (L2/MALL cold). The flush (~0.4 ms) covers the host launch time."""
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(2**31 // 4, dtype=torch.float32, device=dev)
    fn()
    ts = []
    for _ in range(iters):
        _FLUSH.sum()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    return sorted(ts)[len(ts) // 2]


def timeit(fn, iters=ITERS, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def timeit_graph(fn, iters=ITERS):
    """GPU time: capture `iters` calls in one CUDA graph and time its replay (no host overhead)."""
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(3):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / (3 * iters)


def gpu_time(fn):
    if COLD:
        return timeit_cold(fn)
    try:
        return timeit_graph(fn)
    except Exception as ex:  # capture not supported for this path
        print(f"  [graph timing failed: {type(ex).__name__}: {str(ex)[:100]}]", flush=True)
        torch.cuda.synchronize()
        return float("nan")


def make_case(reqs, ratio):
    """reqs: list of (prefix_tokens, new_tokens). Rows = new tokens; row at absolute position p
    sees (p + 1) // ratio compressed keys (SGLang `compress_lens = (pos + 1) // ratio`)."""
    t.setup_seed(t.SEED)
    keys = [(p + n) // ratio for p, n in reqs]
    pages = [max(1, -(-k // PAGE)) for k in keys]
    width = max(pages)
    num_pages = sum(pages) + 1
    perm = torch.randperm(num_pages - 1, device=dev).to(torch.int32)
    req_table = torch.zeros(len(reqs), width, dtype=torch.int32, device=dev)
    off = 0
    for b, n in enumerate(pages):
        req_table[b, :n] = perm[off : off + n]
        off += n
    # fill the plane with random FP4 rows (dense order = page id order)
    kv_bf16 = torch.randn(1, num_pages * PAGE, D, dtype=torch.bfloat16, device=dev)
    ident = torch.arange(num_pages, dtype=torch.int32, device=dev).view(1, -1)
    kv_cache, kv_scale, _, _ = t.create_paged_preshuffle_kv_fp4(kv_bf16, PAGE, num_pages, ident)

    rows_req, row_ends = [], []
    for b, (p, n) in enumerate(reqs):
        rows_req += [b] * n
        row_ends += [(q + 1) // ratio for q in range(p, p + n)]
    T = len(rows_req)
    rows_req = torch.tensor(rows_req, dtype=torch.long, device=dev)
    row_ends = torch.tensor(row_ends, dtype=torch.int32, device=dev)
    qsl = torch.tensor([0] + list(torch.tensor([n for _, n in reqs]).cumsum(0)), dtype=torch.int32, device=dev)

    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev)
    qp, qe = t.fp4_quant_e2m1_with_e8m0(q.reshape(-1, D))
    qp = qp.view(T, H, D // 2)
    qs = t.pack_q_scales(qe.view(T, 1, H, D // 32))  # [T,1,1,4,16,4] == SGLang (T,1,4,16,4)
    w = (torch.randn(T, H, device=dev) * 0.1).to(torch.bfloat16)
    return dict(
        kv_cache=kv_cache, kv_scale=kv_scale, req_table=req_table, row_table=req_table[rows_req],
        row_ends=row_ends, qsl=qsl, qp=qp, qs=qs, w=w, T=T, nreq=len(reqs),
        max_qlen=max(n for _, n in reqs),
    )


def run(name, reqs, ratio, is_decode):
    c = make_case(reqs, ratio)
    T = c["T"]
    # ---- A: SGLang ----
    if is_decode:
        ws = S.prepare_fp4_decode_workspace(c["row_table"], c["row_ends"], page_table_bucket=BUCKET)
        sched = lambda: S.prepare_fp4_decode_workspace(c["row_table"], c["row_ends"], page_table_bucket=BUCKET)
        kw = dict(decode_workspace=ws)
        width = ws.max_seq_len
    else:
        ws = S.prepare_fp4_prefill_workspace(c["row_table"], c["row_ends"], page_table_bucket=BUCKET)
        sched = lambda: S.prepare_fp4_prefill_workspace(
            c["row_table"], c["row_ends"], workspace=ws, page_table_bucket=BUCKET
        )
        kw = dict(prefill_workspace=ws)
        width = ws.max_seq_len

    def A():
        return S.aiter_fp4_paged_mqa_logits(
            q_fp4=c["qp"], q_scale=c["qs"].view(T, 1, 4, 16, 4), k_payload=c["kv_cache"],
            k_scale=c["kv_scale"], weights=c["w"], page_table=c["row_table"],
            c4_seq_lens=c["row_ends"], weight_scale=1.0, is_decode=is_decode,
            page_table_bucket=BUCKET, **kw,
        )

    la = A().clone()
    us_a_eager = float("nan") if COLD else timeit(A)
    us_a = gpu_time(A)
    us_sched = float("nan") if COLD else timeit(sched)

    # ---- B: row-group, request page table ----
    plan = make_fp4_mqa_plan(
        num_seqs=c["nreq"], max_qlen=c["max_qlen"], num_rows=T, heads=H, page_size=PAGE,
        max_seq_len=width,
    )
    out = torch.full((T, width), float("-inf"), device=dev)

    def B():
        flydsl_pa_mqa_logits_fp4_rowgroup(
            plan, c["qp"], c["qs"].reshape(T, -1), c["kv_cache"], c["kv_scale"], c["req_table"],
            c["w"], c["qsl"], c["row_ends"], weight_scale=1.0, out=out,
        )

    B()
    torch.cuda.synchronize()
    us_b_eager = float("nan") if COLD else timeit(B)
    us_b = gpu_time(B)

    col = torch.arange(width, device=dev)[None, :]
    valid = col < c["row_ends"][:, None]
    a, b = la[valid].double(), out[valid].double()
    err = (a - b).abs().max().item() / max(a.abs().max().item(), 1e-12)
    print(
        f"{name:<34} r={ratio} rows={T:>6} keys<={int(c['row_ends'].max()):>6} | "
        f"GPU A {us_a:7.1f} B {us_b:7.1f} us A/B {us_a / us_b:5.2f}x | "
        f"eager A {us_a_eager:7.1f} B {us_b_eager:7.1f} | A sched(eager) {us_sched:5.1f} | rel_err {err:.1e}",
        flush=True,
    )


if __name__ == "__main__":
    only = os.environ.get("ONLY", "")
    prefill = [  # (name, [(prefix, new_tokens), ...])
        ("1x4k fresh", [(0, 4096)]),
        ("1x8k chunk2 (4k on 4k)", [(4096, 4096)]),
        ("1x32k last chunk (4k on 28k)", [(28672, 4096)]),
        ("1x16k fresh chunk", [(0, 16384)]),
        ("1x128k last chunk (4k on 124k)", [(126976, 4096)]),
        ("8x512 fresh", [(0, 512)] * 8),
        ("4x1k on 30k prefix (agentic)", [(30720, 1024)] * 4),
    ]
    decode = [  # (name, bs, ctx, qlen)  qlen>1 = DSpark target verify rows
        ("decode bs1 ctx8k", 1, 8192, 1),
        ("decode bs8 ctx8k", 8, 8192, 1),
        ("decode bs32 ctx32k", 32, 32768, 1),
        ("decode bs64 ctx8k", 64, 8192, 1),
        ("verify bs8 ctx8k q5", 8, 8192, 5),
        ("verify bs32 ctx32k q5", 32, 32768, 5),
        ("decode bs16 ctx64k", 16, 65536, 1),
        ("decode bs64 ctx100k", 64, 102400, 1),
        ("verify bs64 ctx64k q5", 64, 65536, 5),
    ]
    for ratio in (2, 1):
        if only != "decode":
            for name, reqs in prefill:
                run(name, reqs, ratio, is_decode=False)
        if only != "prefill":
            for name, bs, ctx, ql in decode:
                run(name, [(ctx - ql, ql)] * bs, ratio, is_decode=True)
