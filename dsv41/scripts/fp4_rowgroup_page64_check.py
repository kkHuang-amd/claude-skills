#!/usr/bin/env python3
"""P0.2 (FP4_INDEX_PLANE_PORT.md): aiter #6145 row-group FP4 MQA logits on 64-row pages.

Builds the existing page-64 preshuffle FP4 index plane (the layout SGLang's split pool writes),
scores ragged rows (each sequence's qlen rows, causal row_ends) with
`flydsl_pa_mqa_logits_fp4_rowgroup` (via the `query_start_loc` path), and checks it against the
dequantized reference. It also times the old per-row `pa_mqa_logits_fp4` on the same data.

Needs aiter >= 5b2f7d1d1:
  PYTHONPATH=/sgl-workspace/aiter-6145 HIP_VISIBLE_DEVICES=0 python3 fp4_rowgroup_page64_check.py
Env: AITER_DIR (default /sgl-workspace/aiter-6145), for importing op_tests helpers.
Output: one line per config on stdout, then PASS/FAIL.
"""
import importlib.util
import os
import sys

import torch

AITER_DIR = os.environ.get("AITER_DIR", "/sgl-workspace/aiter-6145")
spec = importlib.util.spec_from_file_location(
    "t", os.path.join(AITER_DIR, "op_tests/test_flydsl_pa_mqa_logits_fp4.py")
)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4 import compute_varctx_schedule

dev = "cuda"
PAGE, BLOCK_K, HEAD_DIM = 64, 256, 128


def run(batch, max_ctx, qlen, heads):
    t.setup_seed(t.SEED)
    ctx = torch.tensor(t._make_varctx(batch, max_ctx, PAGE), dtype=torch.int32, device=dev)
    ctx = torch.clamp(ctx, min=qlen)
    max_blocks = -(-max_ctx // BLOCK_K) * (BLOCK_K // PAGE)
    t_max = max_blocks * PAGE
    num_blocks = batch * max_blocks
    kv_bf16 = torch.randn(batch, t_max, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    bt = torch.randperm(num_blocks, device=dev).to(torch.int32).view(batch, max_blocks)
    kv_cache, kv_scale, kv_fp4, kv_e8m0 = t.create_paged_preshuffle_kv_fp4(
        kv_bf16, PAGE, num_blocks, bt
    )
    q = torch.randn(batch, qlen, heads, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    qp, qe = t.fp4_quant_e2m1_with_e8m0(q.reshape(-1, HEAD_DIM))
    qp = qp.view(batch, qlen, heads, HEAD_DIM // 2)
    qe = qe.view(batch, qlen, heads, HEAD_DIM // 32)
    qs = t.pack_q_scales(qe)
    w = (torch.randn(batch * qlen, heads, device=dev) * 0.1).to(torch.bfloat16)
    ws = 1.5
    ref = t.ref_mqa_logits_mixed(qp, qe, kv_fp4, kv_e8m0, w, ctx, next_n=qlen, weight_scale=ws)

    qsl = torch.arange(0, (batch + 1) * qlen, qlen, dtype=torch.int32, device=dev)
    lag = torch.arange(qlen - 1, -1, -1, dtype=torch.int32, device=dev)
    row_ends = (ctx[:, None] - lag).reshape(-1)
    out = torch.full((batch * qlen, t_max), float("-inf"), device=dev)

    def rowgroup():
        t.flydsl_pa_mqa_logits_fp4(
            qp, qs, kv_cache, kv_scale, bt, w, None, t_max, weight_scale=ws,
            kv_block_size=PAGE, row_ends=row_ends, query_start_loc=qsl,
            max_query_len=qlen, out=out,
        )

    rowgroup()
    torch.cuda.synchronize()
    mask = ~torch.isneginf(ref)
    got, want = out[mask].double(), ref[mask].double()
    err = (got - want).abs().max().item()
    ok = err < 1e-3 * want.abs().max().item() and bool(torch.isneginf(out[~mask]).all())
    _, us_rg = t.run_perftest(rowgroup, num_iters=20, num_warmup=3)

    us_old = float("nan")
    try:
        _, cta, total = compute_varctx_schedule(ctx, BLOCK_K, None, t_max, next_n=qlen)
        out2 = torch.full_like(out, float("-inf"))

        def old():
            t.flydsl_pa_mqa_logits_fp4(
                qp, qs, kv_cache, kv_scale, bt, w, ctx, t_max, weight_scale=ws,
                next_n=qlen, block_k=BLOCK_K, kv_block_size=PAGE,
                parallel_unit_num=total, out=out2, cta_info=cta, total_ctas=total,
            )

        old()
        torch.cuda.synchronize()
        err_old = (out2[mask].double() - want).abs().max().item()
        _, us_old = t.run_perftest(old, num_iters=20, num_warmup=3)
        old_note = f"old_err={err_old:.1e}"
    except Exception as e:  # old kernel may not take large next_n
        old_note = f"old=n/a ({type(e).__name__}: {str(e)[:80]})"
    print(
        f"b={batch} ctx<={max_ctx} qlen={qlen} heads={heads}: rowgroup err={err:.1e} "
        f"ok={ok} {us_rg:.1f} us | old {us_old:.1f} us {old_note}",
        flush=True,
    )
    return ok


if __name__ == "__main__":
    configs = [  # (batch, max_ctx, qlen, heads)
        (8, 8192, 1, 64), (32, 8192, 1, 64), (8, 8192, 4, 64), (8, 8192, 6, 64),
        (1, 8192, 128, 64), (1, 16384, 512, 64), (2, 8192, 256, 64), (4, 4096, 1, 32),
    ]
    results = [run(*c) for c in configs]
    print("PASS" if all(results) else "FAIL")
    sys.exit(0 if all(results) else 1)
