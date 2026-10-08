#!/usr/bin/env python3
"""I4: SGLang prefill FP4 index scorer writing bf16 logits vs fp32, then top-k on each (V4.1 shapes).
  PYTHONPATH=/sgl-workspace/aiter-i4:/sgl-workspace/sglang-i4/python HIP_VISIBLE_DEVICES=0 python3 i4_bf16_scorer_bench.py
Check: bf16 logits == fp32 logits rounded to bf16 on every row's valid columns (bitwise).
One request, NEW new tokens at the end of a context of CTX tokens, compress ratio R: lc = CTX // R.
Env: NEW (default 16384), CASES (default "16384:2,32768:2,65536:2,131072:2,131072:1" as ctx:ratio).
"""
import importlib.util
import os

import torch

spec = importlib.util.spec_from_file_location(
    "t", "/sgl-workspace/aiter-i4/op_tests/test_flydsl_pa_mqa_logits_fp4.py"
)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)

from sglang.kernels.ops.attention.dsv4 import fp4_indexer_hip as S
from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import topk_transform_paged_hip

dev, PAGE, D, H, MFMA_M, BUCKET, TOPK = "cuda", 64, 128, 32, 16, 64, 512
NEW = int(os.environ.get("NEW", 16384))
CASES = [tuple(int(v) for v in c.split(":")) for c in os.environ.get(
    "CASES", "16384:2,32768:2,65536:2,131072:2,131072:1").split(",")]


def pack_q_scales(q_e8m0):
    """aiter #6145 pa_mqa_logits_fp4_rowgroup.pack_q_scales (not in this aiter)."""
    b, nn, h, groups = q_e8m0.shape
    m_tiles, k_tiles = h // MFMA_M, groups // 4
    qs_pad = (m_tiles + 3) // 4 * 4
    qe = q_e8m0.reshape(b, nn, m_tiles, MFMA_M, k_tiles, 4).permute(0, 1, 4, 5, 3, 2).contiguous()
    return torch.nn.functional.pad(qe, (0, qs_pad - m_tiles)).contiguous()


def gpu_ms(fn, iters=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return sorted(ts)[len(ts) // 2]


def make_case(ctx, ratio):
    t.setup_seed(t.SEED)
    prefix = ctx - NEW
    keys = ctx // ratio
    pages = max(1, -(-keys // PAGE))
    num_pages = pages + 1
    table = torch.randperm(num_pages - 1, device=dev).to(torch.int32)[:pages].view(1, -1)
    kv_bf16 = torch.randn(1, num_pages * PAGE, D, dtype=torch.bfloat16, device=dev)
    ident = torch.arange(num_pages, dtype=torch.int32, device=dev).view(1, -1)
    kv_cache, kv_scale, _, _ = t.create_paged_preshuffle_kv_fp4(kv_bf16, PAGE, num_pages, ident)
    row_ends = torch.tensor([(p + 1) // ratio for p in range(prefix, ctx)], dtype=torch.int32, device=dev)
    q = torch.randn(NEW, H, D, dtype=torch.bfloat16, device=dev)
    qp, qe = t.fp4_quant_e2m1_with_e8m0(q.reshape(-1, D))
    qs = pack_q_scales(qe.view(NEW, 1, H, D // 32)).view(NEW, 1, 4, 16, 4)
    w = (torch.randn(NEW, H, device=dev) * 0.1).to(torch.bfloat16)
    return dict(qp=qp.view(NEW, H, D // 2), qs=qs, kv_cache=kv_cache, kv_scale=kv_scale,
                row_table=table.expand(NEW, -1).contiguous(), row_ends=row_ends, w=w)


def main():
    print(f"new={NEW} heads={H} topk={TOPK}")
    for ctx, ratio in CASES:
        c = make_case(ctx, ratio)
        ws = S.prepare_fp4_prefill_workspace(c["row_table"], c["row_ends"], page_table_bucket=BUCKET)

        def score():
            return S.aiter_fp4_paged_mqa_logits(
                q_fp4=c["qp"], q_scale=c["qs"], k_payload=c["kv_cache"], k_scale=c["kv_scale"],
                weights=c["w"], page_table=c["row_table"], c4_seq_lens=c["row_ends"], weight_scale=1.0,
                is_decode=False, page_table_bucket=BUCKET, prefill_workspace=ws)

        res = {}
        for name, flag in (("fp32", False), ("bf16", True)):
            S._PREFILL_LOGITS_BF16 = flag
            lg = score()
            ref = lg.clone()
            ids = torch.empty(NEW, TOPK, device=dev, dtype=torch.int32)
            t_sc = gpu_ms(score)
            t_tk = gpu_ms(lambda: topk_transform_paged_hip(lg, c["row_ends"], None, ids, 1, None))
            res[name] = (ref, t_sc, t_tk, lg.dtype)
        f32, b16 = res["fp32"][0], res["bf16"][0]
        lc = int(c["row_ends"].max())
        cols = torch.arange(lc, device=dev)
        valid = cols[None, :] < c["row_ends"][:, None]
        diff = (f32[:, :lc].bfloat16().view(torch.int16) != b16[:, :lc].view(torch.int16)) & valid
        print(f"ctx={ctx:6d} r={ratio} lc={lc:6d} | score fp32 {res['fp32'][1]:6.2f} bf16 {res['bf16'][1]:6.2f} ms"
              f" | topk fp32 {res['fp32'][2]:5.2f} bf16 {res['bf16'][2]:5.2f} ms | dtype {res['bf16'][3]}"
              f" | bf16 != round(fp32): {int(diff.sum())} of {int(valid.sum())}")
        del res, f32, b16, c, ws
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
