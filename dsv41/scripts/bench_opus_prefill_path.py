#!/usr/bin/env python3
"""Per-layer time of the full OPUS prefill path (dequant-gather + CSR + OPUS) vs the served aiter_sparse
kernel, on the unit test's synthetic case. Usage: HIP_VISIBLE_DEVICES=4 python3 bench_opus_prefill_path.py"""
import sys, torch
sys.path.insert(0, "/sgl-workspace/sglang-dsv41/python")
sys.path.insert(0, "/sgl-workspace/sglang-dsv41/test/registered/kernels/ops/attention/dsv4")
from test_hip_opus_prefill import _case, SCALE
from sglang.kernels.ops.attention.dsv4.opus_prefill_hip import OpusPrefillRows, opus_sparse_prefill
from sglang.srt.layers.attention.hip_flash_mla import aiter_sparse_decode_fwd

def timeit(fn, warm=3, iters=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / iters

for reqs in ([(0, 4096)] * 4, [(0, 1024)], [(12288, 4096)]):
    for ratio in (0, 1, 2):
        c = _case(ratio, reqs, torch.device("cuda"))
        cur = timeit(lambda: aiter_sparse_decode_fwd(c["q"].unsqueeze(1), c["swa_cache"], c["swa_idx"], c["sink"], SCALE,
                                                     extra_k_cache=c["c_cache"], extra_indices_in_kvcache=c["c_idx"]))
        mk = lambda: OpusPrefillRows(req_to_token=c["req_to_token"], full_to_swa=c["full_to_swa"], req_pool_indices=c["req_pool"],
                                     seq_lens_cpu=c["seq_lens"], extend_lens_cpu=c["ext_lens"], positions=c["positions"])
        plan_ms = timeit(lambda: mk().compressed(ratio) if ratio else mk())
        plan = mk()
        run = lambda: opus_sparse_prefill(plan, q=c["q"], ratio=ratio, swa_cache=c["swa_cache"], extra_cache=c["c_cache"],
                                          raw_indices=c["raw"], raw_lens=c["raw_lens"], attn_sink=c["sink"], softmax_scale=SCALE)
        new = timeit(run)
        print(f"reqs={reqs[0]}x{len(reqs)} R={ratio} T={c['q'].shape[0]:6d} current={cur:.3f} opus_path={new:.3f} "
              f"(+plan once/fwd {plan_ms:.3f}) ms  speedup={cur/new:.2f}x", flush=True)
