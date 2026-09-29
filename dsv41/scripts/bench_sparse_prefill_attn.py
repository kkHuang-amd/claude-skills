#!/usr/bin/env python3
"""Microbench: DSV4.1 sparse-attention PREFILL on one gfx950 GPU, current SGLang kernel vs aiter OPUS.

Shapes follow DSV4.1-Flash at TP4: H=16 heads/rank, head 512 (448 fp8 NoPE + 64 bf16 RoPE), every query token
attends TOPK=512 main (compressed/top-k) slots + SWA=128 sliding-window slots.
  current   : sglang aiter_sparse_decode_fwd -> aiter pa_decode_sparse (packed 584 B fp8 pages), kv_splits=1
  opus_fp8  : aiter pa_sparse_prefill_fp8_opus (two pools: fp8 NoPE [P,512] + bf16 RoPE [P,64])
  opus_bf16 : aiter pa_sparse_prefill_opus (bf16 [P,512] pools = vLLM's gather->BF16-workspace path; the
              gather/dequant cost itself is NOT included)
Timing only (random data, no accuracy check).
  HIP_VISIBLE_DEVICES=4 python3 bench_sparse_prefill_attn.py [--tokens 1024 4096 16384] [--pool 16384]
"""
import argparse, sys, types
import torch

sys.path.insert(0, "/sgl-workspace/sglang-dsv41/python")
sys.path.insert(0, "/sgl-workspace/aiter/op_tests")
H, D, NOPE, ROPE, PAGE, BYTES = 16, 512, 448, 64, 256, 584
TOPK, SWA = 512, 128
SCALE = D ** -0.5
dev = torch.device("cuda")


def timeit(fn, warm=5, iters=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def pack_cache(slots):
    """Packed fp8 page cache [blocks, PAGE, 1, 584] as the served HIP path stores it (random keys)."""
    nb = (slots + PAGE - 1) // PAGE
    raw = torch.randint(0, 120, (nb, PAGE * BYTES), dtype=torch.uint8, device=dev)  # finite fp8 bytes
    raw.view(nb, -1)[:, PAGE * 576:].fill_(120)  # scale exponents ~2^-7
    rope = (torch.randn(nb, PAGE, ROPE, device=dev) * 0.5).to(torch.bfloat16)
    raw[:, : PAGE * 576].view(nb, PAGE, 576)[:, :, NOPE:] = rope.view(torch.uint8).view(nb, PAGE, 2 * ROPE)
    return raw.view(nb, PAGE, 1, BYTES).view(torch.float8_e4m3fn)


def csr(T, per_row, pool, g):
    idx = torch.randint(0, pool, (T * per_row,), generator=g, device="cpu", dtype=torch.int32).to(dev)
    indptr = torch.arange(0, T * per_row + 1, per_row, dtype=torch.int32, device=dev)
    return idx, indptr


def swa_csr(T):
    """Each token attends the previous SWA tokens of the current chunk (clamped at 0)."""
    t = torch.arange(T, device=dev, dtype=torch.int32).unsqueeze(1)
    idx = (t - torch.arange(SWA, device=dev, dtype=torch.int32)).clamp(min=0).reshape(-1).contiguous()
    return idx, torch.arange(0, T * SWA + 1, SWA, dtype=torch.int32, device=dev)


def bench(T, pool):
    g = torch.Generator().manual_seed(0)
    sink = (torch.randn(H, device=dev) * 0.5).float()
    res = {}
    # --- current SGLang path
    from sglang.srt.layers.attention.hip_flash_mla import aiter_sparse_decode_fwd
    q = (torch.randn(T, 1, H, D, device=dev) * 0.5).to(torch.bfloat16)
    main_cache, swa_cache = pack_cache(pool), pack_cache(max(T, SWA))
    main_idx = torch.randint(0, pool, (T, 1, TOPK), generator=g, dtype=torch.int32).to(dev)
    swa_idx = swa_csr(T)[0].view(T, 1, SWA)
    try:
        res["current"] = timeit(lambda: aiter_sparse_decode_fwd(q, main_cache, main_idx, sink, SCALE,
                                                                extra_k_cache=swa_cache, extra_indices_in_kvcache=swa_idx))
    except Exception as ex:  # report, keep going
        res["current"] = f"ERR {type(ex).__name__}: {str(ex)[:120]}"
    # --- OPUS fp8 (two pools)
    from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_fp8_opus, pa_sparse_prefill_opus
    from test_pa_sparse_prefill import _quantize_nope
    qn, _ = _quantize_nope(torch.randn(T * H, NOPE, device=dev) * 0.5)
    q_nope, q_rope = qn.view(T, H, 512), (torch.randn(T, H, ROPE, device=dev) * 0.5).to(torch.bfloat16)
    kvn_pool, _ = _quantize_nope(torch.randn(pool, NOPE, device=dev) * 0.5)
    kvr_pool = (torch.randn(pool, ROPE, device=dev) * 0.5).to(torch.bfloat16)
    kvn_ext, _ = _quantize_nope(torch.randn(T, NOPE, device=dev) * 0.5)
    kvr_ext = (torch.randn(T, ROPE, device=dev) * 0.5).to(torch.bfloat16)
    pi, pp = csr(T, TOPK, pool, g)
    ei, ep = swa_csr(T)
    try:
        res["opus_fp8"] = timeit(lambda: pa_sparse_prefill_fp8_opus(q_nope, q_rope, kvn_pool, kvr_pool, pi, pp,
                                                                    kvn_ext, kvr_ext, ei, ep, sink, SCALE))
    except Exception as ex:
        res["opus_fp8"] = f"ERR {type(ex).__name__}: {str(ex)[:120]}"
    # --- OPUS bf16 (vLLM-style workspace)
    qb = (torch.randn(T, H, D, device=dev) * 0.5).to(torch.bfloat16)
    kv_pool = (torch.randn(pool, D, device=dev) * 0.5).to(torch.bfloat16)
    kv_ext = (torch.randn(T, D, device=dev) * 0.5).to(torch.bfloat16)
    try:
        res["opus_bf16"] = timeit(lambda: pa_sparse_prefill_opus(qb, kv_pool, pi, pp, kv_ext, ei, ep, sink, SCALE))
    except Exception as ex:
        res["opus_bf16"] = f"ERR {type(ex).__name__}: {str(ex)[:120]}"
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[1024, 4096, 16384])
    ap.add_argument("--pool", type=int, default=16384, help="main (top-k) KV pool slots")
    a = ap.parse_args()
    print(f"H={H} D={D} topk={TOPK} swa={SWA} pool={a.pool}  times in ms/call (one layer)")
    for T in a.tokens:
        r = bench(T, a.pool)
        cur = r["current"]
        line = f"T={T:6d}  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items())
        if isinstance(cur, float):
            line += "  | speedup " + " ".join(f"{k}={cur / v:.2f}x" for k, v in r.items() if k != "current" and isinstance(v, float))
        print(line, flush=True)
