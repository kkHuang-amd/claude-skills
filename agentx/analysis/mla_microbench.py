#!/usr/bin/env python3
"""Standalone microbenchmark of the DSv4 MLA paged-decode kernel.

    HIP_VISIBLE_DEVICES=0 python3 mla_microbench.py [--quick]

Why this exists: in the c128 trace the kernel's per-call time does not follow
any work model. Rank 6 has MORE context and MORE batch than rank 1 and is 20 %
faster; within rank 0 the cost jumps 40 % between bs 18 and bs 20. Clocks are
flat (2372-2393 MHz, throttle 0) and `umc_activity` is 19.5 %, so it is neither
DVFS nor bandwidth. Running the kernel alone, on a dedicated GPU, with shapes
we choose, separates the possibilities that the trace cannot.

Shapes come from the DSv4-Pro config and the launcher, not from guesswork:
  H = 128 (num_attention_heads), D = 512 (head_dim), q bf16
  KV fp8_e4m3 + 1x64 fp32 block scales -> 512 + 32 = 544 B per KV token
  T = bs * 7 (speculative-num-draft-tokens), CONFIRMED by the fused/split
      boundary: split needs T <= 96, and bs 9/10 -> T 63/70 use split while
      bs 14 -> T 98 uses fused, exactly as observed in the trace.
  block_h = next_pow2(min(H, 64)) = 64, so the grid is (T, 2).

The leading hypothesis this tests: at 256 CUs a grid of 2T CTAs is
WAVE-QUANTISED. bs 18 -> 252 CTAs fits one wave; bs 20 -> 280 spills 24 CTAs
into a second, which would explain a 40 % jump for a 11 % batch increase. If
true, the kernel's cost is a step function of bs and its per-call time in the
trace says nothing about how much KV it read.

Prints achieved bandwidth and FLOP/s against MI355X peaks, and inverts the
sweep: which (T, kv_len) reproduces the 330.7 / 266.1 / 166.3 us seen in situ.
"""
import argparse
import math
import sys

sys.path.insert(0, "/sgl-workspace/sglang-MegaMoE/python")

import torch  # noqa: E402

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (  # noqa: E402
    _FP8_DTYPE,
    _FP8_GROUP_SIZE,
    _kv_splits_heuristic,
    _sparse_attn_v4_paged_decode_triton,
)

H, D = 128, 512
BLOCK_H = 64
NUM_CU = 256
BYTES_PER_KV_TOK = D + (D // _FP8_GROUP_SIZE) * 4       # fp8 data + fp32 scales
PEAK_BW = 8.0e12
PEAK_FLOPS = 2.5e15

# Pool big enough that a sweep cannot sit in cache: 4M slots x 512 B = 2 GB.
NSLOT = 4 << 20

# In-situ points to invert against (rank, bs, us/call), from the c128 pdi=24
# steady-state trace.
INSITU = [(1, 14, 330.7), (6, 16, 266.1), (0, 18, 166.3), (0, 20, 231.7),
          (7, 10, 163.5), (3, 9, 117.5)]


def make_inputs(T, kv_len, dev):
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(NSLOT, D, dtype=torch.bfloat16, device=dev).to(_FP8_DTYPE)
    scales = torch.rand(NSLOT, D // _FP8_GROUP_SIZE, dtype=torch.float32,
                        device=dev) * 0.1 + 0.05
    # Random slots: the paged pool is scattered in production, and a
    # contiguous index would hand the kernel a coalescing it does not have.
    idx = torch.randint(0, NSLOT, (T * kv_len,), dtype=torch.int32, device=dev)
    indptr = torch.arange(0, (T + 1) * kv_len, kv_len,
                          dtype=torch.int32, device=dev)
    sink = torch.zeros(H, dtype=torch.bfloat16, device=dev)
    return q, kv, scales, idx, indptr, sink


def bench_ragged(T, lens, dev, iters=30, warmup=8, splits=None):
    """Same as bench() but with a per-token kv_len vector, which is what
    production hands the kernel. Separates 'cost follows the TOTAL KV in the
    batch' from 'cost follows the LONGEST sequence in the batch' -- the two
    have opposite consequences for a token-balancing load balancer."""
    import math as _m
    dev_t = torch.device(dev)
    total = int(sum(lens))
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev_t)
    kv = torch.randn(NSLOT, D, dtype=torch.bfloat16, device=dev_t).to(_FP8_DTYPE)
    sc = torch.rand(NSLOT, D // _FP8_GROUP_SIZE, dtype=torch.float32,
                    device=dev_t) * 0.1 + 0.05
    idx = torch.randint(0, NSLOT, (total,), dtype=torch.int32, device=dev_t)
    indptr = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()),
                          dtype=torch.int32, device=dev_t)
    sink = torch.zeros(H, dtype=torch.bfloat16, device=dev_t)
    args = (q, kv, idx, indptr, sink, 1.0 / _m.sqrt(D))
    kw = dict(kv_scales=sc)
    if splits is not None:
        kw["kv_splits"] = splits
    for _ in range(warmup):
        _sparse_attn_v4_paged_decode_triton(*args, **kw)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        _sparse_attn_v4_paged_decode_triton(*args, **kw)
    b.record()
    torch.cuda.synchronize()
    us = a.elapsed_time(b) * 1000.0 / iters
    del q, kv, sc, idx, indptr, sink
    torch.cuda.empty_cache()
    return us


def bench(T, kv_len, dev, iters=30, warmup=8, splits=None):
    q, kv, scales, idx, indptr, sink = make_inputs(T, kv_len, dev)
    scale = 1.0 / math.sqrt(D)
    args = (q, kv, idx, indptr, sink, scale)
    kw = dict(kv_scales=scales)
    if splits is not None:
        kw["kv_splits"] = splits
    for _ in range(warmup):
        _sparse_attn_v4_paged_decode_triton(*args, **kw)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        _sparse_attn_v4_paged_decode_triton(*args, **kw)
    e.record()
    torch.cuda.synchronize()
    us = s.elapsed_time(e) * 1000.0 / iters
    del q, kv, scales, idx, indptr, sink
    torch.cuda.empty_cache()
    return us


def report(T, kv_len, us):
    splits = _kv_splits_heuristic(T, H, BLOCK_H, num_cu=NUM_CU)
    ctas = T * (H // BLOCK_H) * splits
    waves = ctas / NUM_CU
    kvb = T * kv_len * BYTES_PER_KV_TOK
    qb = T * H * D * 2
    fl = T * H * kv_len * (2 * D) * 2
    t = us * 1e-6
    return dict(T=T, bs=T / 7, kv_len=kv_len, us=us, splits=splits, ctas=ctas,
                waves=waves, kv_mb=kvb / 1e6,
                bw=(kvb + qb) / t, bw_pct=100 * (kvb + qb) / t / PEAK_BW,
                tf=fl / t, fl_pct=100 * fl / t / PEAK_FLOPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}  "
          f"H={H} D={D} block_h={BLOCK_H} KV={_FP8_DTYPE} "
          f"{BYTES_PER_KV_TOK} B/tok  pool={NSLOT} slots")

    bs_list = [9, 10, 14, 16, 18, 20] if a.quick else \
        [9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 24, 28, 32, 40]
    kv_list = [1024] if a.quick else [256, 512, 1024, 2048]

    print("\n=== sweep: bs (T = bs*7) x kv_len ===")
    print(f"{'bs':>4s} {'T':>5s} {'kv_len':>7s} {'splits':>6s} {'CTAs':>6s} "
          f"{'waves':>6s} {'us':>8s} {'KV MB':>7s} {'GB/s':>8s} {'%BW':>6s} "
          f"{'TF/s':>7s} {'%FL':>5s}")
    grid = {}
    for kv_len in kv_list:
        for bs in bs_list:
            T = bs * 7
            r = report(T, kv_len, bench(T, kv_len, dev))
            grid[(bs, kv_len)] = r
            print(f"{r['bs']:4.0f} {r['T']:5d} {r['kv_len']:7d} "
                  f"{r['splits']:6d} {r['ctas']:6d} {r['waves']:6.2f} "
                  f"{r['us']:8.1f} {r['kv_mb']:7.1f} {r['bw'] / 1e9:8.0f} "
                  f"{r['bw_pct']:6.2f} {r['tf'] / 1e12:7.0f} {r['fl_pct']:5.1f}")
        print()

    print("=== inversion: what kv_len reproduces the in-situ us/call? ===")
    print("(linear in kv_len at fixed bs, so interpolate the two nearest)")
    print(f"{'rank':>4s} {'bs':>3s} {'in-situ us':>10s} {'implied kv_len':>15s}")
    for rank, bs, us in INSITU:
        pts = sorted((k[1], v["us"]) for k, v in grid.items() if k[0] == bs)
        if len(pts) < 2:
            continue
        # Fit us = a + b*kv_len over the sweep, then solve for the in-situ time.
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        den = sum((x - mx) ** 2 for x, _ in pts)
        b = sum((x - mx) * (y - my) for x, y in pts) / den if den else 0.0
        aa = my - b * mx
        implied = (us - aa) / b if b else float("nan")
        print(f"{rank:4d} {bs:3d} {us:10.1f} {implied:15.0f}")
    # The heuristic picks kv_splits from capture-time scalars only, and at the
    # production shapes it lands on 1 while the grid covers 0.77-0.98 of a wave.
    # Time is flat in CTA count within a wave and linear in the per-CTA K loop,
    # so splitting K should be nearly free parallelism. Test it directly.
    print("\n=== is the kv_splits heuristic mistuned at the production point? ===")
    print(f"{'bs':>4s} {'kv_len':>7s} {'splits':>6s} {'CTAs':>6s} {'waves':>6s} "
          f"{'us':>8s} {'vs heuristic':>13s}")
    for bs, kv_len in ((14, 708), (16, 551), (18, 321), (20, 244)):
        T = bs * 7
        auto = _kv_splits_heuristic(T, H, BLOCK_H, num_cu=NUM_CU)
        base = None
        for sp in (1, 2, 4, 8):
            us = bench(T, kv_len, dev, splits=sp)
            if sp == auto:
                base = us
            ctas = T * (H // BLOCK_H) * sp
            tag = "  <- heuristic" if sp == auto else (
                f"{100 * (us / base - 1):+12.0f} %" if base else "")
            print(f"{bs:4d} {kv_len:7d} {sp:6d} {ctas:6d} {ctas / NUM_CU:6.2f} "
                  f"{us:8.1f} {tag:>13s}")
        print()

    # Does the cost follow the batch's TOTAL KV or its LONGEST sequence? If the
    # latter, a token-balancing load balancer cannot help this kernel: the
    # longest conversation still lands on somebody.
    print("\n=== total KV or longest sequence? (bs=14, T=98) ===")
    print(f"{'shape':<26s} {'mean':>6s} {'max':>6s} {'total kKV':>10s} {'us':>8s}")
    T = 98
    cases = [
        ("uniform 500", [500] * T),
        ("uniform 1000", [1000] * T),
        ("half 250 / half 750", [250] * (T // 2) + [750] * (T - T // 2)),
        ("one long: 493 + 1x1000", [493] * (T - 1) + [1000]),
        ("one long: 250 + 1x1000", [250] * (T - 1) + [1000]),
    ]
    for name, lens in cases:
        us = bench_ragged(T, lens, dev)
        print(f"{name:<26s} {sum(lens) / T:6.0f} {max(lens):6d} "
              f"{sum(lens) / 1000:10.1f} {us:8.1f}")
    # If cost is set by the longest CTA, split-K should pay HERE even though it
    # lost on uniform shapes -- it shortens the straggler. This is the test of
    # the fix, not of the heuristic.
    print("\n   split-K on the ragged shape (uniform split lost on uniform shapes):")
    print(f"{'   shape':<26s} {'splits':>6s} {'us':>8s}")
    for name, lens in (("one long: 493 + 1x1000", [493] * 97 + [1000]),
                       ("uniform 500 (the floor)", [500] * 98)):
        for sp in (1, 2, 4):
            print(f"   {name:<23s} {sp:6d} "
                  f"{bench_ragged(98, lens, dev, splits=sp):8.1f}")
    print("Rows 1/3/4 hold the MEAN near 500 and vary the MAX. If they differ,")
    print("cost follows the longest sequence and `total_tokens` balancing is")
    print("the wrong knob for this kernel.")

    print("\nAn implied kv_len above the topk cap (1024) means the in-situ time")
    print("cannot be produced by this kernel doing this work -- look for what")
    print("else is inside that duration. Below it, the kernel is honest and the")
    print("rank spread is a KV-length spread.")


if __name__ == "__main__":
    main()
