#!/usr/bin/env python3
"""MLA decode at the REAL kv_len distribution, including the tail.

    HIP_VISIBLE_DEVICES=0 python3 analysis/mla_tail_bench.py [--bs 10 14 18]

`mla_microbench.py` sweeps kv_len <= 2048, but production reaches ~5,000 on the
HCA layers (compress_ratio 128 has no `index_topk` clamp, so kv_len ~ ctx/128
and ISL p99 634,941 / 128 = 4,960). The tail is exactly where a straggler-aware
kernel would win, and it has never been measured.

Three cases per batch size, all on the same ragged vector:

  ragged      the measured distribution (p50 ~1,000, p99 ~3,100, max ~5,000);
  flat        every token at the ragged vector's MEAN -- identical total work,
              zero dispersion. This is the floor a PERFECT straggler fix
              reaches, so `ragged - flat` is the measured achievable win per
              call, as opposed to the modelled one;
  split-K     kv_splits pinned to 1/2/4/8 on the ragged vector. Known to win
              ragged and lose uniform shapes, which is why it cannot be applied
              blind: `_kv_splits_heuristic` reads only capture-time scalars.
"""
import argparse
import sys

sys.path.insert(0, "/sgl-workspace/sglang-MegaMoE/python")
sys.path.insert(0, "/workspace/claude-skills/agentx/analysis")

import torch  # noqa: E402

from mla_microbench import H, D, NUM_CU, BLOCK_H, bench_ragged  # noqa: E402
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (  # noqa: E402
    _kv_splits_heuristic,
)

# LogNormal(ln 1000, 0.486): median 1,000 and p99 = 1000*exp(2.326*0.486) =
# 3,096, which are the two measured anchors. Capped at 5,000, the per-layer
# maximum actually observed (3,958-5,288 across layers).
MU, SIGMA, CAP = 1000.0, 0.486, 5000


def real_lens(T, seed=0, cap=CAP, median=MU):
    g = torch.Generator().manual_seed(seed)
    x = torch.exp(torch.log(torch.tensor(median))
                  + SIGMA * torch.randn(T, generator=g))
    return x.clamp(64, cap).to(torch.int64).tolist()


def q(lens, p):
    s = sorted(lens)
    return s[min(len(s) - 1, int(p * len(s)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, nargs="+", default=[10, 14, 18])
    ap.add_argument("--seed", type=int, default=0)
    # CSA layers (compress_ratio 4) are clamped to index_topk+128 = 1152;
    # HCA layers (compress_ratio 128) have no clamp and reach ~5,000.
    ap.add_argument("--cap", type=int, default=CAP)
    # HCA kv_len ~ context/128, so it GROWS through the run: a few hundred
    # early, ~1,300 at the 165-170k tok/req steady state. A captured graph is
    # reused across all of it, so a static per-layer split must win on the
    # whole range, not just at steady state.
    ap.add_argument("--median", type=float, default=MU)
    ap.add_argument("--splits", type=int, nargs="+", default=[1, 2, 4, 8])
    a = ap.parse_args()
    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}  H={H} D={D} CUs={NUM_CU}")

    for bs in a.bs:
        T = bs * 7
        lens = real_lens(T, a.seed, a.cap, a.median)
        mean = sum(lens) // len(lens)
        flat = [mean] * T
        disp = 1 - mean / max(lens)
        print(f"\n=== bs={bs} T={T}  p50={q(lens, .5)} p99={q(lens, .99)} "
              f"max={max(lens)} mean={mean}  (max-mean)/max={disp:.3f} "
              f"total={sum(lens)} ===")

        us_r = bench_ragged(T, lens, dev)
        us_f = bench_ragged(T, flat, dev)
        heur = _kv_splits_heuristic(T, H, BLOCK_H, num_cu=NUM_CU)
        print(f"ragged (heuristic splits={heur}) {us_r:8.1f} us")
        print(f"flat at the mean, same total    {us_f:8.1f} us   "
              f"win {us_r - us_f:6.1f} us ({100 * (us_r - us_f) / us_r:4.1f} %)"
              "  <- ceiling of a perfect straggler fix")
        for sp in a.splits:
            us = bench_ragged(T, lens, dev, splits=sp)
            print(f"  ragged split-K={sp:<2d}              {us:8.1f} us   "
                  f"{100 * (us - us_r) / us_r:+5.1f} % vs heuristic")


if __name__ == "__main__":
    main()
