#!/usr/bin/env python3
"""I4 microbench: is the prefill select (top-k v2 streaming + layer-20 block-max publish) at the HBM roofline?
  PYTHONPATH=/sgl-workspace/sglang-fp4idx/python HIP_VISIBLE_DEVICES=0 python3 i4_select_bench.py
Compares each op against amax(dim=1) over the same fp32 logits rectangle (one achievable HBM read).
Rows are a 16k-token prefill chunk at the end of a context of LC compressed positions (causal lengths).
Env: ROWS (default 16384), LCS (default 16384,32768,65536,131072).
"""
import os

import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import (
    candidate_block_scores,
    select_candidate_blocks_hip,
    topk_transform_paged_hip,
)

dev = "cuda"
ROWS = int(os.environ.get("ROWS", 16384))
LCS = [int(x) for x in os.environ.get("LCS", "16384,32768,65536,131072").split(",")]
TOPK, CBLK, CTOPK = 512, 8, 2048


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


def main():
    print(f"rows={ROWS} index_topk={TOPK} candidate block={CBLK} x {CTOPK}")
    for lc in LCS:
        torch.manual_seed(0)
        logits = torch.randn(ROWS, lc, device=dev, dtype=torch.float32)
        # causal: row t reaches lc - ROWS + t + 1 positions (at least 1)
        lens = (torch.arange(ROWS, device=dev, dtype=torch.int32) + (lc - ROWS + 1)).clamp_(min=1, max=lc)
        gb = lens.sum().item() * 4 / 1e9  # reachable bytes, what an exact pass must read
        ids = torch.empty(ROWS, TOPK, device=dev, dtype=torch.int32)
        t_amax = gpu_ms(lambda: logits.amax(dim=1))
        t_topk = gpu_ms(lambda: topk_transform_paged_hip(logits, lens, None, ids, 1, None))
        t_bs = gpu_ms(lambda: candidate_block_scores(logits, lens, block_size=CBLK, fill_tail=False))
        t_pub = gpu_ms(lambda: select_candidate_blocks_hip(logits, lens, topk_blocks=CTOPK, block_size=CBLK))
        full_gb = ROWS * lc * 4 / 1e9
        print(
            f"lc={lc:6d} reach {gb:5.2f} GB (rect {full_gb:5.2f}) | amax {t_amax:6.2f} ms ({full_gb / t_amax:5.2f} TB/s)"
            f" | topk {t_topk:6.2f} ms (1-read {gb / t_topk:5.2f} TB/s, 2-read {2 * gb / t_topk:5.2f})"
            f" | blockmax {t_bs:6.2f} ms ({gb / t_bs:5.2f} TB/s) | publish {t_pub:6.2f} ms"
        )
        del logits
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
