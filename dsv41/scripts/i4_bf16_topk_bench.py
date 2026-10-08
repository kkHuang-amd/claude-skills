#!/usr/bin/env python3
"""I4: top-k v2 + layer-20 publish on bf16 logits vs fp32 (needs the SGL_TOPK_BF16 variant, branch dsv41-i4-bf16-logits).
  PYTHONPATH=/sgl-workspace/sglang-i4/python HIP_VISIBLE_DEVICES=0 python3 i4_bf16_topk_bench.py
Check: per row the multiset of selected values equals torch.topk's (ties may pick other positions), and every
index is < the row's length. Timing as in i4_select_bench.py (16384 causal rows, ms, median of 5).
Env: ROWS (default 16384), LCS (default 4096,16384,32768,65536,131072), CHECK_ROWS (default 512).
"""
import os

import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import select_candidate_blocks_hip, topk_transform_paged_hip

dev = "cuda"
ROWS = int(os.environ.get("ROWS", 16384))
LCS = [int(x) for x in os.environ.get("LCS", "4096,16384,32768,65536,131072").split(",")]
CHECK_ROWS = int(os.environ.get("CHECK_ROWS", 512))
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


def check(logits, lens):
    rows = min(CHECK_ROWS, logits.shape[0])
    lg, ln = logits[-rows:].contiguous(), lens[-rows:].contiguous()
    ids = torch.empty(rows, TOPK, device=dev, dtype=torch.int32)
    topk_transform_paged_hip(lg, ln, None, ids, 1, None)
    bad = 0
    for r in range(rows):
        n = int(ln[r])
        k = min(TOPK, n)
        got = ids[r, :k].long()
        if (got < 0).any() or (got >= n).any() or got.unique().numel() != k or (ids[r, k:] != -1).any():
            bad += 1
            continue
        ref = lg[r, :n].float().topk(k).values.sort().values
        if not torch.equal(lg[r, got].float().sort().values, ref):
            bad += 1
    return bad, rows


def main():
    print(f"rows={ROWS} index_topk={TOPK} candidate block={CBLK} x {CTOPK}")
    for lc in LCS:
        torch.manual_seed(0)
        f32 = torch.randn(ROWS, lc, device=dev, dtype=torch.float32)
        lens = (torch.arange(ROWS, device=dev, dtype=torch.int32) + (lc - ROWS + 1)).clamp_(min=1, max=lc)
        out = []
        for name, lg in (("fp32", f32), ("bf16", f32.bfloat16())):
            ids = torch.empty(ROWS, TOPK, device=dev, dtype=torch.int32)
            bad, n = check(lg, lens)
            t_topk = gpu_ms(lambda: topk_transform_paged_hip(lg, lens, None, ids, 1, None))
            t_pub = gpu_ms(lambda: select_candidate_blocks_hip(lg, lens, topk_blocks=CTOPK, block_size=CBLK))
            t_amax = gpu_ms(lambda: lg.amax(dim=1))
            out.append(f"{name}: amax {t_amax:5.2f} topk {t_topk:5.2f} publish {t_pub:5.2f} bad {bad}/{n}")
            del lg
        print(f"lc={lc:6d} | " + " | ".join(out))
        del f32
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
