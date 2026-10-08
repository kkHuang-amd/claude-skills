#!/usr/bin/env python3
"""I4: candidate_block_scores (unmasked fast path) vs a torch reference, edge shapes, fp32 + bf16, FILL_TAIL on/off.
  PYTHONPATH=/sgl-workspace/sglang-i4/python HIP_VISIBLE_DEVICES=0 python3 i4_blockmax_check.py
Env: DTYPES (default fp32,bf16; fp32 only on a branch without bf16 logits).
"""
import os

import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import candidate_block_scores

dev, BS = "cuda", 8


def reference(logits, lens, fill_tail):
    rows, width = logits.shape
    nb = -(-width // BS)
    x = torch.full((rows, nb * BS), float("-inf"), device=dev)
    x[:, :width] = logits.float()
    x[torch.arange(nb * BS, device=dev)[None, :] >= lens[:, None]] = float("-inf")
    ref = x.view(rows, nb, BS).amax(-1)
    blk = torch.arange(nb, device=dev)[None, :]
    last = ((lens - 1) // BS)[:, None]
    ref = torch.where(blk == last, torch.full_like(ref, float("inf")), ref)
    reach = blk * BS < lens[:, None]
    if fill_tail:
        ref = torch.where(reach, ref, torch.full_like(ref, float("-inf")))
    return ref, reach


def main():
    bad = cases = 0
    for width in (1, 7, 8, 4095, 4096, 4097, 9001, 65536, 131071):
        for dtype in [{"fp32": torch.float32, "bf16": torch.bfloat16}[d] for d in os.environ.get("DTYPES", "fp32,bf16").split(",")]:
            for fill_tail in (False, True):
                torch.manual_seed(width)
                rows = 257
                logits = torch.randn(rows, width + 13, device=dev).to(dtype)[:, :width]  # odd stride, view
                if logits.stride(0) * logits.element_size() % 16:
                    logits = torch.randn(rows, width, device=dev).to(dtype)
                lens = torch.randint(1, width + 1, (rows,), device=dev, dtype=torch.int32)
                lens[0], lens[-1] = width, 1
                got = candidate_block_scores(logits, lens, block_size=BS, fill_tail=fill_tail)
                ref, reach = reference(logits, lens, fill_tail)
                mask = torch.ones_like(reach) if fill_tail else reach
                ok = torch.equal(got[mask], ref[mask])
                cases += 1
                if not ok:
                    bad += 1
                    print(f"MISMATCH width={width} dtype={dtype} fill_tail={fill_tail}")
    print(f"block-max check: {cases - bad}/{cases} PASS")


if __name__ == "__main__":
    main()
