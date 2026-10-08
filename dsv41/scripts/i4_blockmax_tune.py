#!/usr/bin/env python3
"""I4: tune the layer-20 publish block-max kernel (_candidate_block_scores_kernel) toward one HBM read.
  PYTHONPATH=/sgl-workspace/sglang-i4/python HIP_VISIBLE_DEVICES=0 python3 i4_blockmax_tune.py
Times the publish split (block-max / top-k over block scores / whole publish) and sweeps BLOCKS_PER_PROGRAM x
num_warps for the block-max kernel; every variant is checked equal to the production launch.
Env: ROWS (default 16384), LC (default 131072), DTYPES (default bf16,fp32).
"""
import os

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4 import candidate_blocks_hip as C

dev = "cuda"
ROWS = int(os.environ.get("ROWS", 16384))
LC = int(os.environ.get("LC", 131072))
DTYPES = os.environ.get("DTYPES", "bf16,fp32").split(",")
CBLK, CTOPK = 8, 2048


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


@triton.jit
def _block_scores_1d_kernel(logits_ptr, seq_lens_ptr, out_ptr, logits_stride, width, num_blocks, out_stride,
                            BLOCK_SIZE: tl.constexpr, BLOCKS_PER_PROGRAM: tl.constexpr):
    """Variant: one contiguous 1-D load of BLOCKS_PER_PROGRAM * BLOCK_SIZE logits, reshaped to [blocks, BLOCK_SIZE]."""
    row = tl.program_id(0)
    block0 = tl.program_id(1) * BLOCKS_PER_PROGRAM
    length = tl.load(seq_lens_ptr + row)
    if block0 * BLOCK_SIZE >= length:
        return
    cols = block0 * BLOCK_SIZE + tl.arange(0, BLOCKS_PER_PROGRAM * BLOCK_SIZE)
    vals = tl.load(logits_ptr + row.to(tl.int64) * logits_stride + cols, mask=(cols < length) & (cols < width),
                   other=float("-inf")).to(tl.float32)
    scores = tl.max(tl.reshape(vals, (BLOCKS_PER_PROGRAM, BLOCK_SIZE)), axis=1)
    blocks = block0 + tl.arange(0, BLOCKS_PER_PROGRAM)
    scores = tl.where(blocks == (length - 1) // BLOCK_SIZE, float("inf"), scores)
    tl.store(out_ptr + row * out_stride + blocks, scores, mask=blocks < num_blocks)


def launch_1d(logits, lens, out, bpp, nw):
    rows, width = logits.shape
    nb = out.shape[1]
    _block_scores_1d_kernel[(rows, triton.cdiv(nb, bpp))](
        logits, lens, out, logits.stride(0), width, nb, out.stride(0),
        BLOCK_SIZE=CBLK, BLOCKS_PER_PROGRAM=bpp, num_warps=nw,
    )


@triton.jit
def _block_scores_rowmajor_kernel(logits_ptr, seq_lens_ptr, out_ptr, logits_stride, width, num_blocks, out_stride,
                                  BLOCK_SIZE: tl.constexpr, BLOCKS_PER_PROGRAM: tl.constexpr):
    """Variant: grid (chunks, rows), so programs launched together walk one row instead of striding rows."""
    row = tl.program_id(1)
    block0 = tl.program_id(0) * BLOCKS_PER_PROGRAM
    length = tl.load(seq_lens_ptr + row)
    if block0 * BLOCK_SIZE >= length:
        return
    cols = block0 * BLOCK_SIZE + tl.arange(0, BLOCKS_PER_PROGRAM * BLOCK_SIZE)
    vals = tl.load(logits_ptr + row.to(tl.int64) * logits_stride + cols, mask=(cols < length) & (cols < width),
                   other=float("-inf")).to(tl.float32)
    scores = tl.max(tl.reshape(vals, (BLOCKS_PER_PROGRAM, BLOCK_SIZE)), axis=1)
    blocks = block0 + tl.arange(0, BLOCKS_PER_PROGRAM)
    scores = tl.where(blocks == (length - 1) // BLOCK_SIZE, float("inf"), scores)
    tl.store(out_ptr + row.to(tl.int64) * out_stride + blocks, scores, mask=blocks < num_blocks)


def launch_rm(logits, lens, out, bpp, nw):
    rows, width = logits.shape
    nb = out.shape[1]
    _block_scores_rowmajor_kernel[(triton.cdiv(nb, bpp), rows)](
        logits, lens, out, logits.stride(0), width, nb, out.stride(0),
        BLOCK_SIZE=CBLK, BLOCKS_PER_PROGRAM=bpp, num_warps=nw,
    )


@triton.jit
def _block_scores_fast_kernel(logits_ptr, seq_lens_ptr, out_ptr, logits_stride, width, num_blocks, out_stride,
                              BLOCK_SIZE: tl.constexpr, BLOCKS_PER_PROGRAM: tl.constexpr):
    """Variant: chunks fully inside the reach load without a mask (vectorizable); only the tail chunk masks."""
    row = tl.program_id(0)
    block0 = tl.program_id(1) * BLOCKS_PER_PROGRAM
    length = tl.load(seq_lens_ptr + row)
    if block0 * BLOCK_SIZE >= length:
        return
    cols = block0 * BLOCK_SIZE + tl.arange(0, BLOCKS_PER_PROGRAM * BLOCK_SIZE)
    base = logits_ptr + row.to(tl.int64) * logits_stride
    end = (block0 + BLOCKS_PER_PROGRAM) * BLOCK_SIZE
    if end <= length and end <= width:
        vals = tl.load(base + cols).to(tl.float32)
    else:
        vals = tl.load(base + cols, mask=(cols < length) & (cols < width), other=float("-inf")).to(tl.float32)
    scores = tl.max(tl.reshape(vals, (BLOCKS_PER_PROGRAM, BLOCK_SIZE)), axis=1)
    blocks = block0 + tl.arange(0, BLOCKS_PER_PROGRAM)
    scores = tl.where(blocks == (length - 1) // BLOCK_SIZE, float("inf"), scores)
    tl.store(out_ptr + row.to(tl.int64) * out_stride + blocks, scores, mask=blocks < num_blocks)


def launch_fast(logits, lens, out, bpp, nw):
    rows, width = logits.shape
    nb = out.shape[1]
    _block_scores_fast_kernel[(rows, triton.cdiv(nb, bpp))](
        logits, lens, out, logits.stride(0), width, nb, out.stride(0),
        BLOCK_SIZE=CBLK, BLOCKS_PER_PROGRAM=bpp, num_warps=nw,
    )


def launch(logits, lens, out, bpp, nw):
    rows, width = logits.shape
    nb = out.shape[1]
    C._candidate_block_scores_kernel[(rows, triton.cdiv(nb, bpp))](
        logits, lens, out, logits.stride(0), width, nb, out.stride(0),
        BLOCK_SIZE=CBLK, BLOCKS_PER_PROGRAM=bpp, FILL_TAIL=False, num_warps=nw,
    )


def main():
    for dt in DTYPES:
        torch.manual_seed(0)
        lg = torch.randn(ROWS, LC, device=dev).to(torch.bfloat16 if dt == "bf16" else torch.float32)
        lens = (torch.arange(ROWS, device=dev, dtype=torch.int32) + (LC - ROWS + 1)).clamp_(min=1, max=LC)
        gb = lens.sum().item() * lg.element_size() / 1e9
        ref = C.candidate_block_scores(lg, lens, block_size=CBLK, fill_tail=False)
        blens = (lens + CBLK - 1) // CBLK
        ids = torch.empty(ROWS, CTOPK, device=dev, dtype=torch.int32)
        t_amax = gpu_ms(lambda: lg.amax(dim=1))
        t_bm = gpu_ms(lambda: C.candidate_block_scores(lg, lens, block_size=CBLK, fill_tail=False))
        t_tk = gpu_ms(lambda: C.topk_transform_paged_hip(ref, blens, None, ids, 1, None))
        t_pub = gpu_ms(lambda: C.select_candidate_blocks_hip(lg, lens, topk_blocks=CTOPK, block_size=CBLK))
        print(f"{dt} rows={ROWS} lc={LC} reach {gb:.2f} GB | amax {t_amax:.2f} | publish {t_pub:.2f} = "
              f"block-max {t_bm:.2f} ({gb / t_bm:.2f} TB/s) + top-k over blocks {t_tk:.2f} + rest")
        reach = torch.arange(ref.shape[1], device=dev)[None, :] < blens[:, None]
        best = None
        for kname, fn, bpp, nw in [("2d", launch, b, w) for b in (256, 1024) for w in (2, 4)] + [
            ("fast", launch_fast, b, w) for b in (64, 128, 256, 512) for w in (1, 2, 4)
        ]:
                out = torch.full_like(ref, float("nan"))
                try:
                    fn(lg, lens, out, bpp, nw)
                except Exception as ex:
                    print(f"  {kname} bpp={bpp} nw={nw}: {type(ex).__name__}: {str(ex)[:120]}")
                    continue
                ok = torch.equal(out[reach], ref[reach])
                t = gpu_ms(lambda: fn(lg, lens, out, bpp, nw))
                if ok and (best is None or t < best[0]):
                    best = (t, kname, bpp, nw)
                print(f"  {kname} bpp={bpp:4d} nw={nw}: {t:.3f} ms ({gb / t:.2f} TB/s) equal={ok}")
        print(f"  best: {best[1]} bpp={best[2]} nw={best[3]} {best[0]:.3f} ms")
        del lg, ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
