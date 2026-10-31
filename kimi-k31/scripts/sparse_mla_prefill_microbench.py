#!/usr/bin/env python3
"""Kimi-K3.1 DSA sparse MLA prefill on one gfx950 GPU (server must be down).
Per-rank shape: H=12 heads, D_V=512, D_TAIL=64, top-k 2048, fp8_ds_mla KV rows (656 B).
Index patterns: 'random' (uniform over the causal prefix, worst-case L2 reuse) and
'clustered' (each 64-query block shares 80% of its keys).
Usage: sparse_mla_prefill_microbench.py [seq=16384] [--extra-configs]"""
import sys

import torch
import triton

from sglang.kernels.ops.attention.dsa import triton_sparse_mla as tsm

SEQ = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 16384
H, DV, DT, TOPK, POOL = 12, 512, 64, 2048, 32768
ROW = DV + 4 * (DV // 128) + 2 * DT  # 656
torch.manual_seed(0)
dev = "cuda"

if "--extra-configs" in sys.argv:
    k = tsm._sparse_mla_fwd_split_dim_kernel
    for bn, w, ns in ((64, 8, 1), (64, 8, 2), (128, 4, 1), (128, 8, 1), (32, 8, 2)):
        k.configs.append(triton.Config({"BLOCK_N": bn}, num_warps=w, num_stages=ns))

# fp8_ds_mla rows: fp8 NoPE | 4 fp32 scales | bf16 RoPE tail
kv = torch.empty(POOL, 1, ROW, dtype=torch.uint8, device=dev)
kv[..., :DV] = (torch.randn(POOL, 1, DV, device=dev) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
kv[..., DV:DV + 16] = torch.full((POOL, 1, 4), 1 / 64, device=dev, dtype=torch.float32).view(torch.uint8)
kv[..., DV + 16:] = (torch.randn(POOL, 1, DT, device=dev) * 0.5).to(torch.bfloat16).view(torch.uint8)
q_nope = torch.randn(SEQ, H, DV, device=dev, dtype=torch.bfloat16) * 0.1
q_rope = torch.randn(SEQ, H, DT, device=dev, dtype=torch.bfloat16) * 0.1
# queries sit at the end of a POOL-long context: every row has a full top-k
pos = torch.arange(POOL - SEQ, POOL, device=dev)


def indices(kind):
    if kind == "random":
        r = torch.rand(SEQ, TOPK, device=dev)
        return (r * (pos[:, None] + 1)).to(torch.int32)
    blk = 64
    shared = (torch.rand(SEQ // blk, TOPK, device=dev) * (pos[::blk, None] + 1)).to(torch.int32)
    idx = shared.repeat_interleave(blk, 0)
    own = (torch.rand(SEQ, TOPK, device=dev) * (pos[:, None] + 1)).to(torch.int32)
    mask = torch.rand(SEQ, TOPK, device=dev) < 0.2
    return torch.where(mask, own, idx).contiguous()


flop = SEQ * H * TOPK * (DV + DT + DV) * 2
gb = SEQ * TOPK * ROW / 1e9
for kind in ("random", "clustered"):
    idx = indices(kind).unsqueeze(1)
    args = dict(q_nope=q_nope, q_rope=q_rope, kv=kv, indices=idx, sm_scale=(DV + DT) ** -0.5, d_v=DV)
    for _ in range(3):
        tsm.triton_sparse_mla_fwd(**args)
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(10):
        tsm.triton_sparse_mla_fwd(**args)
    en.record(); torch.cuda.synchronize()
    ms = st.elapsed_time(en) / 10
    cfg = getattr(tsm._sparse_mla_fwd_split_dim_kernel, "best_config", None)
    print(f"seq={SEQ} {kind:9s} {ms:7.3f} ms  {flop / ms / 1e9:6.1f} TFLOPS(12 heads)  gathered {gb / ms:6.2f} TB/s  best={cfg}", flush=True)
