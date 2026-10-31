#!/usr/bin/env python3
"""C3: split-K sparse MLA (_triton_sparse_mla_fwd_splitk) on long prefill, 1 gfx950 GPU.
Same shapes/data as sparse_mla_prefill_microbench.py; compares the default dispatch
(single-pass split-dim kernel) against kv_splits=1 (fused kernel) and 2..32 (split + reduce).
Accuracy: rel L2 vs an fp32 reference on 128 sampled rows.
Usage: sparse_mla_splitk_microbench.py [seq=16384] [splits=1,2,4,8,16,32] [kind=clustered]
                                       [paths=single,splitk]"""
import sys

import torch

from sglang.kernels.ops.attention.dsa import triton_sparse_mla as tsm

argv = [a for a in sys.argv[1:]]
SEQ = int(argv[0]) if len(argv) > 0 else 16384
SPLITS = [int(s) for s in argv[1].split(",")] if len(argv) > 1 else [1, 2, 4, 8, 16, 32]
KINDS = argv[2].split(",") if len(argv) > 2 else ["clustered"]
# splitk: _triton_sparse_mla_fwd_splitk; single: split-dim kernel with KV_SPLITS (b/f = bf16/fp32 partials)
PATHS = argv[3].split(",") if len(argv) > 3 else ["single", "splitk"]
H, DV, DT, TOPK, POOL = 12, 512, 64, 2048, 32768
ROW = DV + 4 * (DV // 128) + 2 * DT  # 656
torch.manual_seed(0)
dev = "cuda"

kv = torch.empty(POOL, 1, ROW, dtype=torch.uint8, device=dev)
nope8 = (torch.randn(POOL, 1, DV, device=dev) * 0.5).to(torch.float8_e4m3fn)
kv[..., :DV] = nope8.view(torch.uint8)
kv[..., DV:DV + 16] = torch.full((POOL, 1, 4), 1 / 64, device=dev, dtype=torch.float32).view(torch.uint8)
rope = (torch.randn(POOL, 1, DT, device=dev) * 0.5).to(torch.bfloat16)
kv[..., DV + 16:] = rope.view(torch.uint8)
q_nope = torch.randn(SEQ, H, DV, device=dev, dtype=torch.bfloat16) * 0.1
q_rope = torch.randn(SEQ, H, DT, device=dev, dtype=torch.bfloat16) * 0.1
pos = torch.arange(POOL - SEQ, POOL, device=dev)
SM = (DV + DT) ** -0.5


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


def reference(idx, rows):
    v = nope8[:, 0].float() / 64  # scales are 1/64 for every group
    k = torch.cat([v, rope[:, 0].float()], -1)
    out = []
    for r in rows.tolist():
        kk = k[idx[r, 0].long()]  # [TOPK, 576]
        q = torch.cat([q_nope[r].float(), q_rope[r].float()], -1)  # [H, 576]
        p = torch.softmax(q @ kk.T * SM, -1)
        out.append(p @ kk[:, :DV])
    return torch.stack(out)


def timed(fn):
    for _ in range(3):
        o = fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(10):
        fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / 10, o


flop = SEQ * H * TOPK * (DV + DT + DV) * 2
rows = torch.randperm(SEQ, device=dev)[:128]
for kind in KINDS:
    idx = indices(kind).unsqueeze(1)
    ref = reference(idx, rows)
    cases = [("default", lambda: tsm.triton_sparse_mla_fwd(q_nope, q_rope, kv, idx, SM, DV))]
    for s in SPLITS:
        if "splitk" in PATHS:
            cases.append((f"splitk={s}", lambda s=s: tsm._triton_sparse_mla_fwd_splitk(
                q_nope, q_rope, kv, idx, SM, DV, kv_splits=s)))
        if "single" in PATHS and s > 1:
            for pd in (torch.bfloat16, torch.float32):
                cases.append((f"single={s}{'f' if pd == torch.float32 else 'b'}",
                              lambda s=s, pd=pd: tsm._triton_sparse_mla_fwd_single(
                                  q_nope, q_rope, kv, idx, SM, DV, kv_splits=s, partial_dtype=pd)))
    for name, fn in cases:
        try:
            ms, o = timed(fn)
        except Exception as e:  # noqa: BLE001
            print(f"seq={SEQ} {kind:9s} {name:11s} FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)
            continue
        got = o[0, rows].float()
        err = ((got - ref).norm() / ref.norm()).item()
        print(f"seq={SEQ} {kind:9s} {name:11s} {ms:7.3f} ms  {flop / ms / 1e9:6.1f} TFLOPS  relL2 {err:.1e}", flush=True)
