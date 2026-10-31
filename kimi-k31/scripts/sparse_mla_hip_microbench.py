#!/usr/bin/env python3
"""R1: HIP sparse MLA prefill (sparse_mla_prefill_gfx950) vs Triton (SCALE_POST), 1 gfx950 GPU.
Same data as sparse_mla_prefill_microbench.py. Checks rel L2 vs an fp32 reference on 128
sampled rows, out/LSE agreement with Triton, -1 indices and a ragged topk.
Usage: sparse_mla_hip_microbench.py [seq=16384] [kind=clustered] [stages=2,3]"""
import os
import sys

import torch

os.environ.setdefault("SGLANG_ROCM_SPARSE_MLA_SCALE_POST", "1")
from sglang.kernels.ops.attention.dsa import triton_sparse_mla as tsm  # noqa: E402
from sglang.kernels.ops.attention.dsa.sparse_mla_prefill_gfx950 import sparse_mla_prefill_gfx950  # noqa: E402

argv = sys.argv[1:]
SEQ = int(argv[0]) if argv else 16384
KINDS = argv[1].split(",") if len(argv) > 1 else ["clustered"]
STAGES = [int(x) for x in argv[2].split(",")] if len(argv) > 2 else [2, 3]
H, DV, DT, TOPK, POOL = 12, 512, 64, 2048, 32768
ROW = DV + 4 * (DV // 128) + 2 * DT
torch.manual_seed(0)
dev = "cuda"

kv = torch.empty(POOL, 1, ROW, dtype=torch.uint8, device=dev)
nope8 = (torch.randn(POOL, 1, DV, device=dev) * 0.5).to(torch.float8_e4m3fn)
scales = torch.rand(POOL, 1, 4, device=dev) * (1 / 32) + 1 / 128  # distinct per row and group
kv[..., :DV] = nope8.view(torch.uint8)
kv[..., DV:DV + 16] = scales.view(torch.uint8)
rope = (torch.randn(POOL, 1, DT, device=dev) * 0.5).to(torch.bfloat16)
kv[..., DV + 16:] = rope.view(torch.uint8)
q_nope = torch.randn(SEQ, H, DV, device=dev, dtype=torch.bfloat16) * 0.1
q_rope = torch.randn(SEQ, H, DT, device=dev, dtype=torch.bfloat16) * 0.1
pos = torch.arange(POOL - SEQ, POOL, device=dev)
SM = (DV + DT) ** -0.5


def indices(kind, topk=TOPK):
    if kind == "random":
        return (torch.rand(SEQ, topk, device=dev) * (pos[:, None] + 1)).to(torch.int32)
    blk = 64
    shared = (torch.rand(SEQ // blk, topk, device=dev) * (pos[::blk, None] + 1)).to(torch.int32)
    idx = shared.repeat_interleave(blk, 0)
    own = (torch.rand(SEQ, topk, device=dev) * (pos[:, None] + 1)).to(torch.int32)
    return torch.where(torch.rand(SEQ, topk, device=dev) < 0.2, own, idx).contiguous()


def reference(idx, rows):
    v = nope8[:, 0].float() * scales[:, 0].repeat_interleave(128, -1)
    k = torch.cat([v, rope[:, 0].float()], -1)
    out, lse = [], []
    for r in rows.tolist():
        ids = idx[r]
        ok = ids >= 0
        kk = k[ids.clamp(min=0).long()]
        q = torch.cat([q_nope[r].float(), q_rope[r].float()], -1)
        s = (q @ kk.T * SM).masked_fill(~ok[None, :], float("-inf"))
        lse.append(torch.logsumexp(s, -1))
        out.append(torch.softmax(s, -1).nan_to_num(0.0) @ kk[:, :DV])
    return torch.stack(out), torch.stack(lse)


def timed(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(10):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / 10


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


flop = SEQ * H * TOPK * (DV + DT + DV) * 2
rows = torch.randperm(SEQ, device=dev)[:128]
for kind in KINDS:
    idx = indices(kind)
    idx[rows[:8], -300:] = -1  # padded rows
    idx[rows[8], :] = -1  # a row with no key
    ref_o, ref_l = reference(idx, rows)
    tri = lambda: tsm.triton_sparse_mla_fwd(q_nope, q_rope, kv, idx.unsqueeze(1), SM, DV, return_lse=True)
    hip = lambda: sparse_mla_prefill_gfx950(q_nope, q_rope, kv, idx.unsqueeze(1), SM, DV, return_lse=True, stages=STAGES[0])
    (to, tl_), (ho, hl) = tri(), hip()
    ok = torch.isfinite(ref_l)
    sel = torch.arange(len(rows), device=dev) != 8
    print(f"seq={SEQ} {kind}: relL2 vs fp32 triton {rel(to[0, rows][sel], ref_o[sel]):.1e}"
          f" hip {rel(ho[0, rows][sel], ref_o[sel]):.1e}"
          f" | hip vs triton out {rel(ho, to):.1e} lse maxabs {(hl[rows][ok] - ref_l[ok]).abs().max().item():.1e}"
          f" | empty row out {ho[0, rows[8]].abs().max().item():.1e} lse {hl[rows[8], 0].item()}", flush=True)
    cases = [("triton", tri)] + [(f"hip s{st}", lambda st=st: sparse_mla_prefill_gfx950(
        q_nope, q_rope, kv, idx.unsqueeze(1), SM, DV, return_lse=True, stages=st)) for st in STAGES]
    for name, fn in cases:
        ms = timed(fn)
        print(f"seq={SEQ} {kind:9s} {name:7s} {ms:7.3f} ms  {flop / ms / 1e9:6.1f} TFLOPS", flush=True)
    idx2 = indices(kind, 1000)  # topk not a multiple of 32
    o2 = sparse_mla_prefill_gfx950(q_nope, q_rope, kv, idx2.unsqueeze(1), SM, DV, stages=STAGES[-1])
    t2 = tsm.triton_sparse_mla_fwd(q_nope, q_rope, kv, idx2.unsqueeze(1), SM, DV)
    print(f"seq={SEQ} {kind}: topk=1000 hip vs triton out {rel(o2, t2):.1e}", flush=True)
