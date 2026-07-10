#!/usr/bin/env python3
"""
Minimal repro: triton `tl.dot(fp8, fp8)` returns garbage for contraction dim K >= 128 on gfx1250.

Observed on AMD gfx1250 (Instinct, RDNA/CDNA-class), ROCm nightly, triton (custom build shipped in
the sglang gfx1250 docker). A single-tile fp8_e4m3fn x fp8_e4m3fn matmul with small, in-range
values returns ~1e34+ (garbage) once the contraction dimension K >= 128. K <= 64 is correct, and
bf16 x bf16 is correct at all K. There is no error, no NaN at the dot itself — the wrong values
only turn into NaN downstream (e.g. via a softmax exp()).

Impact: any fp8 KV-cache attention that does an fp8xfp8 QK/PV dot with K >= 128 (e.g. DeepSeek MLA,
nope dim K=512) silently produces garbage -> NaN -> degenerate output.

Expected (correct) rel_l2 vs an fp64 recompute of the SAME fp8-rounded values: ~1e-7..1e-5.
Observed on gfx1250: ~0 for K=64, but ~1e34..1e36 for K in {128, 256, 512}.

Run:  python3 triton_fp8_dot_largek_gfx1250_repro.py
"""
import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fn
DEV = "cuda"


@triton.jit
def _tile_dot(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, UPCAST: tl.constexpr):
    # Single program, single tile. No K-loop, no masking (M,N,K are powers of two).
    om = tl.arange(0, M)
    on = tl.arange(0, N)
    ok = tl.arange(0, K)
    a = tl.load(A + om[:, None] * K + ok[None, :])   # [M, K]
    b = tl.load(B + ok[:, None] * N + on[None, :])   # [K, N]
    if UPCAST:                                        # control: cast fp8 -> bf16 before the dot
        a = a.to(tl.bfloat16)
        b = b.to(tl.bfloat16)
    c = tl.dot(a, b)                                  # fp32 accumulate
    tl.store(C + om[:, None] * N + on[None, :], c)


def rel_l2(x, ref):
    x, ref = x.double(), ref.double()
    return (torch.norm(x - ref) / (torch.norm(ref) + 1e-30)).item()


def run(M, N, K, upcast):
    torch.manual_seed(0)
    a = (torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.5)   # small, in fp8 range
    b = (torch.randn(K, N, device=DEV, dtype=torch.bfloat16) * 0.4)
    a8, b8 = a.to(FP8), b.to(FP8)
    ref = a8.double() @ b8.double()                                   # exact math on fp8 values
    c = torch.empty((M, N), dtype=torch.float32, device=DEV)
    _tile_dot[(1,)](a8, b8, c, M=M, N=N, K=K, UPCAST=upcast)
    return rel_l2(c, ref), bool((~torch.isfinite(c)).any().item()), c.abs().max().item()


if __name__ == "__main__":
    print("arch:", torch.cuda.get_device_properties(0).gcnArchName)
    print("triton:", triton.__version__)
    print()
    print("fp8_e4m3fn x fp8_e4m3fn, M=16 N=16, small in-range values (|a|<=~2):")
    print(f"{'K':>6} {'fp8 rel_l2':>14} {'fp8 |c|max':>14} {'bf16 rel_l2':>14}")
    for K in [64, 128, 256, 512]:
        r8, nan8, mx8 = run(16, 16, K, upcast=False)   # fp8 x fp8
        rb, nanb, mxb = run(16, 16, K, upcast=True)    # fp8->bf16 x bf16 (control)
        flag = "  <-- GARBAGE" if r8 > 1.0 else ""
        print(f"{K:>6} {r8:>14.3e} {mx8:>14.3e} {rb:>14.3e}{flag}")
    print()
    print("Correct output would be rel_l2 ~1e-7..1e-5 for all K (as it is for K=64 and for the")
    print("fp8->bf16 control). K>=128 fp8xfp8 returning ~1e34 is the bug.")
