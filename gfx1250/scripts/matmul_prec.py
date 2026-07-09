"""Robust (torch/hipblas) precision test on gfx1250: bf16 matmul vs fp32 matmul vs fp64 ref.
Mirrors the attention P·V dot: A = softmax rows (p, probabilities), B = gaussian (v).
Same VALUES fed as bf16 and fp32; compare each to an fp64 recompute of the SAME values.
"""
import torch

def rel_l2(x, ref):
    x = x.double(); ref = ref.double()
    return (torch.norm(x - ref) / (torch.norm(ref) + 1e-30)).item()

def test(M, N, K, tag):
    torch.manual_seed(0)
    A_f32 = torch.softmax(torch.randn(M, K, device="cuda"), dim=1)   # probs, rows sum 1
    B_f32 = torch.randn(K, N, device="cuda")
    A_bf16 = A_f32.to(torch.bfloat16)
    B_bf16 = B_f32.to(torch.bfloat16)

    C_bf16 = (A_bf16 @ B_bf16).float()      # gfx1250 bf16 matmul (hipblas, fp32 accum)
    C_f32 = (A_f32 @ B_f32)                 # gfx1250 fp32 matmul

    ref_bf16vals = A_bf16.double() @ B_bf16.double()  # same bf16 vals, fp64 accum
    ref_true = A_f32.double() @ B_f32.double()        # ground truth

    print(f"[{tag}] M={M} N={N} K={K}")
    print(f"  HARDWARE accum (same vals, vs fp64):  bf16 matmul={rel_l2(C_bf16, ref_bf16vals):.3e}   "
          f"fp32 matmul={rel_l2(C_f32, ref_true):.3e}")
    tb = rel_l2(C_bf16, ref_true); tf = rel_l2(C_f32, ref_true)
    print(f"  TOTAL vs true fp64 (incl input round): bf16={tb:.3e}   fp32={tf:.3e}   ratio bf16/fp32={tb/max(tf,1e-30):.1f}x")

if __name__ == "__main__":
    for (M, N, K) in [(128, 512, 256), (128, 512, 512), (128, 512, 2048)]:
        test(M, N, K, "attn-PV")
