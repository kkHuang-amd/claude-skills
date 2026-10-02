# Microbench: DSV4 compressor wkv_gate GEMM, pre-41019 (aiter tgemm) vs post-41019 (torch.mm out_dtype=fp32).
# K=7168; N=2048 (attn, ratio 4), 1024 (attn, ratio 128), 512 (indexer, ratio 4).
import torch
from aiter.tuned_gemm import tgemm
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16  # option C (Alan's branch, gfx1250-only today)

def bench(fn, it=200, graph=False):
    """us per call. graph=True replays a captured CUDA graph of `it` calls, which
    removes the per-call CPU dispatch cost (decode runs under CUDA graphs)."""
    for _ in range(20): fn()
    torch.cuda.synchronize()
    if graph:
        g = torch.cuda.CUDAGraph(); st = torch.cuda.Stream()
        with torch.cuda.stream(st):  # aiter allocates its scratch per stream: warm it outside capture
            fn()
        torch.cuda.synchronize()
        with torch.cuda.stream(st), torch.cuda.graph(g, stream=st):
            for _ in range(it): fn()
        run = g.replay
    else:
        run = lambda: [fn() for _ in range(it)]
    run(); torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record(); run(); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1e3

K = 7168
print(f"{'M':>6} {'N':>5} {'A_tgemm':>7} {'now_mm':>7} {'C_a16':>7} {'C/A':>5} {'errA':>7} {'errC':>7}  (us, CUDA graph; err = max|.-fp64 ref|)")
for N in (2048, 1024, 512):
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.02
    for M in (1, 7, 8, 16, 28, 32, 64, 112, 128, 1024, 16384):  # 7/28/112 = c1/c4/c16 verify M
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        a = lambda: tgemm.mm(x, w, otype=x.dtype).float()
        b = lambda: torch.mm(x, w.t(), out_dtype=torch.float32)
        c = lambda: gemm_a16w16(x, w, dtype=torch.float32)  # Triton a16w16, fp32 accumulate + output
        ga, gb, gc = (bench(f, graph=True) for f in (a, b, c))
        ref = x.double() @ w.double().t()
        ea, ec = ((f().double() - ref).abs().max().item() for f in (a, c))
        print(f"{M:>6} {N:>5} {ga:7.1f} {gb:7.1f} {gc:7.1f} {gc/ga:5.2f} {ea:7.4f} {ec:7.4f}")
