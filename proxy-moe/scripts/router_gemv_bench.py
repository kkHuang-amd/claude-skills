"""Baseline for the Laguna fp32 router GEMV on one GPU: logits[M,512] = x[M,16384].float() @ W[512,16384]^T.

Rotates over NL distinct weights (one per layer, as in the server) so a weight is not cache-resident
from the previous call, and reports:
  - torch F.linear fp32 (what LagunaMoEGate runs today) per M, and the hipBLASLt kernel it picks
  - a Triton streaming-read kernel over the same weights (achievable-HBM ceiling for 32 MiB reads)
Times are per call (us) from CUDA-graph replay of NL calls. NOTE: idle clocks (~2.35 GHz), not the
power-capped ~1 GHz seen in serving at high concurrency; memory-bound numbers are less affected.

Usage: python3 router_gemv_bench.py [--ms 1 2 4 8 16 32 64 256] [--layers 12]
"""
import argparse, torch, torch.nn.functional as F, triton, triton.language as tl

ap = argparse.ArgumentParser()
ap.add_argument("--ms", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 256])
ap.add_argument("--layers", type=int, default=12)
ap.add_argument("--iters", type=int, default=20)
a = ap.parse_args()
E, H, NL = 512, 16384, a.layers
torch.manual_seed(0)
Ws = [torch.randn(E, H, device="cuda", dtype=torch.float32) * 0.01 for _ in range(NL)]
wbytes = E * H * 4

def graph_time(fn, calls):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    ts = []
    for _ in range(a.iters):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / calls)
    return sorted(ts)[len(ts) // 2]

@triton.jit
def _read(w_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(pid * BLOCK, n, tl.num_programs(0) * BLOCK):
        acc += tl.load(w_ptr + off + tl.arange(0, BLOCK), mask=off + tl.arange(0, BLOCK) < n)
    tl.store(out_ptr + pid, tl.sum(acc))

props = torch.cuda.get_device_properties(0)
print(f"{props.gcnArchName} CUs={props.multi_processor_count} weight={wbytes / 2**20:.0f} MiB x {NL} layers")

sink = torch.empty(4096, device="cuda")
for grid in (props.multi_processor_count * 4, props.multi_processor_count * 16):
    us = graph_time(lambda: [_read[(grid,)](W, sink, W.numel(), BLOCK=4096) for W in Ws], NL)
    print(f"stream-read 32 MiB, grid {grid}: {us:7.1f} us/call  {wbytes / us / 1e6:6.2f} TB/s")

for m in a.ms:
    x = torch.randn(m, H, device="cuda", dtype=torch.bfloat16)
    us = graph_time(lambda: [F.linear(x.to(torch.float32), W, None) for W in Ws], NL)
    print(f"F.linear fp32 M={m:4d}: {us:8.1f} us/call  eff {wbytes / us / 1e6:6.3f} TB/s")

from torch.profiler import profile, ProfilerActivity
x = torch.randn(2, H, device="cuda", dtype=torch.bfloat16)
with profile(activities=[ProfilerActivity.CUDA]) as p:
    F.linear(x.to(torch.float32), Ws[0], None); torch.cuda.synchronize()
print("kernels at M=2:", sorted({e.name[:110] for e in p.events() if e.device_type.name == "CUDA"}))
