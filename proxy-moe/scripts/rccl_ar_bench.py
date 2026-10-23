"""RCCL all-reduce latency at decode sizes, CUDA-graph replay (baseline for custom AR).

Usage: torchrun --nproc-per-node 8 rccl_ar_bench.py [--rows 2 16 64] [--hidden 16384]
Prints, on rank 0, the per-call median over ranks of the graph-replayed all-reduce in us.
"""
import argparse, os, torch, torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, nargs="+", default=[2, 16, 64])
ap.add_argument("--hidden", type=int, default=16384)
ap.add_argument("--calls", type=int, default=50)
ap.add_argument("--iters", type=int, default=20)
a = ap.parse_args()

rank = int(os.environ["RANK"]); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")
for m in a.rows:
    x = torch.randn(m, a.hidden, device="cuda", dtype=torch.bfloat16)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            dist.all_reduce(x)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(a.calls):
            dist.all_reduce(x)
    times = []
    for _ in range(a.iters):
        dist.barrier(); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1) * 1e3 / a.calls)
    t = torch.tensor(sorted(times)[len(times) // 2], device="cuda")
    allt = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(allt, t)
    if rank == 0:
        v = sorted(float(z) for z in allt)
        print(f"RCCL all_reduce bf16 [{m},{a.hidden}] ({m * a.hidden * 2 / 1024:.0f} KB): "
              f"median-over-ranks {v[len(v) // 2]:.1f} us (min {v[0]:.1f}, max {v[-1]:.1f})")
dist.destroy_process_group()
