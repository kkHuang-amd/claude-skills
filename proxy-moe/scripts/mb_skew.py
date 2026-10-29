# Cross-rank skew microbench: per iter = fixed compute kernel(s) + one all-reduce. GPU events per rank.
# torchrun --nproc-per-node N mb_skew.py --load gemm|stream|light [--iters 2000]
# Socket of a rank = physical GPU id // 2 (MI455 here: 2 partitions per socket); pass --gpus to describe CUDA_VISIBLE_DEVICES.
import argparse, os, statistics as st, torch, torch.distributed as dist
ap = argparse.ArgumentParser()
ap.add_argument("--load", default="gemm"); ap.add_argument("--iters", type=int, default=2000)
ap.add_argument("--warmup", type=int, default=300); ap.add_argument("--ar-bytes", type=int, default=2 << 20)
ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7"); ap.add_argument("--size", type=int, default=4096)
a = ap.parse_args()
dist.init_process_group("nccl"); r, W = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"])); dev = torch.device("cuda")
if a.load == "gemm":      # compute-bound, high power: bf16 4096^3 x2 (~0.3 TFLOP)
    A = torch.randn(a.size, a.size, device=dev, dtype=torch.bfloat16); B = torch.randn(a.size, a.size, device=dev, dtype=torch.bfloat16)
    work = lambda: (torch.mm(A, B), torch.mm(B, A))
elif a.load == "stream":  # memory-bound: read 2 GiB
    X = torch.randn(1 << 30, device=dev, dtype=torch.bfloat16)
    work = lambda: X.sum()
else:                     # light: tiny kernel, should stay far below PPT
    A = torch.randn(256, 256, device=dev, dtype=torch.bfloat16)
    work = lambda: [torch.mm(A, A) for _ in range(4)]
buf = torch.zeros(a.ar_bytes // 2, device=dev, dtype=torch.bfloat16)
ev = [[torch.cuda.Event(enable_timing=True) for _ in range(3)] for _ in range(a.iters)]
for i in range(a.warmup + a.iters):
    if i >= a.warmup: ev[i - a.warmup][0].record()
    work()
    if i >= a.warmup: ev[i - a.warmup][1].record()
    dist.all_reduce(buf)
    if i >= a.warmup: ev[i - a.warmup][2].record()
torch.cuda.synchronize()
comp = torch.tensor([e[0].elapsed_time(e[1]) * 1e3 for e in ev], device=dev)   # us
ar = torch.tensor([e[1].elapsed_time(e[2]) * 1e3 for e in ev], device=dev)
C = [torch.empty_like(comp) for _ in range(W)]; A_ = [torch.empty_like(ar) for _ in range(W)]
dist.all_gather(C, comp); dist.all_gather(A_, ar)
if r == 0:
    C = [c.tolist() for c in C]; A_ = [x.tolist() for x in A_]
    phys = [int(g) for g in a.gpus.split(",")][:W]; sock = [p // 2 for p in phys]
    n = a.iters
    q = lambda v: "/".join(f"{x:.0f}" for x in (st.quantiles(v, n=10)[0], st.median(v), st.quantiles(v, n=10)[-1]))
    print(f"load={a.load} ranks={W} gpus={phys} iters={n}")
    print("compute us p10/p50/p90 per rank:", " ".join(f"g{phys[i]}={q(C[i])}" for i in range(W)))
    same = [st.correlation(C[i], C[j]) for i in range(W) for j in range(i + 1, W) if sock[i] == sock[j]]
    cross = [st.correlation(C[i], C[j]) for i in range(W) for j in range(i + 1, W) if sock[i] != sock[j]]
    if same: print(f"corr same-socket: " + " ".join(f"{x:.2f}" for x in same))
    if cross: print(f"corr cross-socket: min {min(cross):.2f} median {st.median(cross):.2f} max {max(cross):.2f}")
    mm = st.mean(max(C[k][t] for k in range(W)) - st.mean(C[k][t] for k in range(W)) for t in range(n))
    arown = st.median(min(A_[k][t] for k in range(W)) for t in range(n)); armean = st.median(st.mean(A_[k][t] for k in range(W)) for t in range(n))
    print(f"compute max-mean per iter {mm:.1f} us ({100*mm/st.mean(st.mean(c) for c in C):.1f}% of compute) | AR own {arown:.1f} us, mean {armean:.1f} us, wait share {100*(1-arown/armean):.0f}%")
dist.destroy_process_group()
