#!/usr/bin/env python3
"""Host wake-up latency of Event.synchronize() vs Stream/device sync on ROCm (run on an idle GPU).

Per trial: GPU busy kernel (~BUSY_MS) -> record e_pub -> host e_pub.synchronize() (or other mode) -> immediately
record e_after + tiny kernel. GPU-side gap = e_pub.elapsed_time(e_after) = host wake-up + launch latency.
Also runs the same with a spin-wait (while not e.query()) for comparison.
Usage: HIP_VISIBLE_DEVICES=6 python3 hip_event_sync_latency.py [--trials 200] [--busy-ms 5]
"""
import argparse, statistics, torch

ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=200); ap.add_argument("--busy-ms", type=float, default=5.0)
a = ap.parse_args()
x = torch.randn(4096, 4096, device="cuda"); y = torch.empty_like(x)
torch.cuda.synchronize()
# calibrate matmul count for BUSY_MS
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record(); [torch.mm(x, x, out=y) for _ in range(10)]; e.record(); torch.cuda.synchronize()
n = max(1, int(a.busy_ms / (s.elapsed_time(e) / 10))); print(f"mm {s.elapsed_time(e) / 10:.3f} ms")
for mode in ["event_sync", "spin_query", "stream_sync"]:
    gaps = []
    for _ in range(a.trials):
        pub = torch.cuda.Event(enable_timing=True); aft = torch.cuda.Event(enable_timing=True)
        for _ in range(n):
            torch.mm(x, x, out=y)
        pub.record()
        if mode == "event_sync":
            pub.synchronize()
        elif mode == "spin_query":
            while not pub.query():
                pass
        else:
            torch.cuda.current_stream().synchronize()
        aft.record(); y.add_(1.0)
        torch.cuda.synchronize()
        gaps.append(pub.elapsed_time(aft) * 1000)
    gaps.sort()
    print(f"{mode:12s} gap us: median {statistics.median(gaps):7.1f}  p10 {gaps[len(gaps)//10]:7.1f}  p90 {gaps[9*len(gaps)//10]:7.1f}  (busy {n} mm)")
