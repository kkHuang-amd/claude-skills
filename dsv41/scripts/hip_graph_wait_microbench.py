#!/usr/bin/env python3
"""Isolate the HIP 'Event.wait() regresses TPOT' effect (upstream #26672) seen in DSpark overlap scheduling.

Captures a CUDA/HIP graph of N small kernels (~decode-step shaped: many tiny launches) and measures per-replay
time for back-to-back replays under four enqueue patterns:
  sync      : host synchronize before each replay (queue empty at launch; what HIP does today)
  queued    : replays enqueued back-to-back on one stream (no host sync, no event)
  evwait    : like queued, but a side stream records an event and the replay stream waits on it each iteration
              (mirrors schedule_stream -> forward_stream wait_stream / publish_ready.wait())
  evwait_ms : evwait plus a small side-stream kernel before the record (event pending at wait time)
Usage: HIP_VISIBLE_DEVICES=6 python3 hip_graph_wait_microbench.py [--kernels 1200] [--iters 50]
"""
import argparse, time, torch

ap = argparse.ArgumentParser(); ap.add_argument("--kernels", type=int, default=1200); ap.add_argument("--iters", type=int, default=50)
a = ap.parse_args()
dev = torch.device("cuda")
x = torch.zeros(4096, device=dev)
main, side = torch.cuda.Stream(), torch.cuda.Stream()

g = torch.cuda.CUDAGraph()
with torch.cuda.stream(main):
    for _ in range(3): x.add_(1.0)
    torch.cuda.synchronize()
    with torch.cuda.graph(g, stream=main):
        for _ in range(a.kernels): x.add_(1.0)
torch.cuda.synchronize()

def run(mode):
    ev = torch.cuda.Event()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.iters):
        if mode == "sync":
            torch.cuda.synchronize()
        if mode.startswith("evwait"):
            with torch.cuda.stream(side):
                if mode == "evwait_ms": x.mul_(1.0)
                ev.record(side)
            main.wait_event(ev)
        with torch.cuda.stream(main):
            g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / a.iters * 1e3

for m in ["sync", "queued", "evwait", "evwait_ms"] * 2:
    print(f"{m:10s} {run(m):7.3f} ms/replay ({a.kernels} kernels)")
