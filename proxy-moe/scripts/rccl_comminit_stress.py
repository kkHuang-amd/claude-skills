"""Stress the DDA fabric-init race: fresh communicator + two small all-reduces, many times.

Each iteration creates a new NCCL group over all ranks (lazy full init inside the first
all-reduce; with device_id torch would ncclCommSplit, which skips the DDA fabric init), then runs two bf16
all-reduces of 98304 elements (192 KB, the warmup-prefill size that takes DDA flat LL) on a
side stream, so ProcessGroupNCCL does not order them behind the null stream where RCCL
zeroes its DDA scratch/epoch. --busy-ms puts a GEMM on the null stream right before init to
delay that memset and widen the window. A hang shows up as the outer `timeout` firing; each
rank logs its last iteration to OUT/rank<r>.txt.

torchrun --nproc-per-node 4 rccl_comminit_stress.py --iters 150 --out /tmp/stress
(run with NCCL_CUMEM_ENABLE=1; RCCL_DDA_ENABLE=0 is the negative control)
"""
import argparse
import datetime
import os
import random
import time

import torch
import torch.distributed as dist

p = argparse.ArgumentParser()
p.add_argument("--iters", type=int, default=150)
p.add_argument("--count", type=int, default=98304)
p.add_argument("--busy-ms", type=float, default=0.0, help="null-stream GEMM before each init")
p.add_argument("--jitter-ms", type=float, default=0.0, help="random per-rank host delay before init")
p.add_argument("--out", default="/tmp/rccl_comminit_stress")
a = p.parse_args()

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(dev)
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=600))
os.makedirs(a.out, exist_ok=True)
log = open(f"{a.out}/rank{rank}.txt", "w", buffering=1)

side = torch.cuda.Stream()
x = torch.ones(a.count, dtype=torch.bfloat16, device=dev)
busy = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16) if a.busy_ms > 0 else None
rng = random.Random(rank * 7919)
t0 = time.time()

for it in range(a.iters):
    if busy is not None:  # keep the null stream busy so RCCL's init memset lands late
        n = max(1, int(a.busy_ms / 2))
        for _ in range(n):
            busy @ busy
    if a.jitter_ms > 0:
        time.sleep(rng.random() * a.jitter_ms / 1000)
    g = dist.new_group(list(range(world)))  # no device_id: a full init, not ncclCommSplit
    log.write(f"iter {it} init done\n")
    with torch.cuda.stream(side):
        y = x.clone()
        for _ in range(2):
            dist.all_reduce(y, group=g)
    side.synchronize()
    ok = bool(torch.all(y == float(world ** 2)).item())  # 1 -> world -> world^2
    log.write(f"iter {it} ok={ok}\n")
    if not ok:
        print(f"rank{rank} iter {it}: WRONG RESULT {y[:4].tolist()}", flush=True)
    torch.cuda.synchronize()
    dist.destroy_process_group(g)

if rank == 0:
    print(f"PASS {a.iters} iters, world {world}, {time.time() - t0:.0f}s", flush=True)
dist.destroy_process_group()
