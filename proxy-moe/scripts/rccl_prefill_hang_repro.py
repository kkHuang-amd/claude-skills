"""Try to reproduce the server's warmup-prefill hang outside sglang.

Replays, in eager mode (prefill is not graph-captured), the exact collective sequence logged on
every rank before the hang: 25x AllReduce bf16 count 98304 (6 tokens x 16384) then 1x AllGather
bf16 count 25256 per rank (vocab shard of 1 token's logits), on one communicator, with a small
compute kernel between collectives like the model's layers. Repeats ROUNDS times; a round that
does not finish within TIMEOUT seconds counts as a hang.

Usage: NCCL_CUMEM_ENABLE=1 torchrun --nproc-per-node 8 rccl_prefill_hang_repro.py [--rounds 200]
"""
import argparse, os, time, torch, torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=200)
ap.add_argument("--timeout", type=float, default=60)
ap.add_argument("--no-compute", action="store_true")
a = ap.parse_args()

rank = int(os.environ["RANK"]); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")
x = torch.randn(6, 16384, device="cuda", dtype=torch.bfloat16)
w = torch.randn(16384, 2048, device="cuda", dtype=torch.bfloat16)
ag_in = torch.randn(25256, device="cuda", dtype=torch.bfloat16)
ag_out = torch.empty(25256 * dist.get_world_size(), device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()

t0 = time.time()
for r in range(a.rounds):
    for _ in range(25):
        if not a.no_compute:
            y = x @ w            # stand-in for a layer's GEMMs between all-reduces
        dist.all_reduce(x)
    dist.all_gather_into_tensor(ag_out, ag_in)
    ev = torch.cuda.Event(); ev.record()
    start = time.time()
    while not ev.query():
        if time.time() - start > a.timeout:
            print(f"rank{rank}: HANG in round {r} (> {a.timeout}s)", flush=True)
            os._exit(3)
        time.sleep(0.001)
    x.mul_(1.0 / dist.get_world_size())  # keep values bounded
if rank == 0:
    print(f"no hang in {a.rounds} rounds ({time.time() - t0:.1f}s)", flush=True)
dist.destroy_process_group()
