"""Attainable HBM read bandwidth on one GPU (partition): persistent Triton streaming read, swept.

Usage: HIP_VISIBLE_DEVICES=<g> python3 hbm_read_ceiling.py [--gib 4] [--top 10]
Each program walks the buffer with a grid stride, xor-accumulates int32 loads (keeps them alive)
and writes one int per program. Reports TB/s = bytes read / median time (CUDA events, 20 iters).
Buffer >> 256 MB Infinity Cache, so the number is HBM, not cache.
"""
import argparse, itertools, statistics as st
import torch, triton, triton.language as tl


@triton.jit
def read_kernel(x_ptr, out_ptr, n_blocks, BLOCK: tl.constexpr, CACHE: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.int32)
    for b in range(pid, n_blocks, nprog):
        acc ^= tl.load(x_ptr + b.to(tl.int64) * BLOCK + offs, cache_modifier=CACHE)
    tl.store(out_ptr + pid, tl.reduce(acc, 0, _xor))


@triton.jit
def _xor(a, b):
    return a ^ b


def bench(fn, it=20):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(it):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    return st.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gib", type=float, default=4)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    nbytes = int(a.gib * (1 << 30))
    x = torch.randint(0, 1 << 30, (nbytes // 4,), dtype=torch.int32, device="cuda")
    cu = torch.cuda.get_device_properties(0).multi_processor_count
    out = torch.empty(cu * 64, dtype=torch.int32, device="cuda")
    res = []
    for mult, block, warps, cache in itertools.product(
        (1, 2, 4, 8, 16), (1024, 2048, 4096, 8192), (4, 8, 16), ("", ".cg")
    ):
        grid = cu * mult
        n_blocks = x.numel() // block
        try:
            t = bench(lambda: read_kernel[(grid,)](x, out, n_blocks, BLOCK=block, CACHE=cache, num_warps=warps))
        except Exception as ex:  # noqa: BLE001
            print(f"skip grid={grid} block={block} warps={warps} cache={cache!r}: {type(ex).__name__}")
            continue
        res.append((n_blocks * block * 4 / t / 1e12, grid, block, warps, cache))
    res.sort(reverse=True)
    print(f"CUs={cu} buffer={a.gib} GiB configs={len(res)}")
    for bw, grid, block, warps, cache in res[: a.top]:
        print(f"{bw:6.2f} TB/s  grid={grid} ({grid // cu}/CU) block={block} int32 warps={warps} cache={cache or 'default'}")
    y = torch.empty(nbytes // 4 // 4, dtype=torch.int32, device="cuda"); z = torch.empty_like(y)
    t = bench(lambda: z.copy_(y))
    print(f"ref torch copy 1/4 buffer: {2 * y.numel() * 4 / t / 1e12:.2f} TB/s (r+w)")


if __name__ == "__main__":
    main()
