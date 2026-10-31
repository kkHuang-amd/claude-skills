#!/usr/bin/env python3
"""P0.2 of MONO_KERNEL.md: in-kernel hand-off latency and TP8 collectives over IPC peer memory.

(a) one GPU: ping-pong between two CTAs (same XCD: blocks 0 / 8, cross XCD: 0 / 1),
    uncached buffer + system-scope atomics vs hipMalloc buffer + agent-scope atomics
(b) two GPUs: ping-pong rank 0 <-> rank 1 over peer memory
(c) two-shot all-reduce (RS + AG) and (d) all-to-all inside one launch, bf16, two payload formats:
    tagged = 64-bit words (bf16 pair + 32-bit tag, polled per word),
    flag   = raw 128-bit stores + __threadfence_system + one flag per (block, source)
    vs torch.distributed (RCCL) all_reduce / all_to_all_single at the same sizes.
Run with no server up:  torchrun --nproc-per-node 8 p2p_microbench.py
"""
import os

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline

DECL = r"""
int64_t alloc_buf(int64_t nbytes, int64_t uncached);
torch::Tensor ipc_handle(int64_t ptr);
int64_t ipc_open(torch::Tensor h);
void pingpong(int64_t buf, int64_t a, int64_t b, int64_t iters, int64_t out, int64_t base, int64_t sys);
void xpingpong(int64_t local, int64_t remote, int64_t role, int64_t iters, int64_t out, int64_t base);
void coll(int64_t x, int64_t y, torch::Tensor peers, int64_t rank, int64_t n, int64_t tag, int64_t err,
          int64_t fmt, int64_t mode, int64_t blocks);
"""

SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

#define TICK_LIMIT 200000000ull  // 2 s of the 100 MHz wall clock
#define HW_REG_XCC_ID 20

struct Peers { uint64_t p[8]; };

static const uint64_t RS0 = 0, RS1 = 32ull << 20, AG = 64ull << 20, FL = 127ull << 20;

__device__ __forceinline__ uint32_t xcc_id() {
  return __builtin_amdgcn_s_getreg(HW_REG_XCC_ID | (0 << 6) | ((4 - 1) << 11));
}
__device__ __forceinline__ uint32_t ld32s(const uint32_t* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void st32s(uint32_t* p, uint32_t v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ uint32_t ld32a(const uint32_t* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void st32a(uint32_t* p, uint32_t v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ uint64_t ld64s(const uint64_t* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void st64s(uint64_t* p, uint64_t v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ float bflo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bfhi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  uint32_t ua = __float_as_uint(a), ub = __float_as_uint(b);
  ua += 0x7fffu + ((ua >> 16) & 1u);
  ub += 0x7fffu + ((ub >> 16) & 1u);
  return (ua >> 16) | (ub & 0xffff0000u);
}
__device__ __forceinline__ bool late(uint64_t t0, int* err) {
  if (wall_clock64() - t0 > TICK_LIMIT) { atomicExch(err, 1); return true; }
  return false;
}

// (a) ping-pong between CTAs a and b; out = [ticks, xcc(a), xcc(b), timeout]
__global__ void k_pingpong(uint32_t* buf, int a, int b, int iters, long long* out, uint32_t base, int sys) {
  if (threadIdx.x != 0) return;
  const int me = blockIdx.x;
  if (me != a && me != b) return;
  uint32_t* ping = buf;
  uint32_t* pong = buf + 64;
  const uint64_t t0 = wall_clock64();
  for (int i = 1; i <= iters; ++i) {
    const uint32_t v = base + i;
    if (me == a) {
      sys ? st32s(ping, v) : st32a(ping, v);
      int s = 0;
      while ((sys ? ld32s(pong) : ld32a(pong)) != v)
        if (((++s) & 1023) == 0 && wall_clock64() - t0 > TICK_LIMIT) { out[3] = 1; return; }
    } else {
      int s = 0;
      while ((sys ? ld32s(ping) : ld32a(ping)) != v)
        if (((++s) & 1023) == 0 && wall_clock64() - t0 > TICK_LIMIT) { out[3] = 1; return; }
      sys ? st32s(pong, v) : st32a(pong, v);
    }
  }
  if (me == a) { out[0] = (long long)(wall_clock64() - t0); out[1] = xcc_id(); }
  else out[2] = xcc_id();
}

// (b) cross-GPU ping-pong: role 0 writes the peer's ping and polls its own pong; role 1 the reverse
__global__ void k_xpingpong(uint32_t* local, uint32_t* remote, int role, int iters, long long* out, uint32_t base) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const uint64_t t0 = wall_clock64();
  for (int i = 1; i <= iters; ++i) {
    const uint32_t v = base + i;
    int s = 0;
    if (role == 0) {
      st32s(remote, v);
      while (ld32s(local + 64) != v)
        if (((++s) & 1023) == 0 && wall_clock64() - t0 > TICK_LIMIT) { out[3] = 1; return; }
    } else {
      while (ld32s(local) != v)
        if (((++s) & 1023) == 0 && wall_clock64() - t0 > TICK_LIMIT) { out[3] = 1; return; }
      st32s(remote + 64, v);
    }
  }
  out[0] = (long long)(wall_clock64() - t0);
}

// (c)/(d) tagged format: S2 bf16 pairs a slice; mode 0 = two-shot AR, 1 = all-to-all
__global__ void k_tagged(const uint32_t* __restrict__ x, uint32_t* __restrict__ y, Peers P, int rank, int S2,
                         uint32_t tag, int* err, int mode) {
  const int T = gridDim.x * blockDim.x, gid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t rs = (tag & 1u) ? RS1 : RS0;
  uint64_t* my_rs = (uint64_t*)(P.p[rank] + rs);
  uint64_t* my_ag = (uint64_t*)(P.p[rank] + AG);
  const uint64_t hi = (uint64_t)tag << 32;
  const uint64_t t0 = wall_clock64();
  for (int idx = gid; idx < 8 * S2; idx += T) {
    const int j = idx / S2, w = idx - j * S2;
    st64s((uint64_t*)(P.p[j] + rs) + (size_t)rank * S2 + w, hi | x[(size_t)j * S2 + w]);
  }
  if (mode == 1) {
    for (int idx = gid; idx < 8 * S2; idx += T) {
      uint64_t v; int s = 0;
      while (((v = ld64s(my_rs + idx)) >> 32) != tag)
        if (((++s) & 1023) == 0 && late(t0, err)) return;
      y[idx] = (uint32_t)v;
    }
    return;
  }
  for (int w = gid; w < S2; w += T) {
    float a = 0.f, b = 0.f;
    for (int src = 0; src < 8; ++src) {
      uint64_t v; int s = 0;
      while (((v = ld64s(my_rs + (size_t)src * S2 + w)) >> 32) != tag)
        if (((++s) & 1023) == 0 && late(t0, err)) return;
      a += bflo((uint32_t)v); b += bfhi((uint32_t)v);
    }
    const uint64_t o = hi | pack2(a, b);
    for (int j = 0; j < 8; ++j) st64s((uint64_t*)(P.p[j] + AG) + (size_t)rank * S2 + w, o);
  }
  for (int idx = gid; idx < 8 * S2; idx += T) {
    uint64_t v; int s = 0;
    while (((v = ld64s(my_ag + idx)) >> 32) != tag)
      if (((++s) & 1023) == 0 && late(t0, err)) return;
    y[idx] = (uint32_t)v;
  }
}

// flag format: S4 uint4 (8 bf16) a slice; block b owns chunk b of every slice; flags [phase][src][block]
__device__ __forceinline__ void publish(int fast) {
  if (fast) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");  // uncached stores acked, no L2 writeback
  else __threadfence_system();
}
__device__ __forceinline__ void acquire(int fast) {
  if (!fast) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
}
__global__ void k_flag(const uint4* __restrict__ x, uint4* __restrict__ y, Peers P, int rank, int S4,
                       uint32_t tag, int* err, int mode, int fast) {
  const int G = gridDim.x, b = blockIdx.x, t = threadIdx.x, NT = blockDim.x;
  const int C = (S4 + G - 1) / G, lo = b * C, hi = min(lo + C, S4);
  const uint64_t rs = (tag & 1u) ? RS1 : RS0;
  const uint4* my_rs = (const uint4*)(P.p[rank] + rs);
  const uint4* my_ag = (const uint4*)(P.p[rank] + AG);
  uint32_t* my_fl = (uint32_t*)(P.p[rank] + FL);
  const uint64_t t0 = wall_clock64();
  for (int j = 0; j < 8; ++j) {
    uint4* dst = (uint4*)(P.p[j] + rs) + (size_t)rank * S4;
    const uint4* src = x + (size_t)j * S4;
    for (int i = lo + t; i < hi; i += NT) dst[i] = src[i];
  }
  publish(fast);
  __syncthreads();
  if (t < 8) st32s((uint32_t*)(P.p[t] + FL) + (0 * 8 + rank) * G + b, tag);
  if (t < 8) {
    int s = 0;
    while ((int)(ld32s(my_fl + (0 * 8 + t) * G + b) - tag) < 0)
      if (((++s) & 1023) == 0 && late(t0, err)) break;
  }
  __syncthreads();
  acquire(fast);
  if (mode == 1) {
    for (int src = 0; src < 8; ++src)
      for (int i = lo + t; i < hi; i += NT) y[(size_t)src * S4 + i] = my_rs[(size_t)src * S4 + i];
    return;
  }
  for (int i = lo + t; i < hi; i += NT) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int src = 0; src < 8; ++src) {
      const uint4 v = my_rs[(size_t)src * S4 + i];
      const uint32_t w[4] = {v.x, v.y, v.z, v.w};
      for (int k = 0; k < 4; ++k) { acc[2 * k] += bflo(w[k]); acc[2 * k + 1] += bfhi(w[k]); }
    }
    const uint4 o = make_uint4(pack2(acc[0], acc[1]), pack2(acc[2], acc[3]), pack2(acc[4], acc[5]),
                               pack2(acc[6], acc[7]));
    for (int j = 0; j < 8; ++j) ((uint4*)(P.p[j] + AG))[(size_t)rank * S4 + i] = o;
  }
  publish(fast);
  __syncthreads();
  if (t < 8) st32s((uint32_t*)(P.p[t] + FL) + (1 * 8 + rank) * G + b, tag);
  if (t < 8) {
    int s = 0;
    while ((int)(ld32s(my_fl + (1 * 8 + t) * G + b) - tag) < 0)
      if (((++s) & 1023) == 0 && late(t0, err)) break;
  }
  __syncthreads();
  acquire(fast);
  for (int src = 0; src < 8; ++src)
    for (int i = lo + t; i < hi; i += NT) y[(size_t)src * S4 + i] = my_ag[(size_t)src * S4 + i];
}

int64_t alloc_buf(int64_t nbytes, int64_t uncached) {
  void* p = nullptr;
  hipError_t e = uncached ? hipExtMallocWithFlags(&p, nbytes, hipDeviceMallocUncached) : hipMalloc(&p, nbytes);
  TORCH_CHECK(e == hipSuccess, "alloc: ", hipGetErrorString(e));
  TORCH_CHECK(hipMemset(p, 0, nbytes) == hipSuccess);
  TORCH_CHECK(hipDeviceSynchronize() == hipSuccess);
  return (int64_t)p;
}
torch::Tensor ipc_handle(int64_t ptr) {
  hipIpcMemHandle_t h;
  TORCH_CHECK(hipIpcGetMemHandle(&h, (void*)ptr) == hipSuccess);
  auto t = torch::empty({(int64_t)sizeof(h)}, torch::kUInt8);
  memcpy(t.data_ptr(), &h, sizeof(h));
  return t;
}
int64_t ipc_open(torch::Tensor ht) {
  hipIpcMemHandle_t h;
  memcpy(&h, ht.data_ptr(), sizeof(h));
  void* p = nullptr;
  TORCH_CHECK(hipIpcOpenMemHandle(&p, h, hipIpcMemLazyEnablePeerAccess) == hipSuccess);
  return (int64_t)p;
}
void pingpong(int64_t buf, int64_t a, int64_t b, int64_t iters, int64_t out, int64_t base, int64_t sys) {
  hipLaunchKernelGGL(k_pingpong, dim3(256), dim3(64), 0, 0, (uint32_t*)buf, (int)a, (int)b, (int)iters,
                     (long long*)out, (uint32_t)base, (int)sys);
}
void xpingpong(int64_t local, int64_t remote, int64_t role, int64_t iters, int64_t out, int64_t base) {
  hipLaunchKernelGGL(k_xpingpong, dim3(1), dim3(64), 0, 0, (uint32_t*)local, (uint32_t*)remote, (int)role,
                     (int)iters, (long long*)out, (uint32_t)base);
}
void coll(int64_t x, int64_t y, torch::Tensor peers, int64_t rank, int64_t n, int64_t tag, int64_t err,
          int64_t fmt, int64_t mode, int64_t blocks) {
  Peers P;
  for (int i = 0; i < 8; ++i) P.p[i] = (uint64_t)peers[i].item<int64_t>();
  if (fmt == 0)
    hipLaunchKernelGGL(k_tagged, dim3(blocks), dim3(256), 0, 0, (const uint32_t*)x, (uint32_t*)y, P, (int)rank,
                       (int)(n / 16), (uint32_t)tag, (int*)err, (int)mode);
  else
    hipLaunchKernelGGL(k_flag, dim3(blocks), dim3(256), 0, 0, (const uint4*)x, (uint4*)y, P, (int)rank,
                       (int)(n / 64), (uint32_t)tag, (int*)err, (int)mode, (int)(fmt == 2));
}
"""

FUNCS = ["alloc_buf", "ipc_handle", "ipc_open", "pingpong", "xpingpong", "coll"]
BUF = 128 << 20


def build(rank):
    d = "/tmp/k31_p2p_build"
    os.makedirs(d, exist_ok=True)
    kw = dict(name="k31_p2p", cpp_sources=DECL, cuda_sources=SRC, functions=FUNCS, build_directory=d,
              extra_cuda_cflags=["-O3", "--offload-arch=gfx950"], verbose=False)
    if rank == 0:
        m = load_inline(**kw)
    dist.barrier()
    if rank != 0:
        m = load_inline(**kw)
    return m


def main():
    dist.init_process_group("gloo")
    rank, ws = dist.get_rank(), dist.get_world_size()
    assert ws == 8
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    nccl = dist.new_group(backend="nccl")
    m = build(rank)
    say = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)

    # (a) one GPU (rank 0 only)
    if rank == 0:
        out = torch.zeros(4, dtype=torch.int64, device=dev)
        for sys, name in ((1, "uncached+system"), (0, "hipMalloc+agent")):
            buf = m.alloc_buf(1 << 20, sys)
            for a, b in ((0, 8), (0, 1), (0, 4)):
                out.zero_()
                m.pingpong(buf, a, b, 2000, out.data_ptr(), 0, sys)  # warmup
                out.zero_()
                m.pingpong(buf, a, b, 20000, out.data_ptr(), 2000, sys)
                torch.cuda.synchronize()
                o = out.tolist()
                say(f"(a) {name:16s} CTA {a}<->{b}  xcc {o[1]}/{o[2]}  one-way {o[0] * 10.0 / 20000 / 2:7.0f} ns"
                    f"{'  TIMEOUT' if o[3] else ''}")
    dist.barrier()

    # peer buffers
    base = m.alloc_buf(BUF, 1)
    hs = [None] * ws
    dist.all_gather_object(hs, m.ipc_handle(base))
    peers = torch.tensor([base if r == rank else m.ipc_open(hs[r]) for r in range(ws)], dtype=torch.int64)
    dist.barrier()

    # (b) rank 0 <-> 1 (and 0 <-> 4)
    out = torch.zeros(4, dtype=torch.int64, device=dev)
    for other in (1, 4):
        for it, b0 in ((2000, 1 << 20), (20000, 2 << 20)):
            out.zero_()
            dist.barrier()
            if rank in (0, other):
                role = 0 if rank == 0 else 1
                m.xpingpong(base + FLX, int(peers[other if rank == 0 else 0]) + FLX, role, it, out.data_ptr(),
                            b0 + other * 100000)
            torch.cuda.synchronize()
        if rank == 0:
            o = out.tolist()
            say(f"(b) GPU 0<->{other}  one-way {o[0] * 10.0 / 20000 / 2:7.0f} ns{'  TIMEOUT' if o[3] else ''}")
    dist.barrier()

    # (c)/(d)
    err = torch.zeros(1, dtype=torch.int32, device=dev)
    tag = [1000]
    cases = [("AR", 0, 8, 7168), ("AR", 0, 32, 7168), ("AR", 0, 128, 3584), ("AR", 0, 128, 7168),
             ("AR", 0, 128, 10752), ("AR", 0, 384, 7168), ("A2A", 1, 128, 12288), ("A2A", 1, 128, 49152)]
    for kind, mode, rows, h in cases:
        n = rows * h
        x = torch.full((n,), 0.5 * (rank + 1), dtype=torch.bfloat16, device=dev)
        y = torch.empty_like(x)
        mb = n * 2 / 2**20
        line = f"(c) {kind:3s} rows {rows:4d} x {h:5d} = {mb:6.2f} MB:"
        for fmt, fname in ((0, "tagged"), (1, "flag"), (2, "flag-waitcnt")):
            for blocks in (256,):
                def run():
                    tag[0] += 1
                    m.coll(x.data_ptr(), y.data_ptr(), peers, rank, n, tag[0], err.data_ptr(), fmt, mode, blocks)
                bad = 0
                for v in range(12):
                    x.fill_(0.5 * (rank + 1) + (v % 4))
                    run()
                    want = (18.0 + 8 * (v % 4)) if mode == 0 else None
                    if mode == 0:
                        bad += int(not bool((y.float() == want).all()))
                    else:
                        exp = (0.5 * torch.arange(1, 9, device=dev) + (v % 4)).view(8, 1)
                        bad += int(not bool((y.view(8, -1).float() == exp).all()))
                x.fill_(0.5 * (rank + 1))
                torch.cuda.synchronize()
                dist.barrier()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(100):
                    run()
                e1.record()
                torch.cuda.synchronize()
                us = e0.elapsed_time(e1) * 10.0
                ok = int(err.item()) == 0 and bad == 0
                if mode == 0:
                    ok = ok and bool((y.float() == 18.0).all())
                else:
                    ok = ok and bool((y.view(8, -1).float() == (0.5 * torch.arange(1, 9, device=dev)).view(8, 1)).all())
                line += f"  {fname} {us:7.1f} us{'' if ok else ' FAIL'}"
                err.zero_()
                dist.barrier()
        xr = x.clone()
        for _ in range(10):
            if mode == 0:
                dist.all_reduce(xr, group=nccl)
            else:
                dist.all_to_all_single(y, x, group=nccl)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(100):
            if mode == 0:
                dist.all_reduce(xr, group=nccl)
            else:
                dist.all_to_all_single(y, x, group=nccl)
        e1.record()
        torch.cuda.synchronize()
        line += f"  rccl {e0.elapsed_time(e1) * 10.0:7.1f} us"
        say(line)
        dist.barrier()
    dist.barrier()
    dist.destroy_process_group()


FLX = 126 << 20  # cross-GPU ping-pong words, apart from the collective regions

if __name__ == "__main__":
    main()
