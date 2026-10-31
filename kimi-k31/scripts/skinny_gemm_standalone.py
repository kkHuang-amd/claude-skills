#!/usr/bin/env python3
"""Torch-free binary around bf16_skinny_gemm_gfx950 (rocprofv3 aborts inside torch processes).
Usage: skinny_gemm_standalone.py <M> <N> <K> <splits> <w_depth> -> /tmp/k31hip/skg_<...>
Run:   /tmp/k31hip/skg_<...> [iters]   (prints us/iter; weights rotate past the MALL)"""
import os
import re
import subprocess
import sys

M, N, K, S, D = map(int, sys.argv[1:6])
src = open("/sgl-workspace/sglang-k31-amd/python/sglang/kernels/jit/csrc/gemm/bf16_skinny_gemm_gfx950.cuh").read()
kernel = re.search(r"namespace bf16_skinny_gemm_gfx950 \{.*?\}  // namespace bf16_skinny_gemm_gfx950", src, re.S).group(0)
mblk = 8 if M > 64 else (4 if M > 32 else (2 if M > 16 else 1))
KG = int(os.environ.get('KG', '1'))
chunks = K // 128 // S // KG
main = r'''
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <type_traits>
#define SGL_DEVICE __forceinline__ __device__
KERNEL
using namespace bf16_skinny_gemm_gfx950;
int main(int argc, char** argv) {
  const int M = MM, N = NN, K = KK, S = SS, iters = argc > 1 ? atoi(argv[1]) : 50;
  const size_t wbytes = (size_t)(PREB ? (N + 127) / 128 * 128 : N) * K * 2;
  const int ncopy = (int)((640ull << 20) / wbytes) + 4;
  std::vector<uint16_t> h((size_t)N * K);
  for (size_t i = 0; i < h.size(); ++i) h[i] = 0x3c00 | (uint16_t)(i % 61);
  std::vector<__bf16*> w(ncopy);
  for (auto& p : w) { hipMalloc(&p, wbytes); hipMemcpy(p, h.data(), wbytes, hipMemcpyHostToDevice); }
  __bf16 *x, *y; float* ws;
  hipMalloc(&x, (size_t)M * K * 2); hipMemcpy(x, h.data(), (size_t)M * K * 2, hipMemcpyHostToDevice);
  hipMalloc(&y, (size_t)M * N * 2); hipMalloc(&ws, (size_t)S * M * ((N + 127) / 128 * 128) * 4);
  int* cnt; hipMalloc(&cnt, 4096 * 4); hipMemset(cnt, 0, 4096 * 4);
  Params p{}; p.x = x; p.y = y; p.ws = ws; p.counters = cnt; p.n_tiles = (N + kNTile - 1) / kNTile; p.M = M; p.N = N; p.K = K; p.splits = S; p.chunks_per_split = CHUNKS;
  dim3 grid = FUSE == 1 ? dim3(8 * ((p.n_tiles + 7) / 8) * S) : dim3(p.n_tiles, S);
  auto run = [&](int i) {
    p.w = w[i % ncopy];
    skinny_gemm_kernel<MBLK, DEPTH, CHUNKS, ABL, PRE, KGRP, FUSE><<<grid, kThreads * KGRP>>>(p);
    if ((FUSE == 0 || FUSE == 6) && S > 1 && ABL != 6) splitk_reduce_kernel<<<(M * N / 4 + 255) / 256, 256>>>(ws, y, (int64_t)M * N, S);
    if (FUSE == 5 && S > 1 && ABL != 6) splitk_reduce_tiles_kernel<<<(p.n_tiles * M * 128 / 8 + 255) / 256, 256>>>((const __bf16*)ws, y, M, N, p.n_tiles, S);
  };
  for (int i = 0; i < 3; ++i) run(i);
  hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
  hipEventRecord(a);
  for (int i = 0; i < iters; ++i) run(i);
  hipEventRecord(b); hipEventSynchronize(b);
  float ms; hipEventElapsedTime(&ms, a, b);
  printf("M=%d N=%d K=%d S=%d D=%d: %.2f us/iter (incl. launch gaps)\n", M, N, K, S, DEPTH, ms * 1000 / iters);
  return 0;
}
'''
FUSE = int(os.environ.get("FUSE", "0"))
for k, v in (("PREB", 1 if os.environ.get("PRE") == "1" else 0), ("FUSE", FUSE), ("KGRP", KG), ("ABL", int(os.environ.get("ABLATE", "0"))), ("PRE", "true" if os.environ.get("PRE") == "1" else "false"), ("KERNEL", kernel), ("MM", M), ("NN", N), ("KK", K), ("SS", S), ("MBLK", mblk), ("DEPTH", D), ("CHUNKS", chunks)):
    main = main.replace(k, str(v))
os.makedirs("/tmp/k31hip", exist_ok=True)
path = f"/tmp/k31hip/skg_{M}_{N}_{K}_{S}_{D}_{KG}_{FUSE}.hip"
open(path, "w").write(main)
r = subprocess.run(["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx950", "-Wno-unused-result", "-Wno-unused-value", path, "-o", path[:-4]],
                   capture_output=True, text=True)
print("build", "ok" if r.returncode == 0 else "FAILED", path[:-4])
if r.returncode:
    print("\n".join(l for l in r.stderr.splitlines() if "error" in l)[:2000])
