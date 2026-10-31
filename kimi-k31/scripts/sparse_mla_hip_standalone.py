#!/usr/bin/env python3
"""Build a torch-free HIP binary around the sparse_mla_prefill_gfx950 kernel so rocprofv3
can collect counters (rocprofv3 aborts inside torch/triton processes: two LLVMs register
the same cl::opt). Synthetic data only: timing / counters, no accuracy.
Usage: sparse_mla_hip_standalone.py [stages=2] [ablate=0] [seq=16384] -> /tmp/k31hip/smla_<s>_<a>
Run:   /tmp/k31hip/smla_2_0 [iters=20]
Prof:  rocprofv3 --pmc SQ_WAVES SQ_INSTS_VALU ... -- /tmp/k31hip/smla_2_0 3"""
import os
import re
import subprocess
import sys

stages = int(sys.argv[1]) if len(sys.argv) > 1 else 2
ablate = int(sys.argv[2]) if len(sys.argv) > 2 else 0
seq = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
src = open("/sgl-workspace/sglang-k31-amd/python/sglang/kernels/jit/csrc/attention/sparse_mla_prefill_gfx950.cuh").read()
m = re.search(r"namespace sparse_mla_prefill_gfx950 \{.*?\}  // namespace sparse_mla_prefill_gfx950", src, re.S)
kernel = m.group(0)
main = r'''
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#define SGL_DEVICE __forceinline__ __device__
KERNEL
using namespace sparse_mla_prefill_gfx950;
int main(int argc, char** argv) {
  const int T = SEQ, H = 12, TOPK = 2048, POOL = 32768, iters = argc > 1 ? atoi(argv[1]) : 20;
  std::vector<uint8_t> kv((size_t)POOL * kRow);
  for (size_t i = 0; i < kv.size(); ++i) kv[i] = (uint8_t)((i * 2654435761u) >> 24) & 0x77;  // finite fp8
  for (int r = 0; r < POOL; ++r) { float s = 1.0f / 64; for (int g = 0; g < 4; ++g) memcpy(&kv[(size_t)r * kRow + 512 + 4 * g], &s, 4); }
  std::vector<int> idx((size_t)T * TOPK);
  uint32_t x = 1;
  for (int t = 0; t < T; ++t) for (int k = 0; k < TOPK; ++k) {  // clustered: 64-token blocks share 80% of keys
    x = x * 1664525u + 1013904223u; const bool own = (x >> 8) % 5 == 0;
    uint32_t y = own ? x * 2246822519u : ((t / 64) * 2654435761u + k * 40503u) * 2246822519u;
    idx[(size_t)t * TOPK + k] = (int)((y >> 7) % (POOL - T + t + 1));
  }
  std::vector<uint16_t> q((size_t)T * H * 576);
  for (size_t i = 0; i < q.size(); ++i) q[i] = 0x3c00 + (uint16_t)(i % 97);  // bf16 ~ small values
  uint8_t* d_kv; int* d_idx; __bf16 *d_qn, *d_qr, *d_out; float* d_lse;
  hipMalloc(&d_kv, kv.size()); hipMalloc(&d_idx, idx.size() * 4);
  hipMalloc(&d_qn, (size_t)T * H * 512 * 2); hipMalloc(&d_qr, (size_t)T * H * 64 * 2);
  hipMalloc(&d_out, (size_t)T * H * 512 * 2); hipMalloc(&d_lse, (size_t)T * H * 4);
  hipMemcpy(d_kv, kv.data(), kv.size(), hipMemcpyHostToDevice);
  hipMemcpy(d_idx, idx.data(), idx.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(d_qn, q.data(), (size_t)T * H * 512 * 2, hipMemcpyHostToDevice);
  hipMemcpy(d_qr, q.data(), (size_t)T * H * 64 * 2, hipMemcpyHostToDevice);
  Params p{};
  p.q_nope = d_qn; p.q_rope = d_qr; p.kv = d_kv; p.idx = d_idx; p.out = d_out; p.lse = d_lse;
  p.kv_bytes = (uint32_t)kv.size();
  p.sqn_t = H * 512; p.sqn_h = 512; p.sqr_t = H * 64; p.sqr_h = 64; p.H = H; p.topk = TOPK;
  p.qk_scale = 0.0416667f * 1.4426950408889634f; p.store_lse = 1;
  hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
  for (int i = 0; i < 2; ++i) sparse_mla_prefill_kernel<STAGES, ABLATE><<<T, kThreads>>>(p);
  hipEventRecord(a);
  for (int i = 0; i < iters; ++i) sparse_mla_prefill_kernel<STAGES, ABLATE><<<T, kThreads>>>(p);
  hipEventRecord(b); hipEventSynchronize(b);
  float ms; hipEventElapsedTime(&ms, a, b);
  printf("stages=%d ablate=%d seq=%d: %.3f ms/iter\n", STAGES, ABLATE, T, ms / iters);
  return 0;
}
'''
code = main.replace("KERNEL", kernel).replace("STAGES", str(stages)).replace("ABLATE", str(ablate)).replace("SEQ", str(seq))
os.makedirs("/tmp/k31hip", exist_ok=True)
path = f"/tmp/k31hip/smla_{stages}_{ablate}.hip"
open(path, "w").write(code)
r = subprocess.run(["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx950", "-Wno-unused-result", "-Wno-unused-value",
                    path, "-o", path[:-4]], capture_output=True, text=True)
print("build", "ok" if r.returncode == 0 else "FAILED", path[:-4])
if r.returncode:
    print("\n".join(l for l in r.stderr.splitlines() if "error" in l)[:2000])
