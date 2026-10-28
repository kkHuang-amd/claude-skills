// M3: does cluster multicast relieve L2 bandwidth on gfx1250? (MOE_KERNEL_DESIGN_h13-17_1028.md, section 5)
// 96 CUs x one 512-thread WG (16 waves), cluster {1,CS,1}. Each wave streams U x 16 B per lane per iteration.
// MODE 0 private : every WG reads its own region (= M2 baseline, ~88 B/clk/CU when all CUs pull)
// MODE 1 shared  : WGs of a cluster read the SAME addresses with plain global_load_b128
// MODE 2 mcast   : same addresses via cluster_load_b128 with the full cluster mask (one request per cluster if they line up)
// BAR > 0        : cluster barrier (split arrive/wait) every BAR iterations
// Metric: bytes delivered to each WG / its cycles (s_get_shader_cycles_u64), median and min/max over WGs.
// Build: hipcc --offload-arch=gfx1250 -O3 -o cluster_mcast_bw cluster_mcast_bw.hip ; Run: ./cluster_mcast_bw
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

namespace cg = cooperative_groups;
typedef int v4i __attribute__((ext_vector_type(4)));
#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)

constexpr int U = 8;

// JIT > 0: WGs with odd cluster rank s_sleep(JIT) every 8 iterations (emulated drift between cluster members)
template <int MODE, int CS, int BAR, int JIT = 0>
__global__ __launch_bounds__(512) void kern(const v4i* __restrict__ buf, unsigned mask16, int* out, unsigned long long* t, int iters) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x / 32;
  const unsigned wg = blockIdx.y;
  const unsigned region = (MODE == 0) ? wg : wg / CS;           // shared modes: one region per cluster
  const unsigned step = 16 * 32 * U;                             // v4i per WG per iteration (16 waves)
  unsigned base = (region * 7919u * step) & mask16;
  v4i acc = {0, 0, 0, 0};
  if constexpr (CS > 1) {
    cg::this_cluster().barrier_arrive();
    cg::this_cluster().barrier_wait(cg::cluster_group::arrival_token{});
  } else {
    __syncthreads();
  }
  unsigned long long c0 = clock64();
  for (int it = 0; it < iters; ++it) {
    v4i v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const unsigned idx = (base + (wave * U + u) * 32 + lane) & mask16;
      if constexpr (MODE == 2)
        v[u] = __builtin_amdgcn_cluster_load_b128((__attribute__((address_space(1))) v4i*)(buf + idx), 0, (1u << CS) - 1u);
      else
        v[u] = buf[idx];
    }
#pragma unroll
    for (int u = 0; u < U; ++u) acc ^= v[u];
    base = (base + step * 97) & mask16;
    if constexpr (JIT > 0) {
      if ((wg & 1) && (it & 7) == 7) __builtin_amdgcn_s_sleep(JIT);
    }
    if constexpr (BAR > 0 && CS > 1) {
      if ((it + 1) % BAR == 0) {
        cg::this_cluster().barrier_arrive();
        cg::this_cluster().barrier_wait(cg::cluster_group::arrival_token{});
      }
    }
  }
  unsigned long long c1 = clock64();
  out[wg * 512 + threadIdx.x] = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (threadIdx.x == 0) t[wg] = c1 - c0;
}

static std::vector<int> g_last;
template <int MODE, int CS, int BAR, int JIT = 0>
std::vector<int> run(const v4i* d_buf, size_t bytes, int nwg, const char* name) {
  const int iters = 512;
  int* d_out; unsigned long long* d_t;
  CHECK(hipMalloc(&d_out, nwg * 512 * sizeof(int)));
  CHECK(hipMalloc(&d_t, nwg * sizeof(unsigned long long)));
  hipLaunchAttribute attr{};
  attr.id = hipLaunchAttributeClusterDimension;
  attr.val.clusterDim = {1, (unsigned)CS, 1};
  hipLaunchConfig_t cfg{};
  cfg.gridDim = {1, (unsigned)nwg, 1};
  cfg.blockDim = {512, 1, 1};
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  const unsigned mask16 = (unsigned)(bytes / 16 - 1);
  for (int w = 0; w < 4; ++w) CHECK(hipLaunchKernelEx(&cfg, kern<MODE, CS, BAR, JIT>, d_buf, mask16, d_out, d_t, iters));
  CHECK(hipDeviceSynchronize());
  std::vector<unsigned long long> t(nwg);
  CHECK(hipMemcpy(t.data(), d_t, nwg * sizeof(unsigned long long), hipMemcpyDeviceToHost));
  std::sort(t.begin(), t.end());
  const double b = (double)iters * 16 * 32 * U * 16;  // bytes delivered per WG
  std::vector<int> o(nwg * 512);
  CHECK(hipMemcpy(o.data(), d_out, o.size() * sizeof(int), hipMemcpyDeviceToHost));
  printf("%-14s CS=%d BAR=%2d JIT=%2d | delivered per CU %.0f B/clk (med)  [%.0f .. %.0f]  | L2-side per CU %.0f B/clk\n", name, CS, BAR, JIT,
         b / t[nwg / 2], b / t[nwg - 1], b / t[0], (MODE == 0 ? 1.0 : 1.0 / CS) * b / t[nwg / 2]);
  CHECK(hipFree(d_out)); CHECK(hipFree(d_t));
  return o;
}

int main() {
  hipDeviceProp_t p; CHECK(hipGetDeviceProperties(&p, 0));
  const int nwg = p.multiProcessorCount;  // 96: one WG per CU
  const size_t bytes = (size_t)32 << 20;
  v4i* d_buf; CHECK(hipMalloc(&d_buf, bytes)); { std::vector<int> h(bytes / 4); for (size_t i = 0; i < h.size(); ++i) h[i] = (int)(i * 2654435761u); CHECK(hipMemcpy(d_buf, h.data(), bytes, hipMemcpyHostToDevice)); }
  run<0, 1, 0>(d_buf, bytes, nwg, "private");
  auto s2 = run<1, 2, 0>(d_buf, bytes, nwg, "shared");
  auto m2 = run<2, 2, 0>(d_buf, bytes, nwg, "mcast");
  printf("  correctness CS=2: mcast %s shared\n", s2 == m2 ? "==" : "!=");
  auto s4 = run<1, 4, 0>(d_buf, bytes, nwg, "shared");
  auto m4 = run<2, 4, 0>(d_buf, bytes, nwg, "mcast");
  printf("  correctness CS=4: mcast %s shared\n", s4 == m4 ? "==" : "!=");
  // drift: odd-rank WGs sleep every 8 iterations (s_sleep N ~ 64*N cycles)
  run<1, 2, 0, 2>(d_buf, bytes, nwg, "shared+drift");
  run<2, 2, 0, 2>(d_buf, bytes, nwg, "mcast+drift");
  run<2, 2, 8, 2>(d_buf, bytes, nwg, "mcast+drift+bar");
  run<1, 2, 0, 8>(d_buf, bytes, nwg, "shared+drift");
  run<2, 2, 0, 8>(d_buf, bytes, nwg, "mcast+drift");
  run<2, 2, 8, 8>(d_buf, bytes, nwg, "mcast+drift+bar");
  run<2, 2, 0, 32>(d_buf, bytes, nwg, "mcast+drift");
  run<2, 2, 8, 32>(d_buf, bytes, nwg, "mcast+drift+bar");
  return 0;
}
