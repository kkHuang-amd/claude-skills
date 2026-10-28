// M2: global_load_b128 -> VGPR throughput per CU and load-to-use latency on gfx1250 (MOE_KERNEL_DESIGN_h13-17_1028.md, section 5).
// Throughput: every wave streams 16 B per lane (512 B per wave per load, coalesced like the preshuffled weight rows) from a buffer of
// BUF bytes, U independent loads in flight per iteration, xor-accumulated. Bytes per CU / per-wave cycles (s_get_shader_cycles_u64).
// Latency: one wave per CU, dependent pointer chase (each address comes from the previous load), L2-resident buffer.
// Build: hipcc --offload-arch=gfx1250 -O3 -o l2_vgpr_bw l2_vgpr_bw.hip ; Run: ./l2_vgpr_bw
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)

template <int U>
__global__ __launch_bounds__(512) void stream(const int4* __restrict__ buf, unsigned mask, int* out, unsigned long long* t, int iters) {
  const int lane = threadIdx.x & 31;
  const unsigned wave = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  const unsigned step = 32 * U;                       // int4 per wave per iteration
  unsigned base = (wave * 7919u * step) & mask;       // spread waves over the buffer (n16 = mask + 1, power of 2)
  int4 acc = {0, 0, 0, 0};
  __syncthreads();
  unsigned long long c0 = clock64();
  for (int it = 0; it < iters; ++it) {
    int4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = buf[(base + u * 32 + lane) & mask];
#pragma unroll
    for (int u = 0; u < U; ++u) { acc.x ^= v[u].x; acc.y ^= v[u].y; acc.z ^= v[u].z; acc.w ^= v[u].w; }
    base = (base + step * 97) & mask;                  // jump: no reuse within a wave
  }
  unsigned long long c1 = clock64();
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (lane == 0) t[wave] = c1 - c0;
}

__global__ void chase(const int* __restrict__ next, int* out, unsigned long long* t, int iters) {
  int p = threadIdx.x * 16;
  unsigned long long c0 = clock64();
  for (int i = 0; i < iters; ++i) p = next[p];
  unsigned long long c1 = clock64();
  out[threadIdx.x] = p;
  if (threadIdx.x == 0) t[0] = c1 - c0;
}

template <int U>
void run_stream(const int4* d_buf, size_t bytes, int wg_per_cu, int cus) {
  const int blocks = cus * wg_per_cu, waves = blocks * 4;
  const size_t n16 = bytes / 16;
  const int iters = 4096 / U;
  int* d_out; unsigned long long* d_t;
  CHECK(hipMalloc(&d_out, blocks * 128 * sizeof(int)));
  CHECK(hipMalloc(&d_t, waves * sizeof(unsigned long long)));
  for (int w = 0; w < 3; ++w) stream<U><<<blocks, 128>>>(d_buf, (unsigned)(n16 - 1), d_out, d_t, iters);
  CHECK(hipDeviceSynchronize());
  std::vector<unsigned long long> t(waves);
  CHECK(hipMemcpy(t.data(), d_t, waves * sizeof(unsigned long long), hipMemcpyDeviceToHost));
  std::sort(t.begin(), t.end());
  const double bytes_wave = (double)iters * U * 512;
  const double med = (double)t[waves / 2], mx = (double)t[waves - 1];
  // all waves of a CU run concurrently (small VGPR); per-CU bytes = waves_per_cu * bytes_wave over ~the slowest wave
  const double wpc = 4.0 * wg_per_cu;
  printf("buf %4zu MB  waves/CU %2.0f  U=%2d | per-wave %.1f B/clk (med)  per-CU %.0f B/clk (med) %.0f (max-cyc) | cyc/load-round %.0f\n",
         bytes >> 20, wpc, U, bytes_wave / med, wpc * bytes_wave / med, wpc * bytes_wave / mx, med / iters);
  CHECK(hipFree(d_out)); CHECK(hipFree(d_t));
}

// CU-count scaling: one 512-thread block (16 waves) per CU on n CUs (assumes 1 block per CU when blocks <= CUs)
template <int U>
void run_scale(const int4* d_buf, size_t bytes, int nblk) {
  const int waves = nblk * 16, iters = 4096 / U;
  int* d_out; unsigned long long* d_t;
  CHECK(hipMalloc(&d_out, nblk * 512 * sizeof(int)));
  CHECK(hipMalloc(&d_t, waves * sizeof(unsigned long long)));
  for (int w = 0; w < 3; ++w) stream<U><<<nblk, 512>>>(d_buf, (unsigned)(bytes / 16 - 1), d_out, d_t, iters);
  CHECK(hipDeviceSynchronize());
  std::vector<unsigned long long> t(waves);
  CHECK(hipMemcpy(t.data(), d_t, waves * sizeof(unsigned long long), hipMemcpyDeviceToHost));
  std::sort(t.begin(), t.end());
  const double bw = (double)iters * U * 512;
  printf("scale: %3d CUs x 16 waves U=%2d buf %zu MB | per-CU %.0f B/clk (med) | total %.0f B/clk\n", nblk, U, bytes >> 20,
         16 * bw / t[waves / 2], nblk * 16 * bw / t[waves / 2]);
  CHECK(hipFree(d_out)); CHECK(hipFree(d_t));
}

int main(int argc, char** argv) {
  hipDeviceProp_t p; CHECK(hipGetDeviceProperties(&p, 0));
  const int cus = p.multiProcessorCount;
  printf("device %s, %d CUs, L2 %d KB\n", p.gcnArchName, cus, p.l2CacheSize / 1024);
  const size_t maxb = (size_t)512 << 20;
  int4* d_buf; CHECK(hipMalloc(&d_buf, maxb));
  CHECK(hipMemset(d_buf, 1, maxb));
  if (argc > 1) {  // ./l2_vgpr_bw scale
    for (int n : {96, 48, 24, 12}) { run_scale<8>(d_buf, (size_t)2 << 20, n); run_scale<16>(d_buf, (size_t)2 << 20, n); }
    return 0;
  }
  for (size_t mb : {2, 32, 512}) {
    const size_t b = mb << 20;
    for (int g : {1, 2, 4}) {
      run_stream<1>(d_buf, b, g, cus);
      run_stream<4>(d_buf, b, g, cus);
      run_stream<8>(d_buf, b, g, cus);
      run_stream<16>(d_buf, b, g, cus);
    }
  }
  // latency: pointer chase in a 1 MB (L2) and 512 MB (HBM) ring, stride 4 KB + 64 B (new line each hop)
  for (size_t mb : {1, 512}) {
    const size_t n = (mb << 20) / 4, stride = (4096 + 64) / 4;
    std::vector<int> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = (int)((i + stride) % n);
    int* d_next; CHECK(hipMalloc(&d_next, n * 4));
    CHECK(hipMemcpy(d_next, h.data(), n * 4, hipMemcpyHostToDevice));
    int* d_out; unsigned long long* d_t;
    CHECK(hipMalloc(&d_out, 32 * 4)); CHECK(hipMalloc(&d_t, 8));
    const int iters = 4096;
    for (int w = 0; w < 3; ++w) chase<<<1, 32>>>(d_next, d_out, d_t, iters);
    CHECK(hipDeviceSynchronize());
    unsigned long long c; CHECK(hipMemcpy(&c, d_t, 8, hipMemcpyDeviceToHost));
    printf("chase ring %4zu MB: %.0f cyc per dependent load\n", mb, (double)c / iters);
    CHECK(hipFree(d_next)); CHECK(hipFree(d_out)); CHECK(hipFree(d_t));
  }
  return 0;
}
