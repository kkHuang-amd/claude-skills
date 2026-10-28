// M1: issue rate / latency of v_wmma_scale_f32_32x16x128_f4 on gfx1250 (MOE_KERNEL_DESIGN_h13-17_1028.md, section 5).
// Each wave keeps A_N weight fragments (16 VGPR), B_N token fragments (8 VGPR) and A_N*B_N independent accumulators
// (16 VGPR each) in registers and issues ITERS rounds of A_N*B_N WMMAs. (A_N,B_N) = (1,1) is a dependent chain (latency);
// (2,8) is the production per-wave pattern (16 WMMA per k128 step).
// Build: hipcc --offload-arch=gfx1250 -O3 -o wmma_fp4_rate wmma_fp4_rate.hip
// Run:   ./wmma_fp4_rate [wg_per_cu ...]   (block = 128 threads = 4 waves, one per SIMD when 1 WG/CU)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

typedef int v16i __attribute__((ext_vector_type(16)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v16f __attribute__((ext_vector_type(16)));

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)

// VS: 0 = uniform scales (SGPR), 1 = per-lane scales (VGPR), 2 = per-lane, different per operand fragment
template <int A_N, int B_N, int VS = 0>
__global__ __launch_bounds__(128) __attribute__((amdgpu_waves_per_eu(1, 4))) void wmma_loop(const int* in, float* out, unsigned long long* t, int iters) {
  v16i a[A_N];
  v8i b[B_N];
  for (int i = 0; i < A_N; ++i)
    for (int e = 0; e < 16; ++e) a[i][e] = in[(threadIdx.x + 7 * i + e) & 255];
  for (int j = 0; j < B_N; ++j)
    for (int e = 0; e < 8; ++e) b[j][e] = in[(threadIdx.x + 13 * j + e + 64) & 255];
  const int sa = VS ? in[300 + (threadIdx.x & 31)] : in[300], sb = VS ? in[301 + (threadIdx.x & 31)] : in[301];
  int sav[A_N], sbv[B_N];
  for (int i = 0; i < A_N; ++i) sav[i] = VS == 2 ? in[340 + i + (threadIdx.x & 31)] : sa;
  for (int j = 0; j < B_N; ++j) sbv[j] = VS == 2 ? in[380 + j + (threadIdx.x & 31)] : sb;
  v16f acc[A_N * B_N];
  for (int k = 0; k < A_N * B_N; ++k) acc[k] = v16f{};
  __syncthreads();
  unsigned long long c0 = clock64(), r0 = wall_clock64();
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < A_N; ++i)
#pragma unroll
      for (int j = 0; j < B_N; ++j)
        acc[i * B_N + j] = __builtin_amdgcn_wmma_scale_f32_32x16x128_f4(
            a[i], b[j], (short)0, acc[i * B_N + j], 0, 0, sav[i], 0, 0, sbv[j], false, false);
  }
  unsigned long long c1 = clock64(), r1 = wall_clock64();
  float s = 0.f;
  for (int k = 0; k < A_N * B_N; ++k)
    for (int e = 0; e < 16; ++e) s += acc[k][e];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
  if ((threadIdx.x & 31) == 0) {
    int w = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    t[2 * w] = c1 - c0;
    t[2 * w + 1] = r1 - r0;
  }
}

template <int A_N, int B_N, int VS = 0>
void run(int wg_per_cu, int cus, const int* d_in) {
  const int iters = 2048, blocks = cus * wg_per_cu, waves = blocks * 4;
  float* d_out; unsigned long long* d_t;
  CHECK(hipMalloc(&d_out, blocks * 128 * sizeof(float)));
  CHECK(hipMalloc(&d_t, 2 * waves * sizeof(unsigned long long)));
  hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
  for (int w = 0; w < 2; ++w) wmma_loop<A_N, B_N, VS><<<blocks, 128>>>(d_in, d_out, d_t, iters);  // warmup (clock ramp)
  CHECK(hipEventRecord(e0));
  const int reps = 5;
  for (int r = 0; r < reps; ++r) wmma_loop<A_N, B_N, VS><<<blocks, 128>>>(d_in, d_out, d_t, iters);
  CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
  float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); ms /= reps;
  std::vector<unsigned long long> t(2 * waves);
  CHECK(hipMemcpy(t.data(), d_t, t.size() * sizeof(unsigned long long), hipMemcpyDeviceToHost));
  std::vector<double> cyc(waves), mhz(waves);
  for (int w = 0; w < waves; ++w) { cyc[w] = (double)t[2 * w]; mhz[w] = cyc[w] / (t[2 * w + 1] / 100.0); }
  std::sort(cyc.begin(), cyc.end()); std::sort(mhz.begin(), mhz.end());
  const double nw = (double)iters * A_N * B_N;               // WMMAs per wave
  const double wps = wg_per_cu;                               // waves per SIMD (4-wave blocks, if resident)
  const double clk = mhz[waves / 2];
  // aggregate: total WMMAs over all SIMDs vs kernel time x clock
  const double agg = (ms * 1e-3 * clk * 1e6) / (nw * waves / (cus * 4.0));
  printf("VS=%d A_N=%d B_N=%d acc=%2d wg/cu=%d | per-wave cyc/WMMA med %.2f (min %.2f max %.2f) -> per-SIMD %.2f | aggregate %.2f cyc/WMMA/SIMD | clk %.0f MHz | %.3f ms | TFLOP/s %.0f\n",
         VS, A_N, B_N, A_N * B_N, wg_per_cu, cyc[waves / 2] / nw, cyc[0] / nw, cyc[waves - 1] / nw, cyc[waves / 2] / nw / wps, agg, clk, ms,
         nw * waves * 2.0 * 32 * 16 * 128 / (ms * 1e-3) / 1e12);
  CHECK(hipFree(d_out)); CHECK(hipFree(d_t));
}

int main(int argc, char** argv) {
  hipDeviceProp_t p; CHECK(hipGetDeviceProperties(&p, 0));
  const int cus = p.multiProcessorCount;
  std::vector<int> h_in(512);
  for (int i = 0; i < 512; ++i) h_in[i] = 0x22222222 * (i & 1) + i;  // arbitrary fp4 bits
  for (int i = 300; i < 512; ++i) h_in[i] = 0x7f7f7f7f;                                  // e8m0 scale = 1.0
  int* d_in; CHECK(hipMalloc(&d_in, 512 * sizeof(int)));
  CHECK(hipMemcpy(d_in, h_in.data(), 512 * sizeof(int), hipMemcpyHostToDevice));
  printf("device %s, %d CUs\n", p.gcnArchName, cus);
  std::vector<int> wgs;
  for (int i = 1; i < argc; ++i) wgs.push_back(atoi(argv[i]));
  if (wgs.empty()) wgs = {1, 2};
  for (int g : wgs) {
    run<2, 8, 0>(g, cus, d_in); run<2, 8, 1>(g, cus, d_in); run<2, 8, 2>(g, cus, d_in);
    run<4, 8, 0>(g, cus, d_in); run<4, 8, 1>(g, cus, d_in); run<4, 8, 2>(g, cus, d_in);
  }
  return 0;
}
