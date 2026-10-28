// M2b: HBM streaming rate of the P1 weight layout vs bytes in flight per CU (MOE_KERNEL_DESIGN_h13-17_1028.md).
// 4.29 GB of fragment-native weights (E=64, N=8192, K=16384 fp4), every byte read exactly once (>> any cache).
// Persistent grid: G WGs per CU (128 threads = 4 waves each); a work item = 4 waves x 4 weight groups x all 128 k-steps (4 MB).
// Per wave per k-step: 16 x b128 loads = 8 KB (piece u of group j at w_off + u*512 + lane*16, coalesced per instruction).
// MODE 0 (registers): D k-steps of loads in flight per wave (16*D <= 63 = loadcnt limit), xor-consumed.
// MODE 1 (async -> LDS): S k-steps in flight per wave into a per-wave LDS ring (S stages x 8 KB), waited with s_wait_asynccnt.
// Reports TB/s from hipEvent time (4.29e9 B / t) and per-wave bytes in flight per CU.
// Build: hipcc --offload-arch=gfx1250 -O3 -o m2b_hbm_stream m2b_hbm_stream.hip ; Run: ./m2b_hbm_stream
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

typedef int v4i __attribute__((ext_vector_type(4)));
#define GAS __attribute__((address_space(1)))
#define LAS __attribute__((address_space(3)))
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

constexpr int E = 64, N = 8192, K = 16384, KS = K / 128, NG = N / 32;
constexpr int ITEMS = E * NG / 16;  // 16 groups (4 waves x 4) per item -> 1024 items
__host__ __device__ inline size_t w_off(int e, int ng, int ks) { return (((size_t)e * NG + ng) * KS + ks) * 2048; }

__device__ inline const char* src(const char* W, int item, int wave, int j, int ks, int u, int lane) {
  const int e = item / (NG / 16), ng = (item % (NG / 16)) * 16 + wave * 4 + j;
  return W + w_off(e, ng, ks) + u * 512 + lane * 16;
}

template <int D>
__global__ __launch_bounds__(128) void stream_regs(const char* W, int* out) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  v4i acc = {0, 0, 0, 0};
  for (int item = blockIdx.x; item < ITEMS; item += gridDim.x) {
    v4i buf[D][16];
#pragma unroll
    for (int d = 0; d < D; ++d)
#pragma unroll
      for (int q = 0; q < 16; ++q) buf[d][q] = *(const v4i*)src(W, item, wave, q / 4, d, q % 4, lane);
    for (int ks = 0; ks < KS; ks += D) {
#pragma unroll
      for (int d = 0; d < D; ++d) {
#pragma unroll
        for (int q = 0; q < 16; ++q) acc ^= buf[d][q];                        // consume step ks+d
        if (ks + d + D < KS) {
#pragma unroll
          for (int q = 0; q < 16; ++q) buf[d][q] = *(const v4i*)src(W, item, wave, q / 4, ks + d + D, q % 4, lane);
        }
      }
    }
  }
  out[blockIdx.x * 128 + threadIdx.x] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}

template <int S>
__global__ __launch_bounds__(128) void stream_lds(const char* W, int* out) {
  __shared__ __attribute__((aligned(16))) char lds[4 * S * 8192];
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  char* ring = lds + wave * S * 8192;
  for (int item = blockIdx.x; item < ITEMS; item += gridDim.x) {
    for (int ks = 0; ks < KS; ++ks) {
      char* st = ring + (ks % S) * 8192;
#pragma unroll
      for (int q = 0; q < 16; ++q)
        __builtin_amdgcn_global_load_async_to_lds_b128((GAS v4i*)src(W, item, wave, q / 4, ks, q % 4, lane),
                                                       (LAS v4i*)(st + q * 512 + lane * 16), 0, 0);
      __builtin_amdgcn_s_wait_asynccnt(16 * (S - 1));                          // keep S steps in flight
    }
  }
  __builtin_amdgcn_s_wait_asynccnt(0);
  __syncthreads();
  out[blockIdx.x * 128 + threadIdx.x] = *(const int*)(lds + threadIdx.x * 4);
}

template <typename F>
void timeit(const char* name, int g, int cus, double inflight_cu_kb, F launch) {
  hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
  launch(); CHECK(hipDeviceSynchronize());
  const int reps = 3;
  CHECK(hipEventRecord(e0));
  for (int r = 0; r < reps; ++r) launch();
  CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
  float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); ms /= reps;
  const double bytes = (double)E * N * K / 2;
  printf("%-14s WG/CU %d waves/CU %2d | in flight/CU %4.0f KB | %.3f ms  %.2f TB/s\n", name, g, 4 * g, inflight_cu_kb, ms, bytes / (ms * 1e-3) / 1e12);
}

int main() {
  hipDeviceProp_t p; CHECK(hipGetDeviceProperties(&p, 0));
  const int cus = p.multiProcessorCount;
  const size_t wb = (size_t)E * N * K / 2;
  char* W; CHECK(hipMalloc(&W, wb)); CHECK(hipMemset(W, 1, wb));
  int* out; CHECK(hipMalloc(&out, (size_t)cus * 8 * 128 * 4));
  printf("%s, %d CUs, %.2f GB, LDS per block max %zu KB\n", p.gcnArchName, cus, wb / 1e9, p.sharedMemPerBlock / 1024);
  for (int g : {1, 2, 4}) {
    const int grid = cus * g;
    timeit("regs D=1", g, cus, 4.0 * g * 8, [&] { stream_regs<1><<<grid, 128>>>(W, out); });
    timeit("regs D=2", g, cus, 4.0 * g * 16, [&] { stream_regs<2><<<grid, 128>>>(W, out); });
    timeit("regs D=3", g, cus, 4.0 * g * 24, [&] { stream_regs<3><<<grid, 128>>>(W, out); });
    timeit("lds S=2", g, cus, 4.0 * g * 16, [&] { stream_lds<2><<<grid, 128>>>(W, out); });
    timeit("lds S=3", g, cus, 4.0 * g * 24, [&] { stream_lds<3><<<grid, 128>>>(W, out); });
    timeit("lds S=4", g, cus, 4.0 * g * 32, [&] { stream_lds<4><<<grid, 128>>>(W, out); });
  }
  return 0;
}
