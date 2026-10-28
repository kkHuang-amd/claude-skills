// D4 feasibility (MOE_KERNEL_DESIGN_h13-17_1028.md): can the EP1 down epilogue accumulate route-weighted rows straight into
// out[token] with global atomics instead of writing a per-route output (4.3 GB) and running gather-reduce?
// Shapes: T=8192 tokens x top-16 routes = 131072 rows, H=16384. Rows are visited in expert-grouped order (row r -> token tok[r],
// random but each token exactly 16 times), like stage-2 output tiles. Each thread handles 8 consecutive columns of one row.
// MODE 0: today's write   : per-route bf16 store grouped[r][h]            (4.3 GB written)
// MODE 1: today's reduce   : out[t][h] = sum_k w * grouped[row(t,k)][h]   (4.3 GB read + 268 MB bf16 written)
// MODE 2: fp32 atomics     : atomicAdd(out32[tok[r]][h], w * v)            (no-return global_atomic_add_f32)
// MODE 3: pk bf16 atomics  : global_atomic_pk_add_bf16 into out16[tok[r]][h]
// Build: hipcc --offload-arch=gfx1250 -O3 -o m4_atomic_reduce m4_atomic_reduce.hip ; Run: ./m4_atomic_reduce
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)
constexpr int T = 8192, TOPK = 16, ROWS = T * TOPK, H = 16384, VPT = 8;
typedef short v8s __attribute__((ext_vector_type(8)));
typedef short v2s __attribute__((ext_vector_type(2)));

__device__ inline float val(int r, int h) { return (float)((r * 31 + h) & 255) * (1.f / 256.f); }
__device__ inline short f2bf(float f) { return (short)(__float_as_uint(f) >> 16); }

__global__ void k_write(short* grouped) {             // MODE 0
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int r = (int)(i / (H / VPT)), h0 = (int)(i % (H / VPT)) * VPT;
  v8s o;
  for (int j = 0; j < VPT; ++j) o[j] = f2bf(val(r, h0 + j));
  *(v8s*)(grouped + (size_t)r * H + h0) = o;
}
__global__ void k_reduce(const short* grouped, const int* rows_of, const float* w, short* out) {  // MODE 1
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int t = (int)(i / (H / VPT)), h0 = (int)(i % (H / VPT)) * VPT;
  float acc[VPT] = {};
  for (int k = 0; k < TOPK; ++k) {
    const v8s g = *(const v8s*)(grouped + (size_t)rows_of[t * TOPK + k] * H + h0);
    const float wk = w[t * TOPK + k];
    for (int j = 0; j < VPT; ++j) acc[j] += wk * __uint_as_float(((unsigned)(unsigned short)g[j]) << 16);
  }
  v8s o;
  for (int j = 0; j < VPT; ++j) o[j] = f2bf(acc[j]);
  *(v8s*)(out + (size_t)t * H + h0) = o;
}
__global__ void k_atomic_f32(const int* tok, const float* wr, float* out) {  // MODE 2
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int r = (int)(i / (H / VPT)), h0 = (int)(i % (H / VPT)) * VPT;
  float* o = out + (size_t)tok[r] * H + h0;
  const float w = wr[r];
  for (int j = 0; j < VPT; ++j) unsafeAtomicAdd(o + j, w * val(r, h0 + j));
}
__global__ void k_atomic_bf16(const int* tok, const float* wr, short* out) {  // MODE 3
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int r = (int)(i / (H / VPT)), h0 = (int)(i % (H / VPT)) * VPT;
  short* o = out + (size_t)tok[r] * H + h0;
  const float w = wr[r];
  for (int j = 0; j < VPT; j += 2) {
    v2s p = {f2bf(w * val(r, h0 + j)), f2bf(w * val(r, h0 + j + 1))};
    __builtin_amdgcn_global_atomic_fadd_v2bf16((__attribute__((address_space(1))) v2s*)(o + j), p);
  }
}

template <typename F>
float timeit(F f) {
  hipEvent_t a, b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
  for (int i = 0; i < 3; ++i) f();
  CHECK(hipDeviceSynchronize());
  const int reps = 10;
  CHECK(hipEventRecord(a));
  for (int i = 0; i < reps; ++i) f();
  CHECK(hipEventRecord(b)); CHECK(hipEventSynchronize(b));
  float ms; CHECK(hipEventElapsedTime(&ms, a, b));
  return ms / reps;
}

int main() {
  // routing: each token 16 distinct experts of 512; rows grouped by expert (stage-2 order)
  std::mt19937 rng(7);
  std::vector<std::vector<int>> by_e(512);
  for (int t = 0; t < T; ++t) {
    std::vector<int> es;
    while ((int)es.size() < TOPK) { int e = rng() % 512; if (std::find(es.begin(), es.end(), e) == es.end()) es.push_back(e); }
    for (int e : es) by_e[e].push_back(t);
  }
  std::vector<int> tok(ROWS), rows_of(ROWS), cnt(T, 0);
  int r = 0;
  for (int e = 0; e < 512; ++e) for (int t : by_e[e]) { tok[r] = t; rows_of[t * TOPK + cnt[t]++] = r; ++r; }
  std::vector<float> w(ROWS, 1.f / TOPK);
  int *d_tok, *d_rows; float* d_w; short *grouped, *out16; float* out32;
  CHECK(hipMalloc(&d_tok, ROWS * 4)); CHECK(hipMalloc(&d_rows, ROWS * 4)); CHECK(hipMalloc(&d_w, ROWS * 4));
  CHECK(hipMemcpy(d_tok, tok.data(), ROWS * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_rows, rows_of.data(), ROWS * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_w, w.data(), ROWS * 4, hipMemcpyHostToDevice));
  CHECK(hipMalloc(&grouped, (size_t)ROWS * H * 2)); CHECK(hipMalloc(&out16, (size_t)T * H * 2)); CHECK(hipMalloc(&out32, (size_t)T * H * 4));
  const size_t nr = (size_t)ROWS * H / VPT, nt = (size_t)T * H / VPT;
  const float m0 = timeit([&] { k_write<<<nr / 256, 256>>>(grouped); });
  const float m1 = timeit([&] { k_reduce<<<nt / 256, 256>>>(grouped, d_rows, d_w, out16); });
  const float mz32 = timeit([&] { CHECK(hipMemsetAsync(out32, 0, (size_t)T * H * 4)); });
  const float m2 = timeit([&] { k_atomic_f32<<<nr / 256, 256>>>(d_tok, d_w, out32); });
  const float mz16 = timeit([&] { CHECK(hipMemsetAsync(out16, 0, (size_t)T * H * 2)); });
  const float m3 = timeit([&] { k_atomic_bf16<<<nr / 256, 256>>>(d_tok, d_w, out16); });
  printf("today:   per-route bf16 write %.3f ms + gather-reduce %.3f ms = %.3f ms\n", m0, m1, m0 + m1);
  printf("fused:   fp32 atomics %.3f ms (+ zero %.3f ms, + fp32->bf16 convert not counted)\n", m2, mz32);
  printf("fused:   pk-bf16 atomics %.3f ms (+ zero %.3f ms)\n", m3, mz16);
  return 0;
}
