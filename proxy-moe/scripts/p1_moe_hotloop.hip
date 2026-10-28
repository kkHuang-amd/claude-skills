// P1 hot-loop prototype for design D1 (MOE_KERNEL_DESIGN_h13-17_1028.md): EP8 gate-up shape, fp4 x fp4 with e8m0 scales.
// Problem: E=64 experts x 256 rows, N=8192, K=16384 (= one EP8 rank, T=8192). No routing, fragment-native layouts.
// WG = 128 token rows x 512 weight cols, 4 waves; wave = 128 rows x 128 cols = 8 token frags x 4 weight frags = 32 WMMA / k128.
// Weights (and weight scales) go global -> VGPR, optionally multicast (cluster_load_b128 / _b32) to the 2 WGs of a cluster
// (the two 128-row M-tiles of one expert). Tokens (+ scales) go global -> LDS (async b128/b64, NS stages) -> VGPR.
// Correctness: a reference kernel with the same WMMA order but plain global loads must give bit-identical fp32 output.
// Build: hipcc --offload-arch=gfx1250 -O3 -o p1_moe_hotloop p1_moe_hotloop.hip ; Run: ./p1_moe_hotloop
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

namespace cg = cooperative_groups;
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v16i __attribute__((ext_vector_type(16)));
typedef float v16f __attribute__((ext_vector_type(16)));
#define GAS __attribute__((address_space(1)))
#define LAS __attribute__((address_space(3)))
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

constexpr int E = 64, ROWS = 256, N = 8192, K = 16384;
constexpr int KS = K / 128, NT = N / 512, NG = N / 32, MG = ROWS / 16;
constexpr int NWG = E * NT * 2;                 // 2 M-tiles per expert
constexpr int NS = 3;                           // LDS stages for tokens
constexpr int STAGE = 8 * 1024 + 8 * 128;       // 8 token frags (1 KB each) + 8 x 128 B scales

// fragment-native layouts (bytes / ints)
__host__ __device__ inline size_t w_off(int e, int ng, int ks) { return (((size_t)e * NG + ng) * KS + ks) * 2048; }   // + lane*64
__host__ __device__ inline size_t ws_off(int e, int ng, int ks) { return (((size_t)e * NG + ng) * KS + ks) * 32; }   // ints, + lane
__host__ __device__ inline size_t x_off(int e, int mg, int ks) { return (((size_t)e * MG + mg) * KS + ks) * 1024; }  // + lane*32
__host__ __device__ inline size_t xs_off(int e, int mg, int ks) { return (((size_t)e * MG + mg) * KS + ks) * 32; }   // ints, + lane

__device__ inline v16f wmma(v16i a, v8i b, v16f c, int sa, int sb) {
  return __builtin_amdgcn_wmma_scale_f32_32x16x128_f4(a, b, (short)0, c, 0, 0, sa, 0, 0, sb, false, false);
}

template <int MC>
__device__ inline void load_w(const char* W, const int* WS, int e, int ng0, int ks, int lane, v16i (&w)[4], int (&s)[4]) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const v4i* p = (const v4i*)(W + w_off(e, ng0 + j, ks) + lane * 16);   // piece u at + u*512 (coalesced)
    const int* q = WS + ws_off(e, ng0 + j, ks) + lane;
    v4i r[4];
    if constexpr (MC) {
#pragma unroll
      for (int u = 0; u < 4; ++u) r[u] = __builtin_amdgcn_cluster_load_b128((GAS v4i*)(p + u * 32), 0, 3u);
      s[j] = __builtin_amdgcn_cluster_load_b32((GAS int*)q, 0, 3u);
    } else {
#pragma unroll
      for (int u = 0; u < 4; ++u) r[u] = p[u * 32];
      s[j] = *q;
    }
#pragma unroll
    for (int u = 0; u < 4; ++u)
#pragma unroll
      for (int t = 0; t < 4; ++t) w[j][u * 4 + t] = r[u][t];
  }
}

// each thread: 4 x b128 token chunks + 1 x b64 scale chunk per stage (5 async instructions per wave per stage)
__device__ inline void issue_tokens(const char* X, const int* XS, int e, int mt, int ks, char* lds_stage, int tid) {
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    const int c = tid * 4 + u, i = c / 64, off = (c % 64) * 16;
    __builtin_amdgcn_global_load_async_to_lds_b128((GAS v4i*)(X + x_off(e, mt * 8 + i, ks) + off),
                                                   (LAS v4i*)(lds_stage + i * 1024 + off), 0, 0);
  }
  const int i = tid / 16, off = (tid % 16) * 8;
  __builtin_amdgcn_global_load_async_to_lds_b64((GAS v2i*)((const char*)(XS + xs_off(e, mt * 8 + i, ks)) + off),
                                                (LAS v2i*)(lds_stage + 8192 + i * 128 + off), 0, 0);
}

__device__ inline void load_tok(const char* st, int lane, v8i (&tok)[8], int (&ts)[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const v4i* tp = (const v4i*)(st + i * 1024 + lane * 16);
    const v4i t0 = tp[0], t1 = tp[32];
    tok[i] = v8i{t0.x, t0.y, t0.z, t0.w, t1.x, t1.y, t1.z, t1.w};
    ts[i] = *(const int*)(st + 8192 + i * 128 + lane * 4);
  }
}

__device__ inline void mma_step(const v8i (&tok)[8], const int (&ts)[8], const v16i (&w)[4], const int (&ws)[4], v16f (&acc)[32]) {
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i * 4 + j] = wmma(w[j], tok[i], acc[i * 4 + j], ws[j], ts[i]);
}

__device__ inline void store_acc(float* out, int wg, int wave, int lane, const v16f (&acc)[32]) {
#pragma unroll
  for (int a = 0; a < 32; ++a)
#pragma unroll
    for (int el = 0; el < 16; ++el) out[((((size_t)wg * 4 + wave) * 32 + a) * 16 + el) * 32 + lane] = acc[a][el];
}

// MC: weight multicast to the 2-WG cluster; CB: cluster barrier every CB k-steps (0 = none); PD: weight prefetch depth (k-steps)
// ABL bit0: skip weight loads in the loop; bit1: skip token loads in the loop (timing only, output then mismatches)
// TDB: 1 = token VGPR double buffer (v4), 0 = read stage k then compute (v3)
template <int MC, int CB, int PD, int ABL = 0, int TDB = 1>
__global__ __launch_bounds__(128) __attribute__((amdgpu_waves_per_eu(1, 1)))
void hot(const char* W, const int* WS, const char* X, const int* XS, float* out, unsigned long long* t) {
  constexpr int NB = PD + 1;                                        // weight register buffers
  __shared__ __attribute__((aligned(16))) char lds[NS * STAGE];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int pair = blockIdx.y >> 1;
  const int mt = MC ? (int)cg::this_cluster().block_rank() : (int)(blockIdx.y & 1);
  const int e = pair / NT, nt = pair % NT, ng0 = nt * 16 + wave * 4;
  v16f acc[32];
#pragma unroll
  for (int a = 0; a < 32; ++a) acc[a] = v16f{};
  for (int s = 0; s < NS - 1; ++s) issue_tokens(X, XS, e, mt, s, lds + s * STAGE, tid);
  v16i w[NB][4];
  int sw[NB][4];
#pragma unroll
  for (int p = 0; p < PD; ++p) load_w<MC>(W, WS, e, ng0, p, lane, w[p], sw[p]);
  if constexpr (MC) { cg::this_cluster().barrier_arrive(); cg::this_cluster().barrier_wait(cg::cluster_group::arrival_token{}); }
  // token frags double-buffered in VGPRs: step k computes with tk[k&1] while reading stage k+1 into tk[(k+1)&1]
  static_assert(NB % 2 == 0, "NB must be even for the token ping-pong");
  v8i tk[2][8];
  int tsc[2][8];
  if constexpr (TDB) {
    if constexpr (ABL & 2) __builtin_amdgcn_s_wait_asynccnt(0); else __builtin_amdgcn_s_wait_asynccnt(5 * (NS - 2));
    __builtin_amdgcn_s_barrier();
    load_tok(lds, lane, tk[0], tsc[0]);
  }
  unsigned long long c0 = clock64();
  for (int ks = 0; ks < KS; ks += NB) {
#pragma unroll
    for (int h = 0; h < NB; ++h) {
      const int k = ks + h;
      if (k < KS) {
        __builtin_amdgcn_s_barrier();                               // stage (k-1)%NS free (no fence: keeps loads in flight)
        if constexpr (ABL & 2) {} else if (k + NS - 1 < KS) issue_tokens(X, XS, e, mt, k + NS - 1, lds + ((k + NS - 1) % NS) * STAGE, tid);
        else issue_tokens(X, XS, e, mt, KS - 1, lds + ((k + NS - 1) % NS) * STAGE, tid);  // dummy, keeps asynccnt uniform
        if (!(ABL & 1) && k + PD < KS) load_w<MC>(W, WS, e, ng0, k + PD, lane, w[(h + PD) % NB], sw[(h + PD) % NB]);
        if constexpr (TDB) {
          if constexpr (ABL & 2) __builtin_amdgcn_s_wait_asynccnt(0); else __builtin_amdgcn_s_wait_asynccnt(5 * (NS - 2));  // stage k+1 (this wave)
          __builtin_amdgcn_s_barrier();                             // ... and every wave's
          if (k + 1 < KS) load_tok(lds + ((k + 1) % NS) * STAGE, lane, tk[(h + 1) & 1], tsc[(h + 1) & 1]);
          mma_step(tk[h & 1], tsc[h & 1], w[h], sw[h], acc);
        } else {
          if constexpr (ABL & 2) __builtin_amdgcn_s_wait_asynccnt(0); else __builtin_amdgcn_s_wait_asynccnt(5 * (NS - 1));  // stage k (this wave)
          __builtin_amdgcn_s_barrier();
          load_tok(lds + (k % NS) * STAGE, lane, tk[0], tsc[0]);
          mma_step(tk[0], tsc[0], w[h], sw[h], acc);
        }
        if constexpr (MC && CB > 0) {
          if ((k + 1) % CB == 0) { cg::this_cluster().barrier_arrive(); cg::this_cluster().barrier_wait(cg::cluster_group::arrival_token{}); }
        }
      }
    }
  }
  unsigned long long c1 = clock64();
  __builtin_amdgcn_s_wait_asynccnt(0);
  store_acc(out, blockIdx.y, wave, lane, acc);
  if (lane == 0) t[blockIdx.y * 4 + wave] = c1 - c0;
}

// reference: same mapping and WMMA order, all operands straight from global
__global__ __launch_bounds__(128) __attribute__((amdgpu_waves_per_eu(1, 1)))
void ref(const char* W, const int* WS, const char* X, const int* XS, float* out) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int pair = blockIdx.y >> 1, mt = blockIdx.y & 1, e = pair / NT, nt = pair % NT, ng0 = nt * 16 + wave * 4;
  v16f acc[32];
  for (int a = 0; a < 32; ++a) acc[a] = v16f{};
  for (int ks = 0; ks < KS; ++ks) {
    v16i w[4]; int s[4];
    load_w<0>(W, WS, e, ng0, ks, lane, w, s);
    for (int i = 0; i < 8; ++i) {
      const v4i* tp = (const v4i*)(X + x_off(e, mt * 8 + i, ks) + lane * 16);
      const v4i t0 = tp[0], t1 = tp[32];
      const v8i tok = {t0.x, t0.y, t0.z, t0.w, t1.x, t1.y, t1.z, t1.w};
      const int ts = XS[xs_off(e, mt * 8 + i, ks) + lane];
      for (int j = 0; j < 4; ++j) acc[i * 4 + j] = wmma(w[j], tok, acc[i * 4 + j], s[j], ts);
    }
  }
  store_acc(out, blockIdx.y, wave, lane, acc);
}

__global__ void init_bits(unsigned* p, size_t n, unsigned seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    unsigned h = (unsigned)i * 2654435761u ^ seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; p[i] = h;
  }
}
__global__ void init_scales(unsigned* p, size_t n, unsigned seed) {  // e8m0 bytes in 0x7c..0x83 (2^-3 .. 2^4)
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    unsigned h = (unsigned)i * 2654435761u ^ seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    p[i] = ((h & 0x07070707u) + 0x7c7c7c7cu);
  }
}
__global__ void cmp(const unsigned* a, const unsigned* b, size_t n, unsigned long long* bad) {
  unsigned long long c = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) c += (a[i] != b[i]);
  if (c) atomicAdd(bad, c);
}

static bool g_att = false;  // ATT mode: one launch, no reps, no compare
template <int MC, int CB, int PD, int ABL = 0, int TDB = 1>
void bench(const char* name, char* W, int* WS, char* X, int* XS, float* out, const float* outref, unsigned long long* d_t) {
  hipLaunchAttribute attr{};
  attr.id = hipLaunchAttributeClusterDimension;
  attr.val.clusterDim = {1, 2, 1};
  hipLaunchConfig_t cfg{};
  cfg.gridDim = {1, (unsigned)NWG, 1};
  cfg.blockDim = {128, 1, 1};
  cfg.attrs = &attr;
  cfg.numAttrs = MC ? 1 : 0;
  hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
  CHECK(hipLaunchKernelEx(&cfg, hot<MC, CB, PD, ABL, TDB>, (const char*)W, (const int*)WS, (const char*)X, (const int*)XS, out, d_t));
  CHECK(hipDeviceSynchronize());
  if (g_att) { printf("%s: single launch (ATT mode)\n", name); return; }
  const int reps = 3;
  CHECK(hipEventRecord(e0));
  for (int r = 0; r < reps; ++r) CHECK(hipLaunchKernelEx(&cfg, hot<MC, CB, PD, ABL, TDB>, (const char*)W, (const int*)WS, (const char*)X, (const int*)XS, out, d_t));
  CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
  float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); ms /= reps;
  const size_t nout = (size_t)NWG * 4 * 32 * 16 * 32;
  unsigned long long* d_bad; CHECK(hipMalloc(&d_bad, 8)); CHECK(hipMemset(d_bad, 0, 8));
  cmp<<<4096, 256>>>((const unsigned*)out, (const unsigned*)outref, nout, d_bad);
  unsigned long long bad; CHECK(hipMemcpy(&bad, d_bad, 8, hipMemcpyDeviceToHost)); CHECK(hipFree(d_bad));
  std::vector<unsigned long long> t(NWG * 4);
  CHECK(hipMemcpy(t.data(), d_t, t.size() * 8, hipMemcpyDeviceToHost));
  double sum = 0; for (auto v : t) sum += v;
  std::sort(t.begin(), t.end());
  const double med = (double)t[t.size() / 2] / KS;
  // per-CU busy cycles if 1 WG/CU at a time: sum over WGs of loop cycles / 96 CUs
  const double cu_mcyc = sum / 4 / 96 / 1e6;
  const double tflop = 2.0 * E * ROWS * (double)N * K / 1e12;
  printf("%-22s | %s (%llu bad) | cyc/k-step/wave med %.0f [%.0f..%.0f] -> %.2f cyc/WMMA/SIMD | per-CU busy %.2f Mcyc | %.3f ms  %.0f TFLOP/s\n",
         name, bad ? "MISMATCH" : "bit-exact", bad, med, (double)t[0] / KS, (double)t.back() / KS, med / 32, cu_mcyc, ms, tflop / (ms * 1e-3));
}

int main(int argc, char** argv) {
  const int which = argc > 1 ? atoi(argv[1]) : -1;  // run one variant only (see list below); argv[2] == "att": ATT mode
  g_att = argc > 2 && argv[2][0] == 'a';
  const size_t wb = (size_t)E * N * K / 2, wsn = (size_t)E * NG * KS * 32, xb = (size_t)E * ROWS * K / 2, xsn = (size_t)E * MG * KS * 32;
  const size_t nout = (size_t)NWG * 4 * 32 * 16 * 32;
  char *W, *X; int *WS, *XS; float *out, *outref; unsigned long long* d_t;
  CHECK(hipMalloc(&W, wb)); CHECK(hipMalloc(&WS, wsn * 4)); CHECK(hipMalloc(&X, xb)); CHECK(hipMalloc(&XS, xsn * 4));
  CHECK(hipMalloc(&out, nout * 4)); CHECK(hipMalloc(&outref, nout * 4)); CHECK(hipMalloc(&d_t, NWG * 4 * 8));
  init_bits<<<4096, 256>>>((unsigned*)W, wb / 4, 1u); init_bits<<<4096, 256>>>((unsigned*)X, xb / 4, 2u);
  init_scales<<<4096, 256>>>((unsigned*)WS, wsn, 3u); init_scales<<<4096, 256>>>((unsigned*)XS, xsn, 4u);
  CHECK(hipDeviceSynchronize());
  printf("W %.2f GB, X %.0f MB, out %.0f MB, %d WGs\n", wb / 1e9, xb / 1e6, nout * 4 / 1e6, NWG);
  if (!g_att) ref<<<dim3(1, NWG), 128>>>(W, WS, X, XS, outref);
  CHECK(hipDeviceSynchronize());
  if (which < 0 || which == 0) bench<0, 0, 3, 0, 0>("v3 plain PD3", W, WS, X, XS, out, outref, d_t);
  if (which < 0 || which == 1) bench<0, 0, 3, 0, 1>("v4 plain PD3 tokDB", W, WS, X, XS, out, outref, d_t);
  if (which < 0 || which == 2) bench<1, 0, 3, 0, 0>("v3 mcast PD3", W, WS, X, XS, out, outref, d_t);
  if (which < 0 || which == 3) bench<0, 0, 3, 3, 0>("v3 plain -both", W, WS, X, XS, out, outref, d_t);
  if (which < 0 || which == 4) bench<0, 0, 3, 1, 0>("v3 plain -weights", W, WS, X, XS, out, outref, d_t);
  return 0;
}
