// P1 hot-loop prototype for design D1 (MOE_KERNEL_DESIGN_h13-17_1028.md): EP8 gate-up shape, fp4 x fp4 with e8m0 scales.
// Problem: E=64 experts x 256 rows, N=8192, K=16384 (= one EP8 rank, T=8192). No routing, fragment-native layouts.
// WG = 128 token rows x (4 waves x CW x 32) weight cols; wave = 128 rows x CW*32 cols = 8 token frags x CW weight frags.
//   CW=4: 128x128 per wave, 32 WMMA / k128, acc 512 VGPR, 1 wave/SIMD (P1 v3-v5)
//   CW=2: 128x64 per wave, 16 WMMA / k128, acc 256 VGPR, 2 waves/SIMD (2 WGs per CU) (P1 v6)
// Weights (and weight scales) go global -> VGPR, optionally multicast (cluster_load_b128 / _b32) to the 2 WGs of a cluster
// (the two 128-row M-tiles of one expert). Tokens (+ scales) go global -> LDS (async b128/b64, NS stages) -> VGPR.
// Correctness: a reference kernel with the same WMMA order but plain global loads must give bit-identical fp32 output.
// Build: hipcc --offload-arch=gfx1250 -O3 -o p1_moe_hotloop p1_moe_hotloop.hip
// Run:   ./p1_moe_hotloop [variant] [att]   (no args: all variants; "att": one launch, no reference / compare)
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
// Token capacity per expert: MG = 24 groups of 16 rows = 384 rows (routing gives ~256 +- 16 rows per expert at T=8192, top-2 of 64).
// Which M-tiles exist comes from a host-built tile table {expert, m-tile} (like aiter's m_tile_map), so uneven routing
// produces 2 or 3 M-tiles per expert and partially filled last tiles, as in the real grouped GEMM.
constexpr int KS = K / 128, NG = N / 32, MG = 24, MAXTILES = E * 3;
constexpr int STAGE = 8 * 1024 + 8 * 128;       // 8 token frags (1 KB each) + 8 x 128 B scales
// WG = MW x (4/MW) waves: MW waves split the 128 token rows, 4/MW waves split the weight columns (CW frags each)
constexpr int nt_of(int cw, int mw = 1) { return N / ((4 / mw) * cw * 32); }  // weight-column tiles per expert
// (grid = ntiles x nt_of(cw, mw) WGs)

// fragment-native layouts (bytes / ints); a b128 piece u of a lane sits at u*512 + lane*16 (coalesced per instruction)
__host__ __device__ inline size_t w_off(int e, int ng, int ks) { return (((size_t)e * NG + ng) * KS + ks) * 2048; }
__host__ __device__ inline size_t ws_off(int e, int ng, int ks) { return (((size_t)e * NG + ng) * KS + ks) * 32; }   // ints, + lane
__host__ __device__ inline size_t x_off(int e, int mg, int ks) { return (((size_t)e * MG + mg) * KS + ks) * 1024; }
__host__ __device__ inline size_t xs_off(int e, int mg, int ks) { return (((size_t)e * MG + mg) * KS + ks) * 32; }   // ints, + lane

__device__ inline v16f wmma(v16i a, v8i b, v16f c, int sa, int sb) {
  return __builtin_amdgcn_wmma_scale_f32_32x16x128_f4(a, b, (short)0, c, 0, 0, sa, 0, 0, sb, false, false);
}

// NOSC: do not load the weight scales (timing ablation; keeps the previous values)
template <int MC, int CW, int NOSC = 0>
__device__ inline void load_w(const char* W, const int* WS, int e, int ng0, int ks, int lane, v16i (&w)[CW], int (&s)[CW]) {
#pragma unroll
  for (int j = 0; j < CW; ++j) {
    const v4i* p = (const v4i*)(W + w_off(e, ng0 + j, ks) + lane * 16);
    const int* q = WS + ws_off(e, ng0 + j, ks) + lane;
    v4i r[4];
    if constexpr (MC) {
#pragma unroll
      for (int u = 0; u < 4; ++u) r[u] = __builtin_amdgcn_cluster_load_b128((GAS v4i*)(p + u * 32), 0, 3u);
      if constexpr (!NOSC) s[j] = __builtin_amdgcn_cluster_load_b32((GAS int*)q, 0, 3u);
    } else {
#pragma unroll
      for (int u = 0; u < 4; ++u) r[u] = p[u * 32];
      if constexpr (!NOSC) s[j] = *q;
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

// ---- TDM (tensor_load_to_lds), 2D, 4-byte elements. Field layout verified by tdmprobe.hip (2026-10-28):
// g0 = {count=1, lds_addr, gaddr_lo, gaddr_hi[24:0] | type=2 << 30}; g1[0] = wg_mask | data_size(2 = 4 B) << 16;
// tensor_dim0 = row width (elements), tensor_dim1 = rows, tile_dim0/1 = tile width/rows, tensor_dim0_stride = row pitch (elements).
typedef unsigned u4 __attribute__((ext_vector_type(4)));
typedef int i4t __attribute__((ext_vector_type(4)));
__device__ inline v8i tdm_g1(unsigned dim0, unsigned dim1, unsigned tile0, unsigned tile1, unsigned pitch) {
  v8i g1;
  g1[0] = (int)(2u << 16);
  g1[1] = (int)((dim0 & 0xffff) << 16);
  g1[2] = (int)((dim0 >> 16) | ((dim1 & 0xffff) << 16));
  g1[3] = (int)((dim1 >> 16) | (tile0 << 16));
  g1[4] = (int)tile1;
  g1[5] = (int)pitch;
  g1[6] = 0;
  g1[7] = 0;
  return g1;
}
__device__ inline void tdm_load(const void* gsrc, void* lds_dst, v8i g1) {
  const unsigned long long ga = (unsigned long long)gsrc;
  u4 g0;
  g0.x = 1;
  g0.y = (unsigned)(unsigned long long)(LAS char*)lds_dst;
  g0.z = (unsigned)ga;
  g0.w = ((unsigned)(ga >> 32) & 0x1ffffff) | (2u << 30);
  i4t z4 = {};
  v8i z8 = {};
  __builtin_amdgcn_tensor_load_to_lds(g0, g1, z4, z4, z8, 0);
}
// One stage of tokens via TDM, issued by wave 0 only: 8 rows (token groups) x 1 KB at pitch KS*1 KB, and 8 x 128 B scales.
__device__ inline void issue_tokens_tdm(const char* X, const int* XS, int e, int mt, int ks, char* lds_stage, v8i gx, v8i gs) {
  tdm_load(X + x_off(e, mt * 8, ks), lds_stage, gx);
  tdm_load(XS + xs_off(e, mt * 8, ks), lds_stage + 8192, gs);
}

template <int TF>
__device__ inline void load_tok(const char* st, int lane, int i0, v8i (&tok)[TF], int (&ts)[TF]) {
#pragma unroll
  for (int i = 0; i < TF; ++i) {
    const v4i* tp = (const v4i*)(st + (i0 + i) * 1024 + lane * 16);
    const v4i t0 = tp[0], t1 = tp[32];
    tok[i] = v8i{t0.x, t0.y, t0.z, t0.w, t1.x, t1.y, t1.z, t1.w};
    ts[i] = *(const int*)(st + 8192 + (i0 + i) * 128 + lane * 4);
  }
}

// Barrier that the compiler cannot move memory operations across (s_barrier alone is not a memory op for LLVM).
__device__ inline void bar() { asm volatile("" ::: "memory"); __builtin_amdgcn_s_barrier(); asm volatile("" ::: "memory"); }
// Retire this wave's LDS reads before other waves may overwrite the stage (same race class as the aiter s_wait_dscnt fix).
__device__ inline void wait_ds() { asm volatile("s_wait_dscnt 0x0" ::: "memory"); }

// L2 prefetch of one k-step of this wave's weight fragments (+ scales): one global_prefetch_b8 per 128 B line, no VGPR cost.
template <int CW>
__device__ inline void prefetch_w(const char* W, const int* WS, int e, int ng0, int ks, int lane) {
#pragma unroll
  for (int q = lane; q < CW * 16; q += 32)
    __builtin_amdgcn_global_prefetch((GAS const void*)(W + w_off(e, ng0 + q / 16, ks) + (q % 16) * 128), 0);
  if (lane < CW) __builtin_amdgcn_global_prefetch((GAS const void*)(WS + ws_off(e, ng0 + lane, ks)), 0);
}

// WMMAs for token frags [I0, I1) of one k-step
template <int CW, int TF, int I0, int I1>
__device__ inline void mma_range(const v8i (&tok)[TF], const int (&ts)[TF], const v16i (&w)[CW], const int (&ws)[CW], v16f (&acc)[TF * CW]) {
#pragma unroll
  for (int i = I0; i < I1; ++i)
#pragma unroll
    for (int j = 0; j < CW; ++j) acc[i * CW + j] = wmma(w[j], tok[i], acc[i * CW + j], ws[j], ts[i]);
}

template <int CW, int TF>
__device__ inline void mma_step(const v8i (&tok)[TF], const int (&ts)[TF], const v16i (&w)[CW], const int (&ws)[CW], v16f (&acc)[TF * CW]) {
#pragma unroll
  for (int i = 0; i < TF; ++i)
#pragma unroll
    for (int j = 0; j < CW; ++j) acc[i * CW + j] = wmma(w[j], tok[i], acc[i * CW + j], ws[j], ts[i]);
}

template <int NA>
__device__ inline void store_acc(float* out, int wg, int wave, int lane, const v16f (&acc)[NA]) {
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int el = 0; el < 16; ++el) out[((((size_t)wg * 4 + wave) * NA + a) * 16 + el) * 32 + lane] = acc[a][el];
}

// Output modes. OUTM 0: fp32 accumulators (bit-exact check). 1: bf16 (truncate). 2: today-like gate-up epilogue:
// out = silu(gate) * up with weight frags j < CW/2 = gate, j >= CW/2 = up, then e2m1 fp4 (software, scale 1) -> 1/16 of fp32 bytes.
__device__ inline unsigned fp4_code(float x) {  // e2m1 magnitude levels 0, .5, 1, 1.5, 2, 3, 4, 6 (round to nearest)
  const float a = fabsf(x);
  unsigned c = (a > 0.25f) + (a > 0.75f) + (a > 1.25f) + (a > 1.75f) + (a > 2.5f) + (a > 3.5f) + (a > 5.0f);
  return c | (x < 0.f ? 8u : 0u);
}
template <int OUTM, int NA, int CW>
__device__ inline void store_out(float* out, int wg, int wave, int lane, const v16f (&acc)[NA], size_t scale_off = 0) {
  if constexpr (OUTM == 0) {
    store_acc<NA>(out, wg, wave, lane, acc);
  } else if constexpr (OUTM == 1) {
    unsigned short* o = (unsigned short*)out;
#pragma unroll
    for (int a = 0; a < NA; ++a)
#pragma unroll
      for (int el = 0; el < 16; ++el)
        o[((((size_t)wg * 4 + wave) * NA + a) * 16 + el) * 32 + lane] = (unsigned short)(__float_as_uint(acc[a][el]) >> 16);
  } else if constexpr (OUTM == 3) {
    // today-like epilogue with real MX quant: v = silu(gate) * up; per 32 output values (16 in this lane + 16 in lane ^ 16)
    // amax -> e8m0 scale = ceil(log2(amax / 6)); fp4 = e2m1(v / scale); write 8 B of fp4 per lane and 1 scale byte per pair.
    constexpr int TF = NA / CW, HP = CW / 2;
    unsigned* o = (unsigned*)out;
    unsigned char* so = (unsigned char*)out + scale_off;
#pragma unroll
    for (int i = 0; i < TF; ++i)
#pragma unroll
      for (int pj = 0; pj < HP; ++pj) {
        float v[16];
        float amax = 0.f;
#pragma unroll
        for (int el = 0; el < 16; ++el) {
          const float g = acc[i * CW + pj][el], u = acc[i * CW + pj + HP][el];
          v[el] = g * __builtin_amdgcn_rcpf(1.f + __builtin_amdgcn_exp2f(-1.4426950409f * g)) * u;
          amax = fmaxf(amax, fabsf(v[el]));
        }
        amax = fmaxf(amax, __shfl_xor(amax, 16));
        const unsigned bits = __float_as_uint(amax * (1.f / 6.f));
        unsigned ex = (bits >> 23) & 0xff;
        ex += (bits & 0x7fffff) ? 1u : 0u;
        ex = ex < 1u ? 1u : (ex > 253u ? 253u : ex);
        const float inv = __uint_as_float((254u - ex) << 23);  // 2^(127 - ex)
        unsigned w0 = 0, w1 = 0;
#pragma unroll
        for (int el = 0; el < 16; ++el) {
          const unsigned c = fp4_code(v[el] * inv);
          if (el < 8) w0 |= c << (4 * el); else w1 |= c << (4 * (el - 8));
        }
        const size_t pidx = (((size_t)wg * 4 + wave) * (TF * HP) + i * HP + pj);
        o[(pidx * 32 + lane) * 2] = w0;
        o[(pidx * 32 + lane) * 2 + 1] = w1;
        if (lane < 16) so[pidx * 16 + lane] = (unsigned char)ex;
      }
  } else {
    constexpr int TF = NA / CW, HP = CW / 2;  // pairs (gate j, up j + HP)
    unsigned* o = (unsigned*)out;
#pragma unroll
    for (int i = 0; i < TF; ++i)
#pragma unroll
      for (int pj = 0; pj < HP; ++pj) {
        unsigned w0 = 0, w1 = 0;
#pragma unroll
        for (int el = 0; el < 16; ++el) {
          const float g = acc[i * CW + pj][el], u = acc[i * CW + pj + HP][el];
          // silu via hardware v_exp_f32 / v_rcp_f32 (an IEEE divide + expf made this epilogue ~10x slower)
          const unsigned c = fp4_code(g * __builtin_amdgcn_rcpf(1.f + __builtin_amdgcn_exp2f(-1.4426950409f * g)) * u);
          if (el < 8) w0 |= c << (4 * el); else w1 |= c << (4 * (el - 8));
        }
        const size_t idx = ((((size_t)wg * 4 + wave) * (TF * HP) + i * HP + pj) * 32 + lane) * 2;
        o[idx] = w0;
        o[idx + 1] = w1;
      }
  }
}

// CW: weight frags per wave; WPE: waves per SIMD the compiler budgets VGPRs for (1 -> ~1000 VGPR, 2 -> <= 512)
// MC: weight multicast to the 2-WG cluster; CB: cluster barrier every CB k-steps (0 = none); PD: weight prefetch depth (k-steps)
// ABL bit0: skip weight loads in the loop; bit1: skip token loads; bit2: skip weight-scale loads (timing only, output then mismatches)
// TDB: 1 = token VGPR double buffer (read stage k+1 while computing k), 0 = read stage k then compute
// NS: token LDS stages; ONEBAR: drop the "stage free" barrier (safe only with TDB and NS >= 5); PFD: L2 prefetch distance (0 = off)
// MW: waves along M (1: 4 waves split N, each 128 rows; 2: 2x2, each 64 rows x CW frags); TOK: 0 = async per-lane loads, 1 = TDM
// SPLIT: one split barrier per step (requires TDB): dscnt 0 -> stage k+1 landed -> signal -> first half of WMMAs -> wait ->
//        read stage k+1 -> second half. Stage reuse is safe because every wave retired its reads before signalling.
template <int CW, int MW, int WPE, int MC, int CB, int PD, int ABL, int TDB, int NS, int ONEBAR, int PFD = 0, int TOK = 0, int OUTM = 0, int SPLIT = 0>
__global__ __launch_bounds__(128) __attribute__((amdgpu_waves_per_eu(WPE, WPE)))
void hot(const char* W, const int* WS, const char* X, const int* XS, const int4* tiles, float* out, unsigned long long* t, size_t scale_off) {
  const unsigned long long cs = clock64();
  constexpr int NB = PD + 1;                                        // weight register buffers
  constexpr int UNR = (NB % 2 == 0) ? NB : 2 * NB;                  // unroll: weight index h % NB, token index h & 1
  constexpr int NT = nt_of(CW, MW), TF = 8 / MW;
  static_assert(!ONEBAR || (TDB && NS >= 5), "ONEBAR needs TDB and NS >= 5");
  static_assert(!SPLIT || (TDB && !ONEBAR && NS >= 3), "SPLIT needs TDB, no ONEBAR, NS >= 3");
  static_assert(KS % UNR == 0, "branch-free loop: unroll must divide KS");
  __shared__ __attribute__((aligned(16))) char lds[NS * STAGE];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  static_assert(!MC, "routed tile table: weight multicast between M-tiles not supported");
  const int4 td = tiles[blockIdx.y / NT];                           // {expert, m-tile}
  const int mt = td.y;
  const int wm = wave % MW, wn = wave / MW, i0 = wm * TF;
  const int e = td.x, nt = blockIdx.y % NT, ng0 = nt * (4 / MW) * CW + wn * CW;
  v16f acc[TF * CW];
#pragma unroll
  for (int a = 0; a < TF * CW; ++a) acc[a] = v16f{};
  // token tensor as 2D: rows = (expert, 16-row group), row = KS x 1 KB tokens / KS x 128 B scales
  const v8i gx = tdm_g1(256, 8, 256, 8, KS * 256), gs = tdm_g1(32, 8, 32, 8, KS * 32);
  // per-wave pending count of one stage: async = 5 instructions (asynccnt), TDM = 2 instructions in wave 0 (tensorcnt)
  auto issue_stage = [&](int ks, char* st) {
    if constexpr (TOK) { if (wave == 0) issue_tokens_tdm(X, XS, e, mt, ks, st, gx, gs); }
    else issue_tokens(X, XS, e, mt, ks, st, tid);
  };
  // WAIT_STAGES(keep): this wave's token loads for all but the newest `keep` stages have landed (keep must be a constant)
#define WAIT_STAGES(keep) do { if constexpr (TOK) __builtin_amdgcn_s_wait_tensorcnt(2 * (keep)); else __builtin_amdgcn_s_wait_asynccnt(5 * (keep)); } while (0)
  for (int s = 0; s < NS - 1; ++s) issue_stage(s, lds + s * STAGE);
  v16i w[NB][CW];
  int sw[NB][CW];
#pragma unroll
  for (int p = 0; p < PD; ++p) load_w<MC, CW>(W, WS, e, ng0, p, lane, w[p], sw[p]);
  if constexpr (MC) { cg::this_cluster().barrier_arrive(); cg::this_cluster().barrier_wait(cg::cluster_group::arrival_token{}); }
  v8i tk[TDB ? 2 : 1][TF];
  int tsc[TDB ? 2 : 1][TF];
  if constexpr (TDB) {
    if constexpr (ABL & 2) WAIT_STAGES(0); else WAIT_STAGES(NS - 2);
    bar();
    load_tok<TF>(lds, lane, i0, tk[0], tsc[0]);
  }
  if constexpr (PFD > 0)
    for (int p = PD; p < PFD + PD && p < KS; ++p) prefetch_w<CW>(W, WS, e, ng0, p, lane);
  // Branch-free steady state: every step issues the same loads (indices clamped at the tail), so the waitcnt pass sees one
  // straight-line block and can keep PD steps of weight loads in flight (branches made it wait at every join).
  unsigned long long c0 = clock64();
  for (int ks = 0; ks < KS; ks += UNR) {
#pragma unroll
    for (int h = 0; h < UNR; ++h) {
      const int k = ks + h;
      {
        if constexpr (!SPLIT) wait_ds();                            // my reads of the stage being freed are done
        if constexpr (!ONEBAR && !SPLIT) bar();                               // stage (k-1)%NS free (no fence: keeps loads in flight)
        if constexpr (!(ABL & 2)) issue_stage(min(k + NS - 1, KS - 1), lds + ((k + NS - 1) % NS) * STAGE);
        if constexpr (PFD > 0 && !(ABL & 1)) prefetch_w<CW>(W, WS, e, ng0, min(k + PD + PFD, KS - 1), lane);
        if constexpr (!(ABL & 1)) load_w<MC, CW, (ABL & 4) ? 1 : 0>(W, WS, e, ng0, min(k + PD, KS - 1), lane, w[(h + PD) % NB], sw[(h + PD) % NB]);
        if constexpr (SPLIT) {
          wait_ds();                                                // tokens of step k in VGPRs; my reads of stage k done
          if constexpr (ABL & 2) WAIT_STAGES(0); else WAIT_STAGES(NS - 2);  // stage k+1 landed (TDM issuer)
          asm volatile("" ::: "memory");
          __builtin_amdgcn_s_barrier_signal(-1);
          mma_range<CW, TF, 0, TF / 2>(tk[h & 1], tsc[h & 1], w[h % NB], sw[h % NB], acc);
          __builtin_amdgcn_s_barrier_wait(-1);
          asm volatile("" ::: "memory");
          load_tok<TF>(lds + ((k + 1) % NS) * STAGE, lane, i0, tk[(h + 1) & 1], tsc[(h + 1) & 1]);
          mma_range<CW, TF, TF / 2, TF>(tk[h & 1], tsc[h & 1], w[h % NB], sw[h % NB], acc);
        } else if constexpr (TDB) {
          if constexpr (ABL & 2) WAIT_STAGES(0); else WAIT_STAGES(NS - 2);  // stage k+1 (this wave)
          bar();                                                    // ... and every wave's
          load_tok<TF>(lds + ((k + 1) % NS) * STAGE, lane, i0, tk[(h + 1) & 1], tsc[(h + 1) & 1]);  // last step reads a stale stage, unused
          mma_step<CW, TF>(tk[h & 1], tsc[h & 1], w[h % NB], sw[h % NB], acc);
        } else {
          if constexpr (ABL & 2) WAIT_STAGES(0); else WAIT_STAGES(NS - 1);  // stage k (this wave)
          bar();
          load_tok<TF>(lds + (k % NS) * STAGE, lane, i0, tk[0], tsc[0]);
          mma_step<CW, TF>(tk[0], tsc[0], w[h % NB], sw[h % NB], acc);
        }
        if constexpr (MC && CB > 0) {
          if ((k + 1) % CB == 0) { cg::this_cluster().barrier_arrive(); cg::this_cluster().barrier_wait(cg::cluster_group::arrival_token{}); }
        }
      }
    }
  }
  unsigned long long c1 = clock64();
  WAIT_STAGES(0);
  store_out<OUTM, TF * CW, CW>(out, blockIdx.y, wave, lane, acc, scale_off);
  asm volatile("s_wait_storecnt 0x0" ::: "memory");  // epilogue time includes the stores draining
  const unsigned long long ce = clock64();
  if (lane == 0) { const size_t q = (size_t)(blockIdx.y * 4 + wave) * 4; t[q] = c1 - c0; t[q + 1] = c0 - cs; t[q + 2] = ce - c1; }
}

// reference: same mapping and WMMA order, all operands straight from global
template <int CW, int MW>
__global__ __launch_bounds__(128) __attribute__((amdgpu_waves_per_eu(1, 1)))
void ref(const char* W, const int* WS, const char* X, const int* XS, const int4* tiles, float* out) {
  constexpr int NT = nt_of(CW, MW), TF = 8 / MW;
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, wm = wave % MW, wn = wave / MW, i0 = wm * TF;
  const int4 td = tiles[blockIdx.y / NT];
  const int mt = td.y, e = td.x, nt = blockIdx.y % NT, ng0 = nt * (4 / MW) * CW + wn * CW;
  v16f acc[TF * CW];
  for (int a = 0; a < TF * CW; ++a) acc[a] = v16f{};
  for (int ks = 0; ks < KS; ++ks) {
    v16i w[CW]; int s[CW];
    load_w<0, CW>(W, WS, e, ng0, ks, lane, w, s);
    for (int i = 0; i < TF; ++i) {
      const v4i* tp = (const v4i*)(X + x_off(e, mt * 8 + i0 + i, ks) + lane * 16);
      const v4i t0 = tp[0], t1 = tp[32];
      const v8i tok = {t0.x, t0.y, t0.z, t0.w, t1.x, t1.y, t1.z, t1.w};
      const int ts = XS[xs_off(e, mt * 8 + i0 + i, ks) + lane];
      for (int j = 0; j < CW; ++j) acc[i * CW + j] = wmma(w[j], tok, acc[i * CW + j], s[j], ts);
    }
  }
  store_acc<TF * CW>(out, blockIdx.y, wave, lane, acc);
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

struct Bufs { char* W; int* WS; char* X; int* XS; int4* tiles; int ntiles; float* out; float* outref; unsigned long long* t; };
static bool g_att = false;      // ATT mode: one launch, no reps, no compare
static int g_ref_key = 0;       // (CW, MW, routing) the reference output currently holds
static int g_route_id = 0;
constexpr size_t NOUT_MAX = (size_t)MAXTILES * 16 * 4 * 32 * 16 * 32;  // floats, for the largest tile count

template <int CW, int MW, int WPE, int MC, int CB, int PD, int ABL, int TDB, int NS, int ONEBAR, int PFD = 0, int TOK = 0, int OUTM = 0, int SPLIT = 0>
void bench(const char* name, const Bufs& b) {
  constexpr int TF = 8 / MW;
  const int NWG = b.ntiles * nt_of(CW, MW);
  const size_t NOUT = (size_t)NWG * 4 * TF * CW * 16 * 32;
  const size_t scale_off = NOUT / 16 * 4;  // OUTM 3: scale bytes after the fp4 data
  if (!g_att && OUTM == 0 && g_ref_key != CW * 10 + MW + 100 * g_route_id) {
    ref<CW, MW><<<dim3(1, NWG), 128>>>(b.W, b.WS, b.X, b.XS, b.tiles, b.outref);
    CHECK(hipDeviceSynchronize());
    g_ref_key = CW * 10 + MW + 100 * g_route_id;
  }
  auto kern = hot<CW, MW, WPE, MC, CB, PD, ABL, TDB, NS, ONEBAR, PFD, TOK, OUTM, SPLIT>;
  hipLaunchAttribute attr{};
  attr.id = hipLaunchAttributeClusterDimension;
  attr.val.clusterDim = {1, 2, 1};
  hipLaunchConfig_t cfg{};
  cfg.gridDim = {1, (unsigned)NWG, 1};
  cfg.blockDim = {128, 1, 1};
  cfg.attrs = &attr;
  cfg.numAttrs = MC ? 1 : 0;
  hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
  CHECK(hipLaunchKernelEx(&cfg, kern, (const char*)b.W, (const int*)b.WS, (const char*)b.X, (const int*)b.XS, (const int4*)b.tiles, b.out, b.t, scale_off));
  CHECK(hipDeviceSynchronize());
  if (g_att) { printf("%s: single launch (ATT mode)\n", name); return; }
  // steady-state timing like the op test (it runs 101 iters): 10 warmup + 50 timed launches; 3 reps left clocks unsettled
  for (int r = 0; r < 10; ++r) CHECK(hipLaunchKernelEx(&cfg, kern, (const char*)b.W, (const int*)b.WS, (const char*)b.X, (const int*)b.XS, (const int4*)b.tiles, b.out, b.t, scale_off));
  const int reps = 50;
  CHECK(hipEventRecord(e0));
  for (int r = 0; r < reps; ++r) CHECK(hipLaunchKernelEx(&cfg, kern, (const char*)b.W, (const int*)b.WS, (const char*)b.X, (const int*)b.XS, (const int4*)b.tiles, b.out, b.t, scale_off));
  CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
  float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); ms /= reps;
  unsigned long long bad = 0;
  if (OUTM == 0) {
    unsigned long long* d_bad; CHECK(hipMalloc(&d_bad, 8)); CHECK(hipMemset(d_bad, 0, 8));
    cmp<<<4096, 256>>>((const unsigned*)b.out, (const unsigned*)b.outref, NOUT, d_bad);
    CHECK(hipMemcpy(&bad, d_bad, 8, hipMemcpyDeviceToHost)); CHECK(hipFree(d_bad));
  }
  std::vector<unsigned long long> tt((size_t)NWG * 4 * 4);
  CHECK(hipMemcpy(tt.data(), b.t, tt.size() * 8, hipMemcpyDeviceToHost));
  std::vector<unsigned long long> t(NWG * 4);
  double sum = 0, spro = 0, sepi = 0;
  for (size_t w = 0; w < t.size(); ++w) { t[w] = tt[w * 4]; sum += tt[w * 4]; spro += tt[w * 4 + 1]; sepi += tt[w * 4 + 2]; }
  std::sort(t.begin(), t.end());
  const double med = (double)t[t.size() / 2] / KS;
  const double den = 96.0 * 4 * WPE * 1e6;
  const double cu_mcyc = sum / den;
  const double tflop = 2.0 * E * ROWS * (double)N * K / 1e12;
  printf("%-26s | %s | cyc/step med %.0f | per-CU Mcyc: loop %.2f pro %.2f epi %.2f | %.3f ms  %.0f TFLOP/s\n",
         name, OUTM ? "no check " : (bad ? "MISMATCH " : "bit-exact"), med, cu_mcyc, spro / den, sepi / den, ms, tflop / (ms * 1e-3));
}

#include <random>
// Tile table for one routing: rows per expert -> ceil(rows / 128) M-tiles each (capacity 384 rows = 3 tiles)
static std::vector<int> make_tiles(bool uniform, std::vector<int>& rows) {
  rows.assign(E, 0);
  if (uniform) rows.assign(E, 256);
  else {  // like the op test: T = 8192 tokens, top-2 distinct experts out of 64, uniform random, fixed seed
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> d(0, E - 1);
    for (int t = 0; t < 8192; ++t) { int a = d(rng), c = d(rng); while (c == a) c = d(rng); rows[a]++; rows[c]++; }
  }
  std::vector<int> tl;
  for (int e = 0; e < E; ++e) {
    const int nt = std::min(3, (rows[e] + 127) / 128);
    for (int m = 0; m < nt; ++m) { tl.push_back(e); tl.push_back(m); tl.push_back(0); tl.push_back(0); }
  }
  return tl;
}

int main(int argc, char** argv) {
  const int which = argc > 1 ? atoi(argv[1]) : -1;
  g_att = argc > 2 && argv[2][0] == 'a';
  const size_t wb = (size_t)E * N * K / 2, wsn = (size_t)E * NG * KS * 32, xb = (size_t)E * MG * 16 * K / 2, xsn = (size_t)E * MG * KS * 32;
  Bufs b;
  CHECK(hipMalloc(&b.W, wb)); CHECK(hipMalloc(&b.WS, wsn * 4)); CHECK(hipMalloc(&b.X, xb)); CHECK(hipMalloc(&b.XS, xsn * 4));
  CHECK(hipMalloc(&b.out, NOUT_MAX * 4)); CHECK(hipMalloc(&b.outref, NOUT_MAX * 4));
  CHECK(hipMalloc(&b.t, (size_t)MAXTILES * 64 * 4 * 4 * 8)); CHECK(hipMalloc(&b.tiles, MAXTILES * sizeof(int4)));
  init_bits<<<4096, 256>>>((unsigned*)b.W, wb / 4, 1u); init_bits<<<4096, 256>>>((unsigned*)b.X, xb / 4, 2u);
  init_scales<<<4096, 256>>>((unsigned*)b.WS, wsn, 3u); init_scales<<<4096, 256>>>((unsigned*)b.XS, xsn, 4u);
  CHECK(hipDeviceSynchronize());
  for (int route = 0; route < 2; ++route) {  // 0: uniform 256 rows/expert, 1: random top-2 routing (op-test-like)
    std::vector<int> rows;
    const std::vector<int> tl = make_tiles(route == 0, rows);
    b.ntiles = (int)tl.size() / 4;
    g_route_id = route + 1;
    CHECK(hipMemcpy(b.tiles, tl.data(), tl.size() * 4, hipMemcpyHostToDevice));
    int mn = 1 << 30, mx = 0, n3 = 0; for (int e = 0; e < E; ++e) { mn = std::min(mn, rows[e]); mx = std::max(mx, rows[e]); n3 += rows[e] > 256; }
    printf("routing %s: rows/expert %d..%d, %d experts with 3 M-tiles, %d M-tiles total (%.1f%% of tile rows valid)\n",
           route ? "random top-2" : "uniform", mn, mx, n3, b.ntiles, 100.0 * 16384 / (b.ntiles * 128.0));
    //                                   CW MW WPE MC CB PD ABL TDB NS 1BAR PFD TOK OUTM SPLIT
    if (which < 0 || which == 0) bench<4, 1, 1, 0, 0, 3, 0, 1, 5, 0, 0, 1, 0, 1>("v11 fp32 (check)", b);
    if (which < 0 || which == 1) bench<4, 1, 1, 0, 0, 3, 0, 1, 5, 0, 0, 1, 2, 1>("v11 fp4 scale=1", b);
    if (which < 0 || which == 2) bench<4, 1, 1, 0, 0, 3, 0, 1, 5, 0, 0, 1, 3, 1>("v11 fp4 + MX quant", b);
  }
  return 0;
}
