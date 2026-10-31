# DSA sparse MLA prefill — deep optimization section

Owner node: mi355-4. Opened 2026-10-30. Tracker row: `OPT_TRACKER.md` item 2 / 2b.

## Where it stands

Kernel: `_sparse_mla_fwd_split_dim_kernel` (`kernels/ops/attention/dsa/triton_sparse_mla.py`),
single pass, one CTA per query token, all heads of the rank (12, padded to 16), loops over
top-k 2048 in BLOCK_N blocks. KV = `fp8_ds_mla` rows (656 B: 512 fp8 e4m3 NoPE, 4 fp32
per-128 scales, 64 bf16 RoPE).

Precision today (PR #9, `SGLANG_ROCM_SPARSE_MLA_SCALE_POST`): Q bf16; K/V fp8 -> bf16
(exact); QK and PV bf16 MFMA, fp32 accumulate; group scales applied in fp32 after the dot
(QK) / onto P (PV); softmax fp32; P -> bf16; out bf16, LSE fp32.

Microbench (`scripts/sparse_mla_prefill_microbench.py`, seq 16384, per DSA layer per rank):

```text
flag off      5.74 ms   153 TFLOPS(12 heads)   3.85 TB/s gathered
SCALE_POST    5.02 ms   175 TFLOPS             4.40 TB/s
random == clustered indices -> not gather/HBM bound; VALU (convert) bound per CU
(C3: more CTAs per token do not help, 2 CTAs/CU already near peak; not occupancy bound)
```

Roofline (16K tokens, 12 heads, top-k 2048): compute 0.88 TFLOP (1.17 with the 16-row
padding) -> 0.38 ms (0.51) at ~2.3 PF bf16; fp8 MFMA halves that. Gather 22 GB; the KV pool
(~21 MB at 32K context) fits the 256 MB MALL, so the practical ceiling is cache bandwidth,
~1.3-2.2 ms (assumed 10-17 TB/s, not measured). Target: 5.0 -> ~2-2.5 ms bf16.

## Directions

Floor-compliant (bf16 math):
- C1 head packing: 12 real rows padded to 16 waste 25% of every MFMA; try 16x12 tiles or
  2 tokens per CTA (24 rows padded to 32).
- C2 packed fp8 -> bf16 converts (gfx950 packed cvt) or convert once into LDS.
- C3 more parallelism per token: split the 2048 keys over warps/CTAs with an LSE merge.
  REJECTED 2026-10-30 (cross-CTA split is slower at every seq >= 512; see Log).
- C4 async copy / LDS double buffer (currently disabled by `_no_async_copy`; re-check on gfx950).

Rewrite outside Triton (bf16 math, floor-compliant). Triton cannot remove the QK/convert
duplication or the occupancy-1 register bill (see Log, ISA attribution):
- R1 HIP (opus) kernel for fp8_ds_mla: each wave owns its own 32-key tiles (no QK or convert
  duplication, no per-tile cross-warp sync), merge the 4 waves' (m, l, acc) through LDS once
  per token; async direct-to-LDS gather of the raw 656 B rows, 3-4 stages; convert fp8 in
  registers with `v_cvt_scalef32_pk_bf16_fp8` using the row's group scale as the cvt scale
  operand (dequant for free, no SCALE_POST multiplies), once for QK and once via
  `ds_read_b64_tr_b8` for PV (cheaper than a bf16 LDS round trip). Model: ~2-2.5k
  SIMD-cycles per 32-key tile vs 11.7k today; expected 1.5-2.5 ms vs 5.02 (-60..-85 ms per
  16K prefill batch, ~7-9%). Porting opus a16w16 as-is would land ~3.5 ms (~-4%).
- R2 Gluon (Triton 3.8, `triton.experimental.gluon` AMD available): same design with explicit
  layouts, stays in Python; less proven for gather + attention on gfx950.
- FlyDSL: no attention precedent in the `flydsl` corpus (MoE / A2A only).

Experimental, below the precision floor (default-off flag, gsm8k + long-context check,
reported separately):
- E0 unit-scale Q fp8: cast bf16 Q to e4m3 directly (what `USE_FP8_DOT` does for unscaled
  KV). Needs the real Q range first (e4m3 max 448; small values lose precision).
- E1 per-(token, head) Q scale: amax over the 576 dims at load, scale into e4m3, multiply
  the QK partials back. Quant happens inside the attention kernel: Q is read once per CTA
  (~18 KB bf16 vs ~1.3 MB of gathered KV), so fusing it into the producer (absorb bmm /
  RoPE-concat) would only save ~9 KB of reads per token; fuse into the producer only if
  another consumer wants fp8 Q.
- Both: fp8 MFMA for QK (fp8 Q x fp8 K, then x q_scale x k_group_scale) and PV (P x 448
  -> fp8, x fp8 V, then / 448 x v_group_scale); no K/V converts at all.

## Gates

Speed: microbench, then prefill sweep (`scripts/prefill_sweep.sh`) vs flag off.
Accuracy: greedy gsm8k (`TEMP=0 scripts/gsm8k_gate.sh`) vs control (0.98; 188-190/200
identical is the observed run-to-run band), plus a long-context check for E0/E1.

## Log

- 2026-10-30 mi355-4 — R1 HIP kernel E2E behind `SGLANG_ROCM_SPARSE_MLA_PREFILL_HIP=1`
  (dispatch in `triton_sparse_mla_fwd`, long-prefill branch only; SCALE_POST off). Kernel
  now 2.49-2.51 ms clustered / 2.77 random (Triton 5.01) after: buffer LDS DMA with SGPR
  soffset + empty descriptor for idx < 0, permlane16/32 swaps for the row max, mask folded
  into an fma, lazy rescale (only when a head's max grows > 2^8), branch-free index loads
  prefetched S iterations ahead (a load merged at a join made the compiler wait vmcnt(0)
  before every DMA), and `s_barrier` + lgkmcnt instead of `__syncthreads()` (its fence
  waited for the next tile's DMA mid-tile). XCD token remap: no effect (kept, harmless).
  Ablation (torch data): compute+sync 2.00 ms, DMA+sync 1.47 ms -> compute-bound now; counters
  (`scripts/sparse_mla_hip_standalone.py` + `scripts/rocprof_counters.py`, torch-free so
  rocprofv3 works): VALU pipe ~73% busy in compute-only, L2 hit 29%, LDS conflicts minor.
  Prefill TTFT (`results/prefill/i2b_hip_prefill_20261030`) vs SCALE_POST / flags off:
  ```text
  4k 278.3 (294.5 / 301)   8k 484.6 (516.1 / 528)   16k 906.5 (970.9 / 990)
  32k 1816.3 (1941.0 / 1988)   128k median 7484 (7990 / 8237)   -> -5.5..-6.6% vs SCALE_POST
  ```
  Greedy gsm8k (`results/acc/i2b_hip_prefill_greedy_20261030`): 0.985 (control 0.98,
  SCALE_POST 0.975); 191/200 generations identical to control (SCALE_POST 189, band 188-190).
- 2026-10-30 mi355-4 — R1 HIP kernel, WIP (uncommitted, branch `k31/sparse-mla-splitk-long`):
  `jit/csrc/attention/sparse_mla_prefill_gfx950.cuh` + `ops/attention/dsa/sparse_mla_prefill_gfx950.py`
  (`stages` template arg), bench `scripts/sparse_mla_hip_microbench.py`, ISA `scripts/jit_isa.sh
  sparse_mla_prefill_gfx950_2`. Design: CTA per token, 4 waves, wave w = scale group w (its QK
  partial + its 128 output dims), QK as S^T = K Q^T and PV as O^T = V^T P^T (lane % 16 = head),
  partials summed via LDS per 32-key tile, V operand via `ds_read_b64_tr_b8`, exact unit-scale
  converts + fp32 scales after the dots (SCALE_POST numerics). Probed on gfx950:
  16x16x32 bf16 MFMA layout; tr_b8 = in each 16-lane group, lane i elem j <- byte (i&7) of
  src lane 2j+(i>>3); `v_cvt_scalef32_pk_bf16_fp8` scale is power-of-two only (E8M0-like),
  so it cannot carry fp32 group scales; `global_load_lds` 16 B writes base + lane*16.
  Bug found: branching on `tid >> 6` ran both sides' MFMAs (MFMA ignores EXEC) -> wave id via
  `readfirstlane`. Accuracy = Triton (rel L2 2.4e-3 vs fp32, LSE 1e-6, q x30 too; -1 rows,
  empty row, topk 33/1000/2047 ok). seq 16384, clustered / random:
  ```text
  triton SCALE_POST     5.00 / 5.01 ms
  v1 reg-staged loads   3.19 / 3.42   164 VGPR, occ 3, 536 instr/wave-tile (64 cvt, 18 MFMA)
  v2 LDS DMA, 2 stages  2.88 / 3.03   110 VGPR, occ 3, loop 211 VALU + 18 MFMA + 193 SALU
  v2 LDS DMA, 3 stages  3.49 / 3.45   71 KB LDS -> occ 2
  ```
  v2: ~1690 SIMD-cycles per wave-tile vs ~840 of VALU issue, so half is waiting (DMA latency
  with a 1-tile prefetch, 2 barriers/tile, LDS). rocprofv3 --pmc aborts in this env
  ("Option 'spirv-expand-step' registered more than once", LLVM CommandLine clash) and hangs;
  counters not collected yet.
- 2026-10-30 mi355-4 — Triton ISA attribution + aiter opus reference (decides "rewrite?").
  Best config BLOCK_N 32 / 2 warps / 3 stages (`/tmp/isa.py 32,2,3` with TRITON_CACHE_DIR,
  amdgcn -> `/tmp/isa_loop.py`). 411 VGPR (256 arch + 155 AGPR), no spill, **occupancy 1**
  (one wave per SIMD; 2 CTAs fill a CU, which is why seq 512 is already at peak). Main loop,
  per wave per 32-key tile, 925 instr: 192 `v_cvt_scalef32_pk_bf16_fp8` (minimum 64: each
  wave converts the whole K tile for QK, then V again via `ds_read_b64_tr_b8`), 52 MFMA
  16x16x32 bf16 (minimum 34: both waves compute the full QK, they only split D_V for PV),
  119 accvgpr moves, 140 SALU, ~250 other VALU. Measured 11.7k SIMD-cycles per 32-key tile
  vs ~1.1k for the MFMAs alone. The duplication comes from Triton's dot layout for M=16 with
  2 warps and is not controllable from `tl.dot`.
  aiter opus gfx950 sparse prefill (`op_tests/test_mla_sparse_prefill.py`, N 16384, H 16,
  2048 keys per token = `--topk 1024`, its two ranges): bf16 a16w16 (bf16 KV, D 512) 3.67 ms,
  v4 a8w8 (fp8 Q, below floor) 2.81 ms, vs our 5.02 ms (fp8 KV + converts, QK 576). No aiter
  kernel takes fp8_ds_mla with bf16 math. `mla_v4_prefill_a16w16_16mx1_16nx4.hpp` (4 warps
  split keys, cross-warp row max via LDS every tile) is the closest HIP skeleton.
  Impact: sparse MLA is 107 ms of the ~920 ms 16K-token prefill batch after #9 (24 layers).
- 2026-10-30 mi355-4 — C3 split over keys, REJECTED. Results
  `/shared_nfs/kimi-k31/results/sparse_mla_c3_20261030/`, script
  `scripts/sparse_mla_splitk_microbench.py` (rel L2 vs fp32 on 128 sampled rows), clustered
  indices, SCALE_POST=1. (a) Existing split-K path `_triton_sparse_mla_fwd_splitk` at seq
  16384: kv_splits 1 (fused kernel) 15.5 ms, 2 11.1, 4 11.8, 8 13.2, 16 15.9 ms vs 5.02 ms
  default; kv_splits 32 crashes (GPU write fault, int32 offset overflow: the partial buffer
  is 2^32 elements; production only picks split-K at seq*head_blocks <= num_cu, so not
  reachable there). (b) Same split put into the best kernel (`KV_SPLITS` constexpr on
  `_sparse_mla_fwd_split_dim_kernel`, grid (seq, S), int64 partial offsets, existing reduce
  kernel; local uncommitted diff on branch `k31/sparse-mla-splitk-long`, copy in the results
  dir as `split_dim_kv_splits.diff`):
  ```text
  seq 16384  S=1 5.01 | S=2 5.98 (main 5.12 + reduce 0.85) | S=4 6.43 (5.50+0.93) | S=8 7.42 (6.22+1.19)
  seq 1024   S=1 0.332 | S=2 0.382 | S=4 0.401        seq 512  S=1 0.170 | S=2 0.197 | S=4 0.210
  rel L2     2.0e-3 (S=1, fp32 partials) / 2.6-2.7e-3 (bf16 partials)
  ```
  The main kernel alone gets slower as S grows, so grid parallelism is not the limit. seq 512
  (2 CTAs per CU) already runs at 161 TFLOPS vs 175 at 16K, so one or two CTAs per CU nearly
  saturate it: the kernel is limited by per-CU work per gathered key (gather + fp8->bf16
  convert + scale VALU), not by occupancy or latency. Splitting only adds partial traffic and
  a reduce. The same argument limits C1: packing removes padded MFMA rows, but MFMA is not
  the bottleneck (see E0/E1), so C2 (per-key convert cost) is the better next lever.
- 2026-10-30 mi355-4 — E0/E1 fp8 Q, REJECTED (local branch `k31/sparse-mla-fp8-q`, not
  pushed). `SGLANG_ROCM_SPARSE_MLA_FP8_Q` modes on top of SCALE_POST, microbench seq 16384,
  clustered indices, rel L2 error vs fp32 reference (Q x1 / Q x30):
  ```text
  0 bf16 (PR #9)                 5.02 ms   2.0e-3 / 2.3e-3
  1 QK fp8, Q unit scale         5.17 ms   2.0e-3 / 2.6e-3
  2 QK fp8, Q per-row scale      5.20 ms   2.0e-3 / 2.6e-3
  3 QK+PV fp8, Q unit scale      6.72 ms   2.2e-2 / 2.5e-2
  4 QK+PV fp8, Q per-row scale   6.74 ms   2.2e-2 / 2.5e-2
  ```
  The kernel runs at ~7.6% of bf16 MFMA peak, so doubling MFMA throughput buys nothing and
  the extra Q / P quantization costs time; PV in fp8 adds 10x error (P in e4m3). In-kernel
  vs producer-fused Q quant is therefore moot. Next lever is the kernel structure (C1 head
  packing, C3 split over keys), not precision.
- 2026-10-30 mi355-4 — SCALE_POST (bf16, floor-compliant): 5.74 -> 5.02 ms, PR #9.
