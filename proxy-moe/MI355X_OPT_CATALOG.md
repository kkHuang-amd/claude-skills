Owner node: ctheliosmc-rck-g02-h13-17.rck.dcgpu

# Proxy MoE: what was optimized on MI355X (gfx950), for re-evaluation on MI455 (gfx1250)

A summary of the MI355X work, written without changing any source doc. Abbreviations:
- `OPT4` = `OPT_mi355-4.md`
- `OPT231` = `OPT_crsuse2-m2m-231.md`
- `MONO` = `MONO_FFN_crsuse2-m2m-231.md`
- `R1021` = `fixed_seq_len/RESULTS.md`
- `R1022` = `fixed_seq_len/RESULTS_mono_1022.md`

Every number comes from the cited doc and heading. Numbers are not re-measured. All MI355X runs
used 8x MI355X (gfx950, 256 CUs) and the dummy checkpoint (seed 0).

## 1. Target numbers (final MI355X state)

### 1a. fixed_seq_len, final build (R1022, "Mono vs base (same night) and vs 10-21, per point")

The build is the new heads (SGLang `ed3d8b0089`, AITER `e9b0b5979`), prefill SP on, and the FFN
monokernel plugin `fb3193b` (`--monokernel-backend amd`). The monokernel only runs on decode steps of
batch 1..16, so at CONC >= 32 it is idle and the numbers are those of the heads. EP1 everywhere.
- `tp8ep1` = **24 layers, 8 GPUs, TP8 EP1**.
- `tp4ep1` = **12 layers, 4 GPUs, TP4 EP1**.

| Recipe | ISL/OSL | CONC | TTT/GPU | Median ITL (ms) | Median TTFT (ms) | Mono vs base TTT/GPU |
|---|---|---|---|---|---|---|
| tp8ep1 24L | 1024/1024 | 2 / 16 / 64 / 256 / 1024 | 101.9 / 285.6 / 516.6 / 1466.6 / 2248.3 | 4.79 / 13.53 / 29.19 / 36.54 / 75.55 | 110 / 338 / 1011 / 3666 / 14185 | +28.8% (c2), +12.2% (c16) |
| tp8ep1 24L | 8192/1024 | 2 / 16 / 64 / 256 / 1024 | 416.4 / 1040.1 / 1659.1 / 3042.8 / 3795.8 | 4.97 / 13.73 / 29.02 / 37.40 / 69.51 | 287 / 1916 / 7556 / 29306 / 114751 | +26.5% (c2), +9.9% (c16) |
| tp8ep1 24L | 70000/300 | 2 / 8 / 32 | 3429.3 / 3752.9 / 3923.0 | 5.00 / 9.41 / 22.63 | 2208 / 8452 / 32828 | +8.0% (c2), +1.5% (c8) |
| tp4ep1 12L | 1024/1024 | 2 / 16 / 64 / 256 / 1024 | 262.1 / 639.6 / 1160.8 / 3262.4 / 5333.8 | 3.70 / 12.13 / 26.15 / 33.65 / 67.17 | 87 / 262 / 790 / 2853 / 11103 | +19.4% (c2), +8.4% (c16) |
| tp4ep1 12L | 8192/1024 | 2 / 16 / 64 / 256 / 1024 | 1070.2 / 2388.8 / 3835.3 / 7260.4 / 9301.4 | 3.87 / 12.26 / 26.26 / 34.15 / 65.46 | 214 / 1578 / 5912 / 23106 / 81042 | +17.9% (c2), +6.8% (c16) |
| tp4ep1 12L | 70000/300 | 2 / 8 / 32 | 8679.0 / 9355.9 / 9810.5 | 3.89 / 8.06 / 20.01 | 1740 / 7424 / 26784 | +5.6% (c2), +1.6% (c8) |

Pre-monokernel baseline: the full 50-point 10-21 tables are in R1021 "Tables". They used older heads
(no SP, no MoE rows for M <= 1024). Ratios against GB300 are in R1022, "Geomean MI355X / GB300":
- tp4ep1: TTT/GPU 0.891 -> 0.923.
- tp8ep1: TTT/GPU 0.944 -> 1.020.

### 1b. Short bench, 1024 in / 256 out, 64 requests, c16 (OPT231 "TTT/GPU + P90 interactivity" and "Results")

| Config (24L TP8 unless noted) | Out tok/s | TTFT med (ms) | TPOT med (ms) | Source |
|---|---|---|---|---|
| TP8 **EP8** package defaults, pristine | 788.6 | 324.1 | 19.05 | OPT231 Results `c231_tp8_base_ep8def_warm_1021` |
| TP8 **EP8** + AR fusion + tuned + T4 + b_nt fix (best EP8) | 833.8 | 296.8 | 18.06 | OPT231 Results `c231_ep8_t4_bnt2fix_1020` |
| TP8 EP1 new heads, SP off | 938.2 | 337.0 | 15.77 | OPT231 CONTINUE HERE, Next 1(a) |
| TP8 EP1 + monokernel v6 (TPOT only) | — | equal to base | 14.08 (base 15.78) | MONO Results, v6 A/B |
| TP4 12L EP4 best | 1002.6 | 234.0 | 15.10 | OPT231 T11 row (h) |
| TP4 12L EP1 best (no mono) | 1090.3 | 256.0 | 13.73 | OPT231 T11 row (g) |
| TP4 12L EP1 + monokernel v6 (TPOT only) | — | equal | 12.65 (base 13.63) | MONO Results, v6 A/B |

Monokernel decode TPOT, base -> mono v6, 1024/256 (MONO Results row "Serving A/B of fb3193b"):

| Config | CONC | Base (ms) | Mono (ms) | Change |
|---|---|---|---|---|
| TP8 24L | 1 | 5.36 | 3.61 | -32.6% |
| TP8 24L | 16 | 15.78 | 14.08 | -10.8% |
| TP4 12L | 1 | 3.51 | 2.79 | -20.5% |
| TP4 12L | 16 | 13.63 | 12.65 | -7.2% |

### 1c. Comparing with MI455 TP8 EP8 12L

- **No MI355X number exists for 12 layers at TP8, and no fixed_seq_len sweep was run with EP8.**
  - MI355X 12L runs are TP4 on 4 GPUs (`tp4ep1`).
  - MI355X TP8 runs are 24L.
  - MI355X EP8 was measured only on the 1024/256 c16 bench (24L): 788.6 pristine, 833.8 best.
- Closest proxies:
  - **Decode, per layer:** TP8 24L has the same per-GPU per-layer shapes as TP8 12L, but twice the
    layers. A TP8 12L ITL would be about half the TP8 24L ITL plus the fixed per-step cost (sampling,
    logits). This is an estimate, not measured.
  - **12L model:** `tp4ep1` (4 GPUs). Per GPU it reads twice the weight bytes per layer of TP8.
  - **EP choice:** On MI355X, EP1 beat EP8 by +13.2% out tok/s and -13.6% TPOT, but had +17.4% TTFT
    (OPT4 "Summary vs baseline", T8). MI455 currently runs EP8, so this is the first structural item
    to re-test there.
- Other settings that differ from MI455 (`MI455_RUN.md`):
  - mem-fraction 0.88 on MI355X vs 0.80 on MI455.
  - MI455 uses `--attention-backend triton`. The MI355X sweep used the default backend; traces show
    `_unified_attention_gluon_kernel` and fmha (OPT4 "Warm profile").
  - MI355X prefill AR is QR INT8 (item O8 below).

## 2. Optimizations

Each entry lists: Where / Mechanism / Gain / Final / gfx950 assumptions. Gains are serving numbers
unless marked "op".

### O1. EP1 instead of EP8 (`--ep-size 1`, MoE TP8) — T8
- **Where:** server flag. The sweep recipes omit `--ep-size`, which gives EP1.
- **Mechanism:** decode "comm" was really per-layer EP load-imbalance wait. The AR after the MoE
  took 103.5 us, of which 96.7 us was skew; the real AR cost is about 8-10 us (OPT4 "AR wait
  attribution"). EP1 removes the imbalance. The cost: EP1 moe2 writes topk x hidden partials, 8x
  the output traffic, so prefill gets slower.
- **Gain:** TPOT 18.6 -> 16.5 ms, out tok/s 808 -> 893, prefill 425 -> 533 ms per 16k tokens, TTFT
  316 -> 371 ms (OPT4 Task items T8). At TP4: 1089.8 vs 1000.4 out tok/s (OPT231 T11 (f) vs (d)).
- **Final:** yes. Every final MI355X number uses EP1.
- **gfx950 assumptions:** none in the mechanism. The size of the gain depends on how big the
  imbalance wait is compared with the MoE weight-streaming time at 256 CUs / 8 TB/s HBM. Re-measure
  the rank skew on MI455 (`scripts/rank_skew.py`, `ar_wait_attrib.py`).

### O2. Decode AR + RMSNorm fusion (`--enable-aiter-allreduce-fusion`) — T1
- **Where:** server flag. The decode AR becomes `reduce_scatter_cross_device_store` +
  `local_device_load_rmsnorm_naive` (OPT4 Results).
- **Gain:**
  - EP8: decode GPU 19.7 -> 18.7 ms/step, TPOT 19.1 -> 18.6 ms, out tok/s 789 -> 808 (OPT4 T1).
  - TP4 EP4: 981.2 -> 995.6 out tok/s (+1.5%; OPT231 T11 rows (b) -> (c)).
- **Final:** yes. The monokernel path replaces it at decode batch <= 16.
- **gfx950 assumptions:** the aiter custom-AR IPC kernels, and a fully connected 8-GPU XGMI.

### O3. Tuned FlyDSL `fused_moe` rows for prefill (AITER tuned CSV) — T2 / T2b / T11
- **Where:** `aiter/configs/model_configs/proxymoe_fp4_tuned_fmoe.csv` on the AITER branch. Rows:
  - EP1/TP8 M 2048/4096/8192
  - EP8 M >= 2048
  - TP4 rows (AITER fd537aaa4)
  
  The method for EP8: tune an "EP-equivalent" problem (topk/ep = 2 over the 64 local experts,
  `TUNE_MOE_STAGE2_ATOMIC_ONLY=1`), then rewrite topk back to 16 (OPT4 CONTINUE HERE).
- **Gain:**
  - EP1 prefill 533 -> 510 ms, out tok/s 893 -> 902, TTFT 371 -> 359 (OPT4 T2).
  - EP8 prefill 425 -> 410 ms, TTFT 316 -> 308 (OPT4 T2b; op level -4..-5%).
  - TP4 op level: EP1-TP4 M8192 -6.3%; EP4 M4096 -4.7%, M8192 -6.3%. TTFT -1.9% / -2.4% (OPT231 T11).
  - Tuned rows changed output by rel_l2 3.6% vs default. This is a systematic difference in the fp4
    requant path.
- **Final:** yes, prefill rows only. Tuned decode rows were slower than production
  `mfma_moe1_silu_mul_*_pm1_async_v33` and were not installed.
- **gfx950 assumptions:**
  - The CSV is keyed on `gfx950` and `cu_num` 256.
  - The kernel names encode MFMA tile shapes (`t128x64x256`, `persist_cu256`).
  - **All rows must be re-tuned for gfx1250** (different CU count and WMMA instead of MFMA).
  - Tuner fixes needed at H=16384:
    - vectorized `v2_stage1_sorted_ref`
    - `mxfp4_moe_gemm2` HIDDEN_MAX bucket
    - chunked `dequant_w` (OOM fix, 44a1da311)
    - hidden>8192 tuner/dispatcher fix (6ef4d09fd)
  - The `--mxfp4-flydsl` tuner is unusable at H=16384: "B-scale addressing covers at most 32 K
    tiles" (OPT4).

### O4. Decode / large-M MoE rows (M 256/512/1024, plus 16384/32768) — Item 2
- **Where:** AITER `605ade939` (now in `e9b0b5979`). Lookup uses nextPow2(M), so the token=1024 row
  serves M 513..1024.
- **Mechanism:** M <= 1024 ran on default t32 tiles. With about 32 rows per expert, many experts need
  2 M-blocks, so weights are re-read (stated as a hypothesis in the doc). The M1024 row is a copy of
  the M2048 row (t64x128x256), because the tuner's choice at M1024 was wrong.
- **Gain:**
  - Op level: M1024 1,784.7 -> 1,411.7 us (-20.9%), M256 -5.0%, M512 -3.7%. **Slower** at M640 (+6.6%)
    and M768 (+3.2%).
  - Serving 1k/1k: TTT/GPU c256 +2.9%, c512 +3.3%, c1024 +13.2%; decode step -13.4% at c1024
    (OPT231 "Item 2", serving table).
  - With chunk 32768, the M16k/32k rows gave +4.2% / +4.8% TTT/GPU at 70k (OPT231 "Item 3 / G").
- **Final:** yes (pushed).
- **gfx950 assumptions:** same as O3. Also: the tuner-reported time disagreed with the profiler
  measurement (M1024: 1,147 vs 1,751 us), so trust the measurement on gfx1250 too.

### O5. Stage-2 MoE `b_nt` fix (nontemporal B loads in moe2)
- **Where:** AITER `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py`,
  `compile_mixed_moe_gemm2_common`. The code did `del b_nt`, so `_bnt2` in the kernel name was a
  no-op. The fix passes `cache_modifier=b_nt` to stage-2 `load_cell` (commit 4b921e91e; re-applied
  after the upstream merge).
- **Gain:**
  - Op level: EP1 M=16 moe2 139.1 -> 120.4 us (7.03 TB/s); layer -6.6% (OPT231 "D2 design", ROOT CAUSE).
  - Serving EP1: 901.6 -> 931.2 out tok/s, TPOT 16.47 -> 15.89 ms.
  - Serving EP8: 814.0 -> 833.8 (OPT231 "TTT/GPU" table).
- **Final:** yes.
- **gfx950 assumptions:**
  - `buffer_load` cache-policy bits (cache_modifier=2 = streaming / no L2 allocation) as gfx950
    encodes them. gfx1250 uses a different temporal-hint / scope encoding, so check what `b_nt=2`
    lowers to.
  - The 7.13 TB/s achievable-HBM ceiling is from MI355X.

### O6. bf16x3 router GEMM in prefill (`SGLANG_OPT_LAGUNA_ROUTER_BF16X3=1`) — T4
- **Where:** SGLang `models/laguna.py` `LagunaMoEGate`, plus `environ.py`.
- **Mechanism:** the fp32 router weight is split into 3 bf16 parts. One "concat-N" GEMM
  `x @ [P0;P1;P2]^T` produces [M,1536] in fp32, and the 3 slices are summed.
  - fp32 F.linear: 1050 us, rel_l2 2.29e-6.
  - bf16x3: 322 us, rel 8.2e-7, top-16 identical 100%.
  - Decode keeps the fp32 router (weight-read-bound; x3 would read more bytes).
- **Gain:**
  - EP8: TTFT 314.3 -> 294.0 ms, prefill GPU 82.46 -> 74.58 ms/step.
  - EP1: TTFT 364.8 -> 342.9 ms, prefill 101.79 -> 93.99 ms/step.
  - Source: OPT231 "T4 serving A/B".
- **Final:** yes.
- **gfx950 assumptions:** the fp32 hipBLASLt GEMM is compute-bound (~140 TFLOP/s) while bf16 MFMA
  is much faster. Re-check the fp32 / bf16 WMMA throughput ratio on gfx1250.

### O7. Prefill sequence-parallel FFN (`SGLANG_OPT_LAGUNA_PREFILL_SP=1`) — F2
- **Where:**
  - SGLang `ed3d8b0089`: `models/laguna.py` `_sp_forward` / `LagunaMoE.forward_sp`, and
    `moe/moe_runner/aiter.py` (takes pre-quantized fp4 input).
  - `..._SP_MIN_TOKENS` (default 4096).
  - `SGLANG_OPT_LAGUNA_PREFILL_SP_SHARED_QUANT=triton|hip` (default `triton`).
- **Mechanism, per layer:**
  1. Reduce-scatter (RS) the o_proj partial in bf16.
  2. On the M/8 token shard: add + norm, router, hip MXFP4 quant.
  3. All-gather (AG) the top-k results and the fp4 activations + e8m0 scales.
  4. Run the MoE on the pre-quantized input.
  5. One QR INT8 AR for attention + FFN.
- **Eligibility:** extend only, tokens % tp == 0, EP1.
- **Op level, M8192:**
  - RS (RCCL) 653 us + AG (aiter fp4) 158 us = 811 us vs 1034 us for the QR INT8 AR.
  - Pre-quantized `fused_moe` is bit-exact and 62 us faster.
- **Gain (OPT231 "F2 P1"):**
  - 70k c8: TTT/GPU +5.3% (`hip`) / +3.3% (`triton`); TTFT -6.1% / -3.7%.
  - 8k/1k c64: +2.1% / +1.2%.
  - 1024/256: within noise.
- **Final:** in the branch, default off. Turned on in the final sweep (R1022).
- **gfx950 assumptions:**
  - The comm costs (RCCL RS, aiter custom AG, QR AR) were measured on MI355X XGMI.
  - Needs `AITER_CUSTOM_AR_MAX_SIZE=536870912`.
  - The gain is roughly "AR cost minus RS+AG cost", so re-measure the op-level table first.

### O8. Lossy INT8 QuickReduce for prefill AR (`ROCM_QUICK_REDUCE_QUANTIZATION=INT8`) — T0
- **Where:** container environment variable, kept as is. Prefill AR = `quickreduce::...CodecQ8`.
- **Gain (bf16 AR as reference):**
  - Op level, M8192: FP 1301 vs INT8 1034 us (OPT231 "F follow-up").
  - Unsetting it on tp4ep1: TTFT +12% almost everywhere; TTT/GPU geomean -1.4% (1k/1k),
    -4.3% (8k/1k), -9.1% (70k) (R1021 "tp4ep1 with ... unset").
  - INT6 and INT4 would be -11% / -17% AR time, at 4x / 16x the error. Not adopted.
- **Final:** yes.
- **gfx950 assumptions:** quickreduce twoshot kernels on XGMI. QR time looked CU/codec-bound, not
  link-bound (hypothesis). The `gfx1250-quickreduce-optimization` skill covers gfx1250.

### O9. Laguna FFN-only decode monokernel (sgl-amd-mono plugin `laguna/gfx950/`)
- **Where:**
  - Branch `kkHuang-amd/sgl-amd-mono` `feat/laguna-ffn-mono`.
  - The final sweep used `fb3193b`. Push points in MONO: `07f2583`, `31246a4` (TP4), `98ec598`.
  - Flag: `--monokernel-backend amd` (SGLang PR #44353 API).
  - Environment variables:
    - `SGLANG_AMD_MONO_LAGUNA_MAX_ROWS` (default 16)
    - `SGLANG_AMD_MONO_LAGUNA_TP=4|8`
    - `SGLANG_AMD_MONO_CHECK=1|2` (check mode)
    - `LAGUNA_*` switches for sub-optimizations
- **Mechanism:** one persistent launch per layer, covering:
  1. S1 seam: o_proj AR by peer push, rank-ordered sum, residual, RMSNorm.
  2. S2 TP-sharded router: 64 experts per rank, logits exchanged.
  3. S3 routed A4W4 MoE over the expert union.
  4. S4 shared expert, hidden behind the logit exchange.
  5. S5 MoE AR + next norm.
  
  It replaces about 185 us/layer of the SGLang path at bs2 (MONO "Target").
- **Gain:**
  - Layer TP8 M 1/2/4/8/16: 84.8 / 121.3 / 184.0 / 295.4 / 469.8 us ->
    71.4 / 101.7 / 159.7 / 261.3 / 407.6 us (MONO Results, "Seam sum of squares").
  - Serving: see 1a and 1b. TP8 1k/1k TTT/GPU +28.8 / +22.7 / +14.4 / +12.2% at c2/4/8/16.
  - Numerics: check mode next-input rel 2.8-3.3e-3, vs SGLang run-to-run 2.9e-3.
- **Final:** yes (decode batch <= 16).
- **gfx950 assumptions (heaviest of any item):**
  - Kernel launch shape:
    - `MonoSpec devices={"gfx950": DeviceLimits(...)}`
    - 256 CTAs x 512 threads, one CTA per CU (256 CUs), wave64
  - Compute and data layout:
    - `mfma_scaled` 16x16x128 FP4xFP4. Its lane layout equals aiter's (16,16) shuffle tile.
    - `v_cvt_scalef32_pk_fp4_f32` for the in-kernel MXFP4 quantizer.
    - aiter `shuffle_weight(16,16)` / `shuffle_scale` dword layout. The scale-pairing tricks depend
      on it.
  - Memory and cache:
    - `buffer_load` with at most 4 dwords and cache_modifier 2.
    - LDS 53-94 KB per CTA, within the 160 KB gfx950 LDS.
    - Non-coherent per-XCD L2 inside a launch, so CM_DEV stores and loads are used for cross-CTA
      hand-offs.
  - Multi-GPU: XGMI peer push / IPC `PeerBuffer`; TP8 max FLAG_WORDS=8.
  - FlyDSL codegen: `rocdl.sched_barrier`.
  - A4W4 activations: the `gfx1250-mxfp4-a4w4-to-a8w4` skill says gfx1250 lacks
    `V_WMMA_SCALE_F32_32X16X128_F4`, so A4W4 MMA may not port directly.
- **Sub-optimizations in the monokernel (MONO Results; layer us at TP8 unless noted; all bit-exact):**

| # | Switch / commit | What | Measured |
|---|---|---|---|
| a | `LAGUNA_SEAM_TWO_SHOT_MIN` (two-shot from M2) | seam as RS to owner + AG, about 4x fewer XGMI bytes | seam M16 32.9 -> 20.6 us; layer M16 556.0 -> 468.3 |
| b | `router_tp.py` | TP-shard the fp32 router (4 MiB/rank instead of 32 MiB) | router M16 115.3 -> 21.9 us |
| c | commit 640282b | NT (cache_modifier 2) on W13 / W2 loads | S3 -6..7% (M16 541.8 -> 502.8) |
| d | daf6947 | `rocdl.sched_barrier(0)` after each UG quarter's MFMAs | S3 M16 503 -> 455 us (UG 6.43 TB/s) |
| e | 16x16x128 scaled FP4 MFMA in UG/DOWN, replacing scalar decode + FMA | — | S3 M16 1108 -> 751 us |
| f | `LAGUNA_FOLD_RQ` | requant MID inside UG; one grid sync fewer | S3 M16 455.2 -> 447.3 |
| g | `LAGUNA_MS_PACK` (357ff5a) | pack MID e8m0 scales, one dword per expert. Mechanism: ATT showed DOWN is buffer_load-issue bound (52%) | TP4 M8 DOWN 183 -> 161 us |
| h | `LAGUNA_W2S_PAIR` (be7ea97), `LAGUNA_UGS_PAIR` (4909ee3), `LAGUNA_XS_PACK` (67344f9) | fewer scale VMEM instructions, same bytes | DOWN at the W2-only floor; TP8 layer 452.1 us |
| i | `LAGUNA_Q_EARLY` (a39f3cb) | S3 activation quant runs right after the logit push | layer -3.5..-5.6 us |
| j | `LAGUNA_TOPK_FAST` (7399139) | ordered-u32 keys, Batcher presort, `wave_umax` rounds | top-k 7.0 -> 2.4 us; layer -4.3..-5.8 us |
| k | `LAGUNA_U_SCATTER`, `LAGUNA_SH_SPACK` (98ec598) | expert-union build by LDS scatter; S4 scale runs | U scatter M16 -18.4 us; S4 runs TP4 -2.8..-3.3 us |
| l | `LAGUNA_SEAM_NORM_PAR`, `LAGUNA_SEAM_SS_AG` (fb3193b) | all-token norm; sum of squares carried with the AG | layer M16 418.3 -> 407.6 (TP8), 778.0 -> 763.6 (TP4) |

  - **Watch:** the layer kernel cache key once omitted the env switches (fixed in 7399139). Any
    layer A/B that only flips an env needs a key that includes it.

### O10. Smaller and enabling items
- **D, fused MX quant + sort for N <= 16384** (AITER `quant_kernels.cu`, 89fed3572): bit-exact, but
  serving is unchanged (933.6 / 344.8 / 15.85). The 1-GPU 36.2 -> 16.6 us was an eager-timing
  artifact (OPT231 "D result").
- **D1, `AITER_D2_DEFER_TID=1`:** moe2 -1.2% op level. Default off, and the switch was dropped in
  the upstream merge.
- **Enablement for H=16384 (no perf claim):**
  - AITER quant >8192 split path (6819e047a; package `AITER_LARGE_HIDDEN_WORKAROUND`; set to 0 on
    the new base).
  - `load_hip_runtime` fix (cfae83274): CustomAllreduce hung on a second HIP runtime.
  - mega_moe_tp NCK32 (eb42562c0).
- **Warm-vs-cold JIT:** the first M=8192 MoE call costs about 29 s. Always benchmark warm (OPT4
  "JIT finding").

### O11. Open / not adopted (trade-offs)
- **Chunk 32768 at 70k** (R1021 "chunked-prefill-size 8192 vs 32768"):
  - TTT/GPU tp8 +1.3..2.4%, tp4 +5.9..8.7%.
  - TTFT +28.5% (tp8) / +19.7% (tp4) at c2, from a scheduling effect.
  - User decision pending.

## 2b. Dead ends and negative results (all MI355X)

- **DP-attention `--dp 8` on EP8:** 229.1 tok/s, TTFT 9839 ms, TPOT 24.41. Attention GEMMs become
  replicated and the AR-fusion path is lost (OPT231 "T3 results").
- **Overlapping the prefill AR with the MoE on a side stream (incl. CU masks):** both kernels slow
  down. Best case hides 12.5% of one AR. The split penalty for 2 x M4096 is +17.5%. NO-GO
  (OPT231 "F M0").
- **QR INT6 / INT4:** error grows 4x / 16x for 2.5-3.8% projected gain (OPT231 "F follow-up").
- **aiter custom 2-stage AR at prefill sizes:** 1197 us vs QR INT8 1034 us, so prefill AR+norm fusion
  is weak (OPT231 "F2 M2"). AG by peer `copy_`: 1976 us.
- **mega_moe_tp (persistent fused MoE+AR):** +2.8% at M=16, +18.7% at M=64, +17% at M=8. M 1/2/4 are
  unsupported (OPT231 "T9", "Mono feasibility").
- **D2 persistent stream-K moe2:** k=1/2/4 at 271 / 188 / 148 us vs 139 default (OPT231 "D2 design").
- **moe2 tile_n 256:** -0.6%, noise, and confounded by occupancy.
- **moe1 tile / split-K swaps (B):** no movement. The wave-quantization hypothesis was falsified;
  moe1 is VMEM-issue / memory-side limited at 6.4-6.6 TB/s (OPT231 "B result").
- **Tuner decode rows:** EP1 M=16 521 vs 451 us, and EP8 tuner rows without EP masking regress. Not
  installed (OPT4).
- **FP8 router:** 56% of tokens get a different top-16 set (OPT231 "T4").
- **"2 x 1 ms decode AR":** a profiler artifact (rank skew in non-steady steps) (OPT231 "Item A").
- **Monokernel attempts (MONO CONTINUE HERE, "Do NOT retry"):**
  - UG tasks/iteration 2/4
  - UG prefetch depth 2/3 (spills)
  - LDS-staged weight / MID streams (4 variants)
  - seam with 2 pairs per lane
  - DOWN BATCH 2/6/8
  - DOWN first-batch prefetch before the grid sync
  - MID via L2-cached loads
  - UG stream-K (contiguous and tail-split)
  - wave-level UG tasks
  - fusing the shared expert into S3 (no gain)
  - a first all-token norm that polled 4 loads per spin (polling storm, seam 19.9 -> 27.0 us)

## 3. Where the final code lives and how to launch it

**Branches:**

| Component | Repo / branch | Head | Contains |
|---|---|---|---|
| AITER | `kkHuang-amd/aiter` `feat/proxy-model-1020` | `e9b0b5979` (merge with upstream b20f31339; pre-merge `605ade939`) | O3/O4 rows in `model_configs/proxymoe_fp4_tuned_fmoe.csv`, O5 b_nt fix, D fused quant, tuner fixes, quant split path, load_hip_runtime, NCK32 |
| SGLang | `HaiShaw/sglang` `feat/proxy-model-1020` | `ed3d8b0089` | O6 bf16x3 router, O7 prefill SP, monokernel API (#44353) |
| Monokernel | `kkHuang-amd/sgl-amd-mono` `feat/laguna-ffn-mono` | `fb3193b` used in the R1022 sweep | O9 |
| FlyDSL | 0.4.0 | — | required by the new AITER |

On the AITER commits for upstream PRs, see OPT231 CONTINUE HERE item 3.

**Files in this repo:**
- `patches/aiter_2bb549102_proxymoe_tuning.patch`: old-base AITER tuner fixes (HIDDEN_MAX bucket,
  `TUNE_MOE_STAGE2_ATOMIC_ONLY`, vectorized ref). Superseded by the branch.
- `patches/sglang_quark_mxfp4_cuda.patch`: Quark MXFP4 on CUDA for the GB300 runs (non-HIP bf16
  dequant). Not an MI355X optimization.
- `configs/proxymoe_fp4_tuned_fmoe.csv`: early 6-row gfx950 snapshot (EP1 inter 512 / 512 experts and
  EP8 inter 4096 / 64 experts, M 2048/4096/8192). The branch CSV has more rows.
- `configs/tuned_ep1_all14.csv`, `configs/tuned_ep8eq_topk2_all14.csv`: raw tuner output for 14 M
  values each. Every row is `gfx950`, `cu_num` 256.

**Final best launch (R1021 "Config", R1022 "Config", `fixed_seq_len/serve_proxy.sh`, `scripts/sweep_1022.sh`):**
- Environment:
  - `source /data/proxy_moe/mono_env.sh`. These paths are node-local to crsuse2-m2m-231:
    `newbase_env.sh` heads plus the sgl-amd-mono clone on PYTHONPATH.
  - `SGLANG_USE_AITER=1`
  - `SGLANG_OPT_LAGUNA_ROUTER_BF16X3=1`
  - `SGLANG_OPT_LAGUNA_PREFILL_SP=1`
  - `ROCM_QUICK_REDUCE_QUANTIZATION=INT8`
  - `AITER_LARGE_HIDDEN_WORKAROUND=0` on the new base
  - TP4 monokernel only: `SGLANG_AMD_MONO_LAGUNA_TP=4`
- Server:
  - `--tp-size 8 --ep-size 1` (`tp8ep1`, 24L), or `--tp-size 4` on GPUs 0-3 with `proxy_12l`
  - `--enable-aiter-allreduce-fusion --monokernel-backend amd`
  - `--kv-cache-dtype fp8_e4m3 --context-length 73728 --mem-fraction-static 0.88`
  - `--chunked-prefill-size 8192`
  - `--max-running-requests CONC --cuda-graph-max-bs-decode CONC`
  - `--watchdog-timeout 3600`
- Client: `sglang.bench_serving --backend sglang --dataset-name random-ids --tokenize-prompt
  --random-range-ratio 1.0 --request-rate inf`, np = CONC*8 (70k: CONC*4), warmup CONC*2.
- Short-bench repro, without monokernel or SP:
  `source /data/proxy_moe/newbase_env.sh; NL=24 TPN=8 SGLANG_OPT_LAGUNA_ROUTER_BF16X3=1 ONE="<tag> 1 1" bash /data/proxy_moe/repro.sh`
  (OPT231 CONTINUE HERE).

## 4. Porting checklist for gfx1250, ordered by MI355X gain

1. **O9 monokernel** (decode c<=16, up to -32.6% TPOT): needs a gfx1250 port of `laguna/gfx950/*`.
   This means WMMA instead of MFMA, the wave size, CU count / CTA plan, the LDS budget, the
   cache-modifier and coherence model, the A4W4 instruction availability, and the peer fabric.
2. **O1 EP1 vs EP8** (+13% throughput, -14% TPOT, +17% TTFT on MI355X): a flag A/B on MI455.
3. **O4 + O3 tuned MoE rows:** re-tune on gfx1250 / its CU count. The MI355X CSV rows will not match
   the key.
4. **O5 b_nt streaming loads in moe2** (+3.3% out tok/s EP1): verify the gfx1250 lowering.
5. **O7 prefill SP** (+3-5% at 70k) and **O8 QR INT8** (bf16 AR = -9.1% TTT at 70k): re-measure the
   RS / AG / AR op costs on MI455's fabric first.
6. **O6 bf16x3 router** (TTFT -6%) and **O2 AR fusion** (TPOT -2.6%): cheap A/Bs.
