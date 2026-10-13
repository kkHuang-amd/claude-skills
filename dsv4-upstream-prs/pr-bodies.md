Owner node: crsuse2-m2m-243

# PR drafts for the 8 branches (HaiShaw/sglang -> sgl-project/sglang main)

Each block replaces "## Motivation" .. "## Checklist" of the upstream template; Checklist / Review sections and
the bot's CI block stay untouched. Every body ends with the co-author line (user request). Speed rows marked 8K in / 1K out are ours (2 reps per point, speed/results parsed in speed/parsed_243.md).
Accuracy setup for "ours": GSM8K 1319 q 5-shot, DeepSeek-V4-Pro, MI355X x8, TP8 + DP-attention 8 + EAGLE (3 steps),
fp8 KV. Baselines: main + current aiter pin 0.945, main + aiter 02575f1c3a 0.953.

---
## P1 — head `amd/dsv4-fp8-unified-kv-decode`
**Title:** [AMD][DSV4][1/8] fp8 unified_kv decode/verify on FlyDSL v4 nm decode with a latency-model split plan

## Motivation
DeepSeek-V4 fp8 unified_kv decode/verify on gfx950 runs the asm v4 nm kernel per token plus a separate merge and inverse-RoPE quant. This moves it to aiter's FlyDSL v4 nm fused decode and sizes the asm fallback's KV split from aiter's latency model. Includes #41383 (not merged yet).

Depends on: #41383 (open, included here); ROCm/aiter#6126 (in aiter main, needs the aiter pin bump in #44035); ROCm/aiter#5882 (open, carried in-tree until it lands).

## Modifications
- fp8 unified_kv decode/verify use aiter's FlyDSL v4 nm decode (#41383); the kernel is carried in-tree until ROCm/aiter#5882 lands (`SGLANG_DSV4_FLYDSL_DECODE`, default on; `SGLANG_DSV4_FLYDSL_DECODE_SOURCE=auto|aiter|intree`, default auto; 128 local heads opt-in via `SGLANG_DSV4_FLYDSL_DECODE_H128`, default off).
- asm fallback: KV split count from aiter's v4 nm latency model with a constant split_indptr (`SGLANG_DSV4_FP8_DECODE_SPLIT_MODEL`, default on; `SGLANG_DSV4_FP8_DECODE_SPLIT_CTX`, default 200000).
- HCA layers with 128 local heads use aiter's persistent `mla_decode_fwd_v4_nm_ps` (ROCm/aiter#6126) from 15 rows (`SGLANG_DSV4_FP8_HCA_PERSIST`, default on).
- Tests: `test_flydsl_v4nm_decode.py`, plus unified fp8 decode tests.

## Accuracy Tests
GSM8K: main 0.945 -> 0.951 (current aiter pin); 0.953 -> 0.948 (aiter 02575f1c3a).

## Speed Tests and Profiling
From #41383 (8K in / 1K out, step time stock asm -> FlyDSL): TP8 c8 13.28 -> 12.75 ms (-3.9%), TP8 c32 17.13 -> 16.61 ms (-3.1%), DP8 attention c64 21.24 -> 21.01 ms (-1.1%).
Split plan + HCA persistent (asm path, used with DP attention), DP8 attention + EAGLE, median ITL on vs off: 8K in / 1K out c16 6.71 vs 6.76 ms, c64 9.32 vs 9.32 ms; 64K in / 1K out (context length 200000) c8 6.26 vs 6.26 ms, c32 7.60 vs 7.53 ms. No measurable difference in these cells.

Co-authored-by: AMD-yanfeiwang <256076023+AMD-yanfeiwang@users.noreply.github.com>

---
## P2 — head `amd/dsv4-fp8-verify-swa-ring`
**Title:** [AMD][DSV4][2/8] fp8 verify: write the SWA ring in the fused norm/rope kernel

## Motivation
In fp8 target-verify the SWA ring was written by separate copy and scatter launches after the fused norm/rope kernel. Writing it inside that kernel removes 183 launches per verify step.

Depends on: none. Same change as #41905 (open).

## Modifications
- The fused norm/rope kernel writes the SWA ring directly in fp8 verify (same diff as #41905).
- No new env var.

## Accuracy Tests
GSM8K: main 0.945 -> 0.951.

## Speed Tests and Profiling
Verify step, DSpark block 6, 8K in / 1K out: c16 26.94 -> 26.38 ms (-2.1%), c64 45.94 -> 45.33 ms (-1.3%).

Co-authored-by: AMD-yanfeiwang <256076023+AMD-yanfeiwang@users.noreply.github.com>

---
## P3 — head `amd/moe-small-batch-sort`
**Title:** [AMD][MoE][3/8] small-batch sort: histogram + bitonic sort for 32 < P <= 256

## Motivation
The MoE align/sort kernel for 32 < P <= 256 routed pairs is several times slower than needed at decode batch sizes. A histogram + bitonic sort kernel covers this range.

Depends on: none. Same change as #41900 (open).

## Modifications
- New HIP sort kernel for 32 < P <= 256 (same diff as #41900); other sizes keep the existing kernels.
- Output is bit-identical to the existing kernels. No new env var.

## Accuracy Tests
Bit-identical output (unit test 192/192); GSM8K before 0.944 / 0.945, after 0.948 / 0.944 (2 boots).

## Speed Tests and Profiling
Sort launch, E=385 topk 7: P=133-252 14.8 -> 5.2 us, P=70-126 9.5 -> 4.3 us.
Decode step: plain c32 16.41 -> 15.78 ms (-3.9%), DSpark b6 c4 18.20 -> 17.48 ms (-4.0%).

Co-authored-by: AMD-yanfeiwang <256076023+AMD-yanfeiwang@users.noreply.github.com>

---
## P4 — head `amd/dsv4-c4-indexer-qlora-quant-reuse`
**Title:** [AMD][DSV4][4/8] reuse the q_lora fp8 quant for the C4 indexer wq_b

## Motivation
The C4 indexer `wq_b` quantized `q_lora` again although the attention `wq_b` already has the same fp8 tensor. Reusing it removes one quant launch per C4 layer.

Depends on: none. Same change as #41903 (open).

## Modifications
- The C4 indexer `wq_b` takes the attention path's fp8 `q_lora` and scales (same diff as #41903).
- No new env var.

## Accuracy Tests
GSM8K: main 0.945 -> 0.948.

## Speed Tests and Profiling
Decode step: plain c8 11.87 -> 11.79 ms (-0.7%), DSpark b6 c2 14.85 -> 14.76 ms (-0.6%); c16/c64 within noise.

Co-authored-by: AMD-yanfeiwang <256076023+AMD-yanfeiwang@users.noreply.github.com>

---
## P5 — head `amd/dsv4-fp4-indexer-rowgroup`
**Title:** [AMD][DSV4][5/8] FP4 indexer on aiter's row-group kernel for decode and target-verify

## Motivation
The FP4 indexer re-reads each key per query row. aiter's row-group FP4 MQA logits kernel (ROCm/aiter#6145) shares key loads across a request's rows. This uses it for V4.1 low-ratio layers and, opt-in, for the DeepSeek-V4 C4 indexer. Includes #43073 (not merged yet).

Depends on: #43073 (open, included here); ROCm/aiter#6145 (in aiter main, needs the aiter pin bump in #44035).

## Modifications
- V4.1 low-ratio indexer on the row-group kernel with candidate-only scoring (#43073; `SGLANG_HIP_FP4_INDEXER_ROWGROUP`, default on for HIP; page-8 index-K pool `SGLANG_HIP_DSV41_INDEX_K_PAGE8`, default off).
- DeepSeek-V4 C4 indexer decode and target-verify (>= 64 rows) on the row-group kernel, workspace built inside the graph (`SGLANG_HIP_DSV4_C4_INDEXER_ROWGROUP`, default off).
- All three paths fall back to the schedule kernels / page-64 pool with a warning when the installed aiter has no row-group kernel; tests skip there.

## Accuracy Tests
GSM8K: main 0.953 -> 0.948 (default), 0.946-0.954 with `SGLANG_HIP_DSV4_C4_INDEXER_ROWGROUP=1` (aiter 02575f1c3a).

## Speed Tests and Profiling
From #43073, DeepSeek-V4.1-Flash TP2, 68000 in / 200 out, total throughput: c1 +0.7%, c4 +1.2%, c16 +2.4%.
C4 rowgroup, `bench_dsv4_c4_indexer_rowgroup_hip.py`, indexer time per step over 30 C4 layers incl. workspace, max context 65536: decode bs16 ctx 8K 197 -> 117 us (1.68x), bs64 ctx 64K 621 -> 525 us (1.18x); 4-row verify bs64 ctx 64K 1262 -> 1076 us (1.17x), bs16-32 ctx 8K about 5% slower.

Co-authored-by: AMD-yanfeiwang <256076023+AMD-yanfeiwang@users.noreply.github.com>

---
## P6 — head `amd/dsv4-dense-fp8-mxscale`
**Title:** [AMD][DSV4][6/8] optional e8m0 mxscale GEMM for dense 128x128 fp8 linears

## Motivation
DeepSeek-V4's dense block-fp8 weight scales are powers of two, so the dense linears can run on aiter's e8m0 mxscale bpreshuffle GEMM on gfx950, which is faster than the fp32-scale GEMM for a set of shapes.

Depends on: ROCm/aiter#5896 (in aiter main, needs the aiter pin bump in #44035).

## Modifications
- Whitelisted (N, K) dense linears get an e8m0 copy of their weight scale at load and run the e8m0 GEMM; other shapes and non-power-of-two scales stay on fp32 (`SGLANG_DSV4_DENSE_MXSCALE`, default off; `SGLANG_DSV4_DENSE_MXSCALE_SHAPES`, default built-in list).
- Fused input_norm / q_norm producers emit ue8m0 scales when every consumer uses the e8m0 GEMM, so no launch is added.
- Stays off with a warning when the installed aiter has no e8m0 bpreshuffle dispatch; tests skip there.

## Accuracy Tests
GSM8K: main 0.953 -> 0.951 with `SGLANG_DSV4_DENSE_MXSCALE=1` (aiter 02575f1c3a).

## Speed Tests and Profiling
8K in / 1K out, DP8 attention + EAGLE, median ITL off -> on: c16 6.78 -> 6.68 ms (-1.5%), c64 9.80 -> 9.16 ms (-6.5%). Output throughput is within run-to-run noise (about 5%).

Co-authored-by: AMD-yanfeiwang <256076023+AMD-yanfeiwang@users.noreply.github.com>

---
## P7 — head `amd/dsv4-intree-gemm-tune`
**Title:** [AMD][DSV4][7/8] carry aiter DSv4 GEMM tuned rows and mHC tile_m=16 until aiter ships them

## Motivation
Three open aiter PRs (ROCm/aiter#6004, #6029, #6030) only change which existing kernel aiter picks for DeepSeek-V4 shapes. Carrying them in sglang gives the speedup on the current aiter pin.

Depends on: ROCm/aiter#6004, ROCm/aiter#6029, ROCm/aiter#6030 (open, carried in-tree until they land).

## Modifications
- Tuned GEMM rows replace (never append) aiter's rows for the same key; Triton configs and the mHC tile_m=16 decode policy are applied at model init (`SGLANG_AITER_INTREE_GEMM_TUNE`, `SGLANG_AITER_INTREE_MHC_TILE_M16`, both default on).
- Each item is skipped automatically once aiter ships the change. Test: `test_aiter_intree_tune.py`.

## Accuracy Tests
GSM8K: main 0.945 -> 0.948.

## Speed Tests and Profiling
Microbench vs aiter 02575f1c3a: GEMM time -8.7% to -8.9% per tuned shape, mHC -7.2%; same as an aiter build with the PRs.

Co-authored-by: AMD-yanfeiwang <256076023+AMD-yanfeiwang@users.noreply.github.com>

---
## P8 — head `amd/rocm-dockerfile-aiter-pin`
**Title:** [AMD][DSV4][8/8] gfx950 ROCm images: pin aiter to 02575f1c3a

## Motivation
{P1}, {P5} and {P6} need aiter features that are in aiter main but not in the current pin: the v4 nm latency model and persistent decode (ROCm/aiter#6126), the row-group FP4 MQA logits kernel (#6145) and the e8m0 mxscale bpreshuffle GEMM.

Depends on: ROCm/aiter#6042 (open, applied as a resolved patch). Needed by {P1}, {P5}, {P6}.

## Modifications
- `AITER_COMMIT_DEFAULT` of the gfx950-rocm720 / gfx950-rocm724 stages -> 02575f1c3a; other stages unchanged.
- ROCm/aiter#6042 conflicts with #6159 on the new pin, so a resolved patch is applied instead of a cherry-pick. The patch has no trailing whitespace, so pre-commit passes.

## Accuracy Tests
GSM8K: main 0.945 (current pin) -> 0.953 (02575f1c3a).

## Speed Tests and Profiling
N/A (image build only).

Co-authored-by: AMD-yanfeiwang <256076023+AMD-yanfeiwang@users.noreply.github.com>
