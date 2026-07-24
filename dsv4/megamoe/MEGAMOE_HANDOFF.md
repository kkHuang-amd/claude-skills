# FlyDSL MegaMoE on DSV4 (8×MI355X) — handoff (2026-07-16)

Bring-up + evaluation of the **FlyDSL MegaMoE** a2a backend for DeepSeek-V4-Pro on gfx950,
ported onto the DSV4 sglang fork. Bottom line: **works, at parity, shipped as an opt-in backend;
no further low-risk perf lever found.**

## Status
- **Functional + correct**: V4-Pro (Fp8MoEMethod fp4-experts), a8w4, EP8/DP8, cuda-graph.
  full-set **gsm8k 0.937** (n=1319, cuda-graph) = parity with baseline (~0.929).
- **Perf (8k/1k, conc256, same sglang base)**: total tok/s **megamoe 29,482** vs **dp 30,711**
  (96%) vs **mori-ep cap=128 30,203** (98%). TPOT ~equal (59.1 vs 59.8). Decode MoE is at
  parity with the fair EP baseline and **2.2× faster than uncapped aiter**.
- **Conclusion**: goal met. Remaining gap to dp (~4%) is dp's opus a8w4 (no dispatch, hardware-
  tuned) + queueing; not a megamoe defect. No worthwhile low-risk optimization remains (see below).

## Code (all on `/sgl-workspace/sglang-upstream`, branch `feat/mori-ep-decode-small-cap`)
Ported the FlyDSL MegaMoE integration (from PR sgl-project/sglang#31322) onto the fork:
- `python/sglang/srt/layers/moe/mega_moe_flydsl.py` — NEW (the FlyDSL backend).
- `python/sglang/srt/layers/moe/mega_moe.py` — 3 dispatch hooks (`_use_amd_flydsl_mega_moe`).
- `python/sglang/srt/layers/quantization/fp8.py` — megamoe build hook (gated on the FlyDSL flag).
- `python/sglang/srt/environ.py` — `SGLANG_AMD_FLYDSL_*` knobs.
- `python/sglang/jit_kernel/include/sgl_kernel/utils.cuh` — kept HIP `getSMVersion` (needed for
  cuda-graph on this ROCm build; stash-pop conflict resolution).
Working-tree only (not committed). Also a separate clone `/sgl-workspace/sglang-megamoe`
(lixiufei-leo:dev/megamoe-pr) used for the first bring-up.

### Fixes made during bring-up (vs the upstream PR)
1. **Import path**: PR imports `kernels.mega_moe`; current FlyDSL is `kernels.moe.mega_moe`
   (accept both).
2. **forward**: PR calls `forward_bf16`; branch exposes `forward` (accept both, `slice_output=False`).
3. **a8w4 weight layout (accuracy-critical)**: a8w4 → `gate_mode=interleave`, so w1 needs
   `fp4_utils.shuffle_weight_w4(gate_up=True)` + `shuffle_scale_w4`, NOT plain `shuffle_weight`.
   Feeding plain-shuffled w1 → coherent-looking garbage. This is the #1 gotcha (see the posted
   comment on PR #31322).
4. **cohere2_moe `@strict`** (hf_hub≥1.x) — no-op strict (already fixed in the fork).
5. **`MORI_SHMEM_MODE`**: must be STATIC_HEAP (default) NOT ISOLATION for MegaMoE (else
   "not in symmetric heap [0x0,0x0)" NULL-deref).
6. **mtpr**: use 8192 (default). 16384 overflows int32 in the symmetric buffer sizing
   (`struct.error 'i' format`).

## How to run
`run_sgl_dsv4_unified.sh` now has a `megamoe` MODE (added):
```bash
cd /dockerx/home/wunhuang/tmp/useful-scripts/benchmarking/dsv4/
MEM=0.65 SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR=8192 MODE=megamoe PORT=8000 bash run_sgl_dsv4_unified.sh
# needs the FlyDSL workspace: /sgl-workspace/FlyDSL @ branch mega_moe_v1 (kernels.moe.mega_moe)
```
- FlyDSL: `/sgl-workspace/FlyDSL`, branch `mega_moe_v1` (596ec44). compiled `flydsl` pip = 0.2.2.
- MEM=0.65 (megamoe reserves a 40G mori shmem heap; 0.90 would OOM the KV cache).
- A/B driver: `/workspace/ab_megamoe_driver.sh` (dp / mori-ep cap=128 / megamoe, conc256 NP8/WARM2).

## Env knobs added (default-off, safe)
- `SGLANG_AMD_USE_FLYDSL_MEGA_MOE=1` — select FlyDSL over DeepGEMM megamoe.
- `SGLANG_AMD_FLYDSL_KERNELS_PATH` — FlyDSL workspace (unified script sets it).
- `SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR` (8192), `SGLANG_AMD_FLYDSL_MEGA_QUANT`.
- `SGLANG_AMD_FLYDSL_MEGA_TUNE_TOKENS` (experiment; stage1 tile — only 4%, leave 0).
- `SGLANG_AMD_FLYDSL_MEGA_DECODE_MTPR` (experiment; two-instance — HANGS, leave 0, see below).

## Traces (decode, conc128 8k/2000, cuda-graph, 8 ranks each) — for per-layer analysis
- `/workspace/trace_megamoe`   — megamoe mtpr=8192 (the canonical one)
- `/workspace/trace_dp`        — dp (non-EP) baseline
- `/workspace/trace_moriep`    — mori-ep cap=128
- `/workspace/trace_megamoe_tuned` — megamoe tune_tokens=64 (tile experiment)
- `/workspace/trace_mtpr512`   — megamoe mtpr=512 (compact-transition experiment)
- **B200 reference**: `/sgl-workspace/trace_hcOFF_msOFF_c256` — B200 DeepGEMM megamoe (conc256) +
  `server.log`. Shows the single fused `sm100_fp8_fp4_mega_moe_impl` kernel.
- Analysis method: cuda-graph durations ARE real on this build; take per-layer x61 kernels of ONE
  decode step, exclude spin-inflated glue (index_elementwise/alloc_decode). See ../TRACE_PROFILING.md
  + ../MORI_EP_DECODE_ROOTCAUSE.md §5x.

### Per-layer decode breakdown (conc128, ms/step, real compute)
| role | megamoe (mtpr8192) | dp | mori-ep |
|---|---:|---:|---:|
| MoE GEMM (gemm1+gemm2) | 17.5 (222+64us/layer) | 11.1 | 13.8 |
| comm (a2a / collective) | 2.18 (combine; dispatch fused into gemm1) | 2.01 (allgather+RS) | 6.19 (dispatch+combine) |
| **MoE total /step** | ~20.0 (mtpr8192) / 17.2 (mtpr512) | 15.3 | 20.0 |
megamoe stage1 `moe_gemm1` fuses dispatch+gemm1 (222us); vs the fair aiter `EpDispatch(30)+mfma_moe1(113)+glue(~50)≈193` it is competitive. stage2/combine are faster than mori-ep.

## Options evaluated and NOT taken (with reasons)
1. **stage1 tile tuning** (`tune_tokens`): −4% only. Not the lever.
2. **smaller mtpr, single instance**: decode faster but forces smaller prefill chunk → **e2e worse**
   (28.0k vs 29.5k). Wash.
3. **two-instance (prefill 8192 + decode small mtpr)**: right idea, but **HANGS** — separate
   comb_op/shmem/a2a state on the shared mori heap deadlocks under continuous batching (even after
   fixing the lazy-build hang with pre-build). Needs isolated shmem groups; fragile. Left off.
4. **Micro-harness finding (decisive)**: within compact mode mtpr is **irrelevant** (2048→8192
   flat); the server 512↔8192 delta was the compact/non-compact **mode switch** (+11.6%), NOT
   padding. megav1 compact is already 2.2× faster than uncapped aiter. See
   `FLYDSL_MEGAMOE_STAGE1_ANALYSIS.md`.
5. **combine-reduce fusion into gemm2 epilogue**: ~6% decode compute ceiling, but decode is already
   at parity (not the e2e bottleneck) AND it's the same e_vec=2 MFMA acc-layout dead-end as the
   masked-MoE work. High risk, low ROI.
6. **prefill/TTFT**: megamoe prefill compute is FASTER than dp (7360 vs 7005 tok/s); the higher
   TTFT is a queueing effect of the ~4% total-throughput deficit, not a prefill kernel gap.

## Structural gap vs B200 (for context)
B200 DeepGEMM megamoe = a SINGLE fused kernel (`sm100_fp8_fp4_mega_moe_impl`, gemm1+act+gemm2,
dynamic-M): 8.6us/token. AMD FlyDSL = single OP but 3 kernels (gemm1/silu/gemm2): ~17.9us/token
(~2×). Closing it needs CUDA-style gemm1+gemm2 fusion, but the payoff at decode is modest
(a2 small, compute-bound) and ~half the 2× is hardware (B200 fp4 MMA) that fusion can't recover.

## Related docs
- `FLYDSL_MEGAMOE_STAGE1_ANALYSIS.md` — the stage1/compact deep-dive (corrected root cause).
- `../FLYDSL_KERNEL_AUTHORING.md` — the kernel-authoring playbook used here.
- PR feedback posted: sgl-project/sglang#31322 (a8w4 interleave weight-prep + pin FlyDSL commit).
