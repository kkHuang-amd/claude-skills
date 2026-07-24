# MegaMoE handover — resuming after a Docker image change (2026-07-20)

You are switching the container image (likely to get matched aiter/flydsl/**triton-custom** so the
triton a8w8 blockscale GEMM compiles — see "Why change the image"). This file is what you need to
**re-apply the (uncommitted) MegaMoE work in the new image** and re-verify.

Pairs with `MEGAMOE_HANDOFF.md` (full technical record) and
`FLYDSL_MEGAMOE_STAGE1_ANALYSIS.md` (why no further low-risk perf lever).

## Current status (before image change)
- FlyDSL MegaMoE backend works on DSV4 (V4-Pro, a8w4, EP8/DP8, cuda-graph). **gsm8k 0.94**.
- Equal-footing A/B (conc256 8k/1k, BOTH with the CK blockscale workaround on the current broken
  env): **megamoe 27,630 tok/s = 96% of dp 28,774**; TPOT ~equal; ITL +11% (the gemm1 MoE gap).
  Relationship to dp is unchanged from before the FlyDSL update.
- The updated FlyDSL `mega_moe_v1` (`dd215d29`, "packaged operator") gave **no perf change** to
  megamoe (micro-harness flat) — it only repackaged + renamed helpers.

## Why change the image (the problem the new image should fix)
Current env has: **aiter @ main (874840aef), flydsl 0.2.4, triton-custom 3.6.0**. The upgraded
aiter ships a new **triton a8w8 blockscale GEMM** (`gemm_a8w8_blockscale_preshuffle`) that DOES NOT
compile against the pinned triton-custom 3.6.0 → `make_ttgir: PassManager::run failed` during
cuda-graph capture (crashes **all** modes: dp, mori-ep, megamoe — not a megamoe bug).
Current mitigation: `AITER_DISABLE_BLOCKSCALE_TRITON=1` (falls back to CK; slower, hits all modes
equally). The new image (e.g. `rocm/atom-dev:latest`) should have matched aiter/flydsl/triton so
this workaround is unnecessary and perf returns to the faster triton-blockscale path.
See `../../sglang-prefill-coalescer/HANDOVER_blockscale_triton_compile.md` for the root cause.

## MUST re-apply after image change (uncommitted working-tree changes)
Patches saved in `megamoe_image_change_patches/` (this dir; on /dockerx = persists across images):
1. **sglang** `sglang_megamoe.patch` — environ.py knobs, mega_moe.py 3 dispatch hooks, fp8.py
   megamoe build hook, cohere2_moe.py `@strict` no-op, utils.cuh HIP `getSMVersion`.
   ```bash
   cd <sglang-repo> && git apply /dockerx/home/wunhuang/tmp/claude-skills/dsv4/megamoe/megamoe_image_change_patches/sglang_megamoe.patch
   cp .../megamoe_image_change_patches/mega_moe_flydsl.py python/sglang/srt/layers/moe/mega_moe_flydsl.py
   ```
   (mega_moe_flydsl.py is a NEW file — copy it, not in the patch.)
2. **FlyDSL** `flydsl_test_mega_moe_mtpr.patch` — the `--mtpr` micro-harness override (optional but
   useful). Needs FlyDSL branch `mega_moe_v1`.
3. **aiter** `aiter_blockscale_gate.patch` — the `AITER_DISABLE_BLOCKSCALE_TRITON` env-gate. **Only
   needed if the new image still has the triton-blockscale mismatch** (test first; skip if the new
   image's triton compiles the kernel).

If the new image ships its own aiter/flydsl/sglang, re-apply patches onto those (paths may differ;
the patches are small and self-explanatory).

## Runtime deps the new image must provide
- **FlyDSL workspace** for the MegaMoE kernel + `tests/` helpers. Point
  `SGLANG_AMD_FLYDSL_KERNELS_PATH` at it (unified script sets `/sgl-workspace/FlyDSL`).
  NOTE (2026-07-20 re-bringup): the `mega_moe_v1` head has moved PAST the handoff pin and
  drifted the interface in 3 ways, ALL handled in `mega_moe_flydsl.py` (no manual edit needed
  beyond the synced patch): (a) kernel package relocated `kernels.moe.mega_moe` →
  `kernels.mega_moe` (import try/except); (b) `fp4_utils` → `gemm_common_utils` (import
  try/except); (c) the stage1/2 fold removed `mega.stage1.w1`, weights now live on the
  instance as `mega._s1_w1` / `_s1_w1_scale` — `_swap_layer_weights` uses `hasattr(_s1_w1)`
  (the synced `mega_moe_flydsl.py` in this patches dir already has this fix).
- mori (STATIC_HEAP shmem), aiter, the V4-Pro checkpoint at `/dockerx/data/deepseek-ai/DeepSeek-V4-Pro`.

## How to launch + verify (new image)
```bash
cd /dockerx/home/wunhuang/tmp/useful-scripts/benchmarking/dsv4/
# unified script already has a `megamoe` MODE (added). MEM=0.65 (40G mori shmem heap).
# Add AITER_DISABLE_BLOCKSCALE_TRITON=1 ONLY if the new image still can't compile the triton kernel.
SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR=8192 MEM=0.65 MODE=megamoe PORT=8000 bash run_sgl_dsv4_unified.sh
# verify:
lm_eval --model local-completions --model_args model=/dockerx/data/deepseek-ai/DeepSeek-V4-Pro,base_url=http://localhost:8000/v1/completions,num_concurrent=128,max_retries=3,tokenized_requests=False --tasks gsm8k --num_fewshot 5   # expect ~0.94
# A/B (dp vs mori-ep vs megamoe): /workspace/ab_megamoe_driver.sh  (conc256 NP8/WARM2)
```
Gotchas (all already handled in the patched code, listed so you can sanity-check):
- a8w4 → gate_mode=interleave → w1 needs shuffle_weight_w4/shuffle_scale_w4 (else garbage).
- MORI_SHMEM_MODE must be STATIC_HEAP (default), NOT ISOLATION.
- mtpr=8192 (16384 overflows int32).
- cuda-graph needs the HIP getSMVersion (utils.cuh patch).

## Expected results (to confirm the new image is good)
- gsm8k ~0.94. If triton-blockscale works (no CK workaround), decode TPOT/ITL should be BETTER than
  the current CK-workaround numbers (dp TPOT ~60 not ~65; megamoe ITL ~41 not ~47) — i.e. back to
  the pre-upgrade A/B where megamoe was ~29.5k / dp ~30.7k, megamoe = 96% of dp.
- megamoe vs dp relationship should stay ~96% total / ~equal TPOT / ~11-13% ITL gap.

## Bottom line
MegaMoE is functionally done and at ~parity. No further low-risk perf lever (the ~11% ITL gap is
gemm1 dispatch+GEMM vs dp's opus/DP-attention; kernel-owner territory). The image change is to
restore the faster triton-blockscale attention path (a general env fix, not megamoe-specific).
