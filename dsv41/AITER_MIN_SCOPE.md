# Minimal aiter scope for DSV4.1-Flash on the image aiter

Question (user, 2026-09-30 16:00): starting from the image's base aiter (`/workspace/tmp/aiter`), which aiter commits
must be added to reproduce our current DSV4.1-Flash numbers (SGLang rolao/dsv41/opt-branch with router fusion + sort MP)?

## CONTINUE HERE

**Status:** DONE. Min scope answered 18:26 (all pass, no #5561); follow-up (original DSV4.1 on the image aiter) answered
19:26 (runs, GSM8K 0.899). Experiment trees removed; stack kept as patches (see Follow-up).
**Next:** optional -- turn the local stack into a branch/PR on kkHuang-amd/aiter (not done, local only), or re-check
with #5722/#5660/#5519/#5579/#5575 if a workload outside AgentX c1/c2 (e.g. c32/c64, prefill-heavy) regresses.
**Repro:** `setsid nohup bash /workspace/claude-skills/dsv41/scripts/minscope_ab.sh > /shared_nfs/kk/results/DeepSeek-V4.1-Flash/atomport/minscope/run.nohup 2>&1 < /dev/null &`

## Setup

- **Image base aiter:** `/workspace/tmp/aiter` @ `acf8fdf93` (v0.1.21-120), prebuilt JIT `.so`, two uncommitted local
  mods (`csrc/cpp_itfs/torch_utils.py` torch.Stream handle, `pa_mqa_logits_fp4_prefill.py` unbounded kernel cache),
  identical to the ones in our live env. FlyDSL 0.3.2 (image). Not modified; copied to `/sgl-workspace/aiter-minscope`.
- **Candidate stack** (`/sgl-workspace/aiter-minscope`, local branch `minscope-acf8`, never pushed):

| # | local commit | source | why |
|---|---|---|---|
| 0 | a07606042 | image local mods | part of the image |
| 1 | ddd9fb5a9 | #5586 `2fe75b47c` (route untuned a8w8 blockscale GEMMs to Triton above a per-arch M) | prerequisite: #5750 conflicts in `gemm_op_a8w8.py` without it; Python only |
| 2 | 0c38f1b46 | #5750 `e2d019f15` (main squash; native group32 A8W8 Triton GEMM + V4.1 tunes) | required (group32 GEMMs) |
| 3 | 94de118da | #5967 content: MoE tuner GPU datagen + 23 a8w4 tuned rows as standalone `dsv41_tp2_sef_fp8fp4_tuned_fmoe.csv` | #5967 as-is edits `dsv41_fp4_*_fmoe.csv`, created by #5722; #5722 conflicts on acf8 and only adds A4W4 rows we do not use -> content only (user choice) |
| 4 | 857339acb | `aiter/utility/graph_alloc.py` only, from #5430 `6085899fa` | the #5750 group32 kernel imports it; full #5430 conflicts in gfx1250 a16w16 files |
| (5) | - | #5561 patch (FlyDSL stage1 LDS-DMA drain) | only if GSM8K < 0.89 |

- #5586 and #5430 are ancestors of the live env's `1053c79bb` but NOT of `b4d9154d1` or `acf8fdf93`; the earlier
  static analysis (SGLang-visible aiter symbols only) missed them because they are #5750's own prerequisites
  (a code-context dependency and an aiter-internal import). Found by actually cherry-picking + importing on acf8.
- `acf8fdf93` already has `moe_sorting_dispatch_policy` and the multi-phase opus sort (needed by the sort-MP change) and
  loads every `model_configs/*tuned_fmoe*.csv`. #5586 and #5750 touch no `csrc/`, so the image JIT `.so` are reused.
- Not in this base and not added (so their effect is part of the test): #5722, #5660, #5519, #5579, #5575.
- **SGLang:** `/sgl-workspace/sglang-minscope` = `5ec406bb76` + `bc8bae5146` (router fusion + sort MP) = the tree the
  reference numbers were measured on. Same launcher env as the reference (`--fp8-gemm-backend aiter
  --enforce-shared-experts-fusion`, OPUS sparse prefill, TP2 EP1, GPUs 4,5, chunk 16384, PDI 16, SIMULATE_ACC 3.51);
  PYTHONPATH = aiter-minscope + mori only (no pydeps-flydsl-0341).

## Results

- GSM8K (base scope, no #5561): 0.904 / 0.907 / 0.901, mean 0.904 (reference 0.897 / 0.901 / 0.900) -> #5561 NOT needed.
- AgentX c1 (ms_c16k_c1_pdi16): 11,332.2 / P90 348.1 (p50 386.0, TTFT p50/p90 0.78/1.31 s), 284 ok, 0 errors; vs reference
  11,371.9 / 349.5: -0.3% / -0.4% -> PASS. PYTHONPATH = sglang-minscope + aiter-minscope + mori (no aiter-5750, no
  pydeps-flydsl) confirmed in the driver log.
- AgentX c2 (ms_c16k_c2_pdi16): 11,749.3 / P90 325.0 (p50 382.3, TTFT p50/p90 0.46/1.03 s), 429 ok, 0 errors; vs
  reference 11,819.8 / 324.6: -0.6% / +0.1% -> PASS.

| check | reference (live env aiter-5750) | minimal scope (image acf8 + stack) | delta |
|---|---|---|---|
| GSM8K 1319 x3 | 0.897 / 0.901 / 0.900 | 0.904 / 0.907 / 0.901 | +0.005 mean |
| AgentX c1 TTT / P90 | 11,371.9 / 349.5 | 11,332.2 / 348.1 | -0.3% / -0.4% |
| AgentX c2 TTT / P90 | 11,819.8 / 324.6 | 11,749.3 / 325.0 | -0.6% / +0.1% |

## Answer

On top of the image aiter (`/workspace/tmp/aiter` @ `acf8fdf93` with its two uncommitted local mods, FlyDSL 0.3.2),
the minimum to reproduce the current DSV4.1-Flash numbers (GSM8K, AgentX c1/c2) is:

1. **#5586** `2fe75b47c` -- prerequisite of #5750 (without it the #5750 squash conflicts in `aiter/ops/gemm_op_a8w8.py`).
2. **#5430** `6085899fa` -- only the new file `aiter/utility/graph_alloc.py` (imported by the group32 kernel); the rest
   of #5430 (gfx1250 / a16w16) is not needed and conflicts on acf8.
3. **#5750** `e2d019f15` (main squash) -- native group32 A8W8 Triton GEMM + V4.1 tunes.
4. **#5967 content** -- the 23 DSV4.1 TP2 shared-experts-fusion a8w4 tuned FMoE rows (any `model_configs/*tuned_fmoe*.csv`
   file; a standalone CSV was used because #5967's target CSV is created by #5722) + the MoE tuner GPU datagen change
   (tuner only, not needed at runtime).

NOT needed for these numbers: #5561 (GSM8K 0.904 without it), #5722, #5660, #5519, #5579, #5575, #5802, a newer
FlyDSL. All four items are Python/CSV only, so the image's prebuilt JIT `.so` are reused (no C++ rebuild).
Caveat: validated on GSM8K + AgentX c1/c2 (decode-dominated, TP2, chunk 16384); c8-c64 and prefill-heavy traffic were
not re-run on this stack. Local branch: `/sgl-workspace/aiter-minscope` `minscope-acf8` (5 local commits, not pushed).

## Follow-up: original DSV4.1 SGLang on the unmodified image aiter (user 19:20)

Question: without upgrading aiter, does the ORIGINAL DSV4.1 SGLang (RolaoDenthu/sglang `3e8187fa88`, "[AMD] Allow
DeepSeek-V4.1 features on ROCm") run?
- SGLang `/sgl-workspace/sglang-orig41` (detached worktree @ 3e8187fa88); aiter = `/workspace/tmp/aiter` content
  (acf8fdf93 + 2 local mods) via the identical copy `/sgl-workspace/aiter-image` (keeps JIT caches out of the image dir);
  FlyDSL 0.3.2.
- Launch: colleague recipe unchanged except no later optimizations: EXTRA_ARGS empty (no --fp8-gemm-backend aiter, no
  --enforce-shared-experts-fusion), no OPUS sparse prefill env (does not exist at 3e8187fa88), chunk 4096 (recipe
  default); all recipe flags parse at 3e8187fa88 (`--tp` is the unique prefix of `--tp-size`).
- Script `scripts/orig41_ab.sh` (PID 945123, started 19:28), summary `/shared_nfs/kk/results/DeepSeek-V4.1-Flash/atomport/orig41/summary.txt`:
  GSM8K 1319 x3 on an EVAL_ONLY CONC=32 server, then AgentX c1, c2 (PDI 16, PREFIX orig41).
- Reference for the same era: rolao all-opts (09-26) c1 9,818.6 / 267.5, c2 9,832.8 / 258.2.

- **RUNS (19:26):** server ready in ~4 min (no missing-symbol errors), GSM8K 0.901 / 0.896 / 0.901, mean 0.899.
  MoE uses the a8w4 FlyDSL path but with the HEURISTIC fallback kernel choice (76 'using heuristic FlyDSL fallback'
  lines: no tuned rows for these shapes in acf8), plus 90 untuned bf16 GEMM shapes -> perf expected below the tuned env.
- AgentX c1/c2: STOPPED by user at 19:45 (c1 ~15 min in, no result). Answer to the question: YES, the original DSV4.1
  SGLang runs on the unmodified image aiter with correct accuracy (untuned MoE/GEMM shapes, so perf not measured).
- Cleanup 19:50: sglang-orig41 / sglang-minscope worktrees and the aiter-image / aiter-minscope copies removed.
  The minimal-scope stack is kept as patches: /shared_nfs/kk/results/DeepSeek-V4.1-Flash/aiter_minscope_patches/000{1..5}-*.patch
  (git am onto acf8fdf93 to rebuild it).
