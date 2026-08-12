# Handover: run the E30 bf16 MoE emulation ON gfx1250 (decisive kernel-vs-scheme test)

## Why (the one question this answers)
E30 (on gfx950) showed the a8w4 MoE **scheme** is accuracy-neutral: an *idealized* bf16
emulation of a8w4 (fp8 act x fp4 weight, bf16 matmul) scores **0.925 == a4w4** (40Q), matching
native a4w4 ~0.93. Yet the **gfx1250 real flydsl a8w4 kernel scores ~0.85** for the same scheme.

So the gap is either (A) the gfx1250 **flydsl real-kernel numerics** (fixable), or (B) something
about gfx1250 execution that degrades even ideal bf16 a8w4 (deeper). The bf16 emulation is
**arch-independent** (bf16 GEMM works on gfx1250; only fp4/fp8 kernels fault there, E3), so
running the SAME emulation on gfx1250 decides it — **no new kernel needed**:

- **gfx1250 emul-a8w4 ~= 0.925** (>> the real 0.85) => the flydsl a8w4 kernel IS the culprit
  (case A). Fix its large-token/max_m numerics (E16) or replace it. gfx1250 bf16 path is fine.
- **gfx1250 emul-a8w4 < ~0.92** => case B; a new kernel won't help; investigate gfx1250
  execution deeper.

## Prereqs on the gfx1250 box
- The modified sglang tree (this skill's recipe) — `sglang_gfx-1250` equivalent — with the
  working gfx1250 a8w4 recipe (AITER_FORCE_A8W4=1, split_k1 forced, Triton qk-rmsnorm, etc.).
- The hook file `scripts/moe_emul_sitecustomize.py` (from this skill dir). It:
  - neuters `shuffle_weight` / `e8m0_shuffle` / `moe_shuffle_scale` (handles BOTH the gfx950 and
    the gfx1250 `_is_gfx1250` weight-prep branches) so weights keep the clean fp4 layout;
  - replaces `AiterRunnerCore.run` with a bf16 grouped FFN, **bypassing aiter.fused_moe and the
    flydsl kernel entirely**. Env `SGLANG_MOE_EMUL` in {a4w4, a8w4, a16w4}.

## Steps
```bash
mkdir -p /tmp/moeemul && cp <skill>/scripts/moe_emul_sitecustomize.py /tmp/moeemul/sitecustomize.py

# Launch on gfx1250 with the emul hook. Start from your WORKING gfx1250 launch (the one that
# serves a8w4 at ~0.85), and ADD:
#   PYTHONPATH=/tmp/moeemul:<sglang>/python:$PYTHONPATH
#   SGLANG_MOE_EMUL=a8w4
#   --disable-cuda-graph            # the Python hook can't run under cuda-graph replay (REQUIRED)
#   export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
# and keep AITER_FORCE_A8W4=1 etc. (harmless; the MoE is replaced by the emul anyway).
#
# MEMORY on gfx1250 TP1 (model ~353 GB bf16-dequant on ONE ~432 GB card): the emul needs
# ~20-25 GB bf16 dequant scratch. Set --mem-fraction-static ~0.88 (NOT the 0.70 used for the
# gfx950 TP2 script — at TP1 the model won't fit at 0.70). If OOM: lower mem-fraction / KV,
# or reduce --max-running-requests / context. If gfx1250 supports TP2, prefer TP2 (more room).
```

Confirm in the log: `[moe_emul] installed, MODE=a8w4 ...` and `[moe_emul] MODE=a8w4 active=... E=...`.
Sanity `curl /generate` (coherent, not token-0). Then GSM8K:
```bash
python3 -m sglang.test.few_shot_gsm8k --num-questions 40 --parallel 40 --port <port>
```

## Runs to do (each ~tens of min; emul is slow, ~few tok/s)
1. `SGLANG_MOE_EMUL=a8w4`  — THE test. Compare to gfx1250's real-flydsl a8w4 (~0.85).
2. `SGLANG_MOE_EMUL=a4w4`  — cross-check; should also be ~0.925 (bf16 path sanity on gfx1250).
3. (optional) the real gfx1250 a8w4 (no emul, env unset) as the 0.85 anchor if not already have it.

## Read it
| gfx1250 emul-a8w4 (40Q) | meaning |
|-------------------------|---------|
| ~0.92-0.93 (== gfx950 emul, >> 0.85) | **flydsl real a8w4 kernel is the gap** (case A). Actionable: fix `moe_grouped_gemm_mxscale_gfx1250.py` / `gemm_mxscale_gfx1250.py` large-token numerics (E16 growth), validate END-TO-END vs 0.925 (small-token op-test is blind to it, E18/E30). |
| ~0.85 (== real flydsl) | even ideal bf16 a8w4 degrades on gfx1250 (case B) => not the flydsl kernel; investigate gfx1250 execution / a different non-MoE gfx1250 effect. A new triton kernel would NOT help. |

## Note on "one triton a8w4 kernel for both platforms" (the tempting alternative)
Writing a real triton a8w4 kernel to run identically on both is risky as a FIRST step:
aiter's existing **triton fp4 kernels FAULT on gfx1250** (E3: `gemm_a8wfp4`/`gemm_afp4wfp4` ->
memory access fault); gfx1250's working a8w4 backend is flydsl/gluon, not triton. So a triton
kernel may not even run on gfx1250. Do the bf16-emul comparison FIRST (this doc) — it's the
arch-independent "universal a8w4" for the diagnosis and needs no kernel. Only pursue a triton
kernel for perf/production AFTER (a) emul confirms case A, and (b) a minimal triton fp8-scaled
MX-WMMA is verified to compile+run on gfx1250.
