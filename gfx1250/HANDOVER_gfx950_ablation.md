# Handover: gfx950 absorb-BMM ablation (test the quant-matching thesis)

## The idea (why this is the cleanest test)
Every gfx1250 component we checked is **equal-or-more precise** than gfx950 yet gfx1250
scores lower (0.85 vs 0.93). Leading thesis: the Quark PTQ model performs best under its
own quantization error pattern, and gfx1250's *more precise* substitutions are a
distribution mismatch. The one **non-MoE** path that differs for R1 (EXPERIMENT_LOG E25):
- **MLA absorb BMM** (`q_nope @ w_kc`, `attn_out @ w_vc`): **fp4 on gfx950**, **bf16 on
  gfx1250** (gfx950 runs `quark_post_load_weights` to mxfp4-quantize w_kc/w_vc; gfx1250
  skips it via `_use_aiter_gfx95=False`).

**Single-node A/B on gfx950** (no cross-node numerical diff, no confounds): force gfx950
to run the absorb in **bf16** (like gfx1250) and re-measure GSM8K.

## What to run on the gfx950 node
1. Copy `scripts/gfx950_disable_absorb_fp4_sitecustomize.py` (this dir) to a dir as
   `sitecustomize.py`, e.g. `/tmp/abl/sitecustomize.py`.
2. Launch your **normal working gfx950 recipe** (the one that gives 0.93 — TP2, a4w4 MoE,
   cuda-graph fine, radix fine), but add env:
   `PYTHONPATH=/tmp/abl:$PYTHONPATH SGLANG_DISABLE_QUARK_ABSORB_FP4=1`
   Confirm the log prints `[gfx950_abl] quark_post_load_weights -> bf16 absorb`.
3. Run the SAME GSM8K you use for 0.93 (e.g. `benchmark/gsm8k/bench_sglang.py
   --num-questions 200 --parallel 200 --port <port>`), and also a baseline run WITHOUT
   the env (should reproduce ~0.93) for a clean A/B.

## How to read it
- **0.93 -> ~0.85 (drops):** the **fp4 absorb (quant-matching) IS the gap**. gfx1250 is
  "too precise" in the absorb.
  - Fix on gfx1250 CANNOT be "run fp4 absorb": **gfx1250 A0 has no fp4-activation
    scaled-WMMA** (`V_WMMA_SCALE_F32_32X16X128_F4`), so `batched_gemm_afp4wfp4_pre_quant`
    would crash (SQC inst fault — the original reason a8w4 is used).
  - The only feasible gfx1250 route is an **a8w4 absorb** (fp8-act x fp4-weight, via the
    supported fp8 scaled-WMMA, mirroring the MoE a8w4 workaround). CAVEAT: a8w4 is still
    MORE precise than gfx950's fp4 act, so it may only PARTIALLY close the gap.
- **stays ~0.93 (no change):** the absorb is NOT the gap. Remaining suspect is the
  **a8w4-vs-a4w4 MoE scheme** itself. Next: cross-node dump (`HANDOVER_crossnode_dump.md`).
  (Do NOT try "force fp4 absorb on gfx1250" — it is infeasible on A0.)

## Notes
- This does NOT change the MoE (gfx950 stays a4w4). It only changes the absorb BMM
  weight prep, so it isolates the absorb precision cleanly.
- kv_b_proj is bf16 in the R1 checkpoint (excluded from quant), so on gfx950 `w` is bf16
  at the call; the hook simply skips the mxfp4 quant and returns the bf16 split.
- Report both numbers (with/without the env) back so we can act.
