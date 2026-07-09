# Handover: gfx1250 absorb-BMM quantization emulation (complementary to gfx950 ablation)

## Context
gfx1250 A0 has **no fp4-activation scaled-WMMA**, so it CANNOT run the real a4w4 absorb
gemm — that would crash (SQC inst fault; the original reason a8w4 is used). To still test
whether *quantizing the MLA absorb* changes gfx1250 accuracy, we **emulate the quantized
numerics in bf16** (MXFP4/MXFP8 quant->dequant, then the normal bf16 BMM). No fp4 kernel
needed; runs fine on gfx1250.

This complements the gfx950 ablation (`HANDOVER_gfx950_ablation.md`):
- gfx950 ablation: bf16 absorb ON gfx950 — does REMOVING gfx950's fp4 absorb drop 0.93?
- gfx1250 emulation (this): fp4/fp8 absorb ON gfx1250 — does ADDING absorb quant raise 0.85?
Together they bracket the "quantization-matching" thesis.

## Tool (ready): weight-side emulation
`scripts/gfx1250_absorb_quant_emul_sitecustomize.py`, env `SGLANG_ABSORB_QUANT_EMUL`:
- `w_fp4`: replaces attention `w_kc`/`w_vc` with MXFP4 q-dq (bf16). Absorb becomes
  **a16w4** (fp4 weight, bf16 activation). Robust / structure-independent (only touches
  `self.w_kc` / `self.w_vc` on the attention modules at first forward).

Run on gfx1250:
```
PYTHONPATH=/tmp/emul:$PYTHONPATH SGLANG_ABSORB_QUANT_EMUL=w_fp4 bash run_ds-r1.sh
# then GSM8K as usual; compare vs baseline (env unset). Look for [absorb_emul] in log.
```

### Interpretation of the weight-side result
- **0.85 -> rises toward 0.93**: quantizing the absorb WEIGHT to fp4 (matching gfx950)
  recovers accuracy => quant-matching confirmed; the fix direction is "quantize the absorb
  on gfx1250". Next, do the ACTIVATION side to see how close a8w4 (feasible) vs a4w4
  (gfx950, not feasible on A0) gets.
- **no change / drops**: weight-quant of the absorb is not the lever.

## Activation side (a4w4 vs a8w4) — SKELETON to wire against live forward_mla.py
The activation quant must be applied to the absorb BMM inputs (`q_nope` before
`q_nope @ w_kc`, and `attn_output` before `attn_output @ w_vc`). The exact lines shift by
commit, so wire this after reading the live
`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
(bf16 absorb path: the `else` branch around the `w_kc.dtype==uint8` check, ~L495, and the
w_vc BMM ~L874). Use the q-dq helpers already in the tool
(`mxfp4_qdq` for a4w4-emul activation, `mxfp8_qdq` for a8w4-emul activation):

```python
# inside the bf16 absorb path, BEFORE the bmm:
if _MODE == "a4w4":
    q_nope = mxfp4_qdq(q_nope)      # emulate gfx950 exactly (fp4 act x fp4 weight)
elif _MODE == "a8w4":
    q_nope = mxfp8_qdq(q_nope)      # feasible-on-hardware precision (fp8 act x fp4 weight)
# ... existing bf16 bmm(q_nope, w_kc) ...   (w_kc already fp4-qdq by the weight hook)

# and similarly for attn_output before @ w_vc.
```
Cleanest wrap without editing the repo: monkeypatch `DeepseekV2AttentionMLA.forward_absorb_prepare`
and `forward_absorb_core` (re-call the originals is hard since the q-dq is mid-method), OR
temporarily edit forward_mla.py directly for the experiment (revert after). A repo edit is
acceptable here since it's an experiment, not the shipping recipe.

### Interpretation of the full emulation
- **`a4w4` emul on gfx1250 -> ~0.93**: CONFIRMS quant-matching and gives the ceiling. But
  a4w4 is NOT runnable on A0 hardware, so this is the (unreachable) ideal.
- **`a8w4` emul on gfx1250 -> between 0.85 and 0.93**: shows how much the FEASIBLE route
  (fp8-act x fp4-weight absorb, implementable via the supported fp8 scaled-WMMA) can
  recover. This would be the actionable target: implement a real a8w4 absorb BMM on
  gfx1250 (mirror the MoE a8w4 path) instead of bf16.
- If `a8w4` emul stays ~0.85 while `a4w4` emul reaches ~0.93, the gap is intrinsic to A0
  (needs fp4-act, which A0 lacks) — a hardware limitation, not fixable in software.

## Notes
- This is a bf16 EMULATION of quantized numerics — it measures the ACCURACY effect only,
  not performance, and it never invokes an fp4 kernel (safe on A0).
- The provided `mxfp4_qdq` / `mxfp8_qdq` are faithful MX round-trips (e2m1/e4m3 + per-32
  e8m0 scale) but not bit-identical to aiter's kernels; fine for an accuracy trend test.
- If a real a8w4 absorb kernel is later built, validate it the same way as the MoE
  (op-test vs quant-matched torch ref) before trusting it.
