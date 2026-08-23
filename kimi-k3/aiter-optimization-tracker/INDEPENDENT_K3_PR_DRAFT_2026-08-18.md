# Draft: independent Kimi-K3 gfx950 FlyDSL integrations

Proposed branch:

```text
HaiShaw/sglang:perf/k3-gfx950-independent-fusions-clean
base: sgl-project/sglang:main
```

Draft PR created:
[sgl-project/sglang#35287](https://github.com/sgl-project/sglang/pull/35287).

## Suggested title

```text
[AMD] Add independent Kimi-K3 gfx950 FlyDSL integrations
```

## Summary

- Vendor selected Kimi-K3 gfx950 FlyDSL kernels in SGLang with a fail-closed
  source selector.
- Add MLA output-gate, KDA group64/B2, 12-head AITER MLA decode-side padding,
  and an optional latent-tail integration.
- Use one cooperative preactivated MoE producer for exact M2/M4 while larger
  batches retain the existing fallback.
- Add the Kimi-K3 M16384 GEMM tuning profile.
- Require only the AITER revision/toolchain already pinned by main; do not
  require AITER #4617 or #4647.

## Explicit exclusions

This PR does not duplicate:

```text
#33599 attn-residual
#32796 K3 DCP
#34580 Triton MLA decode tune
#33838 MoE layout/copy
#34198 fused KDA decode boundary
#33916 MoE copy/cast removal
#34490 Radix-4 router
#34837 12-head prefill concat/cast
```

MLA Q/cache fusion is also excluded because main's AITER pin lacks the required
batch64/operator semantics.

## Validation

```text
Pinned AITER d9e5ef7 selected FlyDSL tests: 35 passed
Independent branch runtime smoke:
  C2: 8/8 requests, 3270.86 tok/s, 16.65 ms TPOT
  C4: 8/8 requests, 5222.32 tok/s, 19.46 ms TPOT
  GSM8K 50: 1.000, invalid 0.000
  GSM8K 1319: 0.953, invalid 0.001
Forbidden listed-PR path audit: passed
git diff --check: passed
```

Full pinned-AITER server startup could not be measured in the current
container because its installed FlyDSL does not match the old pin
(`T.f8` missing). Focused kernels passed against the pin; the runtime smoke used
the current compatible AITER checkout.

## Runtime flags

```bash
export SGLANG_USE_AITER=1
export SGLANG_AITER_K3_OPT=1
export AITER_FLYDSL_FORCE=1
export AITER_SITUV2_A8W4=1
export AITER_SITUV2_A4W4=0
export SGLANG_K3_FLYDSL_SOURCE=sglang

export SGLANG_K3_AITER_MLA_GATE=1
export SGLANG_K3_AITER_KDA_GROUP64=1
export SGLANG_K3_AITER_B2_FUSIONS=1
export SGLANG_K3_AITER_MOE_PREROUTE_FP8=1
export SGLANG_K3_PREROUTE_PREACTIVATED_SHARED=1
export SGLANG_K3_AITER_M16384_PROFILE=1
```

Optional latent tail remains default-off:

```bash
export SGLANG_K3_AITER_LATENT_TAIL_FP8=1
```

The 12-head padding path has no feature flag; it is selected automatically when
Kimi-K3 uses `--decode-attention-backend aiter`.

## Local commit stack

```text
ef3eb0453 [AMD] Vendor Kimi-K3 gfx950 FlyDSL integrations
8167d911e [AMD] Unify Kimi-K3 M2/M4 MoE preroute
cd484cffc [AMD] Add Kimi-K3 M16384 AITER profile
```
