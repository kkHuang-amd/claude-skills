# Kimi-K3 stale-JIT audit — 2026-08-11

## Motivation

The TP8 all-reduce investigation found one real stale-module error:
`custom_all_reduce.cuh` changed, but an existing
`module_custom_all_reduce.so` was reused because the transitive header change
did not invalidate the build.

This audit checks whether earlier negative AITER experiments had the same
problem.

## Method

- Start from clean AITER/SGLang integration branches.
- Use a new empty `AITER_JIT_DIR` for #4503/#4504.
- Use a fresh Triton cache for #4572/#4577.
- Enable each candidate path.
- Capture explicit coverage-matching TP8 traces.
- Require the candidate kernel name to appear in TP0; flags alone are not
  accepted as dispatch evidence.

Artifacts:

```text
stage2-runs/2026-08-11-stale-jit-audit/
```

## #4503 FP8 latent-MoE tail

Fresh-JIT B1 trace:

```text
11776 launches
latent_moe_tail_b1_bf16_fp8_persistent_gfx950_...
```

Decision: not stale.

Coverage caveat: the original endpoint gate used fixed-length C2/C4 requests.
Those requests remain at batch 2/4 and finish together, so the B1-only kernel
does not appear in a matching C2 trace. The old noise-level C2/C4 result does
not measure the covered path. Keep opt-in off for the selected production
workload, but treat C1 performance as not yet endpoint-validated.

## #4504 FP8 MoE pre-route/shared-down

Fresh-JIT B1 trace:

```text
11776 launches  kimi_k3_b1_tri_projection_bf16_fp8_gfx950_...
11776 launches  kimi_k3_b1_situ_shared_down_bf16_fp8_gfx950_...
```

Decision: not stale.

The same coverage caveat applies: the original fixed C2/C4 endpoint workload
does not enter the B1-only path. Its C1 endpoint effect remains unmeasured.

## #4572 attention-residual gate

Fresh Triton cache, explicit C1 trace:

```text
23 launches   attnres_fwd_kernel
349 launches  current _agg_kernel
```

Decision: not stale and the selective dispatch is active.

The candidate covers only `tokens=1`, `bank_rows=1`, and no fused bank write.
The previous C2 endpoint result is not a strong production gate for this
corner. The direct repeated microbenchmark still shows a small 3.7% kernel
gain, but only for a small subset of attention-residual calls. C1 A/B is
required before calling it endpoint-rejected.

## #4577 GDR KDA decode

Fresh cache C4 trace:

```text
69 launches
gdr_decode_bf16_kh12x128_vh12x128_q1_4w4x16_vs4_kda_0
```

Decision: not stale. The candidate covers C4 and was active in the same class
of workload where endpoint throughput regressed. The earlier rejection versus
the fully-fused production AITER KDA boundary remains valid.

## Audit conclusion

No audited PR result was invalidated by stale JIT.

However, stale-JIT and shape coverage are separate failure modes:

- #4503/#4504/#4572 compiled and dispatched correctly under matching B1/C1
  traces;
- their original C2/C4 fixed-length endpoint benchmarks did not meaningfully
  exercise the narrow B1 paths;
- #4577 did exercise its C4 path, so its negative E2E conclusion is strong.

Next action:

```text
fresh-JIT C1 paired A/B:
  baseline
  #4503 only
  #4504 only
  #4572 selective
```

Until that gate is run, keep #4503/#4504 opt-in off and classify #4572 as
coverage-inconclusive rather than endpoint-rejected.
