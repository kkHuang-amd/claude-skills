# Retained experiment runs

Raw profiler traces were removed after analysis. These folders retain commands,
logs, endpoint JSONL, CSV, summary JSON, and test evidence.

## Current canonical validation

```text
2026-08-12-fresh-integration
  fresh SGLang/AITER migration
  SGLang-vendored FlyDSL validation
  GSM8K, C2-C32, B2 profile
```

## Current decision evidence

```text
2026-08-11-b300-80pct
2026-08-11-b300-mi355x-comparison
2026-08-11-c1-fresh-jit-ab
2026-08-11-production-trace
2026-08-11-stale-jit-audit
2026-08-11-m8192-bf16-tuning
2026-08-11-pr4572-attn-res
2026-08-11-pr4577-gdr-decode
```

## Rejected route prototypes

```text
2026-08-11-route-sort-quant-prototype
2026-08-11-hip-route-quant-fusion
2026-08-11-v3-multiblock-route
2026-08-11-v3r-wave64-role-grid
2026-08-11-v4-route-prep
2026-08-11-generic-fusion
```

Do not rerun these without changing the architecture; see the corresponding
decision documents under `../aiter-optimization-tracker/`.

## Historical baseline and integration

```text
2026-08-06-*
2026-08-07-*
2026-08-10-optimization-execution
2026-08-10-pr-stack-validation
2026-08-10-batch1-fusions
```

These are retained for historical comparison. New work should use the
2026-08-12 fresh integration as its baseline.
