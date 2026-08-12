# AITER Kimi-K3 optimization tracker

This directory is the durable index for Kimi-K3 optimization work in
[`ROCm/aiter`](https://github.com/ROCm/aiter). Keep PR status, branch-only work,
mainline landings, benchmark claims, and local validation results here so a new
session can resume without reconstructing the GitHub audit.

## CONTINUE HERE

- Fresh-clone migration result:
  [`FRESH_INTEGRATION_2026-08-12.md`](FRESH_INTEGRATION_2026-08-12.md)
- Fresh AITER dependency matrix:
  [`AITER_DEPENDENCY_MATRIX_2026-08-12.md`](AITER_DEPENDENCY_MATRIX_2026-08-12.md)
- SGLang-owned Kimi FlyDSL migration:
  [`SGLANG_VENDOR_FLYDSL_2026-08-12.md`](SGLANG_VENDOR_FLYDSL_2026-08-12.md)
- Current pause point and new B300 two-case traces:
  [`PAUSE_TRACK_2026-08-11.md`](PAUSE_TRACK_2026-08-11.md)
- B300 80% single-stream campaign:
  [`B300_80PCT_CAMPAIGN_2026-08-11.md`](B300_80PCT_CAMPAIGN_2026-08-11.md)
- Solidified B2 fusion profile:
  [`B2_FUSION_SOLIDIFICATION_2026-08-11.md`](B2_FUSION_SOLIDIFICATION_2026-08-11.md)
- Generic M-tile fusion study:
  [`GENERIC_FUSION_RESULTS_2026-08-11.md`](GENERIC_FUSION_RESULTS_2026-08-11.md)
- Fresh-chat handoff:
  [`HANDOFF_2026-08-11.md`](HANDOFF_2026-08-11.md)
- Production trace and bottleneck ranking:
  [`PRODUCTION_TRACE_2026-08-11.md`](PRODUCTION_TRACE_2026-08-11.md)
- TP8 all-reduce microbench/E2E reconciliation:
  [`ALLREDUCE_SHAPE_INVESTIGATION_2026-08-11.md`](ALLREDUCE_SHAPE_INVESTIGATION_2026-08-11.md)
- Fresh-cache dispatch audit:
  [`STALE_JIT_AUDIT_2026-08-11.md`](STALE_JIT_AUDIT_2026-08-11.md)
- B300 versus MI355X matched trace comparison:
  [`B300_MI355X_TRACE_COMPARISON_2026-08-11.md`](B300_MI355X_TRACE_COMPARISON_2026-08-11.md)
- Route-sort-quant operator prototype:
  [`ROUTE_SORT_QUANT_PROTOTYPE_2026-08-11.md`](ROUTE_SORT_QUANT_PROTOTYPE_2026-08-11.md)
- Fresh-JIT C1 endpoint A/B:
  [`C1_FRESH_JIT_AB_2026-08-11.md`](C1_FRESH_JIT_AB_2026-08-11.md)
- HIP TopK + MXFP8 quant producer:
  [`HIP_ROUTE_QUANT_FUSION_2026-08-11.md`](HIP_ROUTE_QUANT_FUSION_2026-08-11.md)
- V3 multi-block route prototype:
  [`V3_MULTIBLOCK_ROUTE_2026-08-11.md`](V3_MULTIBLOCK_ROUTE_2026-08-11.md)
- V3-R wave64 role-grid prototype:
  [`V3R_WAVE64_ROLE_GRID_2026-08-11.md`](V3R_WAVE64_ROLE_GRID_2026-08-11.md)
- V4 persistent route-prep prototype:
  [`V4_PERSISTENT_ROUTE_PREP_2026-08-11.md`](V4_PERSISTENT_ROUTE_PREP_2026-08-11.md)
- Batch-1 fusion integration:
  [`BATCH1_FUSION_RESULTS_2026-08-10.md`](BATCH1_FUSION_RESULTS_2026-08-10.md)
- Completed optimization results:
  [`OPTIMIZATION_RESULTS_2026-08-10.md`](OPTIMIZATION_RESULTS_2026-08-10.md)
- Active execution handoff:
  [`EXECUTION_2026-08-10.md`](EXECUTION_2026-08-10.md)
- Current integration state:
  [`INTEGRATION_2026-08-10.md`](INTEGRATION_2026-08-10.md)
- Current validation results:
  [`VALIDATION_2026-08-10.md`](VALIDATION_2026-08-10.md)
- Current optimization priorities:
  [`PRIORITIES_2026-08-10.md`](PRIORITIES_2026-08-10.md)
- Current snapshot: [`snapshots/2026-08-10.md`](snapshots/2026-08-10.md)
- Audited upstream baseline: `ROCm/aiter` `origin/main@7c5e20170`
  (`2026-08-10`)
- The local integration branches now include the original SGLang/AITER stack,
  #4647, M=16384 tuning, and batch-1 PRs #4497/#4499/#4504/#4503.
  TP8 accuracy and C2–C64 endpoint validation passed.
- Search result at capture time: 43 Kimi-K3-related PRs
  (31 open, 9 merged, 3 closed without merge).
- Highest-priority upstream candidates:
  - [#4507](https://github.com/ROCm/aiter/pull/4507): long-context MLA
    split sizing, reported up to 4.80x E2E.
  - [#4450](https://github.com/ROCm/aiter/pull/4450): 12-head MLA split
    scheduling, reported up to 3.78x TPOT improvement.
  - [#4509](https://github.com/ROCm/aiter/pull/4509): split-major MLA grid
    and blocked reduce, stacked on #4507.
  - [#4487](https://github.com/ROCm/aiter/pull/4487): SiTUv2 MoE verify-step
    tuning, reported +18% DSpark output throughput.
  - [#4603](https://github.com/ROCm/aiter/pull/4603): A4W4 MoE retuning and
    graph-memory reduction.
- Exact next step: prototype opt-in HIP multi-stream overlap for BFA tiny
  GEMMs versus the wide projection, then shared all-reduce versus routed MoE.
  A #4504 latency-C1 profile is now validated (+9.59% output, -13.44% token
  capacity); general production remains unchanged. Route prototypes are
  rejected through V4. Multi-CU persistent synchronization is unsuitable for
  B1-B32; further route work requires a one-CTA LDS E896 sorter or a new
  stage1 metadata/scale ABI.

## Tracker layout

```text
aiter-optimization-tracker/
├── README.md
├── BATCH1_FUSION_RESULTS_2026-08-10.md
├── EXECUTION_2026-08-10.md
├── HANDOFF_2026-08-11.md
├── INTEGRATION_2026-08-10.md
├── OPTIMIZATION_RESULTS_2026-08-10.md
├── PRIORITIES_2026-08-10.md
├── VALIDATION_2026-08-10.md
└── snapshots/
    └── 2026-08-10.md
```

Add future audits as dated snapshots instead of overwriting history. Update this
README's `CONTINUE HERE` block to point to the newest snapshot.

## Update procedure

1. Record the exact AITER main commit and audit date.
2. Search PR title/body for `Kimi-K3`, `Kimi K3`, `kimi_k3`, and K3-specific
   shapes/operators.
3. Inspect mainline commits and tuned config diffs; text search alone misses
   shared infrastructure such as #4502.
4. List remote branches containing `kimi-k3` or `k3` and separate K2/K2.5 work.
5. For each performance claim, preserve hardware, quantization, workload,
   baseline, and whether it is kernel-only or end-to-end.
6. Record local validation separately from PR-author benchmark claims.

## Local validation template

```markdown
### YYYY-MM-DD — PR/branch

- AITER base/head:
- SGLang commit:
- GPU / ROCm / Torch / Triton:
- Model / TP / DCP / quantization:
- Command and environment:
- Correctness result:
- Kernel result:
- End-to-end result:
- Logs:
- Decision:
```

## Important caveats

- PR-author measurements use different workloads and are not directly
  comparable.
- The 2026-08-10 local AITER checkout was detached and stale; the audit used the
  fetched `origin/main`, not the worktree contents.
- K3 paths span gfx942, gfx950, and gfx1250. Do not apply a tuning result to a
  different architecture without revalidation.
- Several production paths are opt-in through environment flags or framework
  integration PRs; landing an AITER kernel alone may not activate it.
