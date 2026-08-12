# Kimi-K3 fresh-clone integration — 2026-08-12

## Workspaces

```text
golden:
  /sgl-workspace/sglang
  /sgl-workspace/aiter

fresh:
  /sgl-workspace/sglang-k3-opts-0812
  /sgl-workspace/aiter-mainline-k3-0812
```

Fresh bases:

```text
SGLang 8950e2e0dfb05483ea120612d1156ed509b2dddc
  HaiShaw/perf/k3_opts_0812

AITER  49960c1e77dc4718651498d2c3767844c973b396
  ROCm/aiter main
```

The golden repositories were not modified by the migration.

## AITER dependency audit

At the integration point none of the required capabilities was on AITER main.

```text
#4495 open draft       fused KDA + f_b
#4617 open             caller-provided fused_moe output
#4497 open draft       MLA output gate
#4499 open draft       KDA group64
#4504 open draft       FP8 preroute/shared-down
#4503 open draft       FP8 latent tail
#4647 open             stage1 scratch reuse
#4603 closed unmerged  A4W4 optional profile
```

Ownership decision:

```text
SGLang:
  adapter, capability detection, feature flags, packing,
  warmup, model dispatch and fallback

AITER:
  FlyDSL kernels/wrappers, fused_moe core output API,
  stage1 scratch lifecycle and tuned config files
```

## Fresh AITER stack

```text
89fce019 perf(flydsl): fuse Kimi-K3 KDA decode and f_b          #4495
44eaa76d feat(fused_moe): accept caller-provided output buffers #4617
d5411148 perf(kimi-k3): reuse FlyDSL MoE stage1 scratch         #4647
2845f200 feat(fused_moe): gate FlyDSL stage1 scratch reuse
d599525e tune(kimi-k3): add opt-in M16384 BF16 profile
11460ce4 perf(kimi-k3): add fused MLA output gate               #4497
b89954d8 perf(kimi-k3): add group64 KDA input projection        #4499
98ce357d perf(kimi-k3): add FP8 MoE pre-route fusions           #4504
076b2720 perf(kimi-k3): add FP8 latent MoE tail                 #4503
2eea7204 perf(kimi-k3): extend fusion kernels to batch two
```

Optional A4W4 branch:

```text
profile/k3-a4w4-c16
61bf56be tune(kimi-k3): add optional gfx950 A4W4 profile        #4603
```

## Fresh SGLang stack

The remote base already contains the fused-KDA wiring:

```text
8950e2e0d perf(kimi-k3): fuse ROCm KDA decode boundary
```

New commits:

```text
61c39c7 perf(kimi-k3): integrate AITER MoE optimizations
69bd4eb fix(kimi-k3): support 12-head AITER MLA decode
1544286 perf(kimi-k3): wire AITER MLA output gate
9aeff1f perf(kimi-k3): wire AITER KDA group64 projection
e13b8a7 perf(kimi-k3): wire AITER FP8 MoE pre-route
fa81245 perf(kimi-k3): wire AITER FP8 latent tail
81ce307 perf(kimi-k3): add opt-in batch-two fusions
```

No commits were pushed.

## Focused validation

```text
#4495 KDA decode:         12 passed
#4617 output contract:    passed, including torch.compile/alias cases
#4647 scratch reuse:       1 passed
#4497 MLA gate:            8 passed
#4499 group64:            11 passed
#4504 preroute:            6 passed
#4503 latent tail:         7 passed
B2 extension:             19 passed
SGLang MoE/layout:        11 passed
```

GSM8K:

```text
50 questions:  1.000
200 questions: 0.990
golden:        0.980
```

## Production endpoint validation

TP8, 8192/1024, no radix cache, 64 warmup requests:

```text
C    golden tok/s   fresh tok/s   delta
2       969.04        970.75      +0.18%
4      1746.24       1743.67      -0.15%
8      2885.97       2886.63      +0.02%
16     4437.72       4435.25      -0.06%
32     6202.46       6199.67      -0.05%
```

All points are inside the 0.5% acceptance band.

Capacity:

```text
max_total_num_tokens = 933883
max_running_requests = 256
```

This matches the golden capacity. Scratch reuse and M16384 rows were enabled
through:

```bash
AITER_FLYDSL_STAGE1_SCRATCH_REUSE=1
AITER_K3_M16384_PROFILE=1
```

## Optional B2 validation

```text
metric              production   B2 profile
C2 total tok/s        970.75       1054.53
C2 TPOT                17.62         16.16 ms
C4 total tok/s       1743.67       1743.65
```

B2 reproduces the prior gain and C4 remains flat because M>=4 fails closed.

## Decision

The fresh-clone stack reproduces golden correctness, capacity and performance.
Keep every optional optimization default-off in code; deployment manifests may
enable the selected production flags after policy review.

Artifacts:

```text
stage2-runs/2026-08-12-fresh-integration/
```
