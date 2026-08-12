# Kimi-K3 optimization summary

Updated: 2026-08-12

## Current integration

```text
SGLang remote:
  https://github.com/HaiShaw/sglang
  branch perf/k3_opts_0812
  HEAD f9dd3a0661b472d5fba1632adebcffc5c7c4021e

AITER remote:
  https://github.com/kkHuang-amd/aiter
  branch integration/k3-core-only
  HEAD 284a1eb401bb15f6368a68b34eb0cd693ee1fcd3
```

Kimi-specific gfx950 FlyDSL kernels are maintained in SGLang. AITER retains
only the core dependencies:

```text
#4617 caller-provided fused_moe output
#4647 stage1 scratch reuse
shared FlyDSL helpers/toolchain
```

## Validated production result

```text
Focused vendored tests: 46 passed
GSM8K 50:              1.000
GSM8K 200:             0.990
Max token capacity:    933883

C2:   968.57 tok/s
C4:  1741.98 tok/s
C8:  2881.25 tok/s
C16: 4432.25 tok/s
C32: 6191.41 tok/s
```

All endpoint points are within 0.25% of the selected golden stack.

## Optional B2 profile

```text
C2:      968.57 -> 1054.19 tok/s
C2 TPOT:  17.67 ->   16.16 ms
C4:     1741.98 -> 1743.45 tok/s
```

Enable:

```bash
SGLANG_K3_AITER_MOE_PREROUTE_FP8=1
SGLANG_K3_AITER_B2_FUSIONS=1
```

M>=4 fails closed to the production path.

## Source selection

```bash
SGLANG_K3_FLYDSL_SOURCE=auto
SGLANG_K3_FLYDSL_SOURCE=sglang
SGLANG_K3_FLYDSL_SOURCE=aiter
```

The SGLang-owned M16384 profile is enabled with:

```bash
SGLANG_K3_AITER_M16384_PROFILE=1
```

## Decisions already made

Keep:

```text
fused KDA + f_b
MoE zero-copy output
stage1 scratch reuse
MLA gate
KDA group64
SGLang-vendored Kimi FlyDSL kernels
B2-only optional profile
```

Optional only:

```text
FP8 preroute/shared-down
FP8 latent tail
A4W4 C16 profile
```

Do not retry without an architecture change:

```text
V3/V3-R role-grid + P23
V4 multi-CU persistent route prep
generic TILE_M M4/M8/M16
forced all-reduce global/exact-B32
standalone #4572/#4577 endpoint paths
```

## Remaining optimization work

Highest-value unresolved areas:

```text
fixed tiny-kernel chains
copies/materialization
attention residual and KDA launch boundaries
route / sort / quant handoff
```

Credible route directions:

```text
one-CTA LDS E896 sorter
stage1 ABI consuming route metadata and token-major scale directly
```

Before new kernel work, finish analysis of the retained B300 normal versus
single-stream/no-PDL summaries.

## Detailed sources

```text
aiter-optimization-tracker/SGLANG_VENDOR_FLYDSL_2026-08-12.md
aiter-optimization-tracker/FRESH_INTEGRATION_2026-08-12.md
aiter-optimization-tracker/AITER_DEPENDENCY_MATRIX_2026-08-12.md
aiter-optimization-tracker/B2_FUSION_SOLIDIFICATION_2026-08-11.md
aiter-optimization-tracker/B300_MI355X_TRACE_COMPARISON_2026-08-11.md
aiter-optimization-tracker/PAUSE_TRACK_2026-08-11.md
```
