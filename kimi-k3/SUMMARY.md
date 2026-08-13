# Kimi-K3 optimization summary

Updated: 2026-08-13

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

## 2026-08-13 Triton runtime attribution

The initial reclone used Triton 3.7 and regressed progressively at high
concurrency. Replacing only Triton with the handover commit
`3.6.0+git42270451` recovered C8-C32:

```text
C2:   966.08 tok/s  (-0.26%)  pass
C4:  1743.73 tok/s  (+0.10%)  pass on focused repeat
C8:  2887.31 tok/s  (+0.21%)  pass
C16: 4437.74 tok/s  (+0.12%)  pass
C32: 6176.87 tok/s  (-0.23%)  pass
```

Triton 3.6 improved C8/C16/C32 over the tested 3.7 build by
1.69%/2.98%/5.01%. Treat Triton 3.7 as the main high-concurrency regression
source and retain Triton 3.6 for the handover-equivalent runtime. Capacity
remained 933883 and all requests succeeded.

Details:
`aiter-optimization-tracker/TRITON36_AB_2026-08-13.md`.

Paired C32 traces confirmed that PyTorch/ROCtracer does not expand production
HIP CUDA Graph replays: only one replay was visible during a 15-second active
decode window. The endpoint A/B causally attributes the regression to the
Triton 3.7 runtime/codegen stack, but available traces do not localize it to a
specific Triton kernel. Details:
`aiter-optimization-tracker/TRITON36_37_C32_TRACE_2026-08-13.md`.

Stage-separated all-rank traces refined the attribution:

```text
prefill stage span:       +2.13%
prefill GPU kernel time:  -3.12%
record_param_comms p50:   +5.59%
profiled decode TPOT:     +8.54%
```

Coverage validation later showed that the stage-separated prefill traces expose
only 92 of 552 MoE GPU executions per TP0 (16.7%). A combined prefill+decode
capture records identical CPU MoE counts for 3.6/3.7, but Triton 3.7 hides over
99% of CUDA Graph replay kernels from ROCTracer, including MoE, fused KDA and
MLA merge. Therefore total GPU time and the earlier synchronization attribution
are not valid cross-version comparisons.

The later compact Rank0 trace restores comparable graph visibility and
localizes the prefill loss to
`python/sglang/kernels/ops/attention/extend_attention.py::_fwd_kernel`:

```text
p50:             5990.80 -> 13780.42 us (+130.03%)
total:            143.73 ->   330.90 ms (+187.17 ms)
prefill span:    4153.94 ->  4375.87 ms (+221.93 ms)
explained span:  84.34%
```

Triton 3.7 changes this kernel from 483 VGPR / no private segment to 512 VGPR /
472-byte private segment with 186 scratch spill instructions. Decode has
several smaller 2.7-4.5% regressions but no comparable single-kernel culprit.
Details:
`aiter-optimization-tracker/TRITON36_37_STAGE_ANALYSIS_2026-08-13.md`.

Future SGLang traces must use the validated compact method in
`aiter-optimization-tracker/SGLANG_TRACE_CAPTURE_METHOD_2026-08-13.md`:
one unmerged file per rank, late prefill plus short decode in one manual
profiler session, no stacks/shapes, and a 500 MiB compressed limit per rank.

## Optional Triton 3.7 extend-attention fix

Enable:

```bash
SGLANG_TRITON_37_EXTEND_LQ576_N32=1
```

The gfx950 Lq576/Lv512 extend-attention tile changes from N64 to N32 only on
Triton >=3.7. This removes 472-byte scratch spilling and restores the isolated
kernel from 12.57 to 5.24 ms.

Endpoint result:

```text
C2:   973.94 tok/s
C4:  1748.71 tok/s
C8:  2901.37 tok/s
C16: 4460.33 tok/s
C32: 6198.56 tok/s
```

C32 improves 5.38% over the Triton 3.7 baseline and is 0.12% above the
handover target. Capacity remains 933883. Keep default-off until GSM8K
validation is recorded.

Details:
`aiter-optimization-tracker/TRITON37_EXTEND_N32_RESULTS_2026-08-13.md`.

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

## Validated Radix-4 router profile

Enable:

```bash
SGLANG_K3_RADIX4_TOPK=1
```

Validation:

```text
Focused tests: 45 passed
C2 paired median: 970.38 -> 993.11 tok/s (+2.34%)
C2 TPOT:          17.63 -> 17.21 ms (-2.38%)
C4:             1741.98 -> 1777.66 tok/s (+2.05%)
C8:             2881.25 -> 2930.39 tok/s (+1.71%)
C16:            4432.25 -> 4491.62 tok/s (+1.34%)
C32:            6191.41 -> 6233.41 tok/s (+0.68%)
GSM8K 200:       0.985
capacity:        933883
```

The isolated integration adds deterministic AITER-compatible exact-tie
selection, NaN exclusion and a gfx942/gfx950 guard beyond upstream #34490.
Keep it default-off until those correctness fixes are reconciled upstream.

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
Radix-4 K3 TopK router
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

Completed integration:

```text
SGLang #34490 Radix-4 Kimi-K3 TopK router
measured MI355X saving: 4.2-5.2 us/layer for M1-M64
status: all gates passed; retained default-off
```

This exact Radix-4 nibble-histogram/DPP design was not tested in the prior
V3/V3-R/V4 work. The earlier reports anticipated a broader radix direction,
but V3 used 16 repeated argmax rounds and failed for different architectural
reasons.

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

Next, finish analysis of the retained B300 normal versus
single-stream/no-PDL summaries.

## Detailed sources

```text
aiter-optimization-tracker/SGLANG_VENDOR_FLYDSL_2026-08-12.md
aiter-optimization-tracker/FRESH_INTEGRATION_2026-08-12.md
aiter-optimization-tracker/AITER_DEPENDENCY_MATRIX_2026-08-12.md
aiter-optimization-tracker/B2_FUSION_SOLIDIFICATION_2026-08-11.md
aiter-optimization-tracker/B300_MI355X_TRACE_COMPARISON_2026-08-11.md
aiter-optimization-tracker/PAUSE_TRACK_2026-08-11.md
aiter-optimization-tracker/PR34490_RADIX4_ASSESSMENT_2026-08-12.md
aiter-optimization-tracker/PR34490_RADIX4_RESULTS_2026-08-12.md
```
