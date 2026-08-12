# Kimi-K3 B2 fusion solidification — 2026-08-11

## Scope

The earlier B2/B4 experiment was restored and converted into a fail-closed B2
profile. It extends three existing B1 gfx950 fusion paths to two-token GPU
batches:

```text
KDA group64: q/k/v/g + beta + f_a
MoE preroute: routed-down + shared gate/up + router
Shared expert: SiTU + shared-down projection
```

The latent-tail extension is not included.

## Dispatch

The new bucket is independently gated:

```bash
SGLANG_K3_AITER_B2_FUSIONS=1
SGLANG_K3_AITER_KDA_GROUP64=1
SGLANG_K3_AITER_MOE_PREROUTE_FP8=1
```

`SGLANG_K3_AITER_B2_FUSIONS` defaults off. With it enabled, only actual GPU
batch `M=2` enters the new kernels. B4 and larger fail closed to the existing
production path.

## Correctness

Focused AITER tests cover B1/B2 group64, mixed-precision tri-projection,
shared-down SiTU, and graph replay:

```text
19 passed
```

Python syntax, repository diff checks, and IDE diagnostics passed.

## Paired endpoint result

Configuration:

```text
TP8, 8192 input / 1024 output
no radix cache
64 warmup requests
same source/JIT cache and seed
C2: 16 measured requests
C4: 32 measured requests
```

### C2

```text
metric                    baseline     B2 fusion      delta
total throughput          971.42       1055.92 tok/s  +8.70%
output throughput         107.94        117.32 tok/s  +8.69%
median TPOT                17.61         16.13 ms      -1.48 ms
median TTFT               942.85        938.93 ms      -0.42%
```

This contemporaneous result confirms the prior historical `+8.67%` result.

### Why B4 is excluded

When the fused kernels were allowed to process `M=4`, C4 regressed:

```text
total throughput 1747.18 -> 1674.98 tok/s  (-4.13%)
median TPOT         19.07 ->   19.96 ms
```

After restricting dispatch to B2, C4 returned to baseline:

```text
metric                    baseline     B2-only profile  delta
total throughput          1747.18      1747.33 tok/s    +0.01%
output throughput          194.13       194.15 tok/s    +0.01%
median TPOT                 19.07        19.09 ms        +0.02 ms
median TTFT               1595.89      1589.81 ms        -0.38%
```

## Decision

Retain the implementation as an opt-in, B2-only profile. Do not dispatch B4.
Before making it a default, run:

```text
five-round paired C2
GSM8K 50 -> 200
graph/token-capacity comparison
```

No commit was created.

Artifacts:

```text
stage2-runs/2026-08-11-b300-80pct/solidified-b24/
```

## Generic schedule follow-up

An in-block TILE_M design was tested for M4/M8/M16. M4 shared-down and
tri-projection were individually faster, but their tuned cumulative saving was
only 2.77 us/layer; M8/M16 regressed. The generic experiment was removed and
does not alter this B2 profile.

Details:
[`GENERIC_FUSION_RESULTS_2026-08-11.md`](GENERIC_FUSION_RESULTS_2026-08-11.md).
