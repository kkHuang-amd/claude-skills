# Kimi-K3 SGLang-owned FlyDSL migration — 2026-08-12

## Result

Kimi-specific gfx950 kernels were moved into:

```text
/sgl-workspace/sglang-k3-opts-0812/
python/sglang/kernels/ops/kimi_k3/flydsl/
```

Moved stacks:

```text
fused KDA + f_b
MLA output gate
KDA group64
FP8 preroute / tri / shared-down
FP8 latent tail
B2 extensions
```

The kernels remain dependent on AITER's shared FlyDSL helpers:

```text
buffer_ops
vector
tensor_shim
FlyDSL toolchain bootstrap
```

They no longer require the corresponding Kimi-specific AITER PR kernels.

## Source selection

```bash
SGLANG_K3_FLYDSL_SOURCE=auto     # prefer SGLang, fallback to AITER
SGLANG_K3_FLYDSL_SOURCE=sglang   # force vendored kernels
SGLANG_K3_FLYDSL_SOURCE=aiter    # force upstream AITER kernels
```

Existing adapters are the single compatibility boundary.

## SGLang commits

```text
13e6937 perf(kimi-k3): vendor gfx950 FlyDSL specializations
433af0d tune(kimi-k3): own the opt-in M16384 AITER profile
```

The SGLang-owned profile is enabled with:

```bash
SGLANG_K3_AITER_M16384_PROFILE=1
```

It configures AITER before AITER_CONFIGS is initialized.

## Minimal AITER branch

```text
/sgl-workspace/aiter-mainline-k3-0812
branch: integration/k3-core-only
```

Commits:

```text
638160b6 #4617 caller-provided fused_moe output
f9870683 #4647 stage1 scratch reuse
905928af opt-in scratch reuse flag
```

The Kimi-specific kernel commits are not present on this branch.

## Validation

```text
SGLang-vendored focused tests: 46 passed
SGLang-owned M16384 profile initialization: passed
GSM8K 50:  1.000
GSM8K 200: 0.990
```

Core-only AITER endpoint validation:

```text
C    golden tok/s   vendored tok/s   delta
2       969.04          968.57       -0.05%
4      1746.24         1741.98       -0.24%
8      2885.97         2881.25       -0.16%
16     4437.72         4432.25       -0.12%
32     6202.46         6191.41       -0.18%
```

All points pass the 0.5% acceptance band.

Capacity:

```text
max_total_num_tokens = 933883
max_running_requests = 256
```

Vendored B2 profile:

```text
C2: 968.57 -> 1054.19 tok/s
C2 TPOT: 17.67 -> 16.16 ms
C4: 1741.98 -> 1743.45 tok/s
```

The vendored kernels reproduce the prior accuracy, endpoint, capacity and B2
results without the Kimi-specific AITER PR commits.

No commits were pushed.
