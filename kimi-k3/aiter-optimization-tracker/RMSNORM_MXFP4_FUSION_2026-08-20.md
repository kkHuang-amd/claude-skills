# Kimi-K3 latent RMSNorm + MXFP4 quant fusion — 2026-08-20

## Decision

Retain the standalone AITER RMSNorm+MXFP4 quant integration as a default-off
experimental path, but do not promote it to the production up-only profile.

```text
SGLANG_K3_MOE_LATENT_NORM_QUANT_MXFP4=0  # default
```

The kernel substantially improves the isolated norm+quant chain and passes
full GSM8K, but the complete C2-C64 endpoint result is effectively neutral and
does not meet the 0.5% TTFT promotion gate.

## Flow

Baseline up-only tail:

```text
all-reduce [latent | shared]
latent RMSNorm -> BF16 latent
per-1x32 MXFP4 quant -> values + E8M0 scales
gfx950 ASM latent-up GEMM
add3(shared, prefix)
```

Candidate:

```text
all-reduce [latent | shared]
AITER rmsnorm_quant(group_size=32, shuffle_scale=True)
  -> MXFP4 values + shuffled E8M0 scales
same gfx950 ASM latent-up GEMM
add3(shared, prefix)
```

The collective and shared-buffer semantics remain unchanged. This is not the
generic SGLang `--enable-aiter-allreduce-fusion` path; Kimi-K3 does not use
LayerCommunicator and norms only the latent slice of its heterogeneous buffer.

## Implementation

```text
python/sglang/kernels/ops/kimi_k3/latent_mxfp4_aiter_hip.py
python/sglang/srt/models/kimi_k3.py
python/sglang/srt/environ.py
docs/docs/references/environment_variables.mdx
test/registered/kernels/ops/kimi_k3/test_latent_mxfp4_aiter_hip.py
```

The adapter is gfx950-only, M>=2048, shape/dtype/contiguity checked, and falls
back to the original RMSNorm + quant + ASM chain when unsupported.

## Microbench

```text
M       norm+quant baseline -> fused     speedup   full chain speedup
2048      38.50 -> 21.13 us              1.82x          1.37x
4096      35.98 -> 20.75 us              1.73x          1.02x
8192      35.14 -> 20.55 us              1.71x          1.12x
16384     63.47 -> 27.87 us              2.28x          1.11x
```

The complete ASM latent-up output has relative L2 `~0.033` and cosine
`~0.99945` versus the decomposed baseline, reflecting the fused norm/quant
rounding path.

## Endpoint

Final fixed 8192/1024 C2-C64 sweep, 64 warmups, eight measured requests per
concurrency:

```text
C     tok/s     throughput delta   TTFT ms   TTFT delta   TPOT ms   TPOT delta
2     1147.42         +1.92%         840.48      -0.35%      14.87       -0.07%
4     1986.97         +0.15%        1471.42      +0.54%      16.75       -0.24%
8     3150.25         -0.02%        2283.24      -0.21%      20.65       +0.00%
16    4966.17         +0.10%        3947.03      -0.27%      25.27       -0.20%
32    6979.81         +0.08%        7453.87      -0.28%      34.27       -0.03%
64    9046.80         +0.06%       13955.37      -0.17%      50.15       +0.02%
```

All `1008/1008` requests succeeded. Capacity remains `1,376,952` tokens.

## Accuracy and tests

```text
GSM8K 50:    1.000, invalid 0.000
GSM8K 1319:  0.955, invalid 0.001

Kimi-K3 focused tests: 70 passed
additional subtests:   6 passed
pre-commit:             passed
IDE diagnostics:       no errors
CUDA Graph replay:     passed
```

## Artifacts

```text
/workspace/kimi-k3-runs/rmsnorm-mxfp4-2026-08-20/
  micro/run_microbench.py
  micro/results.json
  micro/run.log
  endpoint/screen/
  endpoint/final/
  sglang.patch
```

Patch SHA-256:

```text
d516ba5de2b115bed8d65b27739409954a9859c9a2b3c8f9b82eb150bccde1fd
```

## Next

The standalone fusion removes the intended launch and memory round-trip but is
too small a fraction of end-to-end latency to produce a material gain. The
next step with plausible E2E impact is a K3-shaped ROCm custom all-reduce that
handles `[M latent | 2M shared]`, norms/quantizes only the first M rows, and
emits MXFP4 values/scales directly while keeping shared rows BF16.
