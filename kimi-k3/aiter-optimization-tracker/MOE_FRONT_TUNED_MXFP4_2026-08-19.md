# Kimi-K3 non-EP tuned MoE front and latent MXFP4 — 2026-08-19

## Decision

Retain both optimizations as independent opt-in features for the TP8 non-EP
profile:

```text
SGLANG_K3_AITER_TUNED_MOE_FRONT=1
SGLANG_K3_AITER_TUNED_MOE_FRONT_MIN_TOKENS=48
SGLANG_K3_AITER_TUNED_MOE_FRONT_MAX_TOKENS=192

SGLANG_K3_MOE_LATENT_MXFP4=1
SGLANG_K3_MOE_LATENT_MXFP4_MIN_TOKENS=2048
```

The AITER tuned BF16 front is restricted to M48-M192 because local microbench
showed regressions at M8/M16 and no material gain at M32. The latent MXFP4 path
passes performance and full GSM8K gates. EP/N4480 behavior is unchanged and is
not claimed by this integration.

## Tested state

```text
SGLang base: 0cd88a5e408d5548b2490b2d40ebf3c306373476
AITER base:  284a1eb401bb15f6368a68b34eb0cd693ee1fcd3
GPU:         8x AMD Instinct MI355X
Torch:       2.9.1+rocm7.2.0.git7e1940d4
HIP:         7.2.26015-fc0010cf6a
Triton:      3.7.0
```

AITER [#4834](https://github.com/ROCm/aiter/pull/4834) contributes 27 gfx950
BF16 tuning rows. SGLang [#35406](https://github.com/sgl-project/sglang/pull/35406)
was manually adapted to preserve the existing K3 preroute, latent-tail,
all-reduce+norm and GEMM+all-gather precedence.

## Implementation

AITER owns the global tuned-GEMM data and dispatch:

```text
aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv
  N4480 rows retained from the PR but not wired in phase 1
  N6016 rows used by the non-EP merged front
```

SGLang owns only the integration policy and gfx950 adapter:

```text
python/sglang/kernels/ops/kimi_k3/latent_mxfp4_aiter_hip.py
python/sglang/srt/models/kimi_k3.py
python/sglang/srt/environ.py
docs/docs/references/environment_variables.mdx
test/registered/kernels/ops/kimi_k3/test_latent_mxfp4_aiter_hip.py
```

For M1-M4 the existing SGLang-vendored FlyDSL preroute remains first. For
M48-M192, the N6016 BF16 merged front can use AITER tuned_gemm. At M>=2048,
the non-EP front splits into an N2432 BF16 head plus an MXFP4 latent-down GEMM;
the latent-up projection also uses MXFP4 after the routed experts and
all-reduce+norm. The existing BF16 weights stay live as fallback.

## Focused validation

```text
AITER config rows: 27
N4480 / N6016:     14 / 13
duplicate shapes: 0

K3 focused tests: 67 passed
additional subtests: 6 passed
pre-commit: passed
IDE diagnostics: no errors
CUDA Graph input-change replay: passed
```

## Kernel microbench

N6016 BF16 front:

```text
M     torch us   tuned us   speedup   selected
8       22.222     24.989    0.889x   FlyDSL
16      22.288     24.247    0.919x   FlyDSL
32      24.404     24.825    0.983x   FlyDSL
48      28.034     25.292    1.108x   FlyDSL
64      35.577     25.010    1.423x   FlyDSL
80      33.491     28.753    1.165x   FlyDSL
96      40.333     32.696    1.234x   Opus
112     36.414     29.401    1.239x   FlyDSL
128     51.700     29.291    1.765x   FlyDSL
192     61.749     38.421    1.607x   FlyDSL
```

Latent MXFP4:

```text
projection   M      BF16 us   MXFP4 us   speedup   relL2    cosine
down         2048    110.416      56.009    1.971x   0.1658   0.98621
down         4096    171.844      79.472    2.162x   0.1657   0.98621
up           2048     86.956      51.411    1.691x   0.1658   0.98621
up           4096    178.014      69.701    2.554x   0.1657   0.98622
```

Packed latent copies add `26.03 MiB` per K3 MoE layer per rank.

## Endpoint matrix

Fixed 8192/1024, TP8, 64 warmups, eight measured requests per concurrency,
seed 42, radix cache disabled:

```text
case        C2       C4       C8       C16      C32      C64 tok/s
base      1141.80  1979.59  3134.19  4920.97  6885.52  8845.80
tuned     1143.99  1975.84  3131.12  4918.44  6894.23  8902.85
MXFP4     1148.25  1972.66  3161.01  4967.23  7008.56  9051.16
both      1145.21  1991.54  3158.53  4989.28  7038.03  9112.40
```

After adding the M192 tuned-front upper bound, the final combined sweep was:

```text
C     tok/s     throughput vs base   TTFT ms   TTFT delta   TPOT ms   TPOT delta
2     1146.71          +0.43%          822.92      -5.56%      14.90       +0.00%
4     1985.17          +0.28%         1449.21      -3.40%      16.75       -0.24%
8     3163.68          +0.94%         2255.18      -4.83%      20.60       -0.48%
16    4991.10          +1.43%         3893.54      -4.65%      25.15       -0.83%
32    7032.36          +2.13%         7321.60      -5.60%      34.11       -1.13%
64    9096.55          +2.83%        13645.77      -5.44%      50.16       -1.94%
```

All measured requests succeeded. Token capacity changes from `1,519,705` to
`1,252,555` (`-17.58%`) because the BF16 latent weights remain live beside the
packed MXFP4 copies.

## Accuracy

```text
GSM8K 50:    1.000, invalid 0.000
GSM8K 1319:  0.946, invalid 0.001
policy gate: >0.94, passed
```

## NEXT OPTIMIZATIONS

Resume here for the next MoE-front session:

1. Add independent latent-down and latent-up MXFP4 gates and run down-only /
   up-only endpoint A/B. Goal: identify whether one packed copy retains most
   of the gain while recovering about half of the 17.58% capacity loss.
2. Extend the AITER MXFP4 latent-up GEMM with a fused
   `shared_output + prefix_sum` epilogue. This is the lowest-risk
   shared/routed tail fusion and removes the final output read/write plus add
   launch.
3. Evaluate an all-reduce + RMSNorm epilogue that emits MXFP4 values and E8M0
   scales directly for latent-up, removing the BF16 latent materialization and
   standalone activation quantization.
4. Retune the N6016 BF16 front on the current Triton 3.7 stack for M8/M16/M32
   and exact CUDA Graph buckets such as M40/M56/M72. Current PR rows are kept
   only for M48-M192 because M8/M16 regress locally.
5. Run single-request 1M context, BEAM/needle 1M correctness, and target
   long-context concurrency tests before making latent MXFP4 a production
   default.
6. If capacity remains unacceptable, evaluate an offline latent-projection
   MXFP4 checkpoint that can release the BF16 down/up weights; revalidate
   small-M latency and full/long-context accuracy.

First action tomorrow: implement separate down/up flags, measure their
per-layer packed memory independently, then run the same 2x2 C2-C64 matrix.

## Artifacts

```text
/workspace/kimi-k3-runs/moe-front-opt-2026-08-19/
  sglang.patch
  aiter.patch
  micro/run_microbench.py
  micro/results-full.json
  micro/run-full.log
  endpoint/run_matrix.sh
  endpoint/{00-base,10-tuned,01-mxfp4,11-both}/
  endpoint/11-both-final/
```

Patch checksums:

```text
SGLang f063c2e00843cfabe7b76938f95b7b98cc6b3535c73dbf842d88be4b387b1bd4
AITER  976fe16bef78de9291914d8ce6a4153c28fd3820b6f0fd282dbc976c26d984f0
```

The server and benchmark processes were stopped after validation. GPU VRAM
returned to the pre-run level on all eight devices.
