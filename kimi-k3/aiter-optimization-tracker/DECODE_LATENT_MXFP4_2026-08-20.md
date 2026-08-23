# Kimi-K3 decode latent-up MXFP4 — 2026-08-20

## Decision

Retain a default-off M32-M256 decode band:

```text
SGLANG_K3_MOE_LATENT_UP_MXFP4=1
SGLANG_K3_MOE_LATENT_NORM_QUANT_MXFP4=1
SGLANG_K3_MOE_LATENT_UP_MXFP4_DECODE=1
SGLANG_K3_MOE_LATENT_UP_MXFP4_DECODE_MIN_TOKENS=32
SGLANG_K3_MOE_LATENT_UP_MXFP4_DECODE_MAX_TOKENS=256
SGLANG_K3_MOE_LATENT_UP_MXFP4_ALL_TOKENS=0
```

BF16 latent-up weights must remain. Low decode M and the M257-M2047 fallback
still use BF16, and the all-token profile regresses low-concurrency endpoint
performance.

## Benchmark methodology correction

Eager microbench initially suggested MXFP4 was faster at M1-M16 and slower at
M32-M256. That result included Python/tuned-gemm dispatch overhead absent from
production CUDA Graph replay.

CUDA Graph chain timing shows MXFP4 is faster at every tested M:

```text
M      BF16 graph us   MXFP4 graph us   speedup
1          18.65           18.31         1.02x
2          19.32           18.50         1.04x
4          19.34           18.49         1.05x
8          19.54           18.48         1.06x
16         19.79           18.60         1.06x
32         25.02           18.79         1.33x
64         23.06           18.93         1.22x
128        25.37           19.64         1.29x
256        32.12           21.19         1.52x
512        40.11           23.56         1.70x
1024       64.84           35.15         1.84x
2048      102.35           40.78         2.51x
```

The candidate uses AITER `rmsnorm_quant(group_size=32,
shuffle_scale=True)` followed by the existing gfx950 ASM latent-up GEMM.

## Scale padding fix

Initial server warmup exposed a gfx950 ASM contract absent from the first
microbench: shuffled activation scales must have a 256-row padded slab. Small-M
allocation of only M rows caused a GPU memory fault. The final adapter allocates
`ceil(M/256)*256` rows of scale storage. Unit and CUDA Graph tests cover M8.

## Endpoint experiments

The all-token profile was rejected because low-concurrency TPOT regressed,
despite improving C32/C64:

```text
all-token vs up-only:
  C2  TPOT +1.34%
  C8  TPOT +1.89%
  C32 throughput +0.90%, TPOT -1.17%
  C64 throughput +0.97%, TPOT -1.14%
```

The selected M32-M256 band, fixed 8192/1024, TP8, 64 warmups, eight measured
requests per concurrency:

```text
C     tok/s     throughput delta   TTFT ms   TTFT delta   TPOT ms   TPOT delta
2     1144.60         +1.67%         838.53      -0.59%      14.91       +0.20%
4     1985.01         +0.05%        1450.79      -0.87%      16.79       +0.00%
8     3151.99         +0.03%        2281.95      -0.26%      20.65       +0.00%
16    4967.35*        +0.13%        3942.61      -0.38%      25.24       -0.32%
32    7049.34         +1.08%        7436.91      -0.50%      33.87       -1.20%
64    9115.55         +0.82%       13893.60      -0.61%      49.69       -0.90%
```

`*` C16 is the focused rerun after the first point was an outlier.

All requests succeeded. Capacity remains `1,376,952` tokens.

## Accuracy

```text
GSM8K 50:          1.000, invalid 0.000
GSM8K 1319:        0.958, invalid 0.002
GSM8K 1319 rerun:  0.954, invalid 0.001
policy gate:       >0.94, passed
```

## Validation

```text
Kimi-K3 focused tests: 71 passed
additional subtests:   6 passed
pre-commit:             passed
IDE diagnostics:       no errors
M8 CUDA Graph replay:  passed
```

## Artifacts

```text
/workspace/kimi-k3-runs/decode-latent-mxfp4-2026-08-20/
  micro/run_microbench.py
  micro/results-graph-full.json
  micro/run-graph-full.log
  endpoint/final/
  endpoint/all-tokens/
  endpoint/band32-256/
  sglang.patch
```

Patch SHA-256:

```text
560b7438b8a95a3c2a2e77db86d4af57c001da9ad78eb5d6c23c2a803444bb81
```

## Weight-retention conclusion

Do not release the BF16 latent-up weight:

1. selected decode coverage starts at M32, so M1-M31 remains BF16;
2. M257-M2047 remains a conservative BF16 fallback;
3. MXFP4 numerical error is rel-L2 approximately `0.165`, cosine `0.986`;
4. keeping BF16 preserves fail-closed behavior for unsupported devices and
   layouts.

Removing BF16 requires a separately validated all-token checkpoint/runtime
policy, which the endpoint tests reject.
