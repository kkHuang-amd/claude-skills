# Kimi-K3 generic fusion schedule study — 2026-08-11

## Goal

Generalize the retained B2 fusion family to runtime M=1-64 while preserving
Kimi-K3/gfx950 dimensions and numerical contracts. Optimized buckets would use
one grid with in-block `TILE_M`; rejected buckets would use production GEMM
fallback.

The proven B2 path remained untouched during the experiment.

## Shared-down TILE_M

The prototype staged `TILE_M x 768` SiTU activations in LDS and loaded each FP8
weight row once for all M rows.

```text
M    production us   TILE_M us   saved us   speedup
4        8.720         7.355       +1.365     1.186x
8        9.539        13.371       -3.832     0.713x
16      10.556        22.093      -11.538     0.478x
```

Correctness:

```text
relative RMSE <= 0.000028
minimum cosine >= 0.9999999
```

M4 passed the per-kernel 5% gate. M8/M16 were rejected.

## Tri-projection TILE_M

The prototype reused routed/shared FP8 and router BF16 weight rows across M,
while preserving the router DPP order and BF16-round-to-FP32 boundary.

```text
M    production us   TILE_M us   saved us   speedup
4       19.690        17.744       +1.946     1.110x
8       19.966        32.758      -12.793     0.609x
```

Correctness:

```text
maximum relative RMSE <= 0.000097
minimum cosine >= 0.9999998
```

M4 passed the isolated 5% gate. M8 was rejected.

## M4 schedule sweep

The accepted M4 candidates were swept over:

```text
rows_per_wave = 1, 2, 4
CU count      = 128, 192, 248, 256
```

Best schedules:

```text
shared-down: rpw=1, CU=248, 7.567 us
tri:        rpw=1, CU=256, 18.072 us
```

The tuned cumulative saving versus production was only about:

```text
shared-down  8.720 - 7.567 = 1.153 us
tri         19.690 - 18.072 = 1.618 us
total                            2.771 us/layer
```

This is far below the required 10 us/layer tri+shared gate.

## Decision

Stop before KDA group64, SGLang bucket wiring, endpoint validation, and the
non-K3 Phase-B shape.

The basic weight-reuse idea works at M4, but the gain is too small because:

- production small-M GEMM already amortizes M efficiently;
- multiple per-token accumulators increase register pressure and instruction
  count;
- M8/M16 LDS/register scaling overwhelms saved weight traffic.

All TILE_M experimental source was removed. The proven, opt-in B2-only profile
remains unchanged and its 19 focused tests still pass.

Future generic work should build on AITER's MFMA `small_m_hgemm` family with a
row-scale/SiTU epilogue rather than extending wave-reduced GEMV kernels across
M.

Artifacts:

```text
stage2-runs/2026-08-11-generic-fusion/
```
