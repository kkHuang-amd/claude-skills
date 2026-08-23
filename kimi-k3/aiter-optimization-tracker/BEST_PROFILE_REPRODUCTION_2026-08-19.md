# Kimi-K3 best-profile reproduction — 2026-08-19

## Configuration

The rebuilt stack used:

```text
SGLang: dc6e5a2cd86f6ba049e5860bf8c809ec0dc7525d
AITER:  284a1eb401bb15f6368a68b34eb0cd693ee1fcd3
GPU:    8x AMD Instinct MI355X
Torch:  2.9.1+rocm7.2.0.lw.git7e1940d4
HIP:    7.2.26015-fc0010cf6a
Triton: 3.6.0+git42270451
FlyDSL: 0.3.0
```

The 2026-08-18 best-profile flags were enabled, including MLA Q/cache,
Radix-4 TopK, KDA/B2, and unified M2/M4 cooperative preactivation. The server
used TP8, BF16 weights, FP8 E4M3 KV cache, Triton top-level/decode attention,
AITER prefill attention, 16384 chunked prefill, and a 256 decode graph.

Benchmark workload:

```text
random input/output: 8192/1024
warmups:             64 per concurrency
measured requests:   8 x concurrency
seed:                42
radix cache:         disabled
```

## Results

```text
C    2026-08-18    2026-08-19    delta
2       1142.09       1134.96    -0.62%
4       1967.69       1964.42    -0.17%
8       3091.64       3083.31    -0.27%
16      4797.55       4790.79    -0.14%
32      6707.67       6525.87    -2.71%  first run
64      8710.86       8652.39    -0.67%
```

All sweep points succeeded: `1008/1008`.

The C32 outlier was rerun with the identical workload:

```text
C32 rerun: 6704.43 tok/s
delta vs 2026-08-18: -0.05%
```

Therefore the rebuilt machine reproduces yesterday's performance within the
normal run-to-run variation. The initial C32 result was a transient outlier,
not a stable regression.

## Artifacts

```text
/workspace/kimi-k3-runs/rebuild-best-profile-c2-c64-2026-08-19/
```

The server and benchmark processes were stopped after validation. GPU VRAM
fell substantially after shutdown; residual VRAM is held by other machine
processes, with no Kimi-K3 server or benchmark process remaining.

