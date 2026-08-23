# Kimi-K3 Triton 3.7 best-profile reproduction — 2026-08-19

## Result

The current container reproduces the validated Kimi-K3 best profile without a
stable C2-C64 performance regression. The Triton 3.7-specific
`SGLANG_TRITON_37_EXTEND_LQ576_N32` workaround remained disabled so this run
measured the existing profile unchanged.

## Environment

```text
SGLang: dc6e5a2cd86f6ba049e5860bf8c809ec0dc7525d
AITER:  284a1eb401bb15f6368a68b34eb0cd693ee1fcd3
GPU:    8x AMD Instinct MI355X
Torch:  2.9.1+rocm7.2.0.git7e1940d4
HIP:    7.2.26015-fc0010cf6a
Triton: 3.7.0
```

The repositories were cloned into isolated directories. The container's
existing `/sgl-workspace/sglang`, `/sgl-workspace/aiter`, Torch, and Triton
installations were not modified. Focused K3 validation completed with
`50 passed, 3 warnings`.

## Workload

```text
input/output:       fixed 8192/1024
warmup requests:    64 per concurrency
measured requests:  8 x concurrency
seed:               42
radix cache:        disabled
KV cache:           FP8 E4M3
attention:          Triton top-level/decode, AITER BF16 prefill
```

The same 2026-08-19 best-profile feature flags were used. Server token capacity
was `1,531,386`, matching the prior run.

## Results

The comparison baseline is the 2026-08-19 Triton 3.6 rebuild, using its C32
rerun because the original C32 point was a documented transient outlier.

```text
C    Triton 3.6   Triton 3.7   throughput delta   TPOT 3.7   TPOT delta
2       1134.96      1144.80          +0.87%         14.87       -0.54%
4       1964.42      1973.81          +0.48%         16.83       -0.41%
8       3083.31      3100.09          +0.54%         20.96       -0.43%
16      4790.79      4816.65          +0.54%         25.98       -0.46%
32      6704.43      6694.50          -0.15%         35.61       +0.03%
64      8652.39      8551.82          -1.16%         52.86       +1.26%
```

All `1,008/1,008` measured requests succeeded.

The first C64 point crossed the 0.5% comparison gate, so it was rerun alone:

```text
C64 rerun: 8663.09 tok/s, 52.36 ms median TPOT
vs 3.6:    +0.12% throughput, +0.31% TPOT
requests:  512/512 successful
```

The C64 rerun removes the apparent regression. No stable throughput or TPOT
regression is demonstrated by this sweep. The recorded Torch package version
string differs slightly from the earlier `2.9.1+rocm7.2.0.lw.git7e1940d4`
string, although both identify the same Torch commit and ROCm/HIP version, so
the comparison should be interpreted as a container-level reproduction rather
than a strict single-variable binary A/B.

## Artifacts

```text
/workspace/kimi-k3-runs/triton37-best-profile-c2-c64-2026-08-19/
  focused-tests.log
  server.log
  c{2,4,8,16,32,64}.{log,jsonl}
  c64-rerun.{log,jsonl}
```

The server and benchmark processes were stopped after validation. GPU VRAM
returned to the pre-run level on all eight devices.
