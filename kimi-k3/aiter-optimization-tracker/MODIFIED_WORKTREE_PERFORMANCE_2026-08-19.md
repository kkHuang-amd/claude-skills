# Kimi-K3 modified-worktree performance validation — 2026-08-19

## Result

The uncommitted 26-file SGLang worktree change does not introduce a measurable
C2-C64 performance regression in the Triton 3.7 best profile.

## Tested state

```text
SGLang base: dc6e5a2cd86f6ba049e5860bf8c809ec0dc7525d
SGLang diff: 26 files, +253/-299
AITER:       284a1eb401bb15f6368a68b34eb0cd693ee1fcd3
Torch:       2.9.1+rocm7.2.0.git7e1940d4
HIP:         7.2.26015-fc0010cf6a
Triton:      3.7.0
GPU:         8x AMD Instinct MI355X
```

The exact tested diff is retained as `worktree.patch` with SHA-256:

```text
1fb4a47c6865f0dfdc8c396716c037f205103ce1f3240638c252fb8798853fe6
```

Fresh AITER and SGLang JIT caches were used. Focused K3 validation completed
with `50 passed, 3 warnings`. Server token capacity remained `1,531,386`.

## Workload

The workload and feature flags match the prior Triton 3.7 best-profile run:
fixed 8192/1024 input/output, 64 warmup requests per point, eight measured
requests per concurrency unit, seed 42, disabled radix cache, FP8 E4M3 KV,
Triton top-level/decode attention, and AITER BF16 prefill.

## Results

The baseline is the prior unmodified Triton 3.7 run. Its verified C64 rerun is
used instead of the documented first-run outlier.

```text
C    baseline    modified    throughput delta   modified TPOT   TPOT delta
2     1144.80     1141.84           -0.26%          14.90 ms       +0.20%
4     1973.81     1973.72           -0.00%          16.83 ms       +0.00%
8     3100.09     3102.84           +0.09%          20.93 ms       -0.14%
16    4816.65     4822.11           +0.11%          25.96 ms       -0.08%
32    6694.50     6751.46           +0.85%          35.41 ms       -0.56%
64    8663.09     8793.22           +1.50%          51.57 ms       -1.51%
```

All `1,008/1,008` measured requests succeeded. The largest negative throughput
delta is C2 at `-0.26%`, inside the existing 0.5% comparison gate. The C32/C64
increases are not claimed as stable gains from a single modified-worktree run.

## Artifacts

```text
/workspace/kimi-k3-runs/triton37-modified-c2-c64-2026-08-19/
  focused-tests.log
  server.log
  c{2,4,8,16,32,64}.{log,jsonl}
  worktree.patch
```

The server and benchmark processes were stopped after validation. GPU VRAM
returned to the pre-run level on all eight devices.
