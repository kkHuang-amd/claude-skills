# Kimi-K3 TP8 all-reduce shape investigation — 2026-08-11

## Why this follow-up was required

An initial operator run appeared to show a 3–6% B32 gain from forcing AITER's
8-GPU one-stage all-reduce, while the 8192/1024 C32 endpoint regressed.

The ROCm Kimi-K3 path does not enable model side streams:

```text
kimi_k3.py: alt_streams = None if _is_hip else [...]
```

The production traces agree: kernel-duration sum and busy-union differ by less
than 0.1% at decode C2/C32. The regression therefore could not be attributed
to overlapping MoE/attention kernels losing resources to all-reduce.

## Instrumentation

Temporary, default-off controls were added to the AITER custom all-reduce
dispatcher:

- log each unique `bytes → stage, blocks` decision;
- force one-stage for an exact byte count;
- force two-stage for an exact byte count;
- reproduce the previous global 512 KiB threshold.

The instrumentation was removed after the experiment.

## Dispatch map

The original 80 KiB threshold uses two-stage for these graph-captured shapes:

```text
bytes   rows at hidden=7168 bf16
86016    6
114688   8
172032  12
229376  16
258048  18
344064  24
458752  32
516096  36
```

The 512 KiB threshold changed all eight to one-stage. The exact-B32 profile
changed only 458752 bytes.

## Correct paired operator benchmark

Eight ranks stayed initialized while every shape captured both algorithms.
Five paired graph-timing rounds were measured; the reported latency is the
maximum rank per round, then the median across rounds.

```text
bytes   rows   1-stage us   2-stage us   1-stage speedup
86016     6       13.961        20.559       1.473x
114688    8       21.818        21.465       0.984x
172032   12       22.411        12.869       0.574x
229376   16       16.321        21.731       1.332x
258048   18       21.603        19.527       0.904x
344064   24       21.261        12.311       0.579x
458752   32       21.467        12.443       0.580x
516096   36       24.077        13.146       0.546x
```

The original apparent B32 gain was a dispatch mismatch: the first test reused
a stale prebuilt `module_custom_all_reduce.so`, so setting the experimental
environment variable did not change the kernel. Rebuilding into a fresh JIT
directory and logging the selected stage exposed the error.

One-stage is actually about 72% slower than two-stage at B32. The stage
crossover is non-monotonic: B6 and B16 favor one-stage, while B12 and B24+
strongly favor two-stage.

## Production traces

Three profiles were compared under the same JIT cache and workload:

```text
baseline:  original 80 KiB threshold
global:    all shapes below 512 KiB use one-stage
exact:     only 458752 bytes use one-stage
```

### C32 decode

```text
baseline 2-stage: p50 11.40 us, p95 17.84 us
global   1-stage: p50 18.24 us, p95 21.96 us
exact    1-stage: p50 18.16 us, p95 21.92 us
```

Critical-tail collective time per generated token:

```text
baseline: 111.14 us
global:   123.32 us  (+11.0%)
exact:    139.89 us  (+25.9%)
```

The exact profile still has unrelated two-stage collectives; the value above
is the sum of its one-stage and two-stage critical tails.

Rank skew did not explain the regression. The one-stage p95 rank spread was
7.72 us versus 10.40 us for baseline two-stage: rank spread improved, while
the kernel itself became slower.

### Prefill

All profiles retained the same small one-stage custom all-reduces:

```text
baseline p50 8.52 us
global   p50 8.40 us
exact    p50 8.44 us
```

The threshold experiments did not materially alter this prefill trace.

## Endpoint results

Contemporaneous C32:

```text
baseline  6199.77 tok/s
global    6085.06 tok/s  (-1.85%)
exact     6096.66 tok/s  (-1.66%)
```

Exact-B32 additional gates:

```text
GSM8K 50 = 1.000
C2  = 968.29 tok/s
C16 = 4439.09 tok/s
C32 = 6096.66 tok/s
```

C2/C16 were effectively flat against the selected-final reference; C32 failed
the production gate decisively.

## Conclusion

There is no microbenchmark/E2E contradiction after verifying dispatch:

1. the initial microbenchmark did not switch algorithms;
2. a correctly forced B32 one-stage kernel is slower in both isolated TP8 graph
   timing and production traces;
3. endpoint throughput regresses by the expected amount;
4. neither same-rank kernel overlap nor rank skew is the cause.

Do not raise the global threshold and do not force B32 one-stage.

The shape sweep reveals a separate future candidate: exact B6 and B16
one-stage dispatch. That should be evaluated independently rather than encoded
as a monotonic byte threshold.

Artifacts:

```text
stage2-runs/2026-08-11-production-trace/ar-shape-investigation/
```
