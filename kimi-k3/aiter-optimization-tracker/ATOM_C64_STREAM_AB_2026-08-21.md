# ATOM Kimi-K3 current-main C64 stream A/B — 2026-08-21

## Decision

Do not enable ATOM-style dual-stream MoE at C64 in SGLang. On the current
ATOM/AITER stack, multi-stream is decisively slower than single-stream:

```text
total-token throughput: multi 8680.09 vs single 9878.91 tok/s (-12.14%)
median TTFT:            multi 15695.19 vs single 14299.91 ms    (+9.76%)
median TPOT:            multi    51.22 vs single    44.65 ms    (+14.70%)
```

Both sides completed 512/512 measured requests. This is a real dispatch
difference, not an ineffective environment flag: the multi graph contains
`maybe_dual_stream_forward`, while the single graph contains no such call.

Combined with the current C2 result (`+1.73%` conservative throughput,
`-2.08%` TPOT), the policy must be token-count/concurrency bounded. Start any
SGLang prototype at exact C2 or at most below the measured C64 crossover;
do not copy ATOM's current default threshold of 1024 unchanged.

## Revisions and runtime

```text
ATOM:     27f8639bad0948755630236aa7796b4b21348668
AITER:    dc4bdf1c142181ad90b7f6948564126df4c05fde
Python:   3.10.12
PyTorch:  2.9.1+rocm7.2.0.git7e1940d4
HIP:      7.2.26015-fc0010cf6a
FlyDSL:   0.3.1
GPU:      8x AMD Instinct MI355X / gfx950
```

The AITER source includes the uncommitted K3 caller-output, stage1 scratch,
and BF16 front-row migration used for the SGLang compatibility work. ATOM and
AITER were loaded through `PYTHONPATH`; neither was installed.

## Workload

The server followed `recipes/Kimi-K3.md`. The serving workload matches the
historical 2026-08-14 C64 comparison:

```text
input/output:       fixed 8192/1024
concurrency:        64
warmup requests:    128
measured requests:  512
random range ratio: 1.0
seed:               0
request rate:       infinite
ignore EOS:         enabled
TP:                 8
CUDA graph mode:    FULL
```

Multi-stream:

```text
ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024
```

Single-stream:

```text
ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0
```

## Results

```text
metric                    multi-stream   single-stream   multi delta
successful requests       512            512             equal
benchmark duration        543.61 s       477.64 s        +13.81%
total-token throughput    8680.09        9878.91         -12.14%
output throughput          964.45        1097.66         -12.14%
median TTFT              15695.19 ms    14299.91 ms      +9.76%
median TPOT                 51.22 ms       44.65 ms     +14.70%
```

## Graph verification

The multi graph contains `torch.ops.aiter.maybe_dual_stream_forward`; its
representative generated graph was:

```text
/tmp/torchinductor_root/br/cbrx7dzhvfrf4jxwbpm57ddnzl2se6zqy4swkovushwnocj5cguh.py
```

The single-server graph cohort contained 368 generated Python graph files and
zero `maybe_dual_stream_forward` occurrences. Its MoE graph used the inline
single-stream `aiter.moe_forward` path.

## Historical comparison

The 2026-08-14 stack measured:

```text
multi  8787.26 tok/s
single 8411.14 tok/s
multi advantage +4.47%
```

Relative to that run, current multi-stream is about `-1.22%`, while current
single-stream is about `+17.45%`. The crossover inversion is therefore driven
primarily by the much faster current single-stream path, including its deferred
routed/shared add and subsequent runtime/kernel improvements. It is not valid
to reuse the historical `+4.47%` C64 claim for current code.

## Implication for SGLang

SGLang already preserves routed/shared/prefix fusion through its `_add3` MoE
tail. A multi-stream prototype should:

1. retain `_add3` instead of adopting ATOM's tuple-return limitation;
2. initially cover exact C2 only;
3. use a token-count gate that disables the path before C64;
4. validate C2/C4/C8/C16/C32/C64 rather than assuming monotonic benefit;
5. preserve TP collective order and FULL HIP graph fork/join topology.

## Artifacts

```text
/workspace/kimi-k3-runs/atom-c64-stream-ab-2026-08-21/
  multi/c64.{json,log}
  single/c64.{json,log}
  multi/server.log
  single/server.log
  multi-run.log
  single-run.log
  run_variant.sh
```

No server or benchmark process remains. All GPUs returned to baseline VRAM.
No source, package, commit, or cache was changed by this A/B.
