# ATOM Kimi-K3 current-main C2 multi-stream A/B — 2026-08-21

## Decision

Current ATOM multi-stream MoE is beneficial at C2 on the matched Kimi-K3
recipe. Across five runs, multi-stream was faster in every run. Use ATOM's
fork/join and collective ordering as the reference for an SGLang prototype,
but retain an opt-in gate until SGLang passes its own matched endpoint and
correctness validation.

The current measured effect is smaller and less stable than the historical
2026-08-14 result. The conservative aggregate is the five-run median:

```text
total-token throughput: 780.43 vs 767.18 tok/s  (+1.73%)
median TPOT:              22.17 vs  22.64 ms     (-2.08%)
```

Five-run means:

```text
total-token throughput: 795.15 vs 764.61 tok/s  (+3.99%)
median TPOT:              21.85 vs  22.68 ms     (-3.63%)
```

## Revisions and runtime

```text
machine:      crsuse2-m2m-030
GPU:          8x AMD Instinct MI355X / gfx950
ATOM:         27f8639bad0948755630236aa7796b4b21348668
AITER:        dc4bdf1c142181ad90b7f6948564126df4c05fde
CK submodule: 15e12dd7f25ee583617c78f66cb502ff9916585f
Torch:        2.9.1+rocm7.2.0.lw.git7e1940d4
Triton:       3.7.0+amd.rocm7.2.0.git89002410
FlyDSL:       0.3.1
```

ATOM and AITER were used directly from isolated source clones through
`PYTHONPATH`; neither was installed into the Python environment. With explicit
user approval, global FlyDSL was upgraded from 0.2.4 to AITER main's pinned
0.3.1. No other package was changed.

The original AITER `d9e5ef7` failed current ATOM's KDA graph capture because
its A8W8 bpreshuffle fallback rejected `M64/N1536/K128`. Isolated AITER main
fixed that dispatch gap. AITER main then required FlyDSL 0.3.1; using 0.2.4
failed while emitting `rocdl.raw.ptr.buffer.load`.

## Recipe and workload

The server followed `recipes/Kimi-K3.md`:

```text
model:                    /shared_nfs/models/Kimi-K3
tensor parallel:          8
KV cache:                 FP8
max model length:         16384
max sequences:            64
max batched tokens:       16384
GPU memory utilization:   0.93
cache block size:         128
prefix caching:           disabled
online quantization:      PTPC FP8 with recipe exclusions
CUDA graph mode:          FULL
```

Matched endpoint workload:

```text
input/output:       fixed 8192/1024
concurrency:        2
measured requests:  16 per round
warmup requests:    4 per round
rounds:             5 per side
random range ratio: 1.0
seed:               0
request rate:       infinite
ignore EOS:         enabled
```

Multi-stream used:

```text
ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024
```

Single-stream used:

```text
ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0
```

## Per-round results

Each row is total-token throughput followed by median TPOT:

```text
round  multi-stream             single-stream            throughput delta
1      775.32 tok/s, 22.20 ms   762.95 tok/s, 22.80 ms   +1.62%
2      780.43 tok/s, 22.17 ms   758.09 tok/s, 22.66 ms   +2.95%
3      779.71 tok/s, 22.18 ms   767.18 tok/s, 22.64 ms   +1.63%
4      800.92 tok/s, 22.15 ms   767.26 tok/s, 22.64 ms   +4.39%
5      839.38 tok/s, 20.56 ms   767.57 tok/s, 22.64 ms   +9.36%
```

All 160 measured requests completed. Multi-stream throughput was higher and
median TPOT lower in all five same-index comparisons.

The multi-stream side warmed upward in rounds four and five while the
single-stream side was stable. Because the two modes require separate FULL
graph captures and were not interleaved in one server process, use the median
gain as the conservative result. Do not substitute the mean or the final round
for a production claim.

## ATOM behavior to reproduce in SGLang

Current ATOM creates one shared alternate stream and threads it into every K3
MoE layer. For token counts `0 < M <= threshold`, it:

1. queues routed pre-all-reduce work on the current stream;
2. runs the shared-expert branch on the alternate stream;
3. runs the shared latent-path all-reduce on that alternate stream;
4. waits before launching the routed all-reduce so the single TP communicator
   preserves collective order;
5. completes routed norm/up projection and joins before adding shared output.

The path is disabled under TBO and PIECEWISE graph mode. ATOM allows eager and
whole-model FULL graph capture because the fork/join topology is captured
consistently.

For SGLang, start default-off at exact C2 and verify:

```text
correct stream/event ordering under HIP graphs
identical TP collective order on all ranks
no overlap with an existing secondary-stream policy
fresh-JIT dispatch evidence
five-round matched C2 endpoint A/B
GSM8K and graph-replay correctness
```

## Artifacts

```text
/workspace/kimi-k3-runs/atom-c2-stream-ab-2026-08-21/
  multi/round{1..5}.{json,log}
  single/round{1..5}.{json,log}
  multi/server-flydsl031.log
  single/server.log
```

Earlier failed-start logs are retained in `multi/server.log` and
`multi/server-isolated-aiter.log` as dependency diagnostics. No ATOM server or
benchmark process remains, and all GPU VRAM was released.
