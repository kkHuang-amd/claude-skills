# Kimi-K3 C64 ATOM/SGLang trace and code comparison — 2026-08-13

## Result

Because the current SGLang path is effectively single-stream, the primary
comparison is ATOM single-stream versus SGLang. ATOM single-stream is 4.81%
faster at C64. ATOM multi-stream is retained as a separate future optimization,
not mixed into the primary kernel/code comparison.

```text
Machine: crsuse2-m2m-002.crusoe.amd.com
GPU:     8x AMD Instinct MI355X

SGLang f9dd3a066 / AITER 284a1eb40:
  8014.22 tok/s, 55.58 ms median TPOT

ATOM 5479c5af3 / AITER e8b4507e5, multi-stream:
  8742.34 tok/s, 49.84 ms median TPOT
  vs SGLang: +9.09% throughput

Same ATOM image, dual-stream disabled:
  8399.50 tok/s, 53.33 ms median TPOT
  vs multi-stream: -3.92% throughput, +7.01% TPOT
  vs SGLang:       +4.81% throughput, -4.04% TPOT, -6.58% TTFT
```

The primary single-stream gap is `385.28 tok/s`. The separate dual-stream
increment is `342.84 tok/s` (+4.08% over ATOM single-stream).

## Old versus current ATOM

The previously measured ATOM stack produced:

```text
ATOM f782218a5 / AITER 284a1eb40:
  8380.39 tok/s, 52.66 ms median TPOT
```

The current-image single-stream result is only 0.23% faster in total throughput
than that older-environment result:

```text
current single vs recorded ATOM:
  throughput: +0.23%
  TPOT:       +1.28% (slower)
  duration:   -0.23%
```

The current multi-stream result is 4.32% faster than recorded ATOM. Numerically,
almost all of that gain disappears when dual-stream is disabled in the current
image. However, the recorded ATOM source already contains the dual-stream code,
so this does not prove a source-code introduction. It indicates that runtime
activation, graph mode, or environment behavior is material.

## Commit ancestry

The image SHAs are not newer than the previously tested SHAs:

```text
ATOM 5479c5af3 is an ancestor of f782218a5.
AITER e8b4507e5 is an ancestor of 284a1eb40.
```

The later ATOM checkout adds:

```text
16c20d30  KDA prefill on AITER; drop FLA dependency
f782218a  attention-residual / FLA fusion changes
```

Those changes should generally favor `f782218a`, not the image's older
`5479c5af`:

```text
f782 attention residual:
  single-pass fusion
  output RMSNorm folded into the store
  deferred MoE routed/shared add folded into the next residual mix

f782 KDA prefill:
  AITER chunk_kimi_delta_attn instead of FLA chunk_kda
```

The following are unchanged between the ATOM SHAs:

```text
kda_attention_with_output opaque boundary
dual-stream MoE registration and 1024-token threshold
online-quant recipe exclusions
CUDAGraph recipe mode
```

The later AITER core-only checkout adds:

```text
gfx950 64-bit address fix
caller-provided fused_moe output buffers
Kimi FlyDSL stage1 scratch reuse and its gate
```

The AITER delta is too narrow to explain the measured gap:

```text
caller output buffers: used by SGLang; reduces allocation/copy overhead
stage1 scratch reuse:  not exercised by the captured Kimi A16W4/Opus paths
qh64 MLA binary:       not exercised; traces use qh16/gqa_d192
```

Therefore the current image's higher endpoint result cannot be explained as
"newer ATOM/AITER commits." Runtime versions and whether the dual-stream branch
actually executes must be treated as first-class variables.

## ATOM dual-stream code path

At C64, ATOM's default
`ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024` selects
`dual_stream_moe_forward`. It:

```text
main stream: routed-expert path
alt stream:  shared-expert GEMMs
join:        before the result add / dependent all-reduce ordering
```

`torch.ops.aiter.maybe_dual_stream_forward` is an ATOM-defined opaque custom op
used to keep this dispatch outside unsafe piecewise tracing. Setting the
threshold to zero calls `single_stream_moe_forward` directly.

ATOM's separate `torch.ops.aiter.kda_attention_with_output` is also an
ATOM-defined opaque splitting boundary. It is registered in the `aiter`
namespace but is not an operator supplied by AITER commit `e8b4507e5`.

## ATOM RTL trace evidence

The matched RTL runs used C64, 64 requests, fixed 8192/64, no warmups and
ignore-EOS. RTL strongly perturbs endpoint latency, so its wall-clock and
per-kernel durations are diagnostic only.

Across all eight ranks:

```text
                         multi-stream       single-stream
active queues                  4                  3
mean kernel count         348,409            516,409
mean kernel sum            58.82 s             65.26 s
mean busy union             58.48 s             65.26 s
mean measured overlap        0.35 s              ~0 s
```

On representative rank 0, the substantive second multi-stream queue contains
about 35.3k operations and 5.99 seconds of summed kernel time. Single-stream
places essentially all work on one queue.

The full-window single-minus-multi kernel-time increase is concentrated in:

```text
KDA family:          +5.73 s/rank mean
copy/fill:           +2.78 s/rank mean
norm/activation:     +1.06 s/rank mean
other:               +0.96 s/rank mean
```

Most of those full-window differences are profiler-amplified prefill effects.
Approximate decode-tail windows have nearly identical kernel counts
(`275.1k` versus `275.6k`) and much smaller non-collective differences. RTL
disproportionately penalizes the multi-stream collective path and even reverses
the unprofiled TPOT ordering. Consequently, use the trace to establish queue
topology and kernel composition, not to estimate the production gain.

## SGLang trace evidence

The matched SGLang PyTorch trace completed on all eight ranks. Rank 0's
approximate decode tail contains:

```text
kernel span:       18.05 s
kernel busy union: 15.45 s
kernel sum:        15.46 s
active streams:         2

main stream:       62,677 kernels, 15.446 s
secondary stream:    664 kernels,  0.011 s
```

SGLang is therefore effectively single-stream for this MI355 production
configuration. The main decode-tail families are:

```text
dense GEMM:          4.07 s
collective:          3.37 s
MLA/attention:       2.89 s
MoE GEMM:            2.28 s
attention residual:  1.44 s
KDA:                 0.59 s
route/sort/top-k:    0.49 s
```

The production command has no EP a2a and does not enable
`SGLANG_K3_AR_FUSION`. On MI355, that fusion is not auto-enabled, so SGLang's
shared/routed side-stream branch is not active. This matches the trace.

Absolute ATOM/SGLang kernel durations are not directly comparable because ATOM
was captured with RTL on Torch 2.13/ROCm 7.14 while SGLang used PyTorch
profiler on Torch 2.9/ROCm 7.2. Kernel names and dispatch structure remain
useful.

## 2026-08-14 same-environment A16W4 kernel check

ATOM `f782218a` was successfully launched in the SGLang Torch 2.9.1,
ROCm 7.2, Triton 3.6 and AITER `284a1eb` environment. Its native Torch profiler
segfaulted, but ROCprofv3 captured a 64/64-request C64 fixed 8,192/64 run.

Exact decode-kernel distributions:

```text
kernel       framework   calls   mean us   p50 us   p10 us   p90 us
gemm1 xcd4   ATOM         5887      97.93    93.08    79.92   118.16
gemm1 xcd4   SGLang TP0   5888     127.16   123.24   114.84   147.16

gemm2        ATOM         5890      53.19    53.60    41.04    63.48
gemm2        SGLang TP0   5888      73.78    73.00    65.16    84.40
```

The representative p50 gap is 1.324x for gemm1 and 1.362x for gemm2. The
previously inspected 70/30 us ATOM calls are low-tail samples, so a single-call
comparison overstated the gap at roughly 2x. The same environment reduces but
does not eliminate the difference.

A matched SGLang ROCprof attempt hit the 300-second scheduler watchdog during
graph warmup and emitted no trace. Therefore the current comparison aligns the
machine, runtime, AITER and workload, but still compares ATOM ROCprof with
SGLang Kineto.

Caller-contract and isolated replay subsequently resolved the discrepancy:

```text
SGLang pre-fix input: [64,3584], stride [6016,1], non-contiguous
ATOM input:           [64,3584], stride [3584,1], contiguous
```

The A16W4 gemm1 port has no runtime row-stride argument. The non-contiguous
input failed isolated correctness on 22.4% of elements, so SGLang now
materializes that A16W4 input. GSM8K 50 remains 1.000. The copy has no endpoint
performance effect (`7354.88 → 7354.28 tok/s` at C64), proving it was a
correctness ABI fix rather than the timing root cause.

The timing root cause is route concentration. The nominally matched random
clients generated different prompts:

```text
SGLang C64: 582 unique experts, max 8 routes/expert
ATOM C64:   454 unique experts, max 13 routes/expert
```

The port pads work per active expert in BM32 blocks. Isolated replay gives:

```text
active experts   stage1 us   stage2 us
454                 124.30       71.25
582                 159.49       88.85
ratio                1.283x       1.247x
```

This reproduces most of the observed 1.324x/1.362x trace gap. The identical
kernel symbol did not imply identical runtime work; SGLang's route distribution
created more active padded expert blocks.

Artifacts:
`/workspace/kimi-k3-runs/atom-c64-same-env-2026-08-14/`.

## Other major stack differences

The remaining 4.81% single-stream ATOM advantage is not isolated. Important
differences include:

```text
ATOM:
  FP8 KV cache
  PTPC-FP8 online quantization for eligible dense projections
  SiTUv2 A16W4 / FP4 MoE kernel stack
  NCCL plus AITER cross-device reductions
  opaque compile boundaries for KDA and dual-stream MoE

SGLang:
  BF16 model dtype and auto KV dtype
  Opus A8W4 stage1/stage2 MoE
  Q8 two-shot quick-reduce
  vendored fused KDA decode + f_b
  fused attention-residual path
```

Both ATOM A/B logs also contain untuned-shape fallback warnings, so the
multi-stream result is not explained by a cleaner tuning-file match.

## MLA pre-attention fusion candidate

The traces expose a concrete per-layer materialization difference before MLA:

```text
SGLang rank0:
  CatArrayBatchedCopy
  1536 calls = 24 MLA layers x 64 decode steps
  167.98 ms summed PyTorch-profiler kernel time

ATOM single-stream rank0:
  aiter::fuse_qk_rope_concat_and_cache_mla_per_head_kernel
  1534 captured calls (approximately the same 24 x 64 coverage)
  9.48 ms summed RTL kernel time
```

Absolute times cannot be subtracted across profilers, but the one-to-one call
counts establish that SGLang materializes a Q concat once per MLA layer per
decode step while ATOM fuses:

```text
Q no-PE copy / optional query quantization
Q PE handling
Q output concatenation
K latent + PE cache write / optional cache quantization
```

Kimi-K3 has no positional rotation. ATOM still uses the fused AITER kernel by
providing an identity RoPE cache (`cos=1`, `sin=0`). SGLang constructs
`KimiK3MLAAttention` with `skip_rope=True`; its generic ROCm fused gate requires
`rotary_emb is not None`, so the Kimi path does not enter that fusion.

The required HIP kernel and public AITER API already exist in the current
AITER checkout:

```text
aiter.fused_qk_rope_concat_and_cache_mla
aiter::fuse_qk_rope_concat_and_cache_mla_per_head_kernel
```

This is primarily a SGLang wiring/fallback task, not a new kernel port.

Recommended implementation shape:

```text
Kimi-specific gfx950 + AITER-backend gate
identity RoPE cache or a dedicated no-RoPE wrapper
produce q_cat directly into caller-owned output
write latent/PE KV cache in the same launch
pass q_cat zero-copy into MLA
save_kv_cache=False
fail closed to the current split/cat/cache path
```

Required gates:

```text
BF16 and FP8 KV-cache correctness
slot_mapping=-1 / CUDA-graph padding
graph replay and fixed BS64
GSM8K smoke
capacity unchanged
paired C32/C64 endpoint
```

Because the raw profiler ratio is cross-tool, first run an isolated matched
operator/chain benchmark. The 1536-call CatArray removal is nevertheless a
high-confidence launch/materialization opportunity.

### Evaluated result

The wiring was implemented and passed operator, graph, dispatch, GSM8K and
capacity checks. It replaced the target CatArray with exactly 24 fused
per-head launches per decode step. Three-round paired endpoint medians were:

```text
BF16 C32: +0.63% throughput, -0.86% TPOT
BF16 C64: +0.53% throughput, -0.65% TPOT

FP8 C32: +3.26% throughput, -4.03% TPOT
FP8 C64: +3.57% throughput, -4.23% TPOT
```

BF16 is below the predeclared 1% C64 gate. FP8 KV passes it, so the integration
is retained default-off and accepted for the Kimi FP8-KV profile. Details:
[`MLA_Q_CACHE_FUSION_RESULTS_2026-08-13.md`](MLA_Q_CACHE_FUSION_RESULTS_2026-08-13.md).

## Recommended next experiments

Continue with:

```text
1. Run ATOM single-stream without PTPC-FP8 online quantization.
   This measures how much of the 4.81% gap comes from eligible dense FP8
   projections versus framework/kernel scheduling.

2. Run ATOM `5479c5af` and `f782218a` single-stream in the same current image.
   This directly measures the attention-residual and AITER-KDA-prefill source
   delta without changing runtime.

3. Run exact SGLang and ATOM single-stream in one identical Torch/ROCm image,
   preferably under the same RTL tracer.

4. Compare the remaining kernel stacks:
   ATOM SiTUv2 A16W4 versus SGLang Opus A8W4 MoE;
   ATOM NCCL/cross-device reductions versus SGLang Q8 quick-reduce;
   ATOM opaque KDA boundary versus SGLang fused KDA decode + f_b.
```

Only after the single-stream gap is isolated, evaluate dual-stream as a
separate SGLang feature:

```text
overlap shared-expert GEMMs with the routed-expert path
preserve collective ordering
independent gate and fail-closed fallback
focused correctness and graph replay
GSM8K smoke
capacity unchanged
paired C32/C64 endpoint
```

The current cross-profiler traces cannot distinguish runtime effects from
kernel-code effects precisely.

## Retained artifacts

```text
ATOM multi RTL:
  /workspace/kimi-k3-runs/c64-stream-ab-2026-08-13/multi/

ATOM single RTL:
  /workspace/kimi-k3-runs/c64-stream-ab-2026-08-13/single/rtl/

SGLang runtime traces:
  /workspace/kimi-k3-runs/c64-repro-2026-08-13/trace/sglang-current/

Analysis JSON/scripts:
  /workspace/kimi-k3-runs/c64-stream-ab-2026-08-13/*analysis.json
  /workspace/kimi-k3-runs/c64-stream-ab-2026-08-13/analyze_rtl_db.py
  /workspace/kimi-k3-runs/c64-repro-2026-08-13/analyze_chrome_trace.py
```
