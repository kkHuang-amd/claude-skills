# Kimi-K3 MLA Q/cache fusion result — 2026-08-13

## Decision

Retain the SGLang Kimi-K3 wiring of AITER's existing
`fused_qk_rope_concat_and_cache_mla` operator behind its default-off flag.

```text
BF16 KV: correct and stable, but below the 1% C64 gate
FP8 KV:  accepted; C32/C64 both improve by more than 3%
```

Do not enable it by default for the BF16 production profile. It is a validated
FP8-KV opt-in profile and a reusable building block for a larger MLA boundary.
The source and focused tests are retained. No commit was created.

## Implementation evaluated

The default-off experiment:

```text
SGLANG_K3_AITER_MLA_Q_CACHE_FUSION=1
```

used AITER's gfx950 per-head HIP kernel with one-row identity RoPE buffers:

```text
q_nope + q_pe + k_nope + k_pe
  -> fused Q materialization / identity RoPE / concat / KV-cache write
  -> q_cat passed zero-copy to AITER MLA
  -> save_kv_cache=False
```

The adapter was fail-closed to Kimi-K3, gfx950, AITER decode, DCP1, the
512+64 MLA layout, supported cache dtypes, int64 slot/position vectors and
caller-provided output.

## Bring-up findings

Two integration issues were found and fixed during evaluation:

```text
1. BF16 KV cache exposes k_scale=None.
   The fused operator does not consume scale for BF16, so a device unit-scale
   tensor was supplied.

2. RadixAttention validates/views K before honoring save_kv_cache=False.
   A shape-only [T,1,576] placeholder was required even though the fused op had
   already written K into the paged cache.
```

Both were integration contracts, not kernel correctness failures.

## Correctness and dispatch

```text
new adapter tests:                         7 passed
vendored FlyDSL + adapter focused tests:  53 passed
post-rejection vendored focused tests:    46 passed
GSM8K 50 with fusion active:               1.000
invalid responses:                         0
max_total_num_tokens:                     934463
baseline max_total_num_tokens:            934463
```

The complete Kimi kernel directory also contains a pre-existing test requiring
the absent `cuda.bindings` package; collection failed before running that
unrelated test. No package was installed or updated.

Fresh C64 dispatch evidence:

```text
aiter::fuse_qk_rope_concat_and_cache_mla_per_head_kernel: 24 calls
target CatArrayBatchedCopy<...,3,64,64>:                    0 calls
```

This confirms one fused launch per MLA layer and complete removal of the target
Q concat/materialization chain.

## Paired endpoint result

Both arms used:

```text
TP8
fixed 8192/1024
64 warmups
8 x concurrency measured requests
seed 42
request rate infinity
radix cache disabled
mem-fraction-static 0.85
Triton main/prefill attention + AITER decode
```

Candidate:

```text
C32 throughput: 6371.18 / 6365.40 / 6362.71 tok/s
C32 TPOT:         37.06 /   37.10 /   37.13 ms

C64 throughput: 8096.87 / 8095.34 / 8092.92 tok/s
C64 TPOT:         55.11 /   55.14 /   55.17 ms
```

Contemporaneous baseline:

```text
C32 throughput: 6328.45 / 6325.46 / 6321.47 tok/s
C32 TPOT:         37.43 /   37.39 /   37.42 ms

C64 throughput: 8058.47 / 8052.90 / 8050.64 tok/s
C64 TPOT:         55.45 /   55.50 /   55.52 ms
```

Median deltas:

```text
C32 throughput: +0.63%
C32 TPOT:       -0.86%

C64 throughput: +0.53%
C64 TPOT:       -0.65%
```

Throughput coefficient of variation was below 0.06% for every arm/point, so
the result is stable rather than noise. All requests succeeded.

## FP8 KV result

The same default-off fusion was evaluated with:

```text
--kv-cache-dtype fp8_e4m3
```

Validation:

```text
GSM8K 50:              1.000
invalid responses:     0
max_total_num_tokens:  1868927
fused launches/step:   24
target CatArray:       0
```

Candidate:

```text
C32 throughput: 6355.60 / 6363.50 / 6359.95 tok/s
C32 TPOT:         36.70 /   36.72 /   36.74 ms

C64 throughput: 8069.16 / 8066.77 / 8066.17 tok/s
C64 TPOT:         54.62 /   54.59 /   54.64 ms
```

Contemporaneous FP8 baseline:

```text
C32 throughput: 6156.12 / 6158.87 / 6159.79 tok/s
C32 TPOT:         38.39 /   38.26 /   38.26 ms

C64 throughput: 7791.77 / 7788.82 / 7787.71 tok/s
C64 TPOT:         57.02 /   57.03 /   57.03 ms
```

Median deltas:

```text
C32 throughput: +3.26%
C32 TPOT:       -4.03%

C64 throughput: +3.57%
C64 TPOT:       -4.23%
```

FP8 KV without the fusion is 2.63-3.28% slower in total throughput than the
matched BF16 baseline. The fusion recovers nearly all of that loss:

```text
FP8-fused vs BF16-fused throughput:
  C32 -0.09%
  C64 -0.35%

FP8-fused vs BF16-fused TPOT:
  C32 -1.02%
  C64 -0.94%
```

The stronger FP8 result is consistent with the fused kernel also owning Q/cache
cast and write boundaries, not only the BF16 CatArray materialization.

## FP8 compact trace

The canonical SGLang compact method captured one late-prefill plus short-decode
C64 8192/64 window with CUDA Graph enabled:

```text
8 independent rank traces
18 MiB per rank
139 MiB total
with_stack=false
record_shapes=false
all gzip archives valid
```

Rank 0 exposes 24
`fuse_qk_rope_concat_and_cache_mla_per_head_kernel` calls in the visible decode
replay and no target MLA Q CatArray. Prefill CatArray events remain because the
feature is intentionally decode-only.

```text
/workspace/kimi-k3-runs/mla-q-cache-fp8-trace-2026-08-13/
```

## Memory sizing

The launch uses top-level Triton attention and AITER decode:

```text
--attention-backend triton
--prefill-attention-backend triton
--decode-attention-backend aiter
```

Therefore SGLang's long-context `attention_backend == "aiter"` 0.85 multiplier
does not apply. The final static fraction remains 0.85.

```text
C64 workload tokens:           589824
C64 with 10% margin:           648806
server token capacity:         934463
FP8-KV server token capacity: 1868927
maximum observed token usage:      0.63
maximum running requests:             64
```

Transient queue depth reached 61 while fixed prompts were admitted in
prefill-sized chunks, but all 64 requests became concurrently running and
token usage remained below capacity. This is not the 16-request KV-capacity
failure seen in pure-AITER-main-backend configurations.

## Acceptance gate

Predeclared gate:

```text
C64 median throughput or TPOT improvement >= 1%
C32 regression <= 0.5%
correctness/capacity matched
```

BF16 KV observed only +0.53% throughput / -0.65% TPOT at C64, below the gate.
FP8 KV observed +3.57% throughput / -4.23% TPOT at C64 and passes the gate.

Retain the implementation default-off. Enable it for the validated Kimi-K3 FP8
KV profile; do not make it a BF16 production default based on this result.

## 2026-08-14 full A8W4 FP8-KV sweep

Commit `e5f2bd991` on `perf/k3_opts_0812` was launched with A8W4, FP8 KV and
the fusion enabled. The fixed 8192/1024 C2-C64 sweep used 64 warmups and eight
measured requests per concurrency unit. All 1,008 requests succeeded.

Compared with the prior same-machine production reproduction:

```text
C    Prior tok/s   Fused tok/s   Delta    Prior TPOT   Fused TPOT
2        980.24       1013.37   +3.38%       17.47       16.88 ms
4       1764.84       1811.92   +2.67%       18.92       18.30 ms
8       2921.82       2980.01   +1.99%       22.15       21.58 ms
16      4492.89       4563.04   +1.56%       27.71       27.09 ms
32      6295.63       6356.34   +0.96%       37.59       36.90 ms
64      8014.22       8072.59   +0.73%       55.58       54.58 ms
```

Every point improves throughput and median TPOT. The gain decreases with
concurrency but remains positive at C64. This comparison is historical rather
than an interleaved paired A/B, so the sub-1% C32/C64 deltas should retain the
usual run-to-run-noise caveat.

The optional B2 profile was then enabled without changing the C2 workload:

```text
default fused C2: 1013.37 tok/s, 16.88 ms median TPOT
B2 fused C2:      1109.82 tok/s, 15.34 ms median TPOT
delta:            +9.52% throughput, -9.12% TPOT
```

All 16 measured requests succeeded. This is 3.43% above the prior recorded B2
C2 result of 1073.02 tok/s and preserves the policy of keeping B2 optional.

## Retained artifacts

```text
/workspace/kimi-k3-runs/mla-q-cache-2026-08-13/paired-summary.json
/workspace/kimi-k3-runs/mla-q-cache-2026-08-13/endpoint/
/workspace/kimi-k3-runs/mla-q-cache-2026-08-13/trace/candidate-dispatch/
/workspace/kimi-k3-runs/mla-q-cache-2026-08-13/trace/candidate-dispatch-v2/
/workspace/kimi-k3-runs/mla-q-cache-2026-08-13/candidate-v2-gsm8k-50.log
/workspace/kimi-k3-runs/mla-q-cache-focused-tests.log
/workspace/kimi-k3-runs/mla-q-cache-2026-08-13/post-reject-focused-tests.log
/workspace/kimi-k3-runs/mla-q-cache-fp8-2026-08-13/paired-summary.json
/workspace/kimi-k3-runs/mla-q-cache-fp8-2026-08-13/endpoint/
/workspace/kimi-k3-runs/mla-q-cache-fp8-2026-08-13/trace/dispatch/
/workspace/kimi-k3-runs/mla-q-cache-fp8-2026-08-13/candidate-gsm8k-50.log
/workspace/kimi-k3-runs/mla-q-cache-fp8-trace-2026-08-13/
/workspace/kimi-k3-runs/a8w4-fused-qkv-fp8kv-sweep-2026-08-14/
```
