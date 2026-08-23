# Triton MLA decode Q/cache fusion — 2026-08-17

## Decision

Accept BF16 Q output + FP8 KV cache write for the optional tuned Triton decode
profile. Reject FP8 Q/Q-PE.

## Implementation

Kimi's existing AITER fused operator supports independent Q-output and
KV-cache dtypes. The Triton decode path now enters the same fused Q/cache
boundary as AITER decode:

```text
Triton decode: BF16 q_out + FP8 kv_cache
AITER decode:  FP8 q_out  + FP8 kv_cache
```

The adapter accepts mixed BF16-Q/FP8-KV and uses separate Q/K scale tensors.
The `triton_mla` runtime backend name is included in the dispatch guard.

## BF16 Q result

```text
operator tests: 9 passed
tuned Triton decode tests: 16 passed, 2095 subtests passed
GSM8K 50:   1.000, invalid 0
GSM8K 1319: 0.951, invalid 0.001
```

Trace:

```text
fuse_qk_rope_concat_and_cache_mla_per_head_kernel: 24 calls
target 3-input Q CatArray:                              0 calls
```

C64 fixed 8,192/1,024:

```text
unfused Triton decode: 8597.81 tok/s, 52.86 ms median TPOT
BF16-Q fused:          8656.21 tok/s, 52.32 ms median TPOT
delta:                    +0.68%,                  -1.02%
```

Median TTFT changes by +0.65%, within run noise.

## Rejected FP8 Q/Q-PE result

The Triton stage1 kernel can consume FP8 Q, but using one FP8 tensor also
quantizes the 64-wide Q-PE region. The experiment used a separate unit Q scale.

```text
GSM8K 50:   1.000
GSM8K 1319: 0.929, invalid 0.001
C64:        8553.47 tok/s, 53.04 ms median TPOT
```

Versus accepted BF16 Q:

```text
throughput: -1.19%
TPOT:       +1.38%
TTFT:       +1.57%
accuracy:    0.951 -> 0.929
```

FP8 Q/Q-PE is rejected and its runtime flag was removed.

## Artifacts

```text
/workspace/kimi-k3-runs/triton-q-cache-fusion-2026-08-17/
  bf16-q/
    c64.log
    gsm8k-50-v2.log
    gsm8k-1319.log
    traces-v2/
  fp8-qpe/
    c64.log
    gsm8k-50.log
    gsm8k-1319.log
    traces/
```
