# Kimi-K3 AITER prefill + tuned Triton decode — 2026-08-17

## Decision

Retain as an optional high-concurrency profile. It materially improves
C16-C64 throughput and prefill latency, but regresses C2 throughput.

Use a top-level Triton backend with explicit phase overrides:

```bash
export SGLANG_AITER_FP8_PREFILL_ATTN=0
export SGLANG_MLA_DECODE_TUNE=1
export SGLANG_K3_AITER_MLA_Q_CACHE_FUSION=1

--attention-backend triton
--prefill-attention-backend aiter
--decode-attention-backend triton
--kv-cache-dtype fp8_e4m3
--mem-fraction-static 0.85
```

This preserves FP8 KV storage. Fresh prefill runs BF16 Opus FMHA directly;
cached-prefix rows are gathered from FP8 cache and cast to BF16 only when
needed.

## Performance

Fixed 8,192/1,024, TP8, 64 warmups, eight measured requests per concurrency
unit, seed 42, radix cache disabled:

```text
C    Total tok/s   Throughput delta   Median TTFT   TTFT delta   Median TPOT
2         982.24          -3.07%          885.08       -2.60%        17.46
4        1794.62          -0.95%         1516.77      -13.29%        18.65
8        2988.43          +0.28%         2397.00       -9.29%        21.80
16       4673.87          +2.43%         4139.17      -11.10%        26.86
32       6598.64          +3.81%         7825.16      -10.11%        36.34
64       8597.81          +6.51%        14622.08      -16.05%        52.86
```

All 1,008 requests succeeded. FP8 token capacity is 1,861,342.

Optional B2 C2:

```text
default C2:  982.24 tok/s, 17.46 ms median TPOT
B2 C2:     1071.03 tok/s, 15.94 ms median TPOT
delta:       +9.04% throughput, -8.71% TPOT
```

B2 recovers most of the tuned-decode C2 loss, but remains 3.50% below the
previous AITER-decode B2 result of 1,109.82 tok/s.

Using top-level `--attention-backend aiter` applies an additional memory
multiplier (`0.93 → 0.7905`) and reduces capacity to 581,538, below the C64
workload. That launch form is rejected.

## Compact C64 trace

The canonical late-prefill plus short-decode method captured eight independent
rank traces with stacks/shapes disabled and CUDA Graph enabled:

```text
8 trace files
18-19 MiB per rank
144 MiB total
all gzip archives valid
all files below 500 MiB
```

TP0 evidence:

```text
AITER/Opus BF16 prefill:
  gqa_d192_v128_kernel
  408 calls, 201.86 ms total, 506.44 us median

Tuned Triton MLA decode:
  _fwd_grouped_kernel_stage1
  24 calls, 2.47 ms total, 103.30 us median

  _fwd_kernel_stage2
  24 calls, 0.21 ms total, 8.66 us median
```

This confirms both PR paths execute in the same trace.

## Artifacts

```text
/workspace/kimi-k3-runs/pr34837-pr34580-sweep-2026-08-17/
  performance-table.csv
  mixed-capacity/c2.log ... c16.log
  mixed-clean/c32.log
  mixed-clean/c64.log
  run_compact_c64_trace.sh
  trace-server.log
  trace-client.log
  trace-analysis-tp0.json
  traces/*.trace.json.gz
  b2-c2/c2.log
```
