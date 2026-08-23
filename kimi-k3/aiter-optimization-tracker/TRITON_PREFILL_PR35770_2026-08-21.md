# SGLang PR #35770 Kimi-K3 Triton prefill integration

Date: 2026-08-21

## Decision

The feature commit `be31ec565` from SGLang PR #35770 was manually ported onto
`perf/k3_opts_0812@455b744aa` without replacing the branch's existing K3,
Triton-3.7, or MXFP4 work.

After validation, the implementation was removed from the current worktree on
2026-08-21 at the user's request. The pre-existing K3/MXFP4 work was preserved
using the recorded pre-port patch snapshot. The following was the evaluated
default-off interface:

```text
SGLANG_K3_TRITON_PREFILL_PROFILE=off|bf16|fp8
default: off
```

Do not promote Triton prefill to production. Isolated kernels pass their
performance and numerical gates, but TP8 endpoint performance and paired
long-context correctness fail.

## Integrated surface

- gfx950 split-dimension absorbed MLA kernel for exact
  `12Q/1KV, Lq576/Lv512`
- fail-closed dispatch for page size, LSE, sink, mask, causal/window/logit-cap,
  skip-stage, shape, layout, and cache-dtype constraints
- gfx950 two-stage scheduling and exp2 accumulation with natural-log LSE output
- native FP8 zero-prefix casts and scaled cached-prefix handling
- `off`: pre-port generic Triton behavior
- `bf16`: two-stage, exp2, and BF16 absorbed split
- `fp8`: BF16 tuning plus native FP8 paths
- backward-compatible `SGLANG_TRITON_FP8_PREFILL_ATTN`, default false

Upstream focused test result:

```text
4 passed
```

## gfx950 HIP-graph micro results

All timings include graph replay. Native zero-prefix FP8 includes Q/K/V casts.

```text
shape                         baseline        candidate       speedup   rel-L2
fresh BF16 8K                 0.742 ms        0.460 ms        1.611x    0.000030
fresh FP8 8K                  0.776 ms        0.426 ms        1.821x    0.036327
fresh BF16 16K                2.542 ms        1.509 ms        1.684x    0.000031
fresh FP8 16K                 2.701 ms        1.407 ms        1.919x    0.036843
absorbed BF16 8K+8K           9.072 ms        4.234 ms        2.142x    0.000044
absorbed FP8 8K+8K generic    3.996 ms        3.715 ms        1.076x    0.000034
```

The micro gates pass: fresh BF16 8K exceeds 1.5x, absorbed BF16 exceeds 1.7x,
fresh FP8 is within the 4% numerical allowance, and the FP8 split agrees with
the corrected generic FP8 path.

## TP8 endpoint A/B

Both sides used the current balanced K3 profile, FP8 KV cache, Triton decode,
chunk/max-prefill 16384, radix cache disabled, fixed seeds and prompts.
The production side used AITER prefill; the candidate used Triton prefill with
the `fp8` profile.

```text
8K/1K total-token throughput (tok/s)
C2     AITER 1138.33    Triton 1128.93    ratio 0.9917
C4     AITER 1986.81    Triton 1957.37    ratio 0.9852
C8     AITER 3147.31    Triton 3116.00    ratio 0.9901
C16    AITER 4957.12    Triton 4891.12    ratio 0.9867
geomean ratio: 0.9884

68K/350 total-token throughput (tok/s)
C2     AITER 10224.78   Triton 8148.79    ratio 0.7970
C4     AITER 12539.29   Triton 9350.73    ratio 0.7457
C8     AITER 14004.23   Triton 9746.35    ratio 0.6960
C16    AITER 15203.22   Triton 10393.59   ratio 0.6836
geomean ratio: 0.7292
```

All requests completed with no retractions or OOM. The 8K gate misses by
1.16%, while the 68K target fails decisively at -27.08% geomean. Triton FP8
also has 40-52% worse 68K TTFT and 12-42% worse 68K TPOT.

Single-point attribution at C2:

```text
profile         8K tok/s    68K tok/s
off               769.03      6653.38
bf16             1106.77      7840.93
fp8              1128.93      8148.79
AITER            1138.33     10224.78
```

The port substantially improves generic Triton, but does not close AITER's
long-context lead.

## Accuracy and long context

```text
                    GSM8K-50       GSM8K-1319      invalid
AITER production       1.000          0.958          0.001
Triton FP8             1.000          0.947          0.001
```

The candidate remains above 0.94 but trails the paired baseline by 1.1
percentage points, just outside the 1 pp gate.

For 32 paired 68K-token prompts with 16 deterministic output tokens:

```text
first-token top-1 match:                 81.25%
all generated-token match:              81.05%
first-token top-20 probability cosine:  mean 0.97965, min 0.95694
maximum retractions:                     0
```

The endpoint exposes top-20 probabilities rather than full logits, so the
cosine is a top-20 proxy. It is already far below the FP8 threshold and the
top-1 gate fails, making a full-logit rerun unnecessary for promotion.

## Artifacts

All raw logs, JSONL responses, server effective-argument logs, scripts, micro
results, endpoint summaries, and paired long-context responses are under:

```text
/workspace/kimi-k3-runs/triton-prefill-pr35770-2026-08-21/
```

No commit or push was created. `preexisting-sglang.patch` and
`final-sglang.patch` preserve the exact before/after worktree states.
