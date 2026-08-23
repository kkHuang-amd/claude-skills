# Kimi-K3 latent down/up MXFP4 split — 2026-08-20

## Decision

Select latent-up-only MXFP4 as the preferred capacity-aware profile:

```text
SGLANG_K3_MOE_LATENT_MXFP4=0
SGLANG_K3_MOE_LATENT_DOWN_MXFP4=0
SGLANG_K3_MOE_LATENT_UP_MXFP4=1
SGLANG_K3_MOE_LATENT_MXFP4_MIN_TOKENS=2048
```

The original master flag remains backwards compatible and enables both down
and up. Up-only preserves the full BF16 N6016 front, avoids the additional
large-M front split, and performs better than down-only at C16/C64.

## Implementation

```text
SGLang base: 455b744aa77b2078de7577619dc12d2775fc1091

modified:
  python/sglang/srt/models/kimi_k3.py
  python/sglang/srt/environ.py
  docs/docs/references/environment_variables.mdx
  test/registered/kernels/ops/kimi_k3/test_latent_mxfp4_aiter_hip.py
```

Only enabled projections are packed after weight loading. Down and up have
independent runtime checks. The existing `gemm_ag_up_proj` remains eligible
when only down is enabled and is disabled only when MXFP4 up is selected.

## Validation

```text
focused K3 tests: 68 passed
additional subtests: 6 passed
pre-commit: passed
IDE diagnostics: no errors
```

## Capacity

```text
baseline:  1,519,705 tokens
down-only: 1,376,952 tokens
up-only:   1,376,952 tokens
both:      1,252,555 tokens
```

One side costs `142,753` tokens (`-9.39%`) and recovers `124,397` tokens
relative to the both-sides profile. Each packed projection is `13.02 MiB` per
K3 MoE layer per rank.

## Fast screening

C2/C16/C64 used 32 warmups and four measured requests per concurrency unit:

```text
case        C2       C16      C64 tok/s
down-only  1096.63  4929.75  9013.35
up-only    1096.94  4951.82  9056.05

up vs down:
  C16 throughput +0.45%, TTFT -1.61%, TPOT -0.51%
  C64 throughput +0.47%, TTFT -1.40%, TPOT -0.26%
```

## Full up-only endpoint

Fixed 8192/1024, TP8, 64 warmups, eight measured requests per concurrency,
seed 42, radix cache disabled:

```text
C     tok/s     throughput delta   TTFT ms   TTFT delta   TPOT ms   TPOT delta
2     1120.13*        -1.05%         843.42      -3.15%      14.92       -0.10%
4     1984.06         +0.23%        1463.56      -2.44%      16.79       +0.00%
8     3150.95         +0.53%        2288.00      -3.45%      20.65       -0.24%
16    4961.12         +0.82%        3957.52      -3.08%      25.32       -0.16%
32    6973.93         +1.28%        7474.61      -3.62%      34.28       -0.64%
64    9041.51         +2.21%       13978.92      -3.13%      50.14       -1.97%
```

`*` C2 uses the mean of two up-only rounds and two same-day baseline rounds.
Makespan throughput is noisy (`~2%` spread within baseline), while median E2E
is neutral (`-0.06%`), TTFT improves `3.15%`, and TPOT is neutral. No decode
regression is demonstrated.

Relative to the both-sides result, up-only retains about `60%` of the C32
throughput gain and `78%` of the C64 gain while recovering almost half of the
lost token capacity.

## Accuracy

```text
GSM8K 50:    1.000, invalid 0.000
GSM8K 1319:  0.950, invalid 0.001
policy gate: >0.94, passed
```

## Artifacts

```text
/workspace/kimi-k3-runs/moe-front-split-2026-08-20/
  split-flags.patch
  run_screen.sh
  down-only/
  up-only/
  run_up_final.sh
  up-only-final/
  base-c2-paired/
```

Patch SHA-256:

```text
10fc31e3b0343f518fc9f00aced4db7d59274cd48a987bb3b3aeae1a903718f6
```

The server and benchmark processes were stopped after validation. GPU VRAM
returned to the pre-run level on all eight devices.

## Next

The MXFP4 latent-up `shared_output + prefix_sum` epilogue was evaluated and
rejected: existing AITER Triton is 11-30% slower than the production ASM path,
and existing FlyDSL GEMM2 remains about 2x slower at M8192 even after removing
the measured add3 cost. Details:
[`MXFP4_TAIL_EPILOGUE_REJECTED_2026-08-20.md`](MXFP4_TAIL_EPILOGUE_REJECTED_2026-08-20.md).

Next evaluate an all-reduce + RMSNorm epilogue that emits MXFP4 activation
values and E8M0 scales directly while retaining the current fast ASM latent-up
GEMM.
