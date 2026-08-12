# Kimi-K3 integrated PR-stack validation — 2026-08-10

## Outcome

- GSM8K 200, 5-shot: **0.990**
- Fixed 8192/1024 sweep: **all 496 measured requests succeeded**
- Full-stack throughput: **954.11 / 1703.90 / 2766.01 / 4151.73 /
  5630.64 tok/s** at concurrency 2/4/8/16/32
- Versus the published SGLang #33838 result: **+4.63% to +9.43% total
  throughput**, with median TPOT improved **5.16% to 9.66%**
- Server was stopped after validation.

Artifacts:

```text
/dockerx/var/amdsgl/kk/workspace/claude-skills/kimi-k3/
  stage2-runs/2026-08-10-pr-stack-validation/
```

## Integrated code

SGLang branch `integration/kimi-k3-pr-stack`:

- sglang#33838
- sglang#34198
- sglang#33916

AITER branch `integration/kimi-k3-pr-stack`:

- Mainline aiter#4463 and aiter#4534
- Staged aiter#4495 and aiter#4617

Runtime:

```text
8× AMD Instinct MI355X / gfx950
Torch 2.9.1+rocm7.2.0
Triton 3.6.0+git42270451
SGLang 0.0.0.dev16334+g8950e2e0d
sglang-kernel 0.4.6.post1, locally built for gfx950
```

AITER was rebuilt from the integration branch using the existing Torch and
Triton, with `--no-deps`.

## Server settings

```text
TP=8
DCP=1
attention backend=triton
dtype=BF16
KV dtype=auto/BF16
mem fraction=0.85
decode CUDA Graph max batch=256
Radix Cache=disabled
```

Environment:

```bash
SGLANG_USE_AITER=1
SGLANG_AITER_K3_OPT=1
AITER_FLYDSL_FORCE=1
AITER_SITUV2_A8W4=1
AITER_SITUV2_A4W4=0
SGLANG_K3_FLYDSL_AR_NORM=1
SGLANG_K3_KDA_FUSED_BACKEND=aiter
```

The fused-KDA adapter reported `enabled=True` and `available=True` on gfx950.
The server process inherited the backend setting, and no fused-KDA rejection
diagnostic appeared. The general dispatcher still logs `TritonKDAKernel`
because that is the fail-closed fallback used outside the covered incremental
boundary.

## Accuracy

GSM8K command profile:

```text
examples=200
threads=64
shots=5
max_tokens=512
temperature=0
```

Result:

```text
score=0.990
latency=31.070 s
output throughput=646.614 tok/s
```

This matches the prior accepted K3 range (0.98–0.99) and exceeds the 0.95 gate.

## Fixed 8192/1024 performance

Methodology matches SGLang #33838:

- random dataset, range ratio 1.0;
- output forced to 1024;
- `num_prompts = 8 × concurrency`;
- `warmups = 2 × concurrency`;
- concurrency 2/4/8/16/32.

| C | Requests | Total tok/s | Output tok/s | TTFT p50 ms | TPOT p50 ms | ITL p50 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 16/16 | 954.11 | 106.01 | 951.35 | 17.89 | 17.89 |
| 4 | 32/32 | 1703.90 | 189.32 | 1601.49 | 19.58 | 19.19 |
| 8 | 64/64 | 2766.01 | 307.33 | 2598.97 | 23.46 | 22.04 |
| 16 | 128/128 | 4151.73 | 461.30 | 4743.20 | 30.13 | 26.66 |
| 32 | 256/256 | 5630.64 | 625.63 | 8951.84 | 42.50 | 35.00 |

## Comparison with SGLang #33838

The #33838 PR body reports the same 8192/1024 request-count and warmup
methodology.

| C | #33838 tok/s | Full stack tok/s | Throughput delta | TPOT delta | ITL delta |
|---:|---:|---:|---:|---:|---:|
| 2 | 882.44 | 954.11 | +8.12% | -8.63% | -7.74% |
| 4 | 1594.32 | 1703.90 | +6.87% | -6.94% | -7.16% |
| 8 | 2527.54 | 2766.01 | +9.43% | -9.66% | -10.15% |
| 16 | 3893.59 | 4151.73 | +6.63% | -6.98% | -7.69% |
| 32 | 5381.41 | 5630.64 | +4.63% | -5.16% | -5.89% |

TTFT is effectively matched at C4/C16/C32, +1.81% at C8, and +11.54% at C2.
The C2 result has only 16 measured requests and is the highest-variance point.

## Fused-KDA A/B within this validation

The first launch accidentally omitted
`SGLANG_K3_KDA_FUSED_BACKEND=aiter`. Its results are retained as a same-code,
same-server-flag MoE-only control:

| C | MoE-only tok/s | Full stack tok/s | Fused-KDA stack delta |
|---:|---:|---:|---:|
| 2 | 923.12 | 954.11 | +3.36% |
| 4 | 1644.32 | 1703.90 | +3.62% |
| 8 | 2618.63 | 2766.01 | +5.63% |
| 16 | 3961.90 | 4151.73 | +4.79% |
| 32 | 5451.55 | 5630.64 | +3.29% |

This A/B uses separate server launches. It supports, but does not by itself
prove, isolated kernel attribution because startup/JIT/warm-state effects are
not perfectly paired.

## Comparison with the 2026-08-07 production AITER-decode baseline

The existing production-oriented baseline is:

```text
stage2-runs/2026-08-07-8k1k-attn-backend-ab/aiter-decode-bf16/
```

| C | Production baseline tok/s | PR-stack tok/s | Delta |
|---:|---:|---:|---:|
| 2 | 921.88 | 954.11 | +3.50% |
| 4 | 1676.44 | 1703.90 | +1.64% |
| 8 | 2781.64 | 2766.01 | -0.56% |
| 16 | 4343.66 | 4151.73 | -4.42% |
| 32 | 6087.04 | 5630.64 | -7.50% |

This is useful operational context but is **not** an apples-to-apples PR
comparison. The production baseline uses Triton prefill plus AITER MLA decode
and 64 fixed warmups at every concurrency. The PR-stack validation follows the
#33838 server/client profile: Triton attention and `2 × concurrency` warmups.
The attention-backend difference is especially material at C16/C32.

## Caveats

- AITER emitted untuned BF16 GEMM warnings for multiple `M=16384` prefill
  shapes and fell back to the torch solution. The results are valid for this
  stack but leave additional prefill tuning opportunity.
- The 2026-08-07 Triton-prefill/AITER-decode sweep used a different attention
  path and 64 warmups, so its higher C16/C32 numbers are not the primary
  apples-to-apples baseline.
- No profiler trace was captured in this run. Verifying removal of all three
  per-layer copy/cast kernels requires a separate trace.

## Decision

Accuracy and serving stability pass. The integrated stack improves the directly
comparable #33838 throughput/TPOT/ITL results at every tested concurrency.
Before upstreaming or committing locally, capture one decode trace to confirm
the expected copy/cast removal and fused-KDA kernel selection.
