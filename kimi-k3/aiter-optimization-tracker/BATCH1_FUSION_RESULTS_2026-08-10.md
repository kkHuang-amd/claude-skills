# Kimi-K3 batch-1 fusion integration results — 2026-08-10

## Scope

Integrated AITER PRs:

- #4497 MLA output gate
- #4499 KDA group64 input projection
- #4504 FP8 MoE pre-route/shared-down
- #4503 FP8 latent MoE tail

Runtime remained:

```text
Torch 2.9.1+rocm7.2.0
Triton 3.6.0+git42270451
8× MI355X / gfx950
```

## Commits

SGLang:

```text
1feb02539  perf(kimi-k3): wire AITER MLA output gate
f765224c7  perf(kimi-k3): wire AITER KDA group64 projection
99a0ce693  perf(kimi-k3): wire AITER FP8 MoE pre-route
9e47efedb  perf(kimi-k3): wire AITER FP8 latent tail
```

AITER:

```text
d7d81c670  perf(kimi-k3): add fused MLA output gate
1405d220f  perf(kimi-k3): add group64 KDA input projection
4ab14dfa2  perf(kimi-k3): add FP8 MoE pre-route fusions
84e23ef4e  perf(kimi-k3): add FP8 latent MoE tail
```

All four SGLang paths are fail-closed, gfx950/B1-specific, and independently
controlled:

```text
SGLANG_K3_AITER_MLA_GATE
SGLANG_K3_AITER_KDA_GROUP64
SGLANG_K3_AITER_MOE_PREROUTE_FP8
SGLANG_K3_AITER_LATENT_TAIL_FP8
```

## Kernel correctness

The old PR kernels were ported to AITER's vendored `buffer_ops`/`vector`
helpers.

| PR | Focused tests |
|---:|---:|
| #4497 | 8 passed |
| #4499 | 11 passed |
| #4504 | 6 passed |
| #4503 | 7 passed |

## Per-feature endpoint gates

All measurements use production-matched Triton prefill + AITER decode, fixed 64
warmups, 8192/1024.

| Stack | GSM8K 50 | C2 tok/s | C4 tok/s | Decision |
|---|---:|---:|---:|---|
| Baseline before batch-1 fusions | 1.000 | 968.45 | 1737.07 | Reference |
| + #4497 | 1.000 | 969.22 | 1743.46 | Enable |
| + #4499 | 1.000 | 970.45 | 1746.95 | Enable |
| + #4504 | 1.000 | 970.10 | 1742.96 | Keep opt-in, disable in production |
| #4497 + #4499 + #4503 | 1.000 | 969.64 | 1739.45 | Keep opt-in, disable in production |
| All four flags | 1.000 | 970.54 | 1744.70 | Correct, but below selected C4 |

#4504 and #4503 are retained as independent opt-in implementations because
their correctness/graph gates pass, but they do not improve the current
production endpoint.

Fresh-JIT audit clarification (2026-08-11):

- explicit B1 traces observed 11,776 #4503 latent-tail launches;
- explicit B1 traces observed 11,776 #4504 tri-projection and 11,776
  shared-down launches;
- the fixed-length C2/C4 benchmark above remains at batch 2/4 and finishes
  requests together, so it does not enter the B1-only paths.

The implementations are not stale. Their disabled-by-default status remains
appropriate for the selected C2+ production workload and #4504's memory cost,
but a coverage-matched C1 A/B is still required to judge B1 performance.

## Selected final stack

Enabled:

```text
SGLANG_K3_AITER_MLA_GATE=1
SGLANG_K3_AITER_KDA_GROUP64=1
```

Disabled:

```text
SGLANG_K3_AITER_MOE_PREROUTE_FP8=0
SGLANG_K3_AITER_LATENT_TAIL_FP8=0
```

Accuracy:

```text
GSM8K 200 = 0.980
```

Production-matched final results:

| C | Pre-fusion tok/s | Selected final tok/s | Delta |
|---:|---:|---:|---:|
| 2 | 968.45 | 969.04 | +0.06% |
| 4 | 1737.07 | 1746.24 | +0.53% |
| 8 | 2882.96 | 2885.97 | +0.10% |
| 16 | 4439.41 | 4437.72 | -0.04% |
| 32 | 6203.83 | 6202.46 | -0.02% |

All 496 measured requests succeeded. C16/C32 changes are within run variance;
the useful endpoint gain is concentrated at C4.

## C64 baseline vs all flags

Matched workload:

```text
concurrency=64
warmups=64
measured requests=512
input/output=8192/1024
```

| Metric | Baseline | All four flags | Delta |
|---|---:|---:|---:|
| Total throughput | 7906.61 tok/s | 7902.34 tok/s | -0.05% |
| Output throughput | 878.51 tok/s | 878.04 tok/s | -0.05% |
| TTFT p50 | 17221.51 ms | 17251.35 ms | +0.17% |
| TPOT p50 | 56.29 ms | 56.34 ms | +0.09% |
| ITL p50 | 40.73 ms | 40.61 ms | -0.29% |

All 512 requests succeeded in both runs. The differences are within run
variance; enabling all four flags provides no C64 throughput benefit.

## AITER #4603 A4W4 retune evaluation

#4603's four feature commits were applied without the unrelated mainline merges.

Validation:

- 17 K3 A4W4 MoE token buckets dispatched correctly;
- 32 routed A4W4 GEMM rows passed production-op correctness;
- GSM8K 200: `0.985`;
- graph memory: `6.27 GB/GPU` versus A8W4's `3.46 GB/GPU`.

Endpoint comparison with the selected A8W4 stack:

| C | A8W4 tok/s | #4603 A4W4 tok/s | Delta |
|---:|---:|---:|---:|
| 2 | 969.04 | 971.92 | +0.30% |
| 4 | 1746.24 | 1725.39 | -1.19% |
| 8 | 2885.97 | 2856.67 | -1.02% |
| 16 | 4437.72 | 4504.10 | +1.50% |
| 32 | 6202.46 | 6161.06 | -0.67% |

#4603 fixes the historical C16 regression and may be useful as a C16-specific
profile, but it does not replace A8W4 as the general production default. The
AITER changes are staged locally and not committed pending a policy decision.

## Memory and graph behavior

- CUDA Graph capture passed with each individual flag and all flags together.
- Graph memory remained about `3.46 GB/GPU`, preserving #4647's memory saving.
- #4504 load-time FP8 packs raised weight memory from about `194.38` to
  `201.11 GB/GPU`; this is another reason not to enable it by default for the
  small endpoint change.

## Artifacts

```text
/workspace/claude-skills/kimi-k3/
  stage2-runs/2026-08-10-batch1-fusions/
```
