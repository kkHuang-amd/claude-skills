# Kimi-K3 handover — 2026-08-12

## Resume point

The SGLang-owned Kimi FlyDSL migration is complete and validated.

```text
SGLang:
  remote: https://github.com/HaiShaw/sglang.git
  branch: perf/k3_opts_0812
  HEAD:   f9dd3a0661b472d5fba1632adebcffc5c7c4021e

AITER core-only:
  remote: https://github.com/kkHuang-amd/aiter.git
  branch: integration/k3-core-only
  HEAD:   284a1eb401bb15f6368a68b34eb0cd693ee1fcd3
```

The core-only AITER branch contains only:

```text
#4617 caller-provided fused_moe output
#4647 stage1 scratch reuse
opt-in scratch reuse gate
```

Kimi-specific KDA/MLA/group64/preroute/latent/B2 kernels are maintained in:

```text
python/sglang/kernels/ops/kimi_k3/flydsl/
```

## Clone on the new machine

Preferred, from remotes:

```bash
git clone --branch perf/k3_opts_0812 \
  https://github.com/HaiShaw/sglang.git sglang-k3-opts-0812

git clone --branch integration/k3-core-only \
  https://github.com/kkHuang-amd/aiter.git aiter-mainline-k3-0812

git -C aiter-mainline-k3-0812 submodule update --init --recursive
```

Offline bundle fallback:

```bash
git clone sglang-k3-opts-0812.bundle sglang-k3-opts-0812
git clone --branch integration/k3-core-only \
  aiter-k3-integration.bundle aiter-mainline-k3-0812
```

The AITER bundle also contains:

```text
main                    full local AITER Kimi stack
profile/k3-a4w4-c16     optional #4603 profile
```

## Runtime

```text
Torch:  2.9.1+rocm7.2.0.git7e1940d4
HIP:    7.2.26015-fc0010cf6a
Triton: 3.6.0
```

Model:

```text
/shared_nfs/huggingface_models/moonshotai/Kimi-K3
```

## Selected production environment

```bash
export PYTHONPATH=$PWD/sglang-k3-opts-0812/python:$PWD/aiter-mainline-k3-0812
export AITER_JIT_DIR=/tmp/aiter-jit-kimi-k3

export SGLANG_K3_FLYDSL_SOURCE=sglang
export SGLANG_K3_AITER_M16384_PROFILE=1
export SGLANG_USE_AITER=1
export SGLANG_AITER_K3_OPT=1
export AITER_FLYDSL_FORCE=1
export AITER_SITUV2_A8W4=1
export AITER_SITUV2_A4W4=0
export AITER_FLYDSL_STAGE1_SCRATCH_REUSE=1
export SGLANG_K3_FLYDSL_AR_NORM=1
export SGLANG_K3_KDA_FUSED_BACKEND=aiter
export SGLANG_K3_AITER_MLA_GATE=1
export SGLANG_K3_AITER_KDA_GROUP64=1

export SGLANG_K3_AITER_MOE_PREROUTE_FP8=0
export SGLANG_K3_AITER_LATENT_TAIL_FP8=0
export SGLANG_K3_AITER_B2_FUSIONS=0
```

Optional B2 profile:

```bash
export SGLANG_K3_AITER_MOE_PREROUTE_FP8=1
export SGLANG_K3_AITER_B2_FUSIONS=1
```

Source fallback:

```bash
SGLANG_K3_FLYDSL_SOURCE=auto
SGLANG_K3_FLYDSL_SOURCE=sglang
SGLANG_K3_FLYDSL_SOURCE=aiter
```

## Validation result

```text
Vendored focused tests: 46 passed
GSM8K 50:             1.000
GSM8K 200:            0.990
Capacity:             933883 tokens
```

Production endpoint:

```text
C    golden tok/s   vendored tok/s   delta
2       969.04          968.57       -0.05%
4      1746.24         1741.98       -0.24%
8      2885.97         2881.25       -0.16%
16     4437.72         4432.25       -0.12%
32     6202.46         6191.41       -0.18%
```

B2 profile:

```text
C2:      968.57 -> 1054.19 tok/s
C2 TPOT:  17.67 ->   16.16 ms
C4:     1741.98 -> 1743.45 tok/s
```

Artifacts:

```text
/workspace/claude-skills/kimi-k3/
stage2-runs/2026-08-12-fresh-integration/
```

## Pending work

### Integration / upstream

- Open/review the SGLang integration PR from `HaiShaw/perf/k3_opts_0812`.
- Track AITER #4617 and #4647. Once merged, rebase the core-only branch onto
  AITER main and drop the corresponding local commits.
- Keep `SGLANG_K3_FLYDSL_SOURCE=auto` compatibility until upstream AITER
  versions are deployed; then delete duplicated SGLang kernels one family at a
  time.
- Decide whether the B2 profile becomes a deployment default. Before enabling
  globally, run five-round paired C2 and record graph/weight capacity.
- Keep the A4W4 branch C16-only; do not use it as the general profile.

### Remaining performance gap to B300

- Fixed small-kernel chains, copies/materialization, attention residual and KDA
  boundaries remain the main serial gaps.
- Route/sort/quant remains unresolved. If work resumes, use either:
  1. one-CTA LDS E896 sorter; or
  2. stage1 ABI consuming route metadata/token-major scale directly.
- Analyze retained B300 normal versus single-stream/no-PDL traces:
  busy union, summed duration, lost overlap categories and PDL contribution.

### Do not repeat

- V3/V3-R role-grid + P23.
- V4 multi-CU persistent route prep.
- Generic TILE_M M4/M8/M16 extension; M4 saved only 2.77 us/layer and larger
  buckets regressed.
- Forced all-reduce threshold/global exact-B32 experiments.
- Standalone #4572/#4577 endpoint paths.

## Documentation

Primary tracker:

```text
/workspace/claude-skills/kimi-k3/
aiter-optimization-tracker/
```

Read in this order:

```text
SGLANG_VENDOR_FLYDSL_2026-08-12.md
FRESH_INTEGRATION_2026-08-12.md
AITER_DEPENDENCY_MATRIX_2026-08-12.md
HANDOFF_2026-08-11.md
PAUSE_TRACK_2026-08-11.md
```

## Excluded local files

The following untracked files are not part of the code or bundles:

```text
git_rebase.sh
xxx
aiter/jit/flydsl_cache/
```

No server or benchmark process should remain running after handover.
