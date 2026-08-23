# Kimi-K3 environment rebuild — 2026-08-19

## Result

The Kimi-K3 development stack was rebuilt under `/sgl-workspace` without
modifying the existing golden repositories:

```text
/sgl-workspace/sglang-k3-opts-0812
/sgl-workspace/aiter-mainline-k3-0812
```

The offline SGLang bundle could not be traversed because it was missing a
parent object, so the repositories were cloned from their recorded remotes.

## Repository heads

```text
SGLang:
  remote: https://github.com/HaiShaw/sglang.git
  branch: perf/k3_opts_0812
  HEAD: dc6e5a2cd86f6ba049e5860bf8c809ec0dc7525d

AITER:
  remote: https://github.com/kkHuang-amd/aiter.git
  branch: integration/k3-core-only
  HEAD: 284a1eb401bb15f6368a68b34eb0cd693ee1fcd3
```

AITER submodule initialization completed successfully.

## Runtime

```text
Torch:          2.9.1+rocm7.2.0.lw.git7e1940d4
HIP:            7.2.26015-fc0010cf6a
Triton:         3.6.0+git42270451
FlyDSL:         0.3.0
sglang-kernel:  0.4.6.post1
GPU:            8x AMD Instinct MI355X
```

SGLang and AITER were installed editable with dependency resolution disabled.
`AITER_USE_SYSTEM_TRITON=1` was used for the AITER installation.

## Focused validation

Command:

```bash
PYTHONPATH=/sgl-workspace/sglang-k3-opts-0812/python:/sgl-workspace/aiter-mainline-k3-0812 \
python -m pytest -q \
  /sgl-workspace/sglang-k3-opts-0812/test/registered/kernels/ops/kimi_k3/flydsl_ops
```

Result:

```text
50 passed, 3 warnings, 30.57 seconds
```

The warnings were an existing unknown `asyncio_mode` pytest option and
Cython deprecation warnings. No server, benchmark, or pytest process remained
running after validation.

## Active baseline flags

```text
SGLANG_K3_FLYDSL_SOURCE=sglang
SGLANG_K3_AITER_M16384_PROFILE=1
SGLANG_USE_AITER=1
SGLANG_AITER_K3_OPT=1
AITER_FLYDSL_FORCE=1
AITER_SITUV2_A8W4=1
AITER_SITUV2_A4W4=0
AITER_FLYDSL_STAGE1_SCRATCH_REUSE=1
SGLANG_K3_FLYDSL_AR_NORM=1
SGLANG_K3_KDA_FUSED_BACKEND=aiter
SGLANG_K3_AITER_MLA_GATE=1
SGLANG_K3_AITER_KDA_GROUP64=1
SGLANG_K3_AITER_MOE_PREROUTE_FP8=0
SGLANG_K3_AITER_LATENT_TAIL_FP8=0
SGLANG_K3_AITER_B2_FUSIONS=0
```

