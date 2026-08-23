# Kimi-K3 KDA whole-input projection integration — 2026-08-19

## Status

SGLang [#35176](https://github.com/sgl-project/sglang/pull/35176), commit
`b2315c53daa22ffa098ca7786bade19834c5e4aa`, was manually adapted into:

```text
/sgl-workspace/sglang-k3-triton37
tested base: e7391224dcd84df91afffb63301fc42930e9e866
```

The integration is uncommitted.

## Adaptation

The upstream ROCm layout is retained:

```text
[q,k,v,g | f_a | b | pad] = N6288
```

Below the configurable token threshold, one BF16 GEMM replaces the wide N6144
projection plus the separate `[f_a|b]` tiny GEMM. The split projections remain
views over the merged buffer, so larger token counts retain the tuned N6144
path.

Two existing K3 optimizations required explicit composition:

1. The AITER group64 path remains first in dispatch order. It continues to
   cover M1 and opt-in M2 before the new BF16 merged path.
2. When the existing fused KDA decode boundary requests `defer_f_b=True`, the
   new path returns `f_a` directly. It does not execute the upstream PR's
   standalone `f_b` GEMM, so the fused recurrence kernel still absorbs that
   projection.

Environment controls:

```text
SGLANG_ROCM_K3_FUSE_KDA_INPROJ=true
SGLANG_ROCM_K3_FUSE_KDA_INPROJ_MAX_TOKENS=256
```

## Files

```text
python/sglang/srt/models/kimi_k3.py
python/sglang/srt/environ.py
docs/docs/references/environment_variables.mdx
test/registered/kernels/ops/kimi_k3/test_kda_inproj_fusion.py
```

## Validation

```text
Targeted integration test:
  3 passed, 6 subtests passed

K3 focused suite plus integration test:
  53 passed, 6 subtests passed, 3 existing warnings

pre-commit on changed files:
  passed

IDE diagnostics:
  no errors
```

The tests pin the N6288 layout and aliases, compare split/fused numerical
results for M1/M4/M8/M33/M128/M256, and verify that deferred `f_b` returns
`f_a` without launching the tiny GEMM.

## Endpoint A/B

The same worktree was launched twice with only the fusion flag changed:

```text
OFF: SGLANG_ROCM_K3_FUSE_KDA_INPROJ=0
ON:  SGLANG_ROCM_K3_FUSE_KDA_INPROJ=1
```

Both runs used TP8, Triton 3.7, fixed 8192/1024 input/output, 64 warmups per
point, eight measured requests per concurrency unit, seed 42, disabled radix
cache, FP8 E4M3 KV, Triton decode, AITER prefill, and the established K3
best-profile flags.

```text
C     OFF tok/s   ON tok/s   throughput delta   OFF TPOT   ON TPOT   TPOT delta
2       1143.36    1144.07            +0.06%       14.88     14.88       +0.00%
4       1976.67    1976.68            +0.00%       16.81     16.81       +0.00%
8       3104.97    3137.16            +1.04%       20.92     20.69       -1.10%
16      4820.53    4910.12            +1.86%       25.96     25.37       -2.27%
32      6750.39    6910.32            +2.37%       35.40     34.49       -2.57%
64      8798.34    8849.86            +0.59%       51.50     51.11       -0.76%
```

Both sweeps completed `1008/1008` requests. The ON C64 point was rerun because
its gain was smaller than C8-C32:

```text
C64 ON rerun: 8849.86 tok/s, 51.11 ms median TPOT
```

The rerun is effectively identical to the first ON result (`8851.46 tok/s`,
`51.13 ms`), confirming that the C64 improvement is stable in this test.

Decision: retain the integration. C2/C4 remain neutral because the existing
small-batch paths take precedence, while C8-C64 improve throughput and TPOT.
Full-stack GSM8K remains the final correctness gate before treating the adapted
composition as production-validated.

Artifacts:

```text
/workspace/kimi-k3-runs/kda-inproj-ab-2026-08-19/
  integration.patch
  off/server.log
  off/c{2,4,8,16,32,64}.{log,jsonl}
  on/server.log
  on/c{2,4,8,16,32,64}.{log,jsonl}
  on/c64-rerun.{log,jsonl}
```
