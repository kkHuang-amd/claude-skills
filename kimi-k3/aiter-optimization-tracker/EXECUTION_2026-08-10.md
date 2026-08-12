# Kimi-K3 optimization execution handoff — 2026-08-10

## CONTINUE HERE

Execution is complete. Final commits, benchmarks, memory results, and deferred
work are recorded in
[`OPTIMIZATION_RESULTS_2026-08-10.md`](OPTIMIZATION_RESULTS_2026-08-10.md).

## Completed

### Phase 1 — decode trace

Passed. See [`TRACE_2026-08-10.md`](TRACE_2026-08-10.md).

Evidence:

- `kimi_k3_kda_decode_fb_bf16_gfx950`: 138 launches on TP0.
- No `[M,3584]` routed-output `aten::copy_`.
- No `[M,3584]` routed-input `aten::contiguous`.
- No `[896]` router-bias `aten::_to_copy`.
- GSM8K 50: `1.000`.

### Phase 2 — AITER baseline commit

Completed:

```text
AITER  2ed06dd11  perf(kimi-k3): integrate fused KDA and MoE output reuse
```

Tests:

```text
AITER focused: 14 passed
SGLang focused: 11 passed
```

Current SGLang commits:

```text
6ac539cac  perf(kimi-k3): integrate AITER MoE optimizations
8950e2e0d  perf(kimi-k3): fuse ROCm KDA decode boundary
```

### Phase 3 — M=16384 tuning

Tuning finished for all seven shapes.

Stable production-op validation:

```text
7/7 checkAllclose passed
```

Selected hipBLASLt gain versus Torch fallback:

| N | K | Kernel gain |
|---:|---:|---:|
| 2304 | 1536 | 4.01% |
| 3072 | 512 | 12.15% |
| 6144 | 7168 | 1.94% |
| 7168 | 1536 | 3.02% |
| 7168 | 3584 | 1.05% |
| 7168 | 4224 | 2.00% |
| 8448 | 7168 | 16.23% |

Artifacts:

```text
m16384-hipblas.csv
m16384-hipblas-profile.csv
m16384-run-config.log
```

The selected kernels themselves passed production-op execution. During the
exhaustive search, one rejected hipBLASLt candidate per shape triggered an
illegal instruction and 180-second timeout. This eventually left GPU2 in a
poisoned KFD state.

## Current blocker

After tuning, GPU2 repeatedly aborted during CUDA-graph capture with:

```text
HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION
```

This persisted even after:

- the unknown KFD PID disappeared;
- VRAM returned to idle;
- the new M=16384 rows were removed;
- the original baseline server was retried.

Per-GPU reset and local-data cleanup are unsupported on this system. The user
selected a node reboot rather than a full AMD driver reload.

The seven M=16384 rows were removed from the source tree before reboot, so both
repositories are clean except the pre-existing untracked
`aiter/jit/flydsl_cache/`.

## Remaining plan

- Finish Phase 3 endpoint validation and commit tuned rows if stable.
- Phase 4 matched production A/B:
  Triton prefill + AITER MLA decode, 64 warmups, C2/4/8/16/32.
- Phase 5:
  #4647, optional #4487, then #4497/#4499/#4503/#4504 one at a time.
