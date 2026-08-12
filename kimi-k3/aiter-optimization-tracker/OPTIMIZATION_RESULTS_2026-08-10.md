# Kimi-K3 optimization execution results — 2026-08-10

## Final baseline

SGLang:

```text
40f2f5128  fix(kimi-k3): support 12-head AITER MLA decode
6ac539cac  perf(kimi-k3): integrate AITER MoE optimizations
8950e2e0d  perf(kimi-k3): fuse ROCm KDA decode boundary
```

AITER:

```text
9732c055d  perf(kimi-k3): reuse FlyDSL MoE stage1 scratch
280a17661  tune(kimi-k3): add gfx950 prefill BF16 configs
2ed06dd11  perf(kimi-k3): integrate fused KDA and MoE output reuse
```

Runtime remained unchanged:

```text
Torch 2.9.1+rocm7.2.0
Triton 3.6.0+git42270451
8× MI355X / gfx950
```

## Phase 1 — trace evidence

Passed:

- fused KDA + `f_b` kernel observed: 138 TP0 launches;
- no `[M,3584]` routed-output copy;
- no `[M,3584]` routed-input contiguous copy;
- no `[896]` per-layer router-bias cast;
- GSM8K 50: `1.000`.

Detailed evidence: [`TRACE_2026-08-10.md`](TRACE_2026-08-10.md).

## Phase 2 — AITER output-buffer baseline

Tests:

```text
AITER: 14 passed
SGLang: 11 passed
```

Real `fused_moe` accepted a matching caller buffer, rejected a mismatched
buffer, and the fake op reproduced the same alias contract.

## Phase 3 — M=16384 BF16 GEMM tuning

The original seven shapes plus one additional serving-discovered shape were
covered.

- Seven hipBLASLt rows: production-op correctness passed.
- One explicit Torch row for `M=16384,N=1536,K=7168`, where safe tuning found
  no faster stable backend.
- GSM8K 200: `0.980`.
- No remaining M=16384 fallback warning after all eight rows were present.

Endpoint comparison using the #33838 methodology:

| C | Before tok/s | Tuned tok/s | Delta |
|---:|---:|---:|---:|
| 8 | 2766.01 | 2784.91 | +0.68% |
| 16 | 4151.73 | 4182.22 | +0.73% |
| 32 | 5630.64 | 5670.92 | +0.72% |

C32 median TTFT improved from `8951.84` to `8742.21 ms` (`-2.34%`).

## Phase 4 — matched production A/B

Configuration:

- Triton prefill;
- AITER MLA decode;
- fixed 64 warmups;
- 8192/1024 random workload;
- C2/4/8/16/32.

Compatibility required:

- zero-pad Kimi-K3's 12 local MLA heads to AITER's 16-head geometry;
- keep NoPE Kimi-K3 off the fused RoPE path.

Results versus the 2026-08-07 production baseline:

| C | Old tok/s | New tok/s | Delta |
|---:|---:|---:|---:|
| 2 | 921.88 | 968.45 | +5.05% |
| 4 | 1676.44 | 1737.07 | +3.62% |
| 8 | 2781.64 | 2882.96 | +3.64% |
| 16 | 4343.66 | 4439.41 | +2.20% |
| 32 | 6087.04 | 6203.83 | +1.92% |

All 496 measured requests succeeded. GSM8K 50 was `1.000`.

## Phase 5 — P1 evaluation

### AITER #4647

Accepted and committed.

- focused tests: 38 passed;
- CUDA-graph memory: `9.16 → 3.47 GB/GPU`;
- saved: `5.69 GB/GPU`;
- GSM8K 50: `1.000`;
- C16: `4439.41 → 4441.57 tok/s`;
- C32: `6203.83 → 6218.27 tok/s`.

The memory saving introduced no throughput regression.

### AITER #4487

Skipped. The current deployment does not enable DSpark; the PR only retunes
verify-step token buckets. Revisit when speculative serving is in scope.

### Batch-1 fusion family

Evaluated in isolated worktrees and not merged because current SGLang has no
call-site wiring for these APIs.

Current FlyDSL required import ports to AITER's vendored `buffer_ops`/`vector`
helpers.

| PR | Kernel suite result | Decision |
|---:|---:|---|
| #4497 | 8 passed | Kernel ready; SGLang MLA-gate wiring required |
| #4499 | 11 passed | Kernel ready; KDA input-projection wiring required |
| #4503 | 7 passed | Kernel ready; latent-MoE-tail wiring required |
| #4504 | 6 passed | Kernel ready; pre-route projection wiring required |

No dead kernel code was added to the validated baseline. These should become
separate SGLang+AITER integration projects, one PR at a time, with C2/C4
endpoint gates.

## Final state

- All planned P0 work completed.
- #4647 completed and committed.
- Optional DSpark work intentionally skipped.
- Batch-1 fusion kernels are correctness-qualified but deferred pending
  framework wiring.
- No Torch or Triton version changed.
