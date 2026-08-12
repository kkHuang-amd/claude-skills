# Kimi-K3 optimization priorities — 2026-08-10

Constraint: keep the current runtime unchanged:

```text
Torch 2.9.1+rocm7.2.0
Triton 3.6.0+git42270451
8× MI355X / gfx950
```

Anything requiring a Torch/Triton replacement is deferred.

## P0 — close the evidence and baseline gaps first

### 1. Capture a decode trace for the committed SGLang stack

Why first:

- GSM8K and serving performance passed, but the current evidence only proves
  that fused KDA was enabled/available and did not reject.
- The trace must confirm:
  - AITER fused KDA + `f_b` projection is selected;
  - routed-output D2D copy is gone;
  - `routed_input.contiguous()` is gone;
  - per-layer router-bias cast is gone.

This needs no package-version change. It is the acceptance gate before
committing AITER #4495/#4617.

### 2. Commit the already validated AITER #4495/#4617 stack

Current state:

- SGLang #34198 is commit `8950e2e0d`.
- SGLang #33838/#33916 is commit `0dae20b83`.
- AITER #4495/#4617 is still staged.

After the trace confirms the expected kernels, commit AITER separately. This
turns the validated environment into a reproducible two-repository baseline.

### 3. Eliminate untuned `M=16384` BF16 prefill GEMM fallbacks

The validated server emitted torch-fallback warnings for these K3 prefill
shapes:

```text
M=16384, N=6144, K=7168
M=16384, N=7168, K=1536
M=16384, N=8448, K=7168
M=16384, N=7168, K=4224
M=16384, N=7168, K=3584
M=16384, N=2304, K=1536
M=16384, N=3072, K=512
```

Action:

1. tune these shapes with the current Torch/Triton;
2. add them to the K3 BF16 tuned config;
3. rerun TTFT and total throughput at C8/C16/C32.

This is the clearest current bottleneck because it is observed locally rather
than inferred from a PR claim. AITER #4479 is useful reference material, but its
posted tuning focuses on other prefill M values, so the exact `M=16384` rows
still need measurement.

### 4. Re-run the production backend profile with the integrated stack

The PR-comparable profile improved #33838 by 4.63–9.43%, but the existing
production profile remains faster at high concurrency:

```text
C16: integrated Triton-attention profile 4151.73 vs production 4343.66 tok/s
C32: integrated Triton-attention profile 5630.64 vs production 6087.04 tok/s
```

Run the integrated stack with matched production settings:

- Triton prefill;
- AITER MLA decode;
- fixed 64 warmups;
- otherwise identical 8192/1024 requests.

This separates PR-stack gains from attention-backend effects without changing
Torch or Triton.

## P1 — high-value same-environment optimizations

### 5. AITER #4647 — reusable MoE stage-1 scratch

Expected benefit: about 6.7 GiB/GPU less graph-capture memory.

Priority rationale:

- directly relevant to K3's 652 graph-captured allocations;
- no Torch/Triton change is indicated;
- larger memory headroom can increase request/KV capacity and reduce startup
  pressure.

Validation:

- graph capture at batch 256;
- pointer stability across layers and replay;
- GSM8K smoke;
- memory before/after;
- C16/C32 throughput.

### 6. AITER #4487 — SiTUv2 MoE `block_m=64`

Expected benefit: reported +18% DSpark output throughput for verify-step token
buckets.

Do this early only if DSpark is part of the target deployment. It does not
require a Torch/Triton change, but its gain is workload-specific and should be
measured under the actual K3 DSpark graph workload.

### 7. Complete the batch-1 fusion family around the current KDA work

Suggested order:

1. #4497 — MLA output gate, reported 2.35× kernel;
2. #4499 — KDA group64 projection, reported 1.63×;
3. #4503 — FP8 latent-MoE tail, reported 1.93×;
4. #4504 — FP8 pre-route projections, reported 1.28–1.90×;
5. #4496/#4498 — BF16 fallback equivalents.

Why after tracing/tuning:

- #4495 is already integrated and establishes the pattern;
- these kernels can reduce batch-1 launch and materialization overhead;
- framework wiring is required, so isolated kernel speedups are insufficient.

Accept only improvements that survive the same 8192/1024 C2/C4 endpoint test.

## P2 — useful, but workload-dependent or lower confidence

### 8. AITER #4603 — A4W4 MoE retuning

Reported c16 gain and major graph-memory reduction are attractive, but the
validated performance default is A8W4 and A4W4 was slower in #33838. Keep this
behind the A8W4 path unless a memory-constrained deployment specifically needs
A4W4.

### 9. AITER #4510 — mixed-MoE stage-2 retune

Reported mean gain is 1.8%. Low implementation risk, but verify that the K3
tuned rows selected by the current A8W4 runtime actually change before spending
an end-to-end benchmark cycle.

### 10. #4494 / #4622 — graph-safe split-K

These are correctness-enabling changes for graph capture. Prioritize only if a
selected K3 GEMM uses split-K or a deadlock/replay issue is reproduced.

### 11. #4617 follow-up tests and fake-op coverage

The output-buffer smoke passed, but add a focused AITER test that exercises both
accepted and rejected caller buffers through the real `fused_moe` API,
including `torch.compile` fake-op alias behavior.

## Deferred — environment or hardware changes

### Torch/Triton-sensitive MLA stack

Defer:

- #4450 — 12-head MLA split scheduling;
- #4480 — FP8 KV small-head MLA;
- #4507 — page-table-derived split sizing;
- #4509 — split-major grid and blocked reduce.

Reason:

- prior isolated work required a Triton 3.7 runtime plus layout-API
  compatibility edits;
- current runtime is Triton 3.6;
- existing measurements also show a batch crossover and integration-sensitive
  FP8 behavior.

Revisit these together in an isolated environment rather than changing the
validated container.

### Non-gfx950 tracks

Defer on this machine:

- gfx942: #4471, #4582, #4645;
- gfx1250: #4482 follow-ups, #4537, #4607.

They may be valuable for their target hardware but cannot be validated on the
current MI355X/gfx950 system.

### Low-priority branch-only work

- shared-GEMM no-split-K branches: prior graph replay regressed 2.9–4.1%;
- `k3-for-amd`: stale integration branch;
- non-causal MLA `msk0` branches: defer until a concrete serving path requires
  them.

## Recommended execution order

```text
1. Decode trace
2. Commit AITER #4495/#4617
3. Tune missing M=16384 BF16 GEMMs
4. Matched production-backend 8k/1k rerun
5. #4647 graph workspace reuse
6. #4487 DSpark tuning, if DSpark is in scope
7. #4497/#4499/#4503/#4504 fusion stack
8. A4W4 and smaller retunes
9. Isolated Triton-3.7 MLA experiment later
```
