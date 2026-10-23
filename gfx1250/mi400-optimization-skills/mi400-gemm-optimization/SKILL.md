---
name: mi400-gemm-optimization
description: Optimize GEMM-like MI400/GFX12 kernels using measured coalescing, tiling, WMMA, staging and pipeline trade-offs.
---
# MI400 GEMM Optimization
## Trigger
GEMM/linear/fused matrix hotspot or profiling evidence showing matrix compute/staging on the critical path.
## Inputs
M/N/K, dtype, layouts/strides, tile geometry, waves, VGPR/LDS use, baseline profile.
## Workflow
1. Verify baseline correctness.
2. Fix/coarsely optimize global access coalescing first.
3. Sweep M/N tiling for reuse versus VGPR pressure.
4. Evaluate WMMA organization appropriate to MI400/GFX12.
5. Evaluate macro-tile cooperative staging versus LDS/synchronization cost.
6. Compare register staging, async copy and TDM where global-to-LDS movement is material.
7. Pipeline next-tile movement only after ordering is proven.
8. Check output-store cost, especially small-K cases.
9. Test cache hints/prefetch only from an explicit reuse hypothesis.
10. Benchmark across representative shapes; retain only measured wins.

## Required references
Read `decision-rules.md`, then the relevant technical references, then `anti-patterns.md` and `optimization-checklist.md`.

## Stop conditions
Stop and return to profiling if GEMM is not on the critical path, if the target shape distribution is unknown, or if correctness cannot be reproduced.
