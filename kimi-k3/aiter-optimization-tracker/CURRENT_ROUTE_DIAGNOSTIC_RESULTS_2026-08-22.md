# Current-route A8W4/A16W4 diagnostic results — 2026-08-22

## Decision

The hardened armed route capture is valid and explains the apparent C64
A16W4 stage1 advantage. ATOM's real common-client wave routes far more
concentrated work than SGLang:

```text
mean active experts: SGLang 133.87, ATOM 18.03
mean BM32 blocks:     SGLang 145.21, ATOM 33.91
```

Matched-route current-kernel replay confirms A8W4 remains faster than A16W4
for both route sets. The trace inversion is a route-workload effect, not an
intrinsic A16W4 kernel win.

The earlier unarmed capture is rejected and retained only under:

```text
/workspace/kimi-k3-runs/common-oai-sglang-atom-traces-2026-08-22/
  route-validation-unarmed-invalid/
```

## Result

The exact common C64 manifest route diagnostic completed sequentially for
SGLang A8W4 and ATOM A16W4. Both eager-only contract runs completed 64/64
requests and produced exactly 92 post-arm `[64,16]` calls on each of eight
ranks. All 1,472 dump files are later than their arm boundary, parse as JSON,
have `total_routes=1024`, and match the `[64,16]` shape. Cross-engine rank/call
alignment is exact: 736/736 pairs, with no left-only or right-only calls.

Evidence root:
`/workspace/kimi-k3-runs/common-oai-sglang-atom-traces-2026-08-22/route-validation/`

Comparison artifacts:
- `comparison/route-analysis.json`
- `comparison/route-layers.csv`
- `comparison/route-deltas.csv`
- `comparison/route-analysis.md`
- `checksums.sha256` (1,480 retained evidence files)

## Aggregate armed route contract

Ranks are identical for these summaries:

```text
                         SGLang        ATOM
active experts mean      133.870       18.033
active experts min/max    57 / 282      16 / 203
BM32 blocks mean         145.207       33.913
BM32 blocks min/max       68 / 288      32 / 208
routes/active mean         8.502       63.359
```

ATOM selects exactly 16 active experts in 91 of 92 captured MoE calls; one
call uses 203. SGLang spreads the same 1024 routes over many more experts.
No aligned expert-count vectors are identical (0/736).

## Matched-route current-kernel replay

HIP graph replay, current AITER `dc4bdf1c`, 10 warmups and 100 iterations:

```text
routes   scope    A8W4 ms   A16W4 ms   A16 vs A8
SGLang   full       8.171       8.452      +3.43%
SGLang   stage1     4.210       5.001     +18.79%
SGLang   stage2     2.807       3.162     +12.62%

ATOM     full       4.765       5.119      +7.44%
ATOM     stage1     1.984       3.088     +55.66%
ATOM     stage2     1.896       1.819      -4.06%
```

The full chain favors A8W4 under both exact route distributions. All 1,104
rows, finite/input-change checks, layouts and 184 numerical comparisons pass.
Details:
[`MATCHED_ROUTE_CURRENT_KERNEL_2026-08-23.md`](MATCHED_ROUTE_CURRENT_KERNEL_2026-08-23.md).

## Operational note

The contracts `sitecustomize.py` initially propagated into ROCm's Python
`rocm_agent_enumerator`, recursively importing AITER/FlyDSL during compiler
architecture detection. Two failed startup attempts are retained as
`sglang-stalled-preflight/` and `sglang-recursive-preflight/`. No requests or
GPU allocations occurred in the first attempt; the second was stopped before
model loading. The successful runs used architecture-only overrides
`FLYDSL_GPU_ARCH=gfx950 HCC_AMDGPU_TARGET=gfx950`; a direct instrumented import
probe passed before retry. Final processes, route JIT/cache directories, and
VRAM were cleaned. Baseline idle VRAM was 297,766,912 bytes per GPU.

Route dumps are eager contract artifacts only; no eager timing is reported.
Matched timing comes from the separate HIP graph microbenchmark.
