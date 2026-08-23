# Matched-route current-kernel microbenchmark — 2026-08-23

## Decision

- PASS: armed routes validated exactly (92 layers/source, preserved counts, no per-token duplicate experts).
- PASS: graph-replay full fused_moe, stage1, and stage2 completed for both sources and A8W4/A16W4 (1,104 OK rows; no skips).
- PASS: all outputs finite and input-sensitive; all 184 A8/A16 numerical comparisons passed.
- This is current-kernel micro evidence only; no endpoint, eager-timing, or promotion claim.

## Runtime

- AITER: `dc4bdf1c142181ad90b7f6948564126df4c05fde` at `/sgl-workspace/aiter-atom-current/aiter/__init__.py`
- SGLang: `455b744aa77b2078de7577619dc12d2775fc1091`
- Device/runtime: `AMD Instinct MI355X` / `gfx950` / torch `2.9.1+rocm7.2.0.git7e1940d4`
- Environment: `AITER_FLYDSL_FORCE=1 AITER_FLYDSL_STAGE1_SCRATCH_REUSE=1`
- Warmup/iterations: `10/100`; HIP graph replay; caller output enabled; weight prep excluded.

## 92-layer sum of layer p50

| Routes | Scope | A8W4 ms | A16W4 ms | A16 vs A8 |
|---|---:|---:|---:|---:|
| sglang | full | 8.171382 | 8.451743 | +3.431% |
| sglang | stage1 | 4.210011 | 5.001037 | +18.789% |
| sglang | stage2 | 2.807474 | 3.161900 | +12.624% |
| atom | full | 4.764780 | 5.119108 | +7.436% |
| atom | stage1 | 1.983877 | 3.088116 | +55.661% |
| atom | stage2 | 1.896093 | 1.819099 | -4.061% |

## Active-expert and BM32 regressions

| Routes | Mode | Scope | Active slope ms/expert | Active R² | BM32 slope ms/block | BM32 R² |
|---|---|---|---:|---:|---:|---:|
| sglang | a8w4 | full | 0.000310909 | 0.990983 | 0.000325080 | 0.993006 |
| sglang | a8w4 | stage1 | 0.000195365 | 0.983500 | 0.000204352 | 0.986297 |
| sglang | a8w4 | stage2 | 0.000100333 | 0.992461 | 0.000104938 | 0.995091 |
| sglang | a16w4 | full | 0.000367241 | 0.938655 | 0.000383318 | 0.937328 |
| sglang | a16w4 | stage1 | 0.000215855 | 0.835202 | 0.000225101 | 0.832517 |
| sglang | a16w4 | stage2 | 0.000136671 | 0.996493 | 0.000142768 | 0.996675 |
| atom | a8w4 | full | 0.000299571 | 0.999097 | 0.000318294 | 0.999097 |
| atom | a8w4 | stage1 | 0.000194178 | 0.999587 | 0.000206314 | 0.999587 |
| atom | a8w4 | stage2 | 0.000086341 | 0.997974 | 0.000091737 | 0.997974 |
| atom | a16w4 | full | 0.000345353 | 0.999513 | 0.000366937 | 0.999513 |
| atom | a16w4 | stage1 | 0.000212332 | 0.998743 | 0.000225603 | 0.998743 |
| atom | a16w4 | stage2 | 0.000127632 | 0.998678 | 0.000135609 | 0.998678 |

## Correctness

- sglang: worst relative L2 `0.129030`; minimum cosine `0.991684`.
- atom: worst relative L2 `0.124334`; minimum cosine `0.992272`.
- Layout/dispatch contracts: A8W4 `shuffle_weight_a16w4(gate_up=True)_GUGU`; A16W4 `shuffle_weight_a16w4(gate_up=False)_GGUU`.
- Full-chain finite/input-change gates: 368/368 pass. Stage hooks: 736/736 stage rows OK; no skips.

## Commands

```bash
python /workspace/useful-scripts/benchmarking/kimi-k3/micro/benchmark_route_modes.py --result-root /workspace/kimi-k3-runs/common-oai-sglang-atom-traces-2026-08-22/route-validation --output-dir /workspace/kimi-k3-runs/matched-route-current-kernel-2026-08-23/route-validation-preflight --validate-routes-only
PYTHONNOUSERSITE=1 PYTHONPATH=/sgl-workspace/aiter-atom-current:/sgl-workspace/sglang-k3-triton37/python AITER_FLYDSL_FORCE=1 AITER_FLYDSL_STAGE1_SCRATCH_REUSE=1 python /workspace/useful-scripts/benchmarking/kimi-k3/micro/benchmark_route_modes.py --result-root /workspace/kimi-k3-runs/common-oai-sglang-atom-traces-2026-08-22/route-validation --aiter-root /sgl-workspace/aiter-atom-current --sglang-root /sgl-workspace/sglang-k3-triton37 --output-dir /workspace/kimi-k3-runs/matched-route-current-kernel-2026-08-23/gpu-benchmark-current-root --warmup 10 --iterations 100
```

## Evidence

- `/workspace/kimi-k3-runs/matched-route-current-kernel-2026-08-23/`
- `gpu-benchmark-current-root/route-mode-results.json`
- `gpu-benchmark-current-root/route-mode-layers.csv`
- `gpu-benchmark-current-root/route-mode-report.md`
- `run.log`
- `checksums.sha256`
- Armed-source checksums: `/workspace/kimi-k3-runs/common-oai-sglang-atom-traces-2026-08-22/route-validation/checksums.sha256`

## Note

The first GPU attempt resolved an incompatible installed AITER tree and failed before timing. It is preserved as `run-import-mismatch.log`; the successful run pins the requested current/default AITER root explicitly.
