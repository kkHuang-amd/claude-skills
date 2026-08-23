# SGLang Kimi-K3 old versus current AITER C2 A/B — 2026-08-21

## Decision

Do not pin the SGLang production profile to current AITER main yet.

After porting the K3-local MoE plumbing and BF16 front tuning, current AITER
main is functionally compatible and close in performance, but misses the
existing 0.5% promotion gate:

```text
five-run median throughput: old 1146.00 vs current 1136.17 tok/s (-0.86%)
five-run median TTFT:        old  844.71 vs current  871.71 ms    (+3.20%)
five-run median TPOT:        old   14.86 vs current   14.99 ms    (+0.87%)
```

All 160 measured requests completed. The remaining gap is reproducible and
small enough for focused attribution, but the TTFT regression and throughput
gate miss prevent promotion.

## Revisions and runtime

```text
SGLang:      455b744aa77b2078de7577619dc12d2775fc1091
old AITER:   b56d27beee1ce0132ca868da24d1a25e88e6ef42
current:     dc4bdf1c142181ad90b7f6948564126df4c05fde
GPU:         8x AMD Instinct MI355X / gfx950
old FlyDSL:  0.3.0 isolated overlay
new FlyDSL:  0.3.1 global runtime
```

The old side used `/sgl-workspace/flydsl-0.3.0-overlay`; the global FlyDSL
installation remained 0.3.1. SGLang and both AITER trees were used directly
through `PYTHONPATH`.

## Migrated K3-local functionality

Current AITER main was missing four local commits from the K3 integration
branch. Their functionality was manually adapted without creating commits:

```text
94b6a3d8  fused_moe caller-provided output buffer
8d681f08  FlyDSL v2 stage1 scratch reuse
284a1eb4  default-off AITER_FLYDSL_STAGE1_SCRATCH_REUSE gate
b56d27be  27 gfx950 BF16 front rows for N4480/N6016
```

Modified current-AITER files:

```text
aiter/fused_moe.py
aiter/ops/flydsl/moe_kernels.py
aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv
op_tests/test_fused_moe_output_buffer.py
op_tests/test_flydsl_stage1_out_cache.py
```

The newer mainline A8W4/A4W4 MoE retunes were retained. Old K3 fmoe CSVs and
the old OPUS adapter stack were not copied wholesale.

## Validation

```text
AITER output-buffer contract:       1 passed
AITER stage1 scratch reuse:         2 passed
SGLang AITER/MXFP4/KDA focused:     14 passed, 6 subtests passed
SGLang vendored K3 FlyDSL suite:    50 passed
IDE diagnostics:                    no errors
```

Current AITER correctly exposes `fused_moe(..., output=...)`; SGLang therefore
keeps the zero-copy MoE output path. Scratch reuse remains opt-in.

## Matched workload

Both sides used the same validated SGLang profile:

```text
input/output:       fixed 8192/1024
concurrency:        2
measured requests:  16 per round
warmup requests:    4 per round
rounds:             5 per side
seed:               42
request rate:       infinite
radix cache:        disabled
TP:                 8
KV cache:           FP8 E4M3
prefill backend:    AITER
decode backend:     tuned Triton
```

## Per-round results

Each row is total-token throughput, median TTFT, and median TPOT:

```text
round  old AITER                        current AITER
1      1141.78 tok/s  845.02  14.87    1135.36 tok/s  871.82  14.99
2      1147.53 tok/s  844.71  14.86    1134.44 tok/s  872.49  14.99
3      1145.32 tok/s  845.16  14.86    1136.31 tok/s  871.71  14.99
4      1146.00 tok/s  842.83  14.86    1136.17 tok/s  871.19  14.99
5      1147.04 tok/s  842.74  14.86    1136.50 tok/s  870.06  14.99
```

Latency values are milliseconds.

## Invalid first current-AITER run

The initial current-AITER run measured only `1037.37 tok/s` (`-9.48%`).
That result is invalid for AITER comparison: the manual CSV migration
temporarily left a second header inside `kimik3_bf16_tuned_gemm.csv`. Pandas
then inferred all lookup-key columns as strings, so BF16 tuned-GEMM lookup
missed and fell back to torch.

The duplicate header was removed and the merged config regenerated. Probes
then confirmed:

```text
M256 N7168 K1536 -> FlyDSL tuned kernel, 12.4777 us recorded row
M2   N6016 K7168 -> FlyDSL tuned kernel, 16.7031 us recorded row
```

The corrected server log contains no torch fallback for the target
M256/N7168/K1536 shape. The invalid artifacts remain preserved under `new/`
for diagnosis; production conclusions use `new-fixed/`.

## Artifacts

```text
/workspace/kimi-k3-runs/sglang-current-aiter-ab-2026-08-21/
  old/round{1..5}.{jsonl,log}
  new/                         # invalid duplicate-header diagnostic
  new-fixed/round{1..5}.{jsonl,log}
  old-run-overlay.log
  new-run.log
  new-fixed-run.log
  run_variant.sh
```

No server or benchmark process remains. All eight GPUs returned to baseline
VRAM usage. No commit or push was created.

## Next attribution

Keep current AITER as an experimental integration target. Before another full
endpoint sweep:

1. capture a compact C2 kernel summary for old and current AITER;
2. compare MoE stage1/stage2, BF16 projection, sort/quant, and collective time;
3. isolate the approximately 27 ms TTFT increase separately from decode TPOT;
4. rerun five-round C2 after any dispatch/config fix;
5. promote only if throughput is within 0.5% and latency does not regress.
