# Dense projection crossover microbenchmark — 2026-08-23

## Status

Implemented the reusable complete-chain benchmark at:

```text
/workspace/useful-scripts/benchmarking/kimi-k3/micro/benchmark_dense_crossover.py
```

No GPU benchmark, server, large tensor allocation, package installation, engine
source edit, commit, or push was performed.

The matrix covers M=2,4,8,16,32,64 for the five representative Kimi projection
shapes. SGLang-only `merged_front` and `kda_inproj` are explicitly marked as
non-like-for-like context. Timed graph chains include activation quantization;
weight quantization/shuffling remains setup work. Output records numerical
checks, dispatch configuration, prepared-weight bytes, and incremental
dual-storage cost. Promotion still requires matched common-client endpoint,
GSM8K, and long-context validation.

## Static validation

```text
python -m py_compile benchmark_dense_crossover.py test_benchmark_dense_crossover.py
python -m unittest -v test_benchmark_dense_crossover.py
python benchmark_dense_crossover.py --help
Cursor lints
```

Result after the replay-gate update: 13 CPU-only tests passed; compile/help and
IDE lint checks passed.
Standalone `ruff` was unavailable and was not installed.

## Runtime smoke required

Before collecting the matrix, smoke one case per mode on gfx950. Confirm:

1. `get_hip_quant(QuantType.per_Token)` produces the scale orientation expected
   by `gemm_a8w8_bpreshuffle` for online-quantized synthetic weights.
2. The current PTPC tuned CSV resolves a supported kernel for each requested
   N/K/M rather than falling into an unavailable CK fallback.
3. `rmsnorm_quant` and `gemm_a4w4` remain graph-capture safe when called through
   the current SGLang latent adapter at M=2 through M=64.
4. AITER dispatch metadata helpers match the actual runtime-selected fallback
   when a tuned row is absent.

## Input-change replay gate added

The benchmark schema is now `kimi-k3-dense-crossover-v2`. Before replay timing,
every captured case replays once with the original activation, clones output A,
negates and copies a deterministic alternate activation into the same static
tensor, replays again, and clones output B. The case fails if either output is
non-finite or if B is bitwise/equal to A. Delta norm, both SHA-256 hashes, and
the alternate-input BF16-reference rel-L2/cosine are recorded. The original
activation is restored and replayed before shared-helper timing warmups.

This gate has CPU/mock contract coverage only. Per instruction, no follow-up GPU
run was performed while adding it.

## gfx950 M=2 runtime smoke — stopped 2026-08-23

Artifacts: `/workspace/kimi-k3-runs/dense-crossover-2026-08-23/smoke/`.
AITER `b56d27beee1ce0132ca868da24d1a25e88e6ef42`; SGLang
`455b744aa77b2078de7577619dc12d2775fc1091`.

Preflight and cleanup found all eight gfx950 GPUs at 0% use and baseline
297,766,912 B VRAM. ROCm SMI retained PID 2327953 as `UNKNOWN` with zero VRAM,
SDMA, and CU occupancy; the PID was absent from `/proc` and `ps`, and no device
users were found.

The requested `latent_up`, M=2, four-mode command exited zero and all four rows
were finite and graph-capture safe. The required smoke is nevertheless marked
**failed/stopped**: `ptpc_fp8` recorded `dispatch.config=null`. No tuned
M=2/N7168/K3584 PTPC row resolved, so the runtime took the default CK fallback
and the benchmark metadata did not identify the actual backend. Per the stop
condition, no supplemental input-change replay was run. RMSNorm+MXFP4 graph
capture passed at M=2, but input-change replay remains unverified. Do not use
these smoke timings as final performance evidence.
