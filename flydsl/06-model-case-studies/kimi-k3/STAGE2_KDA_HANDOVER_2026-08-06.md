# Kimi-K3 Stage 2 KDA FlyDSL versus Triton handover

Date: 2026-08-06

## CONTINUE HERE

The KDA kernel comparison is complete. AITER FlyDSL is the retained
experimental winner; the SGLang Triton implementation was correct and
spill-free but approximately 2x slower.

No server or benchmark is active.

Current SGLang integration changes are uncommitted in:

```text
/sgl-workspace/sglang
branch: perf/k3_moe-opt
base commit: 3aec6fc4610fe7ab86f80730126a80a91a60e3d3
```

The required AITER experiment is isolated in:

```text
/sgl-workspace/aiter-kda-4495
branch: k3-exp-4495
base: 6dc26b7a817bba2ae92ff22a7a63c0c024f4c7e5
PR #4495 functional commits:
  b02ba49ab
  3b0586c82
  4db42b3a8
  455474f9b
```

The AITER worktree also has an uncommitted SGLang compatibility correction:
cache slot zero is valid, so the FlyDSL kernels use `state_idx >= 0` rather
than the draft PR's `state_idx > 0`.

## Retained SGLang integration

Files:

```text
python/sglang/kernels/ops/attention/kda_fused_decode_aiter_hip.py
python/sglang/srt/layers/attention/linear/kda_backend.py
python/sglang/srt/models/kimi_k3.py
```

Enable with:

```text
SGLANG_K3_KDA_FUSED_BACKEND=aiter
```

The integration is fail-closed and only prepares on HIP/gfx950 with the exact
Kimi-K3 TP8 contract. It:

- flattens checkpoint `A_log` from `[1,1,12,1]` to the AITER `[12]` ABI;
- defers the `f_b` projection only for eligible decode;
- fuses `f_b + conv update + KDA recurrence + gated RMSNorm`;
- keeps the original tiny-GEMM plus Triton chain as fallback;
- warms the FlyDSL specialization before CUDA graph capture.

Do not use the earlier sweep under `flydsl-integration/final-c*`: it was
captured before the `A_log` layout fix and therefore measured fallback.
The valid results are under `flydsl-integration/active-c*`.

## Correctness

```text
AITER PR focused suite:             12 passed
Expanded batches/slot-zero suite:   23 passed
Shared B=1..32 oracle:              output/state RRMSE < 1e-3
FlyDSL CUDA graph replay:           passed
Triton focused suite:               5 passed, repeated stable
Full-model GSM8K 200 (FlyDSL):      0.985
```

The active production graph contains:

```text
kimi_k3_kda_decode_fb_bf16_gfx950
69 launches, rank trace total 1.2644 ms
```

The prior packed KDA graph kernel measured approximately 1.8801 ms for the
same 69 launches, a 32.8% KDA-bucket reduction.

## Kernel comparison

Direct identical-contract CUDA graph comparison:

```text
Batch   AITER us   Triton us   AITER speed advantage
1       11.69      29.50       2.52x
2       12.80      30.05       2.35x
4       12.96      30.05       2.32x
8       13.34      30.06       2.25x
12      13.74      31.19       2.27x
16      14.52      31.54       2.17x
24      18.50      37.43       2.02x
32      19.78      38.55       1.95x
```

The standalone Triton kernel used MFMA, 211 VGPR, 106 SGPR, and zero
scratch/spills. It was removed from the parent SGLang working tree because its
latency was not competitive. The implementation remains in isolated worktree:

```text
/root/.cursor/worktrees/k3-kda-7c4e91af/sglang-ff23c246a4b4
branch: perf/k3-kda-fused-decode
```

## Serving comparison

Baseline is the final MoE-copy sweep at SGLang `3aec6fc46`, AITER `6dc26b7a8`.
Both sides use TP8, 8192/1024, random-range-ratio 1.0,
`num-prompts=8*concurrency`, and `warmups=2*concurrency`.

```text
Conc  Baseline TTT  FlyDSL TTT  Delta   TPOT delta  ITL delta
2       882.44         911.52   +3.30%    -3.37%     -3.35%
4      1594.32        1640.52   +2.90%    -3.14%     -3.24%
8      2527.54        2668.38   +5.57%    -5.78%     -6.16%
16     3893.59        4068.83   +4.50%    -4.91%     -5.51%
32     5381.41        5544.38   +3.03%    -3.66%     -4.22%
```

TTFT is effectively unchanged except C32 at +0.50%.

## FP8 KV cache sweep

The same retained FlyDSL KDA profile was rerun with only:

```text
--kv-cache-dtype fp8_e4m3
```

changed from the BF16/auto KV baseline.

```text
Conc  BF16 TTT  FP8 KV TTT  TTT delta  TPOT delta  ITL delta
2       911.52      898.36     -1.44%      +1.53%     +1.55%
4      1640.52     1597.44     -2.63%      +2.45%     +2.55%
8      2668.38     2585.74     -3.10%      +3.39%     +3.34%
16     4068.83     3872.62     -4.82%      +5.45%     +5.50%
32     5544.38     5194.81     -6.30%      +7.39%     +8.06%
```

FP8 KV reduces memory footprint but is slower for this fixed 8192/1024
workload. The regression grows with concurrency, so BF16/auto remains the
performance default.

## Persistent artifacts

```text
/workspace/claude-skills/kimi-k3/stage2-runs/2026-08-06-kda-flydsl-vs-triton
```

Important files:

```text
aiter-focused-tests.log
aiter-expanded-tests.log
shared-oracle.log
graph-oracle.log
flydsl-microbenchmark.log
flydsl-batch-microbenchmark.json
direct-kernel-ab.json
serving-comparison.json
flydsl-integration/gsm8k-200-active.log
flydsl-integration/active-c*/
flydsl-integration/graph-profile/
../2026-08-06-kda-flydsl-fp8kv-sweep/
```

## Next steps

1. Decide whether to upstream the slot-zero correction to AITER PR #4495.
2. Commit/push the SGLang integration only after the AITER dependency revision
   is stable and publicly available.
3. Keep the feature opt-in until the draft AITER PR is merged.
