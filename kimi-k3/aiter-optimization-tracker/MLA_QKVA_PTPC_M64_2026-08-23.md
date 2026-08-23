# Kimi-K3 MLA QKV-A PTPC FP8 M64 candidate

Date: 2026-08-23

## CONTINUE HERE

**Status:** Rejected at the production TP8 targeted GPU-step gate. The SGLang
implementation, test and environment flag were removed; production is unchanged.
**Next:** Implement KDA input-projection MXFP4 at M32+ as a separate default-off,
fail-closed GPU-first candidate. It is not implemented yet.
**Historical flag (removed):** `SGLANG_K3_AITER_MLA_QKVA_PTPC_M64=1`
**Artifacts:** `/workspace/kimi-k3-runs/mla-qkva-ptpc-m64-2026-08-23/`

## Scope and verified production contract

The production Kimi-K3 MLA fused QKV-A projection is local
`M x K` by `N x K` with:

```text
M = 64 only
K = 7168
N = 2112 = Q-LoRA 1536 + KV-LoRA 512 + RoPE 64
```

The originally suggested N=2304 does not match the loaded production module or
ATOM's Kimi-K3 model. Both construct a merged replicated linear with output
partitions `[1536, 576]`.

## Implementation

Rejected SGLang implementation surface, removed after retrace:

```text
python/sglang/kernels/ops/kimi_k3/mla_qkva_ptpc_aiter_hip.py
python/sglang/srt/models/kimi_k3.py
python/sglang/srt/environ.py
docs/docs/references/environment_variables.mdx
test/registered/kernels/ops/kimi_k3/test_mla_qkva_ptpc_aiter_hip.py
```

The two dedicated files were deleted and only the candidate-specific hunks were
removed from the three shared files. Pre-existing latent MXFP4 and other
uncommitted work was preserved. No AITER checkout was modified.

The adapter mirrors ATOM's actual PTPC path:

1. load-time `get_hip_quant(QuantType.per_Token)` weight quantization;
2. `shuffle_weight(..., layout=(16, 16))`;
3. per-call `get_hip_quant(QuantType.per_Token)` activation quantization;
4. `aiter.gemm_a8w8_bpreshuffle(..., dtype=torch.bfloat16)`.

Reference implementation inspected:

```text
/sgl-workspace/ATOM/atom/models/kimi_k3.py
  KimiFullAttention.fused_qkv_a_proj
/sgl-workspace/ATOM/atom/model_ops/linear.py
  LinearBase.online_quantize_weight
/workspace/useful-scripts/benchmarking/kimi-k3/micro/benchmark_dense_crossover.py
  DenseBenchmark._prepare / _runner
```

Activation quantization and the GEMM's required BF16 output conversion are in
the captured production call. Weight quantization/preshuffle is load-time
setup. The original BF16 linear weight and normal projection path remain live.

Exact dispatch requires all of:

```text
explicit flag enabled
PyTorch HIP build and CUDA device available
AITER PTPC quantizer, shuffle and bpreshuffle GEMM APIs available
runtime architecture exactly gfx950
forward mode is decode
hidden shape exactly [64, 7168], BF16, contiguous
prepared weight shape exactly [2112, 7168], AITER FP8, contiguous
weight scale shape exactly [2112, 1], FP32, contiguous
activation, weight and scale on the same CUDA device
```

Any failed preparation or runtime guard leaves or selects
`DeepseekV2AttentionMLA.prepare_qkv_latent`, the existing BF16/current path.
All M other than 64, including C2, fail closed.

## Storage

Per MLA layer:

```text
retained BF16 weight: 30,277,632 B
incremental FP8 + FP32 row scales: 15,147,264 B
dual total: 45,424,896 B
```

Across 24 MLA layers, incremental prepared storage is exactly:

```text
363,534,336 B = 346.693359 MiB = 0.338568 GiB/GPU
```

This is arithmetic/model-weighted storage, not a measured serving-capacity
delta. Capacity measurement was skipped because the earlier targeted GPU-step
gate failed.

## Validation

Current AITER source:
`/sgl-workspace/aiter-atom-current@dc4bdf1c142181ad90b7f6948564126df4c05fde`.

Focused dispatch tests:

```text
9 passed, 1 deselected
```

They cover default-off, exact decode M64 ownership, non-M64 fallback,
non-decode fallback, unsupported shape, dtype and layout fallback.

One-gfx950 numerical/storage/input-change graph test:

```text
1 passed, 9 deselected
prepared bytes: 15,147,264 per layer
finite output and changed-input graph replay: passed
```

Production-faithful focused graph micro, 10 warmups/100 iterations:

```text
BF16:      27.815900 us, rel-L2 0.004169, input-change passed
PTPC FP8:  20.840980 us, rel-L2 0.037615, cosine 0.999292,
           input-change passed
observed ratio: 1.335x
```

The prior canonical full matrix measured `22.562620 us` BF16 versus
`21.511440 us` PTPC (`1.049x`). The focused rerun's PTPC latency is consistent,
but its BF16 baseline is slower. Retain the canonical `1.049x` selection
evidence only as the reason this candidate was selected for retrace; production
TP8 evidence below supersedes it for the decision.

## Production TP8 retrace

Decode targeted-chain artifact:

```text
/workspace/kimi-k3-runs/mla-qkva-ptpc-m64-2026-08-23/
  trace/sglang-c64-decode/analysis/targeted-gap.json
```

Exact TP8 medians:

```text
baseline BF16 QKV-A:           370.099 us
candidate FP8 GEMM:            272.504 us
online activation quant:       103.966 us
candidate complete chain:      376.507 us
candidate chain delta:          +6.007 us (slower)

baseline total graph:           31.012 ms
candidate total graph:          30.566 ms
total graph delta:              -0.447 ms
```

The FP8 GEMM itself saves `97.595 us`, but the required online quantization
costs `103.966 us`; the complete candidate chain is therefore slower by
`6.007 us`. It fails the targeted GPU-step gate.

The total graph is `0.447 ms` shorter, but rank-0 and cross-rank decomposition
show that delta is dominated by routed-MoE, collective and route-work variation.
It must not be attributed to QKV-A, whose directly measured complete chain did
not close.

Same-profile prefill retrace:

```text
/workspace/kimi-k3-runs/mla-qkva-ptpc-m64-2026-08-23/
  trace/sglang-c64-prefill/
```

`analysis/prefill-guard.json` records zero candidate PTPC quant/A8W8 markers on
all ranks, confirming the M64 decode-only guard. The selected complete steps
have a TP8 median of 3,783 kernels; individual baseline/candidate ranks contain
3,783 or 3,784 kernels, so the stronger claim of exactly 3,783 on every rank is
not supported by the saved guard artifact. Absolute prefill spans remain
profiler-perturbed and composition-only.

## Decision

Reject and remove this candidate before capacity, endpoint or accuracy gates:

- targeted production chain regressed by `6.007 us`;
- no endpoint, GSM8K or long-context work is justified;
- all raw trace and micro artifacts remain preserved;
- production defaults and the BF16 path remain unchanged;
- the next GPU-first candidate is KDA input-projection MXFP4 at M32+, still
  default-off and unimplemented.

Graph-external analysis remains deferred. The unrelated total-graph movement is
not a reason to reopen it because the target chain itself already failed.
