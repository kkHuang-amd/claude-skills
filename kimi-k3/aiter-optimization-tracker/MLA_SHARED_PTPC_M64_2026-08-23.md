# Kimi-K3 MLA shared-input PTPC M64

Date: 2026-08-23

## CONTINUE HERE

**Status:** Implemented in SGLang default-off for gfx950 exact decode M64 after
the complete production-faithful micro passed by `1.1471x`. Focused dispatch,
fallback, numerical, storage, and changed-input graph tests pass. Production is
unchanged.

**Next:** Before any promotion, measure the `0.584799 GiB/GPU` capacity cost,
then run matched common-client C64 and a targeted TP8 retrace. Require the
combined QKV-A + gate boundary and total replay to move in the micro-predicted
direction before endpoint correctness gates.

**Flag:** `SGLANG_K3_AITER_MLA_SHARED_PTPC_M64=1`

**Artifacts:**
`/workspace/kimi-k3-runs/mla-shared-ptpc-m64-2026-08-23/`

## Why the earlier rejection was incomplete

The rejected QKV-A-only candidate quantized the same normalized hidden tensor
only for QKV-A. TP8 trace attribution measured:

```text
BF16 QKV-A, 24 layers:       370.099 us
FP8 QKV-A GEMMs:             272.504 us
standalone quant, 24 layers: 103.966 us
candidate chain:             376.507 us
delta:                        +6.007 us
```

That result remains correct for the QKV-A-only topology. It does not evaluate
ATOM's intended shared boundary, where one normalized-hidden `(fp8, scale)`
serves both QKV-A and g_proj.

## Exact ATOM mechanism

Source:

```text
/sgl-workspace/ATOM/atom/models/kimi_k3.py
/sgl-workspace/ATOM/atom/model_ops/layernorm.py
/sgl-workspace/ATOM/atom/model_ops/linear.py
```

`KimiFullAttention` enables fused input-norm quant only when
`fused_qkv_a_proj` and `g_proj` resolve to the same RMSNorm-fusable quant
scheme. The decoder constructs `input_layernorm` with that quant configuration.
The norm returns `(hidden_fp8, hidden_scale)`; attention passes the same pair to
both linears. `Linear.forward(x, x_scale=...)` skips its local activation quant
and dispatches `gemm_a8w8_bpreshuffle` for PTPC FP8. Attention later applies
`attn_out * sigmoid(g_proj(...))`.

ATOM's captured BS64 graph-warmup caller maps do not prove that this sharing was
active in that capture. They contain no `input_layernorm` caller annotation and
map one `dynamic_per_token_scaled_quant` under each projection. Across ranks,
the 24-layer medians are:

```text
ATOM QKV-A A8W8 GEMM:         299.077 us (12.462/layer)
ATOM QKV-A mapped quant:       91.036 us ( 3.793/layer)
ATOM QKV-A mapped clone:       70.879 us ( 2.953/layer)

ATOM g_proj A8W8 GEMM:        274.741 us (11.448/layer)
ATOM g_proj mapped quant:      93.197 us ( 3.883/layer)
ATOM g_proj mapped copy:       61.416 us ( 2.559/layer)
ATOM g_proj mapped clone:      65.613 us ( 2.734/layer)
```

The selected-decode aggregate maps both ATOM `mla_qkv_a` and `mla_gate` to the
same `407.455 us` operation total, so it is ambiguous and must not be used as
proof that ATOM's complete shared boundary is faster. SGLang's corresponding
selected medians are `370.099 us` for QKV-A and `287.039 us` for the mapped
gate. Its fused sigmoid/multiply output-gate kernel alone is `77.779 us` over
24 layers in BS64 warmup attribution (`3.241 us/layer` median). The micro below
therefore retains SGLang's output gate and compares complete boundaries.

## Canonical production-faithful micro

Runner:

```text
/workspace/useful-scripts/benchmarking/kimi-k3/micro/benchmark_mla_shared_ptpc.py
```

Command:

```bash
PYTHONPATH=/sgl-workspace/aiter-atom-current:/sgl-workspace/sglang-k3-triton37/python \
python /workspace/useful-scripts/benchmarking/kimi-k3/micro/benchmark_mla_shared_ptpc.py \
  --output /workspace/kimi-k3-runs/mla-shared-ptpc-m64-2026-08-23/micro.json \
  --m-values 32,64 --warmup 10 --iterations 100 --repeats 5
```

Environment:

```text
GPU:     AMD Instinct MI355X / gfx950
Torch:   2.9.1+rocm7.2.0.git7e1940d4
AITER:   dc4bdf1c142181ad90b7f6948564126df4c05fde
SGLang:  455b744aa77b2078de7577619dc12d2775fc1091 + preserved local changes
```

Weight quantization/preshuffling is outside timing. Every runtime activation
quant/conversion, both projection outputs, and SGLang's fused output gate are
inside the complete graph. Static inputs are changed between replays and both
QKV and gated-attention outputs must change.

### Component and total p50

```text
                                      M32       M64
BF16 QKV-A                         24.000    22.120 us
BF16 g_proj                        17.480    18.880 us
shared PTPC quant                  16.040    17.400 us
prequantized FP8 QKV-A             17.481    17.760 us
prequantized FP8 g_proj            17.060    18.160 us
SGLang fused output gate           16.760    17.381 us
BF16 g_proj + output gate          18.940    20.920 us
FP8 g_proj + gate, prequantized    18.200    18.200 us
complete BF16 boundary             29.240    34.320 us
complete shared boundary           31.720    29.920 us
boundary speedup                   0.9218x   1.1471x
```

Component captures each include common graph replay/event overhead and are not
additive. Complete-boundary distributions are cleanly separated:

```text
M64 BF16:  p50 34.320, p90 34.681, stdev 0.320 us
M64 shared:p50 29.920, p90 30.121, stdev 0.242 us
```

M32 is rejected. Exact M64 passes beyond noise.

### RMSNorm-inclusive variant

The exact available fused API is `aiter.rmsnorm_quant` with group size zero and
unshuffled per-token FP8 scale. Compared with current `aiter.rmsnorm2d_fwd`
plus the BF16 boundary:

```text
M32: 34.200 -> 31.840 us (1.0741x)
M64: 39.561 -> 30.240 us (1.3082x)
```

This fused producer is not integrated. Production K3's attention-residual path
already combines residual aggregation and input RMSNorm in its own kernel.
Calling standalone `aiter.rmsnorm_quant` would replace that faster producer,
not fuse into it. The safe model path quantizes once immediately after the
existing normalized-hidden producer. A future producer fusion would need the
attention-residual aggregation kernel to emit PTPC FP8 plus one FP32 scale per
row in addition to its current outputs.

### Numerical and graph gates

M64:

```text
QKV-A rel-L2 / cosine:            0.037734 / 0.999288
gated output rel-L2 / cosine:     0.015098 / 0.999886
fused-norm QKV rel-L2 / cosine:   0.037735 / 0.999288
fused-norm gated rel-L2 / cosine: 0.015058 / 0.999887
```

All outputs are finite. Baseline, candidate, RMS baseline, and fused-RMS
candidate all pass changed-input graph replay with distinct saved hashes.

## SGLang implementation

New adapter:

```text
python/sglang/kernels/ops/kimi_k3/mla_shared_ptpc_aiter_hip.py
```

It:

1. prepares and preshuffles QKV-A and gate PTPC weights after loading;
2. requires the flag, HIP, CUDA availability, gfx950, exact BF16 contiguous
   `[64,7168]`, exact prepared shapes/scales, and decode mode;
3. quantizes normalized hidden once;
4. lets `prepare_qkv_latent` and the output-gate wrapper consume the same
   `(hidden_fp8, hidden_scale)`;
5. retains SGLang's fused sigmoid/multiply output-gate kernel;
6. keeps BF16 weights and all existing fallbacks live.

Model/config/docs:

```text
python/sglang/srt/models/kimi_k3.py
python/sglang/srt/environ.py
docs/docs/references/environment_variables.mdx
```

Focused test:

```text
test/registered/kernels/ops/kimi_k3/test_mla_shared_ptpc_aiter_hip.py
6 passed
```

No AITER worktree was modified. The interrupted KDA input-projection MXFP4
files and model/environment/documentation hunks were preserved and not changed
as part of this work.

## Storage

```text
QKV-A prepared/layer:           15,147,264 B
gate prepared/layer:            11,016,192 B
combined incremental/layer:     26,163,456 B
combined incremental/24 layers: 627,922,944 B = 0.584799 GiB/GPU

retained BF16/layer:             52,297,728 B
BF16 + prepared/layer:           78,461,184 B
BF16 + prepared/24 layers:    1,883,068,416 B = 1.753744 GiB/GPU
```

This is arithmetic storage, not measured token capacity.

## Decision

Retain the new path default-off and exact-M64-only. M32, C2, prefill, target
verify, all other M, non-gfx950, unsupported layouts/dtypes, preparation
failures, and unavailable AITER APIs remain on the current path. No TP8 server,
endpoint, GSM8K, or long-context run was performed in this phase.
