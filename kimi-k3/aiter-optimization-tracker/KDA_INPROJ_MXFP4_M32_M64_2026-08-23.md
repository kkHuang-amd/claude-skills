# Kimi-K3 KDA input-projection MXFP4 M32/M64 candidate

Date: 2026-08-23

## CONTINUE HERE

**Status:** Implemented in SGLang as a default-off, fail-closed candidate.
Focused CPU dispatch and one-gfx950 numerical/storage/graph gates pass.
Production is unchanged and no promotion is claimed.

**Next:** Measure real TP8 serving capacity, then run the matched common-client
C2/C64 workload and recapture the same complete prefill and decode GPU steps.
Require the targeted C64 `kda_inproj` chain and total replay gap to move in the
predicted direction before endpoint/accuracy gates.

**Flag:** `SGLANG_K3_AITER_KDA_INPROJ_MXFP4_M32_M64=1`

**Artifacts:**
`/workspace/kimi-k3-runs/kda-inproj-mxfp4-m32-m64-2026-08-23/`

## Verified production contract and insertion point

SGLang's existing `SGLANG_ROCM_K3_FUSE_KDA_INPROJ` path merges the TP8-local
BF16 KDA projections as:

```text
[q,k,v,g | f_a | b | pad]
shape: [6288,7168]
split: [4608,1536,128,12,4]
layers: 69
```

The merged parameter is contiguous and the original projection modules are
repointed to views, so retaining the BF16 fallback has no extra duplicate BF16
copy. In production with `SGLANG_USE_AITER=1`, the current fused call routes
through `UnquantizedLinearMethod.apply` to
`aiter.tuned_gemm.tgemm.mm`. The candidate is inserted immediately before this
same fused call in `KimiK3DeltaAttention.forward_qkvbfg_fused`.

The candidate preserves the existing split order and deferred-`f_b` handoff.
Prefill, target verify, M1/M2 group64, and all decode sizes other than exact
M32/M64 continue through the current path.

ATOM was inspected for mechanism context. Its KDA path keeps the tile-aligned
`[q,k,v,g]` projection separate from beta and `f_a`; PTPC KDA input projection
is unsupported in the corrected dense matrix. This candidate therefore uses
SGLang's actual fused `[6288,7168]` contract and AITER MXFP4, not ATOM PTPC.

## Implementation and exact guard

Changed SGLang surfaces:

```text
python/sglang/kernels/ops/kimi_k3/kda_inproj_mxfp4_aiter_hip.py
python/sglang/srt/models/kimi_k3.py
python/sglang/srt/environ.py
docs/docs/references/environment_variables.mdx
test/registered/kernels/ops/kimi_k3/test_kda_inproj_mxfp4_aiter_hip.py
```

Prepared mechanism, using current AITER
`/sgl-workspace/aiter-atom-current@dc4bdf1c`:

1. load-time `get_hip_quant(QuantType.per_1x32)` weight quantization with
   `quant_dtype=dtypes.fp4x2, shuffle=True`;
2. `shuffle_weight(..., layout=(16,16))`;
3. per-call activation quantization with the same per-1x32 MXFP4 mechanism;
4. `aiter.gemm_a4w4(..., dtype=torch.bfloat16, bpreshuffle=True)`.

The complete timed call includes activation quantization, A4W4 GEMM, and BF16
output conversion. Weight preparation is load-time and outside replay timing.
Current AITER reports a null tuned config for both rows, so dispatch is the
supported gfx950 generic ASM path rather than a tuned CSV row.

Runtime dispatch requires all of:

```text
explicit flag enabled at model construction
PyTorch HIP build, CUDA device available, runtime architecture exactly gfx950
AITER per-1x32 quantizer, 16x16 shuffle, fp4x2/e8m0 dtypes and gemm_a4w4 APIs
forward mode exactly decode
M exactly 32 or 64
input exactly contiguous CUDA BF16 [M,7168]
current retained weight exactly contiguous CUDA BF16 [6288,7168]
prepared weight exactly contiguous CUDA fp4x2 [6288,3584]
prepared scale exactly contiguous CUDA e8m0 [6400,224]
all tensors on the same device
```

Any failed preparation or runtime condition leaves the current fused
BF16/group64/split path selected.

## Focused production-faithful graph micro

One MI355X/gfx950, current AITER `dc4bdf1c`, SGLang `455b744a`, 10 warmups,
100 graph replays, seed 20260823. BF16 is
`aiter.tuned_gemm.tgemm.mm`; MXFP4 is the complete mechanism above.

```text
M32 BF16:   29.408700 us
M32 MXFP4:  28.822690 us
delta:      -0.586010 us
speedup:     1.02033x
rel-L2:      0.164591
cosine:      0.986379

M64 BF16:   42.709250 us
M64 MXFP4:  29.916680 us
delta:     -12.792570 us
speedup:     1.42761x
rel-L2:      0.164920
cosine:      0.986323
```

Both BF16 and MXFP4 rows are finite and pass changed-input graph replay with
distinct output hashes. The prior full matrix remains the selection evidence
at `1.044x`/`1.455x`; this focused rerun independently preserves the crossover
direction, with a smaller M32 margin.

## Storage

Measured prepared tensor layout:

```text
packed weight: [6288,3584] fp4x2, 22,536,192 B
scales:        [6400,224] e8m0,     1,433,600 B
incremental per KDA layer:         23,969,792 B
69 KDA layers:                  1,653,915,648 B
                                  1.540329 GiB/GPU
```

The original BF16 weight remains live for all fallbacks. This is exact tensor
storage, not a measured serving-capacity delta.

## Validation

```text
focused candidate + existing in-proj tests:
  11 passed, 14 subtests passed

covered:
  default-off
  exact M32/M64 decode bucket ownership
  other-M fallback, including C2
  prefill fallback
  unsupported shape/dtype/layout/scale fallback
  M32/M64 numerical comparison and BF16 output
  exact prepared storage
  changed-input HIP graph replay
  existing fused layout/split/deferred-f_b behavior
```

Artifacts:

```text
focused-tests-final.log
focused-chain.json
focused-chain.csv
focused-chain.log
prepared-layout.txt
```

## Decision and remaining gates

Retain the candidate default-off. Do not promote yet.

Outstanding gates:

1. real TP8 capacity measurement with the additional 1.540329 GiB/GPU;
2. matched common-client C2/C64 run using the persisted manifests and all
   unrelated flags fixed;
3. same-method complete prefill and decode retrace, including proof of zero
   candidate markers in prefill/C2 and targeted M32/M64 markers in decode;
4. targeted C64 `kda_inproj` and total graph-step attribution;
5. only after those pass: fixed 8K/1K and 68K/350 endpoint matrices, paired
   GSM8K, and paired long-context output/logprob checks.

No TP8 server, endpoint, capacity, production retrace, or accuracy run has been
performed. Graph-external analysis remains deferred.
