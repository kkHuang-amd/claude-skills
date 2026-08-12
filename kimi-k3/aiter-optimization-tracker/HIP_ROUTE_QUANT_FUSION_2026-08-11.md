# Kimi-K3 HIP TopK + MXFP8 quant producer — 2026-08-11

## Goal

Prototype a gfx950 producer that combines:

```text
biased grouped TopK
per-token group-32 MXFP8 quant
```

for the production K3 contract:

```text
E=896, topk=16, hidden=3584, sigmoid + correction bias, B1-B32
```

The producer emits separate AITER-compatible topk IDs/weights and token-major
FP8 activation/scales; no TRT packed-topk ABI is used.

## Implementation

The candidate extended AITER's existing HIP `grouped_topk_kernel`. After
routing, the same wave64 block quantized the activation row:

- four 16-lane subgroups;
- each subgroup processed 32-element MX groups;
- E8M0 scale used the project RoundUp policy;
- FP8 conversion used AITER `scaled_cast`.

A new independent Python/C++ API and kernel name kept the production path
unchanged.

## Correctness

Fresh-JIT checks covered:

- strided K3 router slices;
- B1/B2/B4/B8/B16/B32;
- ties and extreme logits;
- CUDA graph replay.

All outputs were byte-identical:

```text
topk IDs:       exact
topk weights:   exact
FP8 payload:    exact
E8M0 scales:    exact
```

## Producer microbenchmark

Five paired rounds:

```text
tokens  current us  fused us  saved
1         13.930      24.776  -10.846
2         13.809      26.038  -12.229
4         14.315      26.503  -12.187
8         14.652      27.054  -12.402
16        14.690      26.834  -12.144
32        14.734      27.019  -12.285
```

The producer fails the first gate decisively.

## Root cause

AITER's TopK kernel uses one wave64 block per token. Appending 112 MXFP8 groups
to that block serializes quant work behind routing.

The current separate quant kernel distributes groups across many blocks/CUs,
so:

```text
TopK + parallel quant  <  TopK block followed by serialized in-block quant
```

B300's CUDA `route_quant_fused` uses a different launch structure: routing and
quant CTAs can execute concurrently. Reproducing that on HIP requires a new
block-grid/radix design, not an epilogue added to the current one-wave TopK.

## Decision

Stop at the producer gate:

- do not build Phase 1 sort/scale consumer;
- do not modify FlyDSL token-major scale loads;
- do not thread a pre-quant AITER runner contract;
- do not run endpoint tests.

All prototype changes were removed. No commit was created.

Any future full route prototype must provide independent quant CTAs concurrent
with routing, likely alongside a new multi-block TopK/radix implementation.

Artifacts:

```text
stage2-runs/2026-08-11-hip-route-quant-fusion/
```
