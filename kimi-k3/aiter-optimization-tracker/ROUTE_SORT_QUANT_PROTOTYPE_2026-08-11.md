# Kimi-K3 route-sort-quant prototype — 2026-08-11

## Goal

Evaluate whether the MI355X C2 route/sort/quant gap can be reduced by replacing:

```text
Opus expert sort + MXFP8 quant/sorted-scale write
```

with AITER's existing, not-yet-wired FlyDSL:

```text
fused route map + MXFP8 quant + expert scatter + scale preshuffle
```

The comparison starts after TopK; both paths consume identical
`topk_ids [M,16]` and bf16 activations `[M,3584]`.

## Existing prototype

`aiter.ops.flydsl.moe_kernels.flydsl_moe_fused_route_quant_scatter` already
emits:

```text
grouped_a1        [E, max_m, 3584] uint8/fp8 bytes
grouped_a1_scale  [E, max_m, 112]  e8m0 bytes (wmma_rep=1)
masked_m          [E]               valid rows per expert
topids_to_rows    [M, topk]         route -> grouped row
```

It is a grouped/DeepGEMM-style contract, not the current FlyDSL/Opus stage1
contract of unsorted FP8 activations plus sorted/swizzled route scales.

## Temporary prototype fixes

Two dormant-code issues were fixed only for measurement:

1. B1 on gfx950 incorrectly selected a gfx1250-only K-split builder.
2. gfx950 native FP8 conversion requested `i32`; the ROCDL op requires
   `vector<2xi16>`, then a bitcast.

The fixes were removed after the performance gate failed.

## Correctness

For B1/B2/B4/B8/B16/B32, every valid route was checked against
`per_1x32_mx_quant_hip`:

```text
FP8 payload mismatches: 0
E8M0 scale mismatches:  0
```

## Performance

```text
tokens  current us  candidate us  speedup  saved
1         18.690       17.829      1.048x  +0.861 us
2         18.518       19.319      0.959x  -0.801 us
4         19.026       19.157      0.993x  -0.130 us
8         19.113       19.688      0.971x  -0.575 us
16        19.820       20.107      0.986x  -0.287 us
32        21.302       20.260      1.051x  +1.042 us
```

The candidate does not meet the pre-wiring gate of approximately 10 us saved
per layer. It is flat overall and would additionally require a new grouped
stage1 consumer or a token-major scale-gather mode.

## Conclusion

Do not wire the existing fused route+quant+scatter prototype into Kimi-K3
serving.

The B300 route-preparation advantage does not come from sort+quant fusion
alone. The remaining opportunity is earlier in the chain:

- TopK/router fusion;
- route-map generation;
- CUDA route-quant concurrency;
- multi-stream overlap with adjacent GEMMs/collectives.

Any next prototype must include TopK or remove a larger launch boundary rather
than only replacing the already efficient Opus sort + MX quant chain.

Artifacts:

```text
stage2-runs/2026-08-11-route-sort-quant-prototype/
```
