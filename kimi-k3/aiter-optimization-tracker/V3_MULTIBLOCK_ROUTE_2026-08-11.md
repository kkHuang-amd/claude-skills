# Kimi-K3 V3 multi-block route prototype — 2026-08-11

## Goal

V3a was the first gate toward a role-based route/quant grid:

```text
1 route CTA + independent quant CTAs per token
```

Because a single kernel launch has one block size, V3 first needed a
256-thread TopK that could share a grid with 256-thread quant CTAs.

## V3a implementation

An independent gfx950 K3 TopK API was added without modifying production:

- one 256-thread CTA per token;
- each thread scanned about four of 896 experts;
- wave-local argmax;
- four-wave shared-memory block reduction;
- 16 repeated selections;
- sigmoid + correction bias + renorm + routed scaling.

Fresh JIT, strided K3 router slices, random/extreme inputs, and graph replay
were tested at B1/B2/B4/B8/B16/B32.

## Correctness

Random and extreme cases were byte-identical.

The all-tie case failed the strict contract:

```text
topk weights: identical
topk IDs:     5 IDs differ (20 bytes)
```

The current wave64 reduction topology defines a nontrivial tie order. The
four-wave reduction selects a different, still mathematically valid set, but
serving requires exact expert identity.

## Performance

Five paired rounds:

```text
tokens  wave64 us  block256 us  saved
1         11.986      12.826    -0.840
2         11.275      12.425    -1.150
4         11.652      12.449    -0.796
8         12.624      13.090    -0.466
16        12.582      12.988    -0.406
32        12.529      13.080    -0.552
```

V3a is slower at every batch size.

## Root cause

The production kernel completes each top-k round entirely inside one wave.
V3a adds, for all 16 rounds:

- wave partial stores;
- a shared-memory barrier;
- a second wave reduction;
- another block barrier.

That overhead is larger than the benefit of distributing 896 scores over four
waves.

## Decision

Stop at V3a as required by the plan:

- no concurrent quant CTA grid;
- no expert histogram/route slots;
- no precomputed P23;
- no stage1/runner/endpoint changes.

All V3a code was removed. No commit was created.

The broader route optimization now needs a design that preserves the wave64
TopK rather than replacing it. Possible future work must use separate launches
or hardware/graph concurrency, or fuse work into Opus sorting without forcing
TopK onto a 256-thread block.

Artifacts:

```text
stage2-runs/2026-08-11-v3-multiblock-route/
```
