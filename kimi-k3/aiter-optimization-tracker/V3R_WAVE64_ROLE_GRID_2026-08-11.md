# Kimi-K3 V3-R wave64 role-grid prototype — 2026-08-11

## Goal

Preserve the production wave64 TopK body and tie order while allowing routing
and MXFP8 quant CTAs to execute concurrently in one 64-thread launch.

```text
grid.x = M * 5
role 0 = unchanged wave64 TopK
roles 1-4 = independent token-major MXFP8 quant
```

Both token-major and role-major block ordering were evaluated.

## R1 producer

Correctness passed for:

- random, ties, and extreme logits;
- strided K3 router slices;
- B1/B2/B4/B8/B16/B32;
- CUDA graph replay;
- TopK IDs/weights and FP8/E8M0 bytes.

Role-major scheduling was consistently best.

```text
tokens  current us  role-grid us  saved
1         11.870       10.792     +1.079
2         13.534       11.052     +2.482
4         14.104       11.494     +2.610
8         14.588       12.049     +2.539
16        14.699       12.040     +2.659
32        14.737       12.044     +2.693
```

A 15-round B1 repeat measured:

```text
current 11.767 us
role-grid 10.119 us
saved 1.649 us
```

R1 passed its scheduling gate.

## R2 expert metadata

The wave64 route CTA emitted:

```text
expert_counts [896]
route_slots [M,16]
```

Counts summed to `M*16`; slots were dense and unique per expert. A graph-stable
explicit zero was included in the timing.

Metadata overhead over R1:

```text
B1  +2.389 us
B2  +1.667 us
B4  +3.137 us
B8  +5.148 us
B16 +5.447 us
B32 +4.898 us
```

## R3 precomputed P23

A non-EP kernel accepted precomputed counts and rebuilt the standard AITER
route ABI:

- `sorted_ids` token/slot packing;
- padding sentinel;
- `sorted_weights`;
- `sorted_expert_ids`;
- `num_valid_ids`.

The first serial implementation was 50-70 us slower. A parallel revision used:

- 256-thread expert prefix reduction;
- shared route flags;
- deterministic route-order reconstruction;
- parallel padding/expert-block writes.

It reproduced current Opus sorted metadata for block_m 32/64 at B1-B32.

## Complete Phase-1 chain

Current:

```text
wave64 TopK → Opus P0/P23 → fused MXFP8 quant/sorted-scale
```

V3-R:

```text
role-grid TopK+quant+metadata → precomputed P23 → scale-only sort
```

```text
tokens  current us  V3-R us  saved
1         27.309      24.928  +2.381
2         28.242      26.049  +2.193
4         28.612      26.429  +2.183
8         29.399      27.348  +2.052
16        29.624      30.220  -0.596
32        30.152      40.513 -10.361
```

Core payload and route metadata were exact. Any transient sorted-scale
differences were not reproducible and a dedicated B32 rerun found zero real or
padding row differences.

## Decision

V3-R fails the complete-chain gate:

- B1/B2 savings are only about 2.2 us, far below 10 us;
- B16 begins to regress;
- B32 regresses by more than 10 us.

Do not add token-major scale gathering, runner wiring, or endpoint tests.
All V3-R source changes were removed; no commit was created.

The route gap cannot be closed by preserving the current sort ABI with a
per-expert precomputed P23. A future design would need a persistent/current-ABI
kernel with more efficient global prefix/scatter, or a different grouped stage1
layout; both are V4-scale projects.

Artifacts:

```text
stage2-runs/2026-08-11-v3r-wave64-role-grid/
```
