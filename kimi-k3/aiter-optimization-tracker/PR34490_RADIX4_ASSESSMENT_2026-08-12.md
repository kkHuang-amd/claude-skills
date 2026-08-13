# SGLang #34490 Radix-4 Kimi-K3 router assessment — 2026-08-12

## Decision

The queued integration is complete. Retain SGLang
[#34490](https://github.com/sgl-project/sglang/pull/34490) as a validated,
default-off candidate after adding exact AITER tie handling, NaN exclusion and
an explicit gfx942/gfx950 guard.

Measured decision:

```text
45 focused tests passed
C2 paired median: 970.38 -> 993.11 tok/s (+2.34%)
C4/C8/C16/C32: +2.05% / +1.71% / +1.34% / +0.68%
GSM8K 200: 0.985
capacity: 933883
```

Full results:
[`PR34490_RADIX4_RESULTS_2026-08-12.md`](PR34490_RADIX4_RESULTS_2026-08-12.md).

Before this assessment, the exact kernel had not been tested locally. It is
architecturally distinct from the rejected V3/V3-R/V4 route experiments and
therefore qualified as a new approach rather than a retry.

## What was considered previously

The 2026-08-11 route work identified a future multi-block TopK/radix design as
a possible direction after one-wave TopK+quant failed. The exact Radix-4
implementation in #34490 was not designed or evaluated:

```text
V3a:
  256-thread CTA, four-wave shared reduction, 16 repeated argmax rounds
  result: 0.4-1.15 us/layer slower; all-tie expert IDs differed

#34490:
  256-thread CTA, four-bit pivot rounds, packed 16-bin histograms,
  DPP prefix sum and ballot
  reported result: about 4-5 us/layer saved for common decode token counts
```

The prior reports anticipated the broad radix direction, but not this
nibble-histogram/DPP algorithm. The PR was opened on 2026-08-12, after the
local V3/V3-R/V4 experiments were completed on 2026-08-11.

## Relationship to completed experiments

```text
V3 256-thread TopK:
  Similar launch geometry, different selection algorithm.

HIP TopK+quant:
  Appended serialized quant work to the existing router; #34490 replaces only
  the TopK selection.

Route sort+quant prototype:
  Began after TopK and is complementary to #34490.

V3-R/V4:
  Targeted route metadata, sorting, scatter and persistent scheduling rather
  than a standalone Radix-4 selector.
```

## Upstream claim

For graph-captured MI355X, E=896 and topk=16:

```text
M       AITER us   Radix-4 us
1         10.24       5.99
8         10.56       5.75
16        10.64       5.51
64        10.45       5.77
256       10.48       6.38
512       10.97       7.05
1024      11.09       9.44
1536      12.08      11.95
```

The upstream dispatch is limited to row-contiguous `[M,896]`, top-16,
ungrouped routing and `M<=1024`. The PR reports 1-3% E2E improvement, but that
claim is not yet comparable to the fresh vendored production baseline.

## Reusable method learned

### Core observation

The generic K3 router finds top-16 by repeating a maximum-selection round 16
times, so work scales with `topk`. Radix selection instead finds the score
threshold (pivot) by key digits, so the number of rounds scales with key width.

### Algorithm

For each token:

```text
1. Launch one 256-thread CTA.
2. Distribute 896 experts across four wave64 waves.
3. Compute:
     emitted weight = sigmoid(score)
     ranking value  = sigmoid(score) + correction_bias
4. Convert each FP32 ranking value to a monotonic uint32 key.
5. Skip high bits shared by all keys.
6. Resolve the top-k pivot four bits at a time:
     build a 16-bin histogram for the current nibble
     reduce per-wave bin counts
     scan bins from high to low
     select the bin containing the remaining k-th key
     discard keys outside that bin
7. Stop early when every surviving key is a winner.
8. Compact keys above the pivot, then fill remaining slots from pivot ties.
9. Emit plain sigmoid weights and expert IDs; optionally renormalize.
```

### CDNA-specific mapping

The important implementation choices are:

```text
packed histogram:
  Store 16 four-bit counters in one 64-bit register. A nibble can count up to
  15 local values, so split larger per-thread holdings into accumulator chunks.

wave reduction:
  Use DPP stages to reduce packed bin counts while leaving totals in lane 63,
  avoiding a readlane broadcast per histogram word.

bin prefix:
  Transpose descending bin totals into lanes 0-15, then use four DPP prefix
  steps plus a ballot to locate the first cumulative count reaching k.

register residency:
  Keep each thread's ranking keys, sigmoid weights and alive mask in registers;
  use LDS only for cross-wave histogram totals and final winners.
```

The performance principle is to replace 16 synchronization-heavy selection
rounds with at most eight 4-bit key rounds, often fewer after common-prefix
skipping and early exit.

### When to reuse this pattern

This method is a candidate when:

```text
the expert count is large
topk is large enough that repeated argmax is expensive
each token can own one CTA
keys fit in registers
the decode token grid has not saturated the GPU
the output consumer does not require sorted winner order
```

It is less suitable when:

```text
M is large enough that one CTA per token becomes throughput-bound
grouped routing requires group masking before expert selection
exact tie ordering is part of the serving contract
the target GPU lacks the required wave64/DPP behavior
```

### Generalization rule

Do not copy the constants blindly. For another expert shape or GPU, retune:

```text
radix width
histogram counter width
threads and waves per CTA
experts per thread
M coverage cap
tie policy
architecture dispatch guard
```

The transferable idea is **digit-wise pivot selection with packed
wave-register histograms**, not specifically radix-4 or a 256-thread block.

## Risks to resolve

1. The PR adds no focused unit test.
2. Winner collection uses cross-thread atomic increments. All-tie and
   boundary-tie inputs must prove exact expert-ID compatibility with AITER.
3. The wrapper describes CDNA-only support but currently gates on HIP rather
   than an explicit gfx942/gfx950 architecture check.
4. NaN, extreme logits, strided rows, graph replay and output-order behavior
   need focused validation.
5. Upstream AMD GPU jobs were skipped behind the PR gate as of assessment
   time; lint passing is not MI355X runtime validation.

## Integration gates

Use an isolated branch/worktree based on the fresh vendored SGLang stack.

### Gate A — build and dispatch

```text
fresh JIT cache
gfx950 build succeeds
dispatch evidence proves Radix-4 is active for covered shapes
fallback proves AITER remains active outside coverage
```

### Gate B — exact correctness

Compare against AITER for:

```text
M=1/2/4/8/16/32/64/256/512/1024
BF16 and FP32 scores
random and extreme logits
all-tie and boundary-tie cases
NaN behavior
strided score rows
renormalize on/off
CUDA graph replay
```

Any serving-relevant expert-ID mismatch stops integration unless the contract
is explicitly proven safe.

### Gate C — operator performance

Run paired graph-captured measurements at the same M values. Require at least:

```text
3 us/layer median saving for the production decode range
no regression at any covered production shape
stable results across five paired rounds
```

The earlier 10 us/layer route-chain gate does not apply unchanged because
#34490 is a localized dispatch with no stage1 ABI migration.

### Gate D — endpoint

Compare with the fresh vendored baseline using identical workload and warmup:

```text
C2/C4/C8/C16/C32
TPOT, TTFT and total/output throughput
GSM8K 200
max token capacity
```

Keep the feature independently gated and default-off until these gates pass.

## Execution order

```text
1. Integrate and validate #34490.
2. Record accepted or rejected results.
3. Resume retained B300 normal versus single-stream/no-PDL analysis.
```
