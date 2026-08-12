# Kimi-K3 V4 persistent route-prep prototype — 2026-08-11

## Goal

Replace the Opus HBM expert mesh and P0/P23 route scan with one gfx950
persistent FlyDSL launch while preserving the current AITER sorted ABI and
unchanged FlyDSL stage1.

The fixed contract was E=896, topk=16, non-EP, B1-B32, block_m=32/64.

## Baseline

Paired graph replay reproduced the V3-R complete-chain baseline within 10%.
Isolated timings include one graph replay launch overhead, so deltas are the
relevant comparison.

```text
tokens  TopK us  Opus sort us  quant us  chain us
1         15.129       14.539     11.410    29.648
2         15.315       14.696     11.580    30.117
4         15.382       14.693     11.604    30.266
8         15.461       14.612     11.623    30.357
16        15.523       14.678     11.773    30.600
32        15.570       14.617     12.096    31.107
```

## Prototype

The experimental kernel implemented:

1. graph-stable per-bucket workspace;
2. in-kernel count/start reset;
3. route histogram;
4. leader padded prefix over 896 experts;
5. stable token-major route rank and direct scatter;
6. padding sentinel and expert-block writes.

Cross-CU phases used agent-scope atomics and monotonic generation counters, so
graph replay did not require resetting barrier counters.

## Correctness

With agent-scope publication of `starts`, random legal TopK inputs matched Opus
for:

- block_m 32 and 64;
- `sorted_ids`;
- `sorted_weights`;
- `sorted_expert_ids`;
- `num_valid_ids`.

Inputs containing duplicate expert IDs within one token differed because Opus
mesh construction collapses/overwrites those invalid TopK duplicates. Production
TopK emits unique experts, but this edge case was not accepted as a broader ABI.

Small transient graph mismatches also appeared at B8/B16, showing that the
cross-CU publication contract was not yet production-safe.

## Performance

The first version launched one block per CU:

```text
tokens  Opus us  V4 us   delta
1         14.927  257.848 -242.921
2         14.594  251.427 -236.833
4         14.687  283.373 -268.686
8         14.731  270.871 -256.139
16        14.699  275.869 -261.170
32        14.747  262.566 -247.819
```

Limiting the resident grid to blocks that owned route waves did not recover the
cost:

```text
tokens  Opus us  active-worker V4 us
1         14.769       311.408
2         14.634       299.396
4         14.625       226.382
8         14.683       278.095
16        14.835       333.206
32        14.840       339.336
```

The main cost was the serialized 896-expert prefix plus coherent publication.
Replacing 896 agent atomics with plain stores reduced direct timing to about
118 us at B1 and 167 us at B32, but other CUs observed stale `starts` and sorted
payload correctness failed.

## Decision

V4 fails its first metadata/sort gate by more than an order of magnitude:

- count/prefix has no net benefit over Opus;
- sort-alone is nowhere near the required 5 us saving;
- cross-CU graph replay is not production-safe.

The wave64 producer, complete chain, serving wiring, and production validation
were therefore not run. All V4 source changes were removed; no commit was
created.

Further route work should not use a multi-CU persistent barrier for B1-B32.
The remaining credible directions are:

1. a one-CTA LDS oneshot sorter specialized for E=896 and M<=32; or
2. a stage1 ABI change that directly consumes route metadata/token-major scale
   and avoids rebuilding the sorted ABI.

Artifacts:

```text
stage2-runs/2026-08-11-v4-route-prep/
```
