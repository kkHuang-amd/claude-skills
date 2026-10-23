# TDM Decision Rules

## Eligibility
Use TDM only when the target is supported and global-LDS/LDS-global tensor movement or its address generation is material in the profile.

## Mode selection
- Normal: one multidimensional tile with regular descriptor-based traversal.
- Iterative: repeat a smaller tile with step sizes; separately prove aggregate bounds because per-iteration handling does not prove cross-iteration safety.
- Scatter/gather: two-dimensional arbitrary global rows mapped to/from contiguous LDS where supported by the access pattern.
- Multicast: identical data requested by clustered workgroups; measure fallback under drift.

## Descriptor construction
Keep base address, element width, logical dimensions, physical strides, tile dimensions and offsets explicit. Dimensions are used for bounds; strides determine address progression. When shifting a tile by base adjustment, preserve the original physical stride semantics.

## Completion choice
- Only issuing wave consumes: tensor-count/wave-local wait can be sufficient when used correctly.
- Other waves consume: add workgroup synchronization or use LDS-barrier completion that all consumers can observe.
- Independent operations consumed out of order: use distinct synchronization state; issue-order retirement does not provide arbitrary per-operation readiness.

## Throughput choice
- Meet direct-copy granularity/alignment requirements when seeking the bypass/direct path.
- Use both TDM ports only after placing destinations to avoid the same LDS partition and checking shared address-generation contention.
- Keep per-wave in-flight operations within the documented safe limit; HIP compiler-generated paths may insert waits, low-level code must do so explicitly.

## Store/read interaction
TDM store bypass behavior can leave an older normal-load value in L0. Before a subsequent ordinary load of the updated location, use an appropriate cache-management/ordering strategy and test it.
