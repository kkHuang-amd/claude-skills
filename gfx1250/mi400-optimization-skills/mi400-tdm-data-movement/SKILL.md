---
name: mi400-tdm-data-movement
description: Optimize MI450 multidimensional global-memory/LDS movement with TDM while enforcing its correctness constraints.
---
# MI400 TDM Data Movement
## Trigger
Repeated tensor-tile global<->LDS staging, material address-generation overhead, or profiling identifies data movement on the critical path.
## Workflow
1. Describe tensor dimensions, strides, element size, tile and LDS layout.
2. Choose simplest suitable mode: normal, iterative, scatter/gather, or cluster multicast.
3. Verify granularity/alignment and LDS placement.
4. Choose wave-local completion or LDS-barrier coordination.
5. Evaluate one versus two TDM engines without creating LDS/resource conflicts.
6. For multicast verify clustered launch and measure effectiveness under workgroup drift.
7. Compare directly against vector/async-copy baseline.
## Guardrails
- Parameters/control flow must be wave-uniform.
- Validate iterative bounds across iterations.
- Respect the documented per-wave in-flight operation constraint; low-level code must manage waits explicitly.
- Keep dimensions and strides semantically distinct.
- TDM global stores can require cache management before subsequent normal loads of the same data.

## Required references
Read `decision-rules.md`, then descriptors/multicast/hazards, then `anti-patterns.md` and `optimization-checklist.md`.
