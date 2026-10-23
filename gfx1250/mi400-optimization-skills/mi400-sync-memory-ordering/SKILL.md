---
name: mi400-sync-memory-ordering
description: Design and verify MI450 split-barrier, named-barrier, async-copy and cluster synchronization protocols.
---
# MI400 Synchronization and Memory Ordering
## Trigger
Barrier/wait/order optimization, async data movement coordination, race or deadlock analysis.
## Workflow
1. Write producer -> publish -> release/order -> signal -> independent work -> wait -> acquire/order -> consume.
2. Choose narrowest valid scope: wave/workgroup/cluster/device.
3. Identify LDS versus global address space.
4. Follow high-level barrier ordering contracts when used.
5. For low-level hardware barriers, provide required release/acquire ordering explicitly.
6. Put only independent work between arrival and wait.
7. Verify initialization, participant count, phase reuse and lifecycle.
8. Prefer hierarchical/local synchronization over device-wide polling when possible.
