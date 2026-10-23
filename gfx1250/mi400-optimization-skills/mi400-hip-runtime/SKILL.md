---
name: mi400-hip-runtime
description: Apply MI400 HIP runtime features such as cluster launch, runtime prefetch and asynchronous copy safely.
---
# MI400 HIP Runtime
## Trigger
Cluster launch, runtime launch attributes, runtime L2 prefetch, memcpy_async, or fabric-memory runtime work.
## Workflow
1. Query/verify capability for the target.
2. Choose static cluster dimensions when fixed or runtime cluster attributes when dynamic.
3. Validate launch geometry against cluster dimensions and kernel resource use.
4. For runtime L2 prefetch validate exact region bounds and granularity.
5. For memcpy_async prove the double-buffer/consumer synchronization and determine whether the accelerated target path applies.
6. Report fallback behavior and measured effect.
