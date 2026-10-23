---
name: mi400-cache-optimization
description: Optimize MI400 reuse, temporal hints, coherency scope and prefetch based on measured memory behavior.
---
# MI400 Cache Optimization
## Trigger
Cache misses/thrash, memory latency/bandwidth, reuse or prefetch is implicated by profiling.
## Workflow
1. Classify data: streaming, intra-kernel reuse, cross-workgroup reuse, cross-kernel reuse, unknown.
2. Map required coherency scope.
3. Treat RT as ordinary caching baseline.
4. Evaluate NT for low/no reuse traffic.
5. Evaluate HT only for valuable reuse where residency behavior is relevant.
6. Treat LU as a lifetime/reuse hint, not generic bypass.
7. Validate prefetch addresses, alignment/granularity and usefulness.
8. Benchmark hint/prefetch choices and record silicon revision.

## Required references
Read `decision-rules.md`, the topic reference, `anti-patterns.md`, then `optimization-checklist.md`.
