# Profiling Decision Rules

## Select the first tool
- End-to-end gaps, CPU launch delay, communication overlap, page migration, multi-rank behavior: start with ROCm Systems Profiler.
- One or several expensive kernels with unclear limiter: start with ROCm Compute Profiler.
- Counter/roofline result is insufficient or a short instruction region is suspected: use PC sampling or thread tracing.
- Functional crash/incorrect state rather than performance: route to ROCm GDB/debugging before optimization.

## Classify from evidence
### Compute-bound candidate
Evidence should show high matrix/vector execution pressure with memory movement not explaining the elapsed time. Route to GEMM only when the hotspot is matrix-shaped or dominated by matrix execution.

### Memory/data-movement candidate
Evidence should show memory hierarchy traffic, latency/stalls, poor reuse, or explicit staging on the critical path. Route to Cache when reuse/eviction/prefetch is central; route to TDM when multidimensional global-LDS movement/address generation is central.

### Synchronization candidate
Evidence should identify waits, barriers, producer/consumer gaps, or waves stalled while peers progress. Route to synchronization/memory-ordering.

### Launch/runtime candidate
Evidence should identify dispatch, cluster launch configuration, cold-start/prefetch, or async-copy API behavior. Route to HIP runtime.

### Communication candidate
Evidence should identify collective or peer-transfer time, insufficient overlap, or topology-sensitive bandwidth/latency. Route to network/collectives.

## Required evidence record
For every claim record: tool, capture command/configuration, region/kernel, metric/trace observation, alternative explanation, and the measurement that would falsify the hypothesis.
