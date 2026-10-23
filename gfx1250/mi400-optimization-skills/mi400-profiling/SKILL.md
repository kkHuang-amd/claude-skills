---
name: mi400-profiling
description: Diagnose MI400/MI450 performance and route optimization work using ROCm profiling evidence.
---
# MI400 Profiling
## Trigger
Use when performance is poor/unexplained or before selecting an optimization when no bottleneck evidence exists.
## Workflow
1. Reproduce baseline and load shared measurement contract.
2. Use ROCm Systems Profiler when end-to-end CPU/GPU/communication overlap matters.
3. Use ROCm Compute Profiler for kernel roofline, memory hierarchy, counters and hardware-block analysis.
4. Escalate to PC sampling/thread tracing for instruction, wave and stall diagnosis.
5. Classify: compute, memory/data movement, synchronization, launch/runtime, communication, or unknown.
6. Route to the smallest specialized skill set.
## Output
Evidence, classification/confidence, candidate actions, verification plan.

## Required references
Read in this order:
1. `references/decision-rules.md`
2. the profiler-specific reference needed for the investigation
3. `references/anti-patterns.md`
4. `references/optimization-checklist.md`

## Exit criteria
Do not route to an optimization skill until the report contains a reproducible hotspot, evidence, a bottleneck classification, and a metric expected to change if the hypothesis is correct.
