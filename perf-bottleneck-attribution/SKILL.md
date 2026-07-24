---
name: perf-bottleneck-attribution
description: >-
  Avoid misdiagnosing performance bottlenecks. Guards against the common attribution traps when profiling or
  optimizing GPU/CPU/systems code: mistaking a profiler stall-site for the cause, confounded experiments that
  move two variables at once, changing an enabling condition without verifying the actual outcome, asserting
  "X-bound" with no roofline/utilization number, declaring an optimization "exhausted" from a chain of indirect
  nulls, comparing peak hardware specs instead of achieved utilization, and deep-optimizing a structurally
  handicapped design instead of an existing alternative. Use when profiling or optimizing performance,
  diagnosing why a kernel/query/service is slow, deciding whether something is compute/memory/bandwidth/
  latency/occupancy-bound, tuning with a profiler (ATT/nsight/perf/rocprof), or concluding an optimization is
  exhausted or "at the hardware ceiling".
---

# Performance bottleneck attribution — don't fool yourself

Diagnosing *what* limits performance is where most optimization time is wasted. Nearly every misdiagnosis is
the same root error: **a bottleneck was INFERRED from indirect evidence instead of MEASURED**, or **an
intervention changed an enabling condition without changing the actual outcome**. Before acting on "it's
X-bound" or "it's exhausted", run the patterns and the checklist below.

## The one rule
"Bound by resource R" and "R is at its ceiling" are the SAME quantitative claim. If you cannot put a
utilization number on it (achieved-R / peak-R), you have a hypothesis, not a diagnosis.

## Misdiagnosis patterns

### 1. Stall-site != cause
A profiler tells you WHERE a thread/wave is stalled (which instruction/PC), not WHY. A stall attributed to a
*compute* instruction (matmul/tensor-op/FMA) is very often that instruction **waiting for its operands** (a
data-dependency hazard on values still in flight from memory/cache), not the compute unit being saturated.
- Falsify: make the suspected unit cheaper (lower-precision op, fewer FLOPs, smaller tile). If time does not
  improve, that unit was NOT the bound - the stall was upstream (operand/memory delivery).
- Trap: "50% of stall cycles are on the matmul instruction -> compute-bound." A cheaper matmul gives 0
  speedup -> it was operand delivery all along.

### 2. Confounded experiment (one knob, two variables)
If the only knob that reaches the state you want to test ALSO changes something else, a null/regression cannot
be attributed to your hypothesis.
- Falsify: isolate the target variable (a knob that changes ONLY it), or add a second independent probe that
  moves the same variable a different way.
- Trap: "config C reaches higher occupancy but regressed -> occupancy does not help." C also halved compute
  efficiency; the regression was the efficiency loss, so occupancy was never actually tested.

### 3. Enabling condition != outcome (verify the intervention took effect)
"I made X possible" is not "X happened." Measure the OUTCOME quantity, not the precondition you changed.
- Falsify: directly measure the thing you intended to change (actual occupancy/concurrency/in-flight
  requests/cache residency), not the budget or flag that merely permits it.
- Trap: freed the resource so 2 units/slot were *allowed*, saw no change, concluded "more units do not help."
  But the scheduler/launcher only ever placed 1 unit/slot, so the allowed-but-unused headroom changed nothing;
  the outcome was never verified.

### 4. "X-bound" without a roofline number
Do not infer "compute/memory/bandwidth/latency-bound" purely from which levers helped. Measure achieved-X
against peak-X.
- Falsify: achieved = work / time; compare to the spec peak (bytes/time vs peak GB/s; FLOPs/time vs peak;
  achieved occupancy vs max).
- Trap: concluded "bandwidth-bound" from lever behavior; the direct measurement showed ~15% of peak bandwidth
  used -> not bandwidth-bound at all; 85% of the roofline was idle.

### 5. "No win yet" != "impossible / at the ceiling"
A chain of indirect nulls is weak evidence. Only a DIRECT measurement of the bottleneck upgrades "I could not
find a win" to "it is at the limit."
- Falsify: ask "what single direct measurement would prove the ceiling?" (roofline utilization / achieved
  occupancy / achieved BW). If you have not taken it, you have not proven the bound - state it as a hypothesis
  with a confidence level.
- Trap: declared "exhausted / at hardware ceiling" after several nulls; each was overturned by the next direct
  check.

### 6. Peak spec != achieved (systems/hardware comparison)
Two systems or configs with the SAME peak (bandwidth, FLOPs, IOPS) can perform very differently. Peak is a
ceiling; achieved depends on the software keeping the resource busy (parallelism-in-flight, async/pipelining,
batching). So "same peak, therefore mechanism M cannot help" is wrong when M raises *achieved* utilization
toward peak.
- Falsify: compare achieved utilization, never datasheet peaks.
- Trap: "both chips have the same peak bandwidth, so the async/prefetch mechanism cannot matter" - but it does,
  by keeping the pipe fuller (higher achieved BW at the same peak).

### 7. Zoom out: is the design itself handicapped?
Deep local optimization of artifact A can be inferior to simply using an existing alternative B that lacks A's
structural constraint. Before grinding, check: does an existing alternative already dominate, and does my
artifact carry a built-in disadvantage (a constraint the alternative does not have)?
- Falsify: benchmark the alternative EARLY; explicitly name your design's structural constraints.
- Trap: optimized a design forced into low occupancy by its own architecture; an existing alternative without
  that constraint already beat it - the design choice, not the kernel, was the bottleneck.

## Meta-skill: adversarial falsification
For every "it's X-bound / it's exhausted" claim, ask: **"What observable signature must this mechanism have -
and did I MEASURE it, or infer it?"** Actively seek the experiment that would FALSIFY the current diagnosis.
Treat a sharp challenge to a conclusion as a gift: it usually points straight at an unverified assumption.
Hold diagnoses as hypotheses with explicit confidence until a direct measurement pins them.

## Pre-conclusion checklist
Run before saying "X-bound", "occupancy/compute/memory/bandwidth/latency-limited", or "exhausted / at the
ceiling":
- [ ] Roofline: measured the suspected resource's achieved utilization vs its peak? (a number, not an inference)
- [ ] Each key experiment changed ONE variable (no confound)?
- [ ] Verified the intervention changed the OUTCOME, not just an enabling condition?
- [ ] For a "stall on op X": ran the cheaper-X test to separate "unit saturated" from "waiting for inputs"?
- [ ] Cross-system/config claims compare ACHIEVED utilization, not peak specs?
- [ ] Benchmarked the existing ALTERNATIVE; named my design's structural constraints?
- [ ] Stated as a proven bound or a hypothesis, and named the single measurement that would falsify it?

## Worked case study
A full worked example of every pattern (a GPU MoE GEMM successively mis-labeled MFMA/compute-bound ->
occupancy-limited -> bandwidth-bound, each corrected by a direct measurement) is recorded in the
`[CORRECTION]`/`[RETRACTED]` notes of `dsv4/megamoe/GEMM_CODESIGN_WIP.md` and `KERNEL_OWNER_DECODE_PLAN.md`.
