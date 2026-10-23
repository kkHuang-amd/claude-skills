---
name: gfx1250-quickreduce-optimization
description: Develops, reviews, benchmarks, and validates AITER QuickReduce lossy all-reduce kernels on AMD gfx1250 for BF16 TP2/TP4. Use when optimizing PullQ4/INT4 codecs, IPC transport, graph-safe synchronization, peer-pull kernels, or comparing QuickReduce with AITER CAR and PyNCCL.
---
# gfx1250 QuickReduce Optimization

## Scope

Use this workflow for AITER QuickReduce on gfx1250, especially BF16 INT4
PullQ4 at TP2/TP4. Keep experimental routes force-only until both eager and
CUDA-graph results justify production routing.

Do not reset GPUs, reload drivers, rebind PCI devices, or reboot. After a GPU
fault, terminate only related workers, suppress core dumps, and run a small
PyTorch GPU smoke test.

## Start from the relevant code

- Kernel and communication state:
  `aiter/csrc/include/quick_all_reduce.cuh`
- System-scope flag primitives:
  `aiter/csrc/include/quick_all_reduce_base.h`
- Native entry points and buffer sizing:
  `aiter/csrc/kernels/quick_all_reduce.cu`
- Python communicator:
  `aiter/aiter/dist/device_communicators/quick_all_reduce.py`
- AITER tests:
  `aiter/op_tests/multigpu_tests/test_quick_all_reduce.py`
- SGLang benchmark:
  `sglang/benchmark/kernels/all_reduce/benchmark_rocm.py`

## Optimization workflow

1. Establish BF16 TP2/TP4 IPC baselines for:
   - AITER CAR
   - PyNCCL
   - existing QuickReduce
   - candidate PullQ4
2. Use 20 warmups and at least 100 GPU-event iterations. Aggregate with the
   slowest-rank mean.
3. Optimize one hypothesis at a time and retain per-size results.
4. Run focused rows before the complete 1–512 MiB eager/graph sweep.
5. Reject regressions promptly and remove dead experimental code.

Useful gfx1250 techniques:

- Quantize each 32 KiB logical tile once, publish it, then let every rank pull
  peer payloads and write BF16 output once.
- Decode one peer atom and immediately accumulate it to reduce register
  lifetime.
- Load one Q4 scale per eight-thread quantization group and broadcast it with
  a wave32-local shuffle.
- For TP2, stream input load/quantize/publish by atom. Re-measure before using
  this on TP4; retaining the tile can be faster there.
- Mask sub-16-byte tails rather than silently requiring aligned message sizes.
- Tune grid caps only with focused measurements; lower occupancy or more
  grid-stride synchronization can outweigh reduced launch work.
- When per-CTA peer polling dominates, split the operation into three
  same-stream dispatches: bulk quantize, one-CTA publish/wait, and bulk
  reduce. The dispatch boundary plus system-scope publication lets reduce
  CTAs read peer payloads without polling.
- Benchmark bulk below the initially suspected crossover. On gfx1250, the
  three-dispatch path was faster than fused PullQ4 at every measured
  1–512 MiB size despite its extra launches.
- Inspect the JIT code object with `llvm-readelf` and `llvm-objdump` before
  reducing occupancy blindly. The accepted fused kernels used 73–76 VGPRs,
  no spills, and no LDS; lowering the 1,216-CTA cap regressed performance.

## Graph-safe publication protocol

Treat correctness as a state-machine proof, not only a passing benchmark.

- Publish payload stores with a system-scope release flag store.
- Poll peer flags with system-scope acquire loads.
- Place a CTA barrier before publication and after all peer acquires.
- Keep two payload phases and two flag phases selected by epoch parity.
- Maintain one epoch per logical tile. A physical-block epoch is unsafe when
  graph replays change grid sizes.
- Give PullQ4 and legacy two-shot paths separate flag-memory subregions and
  separate epoch arrays. Shared state permits dormant-path ABA after wrap.
- Size epoch arrays for
  `max(kMaxNumBlocks, ceil(max_problem_size / 32 KiB))`.
- Size each legacy flag phase for
  `world_size * logical_tile_count * sizeof(uint32_t)`.
- Size each PullQ4 flag phase for
  `logical_tile_count * sizeof(uint32_t)`.
- Use 64-bit payload offsets when a two-phase allocation can cross 4 GiB.

For the bulk three-dispatch protocol:

- Quantize all local payload before publication.
- In a one-CTA publish/wait kernel, release-publish the local rank epoch and
  assign one thread to acquire-wait each peer epoch.
- Advance the local device epoch only after all peer publications are
  acquired.
- Launch reduce on the same stream; HIP/HSA dispatch completion preserves the
  acquired visibility for the following kernel.
- Keep bulk payload phases, flags, and epoch storage separate from fused
  PullQ4 and reject legacy-codec or fused-RMSNorm use on a PullQ4 communicator.
- Require identical collective order on one stream across all ranks.

One communicator must execute collectives in identical order on one stream on
every rank. Concurrent streams are unsupported; document this in public APIs.

## Adversarial validation

Before accepting the protocol:

1. Assert that the native communicator selected PullQ4; an environment
   variable alone does not prove dispatch.
2. Capture different-size graphs, such as 32 and 64 MiB.
3. Delay one rank to create skew.
4. Queue 1,000 alternating replays without per-replay synchronization.
5. Change inputs every replay.
6. Record a result sample for every replay with nonblocking copies to pinned
   host memory, then validate all samples after synchronization.
7. Use tolerances that reject zero and stale output.
8. Retain captured inputs, outputs, and graphs for their full replay lifetime.
9. Move worker results to CPU before producer processes exit.
10. Cover eager, graph, variable shapes, tiny/nonaligned boundaries, configured
    maximum size, TP2, and TP4.

Ask independent reviewers to separately audit:

- release/acquire ordering and epoch state transitions;
- allocation/index bounds and IPC/VMM size agreement;
- adversarial-test validity and route selection.

Do not declare the protocol legitimate until every concrete counterexample is
resolved and reviewers return no correctness blocker.

## Performance gate

Run BF16 IPC eager and graph sweeps from 1 MiB through 512 MiB with the SGLang
benchmark. Compare against exact AITER CAR rows and PyNCCL where available.

Keep PullQ4 force-only when it loses either eager or graph at a candidate
routing size. Never infer a production win from averages, projected bandwidth,
or a single execution mode.

The measured BF16 crossover for the accepted bulk path was 16 MiB on TP2 and
8 MiB on TP4 versus AITER CAR. At 64–512 MiB it reached 1.86–2.16x CAR speed
on TP2 and 3.19–4.28x on TP4. Keep smaller losing rows visible rather than
hiding them in an average.

Record:

- latency for every TP/mode/size row;
- speedup versus old QuickReduce and AITER CAR;
- unavailable PyNCCL rows and the exact failure;
- accepted and rejected optimization hypotheses;
- correctness and replay commands.

## Final checks

- Rebuild the JIT extension after C++/header changes with `AITER_REBUILD=1`.
- Run clang-format, Python compilation, `git diff --check`, and lints.
- Re-run TP2/TP4 correctness after the final protocol change.
- Re-run the full performance matrix after any layout or synchronization
  change.
- Make initialization failure-safe: Python must destroy a native communicator
  even before it marks itself enabled, and native destructors must continue
  best-effort cleanup without throwing.
- Preserve unrelated workspace changes.
