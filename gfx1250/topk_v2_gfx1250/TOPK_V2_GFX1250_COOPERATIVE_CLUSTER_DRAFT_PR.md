# Draft PR: opt-in cooperative-cluster TopK v2 for gfx1250

> Draft only. The merge-blocking correctness findings have been addressed, but
> integration and performance-signoff work remains below. This document records
> prototype evidence; it is not yet an upstream-readiness claim.

## Title

Add an opt-in gfx1250 cooperative-cluster path for DeepSeek-V4 TopK v2

## Summary

This change adds an experimental AMD gfx1250 route for the DeepSeek-V4 TopK v2
JIT kernel. It replaces the existing two-dispatch histogram/select path with
one cooperative-cluster dispatch when the row geometry is eligible.

Enable it with:

```bash
SGLANG_GFX1250_TOPK_CLUSTER=1
```

The existing path remains unchanged when the variable is unset or not exactly
`1`. CUDA and older ROCm targets retain their existing dispatch.

This is separate from
[PR #41687](https://github.com/sgl-project/sglang/pull/41687), which adjusted
wave-size handling in the AOT `dsa_topk_coop.cuh` kernel and did not add a HIP
cooperative-cluster launch.

## Implementation

- Detect gfx1250 for JIT compilation.
- Launch eligible kernels with `hipLaunchKernelEx` and y-oriented cluster
  dimensions `{1, N, 1}`.
- Use split cluster arrive/wait because combined `this_cluster().sync()` hangs
  with 1,024-thread workgroups on the tested stack.
- Stage per-rank 12-bit histograms in global memory and use
  `cluster_load_b32` register multicast for reduction. Remote LDS through
  `map_shared_rank` is unavailable.
- Use rank-prefix output placement and rank-0 exact tie handling. If a 12-bit
  threshold bin exceeds the 2,048-entry fast buffer, rank 0 re-reads the full
  row and performs unbounded exact radix refinement.
- Use an explicit Python-owned workspace per device/stream. Buffers are
  preallocated outside graph capture, retained with their stream to prevent
  handle reuse, capped at eight streams/device, and sized to about 40 MiB
  rather than the previous fixed approximately 128 MiB.
- Skip the CUDA persistent-pool planning kernel on gfx1250 because this route
  dispatches direct-row kernels only.
- Probe each HIP cluster kernel specialization with
  `hipOccupancyMaxPotentialClusterSize` and
  `hipOccupancyMaxActiveClusters` before its first launch.

Eligibility:

- 2-512 rows (batch 1 stays on the default route);
- C4 sequence length greater than 32,768 for at most 15 rows;
- C4 sequence length greater than 65,536 for larger batches.

## Correctness

Local gfx1250 results:

```text
cluster enabled: 326/326 TopK v2 tests passed
cluster disabled: 326/326 TopK v2 tests passed
duplicate-heavy regression subset: 6/6 passed
```

The suite covers raw indices, page-table output, dual output, ragged inputs,
TopK through 2,048, duplicate-heavy inputs, distinct-value threshold overflow,
two concurrent streams, graph replay, route observability, and dispatch
boundaries.

## Kernel microbenchmark

Configuration: gfx1250, TopK 512, uniform C4 lengths, output-set validation,
20 warmups and 100 requested GPU-event iterations.

| Rows | C4 length | Before: default (µs) | After: cluster (µs) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 131,072 | 20.83 | 17.35 | 1.201x |
| 4 | 131,072 | 24.52 | 18.19 | 1.348x |
| 1 | 262,144 | 25.80 | 22.91 | 1.126x |
| 31 | 131,072 | 43.86 | 25.28 | 1.735x |
| 64 | 131,072 | 48.79 | 39.26 | 1.243x |
| 64 | 262,144 | 69.62 | 41.86 | 1.663x |

These are prototype measurements. Fresh-process repetitions, randomized A/B
order, and dispersion are required before using them as a final PR claim.
Two post-hardening repetitions remained noisy: batch 31 / C4 131,072 improved
1.84x and 2.46x, while small-batch rows ranged from 0.81x to 1.18x. Do not
claim a universal win until the small-batch width policy is retuned with
controlled clocks and randomized order.

### Small-batch retuning

A 48-run sweep compared widths 2/4/8/15 for batches 1 and 4 at C4 lengths
131,072 and 262,144, followed by a 36-run sweep for batches 2/8/15. The
conservative default policy is now:

- batch 1: keep the existing split/streaming path;
- batches 2-8: width 4 for C4 length at most 131,072, width 8 above it;
- batches 9-30: width 8;
- batches above 30: retain width 4 at most 131,072 and width 2 above it.

`SGLANG_GFX1250_TOPK_CLUSTER_SIZE={2,4,8,15}` remains an explicit tuning and
diagnostic override. All four overrides have correctness coverage.

Three fresh-process checks of the tuned defaults produced median speedups of
1.06x for batch 1 / 131,072 (fallback-path timing noise), 1.35x for batch 4 /
131,072, 1.00x for batch 1 / 262,144 (fallback), 1.93x for batch 31 / 131,072,
1.07x for batch 64 / 131,072, and 1.25x for batch 64 / 262,144. Batch-1
fallback removes the prior 0.81x cluster regression; it does not claim a
kernel speedup.

## Profiling trace and kernel breakdown

A focused trace at 31 rows, C4 length 131,072, and TopK 512 captured 100
correctness-checked invocations per route:

```text
Before / cluster off:
  topk_split_select: 100 calls, 3,426.4 µs total, 34.264 µs/call
  topk_split_hist:   100 calls, 1,319.7 µs total, 13.197 µs/call
  route GPU total:              4,746.1 µs, 47.461 µs/call
  dispatches per invocation: 2

After / cluster on:
  topk_small_batch_cluster_kernel:
                     100 calls, 1,735.0 µs total, 17.350 µs/call
  dispatches per invocation: 1

Trace-derived kernel-time speedup: 2.736x
```

Artifacts:

```text
/sgl-workspace/topk_v2_gfx1250/profiles/topk_micro/off_b31_s131072.trace.json
/sgl-workspace/topk_v2_gfx1250/profiles/topk_micro/on_b31_s131072.trace.json
/sgl-workspace/topk_v2_gfx1250/profiles/topk_micro/kernel_breakdown.csv
```

The profile confirms the intended mechanism: the cluster route replaces a
histogram kernel plus selection kernel with one cooperative dispatch. The
trace speedup is larger than the earlier event benchmark, so final reporting
must use repeated runs and include profiler overhead caveats.

## TP4 long-context HiCache serving A/B

Both server lifetimes used identical DeepSeek-V4-Flash settings with
`HIP_FORCE_DEV_KERNARG`, `NCCL_MIN_NCHANNELS`, and `NCCL_MAX_NCHANNELS` unset.
Workload: 131,072 input tokens, 256 output tokens, 16 prompts, concurrency 8,
and four warmups.

Accuracy:

- Before, cluster off: deterministic retrieval 1/1.
- After, cluster on: deterministic retrieval 1/1.
- Serving requests: 16/16 succeeded in both runs, without a GPU fault.

Performance:

| Metric | Before: cluster off | After: cluster on | Change |
| --- | ---: | ---: | ---: |
| Mean TTFT | 53,475.39 ms | 52,133.93 ms | 2.51% better |
| Median TTFT | 52,849.98 ms | 52,001.82 ms | 1.60% better |
| Mean TPOT | 197.80 ms | 201.09 ms | 1.66% worse |
| Median TPOT | 200.08 ms | 201.60 ms | 0.76% worse |
| Total token throughput | 10,108.76 tok/s | 10,157.99 tok/s | 0.49% better |

This serving result is neutral and must not be presented as proof of an
end-to-end speedup. DSV4 uses a 4:1 C4 compression ratio, so a 131,072-token
context is approximately 32,768 C4 entries. The cluster condition is strictly
greater than 32,768 for small batches; decode at this exact shape can therefore
remain on the default route. Prefill can also exceed the 512-row eligibility
limit. The A/B run is useful regression evidence, but may not exercise the new
kernel at all.

## Better serving workload for the feature

Use a decode-heavy workload that is unambiguously above the C4 floor:

- 262,144 input tokens, 1,024-4,096 output tokens, concurrency 8;
- alternatively, just above the boundary (for example 131,073+ input tokens)
  after logging the actual C4 length and confirming a cluster dispatch;
- keep active rows at or below 15 so the lower 32,768 C4 floor applies;
- report TPOT and output-token throughput as the primary metrics;
- treat TTFT and total-token throughput as secondary because long prefill
  dominates both and is often ineligible for this route.

Run at least three interleaved fresh-process A/B rounds and record cold versus
warm HiCache state separately.

### 262K / 1K decode-heavy follow-up

The follow-up used 262,144 input tokens, 1,024 output tokens, 16 prompts,
concurrency 8, and four warmups. This corresponds to approximately 65,536 C4
entries and is above the small-batch cluster floor.

Accuracy was strengthened from one retrieval prompt to eight deterministic
prompts with different seeded insertion positions:

- Before, cluster off: 8/8 recovered the expected verification code.
- After, cluster on: 8/8 recovered the expected verification code.
- Both serving runs completed 16/16 requests without a GPU fault.

| Metric | Cluster off (profile requested) | Cluster on (unprofiled) | Raw difference |
| --- | ---: | ---: | ---: |
| Mean TTFT | 120,111.86 ms | 119,147.06 ms | 0.80% better |
| Median TTFT | 120,097.83 ms | 119,027.55 ms | 0.89% better |
| Mean TPOT | 127.59 ms | 127.23 ms | 0.28% better |
| Median TPOT | 127.61 ms | 127.34 ms | 0.21% better |
| Total token throughput | 8,399.31 tok/s | 8,444.26 tok/s | 0.54% better |
| Output token throughput | 32.68 tok/s | 32.86 tok/s | 0.55% better |

These performance rows are not a valid controlled A/B result. The cluster-off
run requested the live server profiler, while the cluster-on run was
unprofiled after that profiler failed. Even though no cluster-off trace was
written, profiler configuration or partial collection may have added overhead.
Rerun cluster off without profiling after the node is rebooted before quoting
an end-to-end delta. The direct TopK trace remains valid kernel-level evidence.

The 8/8 retrieval check is a meaningful long-context sanity gate, not a
complete accuracy evaluation. Upstream signoff should retain the 306/306
kernel suite and add a larger model-level suite (for example full GSM8K plus a
standard long-context retrieval benchmark).

### E2E profiling limitation

No valid server-level A/B trace was produced. The same live-profiler method
was attempted first on cluster off. After it failed, two recovery paths were
attempted:

1. A second SGLang live torch-profiler capture wedged the TP4 scheduler and
   emitted no rank trace.
2. External `rocprofv3 --attach` required `ROCP_TOOL_ATTACH=1`; enabling that
   caused HSA memory faults during server graph warmup before profiling began.

The faulted server was terminated and GPU memory was released. Do not present
an empty trace directory as profiling evidence. The valid focused TopK A/B
Chrome traces above remain the kernel-level evidence until the ROCm/SGLang
live-profiler issue is fixed.

## Remaining work before upstream review

The hardening revision resolves the singleton stream race, lossy threshold-bin
truncation, dead gfx1250 planning launch, silent allocation fallback, fixed
128 MiB workspace, and lack of route/boundary/graph tests.

Remaining:

1. Exercise explicit allocation failure and multiple physical devices in CI.
2. Decide whether eight retained stream workspaces per device is the right
   production cap and expose memory usage telemetry.
3. Record generated VGPR/LDS/spill data in addition to the runtime cluster
   occupancy probes.
4. Repeat randomized fresh-process benchmarks under controlled clocks and
   retune or disable small-batch shapes that do not consistently win.
5. Record cold JIT latency and code-object growth.
6. Obtain matched server-level traces after the ROCm live-profiler issue is
   fixed.

## Validation commands

```bash
bash /sgl-workspace/topk_v2_gfx1250/scripts/run_cluster_probes.sh
bash /sgl-workspace/topk_v2_gfx1250/scripts/validate_topk_cluster.sh
python3 test/registered/kernels/benchmark/attention/bench_topk_v2_gfx1250.py
```

Focused trace reproduction:

```bash
python3 /sgl-workspace/topk_v2_gfx1250/profile_topk_cluster.py \
  --mode off --batch 31 --seq 131072 --iterations 100 \
  --trace /tmp/topk-off.trace.json

python3 /sgl-workspace/topk_v2_gfx1250/profile_topk_cluster.py \
  --mode on --batch 31 --seq 131072 --iterations 100 \
  --trace /tmp/topk-on.trace.json
```
