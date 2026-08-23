# Kimi-K3 C64 ATOM/SGLang pause — 2026-08-12

## Resumed with current ATOM — 2026-08-13

The user explicitly resumed trace capture on 2026-08-13. The current container
does not contain the original stack, so the new result is a current-version
validation rather than an exact rerun of the SHAs below:

```text
ATOM:    5479c5af3282621a28e87b361a40947e1619a80a
AITER:   e8b4507e588a083534f1a25469b3616085c9813a
Torch:   2.13.0+rocm7.14.0
Triton:  3.7.0
GPU:     8x AMD Instinct MI355X
```

The matched unprofiled C64 endpoint run completed 512/512 requests:

```text
fixed input/output: 8192/1024
warmup requests:    64
ATOM current:       8742.34 total tok/s, 49.84 ms median TPOT
recorded ATOM:      8380.39 total tok/s, 52.66 ms median TPOT
current delta:      +4.32% throughput, -5.36% TPOT
```

The first ATOM PyTorch-profiler run used `--torch-profiler-dir` and `--profile`
but omitted `--mark-trace`. Its eight compressed run traces contain CPU scopes
only, with zero `kernel`, `gpu_memcpy` and `cuda_runtime` events. They were
removed after successful comparable traces became available:

```text
removed: /workspace/kimi-k3-runs/c64-repro-2026-08-13/trace/atom-native/
```

GPU timing was recaptured with ATOM's supported `rocm-trace-lite` full mode.
The single-stream 8192/1024 endpoint produced `8399.50 tok/s` with `53.33 ms`
median TPOT, versus current multi-stream `8742.34 tok/s` and `49.84 ms`.
The old multi/single raw RTL and Perfetto files were later deleted at the
user's request; benchmark JSON/log files remain. Historical details:
[`C64_ATOM_STREAM_AB_2026-08-13.md`](C64_ATOM_STREAM_AB_2026-08-13.md).

Production FULL-CUDAGraph profiling was repeated with
`--torch-profiler-dir`, `--mark-trace`, and benchmark `--profile`. The native
run and per-BS capture traces had complete PyTorch host scopes, but this
Torch 2.13/ROCm 7.14 Kineto build emitted zero GPU activities even though
`ProfilerActivity.CUDA` was requested. Consequently `tools/parse_trace.py`
failed with `No decode gpu_user_annotation found in run trace`.

The completed replacement used simultaneous rank-0 PyTorch CPU profiling plus
RTL full mode on all eight GPUs, then merged the same-run timelines at the BS64
capture boundary and first BS64 decode transition:

```text
/workspace/kimi-k3-runs/c64-graph-host-kernel-2026-08-13/
  atom-c64-full-graph-host-kernel.perfetto.json.gz

5,759,978 total events
4,855,000 KernelExecution events
865,324 cpu_op events
39,590 user_annotation events

/workspace/kimi-k3-runs/c64-single-stream-graph-host-kernel-2026-08-13/
  atom-c64-single-stream-full-graph-host-kernel.perfetto.json.gz

6,464,900 total events
5,572,000 KernelExecution events
874,630 cpu_op events
18,206 user_annotation events
```

The second trace set `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0`. Both merged
traces contain the BS64 capture host hierarchy and production graph replay
kernels. Raw native run/capture traces and RTL SQLite databases are retained
beside them; incomplete CPU-only retries were removed.

## Earlier closure — 2026-08-13

The user stopped the profiling investigation. All raw traces under
`/workspace/kimi-k3-runs/c64/profiles/` were deleted as requested. Do not
resume the old profiling steps below unless explicitly requested.

The validated C64 endpoint numbers and non-trace benchmark logs/results remain
retained.

## CONTINUE HERE

Machine:

```text
crsuse2-m2m-002.crusoe.amd.com
```

The C64 endpoint comparison is complete. ATOM is faster, so matched runtime
profiling was started but intentionally paused at the user's request.

```text
SGLang f9dd3a0661b472d5fba1632adebcffc5c7c4021e
AITER  284a1eb401bb15f6368a68b34eb0cd693ee1fcd3
ATOM   f782218a526611b62745f2c0bb5e94656e04da1e

fixed input/output: 8192/1024
measured requests:  512
warmup requests:    64
concurrency:        64

SGLang: 8014.22 total tok/s, 55.58 ms median TPOT
ATOM:   8380.39 total tok/s, 52.66 ms median TPOT
ATOM delta: +4.57% throughput, -5.25% TPOT
```

Both endpoint runs completed 512/512 requests.

## Profiling attempts

PyTorch runtime profiling is not usable on this stack:

```text
ATOM: rank 4 ModelRunner segfaulted (exit -11) after profiling started.
SGLang: per-BS profiler segfaulted while scanning large graph batches.
```

Restricting SGLang to `--cuda-graph-bs-decode 64` successfully produced all
eight BS64 graph-capture traces. ATOM also produced BS64 graph-capture traces.
However, the SGLang scheduled trace contains only profiler bookkeeping/fill
kernels rather than the graph's runtime kernels, so it is not a valid matched
kernel-duration comparison.

`rocprofv3 --attach` reported success but produced no files at detach/finalize
time. The next approach is to launch each server under `rocprofv3` with a
delayed collection window.

## Exact next step

1. Launch ATOM under `rocprofv3` with:

   ```text
   --collection-period 330:120:1
   --collection-period-unit sec
   --kernel-trace --memory-copy-trace --stats -f csv
   ```

2. Record launch epoch. Wait for the ATOM API to become ready; the prior
   rocprof-launched startup took about 296 seconds.
3. At about launch+335 seconds, run one matched wave:

   ```text
   C64, 64 requests, fixed 8192/64, no warmups, ignore EOS
   ```

4. Stop the server only after launch+455 seconds so rocprof can finalize CSV.
5. Repeat for exact handover SGLang. Choose a delayed window after observing
   its rocprof-launched startup time.
6. Aggregate rank0 kernel CSV by normalized kernel family and compare:

   ```text
   summed kernel duration
   decode-window wall span
   MoE stage1/stage2 and route/sort/quant
   dense BF16/FP8 GEMMs
   KDA/MLA
   all-reduce
   copies/materialization
   ```

7. Write the summary, verify it is reproducible from retained CSV, then remove
   raw graph/runtime traces.

## Artifacts

```text
/workspace/kimi-k3-runs/c64/sglang-c64.log
/workspace/kimi-k3-runs/c64/sglang-c64.jsonl
/workspace/kimi-k3-runs/c64/atom-c64.log
/workspace/kimi-k3-runs/c64/atom-c64.json
/workspace/kimi-k3-runs/c64/profiles/atom/
/workspace/kimi-k3-runs/c64/profiles/sglang/
/workspace/kimi-k3-runs/c64/atom-profile-server-short.log
/workspace/kimi-k3-runs/c64/sglang-profile-server-per-bs.log
/workspace/kimi-k3-runs/c64/sglang-profile-server-bs64.log
```

No server, benchmark, or profiler process was intentionally left running.
