# ATOM Kimi-K3 C64 multi-stream versus single-stream — 2026-08-13

## Result

The current ATOM stack was tested on the same 8x MI355X machine with the
Kimi-K3 TP8 recipe. Disabling ATOM's dual-stream MoE path reduced C64
throughput and increased TPOT:

```text
Machine:  crsuse2-m2m-002.crusoe.amd.com
ATOM:     5479c5af3282621a28e87b361a40947e1619a80a
AITER:    e8b4507e588a083534f1a25469b3616085c9813a
Torch:    2.13.0+rocm7.14.0
Triton:   3.7.0
GPU:      8x AMD Instinct MI355X

multi-stream:   8742.34 total tok/s, 49.84 ms median TPOT
single-stream:  8399.50 total tok/s, 53.33 ms median TPOT
single delta:     -3.92% throughput, +7.01% TPOT
```

Both unprofiled runs used fixed 8192/1024 input/output, concurrency 64, 64
warmup requests, 512 measured requests, seed 42 and ignore-EOS. Both completed
512/512 requests.

## Single-stream control

The only configuration difference was:

```bash
export ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0
```

The default is `1024`. Kimi-K3 uses the dual-stream path at or below that token
count to overlap shared-expert GEMMs on `alt_stream` with routed experts on the
main stream. Setting the threshold to zero disables dual-stream registration
and calls `single_stream_moe_forward` directly.

## Kernel-time trace method

The first ATOM PyTorch-profiler run used `--torch-profiler-dir` and
benchmark `--profile`, but omitted `--mark-trace`. Its eight run traces contain
zero `kernel`, `gpu_memcpy` and `cuda_runtime` events; those files contain
CPU-side scopes only.

For ATOM's native CUDA Graph workflow, the complete command is:

```bash
unset KINETO_CONFIG
python -m atom.entrypoints.openai_server \
  ... \
  --torch-profiler-dir <trace-dir> \
  --mark-trace

python -m atom.benchmarks.benchmark_serving ... --profile
python /app/ATOM/tools/parse_trace.py <rank run trace>
```

`--mark-trace` creates per-batch-size capture traces and is the intended input
to `parse_trace.py`. However, the production rerun on the current
Torch 2.13/ROCm 7.14 stack emitted CPU scopes only in both run and capture
traces despite requesting `ProfilerActivity.CUDA`; the parser therefore could
not find `gpu_user_annotation`. `ATOM_PROFILER_MORE=0` remains recommended to
avoid unnecessarily large shape, stack and memory data.

ATOM's own CI supports `rocm-trace-lite` (`rtl`). Version `0.3.7` was installed
without dependencies. Kimi-K3 uses full CUDA Graph replay, so traces were
captured with:

```bash
rtl trace --mode full -o <trace-dir>/trace.db -- \
  python -m atom.entrypoints.openai_server <Kimi-K3 recipe arguments>
```

ROCm 7.14 satisfies RTL full mode's ROCm 7.13+ requirement. Each trace used one
C64 wave with 64 fixed 8192/64 requests, no warmups and ignore-EOS. Trace-mode
endpoint latency includes profiler overhead and is not a performance result.
The SQLite databases included startup and runtime; the benchmark
timestamp/window was used when extracting matched runtime kernels.

## Trace validation

Before deletion, all eight main rank databases contained measured GPU
operations and per-kernel durations:

```text
multi-stream:  393k-394k GPU ops per rank
single-stream: 557k-566k GPU ops per rank

multi retained size after SQLite analysis/checkpoint:  431 MiB
single retained size after SQLite analysis/checkpoint: 599 MiB
```

Representative rank summaries include kernel call counts, total duration and
average duration. Examples include NCCL, AITER cross-device reduction,
SiTUv2 stage1/stage2 GEMMs, dense GEMMs, attention residual, and KDA kernels.

## Artifacts and deletion status

```text
/workspace/kimi-k3-runs/c64-stream-ab-2026-08-13/multi/
/workspace/kimi-k3-runs/c64-stream-ab-2026-08-13/single/

single/benchmark-c64-8192x1024.json
single/benchmark-c64-8192x1024.log

deleted: multi/trace_*.db, multi-rank0.perfetto.json
deleted: single/rtl/
```

The replacement production FULL-CUDAGraph trace is:

```text
/workspace/kimi-k3-runs/c64-graph-host-kernel-2026-08-13/
  atom-c64-full-graph-host-kernel.perfetto.json.gz

/workspace/kimi-k3-runs/c64-single-stream-graph-host-kernel-2026-08-13/
  atom-c64-single-stream-full-graph-host-kernel.perfetto.json.gz
```

Both merge same-run PyTorch host/operator scopes (including the BS64 capture
hierarchy) with RTL GPU kernel events across all eight GPUs:

```text
multi-stream:  4,855,000 KernelExecution; 865,324 cpu_op
single-stream: 5,572,000 KernelExecution; 874,630 cpu_op
```

The single-stream server was launched with
`ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0`. Both trace-mode request waves are
profiler-perturbed and must not replace the unprofiled benchmark numbers.

No ATOM server, benchmark or profiler process was left running.
