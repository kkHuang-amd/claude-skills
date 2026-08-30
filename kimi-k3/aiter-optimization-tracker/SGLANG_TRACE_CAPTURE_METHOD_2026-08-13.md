# Canonical SGLang trace capture method — 2026-08-13

Use this method for future SGLang CPU/GPU Chrome traces on MI355X.

## Canonical implementation

```text
/workspace/kimi-k3-runs/stage2-runs/
  2026-08-13-triton36-vs37-compact-global-traces/
  run_compact_trace.sh
```

## Required behavior

```text
one independent trace file per TP rank
prefill and decode in the same rank trace
PyTorch cpu_op and GPU kernel events enabled
with_stack=false
record_shapes=false
CUDA Graph remains enabled
do not capture every request step
hard post-capture limit: each compressed rank trace <= 500 MiB
validate every gzip archive before accepting the capture
```

Do not merge TP ranks. Large merged or full-workload traces are difficult or
impossible to open in Perfetto.

## Capture window

For the validated 8192-input/128-output C32 workload:

```text
1. Launch the production-matched server without profiling.
2. Launch 32 requests at concurrency 32.
3. Observe TP0 prefill progress in the server log.
4. After 14 prefill batches, call POST /start_profile.
5. Keep the profiler active across the remaining prefill batches.
6. Wait until TP0 reports a decode batch.
7. Retain a two-second decode sample.
8. Call POST /stop_profile.
9. Verify eight rank traces and enforce the 500 MiB/rank limit.
```

This produces a representative late-prefill plus short-decode trace without
capturing the entire workload.

## Profiler configuration

```text
SGLANG_PROFILE_V2=0
SGLANG_PROFILE_WITH_STACK=false
SGLANG_PROFILE_RECORD_SHAPES=false
```

`/start_profile` payload:

```json
{
  "activities": ["CPU", "GPU"],
  "with_stack": false,
  "record_shapes": false,
  "profile_by_stage": false,
  "merge_profiles": false
}
```

Use one manual profiler session spanning the prefill-to-decode transition.
Do not use `profile_by_stage` for this purpose: stage-boundary profiler
reinitialization produced incomplete MoE GPU coverage.

## Validated output

The retained Triton 3.6/3.7 pair is:

```text
/workspace/kimi-k3-runs/stage2-runs/
  2026-08-13-triton36-vs37-compact-global-traces/
```

Observed sizes:

```text
Triton 3.6: 8.0-8.6 MiB per rank
Triton 3.7: 7.7-8.2 MiB per rank
```

Both versions retained representative PyTorch API, MoE and attention-residual
GPU events. All 16 gzip files passed integrity validation.

## Additional validated capture

The same method was adapted to a matched C64 8192/64 workload for Kimi-K3 FP8
KV with MLA Q/cache fusion enabled:

```text
/workspace/kimi-k3-runs/mla-q-cache-fp8-trace-2026-08-13/
```

It produced eight valid 18 MiB rank traces (139 MiB total), retained late
prefill plus C64 decode, and kept CUDA Graph enabled. See that directory's
`README.md` for the exact configuration and ATOM comparison trace paths.

## Retention

Keep the compact raw traces while they are under active inspection. After
analysis and user approval, retain the capture script, logs, summaries and a
deletion manifest; delete superseded raw traces.
