# SGLang prefill stop-after-wave recapture — 2026-08-22

## Decision

The SGLang-only C2/C64 recapture passed. Use the fresh canonical
`prefill-steps/sglang-c2` and `prefill-steps/sglang-c64` traces for raw
`sglang.vlm.language_model_prefill` scope analysis. The former captures are
preserved as `sglang-c2-early-stop` and `sglang-c64-early-stop`. ATOM was not
rerun.

The valid production-shape selection is the final complete occurrence in each
category, not the earlier profiler-stretched occurrence:

- C2 BS1/8192: CPU `user_annotation` occurrence 1 of 2; GPU
  `gpu_user_annotation` occurrence 1 of 2 on every rank.
- C64 BS2/16384: CPU occurrence 32 of 33; GPU occurrence 33 of 34 on every rank.

Selected C2 CPU scopes are 178.737–196.250 ms and GPU scopes are
468.157–468.158 ms with 3,782–3,783 kernels. Selected C64 CPU scopes are
169.909–185.823 ms and GPU scopes are 466.460–466.464 ms with 3,783–3,784
kernels. GPU scope ends are within 0.005 ms of the final overlapping kernel.
These are one model-forward-scale kernel set and are not the earlier 27–51 s
export-stretched scopes.

## Capture and gates

Both cases used profile summary mode `wave` (`profile_stop_after_wave=true`),
started profiling before the measured wave, and requested stop after the
`wave_completed` perf-counter timestamp:

- C2: wave 38.695031 s, profile 38.699339 s, stop lag 4.293836 ms; completion
  `2026-08-22T11:48:54.499503+00:00`, stop request
  `2026-08-22T11:48:54.503797+00:00`.
- C64: wave 52.652029 s, profile 52.658115 s, stop lag 6.075517 ms; completion
  `2026-08-22T11:57:45.717330+00:00`, stop request
  `2026-08-22T11:57:45.723405+00:00`.

Start/stop HTTP statuses were 200/200. FULL graph, exact manifest, request,
CPU/GPU category, export, annotation, correlation, trace-size, server-health,
and VRAM-release gates passed on ranks 0–7. Trace sizes were 15.96–16.85 MiB
per rank for C2 and 31.37–31.63 MiB per rank for C64, below 500 MiB. VRAM
returned to 283 MiB/GPU and exact rebuildable capture caches were removed.

Raw validation:
`/workspace/kimi-k3-runs/common-oai-sglang-atom-traces-2026-08-22/prefill-steps/SGLANG_PREFILL_SCOPE_VALIDATION.json`

## Commands

```bash
/workspace/useful-scripts/benchmarking/kimi-k3/trace/run_common_prefill_case.sh \
  sglang 2 \
  /workspace/kimi-k3-runs/common-oai-sglang-atom-traces-2026-08-22/prefill-steps/sglang-c2 \
  /workspace/kimi-k3-runs/common-oai-sglang-atom-c2-2026-08-21/prompt-manifest-c2-8192.jsonl.gz

/workspace/useful-scripts/benchmarking/kimi-k3/trace/run_common_prefill_case.sh \
  sglang 64 \
  /workspace/kimi-k3-runs/common-oai-sglang-atom-traces-2026-08-22/prefill-steps/sglang-c64 \
  /workspace/kimi-k3-runs/common-oai-sglang-atom-c64-2026-08-21/prompt-manifest-c64-8k.jsonl.gz
```

No engine source, package, commit, or raw trace was deleted or changed.
