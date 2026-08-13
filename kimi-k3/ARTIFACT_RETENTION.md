# Artifact retention policy

## Keep permanently

```text
README / HANDOFF / SUMMARY documents
benchmark scripts
result JSONL
CSV summaries
analyzer output JSON
filtered client/server logs
focused test logs
git bundles and checksums
canonical canvases
```

## Raw profiler traces

Raw `*.trace.json.gz` files are temporary working artifacts.

Default lifecycle:

```text
capture
analyze
write summary JSON/CSV and decision document
verify that conclusions can be reproduced from retained outputs
delete raw trace
```

Keep a raw trace only while:

```text
the investigation is unresolved
the analyzer is still being developed
the trace is the sole evidence for a correctness issue
```

For new SGLang captures, use
`aiter-optimization-tracker/SGLANG_TRACE_CAPTURE_METHOD_2026-08-13.md`.
Do not merge TP ranks; enforce a 500 MiB compressed limit per rank and capture
only representative late prefill plus short decode steps.

When deleting raw traces, create a manifest containing relative path and byte
size. The 2026-08-12 cleanup manifest is:

```text
RAW_TRACE_DELETION_2026-08-12.tsv
```

## Experiment run folders

Physical run storage:

```text
/dockerx/var/amdsgl/kk/workspace/kimi-k3-runs/stage2-runs
```

The project-local `stage2-runs` path is a symlink to this location. Keep using
the project-local path in reports and scripts so links remain stable.

Each run folder should retain:

```text
README or decision pointer
commands/scripts
small logs
result JSONL/CSV
summary JSON
```

Avoid copying models, JIT caches, compiled extensions, Python environments, or
raw traces into `stage2-runs/`.

## Reproducibility snapshots

`persistent-snapshots/` may contain large runtime archives. Keep only snapshots
that cannot be reconstructed from a pinned image, commit, or package version.
Review snapshots separately from trace cleanup.

## Periodic cleanup

Before machine/container migration:

```text
du -h --max-depth=2 .
verify no raw traces remain
refresh SUMMARY.md
refresh canonical canvases
create and verify handover bundles
```
