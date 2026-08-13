# Kimi-K3 optimization workspace

This directory is the durable home for Kimi-K3 optimization results, migration
state, experiment summaries, and canonical Cursor canvases.

## Start here

```text
start_prompt.md
SUMMARY.md
aiter-optimization-tracker/HANDOFF_2026-08-11.md
aiter-optimization-tracker/SGLANG_VENDOR_FLYDSL_2026-08-12.md
aiter-optimization-tracker/PR34490_RADIX4_RESULTS_2026-08-12.md
aiter-optimization-tracker/SGLANG_TRACE_CAPTURE_METHOD_2026-08-13.md
k3-handover-2026-08-12/HANDOVER.md
```

## Directory layout

```text
aiter-optimization-tracker/   canonical reports and decisions
stage2-runs/                  symlink to external benchmark/run storage
persistent-snapshots/         reproducibility snapshot and Triton runtime
k3-handover-2026-08-12/      portable git bundles and handover docs
canvases/                     canonical topic canvases
legacy/                       archived pre-2026-08-10 documents/canvases
```

New results should be added to `aiter-optimization-tracker/`, not as another
top-level handover file.

Large run artifacts are physically stored under:

```text
/dockerx/var/amdsgl/kk/workspace/kimi-k3-runs/stage2-runs
```

The local `stage2-runs` symlink preserves canonical document and script paths.

## Artifact policy

See [`ARTIFACT_RETENTION.md`](ARTIFACT_RETENTION.md).

Raw `*.trace.json.gz` files were removed on 2026-08-12 after their summaries
and analysis outputs were retained. The deletion inventory is:

```text
RAW_TRACE_DELETION_2026-08-12.tsv
```

## Canvas policy

The canonical canvases live in `canvases/` and use stable topic names. Update
those files in place. Cursor's managed canvas directory is treated as a
generated working copy.

See:

```text
canvases/README.md
canvases/sync_canvases.py
```
