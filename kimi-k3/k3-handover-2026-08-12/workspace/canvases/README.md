# Canonical Kimi-K3 canvases

These three files are the permanent source of truth:

```text
kimi-k3-optimization-scan.canvas.tsx
kimi-k3-hardware-comparison.canvas.tsx
kimi-k3-experiment-history.canvas.tsx
```

## Topic ownership

```text
optimization-scan
  PR inventory, integration state, selected/rejected work, upstream status

hardware-comparison
  B300 versus MI355X endpoint and trace attribution

experiment-history
  timeline, current production metrics, feature policy and next work
```

Do not create another canvas for these topics. Update the existing canonical
file in place using stable rows and experiment names.

## Update rules

Each update should:

```text
replace superseded metrics rather than append duplicates
retain milestone history only when it records a decision
include source/date text inside the canvas
link detailed Markdown reports instead of embedding raw logs
keep rejected experiments in the history canvas
```

## Install into Cursor

Cursor detects canvases only in its managed workspace directory. Treat that
directory as a generated copy:

```bash
python sync_canvases.py \
  --target /root/.cursor/projects/sgl-workspace/canvases \
  --prune-legacy
```

On a different machine, change `--target` to that workspace's Cursor canvas
directory.

The sync script copies only the three stable topic files. `--prune-legacy`
removes the six superseded 2026-08-10/11 canvas names.
