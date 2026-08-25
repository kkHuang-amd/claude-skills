# Kimi-K3 development rules

Extracted verbatim from `start_prompt.md` §3, §9, §10 so an agent can load the
rules **without** paying for the project-state and history files.

Read this file in full. It is small and every line is a constraint.

---

## 1. Working rules

```text
Do not modify the golden repositories.
Use isolated branches/worktrees for experiments.
Keep each optimization independently gated and revertible.
Do not push, commit, or delete data unless explicitly requested.
Verify actual kernel dispatch with fresh JIT evidence.
Do not infer endpoint gains from isolated kernel benchmarks.
For SGLang traces, follow
aiter-optimization-tracker/SGLANG_TRACE_CAPTURE_METHOD_2026-08-13.md:
one unmerged file per TP rank, one manual profiler session spanning late
prefill into a short decode sample, no stacks/shapes, and <=500 MiB per rank.
Update SUMMARY.md, the relevant detailed report, and the canonical canvas after
each accepted or rejected experiment.
```

When resuming, first report:

```text
current SHAs
active feature flags
GPU availability
which pending item will be attempted
the exact acceptance and stop gates
```

---

## 2. Canvas writing rules

Canonical canvas source files live in:

```text
canvases/
```

Cursor's managed canvas folder is a generated working copy. Do not treat the
managed copy as the permanent source.

There are exactly three maintained topics:

```text
kimi-k3-optimization-scan.canvas.tsx
  PR inventory, upstream state, integration status, accepted/rejected work

kimi-k3-hardware-comparison.canvas.tsx
  B300 versus MI355X metrics, trace attribution and hardware/software gaps

kimi-k3-experiment-history.canvas.tsx
  experiment timeline, production metrics, feature policy and next work
```

When adding information:

```text
Use an existing topic canvas whenever the topic matches.
Update superseded metrics in place instead of appending duplicate tables.
Keep one row per PR, feature, experiment or milestone.
Use stable experiment/PR names so future updates can replace the same row.
Retain failed experiments only when they record a reusable decision.
Link detailed Markdown reports instead of embedding raw logs.
Include the source and update date inside the canvas.
Do not create date-suffixed canvases for an existing topic.
```

Only create a fourth canvas when the subject has a genuinely different
lifecycle and cannot fit any existing topic.

After editing a canonical canvas:

```text
1. Run canvases/sync_canvases.py.
2. Check Cursor Canvas TypeScript diagnostics.
3. Update canvases/README.md if a topic contract changes.
4. Add the refreshed canonical canvas to the next handover archive.
```

---

## 3. Artifact retention rules

Large run artifacts are physically stored at:

```text
/dockerx/var/amdsgl/kk/workspace/kimi-k3-runs/stage2-runs
```

The project-local `stage2-runs` path is a symlink to that directory. Continue
using project-local paths in reports and scripts.

Keep:

```text
summary Markdown
result JSONL/CSV
analysis JSON
scripts
small logs
test evidence
canonical canvases
git bundles/checksums
```

Raw `*.trace.json.gz` files are temporary:

```text
capture -> analyze -> summarize -> verify -> delete
```

Do not retain models, JIT caches, compiled extensions, Python environments, or
raw profiler traces under `stage2-runs/`.

Why this matters: the last cleanup removed 360 raw traces and reduced this
workspace from about 25 GiB to about 619 MiB.

Full detail: `ARTIFACT_RETENTION.md`.

---

## 4. Token discipline

Also apply the workspace-wide skills:

```text
../cap-tool-output/SKILL.md            per-command output caps
../reduce-conversation-usage/SKILL.md  session/doc strategy
```

Specific to this project: **never read `SUMMARY.md` in full** (36 KB, ~9k
tokens). List its sections and read only the one you need:

```bash
rg -n '^## ' SUMMARY.md              # section map with current line numbers
sed -n 'A,Bp' SUMMARY.md             # just that section
```

The same applies to `aiter-optimization-tracker/*.md` — those are dated
historical records, not state you need to resume work.
