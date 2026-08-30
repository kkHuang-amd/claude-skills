# Kimi-K3 new-environment start prompt

Use this prompt at the beginning of a new Cursor chat or after moving to a new
container/machine.

---

Continue the Kimi-K3 MI355X/gfx950 optimization project from the durable state
under:

```text
/workspace/claude-skills/kimi-k3
```

Do not restart the investigation from scratch. Treat the documents, canonical
canvases, remote branches, validation results, and rejected-experiment records
below as the source of truth.

## 1. Read these files first

Read in this order:

```text
README.md
SUMMARY.md
ARTIFACT_RETENTION.md
aiter-optimization-tracker/HANDOFF_2026-08-11.md
aiter-optimization-tracker/SGLANG_VENDOR_FLYDSL_2026-08-12.md
aiter-optimization-tracker/FRESH_INTEGRATION_2026-08-12.md
aiter-optimization-tracker/AITER_DEPENDENCY_MATRIX_2026-08-12.md
aiter-optimization-tracker/PAUSE_TRACK_2026-08-11.md
canvases/README.md
```

For portable/offline recovery also read:

```text
k3-handover-2026-08-12/HANDOVER.md
```

## 2. Restore the canonical canvases

From the Kimi-K3 workspace root, run:

```bash
python canvases/sync_canvases.py \
  --target /root/.cursor/projects/sgl-workspace/canvases \
  --prune-legacy
```

If the new Cursor workspace has a different managed canvas directory, replace
the `--target` path accordingly.

## 3. Canvas writing rules

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

## 4. Restore the code branches

Preferred remote clones:

```bash
git clone --branch perf/k3_opts_0812 \
  https://github.com/HaiShaw/sglang.git sglang-k3-opts-0812

git clone --branch integration/k3-core-only \
  https://github.com/kkHuang-amd/aiter.git aiter-mainline-k3-0812

git -C aiter-mainline-k3-0812 submodule update --init --recursive
```

Expected heads recorded at handover:

```text
SGLang f9dd3a0661b472d5fba1632adebcffc5c7c4021e
AITER  284a1eb401bb15f6368a68b34eb0cd693ee1fcd3
```

If the remote branches changed, compare their commits against these SHAs
before continuing.

Offline bundles are available at:

```text
k3-handover-2026-08-12/sglang-k3-opts-0812.bundle
k3-handover-2026-08-12/aiter-k3-integration.bundle
```

Do not commit generated files such as:

```text
aiter/jit/flydsl_cache/
git_rebase.sh
xxx
__pycache__/
```

## 5. Runtime and selected configuration

Validated runtime:

```text
Torch 2.9.1 + ROCm 7.2
HIP 7.2
Triton 3.6
TP8
Kimi-K3 BF16 checkpoint
```

Selected production environment:

```bash
export SGLANG_K3_FLYDSL_SOURCE=sglang
export SGLANG_K3_AITER_M16384_PROFILE=1
export SGLANG_USE_AITER=1
export SGLANG_AITER_K3_OPT=1
export AITER_FLYDSL_FORCE=1
export AITER_SITUV2_A8W4=1
export AITER_SITUV2_A4W4=0
export AITER_FLYDSL_STAGE1_SCRATCH_REUSE=1
export SGLANG_K3_FLYDSL_AR_NORM=1
export SGLANG_K3_KDA_FUSED_BACKEND=aiter
export SGLANG_K3_AITER_MLA_GATE=1
export SGLANG_K3_AITER_KDA_GROUP64=1
export SGLANG_K3_AITER_MOE_PREROUTE_FP8=0
export SGLANG_K3_AITER_LATENT_TAIL_FP8=0
export SGLANG_K3_AITER_B2_FUSIONS=0
```

Optional B2 profile:

```bash
export SGLANG_K3_AITER_MOE_PREROUTE_FP8=1
export SGLANG_K3_AITER_B2_FUSIONS=1
```

`SGLANG_K3_FLYDSL_SOURCE` supports:

```text
auto
sglang
aiter
```

## 6. Current validated results

```text
Vendored FlyDSL focused tests: 46 passed
GSM8K 50:                  1.000
GSM8K 200:                 0.990
Max token capacity:        933883

C2:   968.57 tok/s
C4:  1741.98 tok/s
C8:  2881.25 tok/s
C16: 4432.25 tok/s
C32: 6191.41 tok/s
```

Optional B2 result:

```text
C2:      968.57 -> 1054.19 tok/s
C2 TPOT:  17.67 ->   16.16 ms
C4:     1741.98 -> 1743.45 tok/s
```

Do not claim a new gain unless it is compared against this fresh, vendored
baseline using the same workload and warmup policy.

## 7. Experiments already completed

Read detailed decisions under:

```text
aiter-optimization-tracker/
stage2-runs/README.md
```

Accepted/retained:

```text
fused KDA + f_b
caller-provided MoE output
stage1 scratch reuse
MLA output gate
KDA group64
SGLang-vendored Kimi FlyDSL kernels
optional B2 profile
optional M16384 profile
```

Optional only:

```text
FP8 preroute/shared-down
FP8 latent tail
A4W4 C16 profile
```

Do not repeat without an architectural change:

```text
V3/V3-R role-grid + P23
V4 multi-CU persistent route prep
generic TILE_M M4/M8/M16
route sort+quant-only fusion
one-wave TopK+quant
forced all-reduce global/exact-B32
standalone #4572/#4577 endpoint paths
```

## 8. Pending optimization work

Highest-value unresolved areas:

```text
fixed tiny-kernel chains
copies and materialization
attention residual and KDA launch boundaries
route / sort / quant handoff
```

If route work resumes, choose one of:

```text
one-CTA LDS E896 sorter
stage1 ABI consuming route metadata and token-major scale directly
```

Before implementing another kernel, read:

```text
aiter-optimization-tracker/B300_MI355X_TRACE_COMPARISON_2026-08-11.md
aiter-optimization-tracker/PRODUCTION_TRACE_2026-08-11.md
aiter-optimization-tracker/PAUSE_TRACK_2026-08-11.md
```

Also complete:

```text
Track AITER #4617 and #4647 merges.
Rebase/drop the core-only AITER patches after upstream merge.
Run five-round paired C2 before making B2 a default profile.
Analyze retained B300 normal versus single-stream/no-PDL summaries.
Open/review the SGLang integration PR.
```

## 9. Artifact retention rules

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

The last cleanup removed 360 raw traces and reduced this workspace from about
25 GiB to about 619 MiB.

## 10. Working rules for the new agent

```text
Do not modify the golden repositories.
Use isolated branches/worktrees for experiments.
Keep each optimization independently gated and revertible.
Do not push, commit, or delete data unless explicitly requested.
Verify actual kernel dispatch with fresh JIT evidence.
Do not infer endpoint gains from isolated kernel benchmarks.
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
