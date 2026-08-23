# SGLang versus ATOM C2/C64 trace attribution plan — 2026-08-22

## CONTINUE HERE

**Status:** Complete. The final decision is
[`SGLANG_ATOM_C2_C64_TRACE_ATTRIBUTION_2026-08-22.md`](SGLANG_ATOM_C2_C64_TRACE_ATTRIBUTION_2026-08-22.md).
All four selected decode cases passed 32/32 rank-trace gates; stop-after-wave
prefill scopes replaced the invalid staging-only selections. Armed C64 route
capture, matched-route current-kernel replay, and the full dense crossover
matrix are complete. Production remains unchanged.

**Next:** Select one corrected GPU kernel/op candidate, implement it
default-off and fail closed, rerun the same common-client manifest, then
recapture the same one complete prefill step and one complete decode replay.
Require the targeted GPU-step gap to close. Resume graph-external analysis only
if retraced SGLang GPU work is already ahead while endpoint performance does
not improve.

```text
/workspace/kimi-k3-runs/common-oai-sglang-atom-traces-2026-08-22/
  route-validation/sglang/        # complete, armed, 736 dumps
  route-validation/atom/          # complete, armed, 736 dumps
  route-validation-unarmed-invalid/
```

**Files:**

```text
/workspace/useful-scripts/benchmarking/common_oai_benchmark.py
/workspace/useful-scripts/benchmarking/kimi-k3/trace/
/workspace/useful-scripts/benchmarking/kimi-k3/analysis/
```

**Result root:**

```text
/workspace/kimi-k3-runs/common-oai-sglang-atom-traces-2026-08-22/
```

**Golden/pass criteria:**

```text
same AITER dc4bdf1c and FlyDSL 0.3.1 for all cases
same persisted C2/C64 manifests used by the common-client benchmarks
8 independent valid gzip traces per case
nonzero CPU op, user annotation and GPU kernel coverage
CUDA/HIP graphs remain enabled
each compressed rank trace <= 500 MiB
request/token contract passes
no cross-profiler absolute-time comparisons
```

Do not commit, push, or delete retained traces without an explicit request.

## 2026-08-23 execution agenda — completed

### 1. Resolve the routed-MoE inversion

**Complete:** armed routes are SGLang `133.87` versus ATOM `18.03` mean active
experts and `145.21` versus `33.91` BM32 blocks. Matched current-kernel full
chains favor A8W4 by `3.43%` and `7.44%`; the inversion is route workload.

Complete the armed ATOM C64 route dump, then compare against the completed
armed SGLang dump:

```text
unique active experts per layer
routes per expert
BM32 padded blocks
route concentration
same rank/layer/call alignment
```

Do not use the earlier unarmed route result; it may contain startup/dummy M64
calls and is retained only under `route-validation-unarmed-invalid/`.

Then replay current route distributions through the current kernels:

```text
A8W4 stage1 / stage2
A16W4 stage1 / stage2
complete fused_moe chain including activation quantization
```

Required comparisons:

```text
SGLang routes on A8W4 and A16W4
ATOM routes on A8W4 and A16W4
synthetic matched-active-expert controls
```

This separates route workload from kernel efficiency. Record that A16W4 uses
paired gate/up workgroups while A8W4 uses a different N-grid decomposition;
workgroup count alone is not arithmetic-work evidence.

### 2. Resolve dense-GEMM mechanism and crossover policy

**Complete:** 125/125 supported graph rows passed with seven explicit skips.
The layer-weighted combined prepared cost is `3.048269 GiB/GPU`; use separate
default-off candidates rather than a combined promotion. See
[`DENSE_CROSSOVER_FULL_2026-08-23.md`](DENSE_CROSSOVER_FULL_2026-08-23.md).

First produce a like-for-like operation inventory from the corrected caller
maps. At minimum:

```text
KDA fused input projection
KDA f_b and output projection
MLA q/kv-A, q-B, gate and output projections
merged MoE front N6016
shared-expert gate/up and down
routed latent down/up
attention output projections
```

For every op, record:

```text
SGLang implementation and dtype
ATOM implementation and dtype
input/output/weight shapes
quantization and dequantization chain
selected kernel/config at M2 and M64
graph compatibility and storage requirements
```

Important current policy difference:

```text
SGLang decode M64:
  routed experts: MXFP4/A8W4
  merged latent-down/front: BF16
  latent-up: BF16 because MXFP4 threshold is 2048

ATOM decode M64:
  routed experts: MXFP4/A16W4
  eligible dense projections: online PTPC-FP8
  latent down/up: MXFP4/A4W4 paths observed
```

Build a production-faithful graph-replay microbenchmark matrix:

```text
M = 2, 4, 8, 16, 32, 64
measure complete chains, not GEMM alone:
  input quant / RMSNorm quant
  GEMM
  output conversion / epilogue
```

Questions:

1. Does ATOM's FP8/MXFP4 dense path only win at high concurrency?
2. Which exact M bucket is the crossover for each projection?
3. Does the quantization launch/memory cost make it worse at C2?
4. Can SGLang keep both BF16 and quantized weights and select by M?
5. What capacity cost does dual storage add?

If a crossover exists, design a default-off fail-closed SGLang policy:

```text
small M -> current BF16/tuned path
large M -> selected FP8 or MXFP4 path
unsupported shape/layout/dtype -> BF16 fallback
fixed policy during CUDA graph capture/replay for each graph bucket
```

Validation before endpoint:

```text
numerical microbench
input-change graph replay
dispatch evidence
memory/capacity measurement
```

Endpoint gates:

```text
common-client five-round C2
common-client C64
full GSM8K-1319
no regression outside selected M buckets
```

### 3. Graph-external reconciliation — deferred

Do not make this the next action. Resume only if a candidate retrace shows
SGLang GPU kernels are already ahead while endpoint performance does not
improve. The corrected observations are:

```text
SGLang graph: 31.01 ms
ATOM graph:   28.64 ms
ATOM graph-internal advantage: 2.38 ms

SGLang endpoint TPOT: 31.74 ms
ATOM endpoint TPOT:   44.70 ms
ATOM-SGLang launch cadence delta: +1.765 ms
ATOM-SGLang endpoint TPOT delta:  +12.956 ms
ATOM cadence minus endpoint TPOT: approximately -12.21 ms
```

The approximately `12.21 ms` wave/request-level discrepancy is unresolved.
Do not assign a causal label. If the conditional gate above is met, use
host/API events around the selected replay to investigate:

```text
time between consecutive hipGraphLaunch calls
scheduler idle or queue handling
token result / D2H / detokenizer boundaries
collective completion waits
stream/event synchronization
first-token staggering effects
```

This is separate from dense/MoE kernel optimization. Do not attribute the
endpoint C64 TPOT gap to graph kernels when ATOM's graph is already shorter.

### 4. Finalize prefill attribution — complete

Use only the stop-after-wave SGLang captures. The earlier early-stop captures
contain profiler-export-stretched model annotations and are diagnostic only.

Prefill analysis rules:

```text
one complete full-model scope
C2: BS1 / 8192 tokens
C64: BS2 / 16384 tokens
host/GPU/correlation all present
profiler-observed composition only; do not equate to unprofiled TTFT
```

### 5. Secondary follow-ups

After the route, dense and graph-external questions:

```text
attribute the remaining current-AITER C2 gap (-0.86% throughput)
decide whether current AITER candidate should be retained or removed
reassess SGLang multi-stream priority:
  exact-C2 only
  preserve _add3
  fail closed before C64
```

The common-client results show SGLang already leads ATOM strongly at C2, so
multi-stream is lower priority than preserving/expanding SGLang's decode
advantages and understanding ATOM's high-concurrency scheduler behavior.

### 6. Deliverables and cleanup

Write the final report:

```text
/workspace/claude-skills/kimi-k3/aiter-optimization-tracker/
  SGLANG_ATOM_C2_C64_TRACE_ATTRIBUTION_2026-08-22.md
```

Update:

```text
SUMMARY.md
HANDOFF_2026-08-21.md
aiter-optimization-tracker/README.md
canonical experiment-history canvas (deferred by user; not edited)
```

Preserve all current artifacts. Raw traces, invalid early-stop traces, and
unarmed route diagnostics may be deleted only after conclusions are durable
and the user explicitly approves deletion.

## 2026-08-22 capture clarification

The trace campaign must preserve the best production decode mode. Do not set
either engine to eager execution to make profiling easier:

```text
SGLang: production FULL decode graphs
ATOM:   enforce_eager=false, production FULL graphs
```

Every accepted trace must include:

```text
PyTorch host/API events
user_annotation / record_function scopes
GPU kernel events
CPU-to-GPU correlation or flow metadata
stream identifiers
```

The campaign uses two complementary artifacts:

1. **production replay trace** — timing and critical-path source of truth;
2. **graph warmup/capture attribution trace** — maps graph kernels to the
   framework op/annotation that created them.

The attribution trace is diagnostic only. Its warmup forward may execute
kernels outside replay while constructing the production graph, but the
server's measured decode mode remains graphed. Never report warmup/capture
timings as production performance.

## 2026-08-22 single-step analysis rule

Do not sum every profiled prefill/decode step for the final attribution. Select
one representative step per phase and case:

```text
prefill:
  one full 16,384-token batch (2 requests x 8,192 tokens)

decode:
  one complete steady-state BS2 or BS64 graph replay
  choose the middle eligible step in the decode-only window
```

Eligibility for the decode step:

```text
all 2 or 64 requests have produced a first token
no prefill kernel or EXTEND annotation overlaps the step
graph batch size is exactly 2 or 64
request count is unchanged across the step
step is not the first or last profiler iteration
no request finishes or is retracted in the step
```

Retain the multi-step production trace only to locate this stable interval and
verify that the selected step is representative. Emit a `selected-step.json`
with:

```text
engine / case / rank
annotation and timestamp bounds
graph batch size
context/token metadata when available
kernel count and stream count
route-work metadata reference
selection reason
```

Use the matching single BS2/BS64 graph warmup forward for op-to-kernel mapping.
The main comparison artifacts are therefore:

```text
one prefill step per case
one decode step per case
one graph-capture warmup template per engine/batch size
```

For a fixed step shape and dispatch policy:

```text
projected phase time = representative per-step time * number of steps
```

Report the projection beside unprofiled endpoint time as a reconciliation
check. If it does not reconcile, inspect context-length drift, route
distribution, scheduler overlap, or graph-replay timing visibility rather than
averaging more profiler steps.

## Question to answer

The controlled common-client results are:

```text
                C2 total tok/s   C64 total tok/s
SGLang          1138.02          9800.69
ATOM             768.04          9846.50
ATOM delta       -32.51%           +0.47%

C2 median TPOT:  SGLang 14.90 ms, ATOM 22.62 ms
C64 median TPOT: SGLang 31.74 ms, ATOM 44.70 ms
```

The traces must determine:

1. which decode families explain SGLang's C2 advantage;
2. whether those exact families converge at C64;
3. whether different families instead let ATOM catch up;
4. how much of C64 convergence comes from overlap/scheduling rather than
   faster individual kernels;
5. whether active-expert distribution or padded MoE work differs despite the
   shared prompts.

## Controlled matrix

Run four independent server lifetimes:

```text
case         engine   concurrency   transition wave   decode sample wave
sglang-c2    SGLang       2         exact 8192/64     exact 8192/256
sglang-c64   SGLang      64         exact 8192/64     exact 8192/1024
atom-c2      ATOM          2         exact 8192/64     exact 8192/256
atom-c64     ATOM         64         exact 8192/64     exact 8192/1024
```

Use the short transition wave rather than 1024 output tokens:

- it contains enough decode replays for per-step medians;
- it keeps C64 traces compact;
- unprofiled 8192/1024 common-client results remain the performance source of
  truth;
- profiled throughput must not be reported as endpoint performance.

The matched decode-only window starts only after every request in the wave has
produced its first non-empty streamed token, then records two seconds while
requests remain active. C2 uses 256 output tokens. C64 uses the full 1024
output tokens because ATOM staggers first-token delivery across the wave; a
short output would let early requests finish before the 64th request enters
decode.

All four cases use:

```text
model:       /shared_nfs/models/Kimi-K3
TP:          8
AITER:       /sgl-workspace/aiter-atom-current @ dc4bdf1c
FlyDSL:      0.3.1
KV cache:    FP8
radix/prefix caching: disabled
common streaming /v1/completions request lifecycle
exact server-reported prompt and completion lengths
```

Engine profiles:

```text
SGLang:
  455b744a
  AITER prefill
  tuned Triton decode
  validated K3 flags
  up-only latent MXFP4
  FULL decode graphs

ATOM:
  27f8639b
  current Kimi-K3 recipe
  single-stream MoE (threshold=0)
  FULL graphs
```

ATOM single-stream is the primary trace because it is the exact profile used
for the common-client C2/C64 comparison. The measured C2 multi-stream gain is
only 1.73% and does not explain the 32.5% cross-engine gap. Capture an optional
fifth `atom-c2-multi` case only after the primary four cases if overlap itself
needs attribution.

Do not attempt to make the numerical recipes identical during this campaign.
The goal is to attribute the selected production profiles. Record precision
and backend differences explicitly in every summary.

## Prompt control

Reuse the exact common-client manifests:

```text
C2:
/workspace/kimi-k3-runs/common-oai-sglang-atom-c2-2026-08-21/
  prompt-manifest-c2-8192.jsonl.gz
logical prompt SHA-256:
  185482ba78f0c9d45ba867f8c7714eb95db7b28a181bcfc6929dee2f8ee7a188

C64:
/workspace/kimi-k3-runs/common-oai-sglang-atom-c64-2026-08-21/
  prompt-manifest-c64-8k.jsonl.gz
logical prompt SHA-256:
  e0cc6508eed7c019c221842d0fecf29e867a5e71ccca5021f253fca2174f4e8f
```

The trace driver should load the first 2 or 64 measured prompts from the
corresponding manifest and preserve their request IDs. Do not regenerate
random prompts.

## Capture method

### Primary: matched PyTorch CPU/GPU profiler

Use PyTorch profiler for both engines in the same image:

```text
activities: CPU + GPU
with_stack: false
record_shapes: false
profile_memory: false
one file per TP rank
do not merge ranks
CUDA/HIP graphs enabled
```

The primary run-phase trace must retain both host API and GPU activities.
`with_stack=false` and `record_shapes=false` reduce size but do not disable
CPU operators, record-function annotations, launch APIs, GPU kernels, or
correlation IDs.

SGLang:

```text
SGLANG_PROFILE_V2=0
SGLANG_PROFILE_WITH_STACK=false
SGLANG_PROFILE_RECORD_SHAPES=false
POST /start_profile with merge_profiles=false and profile_by_stage=false
```

ATOM:

```text
--torch-profiler-dir <case>/traces
ATOM_PROFILER_MORE=0
ATOM_ENABLE_DETAILED_ANNOTATION=1
POST /start_profile
POST /stop_profile
```

For each independent case:

1. start the server without active profiling;
2. verify effective args, revisions, AITER path and FlyDSL version;
3. run one unprofiled throwaway wave from the same manifest to warm dispatch
   and graph replay;
4. capture one compact transition trace using the canonical SGLang method:
   late prefill through two seconds of decode;
5. issue the longer decode sample wave without profiling;
6. when the common trace driver has observed the first non-empty streamed token
   from all 2 or 64 unique requests, start the profiler;
7. record exactly two seconds of production graph replay;
8. stop profiling while requests remain active, then let the wave finish;
9. stop the server and wait for VRAM release;
10. validate all rank traces before starting the next case.

For C2, where only one or two prefill batches exist, start the transition
profiler immediately before the short request wave. For C64, SGLang starts
after 14 prefill batches as in the canonical method. ATOM should use detailed
`prefill[]`/`decode[]` annotations or request-progress control to select an
equivalent late-prefill boundary; if that cannot be made deterministic, use
the decode-only trace for cross-engine timing and treat transition traces as
engine-local topology evidence.

### Graph caller-attribution artifacts

Production graph replay can expose one `hipGraphLaunch` host call for many GPU
kernels. To identify which framework op created each graph node, retain graph
construction evidence separately.

ATOM:

```text
--mark-trace
--torch-profiler-dir <case>/graph-capture
ATOM_PROFILER_MORE=0
```

ATOM emits one capture trace per batch/rank containing the warmup forward and
graph capture. Keep only the exact BS2 or BS64 capture files required for this
campaign.

SGLang:

```text
use the recorded canonical CPU/GPU trace method first
retain torch.compile/graph source for BS2 and BS64
retain record_function/user annotations that bracket graph construction
```

If SGLang run-phase traces cannot map replay kernels below `hipGraphLaunch`,
add a production graph-construction capture hook rather than switching decode
to eager. The hook must observe the normal warmup/capture forward and remain
disabled outside this diagnostic.

Generate an explicit caller map:

```text
framework op / annotation
  -> host launch or graph node
  -> GPU kernel name
  -> stream
  -> call count
```

Use profiler correlation IDs and flow events where available. When graph replay
does not preserve per-node host correlation, join runtime kernel names/counts
to the graph warmup/capture trace and generated graph source.

### Fallback: one matched ROCprofiler-SDK method

Before interpreting profiler totals, validate graph visibility:

```text
representative MoE/KDA/MLA replay kernels present
decode call counts scale with 64 output steps
all ranks expose comparable graph replay coverage
```

If either engine's PyTorch trace hides most graph replay kernels, stop the
cross-engine absolute-time comparison and recapture all four cases with one
identical ROCprofiler-SDK/rocprofv3 method. Never compare SGLang PyTorch times
against ATOM RTL/rocprof times.

## Trace validation gates

For every case:

```text
8 trace files exactly
gzip -t passes
each file <= 500 MiB compressed
nonzero kernel events
nonzero cpu_op events
nonzero user_annotation or gpu_user_annotation events
prefill and decode both visible
request wave completes exactly
no server disconnect, OOM or retraction
```

Coverage checks:

```text
expected K3 MoE layer replay families visible
KDA and MLA layer families visible
collective kernels visible
graph replay kernel counts materially exceed one warm/capture iteration
rank0 is not a special coverage outlier
CPU host/API events are present
CPU-to-GPU correlation/flow events are present
caller mapping covers the major KDA/MLA/MoE/collective families
```

If coverage fails, retain the failed trace only until the capture method is
fixed and the conclusion is documented.

## Analysis pipeline

Extend the existing analyzers rather than create one-off parsers:

```text
/workspace/useful-scripts/benchmarking/kimi-k3/analysis/
  analyze_chrome_trace.py
  kernel_families.py
```

Produce one summary JSON per rank and one aggregate per case. Aggregate with
rank median, min, max and coefficient of variation; do not report rank0 alone
as TP8 cost.

The rank aggregate describes the same selected step across TP ranks; it does
not aggregate multiple decode iterations.

### Required metrics

For each case:

```text
GPU kernel busy union
GPU kernel sum
capture span
active stream count
per-stream busy union
top kernel names by total duration and call count
top user annotations
rank imbalance
```

Per family:

```text
MoE stage1
MoE stage2
router / TopK
route sort / quant / scatter
latent down/up projections
shared-expert projections
KDA input projection
KDA recurrence + fused f_b/onorm
MLA Q/cache preparation
MLA decode stage1/stage2
attention residual / add3
RMSNorm / quant
TP all-reduce and other collectives
copies / materialization
unclassified
```

For each major kernel, also emit:

```text
calling framework op or user annotation
host launch API / graph replay owner
graph source location when available
correlation confidence: direct / graph-map / inferred
```

Normalize decode metrics as:

```text
duration in the selected decode step
duration per layer invocation in that step
calls in the selected decode step
busy-union share of the selected step span
```

Kernel sums can double-count concurrent streams. Use busy union and annotated
critical-path spans to decide whether an apparent family cost is hidden by
overlap.

## Difference decomposition

Build two comparison views.

### View A: engine gap at each concurrency

```text
delta_C2(family)  = ATOM_C2 - SGLang_C2
delta_C64(family) = ATOM_C64 - SGLang_C64
```

Rank families by their contribution to:

```text
C2 TPOT difference: +7.72 ms/token for ATOM
C64 TPOT difference: +12.96 ms/token for ATOM
```

This identifies where SGLang is faster at each point, but not why total
throughput converges.

### View B: concurrency scaling within each engine

```text
scale_SGLang(family) = SGLang_C64 / SGLang_C2
scale_ATOM(family)   = ATOM_C64 / ATOM_C2
```

Classify each family:

```text
catch-up:
  ATOM's relative disadvantage shrinks at C64

ATOM overtake:
  ATOM becomes faster in that family at C64

SGLang scaling loss:
  SGLang family cost grows faster with concurrency

overlap gain:
  kernel sum grows but busy-union/critical span does not

neutral:
  similar normalized scaling in both engines
```

The final report must distinguish:

```text
kernel became faster
less work was dispatched
launches were fused/removed
work overlapped on another stream
scheduler changed prefill/decode wave shape
route distribution changed padded expert work
```

## Route-distribution control

Historical traces showed identical MoE symbols but different active-expert
counts, enough to explain large stage1/stage2 timing gaps.

Collect route metadata in a separate, non-profiled diagnostic run using the
shared tensor-contract tooling:

```text
/workspace/useful-scripts/benchmarking/kimi-k3/contracts/
  tensor_contract_dump.py
  compare_tensor_contracts.py
```

For C2 and C64, record:

```text
unique active experts per layer/step
routes per expert
BM32 padded blocks
sorted-token counts
activation/weight dtype and layout
stage1/stage2 selected kernel names
```

Do not instrument route tensors during the timing trace; instrumentation can
change graph capture and scheduling.

## Execution sequence

### Phase 0 — preflight

```text
verify no running server/benchmark
verify 8 idle GPUs and baseline VRAM
record SHAs, package versions and effective configs
run common-client unit tests
run analyzer help/compile smoke
create result directories
```

### Phase 1 — capture four cases

```text
SGLang C2
SGLang C64
ATOM C2
ATOM C64
```

After each case:

```text
stop all processes
verify VRAM release
validate traces
write case runtime.json
write trace inventory with path/size/SHA-256
write production graph-mode evidence
write op-to-kernel caller map
```

### Phase 2 — summarize

```text
analyze all 32 rank traces
validate graph coverage
aggregate rank distributions
generate C2 and C64 engine comparisons
generate within-engine concurrency scaling
```

### Phase 3 — route contracts

```text
capture route metadata separately
quantify active/padded expert work
connect route differences to MoE trace duration
```

### Phase 4 — decision report

Write:

```text
/workspace/claude-skills/kimi-k3/aiter-optimization-tracker/
  SGLANG_ATOM_C2_C64_TRACE_ATTRIBUTION_2026-08-22.md
```

Update:

```text
SUMMARY.md
HANDOFF_2026-08-21.md
aiter-optimization-tracker/README.md
canvases/kimi-k3-experiment-history.canvas.tsx
```

## Stop gates

Stop the campaign and diagnose before proceeding if:

```text
either engine does not use current AITER dc4bdf1c
manifest hashes differ
request token contract fails
CUDA/HIP graph is disabled
fewer than 8 valid traces are produced
GPU replay coverage is missing or asymmetric
trace exceeds 500 MiB/rank
OOM, server disconnect or worker failure occurs
profiler changes kernel dispatch unexpectedly
```

Do not infer a cause from kernel names alone. A conclusion requires matched
call counts, normalized timing, selected kernel/config evidence, and route
work where relevant.

## Expected deliverables

```text
4 runtime manifests
4 trace inventories
32 per-rank summary JSON files
4 aggregate case summaries
2 engine-gap comparisons (C2, C64)
2 concurrency-scaling comparisons (SGLang, ATOM)
route-distribution comparison
one final decision report
updated canonical experiment-history canvas
```

Raw traces remain temporary working artifacts. After the investigation closes,
retain scripts, manifests, summary JSON/CSV and the decision report; delete raw
traces only with explicit user approval and a deletion manifest.
