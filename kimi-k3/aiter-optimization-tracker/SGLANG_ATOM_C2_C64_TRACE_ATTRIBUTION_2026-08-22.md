# SGLang/ATOM C2/C64 trace attribution — final decision

Date: 2026-08-22, finalized 2026-08-23.

## Decision

This campaign identifies actionable GPU work, but it does not justify a
production change. Production remains unchanged: AITER prefill, tuned Triton
decode, the retained SGLang K3 profile, and current default-off policies.
Profile and microbenchmark results alone are not promotion evidence.

The next strategy is deliberately GPU-first:

1. select one kernel/operation candidate at a time from the corrected
   single-step attribution and dense crossover evidence;
2. implement it default-off and fail closed;
3. rerun the same common-client workload and persisted prompt manifest;
4. recapture the same one complete prefill GPU step and one complete decode
   graph replay step;
5. require the targeted GPU step gap to close before endpoint/correctness
   promotion gates.

Graph-external analysis is deferred. Resume it only if a retrace shows SGLang's
GPU kernels are already ahead while matched endpoint performance does not
improve.

Confidence:

- **High:** selected decode replay spans/counts, corrected SGLang BS mapping,
  armed route workload, matched-route current-kernel result, and graph-replay
  dense crossover/storage.
- **Medium:** operation labels derived from exact symbol/count graph maps and
  known Kimi-K3 shapes.
- **Composition only:** profiled prefill scopes. Profiler perturbation prevents
  an absolute prefill latency or TTFT-ratio claim.

## Controlled stack and workload

```text
SGLang: 455b744aa77b2078de7577619dc12d2775fc1091
ATOM:   27f8639bad0948755630236aa7796b4b21348668
AITER:  dc4bdf1c142181ad90b7f6948564126df4c05fde
FlyDSL: 0.3.1
GPU:    8x AMD Instinct MI355X / gfx950
model:  /shared_nfs/models/Kimi-K3
TP:     8
KV:     FP8
graphs: production FULL mode
```

SGLang used AITER prefill, tuned Triton decode, the validated K3 flags,
up-only latent MXFP4, and A8W4 routed MoE. ATOM used its current Kimi-K3
recipe, PTPC-FP8 online quantization, A16W4 routed MoE, and single-stream MoE
(`ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0`). The numerical recipes are
intentionally production-profile comparisons, not identical-precision tests.

Common workload: streaming `/v1/completions`, exact 8,192 input and 1,024
output tokens, saturated request rate, exact server token checks, and required
SSE completion. C2 used five rounds per engine, four warmups and 16 measured
requests per round; all 80/80 requests per engine succeeded. C64 used 128
warmups and 512 measured requests; all 512/512 per engine succeeded. Common
client tests passed 2/2.

Persisted manifests:

```text
C2: /workspace/kimi-k3-runs/common-oai-sglang-atom-c2-2026-08-21/
      prompt-manifest-c2-8192.jsonl.gz
    logical prompt SHA-256:
      185482ba78f0c9d45ba867f8c7714eb95db7b28a181bcfc6929dee2f8ee7a188

C64: /workspace/kimi-k3-runs/common-oai-sglang-atom-c64-2026-08-21/
       prompt-manifest-c64-8k.jsonl.gz
     prompt-content SHA-256:
       e0cc6508eed7c019c221842d0fecf29e867a5e71ccca5021f253fca2174f4e8f
```

## Method and validation

The timing unit is one complete full-model prefill GPU scope and one complete
steady-state decode graph replay per case and TP rank. Rank summaries use the
TP8 median, with min/max/CV retained. Decode caller maps use exact kernel
symbols/counts and graph-construction warmups. The corrected SGLang mapping is
GPU occurrence 1 for BS64 (raw `M=64`) and occurrence 3 for BS2 (`M=2`);
69-layer KDA and 24-layer MLA call patterns disambiguate reused dense symbols.

All 32 selected decode traces (four cases x eight ranks) passed exact graph
batch, directly correlated replay, no-prefill-overlap, nonzero-kernel, selected
summary, and unchanged-raw-trace gates. Every rank in a case has the same
kernel count and one replay stream:

```text
SGLang C2/C64: 1,941 / 2,034 kernels
ATOM C2/C64:   2,796 / 2,796 kernels
```

Prefill uses only the stop-after-wave captures. All eight ranks per SGLang case
passed full-model scope, trace-size, complete-kernel-set, and closing-boundary
gates. The selected complete GPU occurrences are C2 occurrence 1/2 and C64
occurrence 33/34. The earlier early-stop captures are diagnostic only.

Operation durations are exclusive kernel sums inside one selected scope.
Although these accepted replay steps expose one active stream, family sums are
not asserted to be a critical path. SGLang's coarse outer prefill annotation
also leaves lower-confidence `other_dense_gemms` and `other` semantics.

## Complete prefill GPU-step composition

The selected scopes are one full-model step: C2 is BS1/8,192 input tokens and
C64 is BS2/16,384. The profiler-observed spans (SGLang about 468 ms; ATOM about
1,049/1,210 ms) are strongly profiler-perturbed and are not endpoint latency
measurements. No prefill-span ratio, annotation-to-TTFT residual, or TTFT ratio
is claimed. Use only operation composition, call counts, and candidate
direction.

Largest SGLang groups are stable across its selected C2/C64 steps:

```text
collectives              117.336 / 115.247 ms, 187 calls
other_dense_gemms         113.920 / 113.652 ms, 583 calls
other                      61.458 /  61.848 ms, 872 calls
routed_moe_stage1          54.583 /  54.395 ms,  92 calls
attention_residual_add3    50.061 /  50.099 ms, 278 calls
routed_moe_stage2          33.804 /  33.988 ms,  92 calls
kda_recurrence_f_b         20.196 /  20.304 ms, 483 calls
```

ATOM's largest profiled C2 groups are collectives (145.096 ms), routed MoE
stage2/stage1 (64.771/64.517 ms), and `other` (63.675 ms). At C64 they are
collectives (246.742 ms), routed MoE stage1/stage2 (148.105/129.967 ms),
`other` (119.812 ms), and attention residual/add3 (81.550 ms). These values
identify composition and scaling candidates only; route concentration differs
and profiler overhead prevents a causal endpoint prefill comparison.

Actionable prefill candidates therefore require an engine-local, default-off
implementation and same-step retrace. Do not optimize from the apparent
profiled span ratio.

## Complete decode GPU replay gaps

### C2: ATOM graph is 7.989 ms longer

```text
SGLang graph span: 15.578 ms
ATOM graph span:   23.567 ms
ATOM gap:          +7.989 ms
endpoint TPOT:     14.902 -> 22.617 ms (+7.715 ms)
```

Corrected largest signed operation contributors (ATOM minus SGLang):

```text
routed_moe_stage2    +1.665 ms
collectives           +1.206 ms
moe_front_merged      +1.173 ms
shared_expert_down    +0.705 ms
copies                +0.678 ms
mla_decode            -0.640 ms
```

This is the strongest direct target list: the endpoint TPOT difference closely
tracks the graph difference, so C2 candidate work should start with complete
MoE/dense chains and collectives rather than scheduler speculation.

### C64: ATOM graph is 2.377 ms shorter

```text
SGLang graph span: 31.012 ms
ATOM graph span:   28.635 ms
ATOM gap:          -2.377 ms
endpoint TPOT:     31.739 -> 44.695 ms (+12.956 ms)
```

Corrected largest signed operation deltas:

```text
routed_moe_stage1     -2.519 ms
kda_inproj            -1.430 ms
mla_decode            -0.934 ms
collectives           +0.691 ms
other                 -0.676 ms
copies                +0.641 ms
shared_expert_down    +0.480 ms
other_dense_gemms     +0.454 ms
```

The apparent routed-MoE advantage is mostly different route work, not A16W4
kernel superiority. The remaining actionable C64 GPU candidates are KDA input
projection, MLA decode/dense boundaries, copies, and collectives, each subject
to matched implementation and retrace gates.

The earlier claim that `other_dense_gemms` explained `-7.08 ms` is withdrawn.
It came from reversed SGLang warmup occurrence labels. Corrected dense semantics
leave only `+0.454 ms` in that residual bucket.

## Routed MoE correction

The armed eager contract capture used the exact C64 manifest. SGLang A8W4 and
ATOM A16W4 each completed 64/64 requests and produced exactly 92 post-arm
`[64,16]` calls on all eight ranks. All 1,472 dumps are after their arm
boundary, valid JSON, 1,024 routes, and shape-correct; 736/736 cross-engine
rank/call pairs align exactly.

```text
                         SGLang       ATOM
mean active experts      133.87       18.03
mean BM32 blocks         145.21       33.91
routes/active mean         8.50       63.36
```

ATOM selects exactly 16 active experts in 91/92 calls. SGLang spreads the same
1,024 routes across many more experts, producing much more BM32-padded work.

Current-AITER HIP graph replay (10 warmups/100 iterations, summed 92-layer
p50) reverses the raw trace impression:

```text
routes    A8W4 full chain   A16W4 full chain   A16 penalty
SGLang       8.171 ms          8.452 ms          +3.43%
ATOM         4.765 ms          5.119 ms          +7.44%
```

All 1,104 benchmark rows were OK, all 368 full-chain finite/input-change gates
passed, all 736 stage rows were OK, and all 184 A8/A16 numerical comparisons
passed. Therefore the C64 trace inversion is a workload effect; it is not
evidence that A16W4 is intrinsically superior.

A16W4 pairs gate/up workgroups, while A8W4 uses a different N-grid
decomposition. Workgroup count describes launch topology, not arithmetic work,
active-expert work, or a comparable kernel-efficiency denominator. The earlier
unarmed route capture is rejected and retained only as diagnostic evidence.

## Dense mechanisms, crossovers, and storage

The production-faithful matrix measured complete graph-replayed chains
(input/RMSNorm quantization, GEMM, and output conversion/epilogue), not GEMM
alone. It covered M=2/4/8/16/32/64 on current AITER: 132 rows, 125 supported
and passing, seven explicit skips; all 125 supported rows passed finite timing,
finite output, numerical comparison, alternate-input freshness, dispatch, and
layout gates.

Corrected BF16-relative crossover summary:

```text
kda_inproj MXFP4:       wins M32-M64 (1.044x, 1.455x)
latent_up MXFP4:        wins M2-M64
latent_up PTPC FP8:     wins M2-M8 and M32-M64
latent_up RMS+MXFP4:    wins M2-M64
merged_front MXFP4:     wins M64 only
merged_front PTPC FP8:  wins M16-M64
mla_qkv_a PTPC FP8:     wins M64 only (1.049x)
shared_down PTPC FP8:   wins M8 only (1.007x; noise-sized)
kda_mla_output:         no quantized winner
mla_gate:               no quantized winner
mla_qkv_a MXFP4:        no winner
shared_down MXFP4:      no winner
```

PTPC `kda_inproj` is unsupported. PTPC `latent_up` and `merged_front` use
generic null-config fallback rather than tuned ATOM-equivalent-family dispatch.
The M4 BF16 merged-front row is an explicit invalid-tile skip.

Retaining BF16 plus all three fastest prepared families would cost exactly:

```text
latent_up MXFP4, 92 layers:       1,255,604,224 B = 1.169373 GiB/GPU
kda_inproj MXFP4, 69 layers:      1,653,915,648 B = 1.540329 GiB/GPU
mla_qkv_a PTPC FP8, 24 layers:      363,534,336 B = 0.338568 GiB/GPU
total incremental prepared:       3,273,054,208 B = 3.048269 GiB/GPU
```

This is layer-weighted storage. The former 50.320557 MiB is withdrawn as a
model cost; it was only one representative shape per selected family.
Applying the existing latent-up bytes/token calibration estimates a 372,122
token loss and about 1,147,583 tokens remaining from 1,519,705, but this is not
a server measurement. Allocator behavior and the bytes/token slope may differ
for KDA/MLA prepared tensors.

Do not promote the combined 3.048269-GiB policy. Evaluate separate default-off
candidates:

1. **Rejected after retrace:** MLA QKV-A PTPC at M64
   (`0.338568 GiB/GPU` prepared estimate). The production TP8 complete chain
   regressed and the implementation was removed before endpoint gates.
2. **Next separate candidate:** KDA input-projection MXFP4 at M32+,
   with its own `1.540329 GiB/GPU` capacity measurement and gates.
3. **Latent-up:** treat this as a reference to existing endpoint evidence, not
   a new trace-only promotion. The retained up-only and M32-M256 decode reports
   already contain endpoint/correctness/capacity results; any revised threshold
   must be tested independently.

Do not bundle candidates before each closes its targeted GPU gap and passes
capacity, endpoint, and correctness gates.

The first candidate used verified fused N=2112
(1536 Q-LoRA + 512 KV-LoRA + 64 RoPE), not N=2304. Production TP8 retrace
measured `370.099 us` BF16 versus `272.504 us` FP8 GEMM plus `103.966 us`
online activation quantization: `376.507 us` complete, `+6.007 us` slower.
Although the total graph moved `31.012 -> 30.566 ms`, rank and operation
decomposition attributes that movement to routed-MoE/collective/route-work
variation rather than QKV-A. The targeted chain did not close, so the candidate
was rejected and removed before capacity, endpoint or accuracy gates.
Same-profile prefill contains zero candidate markers, confirming the decode-only
guard; prefill spans remain composition-only. Production is unchanged. Details:
[`MLA_QKVA_PTPC_M64_2026-08-23.md`](MLA_QKVA_PTPC_M64_2026-08-23.md).

## Withdrawn or rejected claims

- Withdrawn: SGLang prefill 2–5 ms staging-only selections and all ratios based
  on them. Only complete `sglang.vlm.language_model_prefill` GPU scopes remain.
- Withdrawn: any profiled prefill span or per-token value as an unprofiled TTFT
  ratio or latency claim.
- Withdrawn: C64 `other_dense_gemms -7.08 ms`; corrected residual is
  `+0.454 ms` ATOM minus SGLang.
- Rejected: raw routed workgroup counts as proof of A16W4 kernel superiority.
- Rejected: the unarmed route dump as current-wave evidence.
- Rejected and removed: MLA QKV-A PTPC M64; required online quantization makes
  its production TP8 complete chain `6.007 us` slower than BF16.
- Not claimed: profile/micro results justify production promotion.

## Next experiment gates

For each candidate, independently:

1. implement default-off with exact architecture, shape, dtype/layout, graph
   bucket, and fail-closed BF16/current-path guards;
2. pass focused dispatch/fallback and numerical tests, then input-change HIP
   graph replay including all conversion work;
3. measure real prepared-weight capacity;
4. rerun the same common-client C2 and C64 manifests with all unrelated flags
   fixed;
5. recapture the same one complete prefill GPU step and one complete decode GPU
   replay step;
6. require the targeted operation and total GPU-step gap to move in the
   predicted direction;
7. then run the required fixed 8K/1K and 68K/350 endpoint matrix plus paired
   GSM8K and long-context output/logprob checks.

Only if SGLang's retraced GPU steps are already ahead while endpoint E2E/TPOT
does not improve should the deferred graph-external work below become active.

## Deferred appendix: graph-external reconciliation

This is a caveat and future fallback, not the primary explanation or immediate
task.

Production launch-cadence analysis found:

```text
C2 ATOM-SGLang:
  graph span +7.989 ms; launch cadence +9.083 ms; endpoint TPOT +7.715 ms

C64 ATOM-SGLang:
  graph span -2.377 ms; launch cadence +1.765 ms; endpoint TPOT +12.956 ms
```

For ATOM C64, launch cadence is 32.485 ms, graph span 28.635 ms, and endpoint
TPOT 44.695 ms. Cadence minus endpoint TPOT is approximately `-12.210 ms`.
That wave/request-level discrepancy remains unresolved. It must not be assigned
to scheduler idle, D2H, detokenization, synchronization, or any other cause
without a matched observation. Host categories are clipped unions/sums that may
nest or overlap, and launch cadence is a CPU timestamp domain while graph span
is correlated GPU activity.

## Evidence

```text
Canonical trace evidence:
/workspace/kimi-k3-runs/common-oai-sglang-atom-traces-2026-08-22/
  TRACE_ATTRIBUTION_DRAFT.md
  selected-step-inventory.json
  prefill-steps/SGLANG_PREFILL_SCOPE_VALIDATION.json
  graph-external/graph-external-summary.md
  graph-external/comparisons/graph-external-c2-c64.json
  route-validation/

Route decision:
/workspace/claude-skills/kimi-k3/aiter-optimization-tracker/
  CURRENT_ROUTE_DIAGNOSTIC_RESULTS_2026-08-22.md
  MATCHED_ROUTE_CURRENT_KERNEL_2026-08-23.md
/workspace/kimi-k3-runs/matched-route-current-kernel-2026-08-23/

Dense decision:
/workspace/claude-skills/kimi-k3/aiter-optimization-tracker/
  DENSE_CROSSOVER_FULL_2026-08-23.md
/workspace/kimi-k3-runs/dense-crossover-2026-08-23/full/
```

Raw traces, early-stop diagnostics, and unarmed route evidence remain
preserved. No canvas, server, GPU, package, commit, push, or trace-deletion
action is part of this decision.
