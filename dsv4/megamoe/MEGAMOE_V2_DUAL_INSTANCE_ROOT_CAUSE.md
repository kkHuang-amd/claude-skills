# MegaMoEV2 Dual-Instance SGLang Root Cause and Fix

Date: 2026-07-31  
Platform: 8× MI355X, DeepSeek-V4-Pro, DP8/EP8  
FlyDSL: PR #876 head `2f9da1f4`

## Summary

MegaMoEV2 dual MTPR serving used two operators:

- A: prefill, MTPR8192, compact path
- B: decode, MTPR128, fixed-slot path

Both operators were correct independently, could coexist in the same process,
and passed standalone A→B→A switching, CUDA graph replay, full barriers, and
no-barrier tests. The serving hang was caused by SGLang making two decisions
from rank-local metadata during continuous batching:

1. different ranks selected different operator instances in the same EP step;
2. after instance selection was fixed, ranks could still select different V2
   workload/config buckets.

The complete fix reuses metadata from SGLang's existing MLP-sync all-gather.
It adds no collective, CUDA synchronization, or mode-switch barrier.

## Symptom

The original dual-MTPR server:

- initialized A8192 and B128 successfully;
- completed standalone correctness;
- sometimes completed synchronized prefill→decode workloads;
- hung during 8192-input continuous batching, usually in warmup;
- left all GPUs at 100% utilization with no useful memory traffic;
- eventually triggered the 300-second scheduler watchdog.

Single MTPR8192 serving did not have this failure.

## Rejected hypotheses

The following were tested and rejected:

- generic multi-instance MORI/shared-heap collision;
- symmetric-address aliasing;
- capacity-specific A8192/B128 buffer overlap;
- incomplete CUDA drain between instances;
- missing NCCL or MORI barrier;
- MoE-only CUDA graph replay;
- deterministic config-selector or runtime autotune failure.

The standalone harness passed all of these:

```text
A→A→A
B→B→B
build A+B, run only A
build A+B, run only B
A1→A2→A1
B1→B2→B1
A8192→B128→A8192
B128→A8192→B128
A eager→B graph replay→A eager
cross-capacity switching without per-step barriers
```

A/B symmetric buffers, signal generations, P2P pointer tables, dispatch
tables, and cross-device barrier buffers had zero address overlap.

## Root cause 1: rank-local instance selection

SGLang DP schedulers independently choose a local `ForwardMode`. During the
same MLP/EP step, one rank can be `EXTEND` while another is `DECODE` or
`IDLE`.

The old adapter selected the instance from local mode:

```python
is_decode = forward_batch.forward_mode.is_decode()
selected_mtpr = 128 if is_decode else 8192
```

An env-gated, no-synchronization signature captured the failure:

```text
rank 0:
  mode=EXTEND
  global_has_extend=true
  selected_mtpr=8192

rank 1-7:
  mode=IDLE
  global_has_extend=true
  selected_mtpr=128
```

Rank 0 entered A's MORI rendezvous while ranks 1-7 entered B's independent
rendezvous. Each instance waited for all eight ranks, so neither could
complete.

The standalone harness was extended to replay this exact per-rank sequence:

1. all ranks execute A;
2. rank 0 executes A while ranks 1-7 execute B with idle dummy tokens;
3. all ranks execute A.

The control step completed. The first divergent A/B step timed out before any
rank completed its forward. This established instance divergence as causal.

## Fix 1: rank-global instance policy

SGLang already all-gathers an `is_extend_in_batch` bit during MLP sync:

```text
is_extend_in_batch = max(local_is_extend across DP ranks)
```

The adapter now uses that rank-identical value:

```text
if any rank is EXTEND:
    all ranks use A8192
else:
    all ranks use B128
```

Consequences:

- mixed prefill/decode/idle steps use A8192 on every rank;
- B128 is used only when every rank is decode or idle;
- idle ranks continue to participate with the existing one-token dummy input;
- both instances are prebuilt collectively in deterministic A→B order.

Dual MTPR is explicit opt-in:

```text
SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR=8192
SGLANG_AMD_FLYDSL_MEGA_DECODE_MTPR=128
```

`SGLANG_AMD_FLYDSL_MEGA_DECODE_MTPR=0` remains the safe default and uses the
single MTPR8192 instance.

## Root cause 2: rank-local workload/config bucket

Fix 1 removed instance divergence, but long 8192/1024 warmup could still hang.

MegaMoEV2 config selection uses a workload token count. The adapter read:

```python
get_dp_global_num_tokens()
```

In this SGLang configuration, MLP TP gather is disabled. By design,
`ScheduleBatch.global_num_tokens` is then replaced with a one-element local
list:

```text
rank 0: [8192]
rank 1: [0]
...
```

The all-gathered counts still existed temporarily inside
`MLPSyncBatchInfo`, but their maximum was discarded before `ForwardBatch`
reached the model. As a result, ranks using the same A instance could still
build different `MegaMoEV2Workload` buckets and launch incompatible kernel
configs.

## Fix 2: preserve the existing rank-global max token count

MLP sync now computes and carries:

```python
batch.global_max_num_tokens = max(mlp_sync_info.global_num_tokens)
```

The value is copied into `ForwardBatch`, and the FlyDSL adapter supplies it to:

```python
MegaMoEV2Workload.from_max_tokens(...)
```

This is metadata from the existing MLP-sync all-gather. No new communication
is introduced.

In the long mixed-serving validation:

- 163 eager mixed steps were recorded;
- every step had signatures from all eight ranks;
- all eight ranks had the same `global_max_num_tokens`;
- there were zero instance-selection mismatches;
- there were zero config-bucket mismatches.

## Code changes

SGLang integration commit:

```text
9fbdafdf8 feat: add FlyDSL MegaMoEV2 backend
```

Dual-instance fix:

```text
9d58457ba fix: synchronize FlyDSL dual-instance selection
```

Important files:

```text
python/sglang/srt/layers/moe/mega_moe_flydsl.py
python/sglang/srt/managers/scheduler_components/dp_attn.py
python/sglang/srt/managers/schedule_batch.py
python/sglang/srt/model_executor/forward_batch_info.py
python/sglang/srt/environ.py
test/manual/test_mega_moe_flydsl.py
```

The production commits do not contain debug `cuda.synchronize`, mode-switch
barriers, or profiler workarounds.

## Validation

Functional matrix:

```text
eager 8192/64, c256                 256/256 PASS
CUDA graph 8192/64, c256            256/256 PASS
CUDA graph 8192/1024 + signatures   256/256 PASS
CUDA graph 8192/1024 final         2048/2048 PASS
```

Final 2048-request result:

```text
Successful requests          2048/2048
Request throughput           2.99 req/s
Output throughput            3,061.18 tok/s
Total throughput            27,550.61 tok/s
Mean TTFT                    40,664.54 ms
Mean TPOT                        43.36 ms
Mean ITL                         44.41 ms
```

Matched refresh (`8192/1024`, c256, warmup512, measured2048):

```text
arm                         total tok/s  prefill/rank  decode/rank  step
DP                            31,358.52      6,906.53       895.00  35.75 ms
MegaMoEV1 3b0f818            29,831.76      7,300.63       790.65  40.47 ms
V2 e092035d                  warmup hang             no measured result
V2 2f9da1f4 single           28,041.60      6,371.86       819.44  39.05 ms
V2 2f9da1f4 dual 8192/128    27,550.61      6,366.34       804.26  39.79 ms
```

Dual is 1.75% slower than latest single V2. The global-policy fix resolves the
hang and enables full serving, but dual MTPR is not a performance win for this
matched workload.

## Trace

The fixed same-fresh-launch trace contains both paths:

- compact A8192 prefill;
- fixed-slot B128 CUDA-graph decode.

```text
/tmp/megamoe_dual_global_same_launch_trace/profiles/
  megamoe-dual-global-1785500393.9816146-TP-0-DP-0-EP-0.trace.json.gz
```

Kernel counts in the trace:

```text
compact Stage1/Stage2       122 / 122
fixed-slot Stage1/Stage2  11834 / 11834
```

Final serving logs:

```text
/tmp/megamoe_dual_global_full_final/megamoe/
```

## Operational guidance

- Keep decode MTPR at `0` unless dual MTPR is intentionally enabled.
- Do not select A/B from local `ForwardMode`.
- Do not derive V2 workload/config from local token counts.
- Any future instance or config key must be identical across the full EP
  group before entering MegaMoE.
- Avoid adding runtime collectives inside the operator or CUDA graph replay;
  extend the existing scheduler MLP-sync metadata instead.
- Profiler rank-skew is a separate diagnostic issue and is not part of this
  production fix.

## Post-fix correctness status

The dual-instance rendezvous fix resolves hangs, but it does not establish
model correctness. A later 20-question GSM8K gate found:

```text
MegaMoEV1 control                accuracy 0.850, invalid 0.000
MegaMoEV2 single eager           accuracy 0.000, invalid 1.000
MegaMoEV2 dual graph 8192/128    accuracy 0.350, invalid 0.100
MegaMoEV2 dual eager 8192/128    accuracy 0.400, invalid 0.150
```

Single V2 remains broken without CUDA graph, and increasing max generation
from 2048 to 8192 does not recover valid answers. The performance and
stability results in this document are therefore diagnostic; MegaMoEV2 must
not be promoted until the single-instance end-to-end accuracy regression is
resolved.

### Accuracy resolution

The bad runs used stale `2f9da1f4` sources with a 0.2.4 native runtime.
Remote `mega_moe_v1` was force-updated to `567b43e` with merged PR #876 and
follow-up fixes. After rebuilding the matching native extension, single/dual
20-question accuracy returned to 0.85-0.95 with zero invalid outputs.

A second long-run graph issue remained: local decode graph buckets captured
different MegaMoEV2 configs on different DP ranks. The operator now separates
actual `run_tokens` from rank-identical `config_tokens`; every decode graph
captures config bucket128.

Full 1319-question results after both corrections:

```text
single eager   accuracy 0.925, invalid 0.001
dual eager     accuracy 0.924, invalid 0.001
single graph   accuracy 0.923, invalid 0.000
dual graph     accuracy 0.920, invalid 0.001
```

This supersedes the blocker statement above. Correctness is restored on the
rebuilt latest runtime; performance still requires a fresh matched rerun.
