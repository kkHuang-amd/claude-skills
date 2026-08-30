# DSV4 DP-TBO vs FlyDSL-EP-TBO performance gap — HANDOVER (2026-07-24)

> **Goal complete (2026-07-30).** Final report:
> `FLYDSL_A2A_FINAL_REPORT_2026-07-30.md`. New performance work moves to
> MegaMoE PR #876 using `megamoe/MEGAMOE_PR876_NEXT_CHAT.md`.

## CONTINUE HERE (2026-07-29 final update)

The eager/TBO prefill recv-cap gap is fixed. FlyDSL now computes one shared
logical cap per TBO child from all-rank padded dispatch rows, uses that exact
cap for dispatch encoding, Aiter views, and combine decoding, and keeps physical
shmem unchanged.

Two counterbalanced cap-OFF/ON canonical pairs:

```text
cap OFF mean                         32,737.22 tok/s
cap ON mean                          33,751.42 tok/s
paired throughput gain                   +3.098%
paired TPOT improvement                  +2.225%
paired TTFT improvement                  +5.442%
```

Both pairs are positive (+3.184%, +3.012%). Eager shared cap is now default-on
for DSV4 FlyDSL TBO; explicit rollback remains:

```text
SGLANG_FLYDSL_DYNAMIC_RECV_CAP_EAGER=0
```

The backend comparison has now been repeated with six new counterbalanced
fresh-server pairs:

```text
DP-TBO mean                         33,138.46 tok/s
FlyDSL-TBO eager-cap mean          33,772.24 tok/s
FlyDSL paired advantage                 1.9130%
95% paired CI                    [1.4943%, 2.3317%]
```

All six pairs show FlyDSL ahead. The previous 1.44% DP advantage is closed and
reversed for the fixed 8k/1k c256 workload.

The EP GEMM1 fused-A2 optimization is now promoted and integrated as the
production default. Two counterbalanced EP OFF/ON canonical pairs:

```text
fused-A2 OFF mean                   33,741.99 tok/s
fused-A2 ON mean                    34,664.82 tok/s
paired throughput gain                  +2.735%
paired TPOT improvement                 +1.944%
paired TTFT improvement                 +5.254%
```

The current EP point estimate is 4.61% above the earlier six-pair DP mean, but
that is not a new interleaved DP/EP experiment.

Retain:

```text
GPU_MAX_HW_QUEUES=5
dispatch blocks=80
combine blocks=32
dedicated comm stream=enabled
eager shared recv cap=enabled
FP8-output GEMM1 fused A2=enabled
```

Rollback fused A2 before a fresh launch:

```text
AITER_FLYDSL_EP_NO_FAKE_EXPERT=0
```

Three A2A overlap-friendly kernel experiments were correctness-tested but
rejected by production gates and removed from production code:

- combine cross-PE waiters 256→8: short +0.181%, below +0.3%;
- yielding remote polls (64/256): -0.147% / +0.106%;
- split dispatch data/coordinator kernels: -0.328%.

Do not retry split threshold 0, comm priority -1, aiter-only recv slicing,
wrapped serving PMC/ATT, completed geometry/HW-queue screen, or the three
rejected kernel experiments above without new evidence.

## Objective

Explain and reduce the remaining TBO performance gap on DeepSeek-V4-Pro:

```text
DP-TBO       33,076.53 tok/s
FlyDSL-TBO   32,368.95 tok/s
gap          2.14%
```

Do not re-investigate FlyDSL correctness, dynamic recv, deadlock, or basic TBO
stream wiring unless new evidence contradicts the validated state below.

## Environment

- hardware: 8x MI355X gfx950
- model: `/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro`
- serving: TP8/DP8, DP attention, A8W4, FP8 KV
- workload: fixed 8192 input / 1024 output, concurrency 256
- harness: 2048 prompts, 512 warmups, `sglang-oai`, request-rate infinity
- mem fraction 0.90, CUDA graph ON

Repositories:

- SGLang: `/sgl-workspace/sglang-flydsl-a2a`
- branch: `feat/flydsl-a2a`
- pinned aiter: `/sgl-workspace/aiter`, commit `9127c94a1`
- FlyDSL runtime: `/sgl-workspace/flydsl-0.2.4-py`

Relevant commits:

- `807a16187` — FlyDSL A2A integration
- `15a76b3f9` — vendored A2A kernels
- `394e8955f` — FlyDSL TBO comm stream
- `76e162096` — FlyDSL TBO block64 tuning
- `9b0eb8a55` — PrefillDelayer mixed-slot regression fix

## Current comparison matrix

All delayer-ON rows use the fixed mixed-slot guard.

| Backend | TBO | Delayer | Total tok/s | TPOT | TTFT | GSM8K |
|---|:---:|:---:|---:|---:|---:|---:|
| DP | off | on | 30,895.54 | 55.80 ms | 19.00 s | 0.943 |
| DP | on | on | **33,076.53** | **52.93 ms** | **16.91 s** | — |
| FlyDSL-EP | off | on | 31,227.35 | 55.87 ms | 18.17 s | 0.931 |
| FlyDSL-EP | on | off | 32,087.20 | 53.91 ms | 18.27 s | 0.940 |
| FlyDSL-EP | on | on | **32,368.95** | 54.36 ms | **17.30 s** | 0.936 |

TBO uplift:

- DP: +7.1% throughput, -5.1% TPOT, -11.0% TTFT
- FlyDSL-EP: +3.7% throughput, -2.7% TPOT, -4.8% TTFT

Important: FlyDSL no-TBO is already faster than DP no-TBO. The problem is that
FlyDSL realizes less incremental benefit from TBO.

## Validated FlyDSL TBO implementation

DSV4 TBO is prefill-only. Decode remains the normal CUDA-graph path.

FlyDSL dispatcher under TBO:

- one shared dedicated comm stream for both inner dispatchers
- ready/done events
- compute-stream waits
- event + Python-reference lifetime
- no `record_stream(comm)` deferred-free behavior

The original FlyDSL TBO implementation did not use `async_finish`; all kernels
ran on the compute stream and overlap was zero. The comm-stream fix increased
throughput from 28,931 to 30,406 tok/s.

The default A2A grid then caused CU/cache contention. TBO block sweep:

| Grid | Total tok/s | TPOT |
|---|---:|---:|
| tuning default: dispatch128/combine256 | 30,406 | 56.03 ms |
| block64 for both | **32,087** | **53.91 ms** |
| block32 for both | 22,285 | 84.81 ms |

Current DSV4 launcher uses block64.

## Trace evidence

### FlyDSL TBO before comm-stream fix

- all A2A and compute kernels on the same stream
- A2A events: 1,708
- cross-stream overlap count: 0

### FlyDSL TBO, comm stream, default grid

- A2A stream tid6, compute stream tid8
- A2A work: 674 ms/forward
- overlap: 1,786 / 1,952 events
- overlap fraction: 29.1%
- hidden overlap: ~196 ms/forward
- MoE GEMM: 242 ms/forward
- other GEMM: 423 ms/forward

### FlyDSL TBO, block64

- A2A work: 614 ms/forward
- overlap: 1,952 / 1,952 events
- overlap fraction: 33.7%
- hidden overlap: ~207 ms/forward
- MoE GEMM: 247 ms/forward
- other GEMM: 338 ms/forward

Interpretation: FlyDSL A2A is a GPU kernel using CUs/cache/P2P traffic, not pure
DMA. Reducing the grid gave compute more room while retaining sufficient comm
parallelism. block32 crossed into under-parallelization.

### DP-TBO

- dedicated comm-stream NCCL events: 732
- all 732 overlap compute
- ~49.5% of dedicated-stream NCCL duration is hidden
- total NCCL work: 684 -> 730 ms/forward from no-TBO to TBO (+7%)
- MoE GEMM work: 100 -> 193 ms/forward due two ubatches

NCCL also uses CUs, but its tuned channel/grid footprint interferes less than
the original FlyDSL grid.

## Trace artifacts

FlyDSL:

- no-TBO: `/tmp/flydsl_notbo_prof/`
- TBO before stream fix: `/tmp/flydsl_tbo_prof/`
- TBO comm stream default grid: `/tmp/flydsl_tbo_commstream_prof/`
- TBO block64 fresh matched trace: `/tmp/flydsl_tbo_block64_prof_fresh/`

DP:

- no-TBO: `/tmp/dp_notbo_prof/`
- TBO: `/tmp/dp_tbo_prof/`

Server/load logs use corresponding names under `/tmp/`.

Benchmark artifacts:

- FlyDSL block64:
  - `/tmp/ab_flydsl/bench_tbo_block64/`
  - `/tmp/ab_flydsl/bench_tbo_block64_repeat/`
- FlyDSL TBO guarded delayer:
  - `/tmp/ab_flydsl/bench_flydsl_tbo_guarded_delayer/`
- DP guarded no-TBO:
  - `/tmp/ab_flydsl/bench_dp_mixed_guard_full/`
- DP guarded TBO:
  - `/tmp/ab_flydsl/bench_dp_tbo_guarded_delayer/`

## Memory / scheduler findings already resolved

Do not assume the historical DP `record_stream` memory bug applies to FlyDSL:

- FlyDSL uses preallocated shmem A2A outputs
- no-TBO and TBO KV pool: 8,765,184 tokens
- memory-pool remainder: ~32.3 GB both
- graph memory: 9.24 vs 9.26 GB
- mem0.9 stable

PrefillDelayer regression was separately bisected to:

```text
d03c8cee8 Negotiate PrefillDelayer only after KV-budget admission checks
```

It is fixed by `9b0eb8a55`. Do not benchmark current DP/FlyDSL with the old
mixed-timeout behavior.

## Highest-value next experiments

### 1. Build a critical-path trace parser

Current summaries add kernel durations and can double-count overlap. Build a
parser that reports:

- wall-clock span per forward/layer
- union of busy intervals per stream
- dispatch and combine separately
- exact overlap with other-ubatch attention / GEMM
- stream idle gaps and event-wait gaps
- kernel time inflation under overlap

Use matched fresh-server traces and NVTX `aN` / `bN` operation stages.

### 2. Separate dispatch and combine geometry

Current knob pins both phases to block64. Add independent experimental knobs:

```text
SGLANG_FLYDSL_TBO_DISPATCH_BLOCK_NUM
SGLANG_FLYDSL_TBO_COMBINE_BLOCK_NUM
```

Suggested sweep:

```text
dispatch ∈ {32, 64, 80, 128}
combine  ∈ {32, 64, 128}
```

Screen with trace/microbench, then full serving for finalists. Do not assume
the same block count is optimal for both phases.

### 3. Verify prefill recv-buffer M

Dynamic recv cap is currently decode CUDA-graph only; eager/TBO prefill keeps
the physical cap. Verify the actual aiter input M for both TBO children during
real prefill.

If it is still over-padded, prototype a rank-synchronized op-level prefill cap:

```text
cap >= sum(global child token counts)
```

Requirements:

- same cap/encoding stride on every EP rank
- op-level dispatch and combine agreement
- power-of-two bucketing to bound JIT variants
- no runner-only shmem slicing (previously crashed)

### 4. TBO split ratio

The expert GEMMs are weight-heavy at small M. Two equal ubatches can reload
weights twice with little per-call speed reduction. Sweep
`tbo_token_distribution_threshold` and measure:

- GEMM M and duration for each child
- A2A overlap
- total critical path

### 5. Comm stream priority

Test a lower-priority FlyDSL comm stream so GEMM wins CU scheduling while A2A
still progresses. Verify actual ROCm priority behavior in the trace.

### 6. DSV4 fusion loss

TBO disables cross-layer fused-mHC and uses self-contained layer ops. Quantify
the lost fusion separately from A2A overlap. Avoid redesign until its wall-time
share is measured.

## Methodology guardrails

- Fresh server for every serving datapoint.
- 2048 prompts / 512 warmups for final decisions.
- Change one variable per A/B.
- Verify stream/overlap outcome, not merely flags.
- Do not infer contention only from an A2A stall site.
- Keep correctness gates: focused 8-GPU test + full GSM8K.
- Confirm GPUs return to ~0.3 GB after every run.
- Record all exact env, branch, commit, and trace paths.

## 2026-07-24 critical-path and geometry update

### New analysis tooling

Parser:

```text
/sgl-workspace/sglang-flydsl-a2a/.claude/skills/llm-torch-profiler-analysis/scripts/analyze_tbo_critical_path.py
```

It reports exact interval unions (not summed kernel durations), per-stream busy
time, multi-stream overlap, other-stream hidden/exposed comm, dispatch/combine
separately, matched-kernel inflation, wait calls, rank percentiles, and
CPU-to-GPU `aN`/`bN` stage attribution when CPU spans exist.

Existing GPU-only artifact summary:

```text
/tmp/tbo_critical_path_smoke_v2/tbo_critical_path.{json,md}
```

Matched CPU+GPU traces and analysis:

```text
/tmp/ab_flydsl/prof_flydsl_tbo_cpu_gpu_1step/
/tmp/ab_flydsl/prof_dp_tbo_cpu_gpu_1step/
/tmp/tbo_cpu_gpu_matched_analysis/tbo_critical_path.{json,md}
```

Every rank contains 246 stage spans (`a0..a122`, `b0..b122`). Rank-p50 stage
mapping coverage is 98.9% for FlyDSL and 97.2% for DP. The matched short trace
shows:

| Metric (rank-p50 stage) | FlyDSL-TBO | DP-TBO |
|---|---:|---:|
| A GPU span | 7.52 ms | 4.78 ms |
| B GPU span | 6.71 ms | 4.82 ms |
| A comm | 2.34 ms | 1.51 ms |
| B comm | 2.44 ms | 1.37 ms |
| A hidden comm | 2.19 ms | 1.25 ms |
| B hidden comm | 2.27 ms | 1.18 ms |

These one-step captures include startup/ramp effects. Use them to localize
stage costs, not as serving-throughput estimates.

### Independent geometry knobs

Implemented:

```text
SGLANG_FLYDSL_TBO_DISPATCH_BLOCK_NUM
SGLANG_FLYDSL_TBO_COMBINE_BLOCK_NUM
```

Each phase-specific nonzero value overrides only that phase. Zero/unset falls
back to `SGLANG_FLYDSL_TBO_BLOCK_NUM`, then automatic tuning. Non-TBO ignores
all TBO geometry knobs.

All 12 dispatch `{32,64,80,128}` x combine `{32,64,128}` short screens
completed:

```text
/tmp/ab_flydsl/geometry_screen/ranking.{md,csv}
```

Canonical 2048/512 serving finalists:

| Geometry | Total tok/s | Median TPOT | Median TTFT |
|---|---:|---:|---:|
| dispatch32/combine32 | 32,711.62 | 54.63 ms | 16.00 s |
| **dispatch80/combine32** | **32,793.00** | **54.23 ms** | 16.05 s |
| dispatch80/combine64 | 32,320.43 | 54.63 ms | 16.88 s |
| dispatch64/combine64 control | 32,472.53 | 54.42 ms | 16.67 s |

Artifacts:

```text
/tmp/ab_flydsl/geometry_final/summary.{md,csv}
```

The measured geometry win is +0.987% over the fresh control. Only one full run
per geometry was taken, and control was resumed after a machine restart.

Winner repro:

```bash
SGLANG_FLYDSL_TBO_BLOCK_NUM=0 \
SGLANG_FLYDSL_TBO_DISPATCH_BLOCK_NUM=80 \
SGLANG_FLYDSL_TBO_COMBINE_BLOCK_NUM=32 \
MODE=flydsl-tbo DELAYER=on PORT=8000 \
SGLANG_PYTHONPATH=/sgl-workspace/sglang-flydsl-a2a/python \
bash /workspace/useful-scripts/benchmarking/dsv4/run_sgl_dsv4_unified.sh
```

### Recv, split, and priority probes

Full report:

```text
/tmp/ab_flydsl/residual_probes/residual_summary.{md,csv}
```

Opt-in diagnostics were added:

```text
SGLANG_FLYDSL_TBO_TELEMETRY=1
SGLANG_FLYDSL_TBO_TELEMETRY_SYNC_VALUES=1  # syncs; diagnostic only
SGLANG_FLYDSL_TBO_COMM_STREAM_PRIORITY=<int>
```

All-rank, all-layer recv diagnostic (976 events):

```text
physical recv cap       65,536 rows
actual recv p50             18 rows
actual recv p95          2,048 rows
actual recv max          4,623 rows
max actual/cap            7.05%
```

This proves large padding but not a safe optimization. Setting
`SGLANG_MORI_MOE_MAX_INPUT_TOKENS=8192` kept observed actual recv below the cap
but sliced only aiter inputs while FlyDSL combine retained the 65,536-row
encoding. All ranks hit GPU memory-access faults. The candidate is rejected.

Other controlled probes:

| Probe | Verified outcome | Short throughput delta | Decision |
|---|---|---:|---|
| split threshold 0 | changed 4096/4096 to 0/8192 | -20.355% | reject |
| comm priority -1 | all ranks actual priority -1 | -0.036% | null/reject |

Cross-probe two-step GPU spans were not comparable because profiler windows
captured different work; decisions above use verified intervention plus matched
short throughput.

### Validation

```text
focused parser/telemetry/geometry tests: 38 passed
8-GPU test_flydsl_dynamic_recv_cap.py: exit 0
full GSM8K dispatch80/combine32: accuracy 0.942, invalid 0.000
final GPU state: no KFD PIDs, 297,766,912 bytes/GPU
```

Logs:

```text
/tmp/ab_flydsl/final_dynamic_recv_cap_test.log
/tmp/ab_flydsl/final_d080_c032_gsm8k.log
/tmp/ab_flydsl/final_d080_c032_gsm8k_server.log
```

## 2026-07-25 residual confirmation and contention update

### Six-pair interleaved serving result

Artifacts:

```text
/tmp/ab_flydsl/ab_confirm/raw_pairs.csv
/tmp/ab_flydsl/ab_confirm/summary.{md,csv}
/tmp/ab_flydsl/ab_confirm/statistical_gate.md
```

Schedule alternated DP→FlyDSL and FlyDSL→DP for six paired blocks. Every arm
used a fresh server, 2048 measured requests, 512 warmups, exact 80/32 FlyDSL
geometry, telemetry/profiling off, and clean 0.298 GB/GPU teardown.

| Metric | DP mean | FlyDSL mean | Paired gap | 95% t-CI |
|---|---:|---:|---:|---:|
| Total tok/s | 33,163.67 | 32,685.94 | 1.4404% | [1.2264%, 1.6545%] |
| Median TPOT | 52.95 ms | 54.46 ms | FlyDSL 2.86% worse | [2.29%, 3.44%] |
| Median TTFT | 16.70 s | 16.21 s | FlyDSL 2.93% better | [1.42%, 4.44%] |

The one-sided paired t-test for a positive throughput gap gives `p=5.91e-6`.
No B07/B08 extension was needed.

### Matched stage attribution

Accepted captures:

```text
/tmp/ab_flydsl/gap_profile_confirmed/torch_profiles/{dp,flydsl}/
/tmp/ab_flydsl/gap_profile_confirmed/analysis/tbo_critical_path.{json,md}
/tmp/ab_flydsl/gap_profile_confirmed/matched_profile_summary.{md,json}
/tmp/ab_flydsl/gap_profile_confirmed/kernel_catalog.{md,csv}
```

Both backends have 492 identical A/B stage scopes per rank. Stage mapping is
96.5% for DP and 98.0% for FlyDSL.

| Rank-p50 stage metric | DP | FlyDSL |
|---|---:|---:|
| A span | 5.832 ms | 7.997 ms |
| B span | 5.904 ms | 7.543 ms |
| A comm union | 1.834 ms | 2.740 ms |
| B comm union | 1.834 ms | 2.680 ms |

Communication is mostly hidden in both. FlyDSL A has a heavier exposed p95
tail (0.932 ms). Operation mapping is only ~30%; global forward-window
inference is low confidence and is not used for backend wall-time claims.

rocprofv3 attach discovered no CSV despite successful attachment. Exact
regex-ready kernel names therefore come from the accepted torch traces.

### Controlled overlap on/off

An experimental, default-on diagnostic knob was added:

```text
SGLANG_FLYDSL_TBO_USE_COMM_STREAM=1  # current/default behavior
SGLANG_FLYDSL_TBO_USE_COMM_STREAM=0  # diagnostic serialization only
```

Artifacts:

```text
/tmp/ab_flydsl/contention/contention_summary.{md,json,csv}
```

The comparison preserves eight ranks, 492 stage scopes/rank, 244 calls/rank
for dispatch/combine/GEMM1/GEMM2, identical grids, and observable M=6527/6528.
With the stream enabled all ranks have cross-stream A2A/GEMM overlap; with it
disabled all A2A/GEMM work is on one stream with zero interval overlap.

| Exact-name/stage p50 | stream on | stream off | on/off |
|---|---:|---:|---:|
| Stage A | 8.009 ms | 5.020 ms | 1.596x |
| Stage B | 7.563 ms | 5.243 ms | 1.442x |
| Dispatch | 2.027 ms | 1.029 ms | 1.970x |
| Combine | 3.549 ms | 2.167 ms | 1.638x |
| GEMM1 | 1.241 ms | 1.190 ms | 1.043x |
| GEMM2 | 0.852 ms | 0.903 ms | 0.944x |
| Short input throughput | 54,144 tok/s | 41,619 tok/s | 1.301x |

This directly establishes overlap-induced kernel inflation, primarily in A2A,
while also proving overlap remains strongly beneficial end-to-end. It is
contention evidence, not proof of which resource is limiting.

Wrapped serving PMC failed during warmup with
`HSA_STATUS_ERROR_INVALID_PACKET_FORMAT` for both four-counter and single
`MfmaUtil` jobs. No matched MFMA/CU/L2/GMI counters are valid, and ATT was
correctly skipped.

### Gated fine screen

Artifacts:

```text
/tmp/ab_flydsl/fine_screen/fine_screen_summary.md
/tmp/ab_flydsl/fine_screen/full_finalists/full_finalists_summary.md
```

Short F1 tested dispatch 72/88/96, combine 24/40/48, and
`GPU_MAX_HW_QUEUES` 4/6 around the 80/32 queue5 control. Only dispatch72 passed
the +0.3% short gate. Canonical F2:

```text
dispatch80/combine32   32,777.41 tok/s
dispatch72/combine32   32,794.57 tok/s  (+0.052%)
```

The +0.3% full-serving gate failed, so 80/32 remains the selected geometry and
no extra DP pairs were run. Queue4/5/6 all showed the same three logical
streams and assignments; the queue intervention outcome was unverified.

### Final validation

```text
focused parser/telemetry/geometry/stream tests: 42 passed
8-GPU test_flydsl_dynamic_recv_cap.py: exit 0
full GSM8K dispatch80/combine32: accuracy 0.935, invalid 0.000
pre-commit and IDE lints: pass
final GPU state: no KFD PIDs, 297,766,912 bytes/GPU
```

Final logs:

```text
/tmp/ab_flydsl/final2_dynamic_recv_cap_test.log
/tmp/ab_flydsl/final2_d080_c032_gsm8k.log
/tmp/ab_flydsl/final2_d080_c032_gsm8k_server.log
```

## 2026-07-26 eager shared recv-cap update

### Correct cap contract

Local `_op_cur_tok` differs across EP/DP ranks and is not a safe shared stride.
For each eager TBO child, SGLang now performs one small host all-gather at child
preparation, sums all-rank padded dispatch rows, and stores the result on the
child `ForwardBatch` for reuse by all 61 MoE layers:

```text
cluster_rows = sum(all_rank_child_padded_rows)
recv_cap = min(physical_cap, max(32, next_pow2(cluster_rows)))
```

Data flow:

```text
parent synchronized token metadata
  -> TBO child cluster dispatch rows
  -> FlyDSL dispatch(recv_cap)
  -> logical-cap Aiter views
  -> FlyDSL combine(same recv_cap)
```

Physical shmem remains at the configured capacity. The optimization reduces
logical JIT stride/views and Aiter M; power-of-two bucketing bounds JIT variants.
Existing CUDA-graph/decode dynamic cap behavior is unchanged.

FlyDSL dispatch outputs no longer honor the mori-only
`SGLANG_MORI_MOE_MAX_INPUT_TOKENS` slicing path. Mori backend behavior is
unchanged. This prevents the earlier aiter-only cap mismatch that faulted with
dispatch/combine still using stride 65,536.

Flags:

```text
SGLANG_FLYDSL_DYNAMIC_RECV_CAP_EAGER=1       # default
SGLANG_FLYDSL_DYNAMIC_RECV_CAP_EAGER=0       # rollback
SGLANG_FLYDSL_DYNAMIC_RECV_CAP_VALIDATE=1    # diagnostic sync/assertions only
```

### Correctness and diagnostic evidence

New manual integration:

```text
/sgl-workspace/sglang-flydsl-a2a/test/manual/dsv4/test_flydsl_tbo_dispatcher.py
```

It runs two production child dispatchers over one shared comm stream with
different caps, forward/reverse child ordering, skewed routing, empty ranks,
state isolation, full-cap reference parity, and a deadlock timeout.

Diagnostic serving validated:

- all-rank cap agreement across all observed operations;
- `cap >= cluster dispatch rows`;
- `total_recv <= cap`;
- Aiter consumed rows equal dispatch logical views;
- unsafe mori 8192 slicing was bypassed;
- observed logical JIT buckets were 32 and 1024;
- no GPU fault, overflow, routing mismatch, or deadlock.

Artifacts:

```text
/tmp/ab_flydsl/shared_cap/shared_cap_summary.{md,json,csv}
```

### Performance confirmation

Initial canonical OFF/ON:

```text
OFF   32,686.87 tok/s
ON    33,878.47 tok/s   (+3.646%)
TTFT improvement: 7.11%
TPOT improvement: 2.46%
```

Final counterbalanced confirmation:

```text
/tmp/ab_flydsl/shared_cap_final_pairs/final_pair_summary.{md,json,csv}
```

| Pair | Order | OFF tok/s | ON tok/s | Gain |
|---|---|---:|---:|---:|
| P01 | OFF→ON | 32,756.33 | 33,799.15 | +3.184% |
| P02 | ON→OFF | 32,718.11 | 33,703.69 | +3.012% |

Mean paired gain is +3.098%. Mean TPOT and TTFT improvements are +2.225% and
+5.442%. Every arm completed 2048 requests after 512 warmups and differed only
in the eager-cap flag.

### Definitive eager-cap DP/FlyDSL comparison

Artifacts:

```text
/tmp/ab_flydsl/ab_confirm_eager_cap/raw_pairs.csv
/tmp/ab_flydsl/ab_confirm_eager_cap/summary.{md,csv}
/tmp/ab_flydsl/ab_confirm_eager_cap/statistical_gate.md
```

Six new counterbalanced pairs alternated DP→FlyDSL and FlyDSL→DP. Both arms
explicitly set the eager-cap flag for environment parity (DP ignores it);
FlyDSL used dispatch80/combine32 and the dedicated comm stream. All 12 arms
completed 2048 requests after 512 warmups with exact GPU cleanup.

| Metric | DP mean | FlyDSL mean | Paired `(DP-FlyDSL)/DP` | 95% t-CI |
|---|---:|---:|---:|---:|
| Total tok/s | 33,138.46 | 33,772.24 | -1.9130% | [-2.3317%, -1.4943%] |
| Output tok/s | 3,682.05 | 3,752.47 | -1.9130% | [-2.3318%, -1.4942%] |
| Mean TPOT | 53.03 ms | 53.18 ms | -0.2897% | [-0.6937%, +0.1142%] |
| Mean TTFT | 16.84 s | 15.34 s | +8.9142% | [+8.4145%, +9.4139%] |

Negative throughput gap means FlyDSL is faster. The one-sided paired t-test for
FlyDSL throughput being higher gives `p=3.93e-5`. Every individual pair shows
FlyDSL ahead by 1.36%–2.39%.

FlyDSL TTFT is conclusively lower. Its mean TPOT is 0.29% higher (worse), but
the 95% t-CI crosses zero, so no TPOT difference is established at 95%
confidence.

### A2A kernel experiments

All were implemented behind default-off flags, passed focused 8-GPU
correctness, then failed production gates. Their production code and dedicated
tests were removed afterward.

Artifacts:

```text
/tmp/ab_flydsl/combine_spinners/combine_spinner_summary.{md,json,csv}
/tmp/ab_flydsl/yield_wait/yield_wait_summary.{md,json,csv}
/tmp/ab_flydsl/dispatch_split/dispatch_split_summary.{md,json,csv}
```

Results:

| Experiment | Short delta | Decision |
|---|---:|---|
| combine waiters 256→8 | +0.181% | reject |
| yielding poll interval 64 | -0.147% | reject |
| yielding poll interval 256 | +0.106% | reject |
| dispatch coordinator split | -0.328% | reject |

### Final validation

```text
focused CPU tests: 70 passed
8-GPU dynamic-cap test: pass
8-GPU TBO dispatcher integration: pass
full GSM8K with eager cap default-on: accuracy 0.933, invalid 0.000
pre-commit and IDE lints: pass
final GPU state: no KFD PIDs, 297,766,912 bytes/GPU
```

Final logs:

```text
/tmp/ab_flydsl/final3_default_eager_gsm8k.log
/tmp/ab_flydsl/final3_default_eager_gsm8k_server.log
```

## 2026-07-27 concurrency scaling

One fresh-server run per backend was measured at c128 and c512 with fixed 8k/1k
lengths, `num_prompts=concurrency*8`, and `warmups=concurrency*2`.

Artifacts:

```text
/tmp/ab_flydsl/conc128_512_compare/summary.{md,csv}
```

| Concurrency | Backend | Prompts / warmups | Total tok/s | Mean TPOT | Mean TTFT |
|---:|---|---:|---:|---:|---:|
| 128 | DP-TBO | 1024 / 256 | 23,047.49 | 41.43 ms | 8.72 s |
| 128 | FlyDSL-TBO | 1024 / 256 | 23,285.48 | 41.78 ms | 7.84 s |
| 512 | DP-TBO | 4096 / 1024 | 42,100.72 | 76.76 ms | 33.31 s |
| 512 | FlyDSL-TBO | 4096 / 1024 | 43,755.00 | 75.29 ms | 30.28 s |

FlyDSL throughput advantage is +1.03% at c128 and +3.93% at c512. Both c512
arms completed all 4096 requests without HSA/server failure, despite historical
TBO instability above c256. These are single runs per arm with no confidence
interval; use the six-pair c256 result for the statistically confirmed backend
comparison.

## 2026-07-28 production MoE trace decomposition

Matched 8k/1k c256 traces were captured for DP-TBO and FlyDSL-TBO with
production CUDA graph ON, plus graph-OFF diagnostic cross-checks.

Artifacts:

```text
/tmp/ab_flydsl/moe_trace_8k1k_c256/moe_timing_summary.{md,json,csv}
/tmp/ab_flydsl/moe_trace_8k1k_c256/graph_on_reliability.{md,json,csv}
/tmp/ab_flydsl/moe_trace_8k1k_c256/kernel_catalog.{md,csv}
```

Production graph-ON results (rank p50):

| Stage | Backend | MoE union | Per layer | Whole GPU span | MoE share |
|---|---|---:|---:|---:|---:|
| Prefill | DP-TBO | 720.809 ms | 11.817 ms | 1022.334 ms | 70.5% |
| Prefill | FlyDSL-TBO | 668.024 ms | 10.951 ms | 900.559 ms | 74.2% |
| Decode | DP-TBO | 16.693 ms | 273.65 us | 38.023 ms | 43.9% |
| Decode | FlyDSL-TBO | 18.503 ms | 303.33 us | 39.851 ms | 46.4% |

FlyDSL prefill MoE is 7.3% faster. Its communication union is higher
(10.123 vs 7.896 ms/layer), but GEMM1/GEMM2 are 21.6%/31.5% lower and it
avoids the DP reduction cost.

FlyDSL decode MoE is 10.8% slower. Communication is 72.78 vs 42.02 us/layer
(+73%), while GEMM1/GEMM2 are slightly faster; the extra communication nearly
explains the entire 29.7 us/layer MoE gap.

Current graph replay child timing is conditionally reliable on Torch 2.9.1 /
ROCm 7.2: no <=0.01 us or 0.001 us artifacts were present, and aligned-rank GPU
replay envelopes match same-run server step estimates within about 5%.
Component timing excludes ranks 0/4/6/7 because profiler-stage skew made their
final communication operation spin for 0.58-0.60 s. Expanded graph child events
lack grid/block metadata, so strict graph-ON/OFF shape matching remains
unavailable.

Prefill normalization combines both TBO children and divides by 61 model
layers. Preferred totals are interval unions; component unions overlap and must
not be summed.

Canvas:

```text
/root/.cursor/projects/sgl-workspace/canvases/moe-trace-timing.canvas.tsx
```

## 2026-07-28 non-TBO production prefill MoE

Matched TBO-OFF production prefill traces were captured for `MODE=dp` and
`MODE=flydsl` with the same 8k/1k c256 workload.

Artifacts:

```text
/tmp/ab_flydsl/moe_trace_notbo_8k1k_c256/notbo_prefill_moe_summary.{md,json,csv}
/tmp/ab_flydsl/moe_trace_notbo_8k1k_c256/kernel_catalog.{md,csv}
```

| Backend | MoE union / forward | Per layer | Whole prefill span | MoE share |
|---|---:|---:|---:|---:|
| DP no-TBO | 677.124 ms | 11.100 ms | 1195.423 ms | 56.6% |
| FlyDSL no-TBO | 626.792 ms | 10.275 ms | 1139.523 ms | 55.6% |

FlyDSL plain prefill MoE is 7.4% lower than DP. Its communication and GEMM1
unions are higher, but GEMM2 is 33% lower and it avoids DP's 1.297 ms/layer
reduction cost.

Token work is matched to the TBO captures: 8192-token full forwards, 61 layers,
one GEMM1 call/layer without TBO versus two child calls/layer with TBO.

| Backend | no-TBO MoE union | TBO MoE union | TBO change |
|---|---:|---:|---:|
| DP | 677.124 ms | 720.809 ms | +6.5% |
| FlyDSL | 626.792 ms | 668.024 ms | +6.6% |

TBO increases full-forward MoE busy union because it duplicates child-level
communication/GEMM launch work, but reduces the whole prefill critical path
through overlap. Component unions overlap and are not additive.

### Activation/quant gap root cause

The large prefill activation/quant gap is caused by one extra inter-stage A2
quant kernel per EP MoE invocation.

Trace evidence:

```text
TBO DP:       1 dynamic quant/child; activation union 288.31 us/model-layer
TBO FlyDSL:   2 dynamic quant/child; activation union 923.74 us/model-layer
no-TBO DP:    1 dynamic quant/layer; activation union 229.65 us/layer
no-TBO FlyDSL:2 dynamic quant/layer; activation union 914.12 us/layer
```

Launch geometry identifies the tensors:

```text
EP A1 quant: grid 114688 (TBO) / 229376 (no-TBO)
             = M * hidden(7168) / 64
EP A2 quant: grid 294912 (TBO) / 589824 (no-TBO)
             = M * topk(6) * inter(3072) / 64
```

DP selects a tuned GEMM1 with `_fp8` / `fp8q_sort`; Aiter sets
`fuse_quant=fp8`, so stage1 emits quantized A2 directly. EP local
`(E=48, inter=3072)` has no tuned config and uses the heuristic BF16-output
FlyDSL fallback. With `fuse_quant` empty, Aiter launches a second
`fused_dynamic_mxfp8_quant_moe_sort` for A2 before GEMM2.

The extra A2 quant explains about 98% of the TBO activation gap and 92% of the
no-TBO activation gap. Decode uses small fused quant/sort kernels, so its
activation gap is only about 1.6 us/layer.

Resolved below: EP now selects an FP8-output GEMM1 and a correctness/performance
validated t64x128 GEMM2 for the production token tiers.

## 2026-07-29 EP GEMM1 fused-A2 optimization

The activation/quant root cause above is fixed and promoted.

### Contract and tuning fix

FlyDSL dispatch has no fake top-k expert slot, but Aiter tuning lookup
previously subtracted one whenever `expert_mask` was present. Production logs
therefore used topk=5 and missed E48/inter3072/topk6 rows.

Changes:

```text
/sgl-workspace/aiter/aiter/fused_moe.py
/sgl-workspace/aiter/aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv
/sgl-workspace/sglang-flydsl-a2a/python/sglang/srt/layers/moe/token_dispatcher/flydslep.py
```

- Aiter lookup now supports an explicit no-fake-expert EP contract while
  preserving legacy EP defaults.
- SGLang FlyDSL dispatcher defaults that contract ON; explicit env `=0`
  rolls back.
- Model config token16384 and32768 rows use FP8-output GEMM1:
  `flydsl_moe1_afp8_wfp4_bf16_t128x256x256_bnt0_gui_fp8`.
- Both tiers pair with the validated fast GEMM2:
  `flydsl_moe2_afp8_wfp4_bf16_t64x128x256_atomic_persist_sbm128`.
- No `AITER_CONFIG_FMOE` override is required in production.

### Correctness and kernel-use proof

Artifacts:

```text
/tmp/ab_flydsl/ep_fp8_gemm1/correctness/
/tmp/ab_flydsl/ep_fp8_gemm1/stage2_search/
/tmp/ab_flydsl/ep_fp8_gemm1/kernel_proof/kernel_use_proof.{md,json,csv}
/tmp/ab_flydsl/ep_fp8_gemm1/default_smoke/
```

M tiers 512/1024/4096/8192/16384/32768 passed fused-A2 correctness, scale/layout,
expert-mask/skew and paired GEMM2 checks.

Production trace hard gates:

```text
all 8 ranks x 3 forwards: exact FP8Q GEMM1 + t64x128 GEMM2
dynamic quant calls:      244 -> 122 / full forward
A2 quant grid 294912:     removed
A1 quant grid 114688:     retained
activation union:         -38.26 ms
MoE union:                -49.55 ms
prefill stage span:       -48.32 ms
```

The first source-row GEMM2 (`t64x256 xcd4`) was correct but 5.363 ms and caused
a production regression. A bounded stage2 search selected the t64x128
persistent/sbm128 kernel at 1.985 ms. A hybrid with the old heuristic GEMM2 was
numerically incompatible and rejected. All rejected attempts remain in the
artifact trail.

### Serving and final trace

Artifacts:

```text
/tmp/ab_flydsl/ep_fp8_gemm1/perf/performance_summary.{md,json,csv}
/tmp/ab_flydsl/ep_fp8_gemm1/perf/final_pairs.{md,json,csv}
/tmp/ab_flydsl/ep_fp8_gemm1/perf/final_trace_report.{md,json,csv}
```

| Pair | OFF tok/s | ON tok/s | Throughput |
|---|---:|---:|---:|
| P01 | 33,777.90 | 34,627.47 | +2.515% |
| P02 | 33,706.07 | 34,702.16 | +2.955% |

Mean gains:

```text
total/output throughput  +2.735%
TPOT improvement         +1.944%
TTFT improvement         +5.254%
```

GSM8K accuracy is 0.937 with invalid=0. Focused 8-GPU correctness passed.

Final graph-ON trace:

```text
prefill activation union: 56.367 -> 18.066 ms
prefill MoE union:         665.068 -> 615.521 ms
prefill stage span:        902.095 -> 853.779 ms
decode MoE union:          18.503 -> 18.335 ms (-0.9%)
decode graph envelope:     39.751 ms, server estimate 37.765 ms
decode quant calls:        122 -> 61
```

No small-tier decode regression was found. Production-default smoke without
`AITER_CONFIG_FMOE` or explicit no-fake env selected the new rows on every rank
and forward.

Rollback:

```bash
AITER_FLYDSL_EP_NO_FAKE_EXPERT=0
```

Changes are committed in both repositories:

```text
SGLang  3f0a371a6  perf: enable fused A2 quant for FlyDSL EP
Aiter   bee50d97a  perf: fuse A2 quant for DSV4 FlyDSL EP
```

Aiter PR #4433 also contains merge commit `8525fc016` resolving current-main
conflicts; SGLang PR is #32726.

### Non-TBO fused-A2 prefill trace

Artifacts:

```text
/tmp/ab_flydsl/moe_trace_notbo_fused_a2_8k1k_c256/premerge_runtime/notbo_fused_a2_summary.{md,json,csv}
```

Matched `MODE=flydsl` TBO-OFF production prefill traces:

| Metric | OFF | ON | Change |
|---|---:|---:|---:|
| MoE union / forward | 621.790 ms | 569.707 ms | -8.38% |
| MoE / layer | 10.193 ms | 9.339 ms | -8.38% |
| Whole prefill span | 1129.531 ms | 1082.180 ms | -4.19% |
| Activation/quant / layer | 915.45 us | 284.47 us | -68.9% |

Hard signatures:

```text
quant calls / forward  122 -> 61
A2 grid 589824         removed
A1 grid 229376         retained
GEMM1/GEMM2 calls      61 / 61
```

All ranks/forwards selected the normal topk6 token32768 model row. Rank-p50 is
the primary result; ON p95 is widened by one rank-7 communication outlier.

This capture used an isolated premerge Aiter worktree at feature commit
`bee50d97a`, matching the runtime family used by the original no-TBO baseline.
The latest merged Aiter main introduced an unrelated A8W8 decode-graph route
that could not start on the host's available Triton/ROCm runtime; no graph
disable or source workaround was used.

### Full concurrency/workload uplift matrix

Artifacts:

```text
/tmp/ab_flydsl/fused_a2_matrix/summary.{md,json,csv}
```

All 12 datapoints passed exact baseline/candidate kernel-signature checks.

| Workload | Conc | OFF tok/s | ON tok/s | Throughput | Mean TPOT | Mean TTFT | Median ITL |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8k/1k | 128 | 23,228.78 | 23,597.17 | +1.59% | +1.09% | +4.43% | +0.15% |
| 8k/1k | 256 | 33,858.02 | 34,772.93 | +2.70% | +2.27% | +4.93% | +0.56% |
| 8k/1k | 512 | 43,926.75 | 45,154.43 | +2.79% | +1.51% | +5.99% | -1.64% |
| 70k/300 | 8 | 30,372.40 | 31,266.21 | +2.94% | +0.66% | +4.50% | -0.04% |
| 70k/300 | 16 | 41,184.87 | 42,797.90 | +3.92% | +2.81% | +4.80% | +0.44% |
| 70k/300 | 32 | 50,021.47 | 51,806.45 | +3.57% | +3.74% | +3.40% | -0.77% |

Unweighted mean throughput uplift is +2.36% for 8k/1k and +3.48% for
70k/300. These are one run per arm/concurrency with no confidence interval;
8k server order was OFF→ON and 70k was ON→OFF.

## Reproduction

FlyDSL TBO:

```bash
MODE=flydsl-tbo PORT=8000 \
bash /workspace/useful-scripts/benchmarking/dsv4/run_sgl_dsv4_unified.sh
```

DP TBO:

```bash
MODE=dp-tbo PORT=8000 \
bash /workspace/useful-scripts/benchmarking/dsv4/run_sgl_dsv4_unified.sh
```

Serving:

```bash
WORKLOADS="8192:1024" CONCS="256" NP_MULT=8 WARM_MULT=2 \
BACKEND=sglang-oai \
bash /workspace/useful-scripts/benchmarking/dsv4/sweep_dsv4_sglang_client.sh
```

Trace:

```bash
SGLANG_TORCH_PROFILER_DIR=/tmp/profile ...
curl -X POST http://127.0.0.1:8000/start_profile \
  -H 'Content-Type: application/json' \
  -d '{"num_steps":8,"activities":["GPU"]}'
```
