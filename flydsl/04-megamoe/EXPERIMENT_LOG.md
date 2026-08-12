# MegaMoE optimization experiment log (v4_pro a8w4, gfx950 / 8x MI355X)

Traceable record of every experiment in the MegaMoE dispatch + GEMM optimization work (2026-07).
Perf = `megav1` E2E (stage1+fused-stage2) ms from `tests/kernels/test_mega_moe.py` (cuda-event); oracle relL2
vs torch/atom PASS unless noted. Baseline (compact-only): bs1 **0.151** / bs8 **0.334** / bs64 **0.426** /
bs2048 **1.75**. Full detail: `COMPACT_SINGLE_ROUND_DESIGN.md` §8, `KERNEL_OWNER_DECODE_PLAN.md`,
`FLYDSL_KERNEL_OPT_PLAYBOOK.md`.

## A. Dispatch track (breakthrough #2 / recv)

| # | Experiment | Motivation / hypothesis | Problem hit | Result | Verdict |
|---|-----------|------------------------|-------------|--------|---------|
| 1 | recv 2B-i (draft) | 1 cross-PE round replacing compact's multi-round | — | Correct (bit-eq compact) but bs8 0.419 / bs64 0.518 -> **20-25% slower** | needs opt |
| 2 | M3-opt: fuse done2+recv_num -> 1 round | draft still had 2 rounds | — | combined with #3 | ok |
| 3 | M3-opt: drop 256-serial `my_base` prefix | non-local experts always 0, only epr needed | — | #2+#3 total: bs8 0.353 / bs64 0.452 -> **~5% slower** | still loses |
| 4 | 2B-ii: GEMM gathers staging directly (no scatter/barrier) | remove double-move + barrier | staging 2.8GB tensor arg **i32 overflow** (fixed via addr-based buffer resource) | Correct but bs1 0.235 / bs8 0.469 / bs64 0.586 -> **slower at all bs** (gather starves GEMM); reverted | pruned |

-> **Dispatch verdict: ship compact-only** (all recv paths default-off, not shipped).

## B. GEMM Track B (occupancy / config)

| # | Experiment | Motivation / hypothesis | Problem hit | Result | Verdict |
|---|-----------|------------------------|-------------|--------|---------|
| 5 | Phase 0 profiling (ATT+PMC) | measure before tuning | single-CU ATT swamped by block0 spin; ATT finalize crash; VRAM stragglers->OOM; pgrep self-match | Findings: dispatch ~11.6%, GEMM dominant, **1 wave/SIMD, LDS-bound**, memory-clean | points to GEMM |
| 6 | tile_n=128 (reach 2 waves/SIMD) | lds_out=4*tm*tn is 66% of LDS | — | bs8 +4.5% / bs64 +10% / bs2048 **+18%** (**slower everywhere**) | occupancy not the lever |
| 7 | tile_k=128 | shrink LDS | **ZeroDivisionError -> cross-PE deadlock** (fixed via ceiling-div, as robustness) | won't build | dead end |
| 8 | waves_per_eu=8 | force higher occupancy | — | bs64 0.4296 ~= baseline (**within noise**; hint can't beat hard LDS partition) | no effect |
| 9 | direct-store epilogue | free lds_out 65KB | — | not implemented | pruned (occupancy useless + prefill loses coalescing) |
| 10 | **`b_nt=2` (non-temporal weight loads)** | fp4 weights stream once/tile -> avoid L2 pollution | — | **decode bs8 -3.2% / bs64 -5.1% (tight repeats, non-overlapping clusters)**; prefill bs2048 +5%; bs1 +4.6%, bs512 +1.6% | **decode win** |
| 11 | **`b_nt=1`** | partial non-temporal | — | **prefill bs2048 -2.9%**; decode neutral | **prefill win** |

-> **Per-bucket: decode `b_nt=2`, prefill `b_nt=1`** - the only robust, shippable GEMM win (env-gated
`MEGA_S1_BNT`, default-off, not yet folded into the tune JSON).

## C. Deep GEMM K-loop (MFMA / schedule)

| # | Experiment | Motivation / hypothesis | Problem hit | Result | Verdict |
|---|-----------|------------------------|-------------|--------|---------|
| 12 | K-loop isolation profile (post-filter code.json + per-rank CU scan) | find the true K-loop bottleneck | no GEMM-only compile path; spin swamps accounting (solved via post-filter) | **MFMA-bound 48.5%**, VMEM only ~28%, barrier 9.7% -> overturns the B-VMEM hypothesis | points to MFMA sched |
| 13 | isched/ck_rate sweep (ck 1/2/4, isched 2) | fill MFMA bubbles | **measurement noise**: bs2048 baseline itself +-1.2%, effect within noise | no robust gain (kernel near compute-bound); reverted | no effect |
| 14 | B-weight K+1 prefetch | hide VMEM latency | — | not implemented | pruned (VMEM secondary, MFMA-bound) |
| 15 | build_pipe_schedule reshape | re-order phases | — | not implemented | pruned (same) |

-> **The ~13% vs dp-opus is not reachable via FlyDSL-level scheduling** (dp-opus is hand-tuned assembly).

## Side outputs (not experiments, but landed)
- **divzero fix** in `build_pipe_schedule` / `gemm1.py` (ceiling-div + `max(1,.)`): robustness; no-op for tile_k=256.
- **Methodology**: fused-kernel K-loop isolation profiling; noise floor (bs2048 +-1.2%, bs64 +-0.05% -> small
  effects need median-of-many); kill GPU by numeric PID. Captured in `FLYDSL_KERNEL_OPT_PLAYBOOK.md`.

## One-line summary
~15 experiments run; dispatch (recv / 2B-i / 2B-ii) and GEMM occupancy (tile_n / tile_k / waves_per_eu) all proved
useless or slower; the K-loop is **MFMA/compute-bound with no scheduling headroom**; the **only robust, shippable
win is `b_nt` per-bucket (decode -5%)**. Oracle PASS throughout; compact-only shipping path intact.

## D. ROCm/FlyDSL PR #876 MegaMoEV2 matched serving (2026-07-30)

| Arm | Revision | Total tok/s | Output tok/s | TPOT ms | Correctness | Verdict |
|---|---|---:|---:|---:|---|---|
| DP | SGLang `22faf9fe` + guard `87524746c` | 31,662.73 | 3,518.08 | 54.18 | 2048/2048 | reference |
| old compact MegaMoE | FlyDSL `6fb56158` | 29,526.63 | 3,280.74 | 59.54 | 8-rank smoke PASS; 2048/2048 | reproduced |
| PR #876 MegaMoEV2 | FlyDSL `48214325` | 24,744.00 | 2,749.33 | 93.08 | relL2 6.175e-02; 2048/2048 | reject |

Fixed workload: 8192/1024, concurrency 256, 2048 measured requests, 512
warmups, fresh server per valid arm, CUDA graph enabled. Aiter was
`bee50d97`; mixed-slot guard was enabled.

V2 regressed `16.20%` against old compact and `21.85%` against DP. Its TPOT
was `56.3%` worse than old compact. This exceeds the noise gate by a wide
margin.

PR #876's rank-synchronized autotuner deadlocked in SGLang continuous
batching (`broadcast_object_list`, 300-second scheduler watchdog). The valid V2
run bypassed only that collective and loaded the committed artifact bundles on
each rank. V2 TTFT/ITL were reported as zero by the client and are excluded
from conclusions.

### D1. Matched trace hotspot

Matched profiler-by-stage captures:

```text
/tmp/megamoe_pr876_trace_dp/profiles/
/tmp/megamoe_pr876_trace_v1/profiles/
/tmp/megamoe_pr876_trace_v2/profiles/
/tmp/megamoe_pr876_trace_analysis/megamoe_trace_summary.{md,json,csv}
```

Eager prefill rank-p50 exact interval unions:

| Metric | DP ms/forward | V1 ms/forward | V2 ms/forward | V2 - V1 |
|---|---:|---:|---:|---:|
| whole GPU union | 1,396.855 | 1,410.137 | 2,227.139 | +817.002 |
| MoE union | 868.526 | 910.673 | 1,650.634 | +739.961 |
| Stage1 dispatch + GEMM1 | — | 554.754 | 1,123.504 | +568.750 |
| Stage2 GEMM2 + P2P | — | 209.783 | 301.741 | +91.958 |
| EP combine | — | 145.984 | 213.497 | +67.513 |

Against V1, MoE explains about 91% of the whole-GPU union regression and
Stage1 explains about 77% of the MoE increase. V1 `moe_gemm1_0` is rank-p50
9.09 ms/launch; V2's hottest prefill signature is
`megamoe_stage1_t32x512x256_w8_gm1_dcu128_...` at rank-p50 25.11 ms/launch.

Independent V1-to-V2 server-log cross-checks show 8192-token prefill
throughput/rank `7,194.70 -> 5,574.27 tok/s` (-22.52%) and decode at 32
requests/rank `788.96 -> 679.63 tok/s` (-13.86%), corresponding to
`40.56 -> 47.08 ms` per decode step (+16.09%).

This identifies the Stage1 critical path, not its hardware resource bound.
Persistent Stage1 duration includes communication and synchronization wait;
no compute/bandwidth/occupancy claim is made without direct utilization data.

### D1a. Stage1 microbench correction

An 8-rank serving-matched microbench held MTPR=8192 and directly timed Stage1:

| Tokens/rank | V1 | V2 selected | V2 vs V1 |
|---:|---:|---:|---:|
| 32 | 0.3035 ms | 0.3441 ms (SBM32) | +13.4% |
| 8192 | 3.2596 ms | 2.8369 ms (SBM128) | -13.0% |

V2 config sweeps found SBM32 best at tokens=32
(`0.3441 / 0.3674 / 0.4586 ms` for SBM32/64/128). At tokens=8192,
isolated Stage1 preferred SBM64 (`2.6041 ms`) over SBM128 (`2.8369 ms`),
but the full BF16 E2E microbench was slightly better with the committed joint
SBM128 config (`4.6520 vs 4.6689 ms`, forced SBM64 +0.36%).

Correction: the ~2x V2 Stage1 serving-trace duration is not a 2x raw-kernel
compute regression and is not explained by an obvious config miss. The
serving hotspot includes cross-rank synchronization/wait inside the persistent
Stage1 kernel. Rank-skew/synchronization is now the primary hypothesis.

### D1b. Rank skew and rank-local config divergence

Injecting 0.25 or 1.0 ms launch delay per rank inflated both V1 and V2
persistent kernels by nearly the same amount. At 1.0 ms/rank, rank-0 Stage1
was `10.7011 ms` for V1 and `10.8724 ms` for V2. Differential launch-skew
sensitivity was rejected.

With uneven local tokens
`8192,8192,8192,8192,4096,4096,2048,1024`, V1 used one MTPR8192 config and
measured `2.6065 ms`. V2 rank-local artifact selection measured `3.4294 ms`;
forcing global SBM128 reduced it to `2.5427 ms` (-25.9%, and 2.4% faster than
V1).

Root integration issue: disabling collective autotune avoided a serving
deadlock, but allowed ranks to key artifact lookup by different local token
counts. A diagnostic global-token SBM guard recovered 2.44% serving throughput
and 5.29% prefill throughput in a matched 256-request pair:

```text
rank-local  22,486.55 tok/s, prefill/rank 5,565.81, TPOT 101.49 ms
global SBM  23,035.83 tok/s, prefill/rank 5,860.29, TPOT  99.06 ms
```

Decode throughput was unchanged (`682.06 -> 681.22 tok/s`). Config divergence
is therefore a proven prefill contributor, not the full V2 serving root cause.

### D1c. FlyDSL global workload/config synchronization

Implemented the permanent contract in the isolated PR #876 worktree:

- public rank-invariant `MegaMoEV2Workload`;
- global token bucket drives joint/Stage1/Stage2/P2P-quant artifact selection;
- artifact-only production path performs zero forward broadcasts;
- rank-local tensor shapes removed from config selection cache keys;
- explicit process-group support for live collective warmup;
- removed SGLang hardcoded SBM and sitecustomize autotune monkeypatch.

Validation passed: 55 autotune unit tests, uneven-rank identical config
signatures, V2 oracle bs64/2048/8192, CUDA graph serving, and 2048/2048
requests.

| V2 mode | Total tok/s | TPOT ms | Change |
|---|---:|---:|---:|
| rank-local workaround | 24,744.00 | 93.08 | reference |
| global workload API | 25,907.20 | 88.90 | +4.70% throughput |

Post-fix prefill MoE union fell from `1,650.634` to `1,122.069 ms/forward`.
Stage1 fell from `1,123.504` to `783.059 ms/forward`. V2 remains 12.26% below
V1 serving, so config divergence was substantial but not the only issue.

Artifacts:

```text
/tmp/megamoe_pr876_workload_api_full/megamoe/
/tmp/megamoe_pr876_workload_api_trace/profiles/
/tmp/megamoe_pr876_workload_api_trace_analysis/
/tmp/megamoe_pr876_workload_api_accuracy.log
```

### D1d. Residual Stage1 falsification matrix

Natural, uniform, and round-robin 256-request screens all kept the V2-to-V1
gap at 17–18%. Routing imbalance improves both implementations similarly and
is rejected as the V2-specific cause.

V2 SBM64 matched V1 `num_valid` exactly per rank; SBM128 added only 2–3%
padding. Work volume is insufficient to explain the trace gap.

61-layer replay:

```text
uniform histogram              V1 3.2707  V2 3.1632 ms/layer
recorded per-layer routing      V1 4.5880  V2 5.2130 ms/layer
recorded + alternating weights  V1 4.5806  V2 5.2693 ms/layer
```

Per-layer routing can reproduce the flip, but exact round-robin serving does
not remove the gap. Multi-stream ON/OFF and inter-layer cache traffic were
also null. An env-gated inline device-clock probe produced invalid timestamps
and was fully reverted; a supported ROCm/FlyDSL clock or rocprof phase
decomposition is still required to split planner, payload wait, and GEMM.

### D1e. Dispatch-only / GEMM-only decomposition

Implemented production-safe diagnostic kernels:

- Stage1 `dispatch_only` exits after payload publication and global plan
  rendezvous, with a distinct JIT name;
- GEMM-only uses the existing standalone `compile_gemm1` over frozen payload
  and metadata;
- production full kernel name/path remains unchanged.

Split/full outputs and route metadata are exact for bs64, bs8192, uneven and
idle-dummy ranks under CUDA graph.

```text
shape             dispatch  GEMM-only  full    closure
bs64 SBM32          0.1304    0.2180   0.3403   +2.4%
bs8192 SBM128       1.9666    2.1540   3.1830  +29.5%
bs8192 SBM64        2.0770    2.1572   2.6389  +60.5%
```

SBM64 wins by overlapping two individually similar phases more effectively.
Increasing dispatch CU 32→64 reduced micro full time to 2.670 ms, but serving
improved only 0.27%; dCU128 lost overlap. Active-expert-producer and
cooperative payload copy regressed. `grid_mult=1` was fast in microbench but
deadlocked serving and was rejected.

No structural candidate passed the ≥1% serving gate.

### D1f. PR #876 newest deterministic config selector (`2f9da1f4`)

The newest head removed MegaMoEV2 runtime autotuning and introduced
`mega_moe_config.py`. Standalone decode MTPR64 and prefill MTPR8192 both pass.

```text
decode MTPR64:    Stage1 0.2379 ms, BF16 E2E 0.3459 ms
prefill MTPR8192: Stage1 2.4589 ms, BF16 E2E 4.5166 ms
```

Compact decode with MTPR8192 now selects a small-token config and improves
Stage1 `0.3441 -> 0.2627 ms` (-23.7%); this is independent of fixed-slot.

Single-MTPR8192 serving is flat versus the previous `e092035d` head:
`26,260.65 vs 26,438.62 tok/s`. Dual MTPR8192/128 still hangs after prefill
when switching to decode, even with deterministic prebuild and no runtime
autotune. The blocker is shared mori comb-op/state coexistence, not config
selection.

### D2. Static EPLB placement

A single 8k/1k c256 expert-distribution record was applied identically to V1
and V2 as an initial static EPLB placement. MTPR remained 8192.

| Backend | Placement | Total tok/s | TPOT ms | Change |
|---|---|---:|---:|---:|
| V1 | trivial | 29,526.63 | 59.54 | reference |
| V1 | static EPLB | 30,442.98 | 58.51 | +3.10% |
| V2 | trivial | 24,744.00 | 93.08 | reference |
| V2 | static EPLB | 20,815.65 | 109.94 | -15.88% |

Server logs confirm that V1 prefill improved `7,280.96 -> 7,915.35 tok/s`
(+8.71%) while decode was flat (`787.20 -> 780.79`, -0.81%). V2 degraded
in both prefill (`5,634.79 -> 4,575.37`, -18.80%) and decode
(`679.55 -> 632.27`, -6.96%).

A separate dynamic-EPLB V1 smoke successfully migrated all 61 layers in
6.46 seconds after exposing the shuffled MegaMoE buffers as standard
expert-major parameter views. Migration time is excluded from the static EPLB
comparison.

Verdict: promote the static placement only as a V1 candidate. Reject it for
V2. V2 config sensitivity to the changed token distribution is a hypothesis,
not yet a measured resource-bound diagnosis.

Artifacts:

```text
/tmp/megamoe_pr876_eplb_record/
/tmp/megamoe_pr876_eplb_static_v1/megamoe/
/tmp/megamoe_pr876_eplb_static_v2/megamoe/
```

### D3. Dual-instance standalone reproducer

An 8-rank standalone harness was added for A8192 compact and B128 fixed-slot
MegaMoEV2 instances. It records per-rank heartbeats, output checksums/relL2,
signal generations, symmetric addresses, P2P pointer tables, dispatch tables,
and xdev barrier buffers.

All tested arms passed:

```text
A->A->A, B->B->B
build A+B then run only A / only B
A1->A2->A1, B1->B2->B1
A8192->B128->A8192 and reverse
cross-capacity switching with no per-step barrier
A eager -> B CUDA-graph replay -> A eager, full and no barrier
```

A/B address ranges had zero overlap. Each instance advanced only its own
parity/generation state. Repeated A output was exact; repeated B fixed-slot
output stayed within the existing A8W4 gate (`max relL2 0.063`).

Verdict: the SGLang hang is not a generic FlyDSL/mori multi-instance,
capacity-switch, drain, or decode-graph failure. Scope shifts to SGLang's
per-rank continuous-batching call sequence, adapter instance selection, and
graph/stream integration. B256 and intermediate CUDA/NCCL-only barrier arms
were skipped because the stronger B128/no-barrier case passed.

Artifacts:

```text
/sgl-workspace/sglang-megamoe-pr876/repro_megamoe_dual_instance.py
/sgl-workspace/sglang-megamoe-pr876/run_megamoe_dual_instance_matrix.sh
/sgl-workspace/sglang-megamoe-pr876/megamoe_dual_instance_repro_summary.{md,json,csv}
/tmp/dual_repro_ABA_state.log
```

### D4. Dual-instance SGLang root cause and rank-global fix

An env-gated adapter signature captured the first serving failure without
adding GPU synchronization:

```text
rank 0:   mode=EXTEND global_has_extend=true selected_mtpr=8192
rank 1-7: mode=IDLE   global_has_extend=true selected_mtpr=128
```

The standalone harness was extended with per-rank sequence replay. An
all-rank A control completed, then the recorded rank0=A/rank1-7=B step timed
out before any rank completed the divergent forward. This confirms instance
sequence divergence as causal.

After selecting the instance from rank-global `is_extend_in_batch`, a long
warmup still exposed config divergence: with MLP TP gather disabled,
`global_num_tokens` is intentionally local-only. `MLPSyncBatchInfo` already
has every rank's counts, so the fix carries its maximum as
`global_max_num_tokens` into `ForwardBatch` and uses it for
`MegaMoEV2Workload`. This adds no collective.

Results:

```text
arm                                  result      total tok/s
eager 8192/64 c256                   256/256       34,578.71
graph 8192/64 c256                   256/256       39,571.00
graph 8192/1024 c256 + signatures    256/256       27,100.60
graph 8192/1024 c256 final          2048/2048      27,550.61
```

All 163 signature-bearing mixed steps had eight matching
`global_max_num_tokens` values and zero MTPR policy mismatches.

The final apples-to-apples refresh used `8192/1024`, c256, warmup512 and
measured2048 for every arm:

```text
arm                         total tok/s  prefill/rank  decode/rank  step
DP                            31,358.52      6,906.53       895.00  35.75 ms
MegaMoEV1 3b0f818            29,831.76      7,300.63       790.65  40.47 ms
V2 e092035d                  warmup hang             no measured result
V2 2f9da1f4 single           28,041.60      6,371.86       819.44  39.05 ms
V2 2f9da1f4 dual 8192/128    27,550.61      6,366.34       804.26  39.79 ms
```

Dual is 1.75% slower than latest single V2. It fixes the serving hang but is
not a performance win in the matched workload.

The same fresh-launch trace contains 122 compact prefill Stage1/Stage2 kernels
and 11,834 fixed-slot decode Stage1/Stage2 kernels:

```text
/tmp/megamoe_dual_global_same_launch_trace/profiles/
/tmp/megamoe_dual_global_full_final/megamoe/
```

### D5. MegaMoEV2 GSM8K correctness gate

The original V2 integration had no full GSM8K run. A 20-question, 5-shot
matrix using the same SGLang checkout and model produced:

```text
arm                              accuracy  invalid
MegaMoEV1 control                  0.850     0.000
MegaMoEV2 single eager             0.000     1.000
MegaMoEV2 dual graph 8192/128      0.350     0.100
MegaMoEV2 dual eager 8192/128      0.400     0.150
```

The deprecated lm-eval path was first rejected because its DSV4 reasoning
extraction was incompatible. The in-tree `few_shot_gsm8k` control proves the
final matrix is valid: V1 is coherent while V2 is not. Single V2 still fails
with CUDA graph disabled and with generation cap 8192. Single/dual eager raw
outputs have 0/20 exact parity.

Verdict: stop the full dual GSM8K run. Correctness fails before the dual policy
is evaluated, so V2 serving performance remains diagnostic-only.

#### D5 resolution: latest upstream + rank-identical graph config

Fetched the force-updated `origin/mega_moe_v1` at `567b43e`, created a clean
worktree, and rebuilt its matching LLVM/MLIR/FlyDSL native extension. The
branch includes merged PR #876 (`dc8e153`) and follow-up `ea4ce21`.

Production replay ruled out the adapter packing and base kernels:

```text
W1/W2/scales identical V1/V2
one-layer max relL2 6.8e-5
uneven-rank max relL2 <1e-4
61-layer repeated-weight max cosine diff 3.6e-4
```

Accuracy recovered immediately on the rebuilt latest branch. A remaining full
graph hang was caused by per-rank graph buckets capturing different V2
configs. Added `config_tokens` to MegaMoEV2 and fixed every decode graph to
config bucket128.

```text
arm                         questions  accuracy  invalid
latest single graph                20     0.850    0.000
latest dual graph                  20     0.900    0.000
latest dual eager                  20     0.950    0.000
latest single eager              1319     0.925    0.001
latest dual eager                1319     0.924    0.001
latest single graph fixed        1319     0.923    0.000
latest dual graph fixed          1319     0.920    0.001
```

Verdict: correctness restored; old `2f9da1f4` performance is obsolete for
promotion and must be rerun on the rebuilt latest runtime.
