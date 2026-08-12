# FlyDSL MegaMoE on DSV4 (8×MI355X) — handoff (2026-07-16)

## CONTINUE HERE (2026-08-04, upstream refresh complete)

Status:

- Latest upstream `mega_moe_v1` is `52f335e`, including `9513a2a fix
  deadlock bug`. The clean worktree
  `/sgl-workspace/FlyDSL-mega_moe_v1-latest` was rebased and rebuilt.
  Its current local head is `b153c5d`, which preserves the required
  rank-identical graph config-token fix.
- Dual-instance rank/config divergence is fixed.
- Full graph GSM8K on the refreshed runtime passes: single `0.923`, dual
  `0.926`, invalid `0.000` for both.
- FlyDSL config-token fix after rebase: `b153c5d`; SGLang graph-config fix:
  `f9ac0112e`.
- Old V2 throughput numbers are obsolete.

Matched performance:

Fresh-server counterbalanced `8192/1024`, c256, warmup512, measured2048:

```text
arm       run 1 tok/s   run 2 tok/s   mean tok/s   mean TPOT
single       31,177.38      31,220.53    31,198.96     56.41 ms
dual         31,079.07      31,110.29    31,094.68     56.65 ms
```

Dual is `0.33%` below single throughput and has `0.42%` higher TPOT; this is
near-flat and does not establish a dual-instance performance win. All four
runs completed `2048/2048`; no watchdog, traceback, CUDA error, or runtime
error was found.

Raw logs:

```text
/tmp/megamoe_gsm8k_52f335e_{single,dual}_full/
/tmp/megamoe_52f335e_perf_{single,dual}/
/tmp/megamoe_52f335e_perf_{dual,single}_r2/
```

Next step: upstream or retain `b153c5d` before using upstream FlyDSL without
the local config-token patch. Do not commit the local profiler workaround
files.

Key correctness record:
[`MEGAMOE_V2_DUAL_INSTANCE_ROOT_CAUSE.md`](./MEGAMOE_V2_DUAL_INSTANCE_ROOT_CAUSE.md).

## 2026-07-30 — ROCm/FlyDSL PR #876 first matched serving round

PR #876 was evaluated in isolated worktrees without modifying the existing
FlyDSL or `sglang-flydsl-a2a` dirty worktrees.

Exact runtime:

```text
SGLang base       22faf9fef8048731863fb64c68cae2ab42b9fa4f
mixed-slot guard  87524746c (ported from 2d4cf9bd2)
Aiter             bee50d97a7e74b796bdac8ef0247ecb6706132c3
old FlyDSL V1     6fb56158f1db1ebdda7de5ca086c7b66d9abac1d
PR #876 head      4821432500f199f198c705a33e4eddc7a7d7048b
Torch              2.9.1+rocm7.2.0.git7e1940d4
Triton             3.6.0 (/sgl-workspace/triton-custom)
FlyDSL native      0.2.4 package, PR Python/compiler sources overlaid for V2
```

Workload was 8192 input / 1024 output, concurrency 256, 2048 measured
requests, 512 warmups, fresh server per valid arm, CUDA graph enabled, and
`SGLANG_PREFILL_DELAYER_MIXED_SLOT_GUARD=1`.

```text
arm                    total tok/s   output tok/s   TPOT ms   requests
DP baseline              31,662.73       3,518.08     54.18   2048/2048
old compact MegaMoE      29,526.63       3,280.74     59.54   2048/2048
PR #876 MegaMoEV2        24,744.00       2,749.33     93.08   2048/2048
```

Derived serving deltas:

```text
old MegaMoE vs DP       -6.75%
PR #876 V2 vs old      -16.20%
PR #876 V2 vs DP       -21.85%
V2 TPOT vs old         +56.3% (worse)
```

Correctness:

- old V1 `v4_pro` A8W4 bs64: PASS on all 8 ranks;
- V2 `v4_pro` A8W4 bs64: PASS on all 8 ranks, oracle relL2 `6.175e-02`;
- PrefillDelayer distributed mixed-slot guard case: PASS;
- all serving arms above completed 2048/2048 requests.

Integration finding: PR #876 collective autotune hangs under SGLang
continuous batching because scheduler ranks can reach the autotuner in
different key/order states. The first V2 serving warmup hit the 300-second
watchdog with all schedulers in `broadcast_object_list`. The completed V2 run
disabled only the collective broadcast and loaded the PR-bundled per-rank
tuning artifacts locally; kernel/config selection remained artifact-driven.

The V2 client emitted TTFT/ITL as `0.00`; those fields are invalid and were not
used for the decision. Throughput, successful request count, CUDA-graph
capture, and TPOT were valid.

Decision: **do not promote PR #876 MegaMoEV2 for DSV4 serving in its current
form.** The first matched serving result is a large regression, not a sub-1%
noise result, so no counterbalanced rerun is required before rejection.

### Matched DP/V2 trace attribution

Fresh-server DP, V1, and V2 traces used 8192/1024, concurrency 256, 256 requests,
CUDA graph enabled, and profiler-by-stage captures for eager prefill and
decode. All stage totals below are rank-p50 exact interval unions, normalized
by observed Stage1 multiplicity.

```text
eager prefill metric         DP ms/fwd   V1 ms/fwd   V2 ms/fwd   V2 - V1
whole GPU union              1,396.855   1,410.137   2,227.139   +817.002
MoE union                      868.526     910.673   1,650.634   +739.961
Stage1 dispatch + GEMM1              -     554.754   1,123.504   +568.750
Stage2 GEMM2 + P2P                   -     209.783     301.741    +91.958
EP combine                           -     145.984     213.497    +67.513
```

Against V1, the MoE interval increase explains about 91% of the whole-GPU
union increase. Stage1 contributes 568.750 ms, or about 77% of the MoE
increase, and is the primary hotspot. V1's `moe_gemm1_0` is rank-p50
9.09 ms/launch; V2's hottest prefill kernel is:

```text
megamoe_stage1_t32x512x256_w8_gm1_dcu128_pw0ma0sw1aa0_...
rank-p50 25.11 ms / launch
```

Server-log cross-check:

```text
metric                               DP          V1          V2    V2 vs V1
8192-token prefill / rank      6,892.43    7,194.70    5,574.27     -22.52%
decode, 32 req/rank              896.48      788.96      679.63     -13.86%
derived decode step               35.70       40.56       47.08 ms  +16.09%
```

V1 decode `moe_gemm1_0` is rank-p50 6.28 ms/launch. The V2 decode Stage1 signature is
`megamoe_stage1_t32x128x256_w2_gm4_dcu64_...` at rank-p50 5.90 ms/launch.
Expanded CUDA-graph child-event totals are treated as qualitative; matched
batch logs provide the decode timing conclusion.

These traces locate the regression in the fused Stage1 critical path, but do
not by themselves establish compute-, bandwidth-, or occupancy-bound behavior.
Persistent-kernel duration includes in-kernel communication/synchronization
wait.

### Stage1 microbench falsification and config check

A serving-matched 8-rank microbench fixed MTPR at 8192, used production
7168/3072/E384/topk6 A8W4 shapes, timed only Stage1 under CUDA graph, and
printed the selected config.

```text
tokens/rank  implementation  selected config                         mean ms
32           V1              compact SBM64 t64x256x256 b_nt0          0.3035
32           V2              SBM32 t32x128x256 dCU64 w2               0.3441
8192         V1              compact SBM64 t64x256x256 b_nt0          3.2596
8192         V2              SBM128 t128x512x256 dCU32 w8             2.8369
```

Therefore V2 raw Stage1 is only 13.4% slower at decode, and is 13.0% faster
at full prefill. The ~2x Stage1 duration seen in serving traces is not a raw
compute-kernel difference; it includes substantial persistent-kernel
cross-rank synchronization/wait under serving.

V2 config sweeps:

```text
tokens=32:   SBM32 0.3441 ms (best), SBM64 0.3674, SBM128 0.4586
tokens=8192: SBM32 3.8806 ms, SBM64 2.6041, SBM128 2.8369
```

Although SBM64 is faster for isolated prefill Stage1, the full V2 microbench
shows default joint SBM128 at 4.6520 ms BF16 E2E versus forced SBM64 at
4.6689 ms (+0.36% worse). Thus there is no evidence that the committed joint
config selected the wrong full-MoE configuration.

Corrected attribution: Stage1 is where serving time is exposed, but the
primary open hypothesis is serving-time synchronization/rank skew, not an
incorrect tile or a 2x raw Stage1 compute regression.

### Rank-skew and config-divergence validation

Controlled per-rank launch skew did not show a V2-specific sensitivity:

```text
skew/rank    V1 max Stage1   V2 max Stage1   V2 vs V1
0 ms              3.5556          4.0636      +14.3%
0.25 ms           5.4269          5.9975      +10.5%
1.00 ms          10.7011         10.8724       +1.6%
```

Both persistent kernels absorb launch skew as in-kernel wait, so launch timing
alone does not explain the serving regression.

The decisive probe used uneven local token counts
`8192,8192,8192,8192,4096,4096,2048,1024` with MTPR=8192:

```text
implementation  config policy          Stage1 mean
V1              global MTPR8192 config     2.6065 ms
V2              rank-local artifacts       3.4294 ms
V2              global SBM128               2.5427 ms
```

After collective autotune was disabled to avoid its continuous-batching
deadlock, V2 ranks independently keyed artifacts by local token count. The
1024-token rank selected a different geometry, and rank-local policy was 34.9%
slower than forcing the global-token SBM. With global SBM, V2 was 2.4% faster
than V1 in the same uneven-token microbench.

An env-gated diagnostic
`SGLANG_AMD_FLYDSL_MEGA_V2_SYNC_SBM=1` selects SBM from the DP-global maximum
token count and bypasses the joint-SBM collective. A matched 256-request
serving pair showed:

```text
V2 policy             total tok/s   prefill/rank   decode/rank   TPOT
rank-local artifacts    22,486.55      5,565.81       682.06    101.49 ms
global-token SBM        23,035.83      5,860.29       681.22     99.06 ms
change                    +2.44%         +5.29%        -0.12%     -2.39%
```

Conclusion: rank-local config divergence is a proven contributor to V2
prefill regression, introduced by the workaround for collective-autotune
deadlock. It does not explain the remaining decode gap.

### FlyDSL global workload/config synchronization implementation

The temporary SGLang SBM selection and runtime monkeypatch that disabled
collective autotune have been replaced by a FlyDSL-owned contract:

- `MegaMoEV2Workload` carries a rank-invariant power-of-two token bucket,
  MTPR capacity, forward mode, and optional graph bucket;
- joint SBM, Stage1, Stage2, and automatic P2P quant all key config selection
  from the same global bucket;
- runtime `cur_tok` remains local and controls only actual work;
- production MegaMoEV2 autotuners use committed artifacts/defaults locally
  with zero forward-path broadcasts;
- config cache keys exclude rank-local tensor shapes;
- live tuning remains explicit (`FLYDSL_AUTOTUNE=1`) and supports an explicit
  process group;
- collective callers no longer take a rank-local `_collective_synced_keys`
  fast path.

SGLang now only converts its existing DP-global token counts/forward mode into
the public FlyDSL workload descriptor. It no longer picks SBM or invokes
private Stage1/Stage2 methods. The sitecustomize collective-disable monkeypatch
and `SGLANG_AMD_FLYDSL_MEGA_V2_SYNC_SBM` were removed.

Validation:

```text
FlyDSL autotune unit tests                         55 passed
V2 bs64 oracle                                    PASS, relL2 6.068e-02
V2 bs2048/8192 oracle                             PASS, relL2 5.352e-02 / 5.351e-02
uneven-rank Stage1                                PASS, identical config signatures
decode CUDA graph                                 PASS during full serving
full serving requests                             2048/2048
```

Full 8k/1k c256 result:

```text
V2 rank-local workaround       24,744.00 tok/s   TPOT 93.08 ms
V2 global workload API         25,907.20 tok/s   TPOT 88.90 ms
improvement                       +4.70%               -4.49%
V1 compact baseline            29,526.63 tok/s
remaining V2 gap vs V1            -12.26%
```

Post-fix eager-prefill trace rank-p50:

```text
metric                    V1 ms/fwd   V2 ms/fwd   V2 - V1
whole GPU union            1,410.137   1,614.613   +204.476
MoE union                    910.673   1,122.069   +211.397
Stage1                       554.754     783.059   +228.305
Stage2                       209.783     274.721    +64.938
combine                      145.984     197.747    +51.763
```

The fix removes the large config-divergence penalty and forward-path deadlock,
but does not make V2 beat V1. The remaining gap is now a real serving-level
critical-path issue rather than a rank-local artifact selection bug.

### Stage1 residual-gap debug matrix

Routing screen (256 requests):

```text
routing       V1 tok/s    V2 tok/s    V2 vs V1
natural       29,094.76   24,026.67    -17.42%
uniform       32,008.52   26,483.73    -17.26%
round-robin   32,303.50   26,654.66    -17.49%
```

Uniform routing improves both by about 10%, but leaves the relative gap
unchanged. Expert workload imbalance is rejected as the V2-specific root
cause.

Work/padding counters:

- V2 SBM64 `num_valid` matches V1 exactly on every rank and V2 remains faster
  in microbench;
- V2 SBM128 adds only about 2–3% padded rows;
- work/padding volume cannot explain the +41% serving Stage1 trace gap.

61-layer replay:

```text
routing replay                 V1 ms/layer   V2 ms/layer   result
uniform histogram                 3.2707        3.1632     V2 -3.3%
recorded per-layer routing         4.5880        5.2130     V2 +13.6%
recorded + alternating weights     4.5806        5.2693     V2 +15.0%
```

Real per-layer routing reproduces the direction of the serving flip, and
weight-pointer changes are not material. However, exact round-robin serving
still retains the full V2 gap, so routing/multi-layer accumulation is a
contributor rather than the sole root cause.

Additional falsifications:

- multi-stream OFF→ON is flat for both (`V1 29,094.76→29,069.60`,
  `V2 24,026.67→24,023.33 tok/s`);
- 256 MB cache traffic inserted between layers does not hurt V2
  disproportionately (`V1 3.3808`, `V2 2.6315 ms/layer`);
- env-gated device-clock phase instrumentation was attempted, but current
  FlyDSL inline `s_memtime/s_memrealtime` did not produce usable timestamps;
  the unsafe probe was fully reverted.

Current conclusion: config divergence, routing imbalance, padded work,
launch skew, weight hot-swap, cache traffic, and multi-stream interaction have
all been measured or falsified. The remaining prefill gap requires a supported
device-clock/rocprof phase decomposition of V2's planner/payload
rendezvous/GEMM under the full model; no compute/bandwidth-bound claim is made.

### Stage1 dispatch/GEMM split diagnostic

A compile-time `diagnostic_mode=dispatch_only` was added to the V2 Stage1
compiler with a distinct kernel name; production `full` kernel signatures are
unchanged. GEMM-only replay uses the existing standalone `compile_gemm1`
consumer over frozen dispatch payload/metadata.

Correctness gates:

```text
bs64 / bs8192 / uneven / idle-dummy ranks      PASS
metadata / route keys                           exact
sampled FP8 output                              exact
CUDA graph dispatch-only / GEMM-only / full     PASS
```

Phase timing:

```text
shape/config         dispatch   GEMM-only   sum      full     closure
decode bs64 SBM32      0.1304      0.2180   0.3485   0.3403    +2.4%
prefill SBM128         1.9666      2.1540   4.1206   3.1830   +29.5%
prefill SBM64          2.0770      2.1572   4.2342   2.6389   +60.5%
uneven ranks SBM128    2.1941      1.6216   3.8158   3.3131   +15.2%
idle-dummy SBM128      1.6902      1.5929   3.2831   2.2092   +48.6%
```

Dispatch and GEMM are both material. V2's advantage at SBM64 comes from much
greater overlap, not less standalone work; component times must not be summed
as the critical path.

Single-variable structure sweep:

```text
prefill SBM128             dispatch   GEMM   full      verdict
baseline dCU32 gm3           1.967    2.154  3.183     reference
dCU64 only                   1.596    2.155  2.670     micro +16%, serving +0.27%
dCU128 only                  1.258    2.191  2.957     overlap loss
active_expert_producer       2.226    2.154  4.158     reject
cooperative_payload_copy     3.796    2.151  4.250     reject
grid_mult=1                  1.955    2.128  2.578     server watchdog/deadlock
```

No candidate met the serving promotion gate. dCU64's large microbench win
collapsed to `24,026.67 -> 24,092.11 tok/s` (+0.27%) in a matched short
serving run. `grid_mult=1` hung all schedulers and was rejected.

The split diagnostic is useful and production-safe, but it confirms that the
remaining gap is an overlap/full-model interaction rather than a standalone
dispatch or GEMM kernel that can be optimized independently.

### PR #876 newest update (`2f9da1f4`, 2026-07-31)

The new head removes runtime autotune from MegaMoEV2 and replaces artifact
bundles with a deterministic `mega_moe_config.py` selector. Standalone
production-shape validation:

```text
decode tokens64 / MTPR64      PASS, Stage1 0.2379 ms, BF16 E2E 0.3459 ms
prefill tokens8192 / MTPR8192 PASS, Stage1 2.4589 ms, BF16 E2E 4.5166 ms
decode oracle relL2           5.979e-02 PASS
```

At equal compact MTPR8192/tokens32, the newest selector chooses
`t32x512x256, waves8, gm1, dCU128, async copy, b_nt3` and measures
0.2627 ms versus the older V2 0.3441 ms (-23.7%). This explains why mixed
serving decode improves even without fixed-slot.

Single-MTPR8192 serving completes:

```text
arm/head    total tok/s  prefill/rank  decode/rank  decode step
DP           30,535.06      6,899.73       897.40      35.66 ms
MegaMoEV1    29,094.76      7,239.44       791.13      40.45 ms
e092035d     26,438.62      6,345.57       838.94      38.14 ms
2f9da1f4     26,260.65      6,364.50       820.78      38.99 ms
```

The newest head is statistically flat/slightly worse than `e092035d` in this
single short run. It remains much faster than the original PR #876 V2
integration. `2f9da1f4` is 14.00% below DP and 9.74% below V1 total
throughput; its decode is faster than V1 but slower than DP, while prefill is
slower than both.

The requested dual MTPR (`prefill=8192`, `decode=128`) remains blocked:

- two MegaMoEV2 instances were prebuilt collectively in deterministic
  `8192 -> 128` order;
- latest runtime autotune is no longer involved;
- server completes prefill requests, then hangs at the prefill-to-decode
  transition and produces no completed benchmark result.

This confirms autotune ordering is not the blocker, but does not by itself
prove that two mori comb-op/state machines are structurally incompatible.

Decode MTPR256 was also tested on `2f9da1f4`:

- standalone tokens64 correctness passes (`relL2 4.682e-02`);
- MTPR256 is compact because `FIXED_SLOT_MAX_MTPR=255`;
- BF16 E2E is 0.3641 ms versus 0.3459 ms at fixed-slot MTPR64/128 (+5.3%);
- mixed prefill8192/decode256 serving completes multiple prefill batches, then
  hangs before decode and returns zero completed requests.

Therefore MTPR256 neither preserves fixed-slot performance nor resolves the
SGLang transition hang.

An explicit drain falsification was also run with CUDA graph disabled. Every
mode transition executed, on all eight ranks at the same timestamp:

```text
torch.cuda.synchronize()
EP NCCL barrier
mori shmem_barrier_all()
torch.cuda.synchronize()
```

Logs confirm all ranks crossed `prefill8192 -> decode128` and
`decode128 -> prefill8192` barriers together, yet the next MegaMoE work still
hung. This rejects a simple undrained-op explanation, but the standalone
result below also rejects generic process-global mori state non-isolation.

### Dual-instance standalone falsification (`2026-07-31`)

An 8-GPU harness outside SGLang instantiated independent A8192 compact and
B128 fixed-slot MegaMoEV2 operators with shared local expert weights. The
following all completed:

```text
single controls       A->A->A, B->B->B
unused coexistence    build A+B, run only A or only B
same capacity         A1->A2->A1, B1->B2->B1
cross capacity        A8192->B128->A8192 and reverse
no per-step drain     both cross-capacity directions
CUDA graph            A eager -> B graph replay -> A eager, full/no barrier
```

The state snapshot includes combine buffers, `running`, `count_done`,
`plan_ready`, `payload_ready`, parity/expected generations, P2P pointer
tables, dispatch tables, and xdev barrier buffers. A/B ranges had zero overlap
and each instance advanced only its own generation slots.

Verdict: the dual-instance hang is **not reproducible as a generic
FlyDSL/mori multi-instance failure**, even with no per-step barrier and B CUDA
graph replay. The remaining scope is SGLang integration: per-rank
continuous-batching call order, adapter instance selection, and graph/stream
state. The next diagnostic must record
`(rank, layer, instance, tokens, graph/eager, stream)` at the adapter boundary
and replay the exact observed sequence standalone.

Artifacts:

```text
/sgl-workspace/sglang-megamoe-pr876/repro_megamoe_dual_instance.py
/sgl-workspace/sglang-megamoe-pr876/run_megamoe_dual_instance_matrix.sh
/sgl-workspace/sglang-megamoe-pr876/megamoe_dual_instance_repro_summary.{md,json,csv}
/tmp/dual_repro_ABA_state.log
```

### Dual-instance SGLang root cause and fix (`2026-07-31`)

The serving-only failure was reproduced without profiler or synchronization
instrumentation. During the same MLP-sync step, rank 0 was `EXTEND` and chose
A8192 while ranks 1-7 were `IDLE` and chose B128. Replaying exactly that
A-vs-B split in the standalone harness deadlocked on the first divergent
forward.

Two rank-local decisions had to be removed:

1. Instance selection now uses the already all-gathered
   `ForwardBatch.is_extend_in_batch`: if any rank extends, every rank uses
   A8192; only all-decode/idle steps use B128.
2. When MLP TP gather is disabled, `global_num_tokens` intentionally contains
   only the local count. The MLP sync now also preserves
   `global_max_num_tokens`, and `MegaMoEV2Workload` uses that rank-identical
   value for config selection.

No new runtime collective, CUDA synchronize, or mode-switch barrier was
added. Dual MTPR remains opt-in (`decode_mtpr=0` keeps the safe single
MTPR8192 path).

Validation:

```text
eager 8192/64 c256             256/256 PASS
CUDA-graph 8192/64 c256        256/256 PASS
long 8192/1024 c256 + trace    256/256 PASS, 163/163 steps rank-consistent
final 8192/1024 c256           2048/2048 PASS
```

Final matched performance (`8192/1024`, c256, warmup512, measured2048):

```text
arm                         total tok/s  prefill/rank  decode/rank  step
DP                            31,358.52      6,906.53       895.00  35.75 ms
MegaMoEV1 3b0f818            29,831.76      7,300.63       790.65  40.47 ms
V2 e092035d                  warmup hang             no measured result
V2 2f9da1f4 single           28,041.60      6,371.86       819.44  39.05 ms
V2 2f9da1f4 dual 8192/128    27,550.61      6,366.34       804.26  39.79 ms
```

Dual is 1.75% slower than latest single V2. The fix resolves correctness and
stability, but dual MTPR does not improve matched end-to-end throughput. The
older V1 commit `6fb56158` cannot load against the current FlyDSL 0.2.4
runtime, so the matched V1 arm uses the current `mega_moe_v1` head `3b0f818`.

The fixed same-launch trace contains both A8192 compact prefill and B128
fixed-slot CUDA-graph decode:

```text
/tmp/megamoe_dual_global_same_launch_trace/profiles/
/tmp/megamoe_dual_global_full_final/megamoe/
```

### MegaMoEV2 end-to-end accuracy blocker (`2026-08-03`)

Full GSM8K was not run during the original V2 port; only operator oracle
relL2 and request-completion checks were performed. A matched 20-question,
5-shot smoke now shows a real model-level correctness regression:

```text
arm                              accuracy  invalid
MegaMoEV1 control                  0.850     0.000
MegaMoEV2 single eager             0.000     1.000
MegaMoEV2 dual graph 8192/128      0.350     0.100
MegaMoEV2 dual eager 8192/128      0.400     0.150
```

Raw single output often stops after one word (`Answer: Janet`, `Answer: It`).
Single and dual eager outputs matched exactly on 0/20 questions. Raising the
generation cap from 2048 to 8192 did not fix invalid output, and disabling
CUDA graph did not fix single V2. Therefore this is not an evaluator,
reasoning-truncation, or fixed-slot graph-only issue.

The dual-instance hang fix remains valid, but MegaMoEV2 is not shippable and
its throughput results are diagnostic-only until end-to-end correctness is
fixed. Do not run full dual GSM8K before resolving the single-V2 blocker.

#### Resolution

The local `mega_moe_v1` checkout was stale (`3b0f818`) and PR testing used
`2f9da1f4` source over a 0.2.4 native runtime. Remote `mega_moe_v1` was
force-updated to `567b43e`, including merged PR #876 (`dc8e153`) and
`ea4ce21 fix hang problem and code clean`. A clean worktree and matching
FlyDSL/MLIR native extension were rebuilt.

Production tensor replay then showed:

```text
packed W1/W2/scales V1 vs V2             exact
one-layer V2 vs V1 production artifact   max relL2 6.8e-5
uneven per-rank V2 vs V1                 max relL2 < 1.0e-4
61 repeated-layer V2 vs V1               max cosine diff 3.6e-4
```

Latest 20-question gates recovered:

```text
single graph   0.850, invalid 0
dual graph     0.900, invalid 0
dual eager     0.950, invalid 0
```

Full eager GSM8K also passed:

```text
single eager   0.925, invalid 0.001
dual eager     0.924, invalid 0.001
```

Full graph initially watchdog-hung because different DP ranks can replay
different local decode graph buckets, each with a captured V2 config.
MegaMoEV2 now accepts a separate `config_tokens`; SGLang captures every decode
graph bucket with the same config bucket 128. Token shapes remain graph-local,
but the fused EP config is rank-identical.

Final full graph gates:

```text
single graph   0.923, invalid 0.000
dual graph     0.920, invalid 0.001
```

End-to-end correctness is restored. Performance must be remeasured against
this latest built runtime; numbers from `2f9da1f4` remain historical only.

### Static EPLB comparison

FlyDSL MegaMoE originally cleared the standard expert parameters after
building shuffled weights, making dynamic EPLB migration see zero experts.
The isolated adapter was updated so the standard parameters are expert-major
views of the live shuffled buffers; a dynamic V1 smoke then completed one
61-layer migration in 6.46 seconds without errors.

For steady-state performance, one 8k/1k c256 routing record was captured and
used as the identical static EPLB placement for V1 and V2. MTPR remained 8192.

```text
backend  placement     total tok/s   TPOT ms   change vs own trivial
V1       trivial        29,526.63      59.54   reference
V1       static EPLB    30,442.98      58.51   +3.10%
V2       trivial        24,744.00      93.08   reference
V2       static EPLB    20,815.65     109.94   -15.88%
```

Static-EPLB V2 is 31.63% slower than static-EPLB V1.

Server-log compute cross-check:

```text
backend  metric                          trivial     static EPLB   change
V1       8192-token prefill/rank       7,280.96       7,915.35    +8.71%
V1       decode, 32 req/rank             787.20         780.79    -0.81%
V2       8192-token prefill/rank       5,634.79       4,575.37   -18.80%
V2       decode, 32 req/rank             679.55         632.27    -6.96%
```

Conclusion: EPLB is useful for V1 on this workload but is not a remedy for V2;
it amplifies the V2 regression. The mechanism is not yet proven. A likely
hypothesis is sensitivity of V2's artifact-selected persistent Stage1/Stage2
configs to the changed per-expert/rank token distribution, which requires a
matched EPLB-on trace or direct per-expert work measurement to verify.

Artifacts:

```text
/tmp/megamoe_pr876_eval/dp/
/tmp/megamoe_pr876_eval_old/megamoe/
/tmp/megamoe_pr876_eval_new/megamoe/
/tmp/megamoe_pr876_old_smoke.log
/tmp/megamoe_pr876_new_smoke.log
/tmp/megamoe_pr876_trace_dp/profiles/
/tmp/megamoe_pr876_trace_v1/profiles/
/tmp/megamoe_pr876_trace_v2/profiles/
/tmp/megamoe_pr876_trace_analysis/megamoe_trace_summary.{md,json,csv}
/tmp/megamoe_stage1_v1_t{32,8192}.log
/tmp/megamoe_stage1_v2_t32_m{32,64,128}.log
/tmp/megamoe_stage1_v2_t8192_m{32,64,128}.log
/tmp/megamoe_v2_e2e_t8192_{default,m64}.log
/tmp/megamoe_stage1_skew_v{1,2}_s{0,025,100}.log
/tmp/megamoe_stage1_routing_v{1,2}_recorded.log
/tmp/megamoe_stage1_uneven_v1.log
/tmp/megamoe_stage1_uneven_v2_{localconfigs,globalm128}.log
/tmp/megamoe_pr876_v2_{local,sync}_sbm_smoke/megamoe/
/tmp/megamoe_pr876_workload_api_full/megamoe/
/tmp/megamoe_pr876_workload_api_trace/profiles/
/tmp/megamoe_pr876_workload_api_trace_analysis/
/tmp/megamoe_pr876_workload_api_accuracy.log
/tmp/megamoe_stage1_debug_{natural,uniform,rr}_*/
/tmp/megamoe_stage1_work_v{1,2}_*.log
/tmp/megamoe_stage1_{repeat61,replay61,interleave}_v{1,2}.log
/tmp/megamoe_stage1_phase_split_*.log
/tmp/megamoe_stage1_phase_override_*.log
/tmp/megamoe_phase_dcu64_serving/
/tmp/megamoe_phase_gm1_serving/
/tmp/megamoe_pr876_latest2_{decode,prefill,accuracy}.log
/tmp/megamoe_pr876_latest2_single8192/megamoe/
/tmp/megamoe_pr876_latest2_dual_smoke/megamoe/
/tmp/megamoe_pr876_latest2_decode_mtpr256{,_accuracy}.log
/tmp/megamoe_pr876_latest2_dual256_e2e/megamoe/
/tmp/megamoe_pr876_latest2_dual128_barrier_eager/megamoe/
/tmp/megamoe_pr876_eplb_record/
/tmp/megamoe_pr876_eplb_static_v1/megamoe/
/tmp/megamoe_pr876_eplb_static_v2/megamoe/
/root/.cursor/projects/sgl-workspace/canvases/megamoe-pr876-first-round.canvas.tsx
/root/.cursor/projects/sgl-workspace/canvases/megamoe-pr876-analysis.canvas.tsx
```

Bring-up + evaluation of the **FlyDSL MegaMoE** a2a backend for DeepSeek-V4-Pro on gfx950,
ported onto the DSV4 sglang fork. Bottom line: **works, at parity, shipped as an opt-in backend;
no further low-risk perf lever found.**

## Status
- **Functional + correct**: V4-Pro (Fp8MoEMethod fp4-experts), a8w4, EP8/DP8, cuda-graph.
  full-set **gsm8k 0.937** (n=1319, cuda-graph) = parity with baseline (~0.929).
- **Perf (8k/1k, conc256, same sglang base)**: total tok/s **megamoe 29,482** vs **dp 30,711**
  (96%) vs **mori-ep cap=128 30,203** (98%). TPOT ~equal (59.1 vs 59.8). Decode MoE is at
  parity with the fair EP baseline and **2.2× faster than uncapped aiter**.
- **Conclusion**: goal met. Remaining gap to dp (~4%) is dp's opus a8w4 (no dispatch, hardware-
  tuned) + queueing; not a megamoe defect. No worthwhile low-risk optimization remains (see below).

## Code (all on `/sgl-workspace/sglang-upstream`, branch `feat/mori-ep-decode-small-cap`)
Ported the FlyDSL MegaMoE integration (from PR sgl-project/sglang#31322) onto the fork:
- `python/sglang/srt/layers/moe/mega_moe_flydsl.py` — NEW (the FlyDSL backend).
- `python/sglang/srt/layers/moe/mega_moe.py` — 3 dispatch hooks (`_use_amd_flydsl_mega_moe`).
- `python/sglang/srt/layers/quantization/fp8.py` — megamoe build hook (gated on the FlyDSL flag).
- `python/sglang/srt/environ.py` — `SGLANG_AMD_FLYDSL_*` knobs.
- `python/sglang/jit_kernel/include/sgl_kernel/utils.cuh` — kept HIP `getSMVersion` (needed for
  cuda-graph on this ROCm build; stash-pop conflict resolution).
Working-tree only (not committed). Also a separate clone `/sgl-workspace/sglang-megamoe`
(lixiufei-leo:dev/megamoe-pr) used for the first bring-up.

### Fixes made during bring-up (vs the upstream PR)
1. **Import path**: PR imports `kernels.mega_moe`; current FlyDSL is `kernels.moe.mega_moe`
   (accept both).
2. **forward**: PR calls `forward_bf16`; branch exposes `forward` (accept both, `slice_output=False`).
3. **a8w4 weight layout (accuracy-critical)**: a8w4 → `gate_mode=interleave`, so w1 needs
   `fp4_utils.shuffle_weight_w4(gate_up=True)` + `shuffle_scale_w4`, NOT plain `shuffle_weight`.
   Feeding plain-shuffled w1 → coherent-looking garbage. This is the #1 gotcha (see the posted
   comment on PR #31322).
4. **cohere2_moe `@strict`** (hf_hub≥1.x) — no-op strict (already fixed in the fork).
5. **`MORI_SHMEM_MODE`**: must be STATIC_HEAP (default) NOT ISOLATION for MegaMoE (else
   "not in symmetric heap [0x0,0x0)" NULL-deref).
6. **mtpr**: use 8192 (default). 16384 overflows int32 in the symmetric buffer sizing
   (`struct.error 'i' format`).

## How to run
`run_sgl_dsv4_unified.sh` now has a `megamoe` MODE (added):
```bash
cd /dockerx/home/wunhuang/tmp/useful-scripts/benchmarking/dsv4/
MEM=0.65 SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR=8192 MODE=megamoe PORT=8000 bash run_sgl_dsv4_unified.sh
# needs the FlyDSL workspace: /sgl-workspace/FlyDSL @ branch mega_moe_v1 (kernels.moe.mega_moe)
```
- FlyDSL: `/sgl-workspace/FlyDSL`, branch `mega_moe_v1` (596ec44). compiled `flydsl` pip = 0.2.2.
- MEM=0.65 (megamoe reserves a 40G mori shmem heap; 0.90 would OOM the KV cache).
- A/B driver: `/workspace/ab_megamoe_driver.sh` (dp / mori-ep cap=128 / megamoe, conc256 NP8/WARM2).

## Env knobs added (default-off, safe)
- `SGLANG_AMD_USE_FLYDSL_MEGA_MOE=1` — select FlyDSL over DeepGEMM megamoe.
- `SGLANG_AMD_FLYDSL_KERNELS_PATH` — FlyDSL workspace (unified script sets it).
- `SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR` (8192), `SGLANG_AMD_FLYDSL_MEGA_QUANT`.
- `SGLANG_AMD_FLYDSL_MEGA_TUNE_TOKENS` (experiment; stage1 tile — only 4%, leave 0).
- `SGLANG_AMD_FLYDSL_MEGA_DECODE_MTPR` (experiment; two-instance — HANGS, leave 0, see below).

## Traces (decode, conc128 8k/2000, cuda-graph, 8 ranks each) — for per-layer analysis
- `/workspace/trace_megamoe`   — megamoe mtpr=8192 (the canonical one)
- `/workspace/trace_dp`        — dp (non-EP) baseline
- `/workspace/trace_moriep`    — mori-ep cap=128
- `/workspace/trace_megamoe_tuned` — megamoe tune_tokens=64 (tile experiment)
- `/workspace/trace_mtpr512`   — megamoe mtpr=512 (compact-transition experiment)
- **B200 reference**: `/sgl-workspace/trace_hcOFF_msOFF_c256` — B200 DeepGEMM megamoe (conc256) +
  `server.log`. Shows the single fused `sm100_fp8_fp4_mega_moe_impl` kernel.
- Analysis method: cuda-graph durations ARE real on this build; take per-layer x61 kernels of ONE
  decode step, exclude spin-inflated glue (index_elementwise/alloc_decode). See ../TRACE_PROFILING.md
  + ../MORI_EP_DECODE_ROOTCAUSE.md §5x.

### Per-layer decode breakdown (conc128, ms/step, real compute)
| role | megamoe (mtpr8192) | dp | mori-ep |
|---|---:|---:|---:|
| MoE GEMM (gemm1+gemm2) | 17.5 (222+64us/layer) | 11.1 | 13.8 |
| comm (a2a / collective) | 2.18 (combine; dispatch fused into gemm1) | 2.01 (allgather+RS) | 6.19 (dispatch+combine) |
| **MoE total /step** | ~20.0 (mtpr8192) / 17.2 (mtpr512) | 15.3 | 20.0 |
megamoe stage1 `moe_gemm1` fuses dispatch+gemm1 (222us); vs the fair aiter `EpDispatch(30)+mfma_moe1(113)+glue(~50)≈193` it is competitive. stage2/combine are faster than mori-ep.

## Options evaluated and NOT taken (with reasons)
1. **stage1 tile tuning** (`tune_tokens`): −4% only. Not the lever.
2. **smaller mtpr, single instance**: decode faster but forces smaller prefill chunk → **e2e worse**
   (28.0k vs 29.5k). Wash.
3. **two-instance (prefill 8192 + decode small mtpr)**: right idea, but **HANGS in SGLang**.
   Standalone A/B switching, no-barrier switching, and B graph replay all pass, so isolated shmem
   groups are not yet justified; investigate adapter call order and graph/stream integration.
4. **Micro-harness finding (decisive)**: within compact mode mtpr is **irrelevant** (2048→8192
   flat); the server 512↔8192 delta was the compact/non-compact **mode switch** (+11.6%), NOT
   padding. megav1 compact is already 2.2× faster than uncapped aiter. See
   `FLYDSL_MEGAMOE_STAGE1_ANALYSIS.md`.
5. **combine-reduce fusion into gemm2 epilogue**: ~6% decode compute ceiling, but decode is already
   at parity (not the e2e bottleneck) AND it's the same e_vec=2 MFMA acc-layout dead-end as the
   masked-MoE work. High risk, low ROI.
6. **prefill/TTFT**: megamoe prefill compute is FASTER than dp (7360 vs 7005 tok/s); the higher
   TTFT is a queueing effect of the ~4% total-throughput deficit, not a prefill kernel gap.

## Structural gap vs B200 (for context)
B200 DeepGEMM megamoe = a SINGLE fused kernel (`sm100_fp8_fp4_mega_moe_impl`, gemm1+act+gemm2,
dynamic-M): 8.6us/token. AMD FlyDSL = single OP but 3 kernels (gemm1/silu/gemm2): ~17.9us/token
(~2×). Closing it needs CUDA-style gemm1+gemm2 fusion, but the payoff at decode is modest
(a2 small, compute-bound) and ~half the 2× is hardware (B200 fp4 MMA) that fusion can't recover.

## Related docs
- `FLYDSL_MEGAMOE_STAGE1_ANALYSIS.md` — the stage1/compact deep-dive (corrected root cause).
- `../FLYDSL_KERNEL_AUTHORING.md` — the kernel-authoring playbook used here.
- PR feedback posted: sgl-project/sglang#31322 (a8w4 interleave weight-prep + pin FlyDSL commit).
