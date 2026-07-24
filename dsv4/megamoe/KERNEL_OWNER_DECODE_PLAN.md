# MegaMoE kernel-owner optimization - profile-first budget + decision (2026-07-21)

Profile-first execution of the kernel-owner plan (`/root/.cursor/plans/megamoe_decode_kernel_plan_*.plan.md`).
Goal: speed up BOTH decode latency and prefill throughput; take the easiest overall win first; measure before
committing (two prior dispatch investments - recv 2B-i/2B-ii - did not pay off, see `COMPACT_SINGLE_ROUND_DESIGN.md`
§8.4/§8.5).

## Phase 0 - measured budget (micro-harness, eager, 8-rank, v4_pro a8w4)

Tooling built this session (was a capability gap): `rocprofv3` kernel-trace (per-kernel timing, eager = valid),
source-mapped **ATT** (installed `librocprof-trace-decoder.so` into `/opt/rocm/lib`; `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`;
`hotspot_analyzer.py`), and **PMC** (L2/HBM counters). Note: the ATT finalize crashes on multi-rank torch teardown
(GIL race) AFTER writing the trace - data is valid; must kill stragglers + free VRAM between runs.

### Coarse per-kernel (rocprof kernel-trace; absolute us are rocprof-inflated, ratios valid)
- Decode bs8: stage-1 fused `moe_gemm1_0` ~374 us/call dominates; fused stage-2 (`mfma_moe2_...fusedP2P_pe8`) ~68 us
  -> stage-1 ~= 85% of megamoe.
- Prefill bs2048: `moe_gemm1_0` ~1307 us/call; stage-2 ~575 us -> stage-1 still dominant.
- Clean cuda-event E2E (from the test): decode bs8 0.334 ms, bs64 0.426 ms; prefill bs2048 1.75 ms.

### Dispatch cross-PE round cost (clean A/B, most reliable)
`FLYDSL_MEGAMOE_STAGE1_ANALYSIS.md`: the count-round is **+11.6% of E2E at decode (bs32), mtpr-independent**, and
**~1-2% at prefill** (fixed ~12 us xGMI latency amortized over the big GEMM). This is the ceiling for any
dispatch-round removal, and it is fully serialized/exposed.

### ATT (source-mapped, single CU) - `moe_gemm1_0`
- bs8 AND bs64: the traced CU spends **~100% of stall cycles in `dispatch.py`**, 95% VMEM-wait, dominated by
  `dispatch.py:478` = the spin on the cross-PE count-round completion (`int32_wait_until_greater_than(a_meta,...)`).
- Interpretation: `att_target_cu=1` traces the **block0/coordinator CU**, which runs the cross-PE round then waits;
  it is block0-biased and does NOT give the kernel-wide compute-vs-comm split. What it DOES prove: the grid is
  **gated on the cross-PE round barrier** (block0 spins, all other blocks wait at `:478`) - i.e. the dispatch is a
  serialized latency phase, consistent with the clean ~11.6% decode bound.

### PMC - `moe_gemm1_0` GEMM memory health (bs64)
- L2 hit **51.5%** (real reuse; not pure streaming), 32B fraction **0.0%** (full 64B lines, no spatial waste),
  ~11 GB HBM reads. -> the GEMM memory access pattern is **clean**; there is no obvious bandwidth-waste bug.

### Occupancy (authoritative, from PMC CSV metadata)
`moe_gemm1_0`: **VGPR=228 (Accum=0), SGPR=112, LDS=100352 B (~98 KB/WG), WG=256, Grid=61440**.
- gfx950 (CDNA4): 160 KB LDS/CU, 512 VGPR/SIMD combined.
- limits: VGPR 512/228 = 2 waves/SIMD; LDS 160/98 = 1 WG/CU -> **1 wave/SIMD**; SGPR ample.
- => occupancy = **1 wave/SIMD, LDS-bound**. At 1 wave there is no wave-level latency hiding, so the GEMM leans
  entirely on software prefetch to keep the MFMA pipe fed. Reaching 2 waves/SIMD needs LDS <= 80 KB/WG.

## Decision gate

- **GEMM is the majority of BOTH decode and prefill**, is memory-clean, but runs at **1 wave/SIMD (LDS-bound)** ->
  the highest-leverage, lowest-risk (pure-compute, no cross-PE) target is the GEMM's occupancy / latency-hiding.
  => **Phase 1 = Track B (GEMM decode+prefill optimization).**
- Dispatch rounds (Track A1) cap at **~11.6% decode / ~1-2% prefill**, high risk (fixed-slot cuda-graph corruption +
  cross-PE) -> secondary; only if Track B plateaus and the decode round is still the residual.
- **Track C (comm/compute overlap) is DEPRIORITIZED** (data-driven change from the initial plan): dispatch is
  negligible at prefill (~1-2%, nothing worth overlapping) and cannot be hidden at decode (no concurrent compute).
  The generic "overlap helps prefill" assumption does not hold for this kernel because the dispatch round is a small,
  fixed, latency-bound slice.

## Phase 1 (Track B) plan
1. Low-risk first: sweep `compile_fused_moe_gemm1` knobs (`tile_m/tile_n/tile_k`, `b_nt`, `slice_k`, `waves_per_eu`,
   `use_async_copy`, `xcd_swizzle`, `gate_up_interleave`) A/B vs current at decode (bs8/64) + prefill (bs2048),
   oracle-gated, keep wins as additive tuned configs. Target the 1-wave/SIMD limiter (LDS footprint) and MFMA-pipe
   prefetch (skills: `gemm-optimization`, `prefetch-data-load`, `lds-optimization`, `docs/kernel_tuning_guide.md`).
2. If knobs plateau: targeted LDS reduction to reach 2 waves/SIMD, or deeper software-prefetch of W1/W2 (fp4) in the
   K-loop (`kloop.py` BLoader/KScaleLoader).

Guardrails: oracle relL2 from step 1; graph-safety; additive/default-off; compact stays the shipping fallback.

## Phase 1 (Track B) - executed + findings (2026-07-21)

Current tuned config (v4_pro a8w4, resolve_stage1_config): **tile_m=64, tile_n=256, tile_k=256, waves_per_eu=4,
use_async_copy=True, slice_k=1, xcd=0 (auto-disabled: gx=12 not a divisor of 8)**. LDS 98 KB/WG -> 1 wave/SIMD.

Experiments (env-gated override hook, since removed; A/B vs baseline decode 0.334/0.426, prefill 1.75 ms):
- **waves_per_eu=8** (safe occupancy hint): bs64 megav1 0.4296 vs 0.426 baseline = **no change** (noise), oracle PASS.
  Expected: a hint cannot beat the *hard LDS partition* (1 WG/CU); actual occupancy stays 1 wave/SIMD.
- **tile_k=128** (the real lever - halve X-ping/pong LDS to target 2 waves/SIMD): **BROKEN** - triggers
  `ZeroDivisionError: integer division or modulo by zero` in a layout computation on some ranks -> that rank dies
  -> the surviving ranks deadlock in the cross-PE rendezvous (7 GPUs at 100% spin, 1 idle; timeout). Confirms the
  fused megakernel's tile/LDS config is **tightly coupled** and only the tuned-table configs are valid.

**Phase 1 conclusion (evidenced).** The GEMM is genuinely occupancy-limited (1 wave/SIMD, LDS-bound), confirming
the Phase-0 hypothesis - but it is **not reachable by a config knob sweep**: the only occupancy-relevant safe knob
(waves_per_eu) can't beat the LDS partition, and the impactful knob (tile_k, to shrink LDS) is not a supported
config (layout ZeroDivision -> deadlock). Capturing the ~2x latency-hiding headroom therefore requires
**kernel-owner-level work**, precisely scoped:

1. Fix the tile_k (and general small-tile) layout math so smaller K-tiles are valid configs (root-cause the
   div-by-zero in the tile-chunk / scale-copy layout when tile_k < 256), THEN A/B tile_k=128 for 2 waves/SIMD.
2. OR reduce the fused kernel's per-WG LDS below 80 KB (2 WG/CU) by trimming a co-resident LDS buffer (dispatch
   histogram vs GEMM X-ping/pong vs a-scale staging share the pong arena) - staging/reusing more aggressively.
3. OR deeper software prefetch of W1/W2 (fp4) in the K-loop (`kloop.py` BLoader/KScaleLoader) to hide memory
   latency while stuck at 1 wave/SIMD (the `prefetch-data-load` pattern) - this helps without changing occupancy.

All three are pure-compute (no cross-PE correctness risk) but are real kernel edits with GPU-in-the-loop tuning,
oracle-gated. This is the highest-ROI remaining lever for both decode and prefill (the GEMM is the majority of both).

## Phase 2 (Track C) - not pursued (data-driven)
Deprioritized per Phase-0: dispatch is ~1-2% at prefill (nothing worth overlapping) and cannot be hidden at decode
(no concurrent compute). Overlap is low-value for this kernel. Not implemented.

## Track B execution (2026-07-21) - occupancy lever tested + pruned

Root-caused the LDS limiter: it is NOT the X tile (16KB) or tile_k, but the **CShuffle epilogue scratch
`lds_out = 4*tile_m*tile_n = 65536 B (66% of the 98KB)**, aliased at the pong-arena head (`build_lds_views`/
`plan_lds` in utils.py). tile_k=128 is a dead end (only -8KB AND `ZeroDivisionError` at `build_pipe_schedule`
utils.py:71 -> fixed with ceiling division, no-op for tile_k>=256).

- **A1 (divzero fix): DONE.** `pipe_k_unroll_packed`/`k_unroll_packed` now ceiling-divide; `pipe_n_phases`
  guarded with max(1,...). build_pipe_schedule compiles clean for tile_k 256/128/64. Robustness only.
- **A2 (tile_n=128 occupancy probe): tested, REGRESSES at all bs** (oracle PASS): bs8 0.349 vs 0.334 (+4.5%),
  bs64 0.469 vs 0.426 (+10%), bs2048 2.07 vs 1.75 (+18%). tile_n=128 halves lds_out -> ~65KB -> 2 waves/SIMD,
  but the smaller N-tile's efficiency loss (2x B-load traffic, less MFMA reuse) OUTWEIGHS the occupancy gain,
  worst at prefill. => at tile_n=256 the big-tile / 1-wave/SIMD trade beats tile_n=128.
  **[CORRECTION 2026-07-21, see "Occupancy lever - NOT actually disproven" below]** The stronger claim once
  written here ("occupancy is NOT a beneficial lever") is NOT supported by this test: tile_n=128 confounds two
  effects - occupancy UP (2 waves) AND tile-efficiency DOWN (2x B-traffic, less MFMA reuse). The regression is
  plausibly dominated by the efficiency loss, so a clean 2-wave-at-tile_n=256 config (via epilogue-LDS
  reduction) was never tested. Occupancy remains an OPEN lever, not a disproven one.
- **B (direct-store epilogue): deprioritized, but NOT disproven.** Its benefit = free lds_out for occupancy at
  FULL tile_n=256 (isolates the occupancy gain that A2/tile_n=128 confounded). Prior text pruned it citing "A2
  shows occupancy does not help" (a confounded inference) + a *separate* real risk (loses store coalescing at
  prefill). Net is genuinely unknown -> this is the Phase-4 proposal below. Not worth the
  epilogue-rewrite risk.
- **C (K-loop scheduler): not blindly tuned; instead found a SAFE knob win.** The single-CU ATT can't isolate
  GEMM K-loop stalls (the block0 cross-PE spin-wait busy-loop at dispatch.py:478 accumulates stall cycles across
  all waves -> swamps the stall accounting; GEMM lines show ~0 stall). Per the tuning guide ("don't tune without
  a trace"), I did NOT hand-edit the tuned `_interleaved_half` scheduler. Instead swept the safe `b_nt` B-load
  cache-modifier knob (fp4 weights streamed ~once/tile; non-temporal avoids L2 pollution). **This is a real,
  oracle-PASS, per-bucket win** (megav1 E2E ms, same-session A/B, current tuned b_nt=0):

  | bs | b_nt=0 (base) | b_nt=1 | b_nt=2 |
  | --- | --- | --- | --- |
  | 1    | 0.1529 | - | 0.1599 (+4.6%) |
  | 8    | 0.334  | 0.339 (+1.4%) | **0.323 (-3.2%)** |
  | 64   | 0.426  | 0.430 (+0.8%) | **0.407 (-4.6%)** |
  | 512  | 0.6841 | - | 0.6948 (+1.6%) |
  | 2048 | 1.75   | **1.699 (-2.9%)** | 1.837 (+5%) |

  Non-monotonic -> it is genuinely a per-bucket knob: **decode-serving range bs8-64 -> b_nt=2 (-3 to -4.6%);
  prefill bs2048 -> b_nt=1 (-2.9%); bs1 / bs512 keep b_nt=0.** Pure cache hint: no layout/LDS/grid/cross-PE
  change, oracle relL2 PASS at bs {1,8,64,512,2048}. Landed as an env-gated hook `MEGA_S1_BNT` in
  `resolve_stage1_config` (default-off -> shipping unchanged). **To ship the win, set `b_nt` per bucket in the
  MegaStage1 tune table** (decode buckets=2, prefill=1) after validating across other networks/quants.

**Also landed (robustness, no-op for shipping):** the `build_pipe_schedule` / `gemm1.py` div-by-zero fix
(ceiling `k_unroll_packed` + `max(1, pipe_n_phases)`) so tile_k<256 configs are compilable (verified
tile_k=256/128/64). Not a lever (tile_k reduction is a dead end), just unblocks future small-tile experiments.

## GEMM K-loop deep profile (2026-07-21) - MFMA-bound, not VMEM-bound

Unlocked isolated K-loop profiling despite the fused kernel (no GEMM-only compile path): the 8-rank ATT run
produces per-rank agent dirs, and on some ranks `att_target_cu=1` lands on a pure-GEMM CU (not block0). Scanning
the 8 prefill (bs2048) dirs, `ui_output_agent_11972` was 49% GEMM; post-filtering `code.json` to drop
`dispatch.py` lines then `hotspot_analyzer.py` gives a clean K-loop stall profile:

- **Stall-by-type (K-loop only): MFMA/FMA 48.5%, VMEM-wait 14.9%, VMEM-load 13.0%, barrier 9.7%, LDS-wait 5.2%.**
- Top line `kloop.py:725` (the `mfma_scale_f32_16x16x128_f8f6f4`, fp8xfp4 = 32-cycle) = 48.7% at 83.7% stall rate.
  (`gemm1.py:816` shows 29.7% but is the `kloop.run()` call site = debug-info aggregation artifact, Pattern 5.)

**=> The K-loop is MFMA-pipeline-bound, NOT VMEM-bound.** Consecutive MFMAs write independent accumulators, so
at 1 wave/SIMD the 48.5% MFMA stall is pipeline bubbles / issue-throughput with no second wave to fill them.
This overturns the plan's B-VMEM hypothesis: **Phase 2 (B-weight K+1 prefetch) downgraded** (VMEM ~28%, secondary).

### K-loop scheduling sweep (isched/ck_rate) + the measurement-noise lesson (2026-07-21)
Env-gated `isched`/`ck_rate` ([gemm1.py:174](/sgl-workspace/FlyDSL/kernels/mega_moe/gemm1.py)) sweep at prefill
bs2048 (oracle PASS): ck_rate {1,2,4}, isched 2 all landed 1.694-1.713 ms. BUT careful same-session A/B (100
iters x3) showed the **baseline itself varies 1.700-1.721 (~+-1.2%)** and ck_rate=2 (1.694-1.699) sits *within*
that band -> **no robust isched/ck_rate win**. Lesson (matches `kernel_tuning_guide` "gfx950 +-14%, median-of-7"):
at bs2048 the ~1-3% effects are at the noise floor; single-run A/B is unreliable there. Reverted the hook.
Phases 2/3 (B-prefetch, pipe-reshape) also **not pursued**: they target VMEM (secondary) on a compute/MFMA-bound
loop, so they cannot beat the noise floor either. The ~13% vs dp-opus is not reachable via FlyDSL-level K-loop
scheduling/prefetch tweaks (dp-opus is hand-tuned assembly with a different MFMA schedule/occupancy).

### The b_nt win re-validated ROBUST (2026-07-21)
Because bs2048 was noisy, re-ran the b_nt=2 decode claim at bs64 with **tight repeats**: baseline
0.4285/0.4283/0.4280 vs b_nt=2 0.4068/0.4054/0.4072 -> **-5.1%, non-overlapping clusters (real)**. bs64 variance
is tiny (~+-0.05%), unlike bs2048. So the b_nt per-bucket win (decode b_nt=2) is the one **robust, shippable**
GEMM win from all of Track B; isched/ck_rate/tile_n/direct-store/B-prefetch are pruned.

## E2E validation pass (2026-07-21) - regression + b_nt micro re-confirm + serving-resolution finding

Clean-cache validation of the default-off working tree (5 uncommitted files on `mega_moe_v1`:
dispatch.py/gemm1.py/mega_moe.py/utils.py/test_mega_moe.py). All micro runs `rm -rf ~/.flydsl /tmp/flydsl*`
per env change; GPUs freed to 0.30 GB between runs.

**1. Regression sanity (compact, no env, no --recv).** bs {1,8,64,512,2048}, iters 30: **all FULL-E2E PASS
(all 8 ranks)**, relL2 mega-vs-baseline 2.2-3.1e-3. Default `megav1` ms = 0.152 / 0.338 / 0.433 / 0.687 / 1.690
== historical baseline (0.334/0.426/1.75) -> **the default-off changes do NOT touch the shipping path.**

**2. b_nt micro A/B re-confirmed (iters 100, 3 reps each, cache rebuilt per config).** baseline (no env, b_nt=0
from the mtpr=8192 bucket) vs `MEGA_S1_BNT=2`; median `megav1` ms, all 24 oracle PASS:

  | bs | baseline b_nt=0 | b_nt=2 | delta |
  | --- | --- | --- | --- |
  | 1   | 0.1516 | 0.1621 | **+6.9% (worse)** |
  | 8   | 0.3369 | 0.3217 | **-4.5%** |
  | 64  | 0.4282 | 0.4059 | **-5.2%** |
  | 512 | 0.6816 | 0.6889 | **+1.1% (worse)** |

  bs8/bs64 clusters are non-overlapping (bs64 -5.2% matches the earlier -5.1%); bs1 and bs512 prefer b_nt=0.
  Confirms: **decode range (bs8-64) -> b_nt=2; bs1 / bs>=512 -> b_nt=0.** The win is real and per-bucket.

**3. CRITICAL: how b_nt is resolved != how the tune JSON is keyed (blocks a JSON-only serving ship).**
`resolve_stage1_config` resolves `b_nt` **once at `MegaMoE` construction** via
`mega_tuned_tile(..., mtpr=tune_tokens=max_tok_per_rank)`; the bucket lookup rounds **mtpr** (not the runtime
token count) up to the smallest `num_tokens >= mtpr`. **Both** the compact-primary kernel and the decode
fixed-slot kernel (`compile_fused_moe_gemm1`, mega_moe.py ~L640 and ~L678) are compiled with the **same**
`self._s1_b_nt`. Consequences at the serving config `mtpr=8192`:
  - The resolved bucket is `num_tokens=8192` = the **prefill** bucket -> `b_nt=0`. The committed JSON's
    per-decode-bucket values (currently `b_nt=3` on v4_pro/fp8_ocp num_tokens 1..512) are **never read** at
    mtpr=8192. (This is why the task-1 default matched b_nt=0, not the committed decode b_nt=3.)
  - Because decode+prefill share one resolved `b_nt`, **a JSON-only per-decode-bucket edit cannot make decode
    use b_nt=2 while prefill stays b_nt=0 in a single mtpr=8192 server.** The clean serving A/B the plan
    assumed (per-bucket JSON) is not achievable without a code change.
  - `decode_cap=0` by default -> the decode fixed-slot kernel isn't even built in the micro test (only the
    compact primary runs); env `MEGA_S1_BNT` forces b_nt on that primary, which is what all the micro A/B above
    (and the earlier table) actually measured.

  **To actually ship the decode b_nt=2 win to serving, one of:**
  (A) small kernel-owner change: have the decode fixed-slot kernel resolve its **own** b_nt from the
      `decode_cap` bucket (a second `mega_tuned_tile` lookup keyed on `decode_cap`), then set decode buckets
      b_nt=2 + keep the 8192/prefill bucket b_nt=0 in JSON -> genuine per-mode b_nt, clean serving A/B; OR
  (B) set the **8192 (prefill) bucket** b_nt=2 in JSON -> helps decode but regresses prefill (~+5% at bs2048
      per the b_nt table) -> muddies the throughput A/B (exactly the unfairness the env approach has).
  Recommendation: (A). This is a real (non-default-off) shipping edit and needs its own oracle+serving
  validation, so it was surfaced rather than silently applied.

**4. Option (A) LANDED + validated (2026-07-21).** Implemented per-mode b_nt: the decode fixed-slot kernel
now resolves its OWN `b_nt` from the `decode_cap` tune bucket (a 2nd `mega_tuned_tile` lookup keyed on
`decode_cap`, mega_moe.py in the `decode_cap>0` build block), while the compact/prefill kernel keeps the
mtpr-bucket value. `MEGA_S1_BNT` still force-overrides all modes. JSON updated (v4_pro/fp8_ocp): decode
buckets **8..256 -> b_nt=2**, **1/4/512/>=1024 -> b_nt=0** (matches the measured wins; was uniformly 3).
Validation (clean cache, mtpr=8192):
  - Default bs64 FULL-E2E **PASS**, megav1=0.4298 == baseline (prefill/compact path unchanged; JSON+code edits
    don't touch the default -> `decode_cap=0` skips the decode block and the 8192 bucket b_nt is still 0).
  - `--hang-iters 5 --hang-decode-mtpr 64`: log `[mega] decode kernel b_nt=2 (decode_cap=64 bucket) vs
    prefill/mtpr b_nt=0`; DUAL-MODE **OK**, decode finite, **corrupt_relL2=0.000e+00**.
  - `--hang-decode-mtpr 512`: decode kernel b_nt=0 (bucket 512) -> per-bucket switch confirmed; DUAL-MODE OK.
  So decode can now use b_nt=2 while prefill stays b_nt=0 in one server, correctness-neutral. NOTE for serving:
  it only activates when `SGLANG_AMD_FLYDSL_MEGA_DECODE_MTPR>0` (mega_moe_flydsl.py `decode_cap`); default
  serving (`=0`) is compact-only and unaffected. The clean serving throughput A/B (megamoe vs dp with
  decode_mtpr>0) is the remaining deferred step.

**5. Server gsm8k regression (2026-07-21).** Default megamoe server (MODE=megamoe, MEM=0.65, MTPR=8192, no
`MEGA_DECODE_MTPR` -> compact-only, changes provably inert): full-set **gsm8k Accuracy=0.934** (n=1319,
invalid=0.000) = parity with the 0.937 baseline (within n=1319 variance). Confirms the working-tree changes
(5 default-off files + the inert-at-default b_nt codefix) do not regress accuracy. GPUs freed to 0.30 GB after.

**6. Serving throughput A/B (2026-07-21) - b_nt serving-ship BLOCKED by fixed-slot decode instability.**
conc256, 8k/1k, NP_MULT=8 WARM_MULT=2, `sglang.bench_serving` (sglang-oai), same server args:

  | config | Total tok/s | Output tok/s | Median TPOT (ms) | note |
  | --- | --- | --- | --- | --- |
  | dp (ref) | 30,501 | 3389 | 59.2 | == historical 30,711 (within noise) |
  | megamoe compact (decode_mtpr=0) | **29,482** | 3276 | 59.0 | reproduces baseline EXACTLY; 96.7% of dp; shipping path |
  | megamoe fixed-slot decode, b_nt=0 (decode_mtpr=1024) | **HANG** | - | - | 8/8 schedulers watchdog-timeout (300s) -> crash |
  | megamoe fixed-slot decode, b_nt=2 | (not run) | - | - | same path as above -> would hang; skipped |

  Findings:
  - **The compact-only megamoe shipping path reproduces 29,482 tok/s exactly** -> the default-off working-tree
    changes (incl. the inert-at-default b_nt codefix) do NOT regress serving throughput. dp re-measured 30,501.
  - **The decode b_nt=2 win cannot currently be delivered to serving.** At the fixed serving `mtpr=8192`, the
    only way to give the decode kernel its own b_nt is the fixed-slot decode path
    (`SGLANG_AMD_FLYDSL_MEGA_DECODE_MTPR>0`). That path **deadlocks in real serving** (conc256, cuda-graph): all
    8 EP schedulers hit the 300s watchdog and `scheduler_0 crashed exit -3`. It hung with **b_nt=0**, so it is
    the **fixed-slot decode serving path itself** (the mode-divergence / cross-PE rendezvous instability the
    design docs warned about), NOT the b_nt cache hint. The micro dual-mode test (controlled prefill<->decode
    alternation) passes; real serving (mixed DP-rank modes + chunked prefill) does not.
  - Added a decode-only env knob `MEGA_S1_BNT_DEC` (overrides the decode kernel's b_nt only; prefill untouched)
    to make a clean serving b_nt A/B possible once the fixed-slot decode path is serving-stable. Default-off.
  - **This is NOT a new hang - it is the already-documented dual-mode blocker.** The shipping path is
    compact-only (`decode_cap=0`, no dual-mode) and did NOT hang (29,482). The hang appears ONLY because giving
    decode a separate b_nt REQUIRES the dual-mode fixed-slot decode kernel (`decode_mtpr>0`), and dual-mode is
    already marked **not-shippable** in `DUAL_MODE_DISPATCH_DIAGNOSIS.md` with **two** blockers: (1) the
    divergence / fixed-slot-races-mori-shared-state **hang under serving** (what this run re-hit), and (2)
    fixed-slot decode **cuda-graph correctness corruption** (dual-mode gsm8k 0.125 vs compact 0.854). So
    b_nt-to-serving is doubly gated (a hint riding a broken carrier), and the real unlock is **breakthrough #2**
    (`COMPACT_SINGLE_ROUND_DESIGN.md`: single cross-PE-round dispatch that drops dual-mode entirely, so one
    scheme serves decode+prefill and can carry per-mode b_nt).
  - **Conclusion:** decode b_nt=2 stays a micro-validated (-4.5..-5.2%), oracle-clean, default-off GEMM win;
    shipping it to serving is gated on the dual-mode carrier being fixed (breakthrough #2 or the two dual-mode
    blockers), NOT on the b_nt hint itself. Compact-only remains the stable shipping path (96.7% of dp,
    gsm8k 0.934).

## Net outcome of this project
- Built the profiling capability (ATT + PMC + hotspot) that was a stated gap; produced a measured decode/prefill
  budget and a data-driven track decision (which corrected the initial "overlap helps prefill" assumption).
- Track B, executed: tile_n=256 (1 wave/SIMD) beats tile_n=128 (2 waves) -> the big-tile trade is correct AT
  EQUAL EPILOGUE. **[CORRECTION: this does NOT prove "occupancy is not the lever"** - tile_n=128 confounds
  occupancy with tile-efficiency; a clean 2-wave-at-tile_n=256 via epilogue-LDS reduction is untested -> see
  Phase-4 proposal.] tile_k is a dead end (divzero fixed for robustness only).
- **Found a real, safe, per-bucket win: `b_nt` (non-temporal fp4 weight loads)** -> decode bs8-64 **-3 to -4.6%**,
  prefill bs2048 **-2.9%**, oracle-PASS, no fragility. Landed default-off (`MEGA_S1_BNT`); ready to bake into the
  tune table per bucket after cross-network validation. This is the concrete GEMM decode+prefill speedup the
  project was after (and it composes with, not conflicts with, the compact-only shipping dispatch).
- Remaining GEMM headroom (the ~13% vs dp-opus) would need MFMA-schedule co-design + a way to profile the GEMM
  K-loop in isolation (the single-CU ATT can't, because the cross-PE spin swamps stall accounting) - a genuine
  kernel-owner task, not a config knob.

## GEMM MFMA-schedule co-design (2026-07-21) - EXHAUSTED at FlyDSL level; needs assembly
Executed the kernel-owner MFMA co-design (plan `megamoe_gemm_mfma_codesign_*.plan.md`). Key alignment: a
*universal* K-loop schedule win modifies the single compact primary kernel, which serves BOTH prefill and
decode in serving (decode = cuda-graph replay, decode_cap=0) -> NO dual-mode needed (unlike b_nt). Full WIP
record: `GEMM_CODESIGN_WIP.md`.

**Measurement built (Phase 0).** Reusable A/B (bs64 tight +-0.19% = primary detector, bs2048 median-of-7
+-1.14%), whole-kernel MfmaUtil PMC (busy/active), and a deterministic 8-rank ATT dir-scorer (`att_score.py`)
that auto-finds the pure-GEMM CU + gives a GEMM-only stall breakdown. Baseline: bs64 0.4277 / bs2048 1.6979
ms; ATT GEMM K-loop **MFMA/FMA 49.5%** stall (top kloop.py:725), VMEM-wait 17.3%.

**Phase 1 (low-risk levers) - NULL.** Added env-gated, default-off, cache-separated hooks in `gemm1.py`:
`MEGA_S1_XDL_ARB` (disable_xdl_arb_stall / SCHED_MODE bit4 = back-to-back XDL issue), `MEGA_S1_POST_MISCHED_OFF`
(llvm enable-post-misched=false), `MEGA_S1_ISCHED`/`MEGA_S1_CKRATE`. All oracle PASS. bs64 flat within +-0.2%;
bs2048 best hint -0.5% (inside noise); MfmaUtil flat (33.0 vs 33.1). No robust win.

**Phase 2 (ILP restructuring) - NULL/regress.** Added `MEGA_S1_MFMA_PER_PHASE` (`build_pipe_schedule` override:
fewer phases = more back-to-back independent MFMA). mpp=1 (more phases) = noise; mpp=4 (fewer phases) = **+0.92%
WORSE** at prefill -> the shipping 4-phase split is already near-optimal; more MFMA ILP does not help.

**Decisive conclusion.** bs64 (tight) is FLAT across every schedule/ILP/flag perturbation tried (xdl, post-
misched, isched, ck_rate, mfma-per-phase) on top of prior nulls (isched/ck_rate, tile_n=128, waves_per_eu). The
GEMM runs at 1 wave/SIMD; the 49.5% MFMA stall at 1 wave is NOT reducible by any *schedule* lever tried.
**[CORRECTION 2026-07-21: "occupancy is provably non-beneficial" is too strong - see Phase 4 below.** The ATT
83.7% stall-rate on the MFMA line is exactly the symptom a SECOND wave could fill; whether the loop is
throughput-bound (2 waves useless) or bubble-bound (2 waves helps) is UNTESTED, because the only 2-wave config
(tile_n=128) confounded occupancy with tile-efficiency. Occupancy via epilogue-LDS reduction is an open lever,
not a closed one.]** No FlyDSL-expressible *schedule/ILP/flag* lever moves E2E. The
~13% vs dp-opus (hand-tuned assembly) requires assembly-level MFMA co-design (instruction selection / occupancy
structure the FlyDSL emitter cannot express) - OUT OF FlyDSL SCOPE, a distinct much-larger effort.
**Shipping unchanged: compact-only.** Post-edit DEFAULT path verified byte-identical (oracle PASS, bs64/bs2048
== baseline); all co-design levers are default-off in-tree as future probes. To chase the 13%, escalate to hand
assembly (surface to user).

> Correction (2026-07-21): the "MFMA-throughput-bound, occupancy non-beneficial, nothing left" framing above
> over-stated the occupancy verdict. Only the *schedule/ILP/flag* space is proven exhausted. Occupancy (2
> waves/SIMD) was NOT cleanly tested. See Phase 4.

## Phase 4 (PROPOSED, not yet executed) - occupancy via epilogue-LDS redesign -> 2 waves/SIMD
**The one architecturally-open lever to attack the 1-wave MFMA stall.** All Phase 1/2 schedule work reorders
*within one wave*; it cannot create a second wave. A second wave is exactly what fills the 32-cycle MFMA issue
bubbles that show up as the 49.5% MFMA / 83.7%-stall-rate at 1 wave. The blocker to 2 waves is LDS: 98 KB/WG,
of which the CShuffle epilogue scratch `lds_out = 4*tile_m*tile_n = 65 KB` (66%) dominates; 2 WG/CU needs
<= 80 KB/WG (VGPR allows 2 waves: 228*2=456 <= 512).

Hypothesis: **reduce/relocate the epilogue LDS (direct-store epilogue, or stream/shrink CShuffle, or move
lds_out out of the K-loop-resident arena) to hit <= 80 KB at tile_n=256** -> 2 waves/SIMD at FULL tile
efficiency -> second wave hides the MFMA bubbles -> attacks the prefill gap the schedule levers cannot.

Why it is genuinely open (not disproven): the only 2-wave datapoint (tile_n=128) confounded occupancy UP with
tile-efficiency DOWN and regressed; a 2-wave-at-tile_n=256 config was never built. dp-opus's edge is plausibly
exactly this different occupancy structure.

> **[RETRACTED 2026-07-21 later - the Phase-4 "2-wave" test was INVALID; occupancy is INACCESSIBLE in the fused
> design, NOT disproven. See MEGAMOE_OPT_DIRECTIONS_HANDOVER.md "Direction 0 (CORRECTED)".** The fused kernel is
> persistent: grid ~240 WGs / ~256 CUs = ~1 WG/CU (cross-PE grid barrier needs co-residency). Reducing LDS to
> ALLOW 2 WG/CU does not add a wave because the grid never launches a 2nd WG/CU -> the test ran at 1 wave/SIMD.
> Separately, the achieved-BW diagnostic (FETCH_SIZE/duration) shows the GEMM uses only ~15% of the ~8 TB/s HBM
> peak -> NOT bandwidth-bound either. Corrected: on-chip-latency-bound at a FORCED 1 wave; 2 waves needs a
> non-persistent (un-fused) GEMM. The struck-through claim below is retained only for history:]**
> ~~RESOLVED 2026-07-21 - Phase 4 EXECUTED, occupancy now CONCLUSIVELY DISPROVEN.~~ Built a
> 2-wave-at-tile_n=256 config (env-gated, default-off): `MEGA_S1_SPLIT_OUT=1` force-splits the 64KB CShuffle
> `lds_out` across pong+ping (LDS 100352->83968) AND aliases the phase-0 compact-count histogram at pong_offset
> (disjoint-in-time from GEMM) -> **LDS 80896 B, VGPR 224 -> 2 WG/CU ALLOWED by LDS (but NOT launched - see above)**, oracle
> PASS. A/B vs 1-wave baseline (bs64 0.4277 / bs2048 1.6979): 2-wave = **bs64 0.4281 (+0.09%), bs2048 1.7090
> (+0.65%)** = NO speedup (flat/slightly worse). => the GEMM is genuinely **32-cycle fp8xfp4 MFMA-THROUGHPUT-
> bound**; a 2nd wave finds no spare MFMA issue slot. Occupancy is NOT a lever (now a CLEAN test, not confounded).
> The ~13% vs dp-opus is dp-opus's non-fused, 2-wave-capable, hand-tuned-assembly GEMM vs our forced-1-wave
> FUSED GEMM - NOT an M-efficiency tradeoff [CORRECTED 2026-07-22: per-expert M is INVARIANT to EP degree; the
> earlier "EP small-M vs TP big-M" was wrong]. NOT a FlyDSL-reachable kernel lever. Phase 4 = closed negative.
> Ship compact-only. (Also note the Phase-4 "2-wave clean test" was itself later found INVALID - persistent grid
> forces 1 WG/CU; see the RETRACTED note above.)

Risks / kill criteria (why it may still net-lose):
1. Direct-store loses HBM write coalescing at prefill (large N) -> the store phase may regress by more than the
   K-loop gains. (Independent from the occupancy question; a real downside.)
2. The loop may be genuinely MFMA-throughput-bound (MFMA unit saturated) -> a 2nd wave finds no spare issue
   slot -> no gain. The whole-kernel MfmaUtil PMC (~busy/active) is dispatch-swamped and cannot decide this;
   the DECISIVE test is to actually build a 2-wave config and measure bs2048 (median-of-7) + bs64.
3. Real epilogue rewrite (kernel-owner effort, GPU-in-the-loop, oracle-gated). Even if it works, the 32-cycle
   fp8xfp4 MFMA remains the a8w4 hardware floor (only a4w4=16-cycle or hand-assembly go below it).

Plan (env-gated, default-off, additive; measurement infra from Phase 0 reused - bs64 tight detector +
bs2048 median-of-7 + auto-scored ATT for occupancy/stall-by-type):
- 4a. Cheapest decisive probe FIRST: check whether ANY LDS reduction reaches 2 WG/CU (query the compiled
  occupancy / `plan_lds`); if lds_out can be cut to <=80 KB at tile_n=256 by a *minimal* change (e.g. epilogue
  scratch reuse of the X ping/pong arena after the K-loop, not a full direct-store), A/B that first.
- 4b. If 4a shows 2 waves is reachable and helps: implement the proper direct-store / streamed-CShuffle
  epilogue (`epilogue.py` / `utils.py plan_lds` / `build_lds_views`), oracle-gate, measure store-coalescing
  cost at prefill vs K-loop gain.
- 4c. Gate: keep only if bs2048 median-of-7 robustly improves (non-overlapping clusters) AND bs64 not
  regressed AND oracle PASS. Else conclude occupancy is truly non-beneficial (now with a CLEAN test) and the
  13% is assembly-only.
