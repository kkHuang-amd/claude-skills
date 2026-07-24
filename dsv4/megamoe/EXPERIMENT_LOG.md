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
