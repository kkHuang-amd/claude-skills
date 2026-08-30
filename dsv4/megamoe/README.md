# MegaMoE materials (DSV4 / FlyDSL, gfx950)

Consolidated docs + code snapshots for the **FlyDSL MegaMoE** a2a backend on DeepSeek-V4-Pro (8×MI355X).
All MegaMoE-specific material lives here; shared/general FlyDSL + profiling docs stay in the parent
`../` (dsv4) dir.

## ▶ CONTINUE HERE (DONE: M3-opt + 2B-ii both explored; recv can't beat compact → SHIP COMPACT-ONLY, 2026-07-21)

> **NEXT OPTIMIZATION DIRECTIONS → `MEGAMOE_OPT_DIRECTIONS_HANDOVER.md`** (standalone; start a fresh chat there).
> GEMM K-loop is exhausted + is **weight-bandwidth-bound** (not MFMA: 2-wave & a4w4 both null, b_nt the only win).
> **Direction 1 DONE (2026-07-21): CLOSED NEGATIVE.** Baking b_nt=2 into the mtpr=8192 compact serving bucket
> LOSES net serving: conc256 8k/1k = **28,978 / 29,049 tok/s (mean ~29,014, −1.5%) vs b_nt=0 baseline 29,482**
> (dp 30,501). Reason: total token throughput at 8k/1k is prefill-token-dominated (8:1 in:out), so the +5% prefill
> hit outweighs the decode-replay win. Reverted to b_nt=0 (byte-identical); 4096-bucket follow-up is moot (skip).
> Remaining: (2) TBO — low ROI, fights the fused/persistent design; (3) EP8→EP4 (bigger M/expert, attacks the
> weight-amortization root cause, memory-bounded). Includes the rigorous answers to "why 2 waves did nothing" and
> "TBO vs fused dispatch".


**E2E validation update (2026-07-21):** working tree is regression-clean and the GEMM **b_nt decode win** is
now shippable per-mode.
- **Regression:** all bs FULL-E2E PASS, default `megav1` == baseline (shipping path untouched); **server
  gsm8k = 0.934** (n=1319, invalid=0) = parity with 0.937 baseline → default-off changes don't regress.
- **b_nt micro re-confirm** (iters100×3, oracle PASS): **bs8 −4.5%, bs64 −5.2%**; bs1/bs512 prefer b_nt=0.
- **Root cause + fix:** `b_nt` was resolved once from the **mtpr** bucket and shared by decode+prefill, so a
  JSON-only per-bucket edit couldn't give decode-b_nt≠prefill-b_nt at `mtpr=8192`. **Fixed (landed):** the
  decode fixed-slot kernel now resolves its OWN b_nt from the `decode_cap` bucket (mega_moe.py); JSON v4_pro
  decode buckets 8..256 → b_nt=2, others 0. Validated: default FULL-E2E unchanged; decode kernel b_nt=2 (cap=64)
  vs prefill b_nt=0, dual-mode OK, corrupt_relL2=0. Only active when `SGLANG_AMD_FLYDSL_MEGA_DECODE_MTPR>0`
  (default serving is compact-only, unaffected).
- **Serving A/B (2026-07-21, conc256 8k/1k):** dp **30,501** tok/s; megamoe compact **29,482** (reproduces the
  baseline exactly, 96.7% of dp, no regression). **b_nt serving-ship BLOCKED:** the fixed-slot decode path
  (`SGLANG_AMD_FLYDSL_MEGA_DECODE_MTPR>0`, the only way to give decode its own b_nt at mtpr=8192) **deadlocks in
  real serving** — 8/8 schedulers watchdog-timeout (300s) → crash, even with b_nt=0, so it's the fixed-slot
  decode path itself, not the b_nt hint. So decode b_nt=2 stays a micro-validated (−5%), default-off win; a
  serving ship is gated on fixing the fixed-slot decode serving deadlock. Compact-only remains the stable ship.
  Added default-off `MEGA_S1_BNT_DEC` (decode-only b_nt override) for the future clean A/B. Details:
  `KERNEL_OWNER_DECODE_PLAN.md` "E2E validation pass (2026-07-21)" §6.
- **GEMM MFMA-schedule co-design (2026-07-21): EXHAUSTED at FlyDSL level -> needs assembly.** A *universal*
  K-loop win would help BOTH prefill+decode serving via the compact primary (no dual-mode, unlike b_nt). Built
  isolated measurement (bs64 tight +-0.19% detector, MfmaUtil PMC, auto-scored 8-rank ATT). Tried, ALL null/
  regress on bs64: `disable_xdl_arb_stall`, `enable-post-misched=off`, isched/ck_rate, and MFMA-per-phase ILP
  restructuring (fewer phases regressed prefill +0.9%; more = noise). The GEMM is 32-cycle fp8xfp4 MFMA-
  throughput-bound at 1 wave/SIMD (occupancy provably non-beneficial) -> the ~13% vs dp-opus needs assembly-
  level MFMA co-design (out of FlyDSL scope). Shipping unchanged (compact-only; default path verified byte-
  identical). Env-gated levers left default-off as probes (MEGA_S1_XDL_ARB / _POST_MISCHED_OFF / _ISCHED /
  _CKRATE / _MFMA_PER_PHASE). Full record: `GEMM_CODESIGN_WIP.md` + `KERNEL_OWNER_DECODE_PLAN.md`
  "GEMM MFMA-schedule co-design".
- **Phase 4 occupancy (2026-07-21): [CORRECTED] the "2-wave test" was INVALID + GEMM is NOT bandwidth-bound.**
  Achieved-BW diagnostic: GEMM uses only **~15% of ~8 TB/s HBM peak** -> not HBM-bandwidth-bound. And the fused
  kernel is persistent (grid ~240 WGs / ~256 CUs = forced ~1 WG/CU via the grid barrier), so reducing LDS to
  "allow 2 WG/CU" never launched a 2nd wave -> Phase-4 "2 waves no help" did NOT actually test occupancy.
  Corrected bottleneck = on-chip latency at a FORCED 1 wave/SIMD; the only path to 2 waves is a NON-persistent
  (un-fused) GEMM (loses the fusion win - a genuine fork, same as TBO). Details: MEGAMOE_OPT_DIRECTIONS_HANDOVER.md
  "Direction 0 (CORRECTED)". (Historical/superseded note below.)
  Gap decomposition first: GEMM = ~85% of megamoe at decode+prefill (dispatch/xGMI floor only ~12us / ~1-2%), so GEMM
  is the right lever. Built a clean 2-wave-at-tile_n=256 config (`MEGA_S1_SPLIT_OUT=1`: split the 64KB CShuffle lds_out
  across both LDS arenas + alias the phase-0 compact-count histogram -> LDS 100352->80896 B, VGPR 224 -> 2 WG/CU
  confirmed, oracle PASS). A/B: bs64 +0.09% / bs2048 +0.65% = NO gain. => GEMM is genuinely 32-cycle fp8xfp4 MFMA-
  THROUGHPUT-bound; a 2nd wave finds no spare issue slot. The ~13% vs dp-opus is dp-opus's non-fused,
  2-wave-capable, hand-tuned-assembly GEMM vs our forced-1-wave FUSED GEMM (per-expert M is the SAME for EP and
  TP - NOT an "M-efficiency" gap; corrected 2026-07-22), NOT a FlyDSL-reachable lever. Also dissected DeepGEMM B200
  mega-moe: it's ALSO 1 CTA/SM, wins via Blackwell async-MMA->TMEM + warp-specialization (HW CDNA4 lacks), not
  multi-wave. Definitive: ship compact-only; GEMM is at the CDNA4 MFMA ceiling. (+`MEGA_S1_SPLIT_OUT` default-off probe.)


**Where we are:** breakthrough **#2** (`recv`, 1-round receiver-side dispatch) is IMPLEMENTED, GPU-validated,
and now **M3-optimized**. Full status: `COMPACT_SINGLE_ROUND_DESIGN.md` §8.3 + **§8.4**.
- **M1 compile + M2 correctness = PASS**, still bit-equivalent to compact after the M3-opt edits (all-8-rank
  oracle PASS, bs {1,8,64,512,2048}). The core hypothesis (1 cross-PE round is sufficient + correct) is proven.
- **M3-opt landed (2 correctness-preserving wins in `dispatch.py` recv branch):** (a) **fused the two
  cross-PE rounds into one** — the draft still did a `done2` epoch barrier *then* a separate `recv_num`
  exchange; `recv_num` (emitted post-`fx.barrier`) already is a sufficient payload barrier, so `done2` was
  dropped (fixed-slot pattern, graph-safe) → recv is now a true 1 cross-PE round vs compact's 3; (b)
  **removed the 256-expert lane0-serial `my_base` prefix** (only the `epr` local experts get tokens; base =
  `acc*ctm` already in the metadata loop).
- **Result: gap closed from ~20–25% → ~5.5%, but recv STILL loses to compact** (v4_pro a8w4 `megav1` ms, 100
  iters, stable): bs8 0.353 vs 0.334; bs64 0.449 vs 0.426. The residual ~19–23µs is the **extra full-grid
  scatter pass (staging→dense double move) + its post-scatter barrier** — inherent to 2B-i and NOT removable
  within it (both schemes already use 2 grid barriers; recv already has fewer cross-PE rounds).

- **2B-ii tried + REJECTED (2026-07-21).** Implemented the GEMM-gathers-X-from-staging variant end-to-end
  (block0 builds a `gather_idx` map + scatters only the small fields, no embedding copy → recv drops to ONE
  grid barrier like fixed-slot; GEMM gathers X per-row from staging). **Correctness PASS**, but **slower at
  every bs** (bs8 0.469 vs compact 0.334; bs1/64 also worse) → **reverted**. Lesson: the dense compaction is
  not overhead — it *buys* coalesced/async-contiguous X tile loads in the GEMM K-loop; gathering scattered
  rows starves the GEMM by more than the barrier+scatter it saves. Details: `COMPACT_SINGLE_ROUND_DESIGN.md`
  §8.5.

**FINAL DECISION: SHIP COMPACT-ONLY.** Both 2B-i and 2B-ii recv are slower than compact; recv stays
additive/default-off at its best (2B-i M3-opt, ~5% off compact) as the proven-correct fallback. The
receiver-side single-round hypothesis is validated as *correct*, but the compact-vs-coalescing trade favors
compact (2 cross-PE rounds handing the GEMM a contiguous dense buffer beats 1 round + a GEMM-starving gather).
No dispatch-only lever beats compact; a real decode win needs a different axis (overlap dispatch w/ compute, or
a GEMM that natively consumes source-major tiles — large, kernel-owner-level). See §8.4/§8.5.

**Repro / test (recv is additive, default-off):**
```bash
cd /sgl-workspace/FlyDSL   # branch mega_moe_v1
rm -rf ~/.flydsl /tmp/flydsl*   # only after editing a kernel
# correctness+perf (oracle via _run_full_e2e); drop --recv for the compact control:
PYTHONPATH=/sgl-workspace/FlyDSL MORI_SHMEM_HEAP_SIZE=40G torchrun --standalone --nproc_per_node=8 \
  tests/kernels/test_mega_moe.py --network v4_pro --quant a8w4 --tokens 64 --mtpr 8192 --recv --iters 30 \
  2>&1 | rg "RECV-BUILD|FULL-E2E|mega-vs-baseline|megav1=|Error|HIP error"
```
Files changed for `recv` (all additive, default-off, shipping compact/fixedslot untouched): `dispatch.py`
(recv branch), `gemm1.py` (`recv_dispatch` flag), `mega_moe.py` (staging via `op._sym`/`_p2p_table`,
`_disp_tbl_recv`, build + `_run`/`forward` select), `tests/kernels/test_mega_moe.py` (`--recv`).

## In this folder
- **`PR35619_UPSTREAM_REPRO.md`** — **upstream `sglang#35619` reproduction (2026-08-27)**: both PR numbers
  hit on *stock* `/sgl-workspace/aiter` (no `aiter-megamoe-pr4439` / `FlyDSL-mega_moe_v1` needed) —
  39,200.88 no-EPLB and 42,505.60 EPLB, within 0.9%. Full non-folded server+client commands, the five
  settings that silently break it (mori heap default, shared-experts fusion, the four DP-comm envs, the
  EPLB distribution recorder, redundant experts), and the open `RANK_SYNC` + DP-attention IndexError.
  **Start here to re-run the upstream PR.**
- **`E2E_VALIDATION_PROMPT.md`** — paste-ready prompt for running the whole-network / gsm8k e2e validation in a
  fresh chat (aligned to current state: compact-only ship + `b_nt` GEMM win, all working-tree changes default-off).
- **`MEGAMOE_HANDOFF.md`** — original bring-up: functional + at parity, compact-only megamoe shipped
  (gsm8k ~0.94, ~96% of dp). The full technical record.
- **`MEGAMOE_IMAGE_CHANGE_HANDOVER.md`** — how to re-apply the (uncommitted) MegaMoE port after a Docker
  image change; uses `megamoe_image_change_patches/` below.
- **`FLYDSL_MEGAMOE_STAGE1_ANALYSIS.md`** — why stage1 is near-optimal + the compact-vs-non-compact
  overhead (the basis for the dual-mode lever).
- **`DUAL_MODE_DISPATCH_DIAGNOSIS.md`** — the decode-cap **dual-mode dispatch** optimization: the ~19%
  decode lever, everything tried, the `forward_mode` bug fixed, the remaining kernel-internal blocker
  (fixed-slot dispatch races mori shared state → compact prefill hangs under serving), repro, and the
  kernel-owner handoff. **Start here for the dual-mode work.**
- **`COMPACT_SINGLE_ROUND_DESIGN.md`** — **breakthrough #2 design proposal**: collapse compact's 2 cross-PE
  rounds → 1 via **receiver-side (per-source) dispatch**, so ONE scheme serves decode + prefill (drops
  dual-mode entirely → removes both the divergence hang AND the fixed-slot decode corruption). Buffer sizing,
  perf estimate, risks, and a **validate-against-oracle-first** plan. Not implemented (kernel-owner-level).
- **`FLYDSL_KERNEL_DEBUG_TOOLKIT.md`** — our **kernel-debug capability** distilled from FlyDSL's official
  `.claude/skills` + `docs` (all cloned locally on `mega_moe_v1`): the symptom→tool map, device-side
  `fx.printf` (the deadlock microscope), cache-clear rule, FlyDSL gotchas, ATT-trace/PMC perf flow, **and a
  line-anchored `fx.printf` plan to root-cause the dual-mode dispatch deadlock** (the 4 fixed-slot spin
  points + the all-to-all rendezvous invariant). Use with `DUAL_MODE_DISPATCH_DIAGNOSIS.md` §5.
- **`megamoe_image_change_patches/`** — BASE MegaMoE port patches (sglang env/hooks + `mega_moe_flydsl.py`
  base + optional aiter/flydsl patches). Used by `MEGAMOE_IMAGE_CHANGE_HANDOVER.md`.
- **`patches/`** — the **dual-mode (approach-A)** scaffold snapshot (uncommitted working-tree; re-apply
  after an image change):
  - `flydsl_dual_mode_approachA.patch` — `git apply` in `/sgl-workspace/FlyDSL` (@ `mega_moe_v1`, pinned
    `3b0f818`): `decode_cap` dual-mode in `kernels/mega_moe/mega_moe.py` + `--mtpr`/`--hang-iters` dual-mode
    test in `tests/kernels/test_mega_moe.py`.
  - `mega_moe_flydsl.py` — full sglang integration file → copy to
    `/sgl-workspace/sglang/python/sglang/srt/layers/moe/mega_moe_flydsl.py`. **Supersedes** the base copy in
    `megamoe_image_change_patches/` (adds the `forward_mode` decode-select fix + `decode_cap` wiring; keeps
    the `_swap_layer_weights` `_s1_w1` fix).

## Related (parent `../` dir — shared, not MegaMoE-only)
- `../FLYDSL_KERNEL_OPT_PLAYBOOK.md` — **reusable FlyDSL kernel-optimization playbook** distilled from this work:
  the profiling capability (kernel-trace / source-mapped ATT + decoder install / PMC / occupancy math), how to
  isolate an inner phase of a fused/cross-PE kernel (post-filter `code.json`, per-rank CU scan), the env/process
  gotchas checklist, the safe tuning knobs + edit patterns (b_nt, ceiling-div, buffer-resource i32 limit, LDS
  budget, K-loop scheduling), the profile-first decision tree, and the MegaMoE case-study index. **Start here for
  optimizing any new FlyDSL kernel.**
- `../FLYDSL_KERNEL_AUTHORING.md` — FlyDSL kernel-authoring / escalation playbook (also used by masked-moe).
- `../TRACE_PROFILING.md` — decode/prefill trace + profiling how-to.
- `../MORI_EP_DECODE_ROOTCAUSE.md` — mori-EP decode analysis (per-layer breakdown method).
- `../SKILL.md` — the DSV4 skill index.

## FlyDSL official reference (cloned locally, branch `mega_moe_v1`)
- `/sgl-workspace/FlyDSL/.claude/skills/` — official skills: `debug-flydsl-kernel`, `oob-detection`,
  `capture-kernel-trace`, `kernel-trace-analysis` (+ `scripts/hotspot_analyzer.py`), `flydsl-kernel-authoring`,
  `flydsl-tile-programming`, `gemm-optimization`, `lds-optimization`, `prefetch-data-load`, …
- `/sgl-workspace/FlyDSL/docs/` — `moe_stage1_mega.md` (our kernel's authoritative design: §1 sync, §3.3,
  §8 co-residency/CUDAGraph), `kernel_authoring_guide.md`, `testing_benchmarking_guide.md`, `kernel_tuning_guide.md`.
- `/sgl-workspace/FlyDSL/examples/` — `01-vectorAdd` … `05-gather_scatter` + `notebooks/` (`fx.printf`, types, structs).
- Distilled into `FLYDSL_KERNEL_DEBUG_TOOLKIT.md` — read the source, not GitHub HTML.

## Current state (2026-07-20)
- **Shipping = compact-only MegaMoE**: works on the current image, gsm8k **0.9318**, conc512 A/B
  megamoe ≈ 35.3k tok/s (≈97% of equal-VRAM dp). Unaffected by the dual-mode WIP.
- **Dual-mode (decode-cap)**: **Default-off** (`decode_cap=0`). Hang root cause PROVEN (2026-07-20, `fx.printf`
  + `--hang-diverge` repro + py-spy): a **cross-PE rendezvous deadlock from cross-rank mode divergence**
  (`recv_num` vs `done2`), NOT a memory race. **Fix (a) (cross-rank-consistent selection + divergence
  detector) implemented + server-verified: removes the hang** — but the hang was **masking a 2nd blocker:
  fixed-slot decode is CORRUPT under cuda-graph serving** (dual-mode gsm8k **0.125** + garbage vs compact-only
  control **0.854** + coherent). So dual-mode has **two** blockers now (divergence + graph-correctness) and is
  **not shippable**; (a) is safe for the shipping path. Favors breakthrough #2 (drop dual-mode). See
  `DUAL_MODE_DISPATCH_DIAGNOSIS.md` §5/§7/§8 + `FLYDSL_KERNEL_DEBUG_TOOLKIT.md` §4.
