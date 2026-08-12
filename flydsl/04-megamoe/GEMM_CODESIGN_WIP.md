# MegaMoE stage-1 GEMM MFMA-schedule co-design - WIP scratch (resume here)

Plan: `/root/.cursor/plans/megamoe_gemm_mfma_codesign_*.plan.md`. Goal: speed up the stage-1 fused GEMM
(`moe_gemm1_0`, majority of prefill+decode, MFMA-pipeline-bound at 1 wave/SIMD). A universal K-loop schedule
win ships to BOTH prefill+decode serving via the single compact primary kernel (NO dual-mode). Aggressive
target: approach dp-opus (~5-10% prefill). All edits additive/default-off, oracle-gated.

Repro env: FlyDSL=/sgl-workspace/FlyDSL @ mega_moe_v1; 8x MI355X gfx950; v4_pro a8w4, mtpr 8192.
Tooling (in /tmp): gemm_ab.sh (E2E bs64x3+bs2048x7), gemm_summ.py, pmc_run.sh+pmc_mfma_summ.py (MfmaUtil),
att_capture.sh+att_score.py (8-rank ATT, auto-finds GEMM CU, GEMM-only stall breakdown).

## Phase 0 - baselines (2026-07-21, DONE)
- E2E (oracle 10/10 PASS): bs64 median **0.4277 ms** (spread 0.19%, tight), bs2048 median **1.6979 ms**
  (spread 1.14% = documented noise). Primary win-detector = bs64 (tight); bs2048 = median-of-7.
- MfmaUtil PMC (whole-kernel moe_gemm1_0, busy/active): median **33.1** (SQ_INSTS_VALU_MFMA_F8=17,891,328
  = work-invariant sanity const). Note: whole-kernel incl. dispatch spin -> may be insensitive; weight bs64.
- ATT GEMM K-loop (dir auto-scored; 7/8 dirs = block0 100% dispatch spin, 1 dir 34% GEMM): **MFMA/FMA 49.5%**,
  VMEM-wait 17.3%, barrier 10.6%, VMEM-load 8.4%, LDS-wait 5.3%. Top line **kloop.py:725** (mfma_scale) 49.7%;
  gemm1.py:816 33.5% (kloop.run() call-site aggregation artifact, ignore). Confirms MFMA-issue-bound.

## Phase 1 - low-risk levers (DONE 2026-07-21: NO robust win)
Env-gated (default-off) in gemm1.py: MEGA_S1_XDL_ARB (L1 disable_xdl_arb_stall / SCHED_MODE bit4),
MEGA_S1_POST_MISCHED_OFF (L2 llvm enable-post-misched=false), MEGA_S1_ISCHED / MEGA_S1_CKRATE (L3). Each gets
a module_name tag (_xdl/_pmoff/_is*_r*) so the cache separates variants. All oracle PASS. Screen (bs64 x3 tight,
bs2048 x3) vs baseline (bs64 0.4277 / bs2048 1.6979):
- L1 xdl:   bs64 0.4276 (-0.02%), bs2048 1.6886 (-0.55%)  | MfmaUtil busy/act 33.0 vs base 33.1 = flat
- L2 pm:    bs64 0.4296 (+0.44%), bs2048 1.6966 (-0.08%)
- L1+L2:    bs64 0.4296 (+0.44%), bs2048 1.6987 (+0.05%)
- L3 is=2:  bs64 0.4273 (-0.09%), bs2048 1.6944 (-0.21%)
- L3 ck=2:  bs64 0.4274 (-0.07%), bs2048 1.6880 (-0.58%)
Gate review: NONE beat noise on the tight bs64 detector (+-0.2%); bs2048 hints (L1/L3ck2 ~-0.5%) sit inside the
+-1.14% floor; MfmaUtil flat -> no bubble reduction. Confirms MFMA bubbles are NOT addressable by hint-stream/
flag tweaks (consistent with prior isched/ck_rate null). Escalate to Phase 2 (ILP restructuring) per aggressive
target. Levers kept default-off in tree (harmless).

## Phase 2 - ILP restructuring (DONE 2026-07-21: NO headroom -> throughput-bound)
Added env-gated `MEGA_S1_MFMA_PER_PHASE` (gemm1.py + build_pipe_schedule in utils.py): overrides MFMA-groups-
per-phase (fewer phases = more back-to-back independent MFMA ILP). Default unchanged (len//4). Cache-separated
(_mpp tag). Sweep vs baseline (bs64 0.4277 / bs2048 1.6979), oracle PASS:
- mpp=1 (prefill 8 phases, max interleave): bs64 0.4280 (flat), bs2048 1.6870 (-0.64%, within +-1.14% noise)
- mpp=4 (prefill 2 phases, more back-to-back MFMA): bs64 0.4281 (flat), bs2048 1.7135 (**+0.92% WORSE**)
- mpp=8 (1 phase): INVALID config (B-load distribution loop needs >=2 phases -> KeyError). Not a data point.
Finding: more phases = noise-level; fewer phases = regression -> the shipping 4-phase split is already near-
optimal. More back-to-back MFMA ILP does NOT help => the 49.5% MFMA/FMA stall is the irreducible 32-cycle
fp8xfp4 MFMA throughput at 1 wave/SIMD, NOT removable pipeline bubbles.

### DECISIVE CONCLUSION (all FlyDSL-level K-loop co-design exhausted)
bs64 (tight +-0.19% detector) is FLAT across ALL variants tried this project: L1 xdl_arb, L2 post-misched,
L1+L2, isched=2, ck_rate=2, mpp=1, mpp=4 (+ prior isched/ck_rate, tile_n=128, waves_per_eu, b_nt=cache-only).
The GEMM is MFMA-throughput-bound (32-cycle fp8xfp4) at 1 wave/SIMD; occupancy is provably non-beneficial
(tile_n=128 -> 2 waves regresses, prior Track B) and LDS-bound. There is NO FlyDSL-expressible schedule/ILP/
flag lever left that moves E2E beyond noise. The residual ~13% vs dp-opus (hand-tuned assembly) requires
assembly-level MFMA co-design (different instruction selection / occupancy structure / schedule the FlyDSL
emitter cannot express) -> OUT OF FlyDSL SCOPE. Recommendation: ship compact-only (unchanged); do not bake any
Phase-1/2 lever (all default-off, kept as harmless probes). To chase the 13% would require dropping to hand
assembly (a different, much larger effort) - surface to user.

## Phase 3 - ship/conclude (DONE 2026-07-21: CONCLUDE, ship compact-only)
No robust win in Phase 1 or 2 -> "conclude" branch (not "ship a win").
- Regression guard: post-edit DEFAULT (no env) oracle 6/6 PASS, bs64 0.4282 (+0.12%) / bs2048 1.6923 (-0.33%)
  == pre-edit baseline within noise. Default module_name is byte-identical (xdl/pm/mpp tags all empty when
  unset; build_pipe_schedule mfma_per_phase=None takes the original branch) -> shipping path UNTOUCHED. No need
  to re-run server gsm8k/throughput (default provably == the already-validated gsm8k 0.934 / compact 29,482).
- All Phase-1/2 levers left in-tree but DEFAULT-OFF (env-gated, cache-separated): MEGA_S1_XDL_ARB,
  MEGA_S1_POST_MISCHED_OFF, MEGA_S1_ISCHED, MEGA_S1_CKRATE, MEGA_S1_MFMA_PER_PHASE. Harmless future probes.
- Outcome: FlyDSL-level GEMM K-loop *schedule/ILP/flag* co-design is EXHAUSTED with no reachable headroom.
  CORRECTION (2026-07-21): "occupancy provably non-beneficial" was too strong - only the schedule space is
  proven exhausted. Occupancy (2 waves/SIMD) was NEVER cleanly tested (the one 2-wave config tile_n=128
  confounded occupancy with tile-efficiency). The ATT 83.7% MFMA stall-rate at 1 wave is exactly what a 2nd
  wave could fill -> see Phase 4.

## Gap decomposition (2026-07-21) - is the megamoe cost GEMM or dispatch? -> GEMM (~85%)
rocprofv3 --kernel-trace, per-kernel MEDIAN us (rocprof-inflated, RATIOS valid; /tmp/kt_parse.py):

| bs | FUSED stage1 (disp+GEMM1) | MEGA stage2 (GEMM2+combine) | cross-PE dispatch* | cross-PE combine* |
| --- | --- | --- | --- | --- |
| 64 (decode)  | 297.6 | 99.0 | ~24 | ~29 |
| 2048 (prefill)| 1025.0 | 577.1 | ~200 | ~62 |
(* standalone baseline ep_dispatch/ep_combine kernels = a2a proxy; the FUSED kernel OVERLAPS payload write with
compute, so only the ~12us xGMI sync ROUND is truly exposed - these proxies OVER-estimate the fused a2a cost.)

Decomposition (megamoe MoE = stage1 + stage2):
- **bs64 decode**: total ~396us. GEMM compute (stage1 GEMM1 ~274 + stage2 GEMM2 ~70) ~= **86.7%**; cross-PE
  a2a ~53us (13.3%), of which the IRREDUCIBLE sync round is only ~12us (~3%). Matches doc's "+11.6% count-round".
- **bs2048 prefill**: total ~1602us. GEMM compute ~= **84%+**; a2a sync ~1-2% (doc), payload overlapped.
- Sanity: FUSED megamoe stage1 (297us) is FASTER than the ATOM baseline standalone GEMM1 alone (412us) -> our
  GEMM1 tile is already better-tuned than the ATOM baseline's mixed_moe_gemm1. The ~13% headroom is vs *dp-opus*
  (hand-tuned assembly), NOT vs this baseline.

**CONCLUSION: GEMM is the dominant cost (~85%) at BOTH decode and prefill; the cross-PE dispatch/sync floor is
small (~12us decode / ~1-2% prefill) and NOT the bottleneck.** => Phase 4 (GEMM occupancy) targets the majority
of the time and is well-justified. Leverage: closing the ~13% GEMM headroom (vs dp-opus) x 85% weight ~= ~11%
potential megamoe speedup - more than enough to cover the 3.3% serving gap-to-DP (megamoe 96.7% of DP). The DP
gap is therefore addressable via GEMM, NOT blocked by the a2a hardware floor. (Caveat: this decomposes megamoe
vs ATOM baseline, not vs DP directly; DP's own GEMM not measured here.)

## DeepGEMM sm100_fp8_fp4_mega_moe dissection (2026-07-21) - REFRAMES Phase 4
Fetched deepseek-ai/DeepGEMM `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` (1461 lines).
Key architectural findings vs our FlyDSL kernel:

1. **CUDA is ALSO 1 CTA/SM** - `__launch_bounds__(kNumThreads, 1)` (line 55). It does NOT beat the occupancy
   problem by running 2 CTAs/SM. So my Phase-4 premise ("CUDA reaches 2 waves, we don't") was WRONG.
2. **It hides latency via WARP SPECIALIZATION + async-MMA, not multi-CTA.** Distinct warp roles in one CTA:
   dispatch warps (a2a pull via TMA/NVLink), TMA-load warp for A, TMA-load warp for B, ONE async-MMA issue warp
   (tcgen05 UMMA), scheduler warp, and separate EPILOGUE warpgroups. Producer/consumer via mbarriers + deep
   SMEM pipeline (kNumStages).
3. **The MMA accumulator lives in TENSOR MEMORY (TMEM), not registers.** tcgen05 UMMA is ASYNC: the MMA warp
   issues, results land in TMEM, and SEPARATE epilogue warps read TMEM (SM100_TMEM_LOAD) to do SwiGLU+cast+
   store - overlapping with the next MMA. This DECOUPLES MMA-issue from epilogue, so epilogue SMEM/regs do NOT
   throttle MMA occupancy.
4. **Full gemm1+SwiGLU+gemm2+combine in ONE persistent megakernel** (task scheduler BlockPhase Linear1/Linear2
   + combine loop at the end) - no a2 HBM round-trip. Handles imbalance via per-expert actual token counts
   (scheduler.get_num_tokens) + BLOCK_M tile-padding + ring buffers + persistent task distribution across SMs.

**Why this matters for us (CDNA4 has NO equivalent):** MI355X MFMA is synchronous, accumulators are VGPRs,
there is NO TMEM and NO async-MMA. So CUDA's main latency-hiding mechanism (async-MMA -> TMEM -> separate
epilogue warps) is NOT transferable - it's Blackwell hardware. On CDNA4 the ONLY SIMD-level MFMA-latency hider
is a second wave -> **Phase 4 (2 waves/SIMD) remains the correct AMD-appropriate lever**, it's just NOT how
CUDA does it.

**Transferable ideas (real):**
- (a) **Decouple epilogue store-staging from the MMA-feeding SMEM.** CUDA's epilogue SMEM (`smem_d`) doesn't
  steal from the A/B pipeline SMEM because accum is in TMEM. On AMD our `lds_out` (CShuffle, 65KB) permanently
  reserves LDS the K-loop needs -> 1 wave. Since our epilogue runs AFTER the K-loop, `lds_out` could REUSE the
  freed X ping/pong arena instead of co-reserving -> frees LDS -> 2 waves at tile_n=256. == **Phase 4a**.
- (b) **Warp specialization** (some waves K-loop MFMA, others epilogue) - harder on AMD (MFMA->VGPR coupling,
  handoff via LDS = what CShuffle already does), lower ROI.
- (c) Full gemm1+gemm2 fusion - low ROI (a2 round-trip is VMEM-store ~0%, not our bottleneck) + massive rewrite.

Net: Phase 4a (epilogue LDS reuse -> 2 waves) is the concrete, AMD-appropriate lever; DeepGEMM confirms the
principle (decouple epilogue SMEM from MMA SMEM) even though its mechanism (TMEM) is HW we lack.

## Phase 4 (DONE 2026-07-21) - occupancy 2 waves/SIMD CLEANLY TESTED -> DOES NOT HELP (throughput-bound)
Achieved a clean 2-wave-at-tile_n=256 config (the test prior work never did; tile_n=128 confounded occupancy
with tile-efficiency). Two env-gated (default-off) LDS cuts in gemm1.py/utils.py:
- `MEGA_S1_SPLIT_OUT=1`: force `split_lds_out` (halve the 64KB CShuffle scratch across pong+ping arenas) -> LDS
  100352 -> 83968 B. Still 1 WG/CU (2x83968 > 163840 by ~2KB). Oracle PASS.
- + alias the compact-count histogram (3KB, phase-0, disjoint-from-GEMM) at pong_offset -> LDS **80896 B**
  (< 81920) + VGPR 224 -> LDS/VGPR now ALLOW 2 WG/CU. Oracle PASS.
  **[CORRECTION 2026-07-21 later: 2 waves was NEVER actually achieved.** The fused kernel is persistent - grid =
  ~240 WGs on ~256 CUs = ~1 WG/CU (the cross-PE grid barrier needs co-residency). Allowing 2 WG/CU via LDS does
  NOT add a wave because the grid only launches 1 WG per CU. So this "2-wave test" ran at 1 wave/SIMD -> the
  null result below does NOT test occupancy. See "CORRECTION" at end + MEGAMOE_OPT_DIRECTIONS_HANDOVER.md.]**
Result (A/B vs 1-wave baseline bs64 0.4277 / bs2048 1.6979; oracle PASS):
- 2-wave split: **bs64 0.4281 (+0.09%, flat), bs2048 1.7090 (+0.65%, within noise / slightly worse)**.
=> **2 waves/SIMD gives NO speedup.** The GEMM is genuinely 32-cycle fp8xfp4 MFMA-THROUGHPUT-bound: a 2nd wave
finds no spare MFMA issue slot; the 49.5% MFMA stall is irreducible latency/throughput, NOT removable bubbles.
~~Occupancy is now CONCLUSIVELY ruled out~~ **[RETRACTED 2026-07-21 later: this test was INVALID.** It ran at
1 wave/SIMD (persistent grid = ~240 WGs / ~256 CUs = 1 WG/CU; LDS headroom for 2 WG/CU is unused because the
grid never launches a 2nd WG/CU). So occupancy was NOT tested. Occupancy is **INACCESSIBLE in the fused/
persistent design (grid barrier forces 1 WG/CU), not disproven.** Plus the achieved-BW diagnostic shows
HBM is only ~15% utilized -> NOT bandwidth-bound either. Corrected bottleneck = on-chip latency at forced
1-wave; the only way to 2 waves is a NON-persistent (un-fused) GEMM. See MEGAMOE_OPT_DIRECTIONS_HANDOVER.md
"Direction 0 (CORRECTED)".]**

### FINAL (all FlyDSL levers exhausted, decisively)
**[CORRECTIONS 2026-07-22 - two claims in this + the Phase-5 section below are SUPERSEDED; see
MEGAMOE_OPT_DIRECTIONS_HANDOVER.md for the accurate synthesis: (1) NOT bandwidth-bound - measured achieved HBM =
~15% of peak; the bound is on-chip latency at a FUSION-forced 1 wave/SIMD. (2) "EP small-M vs TP big-M" is FALSE
- per-expert M is INVARIANT to EP degree (an expert processes its full global routing regardless of sharding).
The ~13% vs dp-opus is dp-opus being a non-fused, 2-wave-capable, hand-tuned-assembly GEMM vs our forced-1-wave
FUSED GEMM (same per-expert M, same FLOPs/rank), NOT an M-efficiency tradeoff.]**
schedule/ILP/flag (Phase 1/2) = null; occupancy 2-wave (Phase 4, clean) = null. The EP GEMM is at its CDNA4
MFMA throughput ceiling. The ~13% vs dp is the structural **EP small-per-expert-M vs DP/TP-MoE big-M** GEMM-
efficiency tradeoff (big-M fills MFMA tiles better) + dp's hand-tuned assembly - NOT a fixable FlyDSL kernel
lever. Only a different quant (a4w4=16-cycle MFMA, hurts accuracy) or Blackwell-style async-MMA (HW we lack)
would move it. Ship compact-only. All Phase 1/2/4 levers default-off in-tree as probes.

## Phase 5 (2026-07-21) - warp-spec? + a4w4? -> both dead ends; DIAGNOSIS REFINED to weight-memory-bound
User asked: can we do NVIDIA-style warp-specialization, or try a4w4 (16-cycle MFMA)?

**warp-spec: architecturally NOT applicable/beneficial on CDNA4.** NVIDIA warp-spec overlaps the tensor cores
(a SHARED SM unit) with TMA/ALU/epilogue on SEPARATE units, enabled by ASYNC-MMA (wgmma/tcgen05 -> regs/TMEM).
CDNA4: (1) MFMA is synchronous and occupies the issuing SIMD - there is no async-MMA to overlap with other
work on the same SIMD; (2) MFMA units are PER-SIMD, so to max throughput you want ALL 4 warps doing MFMA, not
some specialized to load/epilogue (that would REDUCE aggregate MFMA throughput); (3) the producer (async
global->LDS copy) is already overlapped via ping/pong. And Phase 4 already showed the loop is not latency/
bubble-bound. => warp-spec cannot add MFMA throughput here; not pursued.

**a4w4: MEASURED, no speedup -> DECISIVE evidence the GEMM is NOT MFMA-arithmetic-bound.** a4w4 = fp4 activation
(half the bytes of fp8) + fp4xfp4 MFMA = **16-cycle** (vs 32 for a8w4 fp8xfp4). Micro A/B v4_pro (oracle PASS):
- bs64: a8w4 0.4277 -> a4w4 0.4443 (**+3.9% SLOWER**; partly the untuned a4w4 tile - v4_pro a4w4 has no tune
  entry), bs2048: a8w4 1.6979 -> a4w4 1.6785 (**-1.1%, flat**).
=> halving the MFMA cost produced ~ZERO GEMM speedup. Plus a4w4 = big accuracy loss (fp4 activations; micro
oracle relL2 0.086 vs a8w4 1e-5; real gsm8k would drop). **Dead end: no speed + worse accuracy.**

**REFINED DIAGNOSIS (correcting the "MFMA-throughput-bound" label):** three orthogonal probes now agree the
GEMM is NOT bound by MFMA arithmetic:
- 2 waves/SIMD (Phase 4): no help -> not MFMA-latency/bubble-bound.
- a4w4 16-cycle MFMA: no help -> not MFMA-issue/throughput-bound.
- b_nt weight non-temporal load: -5% (the ONLY win) -> it's the WEIGHT-MEMORY path.
=> The K-loop ceiling is **WEIGHT-memory traffic / K-loop memory feeding** (fp4 weights streamed each tile -
IDENTICAL in a4w4 and a8w4, which is why a4w4 doesn't help), not MFMA and not activation bytes. Weight-side
levers already explored: b_nt (won, -5%), B-prefetch K+1 (downgraded), XCD-swizzle L2 reuse (v4_pro gx=12 can't
- see moe_stage1_mega.md 5). So the GEMM is still effectively exhausted, but the ACCURATE bound is weight
memory, not MFMA. The ~13% vs dp remains the structural EP small-M vs TP big-M tradeoff (big-M amortizes the
weight load over more tokens -> higher weight-bandwidth efficiency, exactly the axis we're bound on).

## Phase 4 ORIGINAL PROPOSAL (superseded by the DONE result above)
The one architecturally-open lever. Blocker to 2 waves = LDS 98KB/WG, dominated by CShuffle epilogue scratch
lds_out=4*tile_m*tile_n=65KB (66%); need <=80KB (VGPR ok: 228*2<=512). Hypothesis: reduce/relocate epilogue
LDS (direct-store or stream/shrink CShuffle, or reuse X ping/pong arena post-K-loop) to hit 2 waves at FULL
tile_n=256 -> 2nd wave hides the 32-cycle MFMA bubbles the schedule levers cannot. Risks: (1) direct-store
loses prefill store coalescing (may net-lose independent of occupancy); (2) loop may be truly MFMA-throughput-
bound (2 waves useless) - MfmaUtil PMC is dispatch-swamped, so the DECISIVE test is to actually build a 2-wave
config + measure bs2048 median-of-7 + bs64; (3) real epilogue rewrite. Full plan (4a cheap LDS-reuse probe ->
4b direct-store -> 4c gate) in KERNEL_OWNER_DECODE_PLAN.md "Phase 4". Not started.

## Phase 2 - ILP restructuring
(pending)

## Phase 3 - ship/conclude
(pending)
