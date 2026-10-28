Owner node: ctheliosmc-rck-g02-h13-17.rck.dcgpu

# MoE kernel redesign for gfx1250 (design draft, 2026-10-28) -- NO CODE UNTIL REVIEWED

Inputs: `MOE_PREFILL_h13-17_1028.md` (measurements, ATT, upstream check), `GFX1250_CLUSTER_NOTES_h13-17_1028.md`.
Tags: (M) measured on h13-17, (D) derived arithmetic, (A) assumption, (TBD) must be measured before deciding.

## CONTINUE HERE

**Status:** draft v1 reviewed (decisions below); M1 done (WMMA 8 cyc issue / 27 cyc latency, ~1024 VGPR/SIMD). M2 done (L2 aggregate ~88 B/clk/CU is the scarcest; per-CU port ~150). Best paper design: D1 R128xC512 (1.27 Mcyc) or + weight multicast (0.76). M3 done: cluster_load_b128 multicast correct, 1.75x (CS=2) / ~2x (CS=4) delivered per CU; collapses under large drift without barrier. P1 HIP prototype built (bit-exact) but 4.04 Mcyc vs today ~2.62: HBM-bound weights (0.62 Mcyc) were missing from the bounds, latency not hidden at 1 wave/SIMD. ATT done (v3: LDS 25% / async 20% / weights 17% / WMMA 26%; v4 spills drain prefetch). Next: P1 v5 per "P1 ATT" fix plan. Upstream (aiter/FlyDSL main 10-28) has no kernel that avoids the LDS bound.
**Next:** user review -> run microbenchmarks M1-M5 (section 5) -> fill the TBD numbers -> pick design -> hot-loop prototype (P1).
**Pass criteria for P1:** one expert, M=256, N=8192, K=16384 fp4 x fp4: cycles per k128 step per wave < 400 (current 581) with
correct output vs the current kernel; VGPR / occupancy reported.

## 1. Workload (per layer, per rank, TP8 12L proxy, model dim H = 16384, top-16 of 512 experts)

| | EP8 (64 local experts) | EP1 (512 experts, inter sharded) |
|---|---|---|
| routes per rank | 2T | 16T |
| rows / expert | T/32 (256 @ T=8192, 32 @ 1024, <= 2 in decode) | same |
| gate-up GEMM | N = 8192, K = 16384 | N = 1024, K = 16384 |
| down GEMM | N = 16384, K = 4096 | N = 16384, K = 512 |
| FLOP @ T=8192 | gate-up 4.40 T, down 2.20 T | same |
| weight bytes (fp4) | gate-up 4.29 GB, down 2.15 GB | same |
| token input to gate-up | 134 MB | 1.07 GB (route-expanded) |
| per-route down output (bf16) | 0.54 GB | **4.29 GB** (written, then re-read by gather-reduce) |

(D) Weight arithmetic intensity of gate-up = T/8 flop per weight byte (1024 @ T=8192, 128 @ T=1024).
Server runs fp4 act x fp4 weight, mxfp4 (e8m0 per 32) scales, swiglu between stages, top-k weighted gather-reduce.

Time today (M, rocprofv3, rank 0, one 8k prefill = 12 layers):
| | gate-up | down | fused quant | gather-reduce | MoE total | share of prefill |
|---|---|---|---|---|---|---|
| EP8 | 45.5 ms | 26.7 | 10.7 | 10.2 | 93.4 | 41% |
| EP1 | 47.6 | 37.2 | 21.5 | 12.8 | 119.3 | 47% |

## 2. Hardware limits, in cycles (clocks are power-limited: ~1.0-1.4 GHz in op tests and serving (M))

| quantity | value | tag |
|---|---|---|
| CUs per partition | 96 | M (torch) |
| LDS read throughput | 153 B/clk/CU (2 WG x 4 waves), 217 (2 x 8 waves) | M |
| LDS read with concurrent TDM writes | ? | TBD (M4) |
| WMMA `v_wmma_scale_f32_32x16x128_f4` issue interval per SIMD | ? (ATT window: ~18 cyc/WMMA/SIMD incl. stalls) | TBD (M1) |
| L2 -> VGPR (global_load_b128) throughput per CU, and latency | ? | TBD (M2) |
| HBM | ~8 TB/s per partition | A (spec) |
| VGPRs: real per-wave budget and occupancy of the current kernel | doc says acc 256 VGPR but "kernel VGPR 192" -- inconsistent | TBD (M5) |
| LDS capacity per CU | ? (current assumes ~112 KB per WG, 2 WG/CU) | TBD (M5) |
| energy | fewer bytes moved -> higher clock: multicast fast mode = same cycles, +12% clock (1110 vs 990 MHz) | M |

## 3. Where the cycles go today (EP8 gate-up, tile 128x256x128, 4 waves w1x4, 2 WG/CU)

- (M) ATT: s_wait_dscnt 41% of stall+duration, WMMA 33%, barrier 12%; 581 cyc per k128 step per wave.
- (D) LDS traffic per WG per k128 = 48 KB ds_load + 24 KB TDM writes = 72 KB -> 403 MB per CU per layer ->
  **1.86-2.63 Mcyc** at 217-153 B/clk. (M) measured 2383 us at ~1.1 GHz = **2.62 Mcyc** -> matches the 153 B/clk bound exactly.
- (D) WMMA count 3.36e7 = 87.4k per SIMD -> 0.70 Mcyc at 8 cyc/WMMA, 1.40 Mcyc at 16. Unknown which (M1).
- (M) not HBM-bound: 13-40% of HBM BW, L2 hit 65-81%.
- EP1 down (K = 512): fixed cost per tile dominates (ATT: s_wait_tensorcnt 35% incl. output-store drain, barrier 29%, WMMA 5%);
  3.1 ms/layer vs ~0.45 ms compute bound (~7x).

Register-blocking arithmetic per wave (r token rows, c weight cols, per k128; WMMA = 32 weight cols x 16 token rows):
| r x c | acc VGPR | LDS read B/WMMA, both operands in LDS (today) | LDS read B/WMMA, weights direct to VGPR | weight L2->VGPR B/WMMA |
|---|---|---|---|---|
| 128 x 64 (today) | 256 | 768 | 512 | 256 |
| 64 x 128 | 256 | 768 | 256 | 512 |
| 32 x 256 | 256 | 1152 | 128 | 1024 |
| 128 x 128 | 512 | 512 | 256 | 256 |
(D) With weights direct, LDS bytes per WMMA = 32768 / c and weight L2 bytes per WMMA = 32768 / r: c trades LDS traffic for L2 traffic.

Note on "swap-AB": the WMMA already puts weights on the 32-wide side and tokens on the 16-wide side, so minimum token padding is
16 rows already (upstream CSV uses tile_m 16/32 for small T). Swap-AB therefore only changes WG-level tiling, not WMMA padding;
the small-T cost is per-tile fixed cost + weight re-reads (consistent with tile_m=32 being rejected earlier).

## 4. Candidate designs

### D1. Weights global -> VGPR, tokens TDM -> LDS (= Track A, generalized)
- Waves in a WG share the token tile (w1x4) but own disjoint weight columns, so LDS sharing of weights buys nothing.
- (D) LDS traffic per WG per k128 72 KB -> 40 KB (r=128, c=64) -> 1.03-1.46 Mcyc bound; with c=128 (r=64) LDS read halves again
  but weight L2 traffic doubles (512 B/WMMA).
- Weight HBM/L2 bytes unchanged vs today (TDM already reads B per WG); L2 -> VGPR rate must sustain ~61 B/clk/CU at the new
  speed (D, r=128) -> M2 decides whether c=128 is affordable.
- Cost: VGPRs. Weight fragment = 16 VGPR per 32 cols per k128; c=64 -> 32 VGPR per k-step; hiding global latency needs
  prefetch depth ~ latency / step time (TBD M2) -> 64-128 VGPR. acc 256 + weights 64-128 + token frags 64 + misc -> > 256
  -> likely 1 wave/SIMD unless gfx1250's VGPR budget is larger (M5). With 1 wave/SIMD all latency hiding is software pipelining.
- Optional: token tile multicast across a 1x4 cluster (A shared by N-neighbours) to cut L2 bytes/energy -- only with matched loop
  bounds and NO 16-WG barrier (see cluster notes); measure, never assume.
- Risk: if WMMA issue is ~16 cyc (M1), D1 lands at ~1.4 Mcyc = WMMA-bound, still -45% vs 2.62. If ~8 cyc, LDS stays the limit at
  1.0-1.46 and c=128 becomes interesting.

### D2. Tokens global -> VGPR, weights TDM -> LDS (mirror of D1)
- Natural for w4x1 (waves own token rows, share weights). LDS read 32768 / r B/WMMA (256 at r=128).
- Needs WG token tile 4 x r = 512 rows > 256 rows/expert at T=8192 -> heavy padding / cross-expert WGs. Rejected for prefill
  at our rows/expert; reconsider only if rows/expert >= 512.

### D3. Persistent, stream-K-style scheduler over (expert, n-tile, k-chunk) work
- Targets the per-tile fixed cost (EP1 down 7x off bound, small-T prefill, decode): overlap next tile's prologue loads with this
  tile's epilogue store; one launch, grid = #CUs x waves.
- Independent of D1 (D1 is the hot loop, D3 is the outer loop); can be combined.
- Existing knob `persistent_workers2` made EP1 down worse (2331 -> 2652-6396 us) -- that implementation does not overlap
  prologue/epilogue; D3 must, otherwise no gain.

### D4. Fused stage chain / epilogue fusion
- EP1 writes 4.29 GB per-route bf16 output per layer and re-reads it in gather-reduce (D: ~1.07 ms/layer at 8 TB/s ~= the measured
  12.8 ms / 12 layers). Fuse the top-k weighted reduce into the down epilogue: options (a) fp32/bf16 atomics into the token output
  (needs gfx1250 packed bf16 / f32 atomic throughput -- TBD), (b) per-token ownership after a route sort (one WG reduces all routes
  of a token -- breaks expert-major weight reuse), (c) two-level: reduce routes of the same token that land in the same WG first.
- Fuse act + fp4 quant of the gate-up output into the gate-up epilogue (EP1 fused quant 21.5 ms is route-expanded quant of the
  INPUT: quantize once per token, then gather fp4 rows -- 16x fewer quant bytes in EP1).
- Bigger structural change; do after D1 and D3 prove out.

## 5. Microbenchmarks before choosing (each small, GPU0, report cycles with sampled clock)

- M1: WMMA fp4 issue rate: `v_wmma_scale_f32_32x16x128_f4` back-to-back, independent accumulators, 1 / 2 waves per SIMD.
- M2: global_load_b128 L2-hit throughput per CU and load-to-use latency, 1 / 2 waves per SIMD, with the preshuffled weight layout
  (16 B per lane at `kgrp*256 + lane16*16`).
- M3: (optional) cluster_load / TDM multicast of the token tile across 1x4 with no cluster barrier vs separate loads, several
  processes (bimodality check) -- also answers the open slow-mode question (`AITER_FLYDSL_MXFP4_CLUSTER_M=1/2`,
  `AITER_FLYDSL_DISABLE_CLUSTER_SYNC=1` on the existing kernel first; env only).
- M4: LDS read throughput with concurrent TDM writes (today 24 of 72 KB per step are TDM writes).
- M5: code-object VGPR / LDS / occupancy of the current kernel (resolve the 192 vs 256 VGPR question; max VGPR per wave on gfx1250).

Decision rule: M1 + M2 give the D1 bound = max(LDS, WMMA, L2) cycles; go ahead with P1 if the bound is <= 0.7x of 2.62 Mcyc.

## 6. Plan

- P1: hot-loop prototype of D1 (one expert, fixed M=256 N=8192 K=16384, no routing, FlyDSL 0.3.4.1 local env), compare cycles per
  k-step with the current kernel's ATT (581). Keep the current preshuffle / scale layout so the weights need no re-pack.
- P2: integrate into `mxfp4_preshuffle_gfx1250_tdm.py` behind a constexpr / env switch (`AITER_TDM_B_DIRECT`), op test EP8 + EP1,
  then server A/B.
- P3: D3 persistent outer loop for EP1 down / small T; P4: D4 fusions.

## Microbenchmark results

### M1: WMMA fp4 issue rate (2026-10-28, h13-17, GPU0; `/workspace/proxy_work/bench/wmma_fp4_rate.hip`, HIP builtin
`__builtin_amdgcn_wmma_scale_f32_32x16x128_f4(a v16i32, b v8i32, short modC, c v16f32, 0,0,scaleA, 0,0,scaleB, false,false)`,
asm verified = `v_wmma_scale_f32_32x16x128_f4`, operands register-resident, cycles = `s_get_shader_cycles_u64` per wave)
| independent accumulators per wave | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| cyc/WMMA, 1 wave/SIMD | 27.2 | 13.6 | 9.0 | 8.5 | 8.3 |
- **Latency ~27 cyc, issue interval ~8 cyc per SIMD** (2 or 4 waves/SIMD: per-SIMD 8.0-8.3). Need >= 4 independent WMMAs in flight
  per SIMD. Peak = 4 x 131072 / 8 = 65.5k FLOP/clk/CU = 6.3 MFLOP/clk per partition (6.9 PFLOP/s at 1.1 GHz).
- **VGPR file ~1024 per lane per SIMD:** the 354-VGPR kernel (2,8) runs 2 waves per SIMD concurrently (per-wave 16.3 cyc/WMMA at
  2 WG/CU = sharing one WMMA pipe; at 4 WG/CU still pairs). -> up to ~512 VGPR/wave keeps 2 waves/SIMD. The "> 256 VGPR ->
  1 wave/SIMD" assumption in section 4 was WRONG.
- The clock / aggregate columns of the tool are unreliable (kernels 0.03-1 ms, realtime-derived MHz inconsistent); use cycles only.
- (D) EP8 gate-up WMMA bound = 87.4k WMMA/SIMD x 8 = **0.70 Mcyc** (today 2.62 -> WMMA busy ~27%).
- D1 re-evaluated (w1x4 WG, per k128; LDS = token TDM write + 4 waves' token reads):
  | per wave r x c | acc VGPR | LDS B/WMMA | LDS bound Mcyc (153-217 B/clk) | weight L2->VGPR B/WMMA | L2 rate needed at 0.70 Mcyc |
  |---|---|---|---|---|---|
  | 128 x 64 | 256 | 640 | 1.03-1.46 | 256 | 128 B/clk/CU |
  | 64 x 128 | 256 | 320 | 0.52-0.73 | 512 | 256 B/clk/CU |
  | 128 x 128 | 512 (1 wave/SIMD) | 320 | 0.52-0.73 | 256 | 128 B/clk/CU |
  -> M2 (L2 -> VGPR B/clk/CU and latency) picks between 64x128 and 128x128.

### M2: global_load_b128 -> VGPR throughput and latency (2026-10-28, h13-17, GPU0; `/workspace/proxy_work/bench/l2_vgpr_bw.hip`)
(first run was invalid: a 64-bit `%` in the address cost hundreds of cycles; fixed to power-of-2 masks, asm checked for no div.)
- Load-to-use latency (dependent chase): ~320-345 cyc (1 MB and 512 MB rings alike; the 512 MB ring touched only ~17 MB).
- Streaming, all 96 CUs, 2 MB buffer (L2 4 MB): per CU 21-43 B/clk at 4 waves/CU, 72-74 at 8, 78-115 at 16 waves (U = 4-16 loads
  in flight per wave). Needs >= 8 waves x 8 loads in flight per CU.
- **CU-count scaling (16 waves/CU, 1 block/CU):** 96 CUs 81-89 B/clk/CU (total 7.8-8.5 KB/clk), 48 CUs 141, 24 CUs 140-146,
  12 CUs 148-156. -> **per-CU port ~150 B/clk, L2 aggregate ~8.5 KB/clk per partition = ~88 B/clk/CU when all CUs pull.**
  With all CUs busy, **L2 bandwidth is the scarcest resource** (88 B/clk/CU < LDS 153-217).
- Caveats: shader-clock cycles; L2 may run in another clock domain, ratio may move under the power cap; real kernel L2 hit rate
  65-81% (misses go further). TDM's L2 -> LDS path not measured separately (assumed to share the same L2 bandwidth).

### Design bounds after M1 + M2 (EP8 gate-up, Mcyc; L2 88 B/clk/CU, LDS 153-217, WMMA 8 cyc)
WG = R token rows x C weight cols, 4 waves; "direct" = weights global -> VGPR (each wave owns C/4 cols), tokens TDM -> LDS.
L2 B/WMMA = 32768 (1/C + 1/R_eff); LDS B/WMMA (direct) = 32768 (1 + waves) / C.
| design | acc VGPR/wave | L2 | LDS | WMMA | bound |
|---|---|---|---|---|---|
| today (both via LDS, 128x256 w1x4) | 256 | 1.53 | 1.86-2.64 | 0.70 | 1.86-2.64 (measured 2.62) |
| D1 R128 C256 | 256 | 1.53 | 1.03-1.46 | 0.70 | 1.53 |
| **D1 R128 C512** | 512 | 1.27 | 0.52-0.73 | 0.70 | **1.27 (2.1x)** |
| D1 R256 C256 | 512 | 1.02 | 1.03-1.46 | 0.70 | 1.03-1.46 |
| **D1 R128 C512 + weight multicast x2 (cluster_m=2, same expert)** | 512 | 0.76 | 0.52-0.73 | 0.70 | **0.76 (3.4x)** |
- 512 acc VGPR + weight prefetch + token frags > 512 -> 1 wave/SIMD (user: acceptable if fast); needs >= 4 independent WMMAs in
  flight (have 32) and ~330-cyc load latency covered by software prefetch (~1-2 k-steps ahead).
- Weight multicast assumes one L2 read serves both CUs (unverified) and lockstep WGs on the same expert (rows/expert >= 256);
  VGPR multicast primitive found so far is `cluster_load_b32` (4 B/lane) -> 4 loads per 16 B fragment. -> M3 must measure:
  (a) does multicast cut L2 aggregate load, (b) cluster_load_b32 throughput, (c) drift / bimodality without a 16-WG barrier.
- Small-T / EP1 down still need D3; this table is the T=8192 gate-up only.

### M3: cluster multicast vs L2 bandwidth (2026-10-28, h13-17, GPU0; `/workspace/proxy_work/bench/cluster_mcast_bw.hip`)
96 CUs x one 16-wave WG, cluster {1,CS,1} via `hipLaunchKernelEx` + `hipLaunchAttributeClusterDimension`, 8 x 16 B loads in flight
per wave, 32 MB buffer, bytes delivered per WG / its cycles; 3 + 2 separate processes, all consistent (no bimodality here).
**`__builtin_amdgcn_cluster_load_b128` exists** (asm `cluster_load_b128 v[..], v, s[..]`; also b64) -> one 16 B weight fragment per
lane per instruction (correction: earlier notes said only b32).
| mode | delivered B/clk/CU (median) |
|---|---|
| private (each WG own data) | 80-84 (= M2 L2 aggregate) |
| shared: same addresses, plain global_load_b128 | ~82 (L2 does NOT merge identical requests) |
| **multicast cluster_load_b128, CS=2** | **141-148 (1.75x)** |
| **multicast, CS=4** | **145-170 (~per-CU port limit)** |
| multicast + cluster barrier every 8 iters | CS=2 128-129, CS=4 142-148 (-10..15%) |
- Correctness: multicast output == plain-load output (CS 2 and 4, index-pattern buffer).
- Drift (odd-rank WG `s_sleep(N)` every 8 iters; one iter ~440 cyc): N=2/8 (~128-512 cyc) -> multicast unaffected (144-153);
  N=32 (~2k cyc per 8 iters) -> **no barrier 41 B/clk (half of private!)**, with barrier 95-98 (~= the slow-member limit ~94).
  -> small drift is absorbed; large sustained drift without a barrier collapses below private. Keep a periodic cluster barrier
  as a safety net only where members can drift (cost 10-15% when in lockstep), or make members' work identical.
- Consequence: "D1 R128 C512 + weight multicast x2" is feasible on paper: delivered 320 B/WMMA -> 0.75 Mcyc at ~150 B/clk/CU,
  L2-side 192 B/WMMA -> 0.76 Mcyc, LDS 0.52-0.73, WMMA 0.70 -> **balanced at ~0.76 Mcyc (3.4x vs 2.62)**.
  Requires both WGs of a cluster on the same expert, same k loop (M-tiles of one expert: rows/expert >= 2 x 128).
- Open: bimodality of the existing 16-WG-barrier kernel still unexplained (not reproduced by this microbench).

### P1: HIP hot-loop prototype of D1 (2026-10-28, h13-17, GPU0; `/workspace/proxy_work/bench/p1_moe_hotloop.hip`,
copy in `proxy-moe/scripts/`). Full EP8 gate-up shape (64 experts x 256 rows, N 8192, K 16384), 2048 WGs of 128x512, 4 waves of
128x128 (32 WMMA per k128), `amdgpu_waves_per_eu(1,1)` (up to ~1024 VGPR/wave, verified: 680-833 VGPR without spill), weights
global -> VGPR (plain or `cluster_load_b128` multicast to the 2 M-tiles of an expert), tokens async global -> LDS (3 stages) -> VGPR.
**All variants bit-exact vs a plain-load reference kernel.** Clock during runs ~1.15 GHz (per-CU busy cycles / ms).
| version | cyc per k-step per wave | per-CU busy Mcyc | ms |
|---|---|---|---|
| v0: `__syncthreads` (fence -> `s_wait_loadcnt 0` kills prefetch), per-frag LDS waits | 2642 (mcast 5269) | 7.14 | 6.04 |
| v1: raw `__builtin_amdgcn_s_barrier`, all token frags read before WMMAs | 1869 (mcast 4728) | 5.25 | 4.49 |
| v2: weight prefetch depth 3 (PD1 2146, PD2 1942) | 1602 (mcast 4541) | 4.76 | 4.14 |
| v3: coalesced fragment layout (one b128 = 512 B contiguous per wave) | **1419 (mcast 1355)** | **4.04** | 3.28 |
| v4: + token VGPR double buffer (LDS reads of k+1 overlap WMMAs of k); VGPR 978, spills 132 B | 1630 (mcast 1806) | 4.55 | 3.97 |
Ablations (v3 / v4): no weight loads 942 / 783; no token loads 1243 / 1306; neither 780 / **499** (32 WMMA alone = 260).
VGPR scales do not slow the WMMA (M1 rerun: 8.1-8.3 cyc with per-lane / per-fragment VGPR scales).
**Status: prototype is SLOWER than today's kernel (4.04 vs ~2.62 Mcyc).** Lessons:
1. Missing bound in section "Design bounds": **weights must stream from HBM**: 4.29 GB at ~8 TB/s = 0.54 ms = **0.62 Mcyc at 1.15 GHz**,
   about equal to the WMMA bound (0.70). Real fp4 peak (~7 PFLOP/s at 1.15 GHz) puts the machine balance at ~900 flop per HBM byte,
   and T=8192 gate-up is 1024 -> compute and HBM are balanced; any exposed HBM latency shows up directly.
2. With 1 wave/SIMD and both barriers per k-step, the 4 waves run in lockstep: nothing hides the weight-load latency except the
   register prefetch depth, which is capped by VGPRs (512 acc + 4 x 68 weight buffers + 2 x 72 token buffers ~ 1000).
3. Never use `__syncthreads()` inside a pipelined loop on gfx1250 (implicit fence waits for all loads); use `__builtin_amdgcn_s_barrier()`.
4. Fragment layouts must be coalesced per instruction (lane-contiguous 16 B), not per lane.
Next options: (a) ATT of v3/v4 to see the real stall mix; (b) stage weights through LDS with async loads (deep, VGPR-free prefetch;
LDS ~832 B/WMMA vs today 1152); (c) 2 waves/SIMD with 128x64 per wave (acc 256) to regain wave-level latency hiding;
(d) reduce VGPR (PD2 + single token buffer) to remove the v4 spills.

### P1 ATT (2026-10-28, h13-17; `rocprofv3 --att --att-target-cu 1 --att-shader-engine-mask 0x1 --kernel-include-regex hot`,
one launch via `./p1_moe_hotloop <variant> att`; traces `/shared_nfs/kk/results/proxy_12l/p1_att_h13-17_1028/{v0=v3, v1=v4}`;
analysis `scripts/att/att_hot_loop_gfx1250.py` + new `scripts/att/att_wait_breakdown.py <dir> lo hi`). Share of stall+duration in the
hot loop (4 k-steps, 128 WMMA per iteration; ~1350 cyc/step):
| opcode | v3 (no token DB, VGPR 834, no spill) | v4 (token DB, VGPR 978, 132 B spill) |
|---|---|---|
| v_wmma | 25.9% | 20.6% |
| s_wait_dscnt (LDS token reads, not overlapped) | 24.8% | 3.4% |
| s_wait_asynccnt (token stage not landed: only 2 stages ahead) | 20.0% | 10.7% |
| s_wait_loadcnt(_dscnt) (weights; in v4 also spill reloads) | 16.8% | **49.1%** |
| s_barrier_wait | 6.9% | 10.7% |
- v4: 24 scratch ops + 5 `s_wait_loadcnt 0x0` in the function -> each spill reload drains the weight prefetch. The token double
  buffer works (dscnt 25% -> 3%) but its spill costs more than it saves.
- Fix plan (P1 v5): free ~150 VGPR so token DB fits without spills (weights PD 3 -> 2 with NB = 3 rotation via 3x unroll, or
  half-step token DB: 4 of 8 frags ahead); token LDS stages 3 -> 4-5 (asynccnt 20%); keep 1 barrier per step if NS >= 4;
  then re-ATT. Expected from the ablations: ~500 cyc/step core + exposed weight wait.
- Traps / design misses from P1 recorded for reuse in `amd-kernel-opt-techniques-gfx1250/GEMM_KERNEL_DESIGN_TRAPS.md`.

## Decisions (user, 2026-10-28)

1. Microbenchmarks first, starting with M1 (WMMA fp4 issue rate). 2. Do D3 (persistent, EP1 down) in parallel with D1.
3. 1 wave/SIMD is acceptable if the measured performance is good. 4. D4 (op interface change: fused reduce / quant) may be planned.

## Open questions for review (answered above)

1. Scope: start with gate-up only (largest, both EPs) or also EP1 down (largest gap to bound, 7x) in parallel via D3?
2. Is giving up 2 waves/SIMD acceptable if M5 shows D1 cannot fit in 256 VGPRs?
3. D4 changes the MoE op interface (fused reduce / quant) -- OK to plan for that, or keep the current op boundaries?
