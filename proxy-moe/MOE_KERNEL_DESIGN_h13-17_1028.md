Owner node: ctheliosmc-rck-g02-h13-17.rck.dcgpu

# MoE kernel redesign for gfx1250 (design draft, 2026-10-28) -- NO CODE UNTIL REVIEWED

Inputs: `MOE_PREFILL_h13-17_1028.md` (measurements, ATT, upstream check), `GFX1250_CLUSTER_NOTES_h13-17_1028.md`.
Tags: (M) measured on h13-17, (D) derived arithmetic, (A) assumption, (TBD) must be measured before deciding.

## CONTINUE HERE

**Status:** draft v1 reviewed (decisions below); M1 done (WMMA 8 cyc issue / 27 cyc latency, ~1024 VGPR/SIMD). M2 done (L2 aggregate ~88 B/clk/CU is the scarcest; per-CU port ~150). Best paper design: D1 R128xC512 (1.27 Mcyc) or + weight multicast (0.76). M3 done: cluster_load_b128 multicast correct, 1.75x (CS=2) / ~2x (CS=4) delivered per CU; collapses under large drift without barrier. P1 HIP prototype built (bit-exact) but 4.04 Mcyc vs today ~2.62: HBM-bound weights (0.62 Mcyc) were missing from the bounds, latency not hidden at 1 wave/SIMD. ATT done (v3: LDS 25% / async 20% / weights 17% / WMMA 26%; v4 spills drain prefetch). P1 v5 best 3.60 Mcyc (still > today 2.62; structure floor 1.18). Revised realistic gain 1.5-2.0x on gate-up (~8-10% prefill). v6 (2 waves/SIMD) 3.98 Mcyc, race fixed, L2 prefetch worse. Weight streaming from HBM looks like the real limiter. M2b done: HBM streams fine (async->LDS 8.8 TB/s at 4 waves/CU; regs <= 5.2); branch-free loop fixed the weight waits (v6 3.33 Mcyc, still > today 2.62); token/LDS path now dominates. v7 (2x2 waves) 3.14 Mcyc / 2.51 ms, ~ today in ms (not 1:1). Same-session A/B: v7 2.87-3.10 ms vs today 2.31-2.38 ms (~25-30% slower). TDM tokens added (HIP TDM verified); best v10 = 128x128/wave 1 wave/SIMD TDM PD3: loop 2.38 Mcyc, ms still 10-25% > today (non-loop overhead). Matched-epilogue A/B: v10 median 2.36 ms vs today 2.34 ms (parity). v10 ATT: WMMA 43%, barrier 21%, dscnt 20%, weights 7% -> LDS problem solved; With real routing + MX quant v11 = 3.23 ms vs today 2.30 ms (~40% slower; the -12% was uniform routing + scale-1 epilogue). D1 parked (realistic: ~40% slower, best case ~10% faster after fixes). D3 started: today's persistent mode has zero task overlap; EP1 down HBM floor 0.81 ms (4.3 GB per-route output), 0.34 with fused reduce (D4). D3 in today's kernel (E1 overlap, E2 tile map in LDS) does not beat non-persistent (best persistent 3315 vs 2448 us). D4 atomic fused reduce infeasible (pk-bf16 atomics 3.56 ms vs write+reduce 2.30 ms). No redesign beats today; B (gather-reduce) EP compact mode (auto by T): microbench EP8 0.58 -> 0.23 ms; **server EP8 TTFT -2.6% (c=2) / -3.0% (c=16)**, smoke pass, c=2 ITL flat, c=16 4-round A/B: throughput +2.1%, TTFT -3.0%. Local aiter edit, uncommitted; ON in the overnight 1028 sweep. Upstream (aiter/FlyDSL main 10-28) has no kernel that avoids the LDS bound.
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

### P1 v5 (2026-10-28, h13-17, GPU0): PD/NS template params, token double buffer without spills, optional single barrier
| variant | VGPR / scratch | cyc/k-step/wave | per-CU busy Mcyc |
|---|---|---|---|
| v3 PD3 NS3 (reference point) | 834 / 0 | 1428 | 4.06 |
| v5 PD2 token-DB NS3 | 896 / 0 | 1387 | 3.83 |
| v5 PD2 token-DB NS4 / NS5 | 899 / 888, 0 | 1467 / 1474 | 3.91 / 3.88 |
| **v5 PD2 token-DB NS5 one barrier** | 888 / 0 | **1372** | **3.60** |
| same + weight multicast x2 | 887 / 0 | 1381 | 3.64 |
| same, no weight / token loads | 822 / 0 | 431 | 1.18 |
All bit-exact. Still slower than today (~2.62). Multicast does not help -> not L2-bandwidth-bound; weight-load latency is exposed.
The structure itself (1 wave/SIMD, 4 lockstepped waves, barrier per step) has a floor of ~1.18 Mcyc even with no memory traffic.

### Revised gain estimate with the missed bounds (EP8 gate-up, T=8192, Mcyc at ~1.15 GHz; today measured ~2.62)
HBM floor = weights once + tokens + bf16 output = 4.7 GB -> 0.67 Mcyc (weights twice: 1.29).
| design | L2 | LDS | WMMA | HBM | bound | x vs today |
|---|---|---|---|---|---|---|
| D1 128x64/wave, WG 128x256, 2 waves/SIMD | 1.53 | 1.03-1.46 | 0.70 | 0.67 | 1.53 | 1.7x |
| + weight multicast x2 | 1.02 | 1.03-1.46 | 0.70 | 0.67 | 1.03-1.46 | 1.8-2.5x |
| D1 128x128/wave, WG 128x512, 1 wave/SIMD (P1) | 1.27 | 0.52-0.73 | 0.70 | 0.67 | 1.27 | 2.1x |
| + weight multicast x2 | 0.76 | 0.52-0.73 | 0.70 | 0.67 | 0.76 | 3.4x (theory) |
- With four resources within 10% of each other (last row) a kernel cannot saturate all of them; and the P1 structure has a
  measured compute-only floor of 1.18 Mcyc. Realistic: P1 path ~1.3-1.6 Mcyc if latency gets hidden (1.6-2.0x);
  the 2-waves/SIMD variant keeps wave-level latency hiding, bound 1.03-1.53 (realistic ~1.5-1.8x).
- End to end (EP8, rank 0, one 8k prefill ~225 ms): gate-up 45.5 ms -> at 1.6-2.0x saves ~17-23 ms = ~8-10% of prefill time;
  applying the same to down (26.7 ms) adds maybe ~3-5%; EP1 down (37.2 ms, 7x off bound) needs D3 instead.

### P1 v6: 2 waves/SIMD (user, 2026-10-28) + race fix + L2 prefetch (h13-17, GPU0)
Prototype now templated on CW (weight frags per wave): CW=2 = 128x64 per wave, acc 256, `amdgpu_waves_per_eu(2,2)`, WG 128x256,
2 WGs per CU, 4096 WGs; VGPR 456-468 with a 36 B spill (2 reloads in the loop); token double buffer at CW=2 hits 512 VGPR and spills 448 B.
**Race found:** v6 without token double buffer gave 24k-114k wrong outputs (varying per run). `__builtin_amdgcn_s_barrier()` is not a
memory operation for LLVM: the WMMAs consuming stage k (and with them the dscnt wait) were scheduled after the next step's barrier
and async writes, so other waves overwrote a stage whose LDS reads were still in flight. Fix: `s_wait_dscnt 0x0` (inline asm with a
"memory" clobber) before the stage-free barrier + compiler memory barriers around `s_barrier` -> all variants bit-exact.
Same race class as the upstream aiter `s_wait_dscnt(0)` fix.
| variant (all bit-exact) | VGPR / scratch | cyc/k-step/wave | per-CU busy Mcyc |
|---|---|---|---|
| v5 CW4 1 wave/SIMD PD2 TDB NS5 1bar | 888 / 0 | 1354 | **3.56** |
| v5 + L2 prefetch 8 / 16 steps ahead (`global_prefetch_b8` per 128 B) | 892 / 0 | 1438 / 1523 | 3.95 / 4.10 |
| v6 CW2 2 waves/SIMD PD3 NS3 | 462 / 36 | 1412 | 3.98 |
| v6 + L2 prefetch 8 / 16 | 468 / 36 | 2548 / 2521 | 6.91 / 6.78 |
| v6 + weight multicast + L2 prefetch 16 | 466 / 36 | 1988 | 5.43 |
| v6 no loads (core) | 384 / 0 | 670 | 1.82 (LDS-bound: 640 B/WMMA as predicted) |
- 2 waves/SIMD did NOT hide the weight-load cost either (~2.2 Mcyc exposed in both v5 and v6). L2 prefetch makes it worse.
- Weights-only ablation (v3) streamed weights at only ~1.5-2.9 TB/s. Hypothesis (unverified): HBM latency under load is far
  above the 330-cyc L2 latency, so ~270 KB in flight per CU would be needed for 8 TB/s -- more than registers can hold; this
  would explain why today's kernel stages weights through LDS (TDM, 4 buffers). M2 never measured true HBM streaming
  (its "512 MB" cases were partly cache-resident).
- Next decision point: M2b = HBM streaming rate of the exact weight access pattern vs bytes in flight per CU (4.29 GB, read once,
  registers vs async-to-LDS vs TDM) before any further prototype work.

### M2b: HBM streaming of the P1 weight layout (2026-10-28, h13-17, GPU0; `/workspace/proxy_work/bench/m2b_hbm_stream.hip`)
4.29 GB read once, persistent grid (G WGs x 4 waves per CU), 8 KB per wave per k-step (16 b128); TB/s = 4.29e9 B / kernel time.
Per-wave in-flight b128 loads are capped by the 6-bit counters (loadcnt / asynccnt <= 63 -> <= 3 steps of 16 loads per wave).
LDS per block max reported 320 KB.
| waves/CU | registers D=1 / 2 / 3 steps in flight | async -> LDS S=2 / 3 / 4 stages |
|---|---|---|
| 4 | 4.72 / 4.11 / 5.18 | 6.58 / **8.82** / 7.43 |
| 8 | 4.96 / 5.04 / 5.49 | 5.43 / 6.01 / 8.02 |
| 16 | 7.10 / 5.25 / 6.30 | 6.88 / 6.60 / 5.89 (LDS-capacity-limited occupancy) |
- **The "~270 KB in flight per CU" hypothesis is wrong:** async->LDS reaches ~8.8 TB/s with 4 waves and 96 KB in flight per CU
  (-> loaded HBM latency ~1000-1200 cyc). The register path is less efficient for the same bytes in flight (<= 5.2 TB/s at 4 waves,
  7.1 at 16; cause unknown).
- **HBM rate is not why P1 is slow:** weights-only streaming via registers at 4 waves/CU takes 0.83-1.05 ms (~1.0-1.2 Mcyc), the
  compute-only core 1.18 Mcyc; perfect overlap would give ~1.2 Mcyc, measured 3.56 -> **loads and compute do not overlap** in
  the lockstep 1-wave/SIMD structure. Runs are single-shot (noise ~10%: D=2 slower than D=1).
- Options: (1) weights via async->LDS into a per-wave private ring (no cross-wave barrier for weights; LDS traffic +512 B/WMMA ->
  ~830 B/WMMA total, LDS bound 1.34-1.9 Mcyc, i.e. at best ~1.4-1.9x vs today); (2) find why loads and WMMAs serialize in v5
  (ATT of v5: wait sites per step) -- the remaining upside of the register path is ~1.2 vs 3.56 Mcyc; (3) stop D1, move to D3.

### P1 v5/v6 ATT -> branch-free loop (2026-10-28, h13-17, GPU0; ATT `.../p1_att_h13-17_1028/{v5,v6bf}`)
- v5 ATT: s_wait (loadcnt) 43%, WMMA 25%, barrier 23%. The compiler waited `s_wait_loadcnt 0x3..0x0` right after issuing the
  next step's 20 weight loads (expected ~0x28): every step drained the whole weight prefetch.
- Not the scales (skipping weight-scale loads: v5 3.37 -> 3.13, v6 4.23 -> 3.60 only) and not VGPR-MSB (a <= 256-VGPR CW=1
  build with no `s_set_vgpr_msb` also waited 0x4 instead of 0xa).
- **Cause: per-step `if` guards (k < KS, k+PD < KS, real/dummy token issue, k+1 < KS) split the loop body into many basic
  blocks; the waitcnt pass merges pending-load state at each join conservatively.** Fix: clamp indices (`min(k+PD, KS-1)`),
  unroll dividing KS, no branches -> loop = 2 basic blocks, waits become 0x1e for PD=3 (= 3 steps x 10 loads).
- Note: ATT `code.json` disassembly shows VGPR numbers modulo 256 (ignores `s_set_vgpr_msb`); use the compiler `.s` for registers.
| branch-free variant (all bit-exact) | VGPR / scratch | cyc/k-step/wave | per-CU busy Mcyc |
|---|---|---|---|
| v5 CW4 PD1 TDB NS5 1bar | 736 / 0 | 1303 | 3.36 |
| v5 CW4 PD3 TDB NS5 1bar | 870 / 28 | 1403 | 4.08 (spill) |
| v6 CW2 PD1 NS3 | 371 / 0 | 1270 | 3.42 |
| **v6 CW2 PD3 NS3 / NS4** | 456 / 0 | 1230 / 1237 | **3.36 / 3.33** |
- v6 ATT after the fix: s_wait_dscnt 37% (token LDS reads), s_wait_asynccnt 19% (token stages), weight loadcnt 14% (was ~45%),
  WMMA 13%, barrier 8% -> now the token/LDS path dominates, as the bounds table predicted for CW2 (640 B/WMMA LDS).
- **Status: best prototype 3.33 Mcyc vs today ~2.62 (still 27% slower).** Every fix gains 10-20%; remaining gap to the ideal
  overlap (~1.2) is spread over LDS reads, token async latency, weights and barriers.

### P1 v7: 2x2 wave grid (user: continue D1, 2026-10-28, h13-17, GPU0)
New template param MW (waves along M): MW=2, CW=4 -> each wave 64 token rows x 128 weight cols (4 token frags x 4 weight frags,
16 WMMA/k-step), acc 256 VGPR, 2 waves/SIMD; WG 128x256; the 2 waves of a column load the same weights (expected to hit the CU
cache, not verified). Token LDS reads per WMMA halve (512 -> 256 B).
| variant (all bit-exact) | VGPR / scratch | cyc/k-step/wave | per-CU busy Mcyc | ms |
|---|---|---|---|---|
| v6 CW2 MW1 PD3 NS3 (reference point) | 456 / 0 | 1217 | 3.35 | 3.27 |
| v7 CW4 MW2 PD1 NS3 / NS4 | 442 / 0 | 1277 / 1351 | 3.49 / 3.65 | 3.08 / 3.29 |
| v7 PD3 NS3 | 512 / 1348 (spill) | 8372 | 22.75 | 18.4 |
| **v7 PD1 token-DB NS5 one barrier** | 459 / 0 | 1189 | **3.14** | **2.51** |
| v7 no loads (core) | 357 / 0 | 576 | 1.57 | 1.25 |
- In ms, v7 (2.51) is now close to today's op-test gate-up (2.32-2.47 ms), but not comparable 1:1 (op test includes routing /
  epilogue, different data; "today = 2.62 Mcyc" was us x an assumed 1.1 GHz). Next: same-session A/B of today's kernel vs v7.
- Core alone 1.57 Mcyc = 18 cyc/WMMA/SIMD vs 8 ideal and an LDS bound of 0.52-0.73 -> the compute core (2 barriers per step,
  LDS latency, waits) is the next target; ATT of the core variant.

- **Same-session A/B (GPU0, alternating, 2 rounds):** today's kernel op-test gate-up 2376 / 2310 us (pass) vs v7 best
  (PD1 token-DB NS5 one barrier) 3.104 / 2.869 ms -> **v7 is ~25-30% slower than today** (the single 2.51 ms run was lucky).
  The prototype also writes a 537 MB fp32 output vs today's ~34 MB fp4 epilogue, so this slightly favours today.
- **v7 core ATT** (`.../p1_att_h13-17_1028/v7core`, core variant has no token double buffer): s_wait_dscnt 48%, WMMA 32%,
  loadcnt_dscnt 8%, barrier 3%. Recomputed LDS traffic for v7: reads 4 KB tokens + 0.5 KB scales per wave per step (288 B/WMMA)
  + async writes 9 KB per WG per step (144 B/WMMA) = **432 B/WMMA (the bounds table said 320: scales and write bytes were
  undercounted) -> LDS bound 0.70-0.99 Mcyc**; core measured 1.57 (LDS latency not hidden in this variant).

### P1 v8-v10: TDM for tokens (user: option b, 2026-10-28, h13-17, GPU0)
**HIP TDM verified** (`scripts/tdmprobe.hip`): `__builtin_amdgcn_tensor_load_to_lds(u32x4 g0, i32x8 g1, i32x4 0, i32x4 0, i32x8 0, cpol)`
(2D form, groups 2-4 zero) + `__builtin_amdgcn_s_wait_tensorcnt(n)`. g0 = {count=1, lds_addr, gaddr_lo, gaddr_hi[24:0] | 2<<30};
g1[0] = wg_mask | data_size<<16 (2 = 4 B); tensor_dim0 = row width (elements), tensor_dim1 = rows, tile_dim0/1, and the
**row pitch goes in tensor_dim0_stride** (pitch in dim1_stride gave wrong rows). Layout: g1[1]=dim0_lo<<16, g1[2]=dim0_hi|dim1_lo<<16,
g1[3]=dim1_hi|tile0<<16, g1[4]=tile1, g1[5]=dim0_stride_lo, g1[6]=dim0_stride_hi|dim1_stride_lo<<16, g1[7]=dim1_stride_hi.
Prototype: template TOK=1 -> wave 0 issues 2 TDM per stage (8 x 1 KB tokens at pitch KS KB, 8 x 128 B scales).
| variant (all bit-exact) | VGPR / scratch | cyc/k-step/wave | per-CU busy Mcyc |
|---|---|---|---|
| v7 2x2, async tokens, PD1 TDB NS5 | 459 / 0 | 1204 | 3.12 |
| v8 2x2, **TDM** tokens, PD1 TDB NS5 / NS6 (no TDB) | 451 / 434 | 1065 / 1040 | 2.96 / 2.89 |
| v8 TDM, no weight loads | 383 / 0 | 447 | 1.23 |
| v9 2x2 1 wave/SIMD PD3 / PD7 / PD3 NS8 | 596 / 852+76 / 598+12 | 620 / 2039 / 1157 | 3.33 / 10.4 / 5.8 |
| v8 + weight multicast | 440 / 0 | 1875 | 5.08 |
| v10 128x128/wave (MW1 CW4) 1 wave/SIMD TDM PD1 | 776 / 0 | 1087 | 2.99 |
| **v10 128x128/wave 1 wave/SIMD TDM PD3 TDB NS5 1bar** | 874 / 0 | **840** | **2.38** |
| v10 + weight multicast PD1 / PD3 | 740 / 878 | 1189 / 878 | 3.16 / 2.55 |
- **2x2 (v7/v8) was L2-bound, not latency-bound:** the two M-waves of a column each load the same weights (no CU-cache sharing
  visible) and the two M-tile WGs again -> weights read 4x from L2: 512 + 144 (tokens) = 656 B/WMMA -> 2.6 Mcyc at 88 B/clk/CU
  = measured 2.9-3.0. The design assumption "the second wave hits the CU cache" was wrong.
- 128x128/wave loads each weight once per WG (L2 ~328 B/WMMA -> 1.3 Mcyc bound) and, with TDM tokens + PD3, is the best so far.
- Weight multicast slowed every real-kernel variant (2.38 -> 2.55; 2.93 -> 5.08) although M3 showed 1.75x in a pure stream.
- **Same-session A/B (GPU0, 3 alternating rounds):** today 2391 / 2240 / 2457 us vs v10 PD3 2.86 / 2.97 / 2.52 ms (loop
  2.30-2.42 Mcyc) -> v10 still ~10-25% slower in ms. Loop cycles vs ms imply ~0.8-0.9 GHz -> large non-loop overhead (prologue,
  537 MB fp32 output store, 22 WG rounds at 1 WG/CU) not yet measured; today's kernel writes ~34 MB (fused act+quant).

### P1 v10 with a matched epilogue (user, 2026-10-28, h13-17, GPU0)
Template OUTM: 0 fp32 (537 MB, bit-exact check), 1 bf16 (268 MB), 2 today-like: silu(gate)*up (weight frags j < CW/2 gate,
j >= CW/2 up) -> software e2m1 fp4 with scale 1 (33.5 MB, same bytes as today). gfx1250 has no
`__builtin_amdgcn_cvt_scalef32_pk_fp4_f32` (needs fp4-cvt-scale-insts, gfx950 only). Per-wave cycle stamps now split prologue /
loop / epilogue (stores drained with `s_wait_storecnt 0`). First fp4 epilogue used `g / (1 + __expf(-g))`: epilogue 2.64 Mcyc;
with `__builtin_amdgcn_exp2f` + `__builtin_amdgcn_rcpf` 0.19 Mcyc.
| output (v10: 128x128/wave, 1 wave/SIMD, TDM tokens, PD3) | loop | prologue | epilogue (per-CU Mcyc) |
|---|---|---|---|
| fp32 | 2.43 | 0.04 | 0.24 |
| bf16 | 2.46 | 0.04 | 0.20 |
| silu*up fp4 (fast exp/rcp) | 2.34-2.61 | 0.04 | 0.19 |
- Non-loop time is small (prologue 0.04, epilogue ~0.2); the earlier "0.8-0.9 GHz implied" estimate was wrong.
- **Same-session A/B, 3 alternating rounds (GPU0):** today 2377 / 2336 / 2212 us (median 2.34 ms); v10 fp4 2.487 / 2.140 / 2.360 ms
  (median 2.36); v10 bf16 2.71 / 2.69 / 2.75 ms. -> **v10 with a today-sized epilogue is at parity with today's kernel (within the
  ~+-10% run-to-run noise)**, not yet faster. Caveats: no routing / real per-32 amax quant in the prototype; today's op test
  includes its real epilogue. Loop is ~90% of the time -> remaining upside is in the loop (2.3-2.6 Mcyc vs bounds ~1.0-1.3).

### P1 v10 ATT (2026-10-28, h13-17; `.../p1_att_h13-17_1028/v10`, variant fp4 epilogue, ~880 cyc/k-step)
| opcode | v10 | v8 (2x2, TDM) | today's kernel (EP8 gate-up) |
|---|---|---|---|
| v_wmma | **43.0%** | ~13% | 33% |
| s_barrier_wait | **21.4%** | ~8% | 12% |
| s_wait_dscnt | 19.9% | 37% | 41% |
| s_wait_loadcnt (weights) | 7.4% | 14% | - |
- **The LDS-bandwidth problem of today's kernel is solved in v10** (LDS ~360 B/WMMA vs 1152; dscnt share 41% -> 20%; WMMA share
  33% -> 43%), but time is lost to the 4 lockstepped waves at 1 wave/SIMD (barrier 21%) and to the per-step
  `s_wait_dscnt 0x0` stage-free waits (~2.6% each, 4 per 4-step iteration). Weights (PD3) are mostly hidden (7%).
- Next fixes: (1) split barrier: `s_barrier_signal` right after the tensorcnt wait, first half of the step's WMMAs, then
  `s_barrier_wait` before reading the next stage; (2) more token stages so the stage-free dscnt wait can lag several steps.
  If most of barrier + dscnt time (41%) is recovered: loop ~2.4 -> ~1.4 Mcyc (L2 bound 1.3) = ~1.6x vs today.

### P1 v11: split barrier (2026-10-28, h13-17, GPU0)
Per step (template SPLIT=1, TDB): TDM issue (wave 0) -> weight loads k+PD -> `s_wait_dscnt 0` (this step's tokens are needed anyway;
retires reads of the stage freed next step) -> tensorcnt wait for stage k+1 -> `__builtin_amdgcn_s_barrier_signal(-1)` -> first half
of the step's WMMAs -> `__builtin_amdgcn_s_barrier_wait(-1)` -> read stage k+1 tokens -> second half. One barrier per step, stage
reuse stays race-free (every wave retires its reads before signalling). All fp32 variants bit-exact.
| variant | VGPR | cyc/k-step | loop Mcyc |
|---|---|---|---|
| v10 (baseline, fp4 out) | 870 | 870 | 2.53 |
| v11 split NS3 fp32 / fp4 | 904 / 902 | 752 / 727 | 2.21 / 2.12 |
| v11 split NS5 fp32 / fp4 | 872 / 870 | 817 / 725 | 2.31 / 2.06 |
- **Timing method fixed:** the prototype timed 3 launches after 1 warmup (~10 ms): back-to-back runs implied 0.88-1.34 GHz and
  ms did not follow cycles. Now 10 warmup + 50 timed launches (op test uses 101 iters) -> stable to ~+-3%.
- **Same-session A/B, steady state, 3 alternating rounds (GPU0, fp4 epilogue):**
  | round | today | v10 | **v11 split NS5** | v11 split NS3 |
  |---|---|---|---|---|
  | 1 | 2226 us | 2.255 ms | 2.045 ms | 2.139 ms |
  | 2 | 2319 | 2.243 | 2.045 | 2.120 |
  | 3 | 2390 | 2.234 | 2.142 | 2.125 |
  -> **v11 NS5 median 2.05 ms vs today 2.32 ms = 12% faster** (v10 -3%). Caveats: prototype has no routing and no per-32 amax
  quant (scale 1); today's op test includes its real epilogue. Loop 2.06-2.25 Mcyc vs bound ~1.3 -> still ~1.6x headroom.

### P1 v11 with routing + real MX quant (user: "補上2", 2026-10-28, h13-17, GPU0)
Prototype now uses a host tile table {expert, m-tile} (token capacity 384 rows / expert) and two routings: uniform 256 rows/expert
(128 M-tiles) and op-test-like random top-2 of 64 (seed 1234: rows/expert 219..314, 31 experts need a 3rd M-tile, 159 M-tiles,
80.5% of tile rows valid). New OUTM=3: silu(gate)*up, per-32-value amax (16 values in-lane + `__shfl_xor(.,16)`), e8m0 scale,
e2m1 fp4 + scale bytes. The op test's default routing is random (`_make_topk`, `fused_topk` on random scores);
`AITER_MOE_EXPERT_BALANCE=1` gives balanced routing.
| case (2 alternating rounds, steady-state timing) | today | v11 fp4 scale=1 | v11 fp4 + MX quant (loop / epi Mcyc) |
|---|---|---|---|
| balanced / uniform routing | 2273 / 2400 us | 1.977 / 1.977 ms | 2.697 / 2.713 ms (2.24-2.28 / 0.63-0.68) |
| random routing | 2321 / 2273 us | 2.412 / 2.415 ms | 3.221 / 3.243 ms (2.84-2.94 / 0.82-0.84) |
- **The earlier "12% faster" compared uniform-routing v11 with random-routing today and a scale-1 epilogue -> not a fair result.**
  Under the real workload (random routing + MX quant) v11 = 3.23 ms vs today 2.30 ms (**~40% slower**).
- Gap 1: my MX-quant epilogue costs 0.6-0.8 Mcyc (scale-1 version 0.2): `__shfl_xor` (ds_bpermute) + 16 compares per value;
  today's epilogue also skips whole 16-row blocks past the expert's valid rows (`if wmb + wm*16 < mn_oob`, kernel ~1791).
- Gap 2: v11 slows 22% with random routing (more, partly empty M-tiles), today's kernel ~0%. Today's TDM A-load is bounded by
  `mn_oob` (valid rows only, zero-fill); a main-loop WMMA skip for empty row blocks was not found yet -> cause still open
  (check: today's tile count / layout alignment for this routing; whether balanced mode really gives 256 rows/expert).
- Next options: (a) explain gap 2 (read today's tile_map/layout for the op-test routing); (b) skip empty 16-row blocks in our
  loop (WMMA rows) and epilogue + bound the TDM token rows; (c) cheaper MX epilogue (DPP/permlane instead of ds_bpermute).

## D3 (persistent / overlapped tiles) -- started 2026-10-28 (user: start from today's kernel, diagnose first)

### Step 1: today's persistent path (`mxfp4_preshuffle_gfx1250_tdm.py`, `persistent_workers`, ~2309-2400)
`for task_id in range(block_idx.x, tasks, persistent_workers): run_task(...)` calls the full `gemm_task` per task: prologue TDM
loads -> k-loop -> epilogue store -> `tensor_wait(0)` (output store drained) -> `workgroup_barrier()`. **No overlap between tasks**;
`next_stage_buf` only overlaps k-tiles inside one task. Persistent mode thus removes launch cost but also the overlap that several
co-resident WGs per CU give in the normal grid -> slower.

### Step 2: EP1 down measurements (h13-17, GPU0; CSV row = EP1 rule tiles + persistent_workers2; `/tmp/up/d3/ep1_pw*.csv`)
- op test gemm2 (down): rule 2482 us; CSV pw2=0 2448; pw2=192 4262; pw2=384 3000 (all pass).
- ATT (`/shared_nfs/kk/results/proxy_12l/d3_att_h13-17_1028/pw{0,384}`; needs `PYTHONPATH=proxy-moe/scripts/rocprof_env_fix` --
  without it rocprofv3 aborts with `Option 'spirv-expand-step' registered more than once`, see MOE_PREFILL_f19-14_1028.md):
  | opcode | pw2=0 (~5700 cyc per tile, no loop) | pw2=384 |
  |---|---|---|
  | s_wait_tensorcnt (prologue loads + output store drain) | 34.0% | 21.8% |
  | s_barrier_wait | 29.0% | 24.4% |
  | s_wait_dscnt | 15.9% | 14.1% |
  | s_wait_loadcnt | - | 27.7% (per-task serial global loads) |
  | v_wmma | 4.9% | 4.4% |
- **Bound correction:** EP1 down writes a per-route bf16 output (16 routes x 8192 x 16384 x 2 B = **4.29 GB per layer**) on top of
  2.15 GB of weights; op test reports 2664 GB/s x 2.48 ms = 6.6 GB. HBM floor = **0.81 ms/layer** (the "0.45 ms compute bound"
  above ignored the output). Compute floor 0.31 ms. With the top-k weighted reduce fused into the epilogue (one T x H output,
  D4) the floor is 0.31-0.34 ms and the separate gather-reduce kernel (12.8 ms per 8k prefill) disappears.
- -> D3 alone (overlap prologue / store across tiles): at most 2.48 -> ~0.8-1.0 ms per layer. D3 + D4: toward ~0.35 ms + no gather.

### Step 3: D3 experiments in today's kernel (user: "先做 D3", 2026-10-28, h13-17, GPU0, EP1 down op test)
Local, uncommitted edits in `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py` (original saved as
`/workspace/proxy_work/mxfp4_preshuffle_gfx1250_tdm.py.before_d3`; full local diff incl. earlier edits
`/workspace/proxy_work/d3_kernel_1028.patch`), both default OFF and only active with persistent workers:
- E1 `AITER_TDM_PW_OVERLAP=1`: C staging gets its own LDS region (`c_ptr`, `stCo_idx`; ARENA_B = inputs only) and the per-task
  `tensor_wait(0)` + `workgroup_barrier()` after the output TDM store are dropped (one `tensor_wait(0)` after the worker loop).
  Safe because the next task's `pipeline_fence(0)` before its epilogue retires the previous store before C is rewritten.
- E2 `AITER_TDM_TILEMAP_LDS=1`: first task of a worker copies the tile map (n_experts i32) into LDS; has_work / bisect /
  prev_end / mn_oob read LDS (`tm_get`).
- **Trap: FlyDSL's JIT cache key ignores module-level env globals** -> the first A/B silently reused the OFF binary (identical
  times). Use `FLYDSL_RUNTIME_ENABLE_CACHE=0` for any env-switched kernel variant (as the AITER_TDM_PROBE runs did).
| gemm2 us (all pass) | E off | E1 | E2 | E1+E2 |
|---|---|---|---|---|
| persistent 192 workers | 4259 | 3482 | 4200 | 3315 |
| persistent 384 workers | 2997 | 3395 (C region -> 104 KB LDS/WG, only 3 WGs/CU fit) | 2998 | 3597 |
| non-persistent (baseline) | **2448** | | | |
- E1 helps at 2 WGs/CU (-18%) but costs occupancy at 4; E2 is small (the dependent bisect is not the main cost; the 28%
  `s_wait_loadcnt` in persistent mode is still unexplained). **No persistent variant beats the non-persistent 2448 us.**
- Occupancy / tile sweep, non-persistent (m2/n2/k2/nb2/nw2): rule 64/256/128/3/4 2483; nb2=2 2562; nb2=4 2825; n2=128: 2692-2779;
  m2=32 3452 -> nothing beats the rule (matches the earlier knob sweep).
- Earlier probes (MOE_PREFILL doc, EP1 down): full 2362 / no-WMMA 2235 / TDM-loads-only 1406 us -> **epilogue (per-route output
  store) ~830 us = ~35%**, loads ~1.4 ms (2.15 GB weights re-read by ~4 M-tiles per expert).
- **Conclusion so far:** overlapping tiles inside today's kernel (D3) does not pay; real cross-task overlap would need splitting
  `gemm_task` into setup/prologue and mainloop/epilogue (large refactor) and is capped by the HBM floor (0.81 ms) anyway.
  The output store is a measured ~35% of EP1 down and the gather-reduce is another ~1.07 ms/layer -> **D4 (fused weighted
  reduce, no per-route output) attacks the largest measured costs.**

## D4 (fused top-k weighted reduce) -- feasibility (user: "做D4", 2026-10-28, h13-17, GPU0)
User note: none of the redesigns so far beat today's kernel; the earlier wins came from rule/config changes in today's kernel.
-> D4 approach: reuse existing mechanisms, measure the upside before any kernel change.
- Existing pieces: the EP-scatter epilogue already multiplies by the route weight and writes rows to destination positions
  (P2P combine); the non-EP path writes `grouped_out` (per-route bf16) then runs `flydsl_moe_gather_reduce`
  (out[t] = sum_k w[t,k] * grouped[row(t,k)]). No atomic-accumulate epilogue exists.
- Microbenchmark `scripts/m4_atomic_reduce.hip` (T=8192, top-16 of 512, H=16384, rows in expert-grouped order, asm checked):
  | path | ms per layer |
  |---|---|
  | today: per-route bf16 write 1.319 + gather-reduce 0.980 | **2.300** |
  | fused, `global_atomic_add_f32` into a T x H fp32 buffer | 11.712 (+0.187 zero-fill, + convert) |
  | fused, `global_atomic_pk_add_bf16` into T x H bf16 | 3.563 (+0.077 zero-fill) |
- **Atomic accumulation is slower than write + gather-reduce on gfx1250 at this size (~3e11 pk-bf16 atomic ops/s)** -> the fused
  reduce as designed does not pay. Token-major ownership (one WG reduces all 16 routes) would need 16 experts' weights per WG;
  rows inside a tile are distinct tokens, so no in-tile pre-reduction. D4 as designed is dropped.
- The measured gather-reduce (0.98 ms) matches the serving profile (12.8 ms / 12 layers = 1.07 ms).

## Status of the redesign track (2026-10-28)
D1 (weights in VGPRs): realistic workload ~40% slower than today. D3 (tile overlap in today's kernel): no gain vs non-persistent.
D4 (atomic fused reduce): atomics slower than today's write + reduce. **No redesign has beaten today's kernel.** Lessons and measured
hardware numbers are in `amd-kernel-opt-techniques-gfx1250/GEMM_KERNEL_DESIGN_TRAPS.md`.

## Prototype-validated techniques applied to today's kernel (user: "優先嘗試已經在原型裡驗證過的做法", 2026-10-28, h13-17, GPU0, EP8)
ISA via `FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=<dir>` (final ISA in `<dir>/<kernel>/22_final_isa.s`).
1. **Bigger per-wave tile.** No spills in any variant: rule 128x256 w1x4 = 382 VGPR, 110 KB LDS (2 waves/SIMD); w1x2 (128x128/wave)
   665 VGPR; 128x512 w1x4 (128x128/wave, closest to P1 v10) 664 VGPR, 134 KB LDS, 1 wave/SIMD. Single runs suggested -5%, but a
   3-round alternating A/B with stage 2 pinned to the rule (`AITER_TDM_TILE_M2/N2/K2/NUM_BUFFERS2`; stage 2 otherwise inherits
   stage-1 env tiles!) gave base 2208 / 2215 / 2350 vs 128x512 2214 / 2217 / 2214 us -> **no measurable gain** (base noise ~+-4%).
   With A and B both in LDS the bigger tile cuts LDS bytes/WMMA ~28% but halves occupancy.
2. **Split fence** (`AITER_TDM_SPLIT_FENCE`, local edit in `k_step`, uses the existing `pipeline_fence_signal/_wait` that the dense
   `gemm_a4w4_256x256` / `gemm_a8w8_256x256` kernels already use). In the default path (tile_k 128, no next-stage prefetch) the
   fence precedes the tile's first LDS read, so nothing can overlap (ISA showed signal immediately followed by wait). It applies only
   with `AITER_TDM_NEXT_STAGE_PREFETCH=1`, which itself costs +3..12% (base 2378 -> 2451, 128x512 2245 -> 2515 us); split recovers
   part (2432 / 2434) -> **net loss**.
3. **Branch-free loop / exact waits.** Today's EP8 gate-up hot loop is already 1 basic block, 16 WMMA, `s_wait_tensorcnt 0x4`,
   `s_wait_dscnt 0x13` + `0x0`, one barrier -> nothing to fix (the prototype's problems were self-inflicted).
-> **None of the three P1 techniques improves today's kernel on its own.** The P1 loop-level win came from the combination
   "weights not through LDS + 128x128/wave + TDM tokens + split barrier"; today's kernel cannot take the pieces individually.

## B: gather-reduce kernel (user: "先做B", 2026-10-28, h13-17, GPU0)
Kernel `aiter/ops/flydsl/kernels/moe_gather_reduce.py` (315 lines): grid (T, n_iters=8 for H=16384), 256 threads, 16 B per thread per
route row; the first topk lanes put rows/weights in LDS + barrier, then `for k in range_constexpr(topk)` buffer loads (zero-length
descriptor for EP `DROPPED_ROUTE_ROW`). Bench `scripts/gather_reduce_bench.py` (calls `flydsl_moe_gather_reduce`):
- EP1 (T=8192, top-16 of 512 all live, H=16384): **0.93 ms = 4.9 TB/s** (rocprof serving 12.8 ms / 12 = 1.07 ms). vec 2 / 4 same;
  **vec_dwords=8 is broken** (`unit_bytes 32 is not a buffer access width`).
- **EP8 real (16 slots per token, ~2 live on a 64-expert rank, others DROPPED, as the EP route kernel produces): 0.58 ms = 1.4 TB/s
  of live bytes**, vs 0.226 ms for a compact top-2 map -> ~0.35 ms per layer spent on dropped slots.
- Change (local, uncommitted; originals `/workspace/proxy_work/moe_gather_reduce.py.orig_1028`,
  `/workspace/proxy_work/grouped_moe_gfx1250.py.before_gr_1028`): builder arg `ep_compact` (kernel suffix `_epc`) = whole-row
  block (grid y = 1, loop over the 8 chunks) + each route in a block-uniform `if row >= 0`. The wrapper enables it only for EP
  route maps (`num_valid_tokens is not None`); `AITER_GR_EP_COMPACT=0` disables.
  | gather-reduce ms (all max|err| at bf16 rounding) | default | rowloop only | skip only | both (= ep_compact) |
  |---|---|---|---|---|
  | EP1 all live | **0.928** | 1.018 | 1.267 | 0.970 |
  | EP8 real | 0.578 | 0.530 | 0.383 | **0.227** |
  Wrapper A/B (2 alternating rounds): EP8 0.585 / 0.580 -> 0.245 / 0.229 ms (**~2.4x**), EP1 0.931 / 0.929 vs 0.940 / 0.933 (unchanged,
  not enabled there). Expected serving effect EP8: gather ~10.2 -> ~4 ms per 8k prefill (~6 ms, ~2.7% of TTFT).
- **Server A/B #1** (`scripts/gr_ab_1028.sh`, EP8 TP8 12L all-on, 8k/1k, arms off/on/off/on = whole-row + skip for every T;
  `/shared_nfs/kk/results/proxy_12l/gr_ep_compact_h13-17_1028/`, all smoke passed), ms:
  | | off1 | on1 | off2 | on2 | change |
  |---|---|---|---|---|---|
  | c=2 mean / median TTFT | 290.3 / 234.4 | 281.1 / 224.2 | 289.3 / 231.0 | 281.0 / 224.7 | **-3.0% / -3.6%** |
  | c=16 mean / median TTFT | 1481.1 / 1396.1 | 1437.6 / 1354.2 | 1484.7 / 1399.0 | 1433.7 / 1347.8 | **-3.1% / -3.2%** |
  | c=2 median ITL | 6.33 | 6.77 | 6.08 | 6.71 | +7% (worse) |
  | c=16 median ITL | 15.81 | 17.82 | 16.60 | 16.19 | mixed |
- Small-T sweep (EP8 real map, ms; mode 0 = original, 1 = skip dropped only, 2 = skip + whole-row block):
  | T | 2 | 16 | 64 | 256 | 1024 | 4096 |
  |---|---|---|---|---|---|---|
  | mode 0 | 0.014 | 0.013 | 0.013 | 0.013 | 0.065 | 0.294 |
  | mode 1 | 0.014 | 0.013 | 0.013 | 0.013 | 0.048 | 0.202 |
  | mode 2 | 0.015 | 0.020 | 0.013 | 0.015 | **0.030** | **0.117** |
  -> default `AITER_GR_EP_COMPACT=3` = auto: mode 2 for T >= `AITER_GR_EP_ROWLOOP_MIN_TOKENS` (1024), else mode 1. The decode
  gather difference at T=2 is ~1 us per call (12 us per step), so it cannot explain the +0.4-0.6 ms c=2 ITL of A/B #1 -> re-check.
- **Server A/B #2** (arms off / auto / off / auto, `AITER_GR_EP_COMPACT=0/3`;
  `/shared_nfs/kk/results/proxy_12l/gr_ep_compact_auto_h13-17_1028/`, all smoke passed), ms (throughput tok/s):
  | | off1 | auto1 | off2 | auto2 | change |
  |---|---|---|---|---|---|
  | c=2 mean / median TTFT | 287.1 / 231.4 | 281.7 / 228.2 | 289.4 / 231.0 | 280.4 / 222.1 | **-2.6% / -2.6%** |
  | c=2 median ITL | 6.29 | 6.24 | 6.32 | 6.33 | flat (A/B #1 regression gone) |
  | c=2 output throughput | 300.4 | 303.9 | 299.0 | 298.9 | flat |
  | c=16 mean / median TTFT | 1480.0 / 1393.2 | 1435.0 / 1346.8 | 1479.2 / 1394.7 | 1435.9 / 1353.4 | **-3.0% / -3.1%** |
  | c=16 median ITL | 16.09 | 16.27 | 15.90 | 16.80 | +3.6% avg (noisy; A/B #1 also mixed) |
  | c=16 output throughput | 845.5 | 843.0 | 851.4 | 824.7 | -1.5% |
- **Result: EP8 8k/1k TTFT -2.6% (c=2) / -3.0% (c=16), reproducible across 4 off/on pairs (2 A/B runs); c=2 decode unchanged.**
  c=16 ITL / throughput within run-to-run noise or slightly worse (decode at T=16 runs mode 1 = same kernel time as mode 0 in
  the microbench) -> confirmed below.
- **Server A/B #3, c=16 only, 4 alternating rounds per arm** (`/shared_nfs/kk/results/proxy_12l/gr_ep_compact_c16x4_h13-17_1028/`,
  8/8 smoke passed): output throughput off 809.3 / 821.3 / 830.6 / 820.2 (avg 820.3) vs auto 848.7 / 829.7 / 829.4 / 843.6
  (avg 837.8) tok/s = **+2.1%**; mean TTFT 1657.3 -> 1607.1 ms (**-3.0%**); median ITL 16.39 -> 15.96 ms (-2.6%). The c=16
  "-1.7%" of A/B #2 was noise. (TTFT is higher than in A/B #1/#2 because no c=2 run warmed the server first.)
  **-> gather-reduce EP compact (auto) is kept ON in the overnight 1028 sweep.** Patch (incl. earlier local edits of grouped_moe_gfx1250.py):
  `/workspace/proxy_work/gr_ep_compact_1028.patch`.

## Decisions (user, 2026-10-28)

1. Microbenchmarks first, starting with M1 (WMMA fp4 issue rate). 2. Do D3 (persistent, EP1 down) in parallel with D1.
3. 1 wave/SIMD is acceptable if the measured performance is good. 4. D4 (op interface change: fused reduce / quant) may be planned.

## Open questions for review (answered above)

1. Scope: start with gate-up only (largest, both EPs) or also EP1 down (largest gap to bound, 7x) in parallel via D3?
2. Is giving up 2 waves/SIMD acceptable if M5 shows D1 cannot fit in 256 VGPRs?
3. D4 changes the MoE op interface (fused reduce / quant) -- OK to plan for that, or keep the current op boundaries?
