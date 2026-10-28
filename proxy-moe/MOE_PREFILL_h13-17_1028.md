Owner node: ctheliosmc-rck-g02-h13-17.rck.dcgpu

# Prefill MoE tile optimization, continued on h13-17 (2026-10-28)

Takes over from `MOE_PREFILL_f19-14_1028.md` (owner f19-14, closed: GPU 2/3 SMU not responding). Carried-over result:
grouped a8w4 MoE, T=8192, one EP2 rank of the TP2 4L model (E=256, topk 8, H 16384, I 4096): stage1 `AITER_TDM_TILE_M/N/K/NUM_BUFFERS=128/256/128/4`
+ stage2 `..._M2/N2/K2/NUM_BUFFERS2=128/256/256/2` = 14.40 ms vs defaults 16.46 ms (-12.6%), f19-14 only, idle clocks.

## CONTINUE HERE

**SUPERSEDED 2026-10-29: the next round continues in `OPT_h13-17_1029.md` (read that CONTINUE HERE instead).**

**Status (2026-10-28, h13-17):** Step 0 done (see "Step 0 result" below): no upstream kernel avoids weights-in-LDS; upstream LDS race fix ported. Upstream A/B done (no win over rule). Cluster notes written. **Track B started: design draft `proxy-moe/MOE_KERNEL_DESIGN_h13-17_1028.md` (awaiting review).** TP8 runs all f19-14 optimizations (8k/1k TTFT -34..39%),
DDA on (CAR <= 1 MB), prefill MoE tile rule + alignment-gap tile skip (EP8 TTFT -12%, EP1 -5%, 0 faults, EP8 bit-identical).
Gate-up (both EPs) diagnosed as **LDS-throughput-bound** (needs ~250 B/clk/CU vs 153-217 measured). Config space exhausted.
**Next, in order (user, 2026-10-28):**
**Step 0 -- check upstream first** (standing rule in MI455_RUN_h13-17.md): look for a newer / different gfx1250 grouped MoE GEMM in
ROCm/FlyDSL (installed 0.3.4.1 is old: aiter's own `gemm_a16w16_gfx1250` already needs `flydsl.runtime.device.check_smem_capacity`) and
ROCm/aiter main (FlyDSL or gluon MoE paths). Record per candidate: data flow (do weights / tokens go through LDS?), WMMA mapping, supported
formats (fp4 act x fp4 weight, preshuffle), EP support. If one avoids the LDS bound, benchmark it on the op-test shapes before writing code.
**Step 1 -- two tracks, decide after Step 0:**
- Track A (incremental, current kernel): "weights direct global -> VGPR", details below.
- Track B (from scratch, user request): design a new prefill/decode MoE kernel NOT bound to the current FlyDSL grouped_moe structure.
  Write the design doc first (`proxy-moe/MOE_KERNEL_DESIGN_<host>_<date>.md`), no code until reviewed:
  1. Workload: shapes per EP (EP1 E512/inter512, EP8 E64/inter4096, model 16384), T from 1 (decode) to 8192 (prefill), rows/expert
     = T x 16 / 512, routing skew; what the server actually runs (fp4 act x fp4 weight, mxfp4 scales, swiglu, weighted gather-reduce).
  2. Hardware limits in CYCLES (clocks are power-limited, ~1.0-1.4 GHz in op tests and serving): HBM ~8 TB/s per partition, LDS read
     ~153-217 B/clk/CU, WMMA fp4 issue rate (measure), VGPR 512/wave and waves/SIMD trade-off, LDS capacity, TDM / cluster multicast,
     energy per byte (PPT-limited: fewer bytes -> higher clock).
  3. Roofline per shape: minimum HBM bytes, minimum on-chip (L2->LDS->VGPR) bytes per WMMA for a given accumulator budget, WMMA time.
  4. Candidate architectures to compare on paper: weights never through LDS (each wave owns weight columns, global->VGPR prefetch);
     swap-AB for small rows/expert; persistent / stream-K over (expert, n) work with overlapped prologue/epilogue; fused stage chain
     (route + quant + gate-up + act/quant + down + weighted reduce, no 4.3 GB per-route write in EP1); per-stage row alignment.
  5. Pick one, estimate per shape vs the current kernel, then prototype the hot loop alone (one expert, fixed M/N/K) before integration.
**Step 0 result (2026-10-28, h13-17; snapshots FlyDSL main `64e409b`, aiter main `bbef5a6`, both 10-28, shallow clones in `/tmp/up/`):**
no upstream gfx1250 grouped MoE GEMM keeps weights out of LDS. Candidates:
- aiter main `mxfp4_preshuffle_gfx1250_tdm.py` (same kernel, +92 lines vs our orig): (a) `a_preshuffled` A layout (M/16,K/128,4,16,16B),
  no LDS pad, lane reads at `kgrp*256+lane16*16`/`+512` like B; fp4 act only; producer `moe_apreshuffled_quant_scatter.py` (new);
  enabled per tuned-CSV row `a_preshuffle=1`; (b) per-format IGLP params, "prefill" = 256x256x256 (a4w4 mg 13x11, ds_first 16);
  (c) **race fix: `s_wait_dscnt(0)` before the REUSE fence when issue_fn recycles the LDS buffer** -- candidate for our EP1 fault.
  Upstream prefill CSV (DS-like 7168/2048 E64 topk8, T>=2048): a_preshuffle=1, tiles 256^3 both stages, warp 2x2, nbuf 4.
  New CSV knobs: split_k, persistent_workers(2), lds_soa_load_interleave, rolled_drain, tdm_b_th. Needs flydsl==0.4.0 (pip has 0.4.0/0.4.1).
- FlyDSL main `kernels/moe/moe_a8w4_mxscale_gfx1250.py` (530 lines): same TDM design (B via LDS, `B_LDS_ROW`), WMMA 16x16x128, cluster_n mcast.
- aiter gluon `_gluon_kernels/gfx1250/moe/moe_op_gemm_{a4w4,a8w4,a16w4}.py`: x, w, scales all TDM -> LDS.
- aiter `mega_moe_gfx1250/` and `moe_g2l_lut.py`: identical to local branch (nothing new).
- **Race fix (c) ported locally** (h13-17, uncommitted, `k_step` before the fence; patch now `/workspace/proxy_work/aiter_local_1028b.patch`).
  Op test GPU0, T=8192, base -> fix (us): EP8 gemm1 2336 -> 2316, gemm2 1121 -> 1159; EP1 gemm1 2407 -> 2320, gemm2 2468 -> 2480; all pass.
  = perf-neutral (noise ~5%). EP1 both-tile fault re-test with the fix still TODO (server, several starts).
- **Blocked:** running the upstream clone (`/tmp/up/aiter_main` + `pip --target /tmp/up/fly040 flydsl==0.4.0`) was denied by the
  auto-mode classifier (external code). Needs user OK before benchmarking upstream a_preshuffle 256^3 on our shapes.
  Isolated env now exists (user OK): venv `/tmp/up/venv040` (--system-site-packages, flydsl 0.4.0; main env stays 0.3.4.1).
  User OK'd running the clone. venv setup: `zz_optvenv.pth` = `site.addsitedir(/opt/venv/...site-packages)` (torch / editable triton;
  flydsl 0.4.0 still wins), pytest added, `/tmp/up/aiter_main/3rdparty/composable_kernel` -> symlink to local CK (same commit `96afae1d`,
  carries the brace fix). Needs `--no-check-aot-cache`. CSV rows must use cu_num=96 (this partition). A/B script `/tmp/up/scripts/up_ab.sh`,
  CSVs `/tmp/up/csv/{ep8,ep1}_256_ap{0,1}.csv` (upstream a_preshuffle prefill row, shape fields swapped), logs `/tmp/up/logs/up_*.log`.
  Upstream EP8, no CSV match (fallback tile_m=64 n_warp=4 nbuf=2): gemm1 2923 / gemm2 1481 us, pass (local w/ rule: 2316 / 1159).
  **Upstream A/B, EP8 op test (GPU0, T=8192, 1 run each, noise ~5%), gemm1 / gemm2 / sum us, all pass:**
  | variant | gemm1 | gemm2 | sum |
  |---|---|---|---|
  | local branch + rule + race fix (0.3.4.1) | 2316 | 1159 | 3475 |
  | upstream, no CSV (tile_m 64, nbuf 2) | 2923 | 1481 | 4404 |
  | upstream, our rule tiles via env | 2324 | 1132 | 3456 |
  | upstream CSV 256^3 w2x2 nbuf4 cluster_n4 nsp1 soa1, a_preshuffle=0 | 2057 | 1226 | 3283 |
  | same, a_preshuffle=1 | 2144 | 1104 | 3248 |
  -> same tiles: upstream == local (kernel changes alone buy nothing). Upstream's 256^3 prefill row: gemm1 -11%, sum -6%, unlike our
  earlier 256x256 w2x2 (+6..+22%) -- that row also sets cluster_n=4 (weight multicast), next_stage_prefetch=1, lds_soa_load_interleave=1.
  a_preshuffle: gemm1 +4%, gemm2 -10% (within ~2x noise; repeat). EP1 hits the 2bafed472 int32 overflow upstream too (unfixed on main);
  `_flat_base` fix applied to the /tmp clone only (orig `/tmp/up/grouped_gemm_mxfp4.py.orig`).
  **Upstream EP1 (GPU0):** EP8-rule tiles via env (NOT the EP1 rule) 2308 / 2552; 256^3 ap0 2052 / 3006; ap1 2039 / 2994, all pass.
  -> gemm1 -11% again, but 256^3 gemm2 +18% for inter 512 -> hybrid: stage1 256^3 row, stage2 keep the rule.
  **Local kernel reproduces it (no upstream code needed; local supports all knobs except a_preshuffle):** GPU2, EP8, CSV 256^3 ap0:
  gemm1 2181 / gemm2 1212 vs rule 2466 / 1211 (gemm1 -11.6%). Hybrid CSVs `/tmp/up/csv/{ep8,ep1}_hyb.csv`
  (stage2 = rule tiles, m/n_warp2 1/4, cluster2 -1), script `/tmp/up/scripts/loc_hyb.sh`, logs `/tmp/up/logs/loc_*`.
  **Noise warning:** same config on GPU2 run-to-run: 256ap0 gemm1 2181 -> 2401, hyb gemm1 2873 -> 2205 (+-15%!). Single runs are NOT
  enough. Hybrid gemm2 consistently slow (EP8 1461/1489 vs rule 1211/1218 at identical stage2 tiles; EP1 4250 vs 2430) -> a row-level
  knob (next_stage_prefetch=1 / lds_soa_load_interleave=1 / waves_per_tensor) hurts stage2 with rule tiles. Running interleaved
  3x3 medians: `/tmp/up/scripts/loc_rep.sh <ep> <gpu> <reps>` -> `/tmp/up/logs/rep_{ep8,ep1}.txt` (EP8 GPU2, EP1 GPU4).
  **3x3 interleaved result (local kernel, T=8192, us, median [gemm1 range]):**
  | | EP8 gemm1 | EP8 gemm2 | EP8 sum | EP1 gemm1 | EP1 gemm2 | EP1 sum |
  |---|---|---|---|---|---|---|
  | rule | 2383 [2371-2458] | 1226 | 3609 | 2541 [2523-2790] | 2488 | 5029 |
  | 256^3 ap0 row | 2850 [2509-2875] | 1165 | 4015 | 2813 [2292-2872] | 3258 | 6071 |
  | hybrid | 2238 [2211-2842] | 1410 | 3648 | 2850 [2515-2947] | 4243 | 7093 |
  -> **256^3 stage1 is bimodal (~2200 vs ~2850 per process), rule is stable.** The earlier "-11%" was the fast mode. Median: no win,
  rule stays. Hybrid gemm2 always slow (row-level knob). Unexplained: what selects the mode (hypothesis: cluster_n=4 cluster placement /
  power state; test cluster_n=1 256^3 and per-process repeat). Upstream adoption: nothing to take except the race fix (already ported).
  **Bimodality probe (user: option 3, 2026-10-28).** EP8 256^3 row, 5 interleaved rounds per GPU, gemm1 us (all pass):
  | cluster_n | GPU2 | GPU4 |
  |---|---|---|
  | 4 (row) | 2190 2157 2577 2873 2798 | 2809 2810 2477 2234 2815 |
  | 1 | 2358 2341 2521 2429 2599 | 2596 2515 2593 2824 2563 |
  | -1 (default) | 2477 2309 2480 2347 2412 | 2716 2656 2547 2683 2543 |
  -> fast mode (~2160-2230) only with cluster_n=4; c1/-1 narrower, median no better than rule (GPU2 rule 2383). GPU4 slower than GPU2.
  Logs show same kernel / no compile difference between fast and slow runs; timing = 5 warmup + 101 iters, graph off, fixed seed.
  Next probe running: `/tmp/up/scripts/clk_probe.sh 2 5 3000` (c4 vs c1, --iters 3000, amd-smi clock/power/violation every ~0.3 s,
  `/tmp/up/logs/clk_smi_g2.txt`, `clk_runs_g2.txt`) -> is the mode a clock effect or cluster mechanics (compare in cycles)?
  **Clock probe result (GPU2, --iters 3000, gemm1 window = first busy block, amd-smi GFX% saturates at 50):**
  | run | gemm1 us | clk MHz | power W | PVIOL % | gemm1 Mcyc |
  |---|---|---|---|---|---|
  | c4 fast (3 runs) | 2200-2225 | 1110-1112 | 1590-1620 | 33-34 | 2.44-2.47 |
  | c4 mid (1) | 2429 | 1102 | 1445 | 50 | 2.68 |
  | c4 slow (1) | 2831 | 1136 | 1311 | 60 | 3.22 |
  | c1 (5 runs) | 2427-2480 | 986-995 | 1410-1421 | 66 | 2.42-2.46 |
  gemm2 (c1 and c4 identical): ~1120 MHz, 1.31-1.34 Mcyc.
  -> **slow mode is NOT a clock effect**: it runs at a slightly HIGHER clock with LOWER power but +30% cycles = stalls (cluster
  mechanics). **c4 fast mode == c1 in cycles**; its whole time gain is clock (+12%: ~1110 vs ~990 MHz) -- multicast removes
  redundant A reads, so less energy per cycle under the power cap. Confirms "fewer bytes -> higher clock" (PPT-limited).
  Implication for Track A/B: energy per byte moved is a first-order lever, not just LDS cycles.
  Open: what fixes the mode per process (buffer address / alignment? cluster -> CU placement?). Next probe: repeat the timing
  several times inside ONE process (mode stable within a process -> per-process state such as addresses), plus rule's cycles.
  **Cluster knowledge collected -> `proxy-moe/GFX1250_CLUSTER_NOTES_h13-17_1028.md`.** Key code fact: CSV cluster_m=-1 with
  cluster_n=4 becomes cluster_m=4 for stage1 = 16-WG cluster with a cluster barrier every num_buffers k-steps, while B multicast is
  ~never active at ~256 rows/expert. Next probe (env only): `AITER_FLYDSL_MXFP4_CLUSTER_M=1` / `2` and `AITER_FLYDSL_DISABLE_CLUSTER_SYNC=1`.
  Caveat for that A/B: 256-row tiles pad heavily at EP8/EP1 ~256 rows/expert (T=8192); our earlier 256x256 w2x2 was +6..+22%.
**Track A details -- implement "weights direct global -> VGPR" in `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`**
(see "Main-loop LDS pipeline investigation" + "Register-blocking arithmetic" below). Facts for the change:
- Naming: `a` = tokens (activations), `b` = weights. WMMA `v_wmma_scale_f32_32x16x128_f4`: first operand = 32 weight cols (16 VGPR),
  second = 16 token rows (8 VGPR). Per wave (tile 128x256 w1x4): 2 weight frags x 8 token frags = 16 WMMA, acc 256 VGPR, kernel VGPR 192.
- Weight LDS layout == global layout (no pad): per 16-col group ("outer row") per k128 a contiguous 1 KB (`B_LDS_ROW = tile_k*8`);
  global outer row = `expert * (N/16) + n/16`, row pitch `Kp16 = K/2*16`; lane reads 16 B at `kgrp*256 + lane16*16` and `+512`
  (`lds_b_lane_off`, `load_b`, kernel ~1020 / ~1082). Weight scales (SB) can stay TDM -> LDS.
- Plan: constexpr switch (env `AITER_TDM_B_DIRECT`, add to cache_tag + kernel name); drop the B job from the TDM job list and fix the
  per-stage `tensor_wait` counts; `buffer_load_dwordx4` the weight frags with a cross-iteration prefetch (2 deep = +64 VGPR; keep total
  <= 256 so 2 waves/SIMD survive); validate pass + per-kernel us + ATT (s_wait_dscnt share) + VGPR, then server A/B (EP8 + EP1).
- Expected: LDS traffic per WG per k128 72 KB -> 40 KB; gate-up iteration ~581 cyc -> toward WMMA-bound ~256-350 cyc.
**Local, uncommitted state (h13-17, do not assume on another node):** aiter `/sgl-workspace/aiter` branch feat/proxy-model-gfx1250-1023
@ 7ea6f2f46 + edits, full diff in `/workspace/proxy_work/aiter_local_1028.patch`: grouped_moe_gfx1250.py (prefill tile rule EP8/EP1,
`_layout_align`), grouped_gemm_mxfp4.py + kernel (`layout_align_m` gap-tile skip), kernel TEMP probe `AITER_TDM_PROBE` (epilogue
re-indented; original `/workspace/proxy_work/mxfp4_preshuffle_gfx1250_tdm.py.orig_1028` = pre-probe AND pre-layout_align version),
3 image edits, CK `amd_tdm_store` brace fix in 3rdparty (orig `/workspace/proxy_work/amd_buffer_addressing.hpp.orig_1028`).
Before committing aiter: remove the probe, keep rule + layout_align_m.
**Best config (TP8, 12L):** `/workspace/proxy_work/allon_tp8.env` (DDA on) + `--attention-backend aiter --page-size 64`, EP1 adds
`AITER_CONFIG_GROUPED_FMOE=/workspace/proxy_work/ep1_grouped_fmoe_gfx1250_v3.csv`; tile rule on by default (`AITER_TDM_PREFILL_TILE_RULE`).
**Repro:** op test (per-kernel us): `cd /sgl-workspace/aiter && HIP_VISIBLE_DEVICES=0 python3 op_tests/flydsl_tests/test_flydsl_grouped_gemm.py
--no-check-aot-cache --scenario kernel --data-init uniform --scale-init auto --data-format a4w4 --experts 64 --topk 2 --model-dim 16384
--inter-dim 4096 --tokens 8192 --act silu --no-bias` (EP1: `--experts 512 --topk 16 --inter-dim 512`); server: `EPS=8 CONCS="2 16" bash
/workspace/proxy_work/allon_tp8_1028.sh <tag>`; ATT: see "Prefill MoE GEMM diagnosis" (`proxy-moe/scripts/att/att_hot_loop_gfx1250.py`).
**Pass criteria:** op test pass=True; EP8 gate-up < 2300 us (rule baseline 2274-2370, op test runs power-limited at ~1.1-1.4 GHz);
server 8k/1k TTFT EP8 c=2 ~289 ms / c=16 ~1481 ms, EP1 342 / 1781 (rule on); smoke passed; no Memory access fault.

**Open items (2026-10-28):**
- Tile (paused): (1) shape rule = tiles only for inter 4096 + prefill token counts (threshold T from an op-test token sweep first),
  never EP1 / decode; (2) ISA resource check (VGPR / scratch / LDS) of the two tiles; (3) EP8 server A/B on all-on, several starts
  (fault risk), + teacher-forced check; (4) HBM bytes / L2 hit per kernel; (5) kernel-level 16.46 -> 14.40 repro (f19-14 bench not on this node).
- **Kernel (user: the fundamental fix, 2026-10-28): swap-AB grouped MoE GEMM for small rows/expert.** Rows per expert = T x topk / E_total
  (same for EP1 and EP8: T=1024 -> ~32), so 64/128-row M tiles are mostly padding at small T and in decode. Put the weight rows on
  the WMMA M side (EP1 gate-up N=1024, down N=16384 -> full 128-row tiles) and tokens on the N side (pad only to WMMA_N=16).
  Needs a new weight preshuffle/scale layout and A/B roles swapped in the TDM pipeline; benefits small-T prefill and decode.
  Cheaper interim tested and REJECTED: tile_m=32 at small T (`tile32_tokens_h13-17_1028/`, whole MoE, 2 rounds): EP1 T=512 -2.4%,
  1024 +9.5%, 2048 +0.6%; EP8 512 -1.2%, 1024 +12.9%, 2048 +54%. So padding compute is NOT the dominant small-T cost (per-tile fixed
  cost + weight re-reads are) -> swap-AB's small-T gain is uncertain; ATT the small-T / decode kernel before starting it.
  Prefill coalescing (more tokens per MoE call) remains a scheduler-side option.
- Fault: EP1 with BOTH stage tiles faults 3/3 (rocdevice: RCCL `ncclDevKernel_Generic_32`); s1-only / s2-only / serialized clean (1 run each). Race suspected.
- Perf: EP1 MoE prefill CSV rows (T 1024-8192 `no grouped CSV config matched`); a16w16 M=32-64 gluon tuning (decode qkv/o + lm_head);
  AR wait / rank skew (decode c=2 AR 36% of GPU time); all-on not yet measured at 1k/1k EP1, 70k, c=64 EP1.
- Housekeeping: upstream CK fix; aiter local edits uncommitted; push claude-skills; unexplained: decode-vs-prefill logprob gap ~0.45,
  EP8 c=16 ITL drift in the tile A/B; f19-14 `allon.env` / CSVs / bench when that node can upload.
**Files:** `/workspace/proxy_work/{allon_tp8.env, allon_tp8_1028.sh, ar_server_ab_1028.sh, tf_tp8_1028.sh, bf16_prefill_torch_tp8.csv, bench/attn_proj_tp8_bench.py}`.

## Environment (h13-17, 2026-10-28)

- 8x gfx1250 (`gfx_target_version 120500`), all idle (0% GFX, no processes). Physical device = PCI domain; each has 2 partitions
  (location_id 1024/1025): GPU 0+1 = domain 1, 2+3 = domain 2, 4+5 = domain 3, 6+7 = domain 4 (assuming GPU index follows KFD node order 2..9).
- dmesg: only boot-time `RAS early init service failure! ret:-95` per device, SMU initialized OK; no amdgpu runtime / pcie_pl errors.
- This is a fresh container: aiter `/sgl-workspace/aiter` = ROCm main `2bb549102` (detached, clean clone + 2 local mods), NOT the kk branch;
  `kkHuang-amd/aiter feat/proxy-model-gfx1250-1023` tip = `7ea6f2f46` (exists on the fork). sglang = `5b1cda03bb release for testing`,
  not `HaiShaw/feat/proxy-model-gfx1250-1023`.
- `/shared_nfs` here is a node-local md RAID, not shared with f19-14: `/shared_nfs/kk/proxy_work_f19-14_1027/` does not exist on this node
  (no `moe_ab_1028.sh`, no `bench/moe_prefill_bench.py`). `/workspace/proxy_work/` (node-local) still has the EP1/EP8 CSVs and `proxy_12l`.
- Switched (same procedure as MI455_RUN_h13-17.md "Container rebuild 2026-10-26"): image edits backed up to
  `/workspace/proxy_work/imagebase_backup_1028/` and stashed (`imagebase-1028`); aiter -> local branch `feat/proxy-model-gfx1250-1023` = kk @ `7ea6f2f46`
  + re-applied all 3 image edits (uncommitted): `csrc/cpp_itfs/torch_utils.py` torch.Stream fix, `pa_mqa_logits_fp4_prefill.py`
  lru_cache -> cache (as a patch; file differs on the branch), gfx950 `mla_a8w8_qh64_..._nm.co` (branch blob == image base);
  stale image JIT `.so` + `build/` moved to `imagebase_backup_1028/aiter_jit/` (`Relu2` start failure, as on 10/23); sglang -> haishaw `feat/proxy-model-gfx1250-1023` @ `08db4167c7`.
  RCCL already 2.32.3 (`ncclGetVersion` 23203). Both installs are editable (import from `/sgl-workspace/*`).
- Baseline to compare (fixed_seq_len_mi455/RESULTS.md, 10/23 config): 8k/1k mean TTFT EP1 c=2 445 ms, c=16 3226 ms; EP8 418 / 3060 ms.

## Results

### TP8 server A/B of the f19-14 tiles on the OLD h13-17 config (2026-10-28 01:01-01:36 UTC, h13-17, GPUs 0-7)

`/shared_nfs/kk/results/proxy_12l/tile_ab_tp8_h13-17_1028/` (first attempt in `fail1_stale_jit/`: stale image JIT `.so`, no `Relu2`).
12L proxy, 8k/1k, triton attention, arms A -> B (A + `AITER_TDM_TILE_*` 128/256/128/4 ; 128/256/256/2) -> A2. Mean / median TTFT ms, median ITL ms.
NOTE: A is the old h13-17 config, not the current best (see CONTINUE HERE). Smoke passed in every arm that started.

| EP | arm | c=2 TTFT mean / med | c=16 TTFT mean / med | c=16 ITL |
|---|---|---|---|---|
| 1 | A | 557 / 424 | 2918 / 2750 | 19.42 |
| 1 | B | FAULT | - | - |
| 1 | A2 | 554 / 419 | 2924 / 2758 | 19.44 |
| 8 | A | 527 / 408 | 2756 / 2597 | 19.22 |
| 8 | B | 486 / 377 | 2563 / 2442 | 19.15 |
| 8 | A2 | 525 / 402 | 2758 / 2605 | 16.47 |

- EP8: B vs mean(A, A2) TTFT -7.6% (c=2) / -7.0% (c=16); A/A2 TTFT within 0.4%. A2 c=16 ITL 16.47 vs A 19.22 is unexplained drift
  (TTFT unaffected) -> ITL from this run is not usable.
- EP1 B: `Memory access fault by GPU node-3 (GPU 1) ... Page not present`, write fault (dmesg GCVM_L2_PROTECTION_FAULT, PERMISSION_FAULTS 0x5),
  during the first 8k prefill (smoke with short prompts passed). Only change vs A is the tile env, which also overrides the EP1 CSV v3 rows.
  Shape E512 top16 inter512 (stage2 K=512) was never validated with these tiles. Not yet isolated (stage1 vs stage2, prefill vs decode row).
  The next arms started cleanly, so the GPU recovered.

### TP8 all-on (f19-14 10/27 config ported) vs old h13-17 config (2026-10-28 01:37-01:48 UTC, h13-17, GPUs 0-7)

`/shared_nfs/kk/results/proxy_12l/allon_tp8_h13-17_1028/`, `allon_tp8.env` + `--attention-backend aiter --page-size 64`, 12L, 8k/1k, smoke passed both.
Old = mean of A / A2 from the tile run above (same session, triton attention). One all-on run per EP (no repeat yet).
Mean TTFT ms (median), median ITL ms, output tok/s.

| EP | c | old TTFT | all-on TTFT | old ITL | all-on ITL | old tok/s | all-on tok/s |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 556 (422) | **364 (284), -34%** | 5.78 | 5.64 (-2%) | 312 | 333 (+7%) |
| 1 | 16 | 2921 (2754) | **1886 (1771), -35%** | 19.43 | 15.57 (-20%) | 678 | 832 (+23%) |
| 8 | 2 | 526 (405) | **330 (263), -37%** | 6.36 | 6.32 (-1%) | 281 | 296 (+6%) |
| 8 | 16 | 2757 (2601) | **1695 (1597), -39%** | 19.22 / 16.47 (A/A2 drift) | 16.29 | 691 / 730 | 822 (+13-19%) |

- Log check (EP1): backend aiter, page 64, fp8 KV; TP8 attn CSV hit for every prefill M (misses only M <= 64 = decode, by design);
  router bf16x3 (N 1536 fp32-out) on gluon default; lm_head via aiter tgemm (N 25256). FlyDSL extend kernels compiled.
  vectorized_5d / FlyDSL pa_decode NOT yet confirmed from a kernel trace.
  ISL 8192 / OSL 1024 fixed (`--random-range-ratio 1.0`); c=2 = 16 prompts, c=16 = 128 prompts.
- **Decode kernel trace confirms the attention path** (`/shared_nfs/kk/results/proxy_12l/allon_prof_tp8_h13-17_1028/`, EP8, 1k/1k,
  torch profiler 20 steps, `/workspace/proxy_work/p0_profile_env.sh` = mi455_p0_profile.sh + ENVF/ATTN/EXTRA_ARGS): every layer runs FlyDSL
  `pa_decode_tile_kernel_kvfp8_qbf16_d128_bs64_gqa16_mtp1{_sw2048}` (9 sliding + 3 full) + plan/reduce; no triton decode attention.
  pa_decode only reads the SHUFFLE 5D cache, so vectorized_5d is live. `_qk_norm_rope_kv_kernel` 12/step (fused prep live).
  c=2 6.50 ms/step GPU: AR `ar_ll_gfx1250` 36%, MoE 25%, a16w16 10%, attn 6.7%. c=16 15.6 ms/step: MoE 47%, AR 30%.
- **SILU_MUL unblocked:** aiter `module_activation` build error = stray `}` in CK `ck_tile/core/arch/amd_buffer_addressing.hpp`
  `amd_tdm_store` (line 3148, before `#else`; `amd_tdm_load` is correct). Only compiled with CK_TILE_ENABLE_TDM_FEATURE (gfx1250).
  Removed locally (uncommitted, in aiter's 3rdparty/composable_kernel @ `96afae1d5`; original saved `/workspace/proxy_work/amd_buffer_addressing.hpp.orig_1028`).
  `aiter.silu_and_mul` builds, max |d| vs torch 0.015 (bf16). Upstream CK fix not filed yet.
  Server A/B running: `/tmp/allon2.sh` -> `/shared_nfs/kk/results/proxy_12l/allon2_tp8_h13-17_1028/` (per EP: all-on repeat `allon_r2`, then `allon_silu`).

### All-on repeat + SILU_MUL A/B (2026-10-28 01:56-02:16 UTC, h13-17; `allon2_tp8_h13-17_1028/`, 8k/1k)

Mean TTFT ms (median), median ITL ms, output tok/s. r1 = first all-on run above. Smoke passed all; `module_activation` loaded on all 8 ranks.

| EP | c | all-on r1 | all-on r2 | + SILU_MUL |
|---|---|---|---|---|
| 1 | 2 | 364 (284), 5.64, 333 | 362 (280), 5.30, 351 | 364 (285), 5.68, 330 |
| 1 | 16 | 1886 (1771), 15.57, 832 | 1881 (1769), 15.56, 833 | 1880 (1767), 15.35, 841 |
| 8 | 2 | 330 (263), 6.32, 296 | 329 (261), 6.31, 297 | 342 (263), 6.21, 304 |
| 8 | 16 | 1695 (1597), 16.29, 822 | 1693 (1591), 16.57, 814 | 1692 (1592), 16.52, 815 |

- All-on reproduces: TTFT r1/r2 within 0.6%; c=2 ITL varies up to 6% run to run (EP1 5.64 vs 5.30), c=16 ITL within 2%.
- SILU_MUL: no measurable effect (all deltas inside r1/r2 spread). Keep it in the all-on set (works now) but it is not a gain at 8k/1k.
- Greedy exact-output check running: `/workspace/proxy_work/greedy_tp8_1028.sh` -> `/shared_nfs/kk/results/proxy_12l/greedy_tp8_h13-17_1028/`, log `/tmp/greedy_tp8_1028.log`.

### Correctness on dummy weights: greedy and teacher-forced logprobs (2026-10-28, h13-17)

- Greedy exact-output (`greedy_tp8_1028.sh` -> `greedy_tp8_h13-17_1028/`) is NOT usable on the 12L dummy ckpt: the old config is
  deterministic (ref vs ref2 identical), but ANY config change (incl. old config EP8 -> EP1) diverges at token 0. First-token logprob
  ~ -9.8 vs uniform -12.2, i.e. top-1 prob ~5e-5 with near-ties, so argmax flips on normal numeric noise. GSM8K is meaningless here too.
- Teacher-forced instead (`/workspace/proxy_work/{tf_dump.py,tf_compare.py,tf_dump_env.sh,tf_tp8_1028.sh}` ->
  `/shared_nfs/kk/results/proxy_12l/tf_tp8_h13-17_1028/`): 9 prompts (16..8000 tokens) + 32 greedy tokens; prefill logprob of every
  position in the last 1024 prompt positions (memory cap), |d logprob|:

| comparison | mean | p99 | max |
|---|---|---|---|
| old EP8 vs old EP8 rerun | 0 | 0 | 0 |
| old EP8 vs old EP1 (EP-only numeric baseline) | 0.196 | 0.646 | 1.015 |
| old EP8 vs all-on EP8 | 0.246 | 0.811 | 1.329 |
| old EP8 vs all-on+SILU EP8 | 0.250 | 0.826 | 1.190 |
| old EP1 vs all-on EP1 | 0.253 | 0.830 | 1.530 |
| old EP1 vs all-on+SILU EP1 | 0.250 | 0.840 | 1.213 |

  All-on is ~1.25-1.3x the EP-only baseline (it changes attention backend, KV layout, GEMM backends at once), same growth with prompt
  length (baseline 0.10 -> 0.23, all-on 0.10 -> 0.30 from 16 to 8000 tokens), no outlier prompt.
- Decode vs prefill self-consistency (decode-path logprob of the 32 generated tokens vs prefill recompute, same config), mean / p99:
  old 0.44 / 1.36 (EP8), 0.48 / 1.37 (EP1); all-on 0.52 / 1.37, 0.45 / 1.28; +SILU 0.46 / 1.17, 0.49 / 1.38. Same level for every config,
  so the FlyDSL pa_decode path is no worse than triton decode (the ~0.45 gap itself is common to all; likely fp8 KV read in decode vs bf16 in prefill, unverified).
- Verdict: no sign of a numeric error from the all-on config on TP8. Limit: dummy weights only show "no larger deviation than normal noise", not task accuracy.

### EP1 tile fault investigation (2026-10-28 ~02:50 UTC, h13-17; `/shared_nfs/kk/results/proxy_12l/ep1_tile_fault_h13-17_1028/`)

- Op test (`test_flydsl_grouped_gemm.py`, GPU 0 alone, EP1 rank E512 top16 inter512, a8w4, no bias) never faults, all `pass=True`:
  T=8192 default / s1-only / s2-only / both; server env (CSV v3 + tile env, CSV n_warp kept) at T=1..64; T=8193/8191/6000/4097;
  skewed routing `AITER_MOE_NUM_EXPERT_ACTIVATED`=16/64/200 at T=8192. `PYTORCH_NO_HIP_MEMORY_CACHING=1` is unusable (hipgraph capture fails).
- Perf on EP1 (op test, us): T=8192 default 10175, s1-only 10336, s2-only 10964 (+7.7%), both 10167 -> no gain on EP1.
  Decode T=1..64 with the tile env on top of CSV v3: 1.4-2x SLOWER (e.g. T=16 675 -> 1039 us). The env tiles must never be global for EP1.
- Buffer sizing is not the cause: `contiguous_m` uses `_align_m = max(tile_m, tile_m2)` after the env override (grouped_moe_gfx1250.py ~678).
- **The faulting kernel in the EP1 B server log is RCCL, not MoE:** rocdevice line `kernel: ncclDevKernel_Generic_32` (8k-prefill AR,
  256 MB > custom AR max 1 MB -> RCCL). Either an RCCL fault coinciding with the tile arm, or MoE corrupting memory RCCL then touches.
  Repro running: `/tmp/ep1_tile_repro.sh` -> `/shared_nfs/kk/results/proxy_12l/ep1_tile_repro_h13-17_1028/` (EP1 + tiles on old config, then on all-on).
- **Repro (02:43-02:50 UTC): deterministic.** EP1 + tile env faults at the first 8k prefill on BOTH the old config and all-on
  (`ep1_tile_repro_h13-17_1028/`; old: rocdevice `ncclDevKernel_Generic_32` x2; all-on: fault, no kernel line). EP1 without tiles: 5/5 server
  runs today clean. So tile-triggered, most likely an MoE OOB write on the server path that RCCL then trips over (not proven).
  Bisect running: `/tmp/ep1_tile_bisect.sh` -> `ep1_tile_bisect_h13-17_1028/` (s1-only, s2-only, both + AMD_SERIALIZE_KERNEL=3; old config, c=2).
- **Bisect (02:51-02:59 UTC, `ep1_tile_bisect_h13-17_1028/`, old config, c=2, one run each): NO fault in any arm.**
  s1-only (s2 default) clean, c=2 TTFT 533 ms; s2-only clean, 568 ms; both + `AMD_SERIALIZE_KERNEL=3` clean (759 ms, serialized).
  Both tiles without serialization: 3/3 faulted (tile run, repro old, repro all-on). Reading: needs BOTH stages at tile_m=128 AND
  concurrent execution -> looks like a race (cross-stream or cross-kernel ordering), not a plain OOB; consistent with the fault
  landing in the RCCL kernel. No explicit tile_m == tile_m2 branch in grouped_moe_gfx1250.py. Caveat: s1/s2-only are single runs.
- Practical: the tiles give EP1 nothing (prefill flat, decode 1.4-2x slower, s1/s2-only TTFT no better than A 557 ms), so EP1 must not get them.

### AR backends on TP8: aiter CAR vs RCCL DDA on / off (2026-10-28 ~03:00-03:35 UTC, h13-17; `/shared_nfs/kk/results/proxy_12l/ar_dda_h13-17_1028/`)

- CORRECTION: the RCCL in this container is NOT the 9c52e34 / 16ded0c fix build (ncclGetVersion 23203 matched, files / build-id do not).
  It is the image's own RCCL 2.32.3 (`_rocm_sdk_libraries/lib/librccl.so.1`, built from `/sgl-workspace/rocm-systems/projects/rccl`, 10-27 03:51).
  The user confirmed this build contains the DDA hang fix. All runs above used it with `RCCL_DDA_ENABLE=0`.
- bf16 [M,16384], TP8, CUDA graph. RCCL: `scripts/rccl_ar_bench.py` (NCCL_CUMEM_ENABLE=1), median over ranks (max). CAR: aiter
  `test_custom_allreduce.py -t 8 -m graph -s M,16384` (env fix PYTHONPATH), min-max over ranks, err ~1% (bf16). Slowest rank compared; harnesses differ.

| size | RCCL DDA off | RCCL DDA on | aiter CAR |
|---|---|---|---|
| 64 KB (M=2) | 34.4 | **11.8 (12.0)** | 9.7-12.6 |
| 128 KB (M=4) | 35.2 | **13.0 (13.2)** | 11.8-14.6 |
| 256 KB (M=8) | 38.5 | **16.9 (17.0)** | 19.4-21.3 |
| 512 KB (M=16) | 42.3 | **29.1 (29.3)** | 29.6-33.0 |
| 1 MB (M=32) | 49.8 | **45.6 (45.9)** | 46.0-48.5 |
| 2 MB (M=64) | 68.7 | **37.0 (37.3)** | 89.5-92.3 |
| 32 MB (M=1024) | 321 | **213** | - |
| 64 MB (M=2048) | 450 | **354** | - |
| 128 MB (M=4096) | 784 | **690** | - |
| 256 MB (M=8192) | **1115** | 1146 | - |

- DDA on beats DDA off at every size up to 128 MB (2.9x at 64 KB, 1.9x at 2 MB, 1.5x at 32 MB), ties at 256 MB (+3%).
- DDA on vs CAR (slowest rank): equal at 64-128 KB (within ~5%), DDA 11-20% faster at 256 KB-1 MB, 2.5x faster at 2 MB.
  DDA off repeat (dec_off2) within 2% of dec_off. bench teardown hangs (exit 124 after printing, known).
- Next: server A/B on all-on EP8: A = current (CAR <= 1 MB + DDA off), B = CAR <= 1 MB + DDA on, C = no CAR (`--disable-custom-all-reduce`) + DDA on, A2.
- **Server A/B (03:31-04:12 UTC, `ar_ab_tp8_h13-17_1028/`, all-on EP8, `/workspace/proxy_work/ar_server_ab_1028.sh`), all smoke passed, DDA-on starts clean (2/2).**
  Mean TTFT ms / median ITL ms / output tok/s. A/A2 spread: <= 1.7% on tok/s and TTFT.

| point | A | B (CAR + DDA) | C (DDA only) | A2 |
|---|---|---|---|---|
| 8k/1k c=2 | 329 / 6.31 / 298 | 328 / 6.12 / 309 | 328 / 6.15 / 307 | 328 / 6.23 / 302 |
| 8k/1k c=16 | 1692 / 16.03 / 829 | 1692 / 16.08 / 828 | 1694 / 16.80 / 800 | 1691 / 16.14 / 827 |
| 1k/1k c=2 | 92 / 6.06 / 330 | 90 / 6.20 / 318 | 88 / 5.89 / 340 | 92 / 6.12 / 324 |
| 1k/1k c=64 | 1029 / 29.61 / 2053 | 1009 / 29.24 / 2090 | 1010 / 29.72 / 2053 | 1012 / 29.61 / 2059 |

  vs mean(A, A2): B = -1..+3% everywhere (1k c=64 tok/s +1.6%, 8k c=2 +2.8%; 1k c=2 -2.5%), i.e. at noise level.
  C mixed: 1k c=2 +4.1% tok/s / TTFT -4.9%, but 8k c=16 -3.3% tok/s, ITL +4.4%. The microbench wins (DDA 11-20% at 256 KB-1 MB,
  12-34% at 32-128 MB) do not show up end to end: AR kernel speed is not what limits the step (earlier finding: AR time is mostly
  wait from per-rank skew, MI455_RUN_h13-17.md "EP1 + CSV v3 profile"; not re-verified here). Single runs for B and C.

### Decode AR wait + prefill profiling attempt (2026-10-28 04:17-05:00 UTC, h13-17)

- Script `/workspace/proxy_work/prof_stage_1028.sh` (torch profiler, `profile_by_stage`, 8k/1k load, all-on EP8).
- **Prefill (EXTEND) under the torch profiler hangs, DDA on AND off** (`prof_stage_allon_ep8{,_ddaoff}_h13-17_1028/`): DECODE stage
  (5 steps, all 8 ranks) is captured, then the first profiled prefill hangs with `queue_interposition.cpp:470 Async signal handler still
  waiting` (rocprofiler interposition), server never recovers; killed by hand (-9). Not DDA-related. Prefill breakdown still missing ->
  next try rocprofv3 --kernel-trace (not --att) or a non-profiler timing split.
- **Decode AR wait, EP8 c=2, 8k ctx (6.6 ms/step GPU):** tools needed new kernel names (`ar_ll_gfx1250`, `a8w4_tdm`): patched copies
  in /tmp/ana (shared scripts not yet updated). `ar_wait_attrib`: mean AR 84 us/call, fit AR = 11.3 + 1.06*skew (r=0.99), min-over-ranks
  p50 11.5 us -> ~1.8 ms/step (~27% of the step) is waiting for the slowest rank. `ar_segment_skew`: busy max-mean 1.73 ms/step vs span
  1.81 -> kernel-time differences, not host idle. Largest spread: MoE gate-up 0.50, MoE down 0.36, a16w16 0.24, pa_decode 0.20 ms/step.
  `rank_skew`: ranks 0,1,6,7 (physical devices domain 1 and 4) are slower in ALL categories (rest 13.7-14.4 vs 12.5-12.7 ms / 5 steps),
  so it looks device-level, not only EP8 MoE load imbalance. One GFX clock sample under (other) load: 2044-2119 MHz, 890-995 W, no clear outlier.
  Per-GPU microbench running: `/tmp/per_gpu.sh` -> `/shared_nfs/kk/results/proxy_12l/per_gpu_bench_h13-17_1028/`.
- **Per-GPU microbench (05:00-05:20 UTC, `per_gpu_bench_h13-17_1028/`, one GPU at a time, sibling idle, 2 rounds):**
  `/workspace/proxy_work/bench/per_gpu_bench.py` + MoE op test (EP8 rank, a8w4, T=16).
  HBM copy (r+w): GPU 0/1 2.92-2.96 TB/s, GPU 2-7 2.57-2.62 TB/s (device 1 ~13% FASTER). a16w16 M16 N16384 K2048: 85.7-87.7 us on all.
  MoE T=16: 513-556 us, round-to-round spread per GPU up to 6%, no GPU consistently slow. bf16 8192^3 torch.matmul 30-33 TFLOP/s on all
  (far below WMMA peak 1265 -> torch's bf16 GEMM path on gfx1250 looks wrong / untuned; not investigated).
  Verdict: devices 1 and 4 (ranks 0,1,6,7) are NOT slower in isolation; their in-server slowness is load-dependent (all 8 partitions
  busy: shared device power / clocks?) or workload-dependent. Next: sample clocks / power per GPU during a real decode run.
- Gotcha (again): `rg` is an interactive-shell function only; scripts run by bash must use grep.

### bf16 GEMM layout check + FlyDSL WMMA a16w16 (2026-10-28 ~05:40 UTC, h13-17, GPU 0; `/workspace/proxy_work/bench/flydsl_a16w16_bench.py`, `/tmp/flydsl_a16w16.log`)

- The 30-33 TFLOP/s of the per-GPU bench is layout-specific: torch bf16 8192^3 NT (`a @ b.t()`, `F.linear`) 720-725 TFLOP/s (57% of
  WMMA peak); NN (`a @ b`) 33 and TN 26 TFLOP/s (20x slower). Model linears are NT (`F.linear`, W [N,K]) -> not on the slow path.
- aiter FlyDSL `gemm_a16w16_gfx1250` (WMMA, TDM; auto-detects W [N,K] or [K,N]) needs `flydsl.runtime.device.check_smem_capacity`,
  absent in the installed FlyDSL 0.3.4.1 -> ImportError for every tile. With a no-op shim in the bench only (LDS guard), best of 15 tiles:
  8192^3 NT 515 TFLOP/s (0.81x hipBLASLt), NN 472 (14.4x torch NN), TP8 qkv M8192 N2304 K16384 0.96x, o M8192 N16384 K2048 0.89x.
  -> no gain for the model's prefill projections today; it only fixes NN, which the model does not use. Not wired in.
- sglang `82162c0b22` (user pointer, "fp32 GEMM optimization") = FlyDSL fp32 router GEMV (`laguna_router_gemv_gfx1250.py`,
  `SGLANG_OPT_LAGUNA_ROUTER_FLYDSL`), already ON in all-on (decode trace: `laguna_router_gemv2_f32_m2...`, `laguna_router_gemv_wmma_f32_m16_k64`).

### 8k prefill kernel breakdown via rocprofv3 kernel-trace (2026-10-28 ~05:35 UTC, h13-17; `rocprof_prefill_allon_ep8_h13-17_1028b/`)

- Works (no hang, unlike torch profiler EXTEND): `/workspace/proxy_work/rocprof_prefill_1028.sh` (server under `rocprofv3 --kernel-trace`,
  4 sequential 8192-token requests with max_new_tokens=1 after a 20 s idle gap) + `/workspace/proxy_work/rp_prefill_analyze.py`.
  Gotcha: SIGINT the `sglang::scheduler` processes first; killing the launcher SIGKILLs them before rocprofv3 flushes (first try: 0 files).
- All-on EP8, DDA on. Client latency 224-230 ms/request; GPU kernel time 225-239 ms per 8k prefill across ranks. Rank 0:
  MoE 93.4 ms (41%) = gate-up a8w4 45.5 + down 26.7 + fused quant 10.7 + gather-reduce 10.2;
  AR 63.1 ms (27%) = RCCL `ncclDevKernel_Generic_32` 25 x ~2.5 ms (256 MB each; microbench 1.15 ms -> ~half is wait; rank min/max 49.6/64.9 ms);
  GEMM 38.3 ms = hipBLASLt attn qkv/o 24.8 (24 calls) + gluon a16w16 BLOCK_M128 13.4 (12 calls = bf16x3 router, ~1.1 ms/layer);
  attention fmha prefill 7.4; norm 8.4; quant 8.8 (shared-expert fp4 GEMM 5.7 + mxfp4 quant 2.9); router triton 1.3.

### Prefill MoE tile shape rule (2026-10-28 05:00-05:50 UTC, h13-17)

- Op test token sweep (GPU 0, EP8 rank E=64 top-2 inter 4096, 2 rounds, `ep8_tile_tokens_h13-17_1028/`), tiles vs kernel defaults:
  fp8 activations (a8w4 format): T 256-1024 +25..28%, 2048 +0.3%, 4096 -11.3%, 8192 -10.9%.
  **fp4 activations (a4w4 format = what the server runs: kernel `a8w4_tdm_fp4_*`, `q_dtype_a=float4_e2m1fn_x2`)**:
  1024 +11.6%, 2048 -8.6%, 4096 -10.8%, 8192 -19.6% (4.82 -> 3.88 ms). All pass=True. NOTE: f19-14's sweep and A/B used a8w4 (fp8 act).
- **Code (aiter, local, uncommitted):** `grouped_moe_gfx1250.py`, before the env overrides: if no CSV row, model_dim 16384,
  inter_dim 4096, token_num >= 2048 and `AITER_TDM_PREFILL_TILE_RULE` (default 1) -> stage1 128/256/128/4, stage2 128/256/256/2.
  Env `AITER_TDM_TILE_*` still wins. EP1 (inter 512) and decode untouched.
- **ISA check (rocprofv3 kernel trace, `ep8_tile_isa_h13-17_1028/`): scratch 0 for both new kernels.** gate-up t128x256x128 b4:
  VGPR 224, LDS 145408 B (default t64 b3: 128 / 81408); down t128x256x256 b2: VGPR 232, LDS 141312 B (default 120 / 81408).
- **Server A/B, all-on EP8, 8k/1k, off/on/off2/on2 (`tile_rule_ab_ep8_h13-17_1028/`), smoke passed all, 0 faults in 4 starts:**
  mean TTFT c=2 off 329 / 329, on 289 / 290 (**-12.1%**); c=16 off 1694 / 1693, on 1483 / 1479 (**-12.5%**). ITL unchanged
  (c=16 16.13-16.60 both), c=16 tok/s 812-827 -> 832-841. Teacher-forced check of rule-on running (`tf_tp8_h13-17_1028/rule_ep8.json`).
- Still far from the bound (next section): per-partition HBM ceiling is ~8 TB/s (gfx1250 skill; 5.5 was a small-buffer number),
  EP8 gate-up T=8192 needs >= 0.54 ms (weights) / ~0.9 ms (fp8xfp4 WMMA at idle clocks), measured ~3.8 ms default, ~3 ms est. with tiles.
- Teacher-forced, rule on (EP8, current all-on incl. DDA on): vs old ref mean 0.247 / p99 0.834 (all-on rule off was 0.246 / 0.811;
  EP-only baseline 0.196 / 0.646). vs all-on rule off (dumped before DDA on): the 8000-token prompt (one 8032-token prefill, rule active)
  is BIT-IDENTICAL (0.0000); shorter prompts differ (0.19 mean) -> attributed to the DDA AR change (24-100 MB ARs moved from non-DDA to
  DDA), not the tiles. Reading: the tile change only regroups M/N (and K in 128/256 chunks in order), so the accumulation order is unchanged.

### EP1 8k prefill breakdown (rocprofv3, 2026-10-28 ~06:05 UTC, `rocprof_prefill_allon_ep1_h13-17_1028/`)

Client 251-255 ms/request; GPU kernel 247-268 ms/prefill across ranks. Rank 0 (per 8k prefill, 12 layers):
MoE 119.3 ms (47%) = gate-up `..._K16384_e512` 47.6 + **down `..._K512_e512` 37.2** + fused quant (route-expanded, 16 routes) 21.5 + gather-reduce 12.8;
AR 58.0 (23%, rank min/max 43.5/69.5); GEMM 42.2 (hipBLASLt attn 29.0 + bf16x3 router 13.0); attn 6.8; norm 9.2; quant 10.3.
vs EP8 (same total FLOPs): gate-up 47.6 vs 45.5, down 37.2 vs 26.7, quant 21.5 vs 10.7, gather 12.8 vs 10.2.
Bounds per layer at idle clocks (8 TB/s, 4896 TFLOP/s): down weights 2.1 GB -> 0.27 ms, 2.2 TFLOP -> 0.45 ms; measured 3.1 ms/layer
(~7x) -> EP1 down (K=512, small-K) is the furthest from its bound of all MoE kernels.

### Prefill MoE GEMM diagnosis: serving clocks, probes, counters, clusters, ATT (2026-10-28 06:00-06:40 UTC, h13-17, GPU 0 op tests)

- **Serving clocks under sustained 8k prefill** (`/workspace/proxy_work/prefill_clock_1028.sh`, EP8 all-on, 8 concurrent 8k max_new_tokens=1,
  41.5k prefill tok/s, `prefill_clock_ep8_h13-17_1028/`): GFX 1022-1090 MHz, socket power 1494-1542 W, gfx_act 100%, hotspot 65-67 C
  (PPT-limited, not thermal). Compute bound at serving ~4896 x 1.06/2.2 ~ 2350 TFLOP/s (fp8xfp4; fp4xfp4 peak not measured).
  Per layer at serving: EP8 gate-up >= 1.87 ms (meas. 3.79 default), EP8 down >= 0.94 (2.23), EP1 gate-up >= 1.87 (3.97), EP1 down >= 0.94 (3.10).
- **Weights-only probes** (re-added `AITER_TDM_PROBE` in `mxfp4_preshuffle_gfx1250_tdm.py`, LOCAL TEMP edit, original saved at
  `/workspace/proxy_work/mxfp4_preshuffle_gfx1250_tdm.py.orig_1028`; 1 = no WMMA, 2 = no WMMA + no epilogue; probe runs fail accuracy as
  expected). T=8192 fp4 act, `--scenario kernel`, 2 rounds, us full / no-WMMA / TDM-only (`moe_probe_prefill_h13-17_1028/`):
  EP8 gate-up 2373 / 1726 / 1676; EP8 down 1155 / 874 / 826; EP1 gate-up 2942 / 2809 / 2783; EP1 down 2362 / 2235 / 1406 (epilogue 830).
- **Counters** (`moe_counters_prefill_h13-17_1028/`, FETCH_SIZE / GL2C_HIT+MISS / GL2C_EA_RDREQ in 3 passes): HBM 1.1-3.2 TB/s
  (13-40% of 8 TB/s), FETCH only ~1.3x the weight bytes, L2 hit 65-81%. So NOT HBM-bound: the redundant A/B re-reads hit L2.
  L2 requests (hit+miss) 8.2e7 EP8 gate-up / 1.2e8 EP1 gate-up (~10-15 GB if 128 B each, assumption) -> on-chip traffic is large.
- **Cluster multicast does not help** (`/workspace/proxy_work/cluster_sweep_1028.sh`, one-row CSV, `moe_cluster_prefill_h13-17_1028/`):
  1x1 best everywhere; 1x2 / 1x4 (share A) +10..+80%, 2x1 (share B) +1..+75%, 2x2 worse. All pass.
- **ATT** (`rocprofv3 --att --att-target-cu 1 --att-shader-engine-mask 0x1`, 1 dispatch, no hang; analysis `/tmp/ana/att_hot_loop.py` =
  att_loop_stalls.py adapted to v_wmma + pick loop by executed count; `moe_att_prefill_h13-17_1028/`):
  - EP8 gate-up hot loop (83 instrs, 16 `v_wmma_scale_f32_32x16x128_f4`, 27 ds_load per iter): share of stall+dur: **s_wait_dscnt 40.6%**,
    WMMA 33.0%, barrier 12.0%, ds_load 7.2%, s_wait_tensorcnt 4.5%. Top stalls: `s_wait_dscnt 0x0` 12.2%, `s_wait_dscnt 0x13` 8.0%,
    `s_barrier_wait` 5.9%. -> LDS -> VGPR feed of the WMMAs is the limiter, NOT the TDM global loads. w1x4: every wave reads the whole A tile.
  - EP1 down (K=512, fully unrolled, 1712 short waves of ~6.6k cyc): **s_wait_tensorcnt 35.3%** (12.8% at the final `s_wait_tensorcnt 0x0` =
    waiting for the output store), **barrier 28.7%**, s_wait_dscnt 16.5%, WMMA 4.9%. -> per-tile fixed cost (prologue load wait, barriers,
    epilogue store drain) dominates; compute is ~5%.
- **Existing-knob sweeps** (`/workspace/proxy_work/knob_sweep_1028.sh` + `knob_cases*_1028.txt`, `moe_knobs_prefill_h13-17_1028/`, us, all pass):
  gate-up knobs (warp 2x2, INTERLEAVED_LDS_LOAD, PLANAR_LDS, MMA_GROUP 4/16, DS_FIRST_N 4): all within +-2% of base (noise ~5%),
  PLANAR_LDS +11% / +64% on EP1. EP1 down: persistent_workers2 96/192/384 -> 6396 / 3814 / 2652 (base 2331), tile_k2 512 2520,
  tile_m2 128 2775 -> nothing helps. Bigger per-wave tile (128x128 per wave via w1x2 or 128x512 w1x4): EP8 ~-2% (noise).
  EP1 stage1 tile_m 128: gate-up -19..-23% (2947 -> 2260 with 128x512 w1x4) but down +15% (2336 -> 2690-2719) because both
  GEMMs share one contiguous layout aligned to max(tile_m, tile_m2) (`_align_m`, grouped_moe_gfx1250.py ~678) -> net only -4..-6%.
  -> config alone cannot fix the LDS-feed limit; candidate kernel change #1: per-GEMM row alignment (keep EP1 gate-up at 128 without
  padding down to 128).

### Kernel change #1: skip alignment-gap tiles + EP1 prefill tile rule (2026-10-28 ~07:00-07:40 UTC, h13-17; aiter LOCAL, uncommitted)

- Finding: psum (`moe_contiguous_psum.py`) stores each expert's ACTUAL end, starts are aligned to `_align_m = max(tile_m, tile_m2)`.
  When a GEMM's tile_m < _align_m, tiles lying wholly in the gap [prev expert actual end, this expert aligned start) bisect to the
  next expert and compute padding rows (EP1 gate-up 128 / down 64: ~half the experts get one wasted 64-row down tile).
- Change: `launch_gemm_a8w4_tdm(..., layout_align_m=0)` (constexpr, in cache_tag, kernel name suffix `_la<N>` when > tile_m);
  in `gemm_task`, after the expert bisect, `has_work &= blk_m >= align_up(psum[expert-1], layout_align_m)` (only when
  layout_align_m > tile_m and no cluster, so cluster rendezvous counts are unchanged). Plumbed through
  `grouped_gemm_mxfp4.flydsl_grouped_gemm_a8w4_masked(layout_align_m=)` and all 6 launches in `_grouped_a8w4_tdm_moe`
  (`_layout_align = 0 if _compact else _align_m`). No effect when tile_m == _align_m (EP8 rule tiles, decode).
- Op test (us, 2 rounds, all pass): EP1 gate-up 128x256 / down 64: down 2690-2719 -> 2400; GEMMs total 5286 -> 4731 (-10.5%).
  EP8 base and EP1 base unchanged.
- EP1 whole-MoE token sweep (fp4, `ep1_tile_tokens_h13-17_1028/`): new vs default T=1024 +16%, 2048 +0.8%, 4096 -5.0%, 8192 -7.9%.
- Rule extended (same `AITER_TDM_PREFILL_TILE_RULE`): model_dim 16384, inter_dim 512, token_num >= 4096, no CSV row ->
  stage1 128/256/128/3, stage2 64/256/128/3. Server A/B EP1 running: `/tmp/tile_rule_ab_ep1.sh` -> `tile_rule_ab_ep1_h13-17_1028/`.
- **EP1 server A/B (06:47-07:07 UTC, all-on EP1, 8k/1k, off/on/off2/on2, `tile_rule_ab_ep1_h13-17_1028/`): smoke passed, 0 faults in 4 starts.**
  mean TTFT c=2 362 / 362 -> 342 / 342 (**-5.6%**); c=16 1874 / 1876 -> 1783 / 1780 (**-5.0%**); ITL unchanged (15.3-15.9).

### Main-loop LDS pipeline investigation (2026-10-28 ~08:00 UTC, h13-17)

- ISA of the EP8 gate-up hot loop (tile 128x256x128 b4, w1x4; from ATT `code.json`, idx 663-745): per iteration = one k128 step:
  `s_wait_tensorcnt 0x4` + `s_barrier_wait` (20% of stall) -> 2 `tensor_load_to_lds` (next stage) -> 27 `ds_load` (A 2 frags = 8 b128,
  B 8 frags = 16 b128, 3 scale 2addr_b32) for THIS step -> `s_wait_dscnt 0x13` (19.8%) -> 1 WMMA -> `s_wait_dscnt 0x0` (30.4%) -> 15 WMMA.
  The 2 rmem slots only prefetch within a k-tile, so with tile_k=128 nothing overlaps.
- tile_k 256/512 (2-4 k-steps per tile, prefetch active), 2 rounds: EP8 gate-up 2206-2402 vs base 2327-2359; EP1 2279-2349 vs
  2288-2459 -> no gain. ATT of EP8 k256 b3 (`moe_att_prefill_h13-17_1028/ep8_g1_k256/`): loads now interleave with WMMAs
  (waits spread over 5 sites) but s_wait_dscnt share is unchanged (42.1% vs 40.6%), WMMA 31.8% -> NOT exposed latency.
- Working hypothesis (unverified): LDS throughput: per WG per k128 ~48 KB ds_load reads + ~24 KB TDM writes vs 64 WMMAs; at an
  assumed ~128 B/clk that is ~550 clk of LDS vs ~256 clk of WMMA. Next: LDS bandwidth microbench on gfx1250 (ds_load_b128 alone and
  with concurrent TDM writes) + exact byte accounting per config; if confirmed, redesign to cut LDS bytes (A direct global->VGPR,
  since waves do not share A rows; more WMMA reuse per B fragment).
- **LDS read bandwidth microbench** (gluon, `/workspace/proxy_work/bench/lds_bw_gfx1250.py`, GPU 0, int32 b128 loads from a 4-way
  buffered [64,128] tile, xor-accumulated; B/clk assumes 2.2 GHz, clock not measured): progs/CU x warps: 1x4 135, 2x4 161, 4x4 170,
  **2x8 220 B/clk/CU**; 2x4 with 32 B/thread 117. (32x128 8-warp case read 1700 B/clk = compiler-folded loads, discarded.)
- vs EP8 gate-up (ATT): 128 iterations (K=16384/128), mean loop window ~74.4k cyc/wave -> ~581 cyc per k128 iteration; per WG per
  iteration ~48 KB ds_load + ~24 KB TDM writes; at 2 WG/CU (LDS 112 KB each, assumed) ~250 B/clk/CU of LDS traffic >= the measured
  read ceiling -> LDS-throughput-bound (consistent with s_wait_dscnt ~41% regardless of tile_k). WMMA issue ~18 cyc/WMMA/SIMD in that
  window -> ~2x compute headroom if LDS traffic halves. Open contradiction: the 128x128-per-wave config (w1x2, -33% LDS bytes/WMMA)
  did not speed up -> ATT it next (2 waves/WG lowers LDS BW per the microbench; VGPR/occupancy).
- **ATT time unit = shader cycles** (`realtime.json`: 280 ATT units per 12 ticks of the 100 MHz realtime counter -> 2.33 GHz during the
  single traced dispatch). Bigger per-wave tile (w1x2) ATT: WMMA 59.7% / dscnt 26.8%, but **1 wave per SIMD** (w1x4: 2) -> fewer LDS bytes
  but no latency hiding; 256x256 w2x2/w2x4 gate-up (stage2 kept 128): +6..+22% (confounded by 256-row expert padding at ~256 rows/expert).
- **LDS ceiling re-measured with clock sampling**: 2.36 GHz during the run; 2 progs x 4 warps 153, 2 x 8 217, 4 x 8 202 B/clk/CU.
  Kernel needs ~250 B/clk/CU (2 WG/CU x 72 KB / 581 cyc) -> LDS-throughput-bound CONFIRMED in cycles.
- **CORRECTION: op tests are power-limited too.** GPU 0 alone during the EP8 MoE op test: GFX 1.14-1.4 GHz in the GEMM phases (2.1 GHz in
  setup), socket power up to 1.46 kW (single partition reaches PPT). So op-test us are NOT "idle / restored-clock" numbers; the
  EP8 gate-up 2.3 ms ~ 0.96 ms of cycles at 2.33 GHz. Earlier "idle-clock" bound estimates in this doc are invalid; compare in cycles.
- **Register-blocking arithmetic:** per wave 128 token rows x 64 weight cols (WMMA A operand = weights, 32 cols; B = tokens, 16 rows),
  16 WMMA, 256 acc VGPRs. LDS bytes/WMMA = 64 B x (r + c) / ((r/16)(c/32)); at a fixed 256-VGPR accumulator r=128, c=64 is already the
  minimum (768 B/WMMA); larger accumulators drop to 1 wave/SIMD. -> **Redesign: weights direct global -> VGPR** (waves own disjoint weight
  columns, so LDS sharing buys nothing), tokens stay TDM -> LDS: per WG per k128 72 KB -> 40 KB LDS traffic (-44%), ~+64 VGPR for a
  weight double buffer. Next: read the weight TDM/LDS path and the preshuffle layout to confirm per-lane contiguous 16 B loads.
