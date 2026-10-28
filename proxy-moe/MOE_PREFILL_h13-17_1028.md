Owner node: ctheliosmc-rck-g02-h13-17.rck.dcgpu

# Prefill MoE tile optimization, continued on h13-17 (2026-10-28)

Takes over from `MOE_PREFILL_f19-14_1028.md` (owner f19-14, closed: GPU 2/3 SMU not responding). Carried-over result:
grouped a8w4 MoE, T=8192, one EP2 rank of the TP2 4L model (E=256, topk 8, H 16384, I 4096): stage1 `AITER_TDM_TILE_M/N/K/NUM_BUFFERS=128/256/128/4`
+ stage2 `..._M2/N2/K2/NUM_BUFFERS2=128/256/256/2` = 14.40 ms vs defaults 16.46 ms (-12.6%), f19-14 only, idle clocks.

## CONTINUE HERE

**Status (2026-10-28 04:15 UTC, h13-17):** TP8 runs every f19-14 10/27 optimization ("all-on"); 8k/1k TTFT -34..39% vs the old
h13-17 config on EP1 and EP8, reproduced, smoke + teacher-forced logprob check OK. AR default changed to CAR <= 1 MB + RCCL DDA on
(arm B; neutral end to end, faster in microbench). Prefill MoE tile work paused; next item to be picked by the user (list below).
**Best config (TP8, 12L):** `/workspace/proxy_work/allon_tp8.env` (DDA on since 04:15) + `--attention-backend aiter --page-size 64`,
EP1 adds `AITER_CONFIG_GROUPED_FMOE=/workspace/proxy_work/ep1_grouped_fmoe_gfx1250_v3.csv`. Code: aiter kk `7ea6f2f46` + 3 image edits
+ local CK `amd_tdm_store` brace fix; sglang haishaw `08db4167c7`; image RCCL 2.32.3 (has the DDA fix per user).
**Repro (one EP, 8k/1k c=2,16 + smoke):**
```bash
EPS=8 CONCS="2 16" bash /workspace/proxy_work/allon_tp8_1028.sh <run-tag>   # -> /shared_nfs/kk/results/proxy_12l/<run-tag>/
```
**Pass criteria:** smoke passed=True, bench 100% success; 8k/1k mean TTFT EP8 c=2 ~330 ms, c=16 ~1693 ms (EP1 ~363 / ~1881).

**Open items (2026-10-28):**
- Tile (paused): (1) shape rule = tiles only for inter 4096 + prefill token counts (threshold T from an op-test token sweep first),
  never EP1 / decode; (2) ISA resource check (VGPR / scratch / LDS) of the two tiles; (3) EP8 server A/B on all-on, several starts
  (fault risk), + teacher-forced check; (4) HBM bytes / L2 hit per kernel; (5) kernel-level 16.46 -> 14.40 repro (f19-14 bench not on this node).
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
