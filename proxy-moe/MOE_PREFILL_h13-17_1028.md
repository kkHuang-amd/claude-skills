Owner node: ctheliosmc-rck-g02-h13-17.rck.dcgpu

# Prefill MoE tile optimization, continued on h13-17 (2026-10-28)

Takes over from `MOE_PREFILL_f19-14_1028.md` (owner f19-14, closed: GPU 2/3 SMU not responding). Carried-over result:
grouped a8w4 MoE, T=8192, one EP2 rank of the TP2 4L model (E=256, topk 8, H 16384, I 4096): stage1 `AITER_TDM_TILE_M/N/K/NUM_BUFFERS=128/256/128/4`
+ stage2 `..._M2/N2/K2/NUM_BUFFERS2=128/256/256/2` = 14.40 ms vs defaults 16.46 ms (-12.6%), f19-14 only, idle clocks.

## CONTINUE HERE

**NOW (user, 2026-10-28 ~01:30 UTC): tile work PAUSED. Priority = get every f19-14 10/27 optimization working on TP8.**
Ported config: `/workspace/proxy_work/allon_tp8.env` (+ `--attention-backend aiter --page-size 64`), TP8 attention-projection CSV
`/workspace/proxy_work/bf16_prefill_torch_tp8.csv` (14 rows, qkv 2304x16384 + o 16384x2048, padded M 128..8192 -> torch; microbench
`bench/attn_proj_tp8_bench.py` on GPU 0: hipBLASLt 1.0-2.9x faster than gluon at every M 64..8192, `/tmp/attn_proj_tp8.log`).
RUNNING: `/workspace/proxy_work/allon_tp8_1028.sh` -> `/shared_nfs/kk/results/proxy_12l/allon_tp8_h13-17_1028/` (EP1, EP8; smoke, 8k/1k c=2/16).
Still blocked from f19-14: `SGLANG_OPT_USE_AITER_SILU_MUL` (module_activation build error). Not ported: a16w16 M=32-64 tuning (open item there too).

**Earlier status (2026-10-28, h13-17):** env switched (below); user gave all 8 GPUs. DONE: server A/B of the f19-14 tiles on TP8 EP1 / EP8,
12L proxy, 8k/1k c=2 and c=16, arms A (best config) -> B (A + tile env) -> A2 per EP, smoke per arm.
`/workspace/proxy_work/tile_server_ab_1028.sh` -> `/shared_nfs/kk/results/proxy_12l/tile_ab_tp8_h13-17_1028/` (`driver.log`).
Caveat: the `AITER_TDM_TILE_*` env overrides EVERY grouped MoE call (decode too, and the EP1 CSV v3 rows), so B's ITL is not a
prefill-only effect; TTFT is the metric for this A/B.
Correction (user): arm A here is the OLD h13-17 config (triton attention, 10/23-26 env), NOT the current best. It lacks the f19-14
10/27 config (`TP2_4L_f19-14_1027.md` "CLEAN COMPARISON": `allon.env` + `--attention-backend aiter` + `SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d`
+ `--page-size 64` + FlyDSL pa_decode / extend + `SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1` + attention / `bf16_prefill_torch.csv`; 8k1k c16 TTFT -28.7% on TP2).
`allon.env` and the CSVs live only on f19-14 (`/shared_nfs/kk/proxy_work_f19-14_1027/`), not on this node. Redo the A/B with A = that config on TP8.
**Next:** after the A/B, the remaining f19-14 items (repro kernel A/B, ISA check, no-CSV default / shape rule, GSM8K).
**Files:** `proxy-moe/scripts/gfx1250_moe_tile_sweep.sh`; aiter `grouped_moe_gfx1250.py` (no-CSV fallback).
**Results:** `/shared_nfs/kk/results/proxy_4l/<tag>_h13-17_1028/` (plus `proxy_12l/` for TP8 server runs).

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
