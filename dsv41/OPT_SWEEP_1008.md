# DSV4.1-Flash AgentX sweep: CI recipe 37423021942 + TP4_GAP items 0-6 (2026-10-08)

Owner node: crsuse2-m2m-255

## CONTINUE HERE

**Status (crsuse2-m2m-255, 2026-10-09 05:25): DONE.** Smoke GSM8K TP2 0.897 / TP4 0.902 (QR INT8). All 12 points rc=0
(TP4 c1 first try invalid, rerun opt1008_tp4_c1_r2). vs CI 37423021942 (table at the end): TPOT -3..-10% and P90 +4..+17% at
every point; TTT -1..+9% (mostly prefill, small moves); TTFT p50 -14..-37% at c<=4 (both TP) and c64, ~0 at c8/c16 TP4, c32 TP2.
vs B200 vLLM TP4: c1 -16.5/-25.5%, c8 -7.1/-18.5%, c16 -2.7/-24.2% (TTT/P90; TP4_GAP had c8 -7.7/-24.6, c16 -4.4/-31.8).
Combined effect only (mainline + items 0-6 + OPUS + QR INT8); no per-item attribution.
**Next:** per-item attribution if wanted (one flag off at a time at TP4 c8/c16); fix the smoke env check (read the launch_server
parent, not sglang::scheduler); understand the TP4 c1 server_watch stop (server_watch_probe.py logs the failing pid).
**Files:** `scripts/sweep_ci37423_opt.sh` (driver; CONCS/TSUF knobs), `after_tp4_rerun.sh`, `opt1008_cmp_watch.sh`,
`server_watch_probe.py`, `agentx_agg_table.py --ref/--row`, `scripts/tp4_gap_lane.sh` (now TP / OPUS knobs),
`scripts/agentx_colleague_mi355x_sglang.sh` (new CUDA_GRAPH_MAX_BS knob, default 64).
**Repro:**
```bash
S=/workspace/claude-skills/dsv41/scripts
MODE=smoke LANE=tp4 bash $S/sweep_ci37423_opt.sh; MODE=smoke LANE=tp2 bash $S/sweep_ci37423_opt.sh
LANE=tp2 bash $S/sweep_ci37423_opt.sh; LANE=tp4 bash $S/sweep_ci37423_opt.sh   # sequential only (GPUs 0,1 / 4-7)
```
**Pass criteria:** every point rc=0, 0 errors; compare TTT / P90 against CI run 37423021942 per point.

## Recipe (InferenceX PR #3696 @ 47adb59a, run 37423021942)

configs/amd-master.yaml `dsv41flash-fp4-mi355x-sglang-agentic-dspark`: TP2 c1-c64, TP4 c1-c16, ep1, no DP, DSpark
block 5, `--enforce-shared-experts-fusion --fp8-gemm-backend aiter`, prefill graph disabled + decoder SWA bounded replay,
per point: pdi 16 (c<32) / 4, chunk 16384 / 4096 at c64, mem 0.70 (TP2 c16/c32 0.80, c64 0.85), swa-prefix-tails
64*conc in [128, 4096], max-running 2*conc, decode graph bs 64 (c64: 128). CI: no OPUS sparse prefill, QR NONE.
**Deviation from CI (user, 2026-10-08):** SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1 and ROCM_QUICK_REDUCE_QUANTIZATION=INT8
(QR_QUANT=INT8) in both smoke and sweep. Note: qr_int8_gsm8k (mi355-4, DP2) gave 0.887, below the 0.895 gate.
Copy of the recipe files: /tmp/ix37423/ (node-local).

## Code state (crsuse2-m2m-255)

- sglang /sgl-workspace/sglang branch dsv41-opt-1008 = origin/main 45abea0269 + cherry-picks of HaiShaw perf/v41-*:
  i0 index-q-prefill-fuse, i1 indexer-bf16-logits, (i2 block-max: already contained in i1, skipped), i3
  mhc-ar-boundary-stats, i4 small-moe-sort (environ.py conflict: kept both vars), i5 flydsl-mxfp8-gemm, i6
  woa-fuse-mxfp8-quant. Image pyproject edits re-applied (uncommitted).
- aiter /sgl-workspace/aiter branch dsv41-opt-1008 = origin/main 6264f8f5c (has #5896, #5967) + kkHuang-amd
  perf/v41-indexer-bf16-logits (5422ea8ef). Re-applied image steps (uncommitted): torch.Stream fix, #6042
  pa_decode_sparse (conflict resolved: comment only, row_tiles already defined), mla_v4 .co from sglang. lru_cache fix is
  upstream now. Old JIT .so + jit/build -> /sgl-workspace/aiter_jit_backup_e7d2453_1008.
- Backups (HEADs + local diffs): /shared_nfs/kk/results/DeepSeek-V4.1-Flash/repo_backup_1008/. Old stashes: stash@{0}.
- Full AOT build: `PREBUILD_KERNELS=1 GPU_ARCHS=gfx950 python setup.py build_ext --inplace` (incl. FlyDSL run_aot) +
  `pip install -e .`, log repo_backup_1008/aiter_aot_build.log.

## Flags on (sweep_ci37423_opt.sh)

| item | flag |
|---|---|
| 0 index-Q fuse | none (always on) |
| 1 bf16 logits | SGLANG_DSV41_PREFILL_LOGITS_BF16=1 (NIAH accuracy A/B still not done) |
| 2 block-max | none |
| 3 mHC AR+stats | SGLANG_ROCM_MHC_ALL_REDUCE_STATS=1 (TP4 decode path) |
| 4 small sort | SGLANG_AITER_SMALL_MOE_SORT_MAX_PAIRS=64 (default) |
| 5 FlyDSL MXFP8 | SGLANG_ROCM_MXFP8_AITER_PRESHUFFLE=1 |
| 6 wo_a fuse | SGLANG_HIP_WO_A_MXFP8=1 (default) |

## Log (append-only)

- 2026-10-08 11:33-11:48 (crsuse2-m2m-255): first smoke attempt void. Both smokes ran install_agentic_deps at once on the
  shared /workspace/agentx-runtime/venv; tp4 failed (uv venv exists), tp2 killed. Root cause: benchmark_lib.sh:3080
  re-derives AIPERF_VENV from AIPERF_RUNTIME_DIR, so the run shim's per-port AIPERF_VENV never took effect. Fix:
  agentx_colleague_run.sh exports AIPERF_RUNTIME_DIR=/workspace/agentx-runtime/p$PORT (uv + cache seeded by hardlink).
- 2026-10-08 13:05 (crsuse2-m2m-255): sweep TP2 c1 ready 12:45, profiling OK. Verified on the launch_server parent (pid 299334):
  QR INT8, OPUS 1, SIMULATE_ACC_LEN 3.51 (benchmark mode, as CI). sglang::scheduler children do not carry these vars in
  /proc/<pid>/environ -> the smoke env check in sweep_ci37423_opt.sh must read the launch_server parent (fix later).

- 2026-10-08 14:2x (crsuse2-m2m-255): vs-CI rows are appended by scripts/opt1008_cmp_watch.sh (polls lane files every
  30 s; a tail -F Monitor missed the 13:49/13:50 lane lines on NFS). tp2_c1 row added by hand (finished before the watcher).

- 2026-10-08 22:54 (crsuse2-m2m-255): opt1008_tp4_c1 rc=1 after ~13 min profiling. InferenceX
  infx/bench_serving/server_watch.py (checks the server + 4 sglang::scheduler pids every 2 s, start-time match, not Z/X)
  reported "required worker exited" and stopped the client; the launcher then SIGTERMed a server that was still
  decoding (server.log normal to 22:54:15, no Traceback; dmesg clean, no OOM). Which pid failed the check is not logged.
  Point invalid; rerun opt1008_tp4_c1_r2 queued after c16 (after_tp4_rerun.sh pid 354343; sweep script got CONCS/TSUF knobs).

## Results vs CI run 37423021942 (append-only; each cell = this / CI (delta); written by scripts/opt1008_cmp_watch.sh)

CI ref: /shared_nfs/kk/results/DeepSeek-V4.1-Flash/ref_ci_37423021942/agg_bmk.json (image 20261006, no OPUS, QR NONE, no items 0-6).

| node / date / cfg | tp | conc | TTT/gpu | P90 intvty | TPOT p50 ms | TTFT p50 s | ok/total |
|---|---|---|---|---|---|---|---|
| m2m-255 10-08 opt1008_tp2_c1 rc=0 | 2 | 1 | 11,690.4 / 11,288.0 (+3.6%) | 365.8 / 345.0 (+6.0%) | 2.410 / 2.500 (-3.6%) | 0.80 / 0.84 (-5.5%) | 289/300 |
| m2m-255 10-08 opt1008_tp2_c2 rc=0 | 2 | 2 | 12,021.8 / 12,160.8 (-1.1%) | 347.5 / 330.6 (+5.1%) | 2.400 / 2.500 (-4.0%) | 0.40 / 0.57 (-29.0%) | 445/467 |
| m2m-255 10-08 opt1008_tp2_c4 rc=0 | 2 | 4 | 17,723.2 / 16,575.1 (+6.9%) | 309.5 / 297.9 (+3.9%) | 2.460 / 2.540 (-3.1%) | 0.34 / 0.48 (-27.7%) | 665/709 |
| m2m-255 10-08 opt1008_tp2_c8 rc=0 | 2 | 8 | 29,932.5 / 29,765.3 (+0.6%) | 257.4 / 239.3 (+7.5%) | 2.940 / 3.180 (-7.5%) | 0.35 / 0.41 (-14.3%) | 1380/1467 |
| m2m-255 10-08 opt1008_tp2_c16 rc=0 | 2 | 16 | 55,251.5 / 54,414.2 (+1.5%) | 175.6 / 157.4 (+11.5%) | 3.620 / 4.000 (-9.5%) | 0.39 / 0.47 (-15.7%) | 2574/2751 |
| m2m-255 10-08 opt1008_tp2_c32 rc=0 | 2 | 32 | 109,162.2 / 104,531.7 (+4.4%) | 101.6 / 87.2 (+16.5%) | 5.490 / 6.020 (-8.8%) | 0.60 / 0.59 (+1.1%) | 4519/4873 |
| m2m-255 10-08 opt1008_tp2_c64 rc=0 | 2 | 64 | 145,956.7 / 137,862.5 (+5.9%) | 54.0 / 49.9 (+8.3%) | 14.360 / 15.230 (-5.7%) | 1.43 / 1.75 (-18.2%) | 8141/8848 |
| m2m-255 10-08 opt1008_tp4_c1 rc=1 | 4 | 1 | 6,333.2 / 6,019.4 (+5.2%) | 456.4 / 406.5 (+12.3%) | 2.090 / 2.230 (-6.3%) | 0.39 / 0.88 (-55.1%) | 108/119 |
| m2m-255 10-08 opt1008_tp4_c1 INVALID | 4 | 1 | row above: client stopped at 22:54 after ~13 min profiling (108 req) by InferenceX server_watch ('required worker exited'); server log, dmesg clean; rerun as opt1008_tp4_c1_r2 after c16 | | | | |
| m2m-255 10-09 opt1008_tp4_c2 rc=0 | 4 | 2 | 6,748.6 / 6,169.7 (+9.4%) | 396.9 / 350.6 (+13.2%) | 2.130 / 2.240 (-4.9%) | 0.37 / 0.59 (-36.8%) | 478/500 |
| m2m-255 10-09 opt1008_tp4_c4 rc=0 | 4 | 4 | 9,025.2 / 8,807.1 (+2.5%) | 386.4 / 331.6 (+16.5%) | 2.130 / 2.300 (-7.4%) | 0.36 / 0.50 (-28.3%) | 680/724 |
| m2m-255 10-09 opt1008_tp4_c8 rc=0 | 4 | 8 | 15,366.9 / 15,361.7 (+0.0%) | 313.6 / 282.2 (+11.1%) | 2.510 / 2.720 (-7.7%) | 0.30 / 0.30 (-0.2%) | 1394/1481 |
| m2m-255 10-09 opt1008_tp4_c16 rc=0 | 4 | 16 | 28,619.3 / 28,106.5 (+1.8%) | 240.8 / 214.6 (+12.2%) | 2.940 / 3.270 (-10.1%) | 0.40 / 0.41 (-2.3%) | 2654/2831 |
| m2m-255 10-09 opt1008_tp4_c1_r2 rc=0 | 4 | 1 | 6,428.8 / 6,019.4 (+6.8%) | 434.7 / 406.5 (+6.9%) | 2.110 / 2.230 (-5.4%) | 0.66 / 0.88 (-24.1%) | 303/314 |
