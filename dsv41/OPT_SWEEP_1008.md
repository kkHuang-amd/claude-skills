# DSV4.1-Flash AgentX sweep: CI recipe 37423021942 + TP4_GAP items 0-6 (2026-10-08)

Owner node: crsuse2-m2m-255

## CONTINUE HERE

**Status (crsuse2-m2m-255, 2026-10-08 11:31):** chain_opt1008.sh running (setsid, pid 286414, restarted 12:04 after the aiperf venv race fix): waits for the aiter AOT build ->
`import aiter` -> TP2 + TP4 smokes in parallel (EVAL_ONLY=true, scheduler env checked for no SGLANG_SIMULATE_ACC_LEN
and all opt flags, GSM8K 1319 5-shot) with QR INT8; if either < 0.895 redo with QR NONE (user); first passing QR -> sweep, TP2 then TP4 serialized on /tmp/opt1008_sweep.lock (user: never TP2+TP4 at once, one lane). Progress agentx/chain_opt1008.txt.
**Next:** read chain_opt1008.txt; when lanes finish, table with scripts/agentx_agg_table.py vs CI (/tmp/ix37423/ci_table.md).
**Files:** `scripts/sweep_ci37423_opt.sh` (driver), `scripts/tp4_gap_lane.sh` (now TP / OPUS knobs),
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
