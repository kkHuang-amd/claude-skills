# RUNBOOK — DeepSeek-V4.1-Flash on MI355X with SGLang

TP4 baseline (2026-09-24): GSM8K 5-shot/1319 ≈ 0.90–0.91; PR-method bs1 147.7 tok/s (DSpark off) / 621 (on).
Best config (TP2 AgentX): last section. `D=/workspace/claude-skills/dsv41`. Old detail: doc_backup_20260929/RUNBOOK.md.

## 0. Base container

SGLang ROCm dev image `GPU_ARCH=gfx950-rocm720` (ROCm 7.2.0, Python 3.10, `/opt/venv`), torch 2.9.1+rocm7.2.0, triton 3.7.0,
editable aiter `/sgl-workspace/aiter` (pinned by `setup_env.sh`: v0.1.22.post1 `b4d9154d1` + #5561), 8× MI355X.
`launch_server.sh` overrides container env `ROCM_QUICK_REDUCE_QUANTIZATION=INT8`→`NONE` and `SGLANG_USE_ROCM700A=1`→`0`
(DP-attention only). Kept: `HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1 NCCL_MIN_NCHANNELS=112 SGLANG_SET_CPU_AFFINITY=1
AITER_USE_SYSTEM_TRITON=1 PYTHONPATH=/sgl-workspace/mori:/sgl-workspace/aiter`.

## 1. Setup (idempotent)

```bash
bash $D/scripts/setup_env.sh        # ~5 min first time (sgl-kernel build), seconds afterwards
```

1. Clones `kevin-mii/sglang` `dsv41-amd-main` to `/sgl-workspace/sglang-dsv41` (+ `upstream` remote); validated `e2e824dc58`.
2. Checks the aiter pin, applies `patches/aiter_5561_*` (race fix; #5802 conflicts and is unused at `BOUND=0`).
3. Rebuilds sgl-kernel (AOT, `/tmp` copy; needs `sort_output`) and **replaces `site-packages/sgl_kernel`** with the
   egg's copy (else shadowed); original -> `/shared_nfs/kk/dsv41/sgl_kernel_backup_orig`.
4. Checks the model: `/shared_nfs/models/deepseek-ai/DeepSeek-V4.1-Flash` (fallback `/shared_nfs/deepseek-ai/...`).

The branch runs via `PYTHONPATH`; installed editable sglang (`/sgl-workspace/sglang`, main) stays untouched.

## 2. Launch (TP4)

```bash
cd /shared_nfs/kk/dsv41
nohup bash $D/scripts/launch_server.sh > server.log 2>&1 &                                  # DSpark off, GPU0-3 :30000
DSPARK=1 GPUS=4,5,6,7 PORT=30001 nohup bash $D/scripts/launch_server.sh > server_dspark.log 2>&1 &
grep -E 'ready to roll|Traceback|Error' server.log | tail -3      # ~5-15 min
```

Perf settings (`PERF=1`) need local branch `opus-prefill`; recreate it in a fresh container:

```bash
git -C /sgl-workspace/sglang-dsv41 checkout -b opus-prefill e2e824dc58 && git -C /sgl-workspace/sglang-dsv41 am -k \
  $D/patches/sglang_local_opus_prefill_*.patch $D/patches/sglang_local_kvstore_int64_0001.patch  # int64 = #41159, needed for long context
PERF=1 nohup bash $D/scripts/launch_server.sh > server.log 2>&1 &
```

`PERF=1` = OPUS sparse prefill (`SGLANG_OPT_DSV41_OPUS_PREFILL=1`, TTFT -10%) + `QR=INT8` (TTFT -7%, not bit-deterministic)
+ `--enable-mixed-chunk` (DSpark off only; TTFT 2273 -> 1520 ms, TPOT +6%). c32 DSpark off: 2341 out tok/s, TTFT 1520 ms.
Bit-reproducible: no `PERF=1` or `PERF=1 QR=NONE`. Never bench with `SGLANG_DEBUG_DSV41_OPUS_PREFILL_CHECK=1` (syncs).

## 3. Validate

```bash
TAG=check PORT=30000 bash $D/scripts/run_gsm8k.sh          # expect Accuracy >= 0.89 (±1pt run-to-run)
TAG=check PORT=30000 bash $D/scripts/run_pr_style_c1.sh    # expect ~148 (off) / ~620 on :30001 (DSpark)
TAG=check PORT=30000 bash $D/scripts/run_throughput.sh     # full sweep -> results/perf.md
# or after launch: TAG=x PORT=.. SERVER_LOG=.. SERVER_PID=.. bash $D/scripts/pipeline_eval.sh
```

Compare with the PR only via `run_pr_style_c1.sh` (no TTFT, 6-run median; sweep reads ~5–20% lower).

## 4. Stop / clean up

```bash
ps -eo pid,comm,args | awk '$2=="python3" && /sglang.launch_server/{print $1}' | xargs -r kill
ps -eo pid,comm | awk '$2 ~ /^sglang::/{print $1}' | xargs -r kill     # renamed scheduler/detokenizer children
rocm-smi --showmeminfo vram | grep Used        # wait until ~0 (GPU[7] = HIP device 5 here)
```

Never `pkill -f`/`pgrep -f` a pattern that appears in your own command line.

## 5. Revert

sgl-kernel: copy `/shared_nfs/kk/dsv41/sgl_kernel_backup_orig/sgl_kernel` back over `/opt/venv/lib/python3.10/site-packages/sgl_kernel`.
Pre-upgrade aiter (acf8fdf93): stash@{0} + aiter_pre_upgrade_full.diff; JIT: /sgl-workspace/aiter_jit_backup_acf8fdf93.
Older local aiter edits: aiter_preexisting_local.diff (both in /shared_nfs/kk/dsv41/).

## ATOM-port worktree (current best config)

Fresh container: run section 1 (`setup_env.sh`), then

```bash
bash $D/scripts/setup_atomport_env.sh              # idempotent; VERIFY_ONLY=1 = check only; prints the LAUNCH line
```

It creates (all via PYTHONPATH; the image's editable sglang/aiter and global flydsl 0.3.2 stay untouched):
- sglang `/sgl-workspace/sglang-rolao-opt`: worktree of sglang-dsv41 + remote `rolao` (RolaoDenthu/sglang), branch
  `atomport-mxfp8-producers` @ `026da361c0` (= pushed `dsv41/opt-branch`).
- sglang patch `patches/sglang_local_engram_host_devptr_0001.patch` applied to that worktree (engram host table must use the
  device VA; without it TP2 faults at prefill graph capture on nodes where `scripts/hostreg_devptr_check.py` says same=False).
- aiter `/sgl-workspace/aiter-5750`: worktree of `/sgl-workspace/aiter` at ROCm/aiter PR #5750 head `1053c79bb`, CK submodule,
  `patches/aiter_local_5750_worktree_0001.patch` (#5561 + MoE-tuner GPU data gen + 2 local fixes), tuned FMoE CSV
  `patches/aiter_local_dsv41_tp2_sef_fp8fp4_tuned_fmoe.csv` -> `aiter/configs/model_configs/`. aiter merges the
  `model_configs/*tuned_fmoe*.csv` of the aiter that is IMPORTED; restart the server after changes.
- flydsl 0.3.4.1 (PyPI) in `/sgl-workspace/pydeps-flydsl-0341` (`pip --no-deps --target`).
- AgentX deps (`/workspace/InferenceX`, aiperf venv) are only checked; build them with the agentx skill.

Knobs are `AP_*` (the image exports `AITER_COMMIT`, so unprefixed names collide), e.g. build a second copy:
`AP_SGL_DIR=... AP_BRANCH= AP_AITER_DIR=... AP_FLYDSL_DIR=...`. First server start JIT-builds the aiter-5750 modules.
Verified 2026-09-29: a copy built this way is file-identical to the live env (except the uncommitted
SGLANG_HIP_SPEC_EVENT_WAIT experiment in overlap_utils.py, which is intentionally not reproduced).

```bash
cd /shared_nfs/kk/dsv41/agentx && PYTHONPATH=/sgl-workspace/pydeps-flydsl-0341:/sgl-workspace/aiter-5750:/sgl-workspace/mori \
  SRC=/sgl-workspace/sglang-rolao-opt/python SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1 \
  EXTRA_ARGS="--fp8-gemm-backend aiter --enforce-shared-experts-fusion" SERVER_ONLY=1 OPUS=0 TP=2 EP_SIZE=1 GPUS=4,5 \
  CONC=1 PREFILL_DECODE_INTERVAL=16 TAG=<tag> \
  setsid nohup bash /workspace/claude-skills/dsv41/scripts/agentx_colleague_run.sh > <tag>.nohup 2>&1 < /dev/null &
```

`EVAL_ONLY=true`: GSM8K, real acceptance; drop `SERVER_ONLY`: run AgentX; series:
`scripts/agentx_series.sh RUNS="conc:pdi ..." SCRIPT=agentx_colleague_run.sh PREFIX=...`.

## Known failure signatures

| Symptom | Cause / fix |
|---|---|
| `Unsupported kernel config for moe heuristic dispatch` | `AITER_BF16_FP8_MOE_BOUND` unset; use `0`. |
| `deepseek_v4_topk_transform_512 predates sort_output` | Old sgl-kernel (perf-only fallback); rerun setup step 3. |
| New sgl-kernel built, schema still old | Egg shadowed by `site-packages/sgl_kernel`; replace it. |
| Works interactively, not under nohup | Script uses `rg`; use `grep -E`. |
| "Write access to a read-only page" at long context | Missing #41159 (`sglang_local_kvstore_int64_0001.patch`). |
| TP2 `Memory access fault` at prefill graph capture (4096), both ranks, dmesg PERMISSION_FAULTS on unmapped VA | Engram host VA passed to engram_gather; apply `sglang_local_engram_host_devptr_0001.patch`. Check node: `scripts/hostreg_devptr_check.py`. |
