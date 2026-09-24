# RUNBOOK — DeepSeek-V4.1-Flash on 4×MI355X with SGLang (PR #39857)

Reproduces the environment validated on 2026-09-24. Results for this state: GSM8K 5-shot/1319 ≈ 0.90–0.91
(DSpark off and on); PR-method bs1 decode 147.7 tok/s (DSpark off) / 621 tok/s (DSpark on).
Everything below is automated by `scripts/`; commands assume `D=/workspace/claude-skills/dsv41`.

## 0. Base container

| Item | Value |
|---|---|
| Image family | SGLang ROCm dev image, `GPU_ARCH=gfx950-rocm720` (ROCm 7.2.0, Ubuntu 22.04, Python 3.10, `/opt/venv`) |
| torch / triton | 2.9.1+rocm7.2.0 / 3.7.0 (`TRITON_COMMIT=42270451`) |
| aiter | editable `/sgl-workspace/aiter` @ `acf8fdf9307431ece8ee275971c41cb3d1a7020b` |
| flydsl | 0.3.2 |
| Hardware | 8× MI355X (gfx950); each server uses 4 GPUs |
| Upstream alternative | `lmsysorg/sglang:dev-dsv41-mi35x` (cookbook preview image, not what we ran) |

The container's global env contains values that differ from the PR. `launch_server.sh` overrides them:
`ROCM_QUICK_REDUCE_QUANTIZATION=INT8`→`NONE` and `SGLANG_USE_ROCM700A=1`→`0`. ROCM700A only affects
DP-attention gathers, so it has no effect at TP4 without `--enable-dp-attention`.
Container vars kept as-is: `HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1 NCCL_MIN_NCHANNELS=112
SGLANG_SET_CPU_AFFINITY=1 AITER_USE_SYSTEM_TRITON=1 PYTHONPATH=/sgl-workspace/mori:/sgl-workspace/aiter`.

## 1. Setup (idempotent)

```bash
bash $D/scripts/setup_env.sh        # ~5 min first time (sgl-kernel build), seconds afterwards
```

The script does the following:
1. Clones `kevin-mii/sglang` branch `dsv41-amd-main` to `/sgl-workspace/sglang-dsv41` if it is missing, and adds
   the `upstream` remote. Validated HEAD: `e2e824dc58`.
2. Checks the aiter pin and applies `patches/aiter_5561_*` and `patches/aiter_5802_*`, skipping any that are
   already applied. Both are unmerged upstream as of 2026-09-24. Neither changed throughput. #5561 is a race
   fix, so keep it. #5802 is unused at `BOUND=0`.
3. Rebuilds sgl-kernel (AOT) from the branch so `deepseek_v4_topk_transform_512` has `sort_output`.
   It builds in a `/tmp` copy and never modifies the repo. It then **replaces `site-packages/sgl_kernel` with the
   egg's copy**, because the egg is otherwise shadowed. The original is backed up to
   `/shared_nfs/kk/dsv41/sgl_kernel_backup_orig`.
4. Checks that the model exists at `/shared_nfs/models/deepseek-ai/DeepSeek-V4.1-Flash`.

We run the branch through `PYTHONPATH` (inside `launch_server.sh`). The pip-installed editable sglang
(`/sgl-workspace/sglang`, main) is left untouched.

## 2. Launch

```bash
cd /shared_nfs/kk/dsv41
nohup bash $D/scripts/launch_server.sh > server.log 2>&1 &                                  # DSpark off, GPU0-3 :30000
DSPARK=1 GPUS=4,5,6,7 PORT=30001 nohup bash $D/scripts/launch_server.sh > server_dspark.log 2>&1 &
grep -E 'ready to roll|Traceback|Error' server.log | tail -3      # ready in ~5-15 min (weights 476 GB + graphs)
```

Load-bearing settings (details in SKILL.md):
- `AITER_BF16_FP8_MOE_BOUND=0` is **not in the cookbook**. Without it, prefill graph capture crashes with
  `Unsupported kernel config for moe heuristic dispatch`.
- `SGLANG_USE_AITER=1`
- `AITER_FLYDSL_FORCE_REDUCE=1` gives deterministic output.
- `--disable-radix-cache` and `--cuda-graph-backend-prefill breakable`.
- FP8 KV cache and page size 256 are set automatically.

## 3. Validate

```bash
TAG=check PORT=30000 bash $D/scripts/run_gsm8k.sh          # expect Accuracy >= 0.89 (spread ±1pt run-to-run)
TAG=check PORT=30000 bash $D/scripts/run_pr_style_c1.sh    # expect ~148 (off) / ~620 on :30001 (DSpark)
TAG=check PORT=30000 bash $D/scripts/run_throughput.sh     # full sweep -> results/perf.md
# or everything after launch: TAG=x PORT=.. SERVER_LOG=.. SERVER_PID=.. bash $D/scripts/pipeline_eval.sh
```

Compare against the PR only with `run_pr_style_c1.sh`. The PR metric excludes TTFT and takes the median of
6 runs. The sweep's `Output token throughput` includes TTFT and reads roughly 5–20% lower.

## 4. Stop / clean up

```bash
ps -eo pid,comm,args | awk '$2=="python3" && /sglang.launch_server/{print $1}' | xargs -r kill
rocm-smi --showmeminfo vram | grep Used        # wait until ~0 before relaunching
```

Never use `pkill -f`/`pgrep -f` with a pattern that also appears in your own command line.

## 5. Revert to the pristine container

```bash
for p in $D/patches/aiter_*.patch; do git -C /sgl-workspace/aiter apply -R "$p"; done
SP=/opt/venv/lib/python3.10/site-packages
rm -rf $SP/sgl_kernel && cp -a /shared_nfs/kk/dsv41/sgl_kernel_backup_orig/sgl_kernel $SP/  # or ..._backup_0.4.7
```

Local aiter also carries two edits that predate this work: `pa_mqa_logits_fp4_prefill.py` `lru_cache→cache`
(same as the Dockerfile sed) and `csrc/cpp_itfs/torch_utils.py`. They are saved in
`/shared_nfs/kk/dsv41/aiter_preexisting_local.diff`.

## Known failure signatures

| Symptom | Cause / fix |
|---|---|
| `Unsupported kernel config for moe heuristic dispatch` at prefill graph capture | `AITER_BF16_FP8_MOE_BOUND` is unset. Use `0`. |
| log: `deepseek_v4_topk_transform_512 predates sort_output` | Old sgl-kernel is loaded. Rerun setup step 3 (a perf-only fallback). |
| New sgl-kernel built but schema still old | The egg is shadowed by `site-packages/sgl_kernel`. Replace that dir. |
| Script works interactively but not under nohup | It uses `rg`. Scripts must use `grep -E`. |
