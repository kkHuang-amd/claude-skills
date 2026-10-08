# Prompt: reproduce OPT_SWEEP_1008 on another node

Paste after NEW_WORKSPACE_PROMPT.txt. Origin: crsuse2-m2m-255 (2026-10-08), doc dsv41/OPT_SWEEP_1008.md.

---

請用中文回答。任務：在這台 node 上重做 crsuse2-m2m-255 在 2026-10-08 做的 DSV4.1-Flash 優化 sweep（參考
/workspace/claude-skills/dsv41/OPT_SWEEP_1008.md，但那份 doc 的 owner 是 m2m-255，你不要改它；本 node 開自己的
doc：dsv41/OPT_SWEEP_1008_<hostname>.md，頂端寫 `Owner node: <hostname>` 和 CONTINUE HERE）。
先 `git -C /workspace/claude-skills pull --rebase`，並讀 dsv41/SKILL.md 和 agentx skill 的 Gotchas。

## 0. 跨 node 衝突（先處理，再跑任何東西）
/workspace 與 /shared_nfs 是多台 node 共用的。下列路徑要加上本 node 的 hostname，否則會和 m2m-255 互相覆蓋或刪除：
- aiperf runtime：agentx_colleague_run.sh 的 `AIPERF_RUNTIME_DIR=/workspace/agentx-runtime/p$PORT` -> 加 hostname
  （benchmark_lib.sh:3080 會用 AIPERF_RUNTIME_DIR 重算 AIPERF_VENV，且每次 rm -rf venv；只 export AIPERF_VENV 無效）。
- 進度檔：chain_opt1008.sh 的 `agentx/chain_opt1008.txt`、agentx_lane.sh 的 `agentx/lane_${PORT}.txt` -> 加 hostname。
- 結果 tag：sweep_ci37423_opt.sh 用 `TAGP=opt1008_<hostname短名>`（smoke 與 GSM8K log 名稱都由 TAGP 決定）。
用 knob（環境變數，預設保持原值）來做，不要寫死；改完 `bash -n`。

## 1. 更新 sglang / aiter 到 mainline（不要丟掉本地修改）
- 先備份：每個 repo 的 HEAD 和 `git diff HEAD --binary` 存到
  /shared_nfs/kk/results/DeepSeek-V4.1-Flash/repo_backup_1008_<hostname>/，再 `git stash`。
- git identity（repo-local）：kkHuang-amd / 43161300+kkHuang-amd@users.noreply.github.com。
- sglang（/sgl-workspace/sglang）：`git checkout -b dsv41-opt-1008 origin/main`，依序 cherry-pick -x
  HaiShaw/sglang 這些 branch 的單一 commit：
  i0 perf/v41-index-q-prefill-fuse, i1 perf/v41-indexer-bf16-logits, i2 perf/v41-candidate-blockmax-fastpath,
  i3 perf/v41-mhc-ar-boundary-stats, i4 perf/v41-small-moe-sort, i5 perf/v41-flydsl-mxfp8-gemm,
  i6 perf/v41-woa-fuse-mxfp8-quant。
  已知衝突：i2 與 i1 衝突在 candidate_blocks_hip.py -> i1 已包含 i2（多了 bf16 的 `.to(tl.float32)`），保留 HEAD，
  `cherry-pick --skip`；i4 衝突在 srt/environ.py -> 兩個 env var 都保留（SGLANG_ROCM_MHC_ALL_REDUCE_STATS 與
  SGLANG_AITER_SMALL_MOE_SORT_MAX_PAIRS）。之後 `git stash pop`（image 的 pyproject 修改）。
  先確認 mainline 沒動 python/sglang/kernels/aot 和 pyproject（m2m-255 時沒有，image 的 sgl_kernel 可沿用）。
- aiter（/sgl-workspace/aiter，editable install）：`git checkout -b dsv41-opt-1008 origin/main`（需含 #5896、#5967），
  cherry-pick kkHuang-amd/aiter perf/v41-indexer-bf16-logits。再重新套用 image（sglang docker/rocm.Dockerfile）的步驟，
  逐項檢查是否已在 main：torch_utils.py 的 torch.Stream 修正（用 backup diff `git apply -3 --include=...`）、
  #6042 pa_decode_sparse.py（衝突時只取註解，row_tiles 在 main 已定義，不要重複）、mla_v4 `.co` 從
  sglang/python/sglang/kernels/ops/attention/dsv4/asm/gfx950/mla_v4/ 複製；pa_mqa_logits_fp4_prefill 的
  lru_cache->cache 在 m2m-255 時已在 main（確認）。#5967 CSV 已在 main，不要再套。
- 舊 JIT：aiter/jit/*.so 和 aiter/jit/build 移到 /sgl-workspace/aiter_jit_backup_<old-sha>_<date>（flydsl_cache 保留）。

## 2. aiter 完整 AOT build（照 Dockerfile）
背景執行，log 寫檔，只看結尾：
`cd /sgl-workspace/aiter && AITER_USE_SYSTEM_TRITON=1 SETUPTOOLS_SCM_PRETEND_VERSION= PREBUILD_KERNELS=1 GPU_ARCHS=gfx950
python setup.py build_ext --inplace && GPU_ARCHS=gfx950 pip install --config-settings editable_mode=compat -e .; echo BUILD_EXIT=$?`
Pass：BUILD_EXIT=0、`import aiter` 成功、log 裡每個 `FlyDSL ... AOT: compiled N ok, M failed` 都是 0 failed
（m2m-255：129 個 .so，GEMM 6595 / MOE 2534 / MXFP4_MOE 1125 ... 全部 0 failed，約 25 分鐘）。

## 3. Sweep 設定 = InferenceX run 37423021942 + 優化
Recipe：SemiAnalysisAI/InferenceX PR #3696 @ 47adb59a 的
benchmarks/single_node/srt-slurm-recipes/dsv41flash/sglang/mi355x-fp4-mtp/agentic.yaml 與 configs/amd-master.yaml
`dsv41flash-fp4-mi355x-sglang-agentic-dspark`（用 `gh api ...contents/<path>?ref=47adb59a` 讀）。
TP2 c1/2/4/8/16/32/64、TP4 c1/2/4/8/16，ep1、無 DP、DSpark block 5、`--enforce-shared-experts-fusion
--fp8-gemm-backend aiter`。driver：dsv41/scripts/sweep_ci37423_opt.sh（已實作以下全部）：
- 每點：pdi 16（c<32）/ 4；chunk 16384 / c64 4096；mem 0.70（TP2 c16/c32 0.80；c64 0.85；TP4 c16 是 0.70）；
  max-running 2*conc；decode graph bs 64，c64 為 128（CUDA_GRAPH_MAX_BS=128）。
- 優化（全開）：SGLANG_DSV41_PREFILL_LOGITS_BF16=1、SGLANG_ROCM_MHC_ALL_REDUCE_STATS=1、
  SGLANG_AITER_SMALL_MOE_SORT_MAX_PAIRS=64、SGLANG_ROCM_MXFP8_AITER_PRESHUFFLE=1、SGLANG_HIP_WO_A_MXFP8=1
  （i0、i2 無 flag）。
- 使用者額外要求（偏離 CI）：SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1，QR_QUANT=INT8
  （-> ROCM_QUICK_REDUCE_QUANTIZATION=INT8）。

## 4. 開 sweep 前先做 GSM8K 驗證（所有優化都打開）
- smoke：`MODE=smoke LANE=tp2|tp4 bash sweep_ci37423_opt.sh`（server only，CONC 8 設定，EVAL_ONLY=true + GSM8K 1319
  5-shot）。**一定要 EVAL_ONLY=true**：否則 launcher 會設 SGLANG_SIMULATE_ACC_LEN，GSM8K 會掉到 ~0.37（無效）。
  跑完要親自確認 smoke 的 server.log 沒有任何 `SGLANG_SIMULATE_ACC` 行，且上面的優化 flag 都在
  （script 內建的 scheduler environ 檢查在 m2m-255 抓不到 pid，輸出空白，不能當作驗證）。
- 判定：TP2 與 TP4 都 >= 0.895 才算過（m2m-255：TP2 0.897、TP4 0.902，QR INT8）。若任一 < 0.895，把 QR 改回
  NONE（QR_QUANT=NONE）重跑 smoke，過了就用 NONE 跑 sweep；兩種都沒過就停下來回報，不要開 sweep。
- 兩個 smoke server 同時開時，install_agentic_deps 會搶同一個 aiperf venv（第 0 節的 per-port/hostname runtime dir
  修好之後才可平行）；保險起見 smoke 也可依序跑。

## 5. Sweep：一次只跑一個 server
- TP2 和 TP4 **不可同時跑，也不可開多個 lane**：先 `LANE=tp2` 的 7 點跑完，再 `LANE=tp4` 的 5 點
  （sweep_ci37423_opt.sh 用 /tmp/opt1008_sweep.lock flock 串行，TP4 晚 60 s 搶鎖；最穩是同一支 script 依序呼叫）。
- 整條流程（build -> smoke -> 判定 -> sweep）用 setsid nohup 的 chain 背景執行（參考 scripts/chain_opt1008.sh，
  但它的最後一段會同時開兩條 lane，要改成依序）。
- 不要在 script 執行中 in-place 修改它（bash 邊讀邊執行）；要改就寫新檔再 mv。
- 停 process 用 PID（從 /proc/<pid>/environ 判斷 HIP_VISIBLE_DEVICES / LANE），不要用 pkill -f 或含 script 名的
  pattern（會殺到自己的 shell）。
- 監看只用一個 channel（Monitor `tail -F` 進度檔，管線用 `awk '{print; fflush()}'`，不要用會 buffer 的 cut）。

## 6. 結果
- 每點 rc=0、0 error；結果在 /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/<TAGP>_tp<TP>_c<C>/。
- 表格：`python3 dsv41/scripts/agentx_agg_table.py <各點 json>`；CI baseline 用同一支 script 讀 run 37423021942 的
  artifact `results_bmk`（agg_bmk.json，`gh api .../actions/artifacts/<id>/zip`）。逐點比 TTT/gpu 與 P90 interactivity。
- 結果與每列 node+date 寫進自己的 doc（append-only）。不要 commit/push，除非我要求；commit message 不加任何 AI
  attribution。
