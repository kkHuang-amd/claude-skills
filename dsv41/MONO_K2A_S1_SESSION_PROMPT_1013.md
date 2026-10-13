# Prompt: SGLang DSV4.1 mono -- task 17 K2a stage S1 on mi355-4 (from 2026-10-13)

Paste as the first message of a new session, after NEW_WORKSPACE_PROMPT.txt.

---

請用中文回答。

先 `git -C /workspace/claude-skills pull --rebase --autostash`（本地有未提交修改；有衝突就停下來回報），然後只讀：
1. /workspace/claude-skills/dsv41/MONO_KERNEL_mi355-4.md 的 CONTINUE HERE 和 Task list（owner = mi355-4；source of truth）
2. /workspace/claude-skills/dsv41/K2A_PORT_PLAN_mi355-4.md 全文（158 行，task 17 的計畫）
3. 需要細節時才讀 MONO_KERNEL_mi355-4.md 的 "Indexer / compressor layers: design"、"Task 9"（每步 indexer kernel 成本）、
   "Task 14"（為什麼瓶頸是長 ISL）小節，不要整份重讀

現況（細節以文件為準）：
- sglang：/sgl-workspace/sglang 分支 dsv41-mono-ffn；aiter：/sgl-workspace/aiter 分支 dsv41-opt-1008。全部未 commit。
  最新凍結副本：/sgl-workspace/snap_task9c_1012（40 層 mono + nosync + draft mono + sim-accept fold）。
- 8 個 indexer 層（2,8,14,20,24,28,32,36）是 hybrid：K1 輸出 x / q_lora -> SGLang indexer（+ 2/8/14/20 的 compressor）
  在 K1 和 K2 之間跑 -> K2 用新的 top-k 建 key table 再做 attention + FFN。
- AgentX：除了 TP4 c1 P90 落後 ATOM 1010 約 2.2%，其他點都領先。c1 的尾端是長 ISL request（ITL p90 1.665 vs ATOM 1.628），
  c1 已經沒有 host 空泡 -> 要降 indexer 隨 context 增加的成本。
- ATOM 原始碼：/tmp/mono_src/ATOM（origin/main 已 fetch；K2a 相關在 atom/models/deepseek_v41/mono/，用
  `git -C /tmp/mono_src/ATOM show origin/main:<path>` 讀）。若 /tmp 被清掉，重新 clone https://github.com/ROCm/ATOM.git。

這個 session 的目標：task 17 的 S1（只做 S1，S2/S3 之後再說）
- 把 ATOM 的 FP4 scorer（index_score_fp4.py）+ 精確 top-k（topk.py、index_score.py 的 tree 部分）+ index_plan 的 FP4 部分
  vendor 到 mono/kernels/indexer/，包成獨立的 persistent launch（"K2i"），開關 SGLANG_MONO_DSV41_K2A=1（需 INDEXER=1）。
- K2i 取代 SGLang 的 logits / top-k / candidate kernel，輸出 sparse_raw_indices / sparse_page_indices；
  compressor 和 SGLang 的 q 路徑（wq_b、gemv、q-pack）暫時不動；kt_kernel 和 K2 不動。
- 保留 SGLang 的 page-64 index plane，只改讀 key 的定址（計畫第 3 節）；ATOM host 端的 walk plan 改成 device kernel（HIP graph 相容，第 4 節）。
- 先跟我確認實作切分和順序，再開始寫。

驗證順序（每步過了才往下）：
1. 數值（計畫第 5 節 S1 的 (a)-(e)）：logits vs flydsl_pa_mqa_logits_fp4（8k/128k/512k）、相同 logits 下 top-k 集合、
   每列 sparse_raw_indices 的 Jaccard（median 1.0、p1 >= 0.99）、K2 輸出誤差、walk_plan kernel == host 版本。
2. GSM8K TP4 >= 0.895（EVAL_ONLY, parallel 8），外加 128k needle 檢查。
3. TPOT A/B：同一份凍結副本切換 SGLANG_MONO_DSV41_K2A=0/1，ISL 8k / 128k / 512k，c1/c4 x3（scripts/task9b_ab.sh，
   LOAD_EXTRA / AB_CONCS 可用；512k 需改 --isl）。
4. 有收益才跑 AgentX TP4 c1（和 c4），跟 task9c_tp4_c1 與 ATOM 1010 比 median ITL、per-request join 與 P90
   （scripts/agentx_join_itl.py、agentx_table.py；注意 ATOM c1 P90 有「斷崖」假象，主要看 median 和長 ISL bin）。

規則：
- 只用 GPU 4-7，不跟別人共用；用前 `rocm-smi --showmemuse` 確認空閒。TP2 和 TP4 不同時跑。
- A/B 和 AgentX 一律從凍結副本跑（`cp -a /sgl-workspace/sglang/python <snap>/` 然後 `SRC=<snap>/python`）；執行中的 script 不要原地改。
- 從 /tmp 啟動 server（不要在 sglang/srt/models 底下啟動）。
- profile 用 PROF_ACT=GPU；nosync 之後 torch profiler 會漏 verify graph 的 kernel。rocprofv3 在這台不能用
  （attach 沒輸出；包住 server 會 HSA_STATUS_ERROR_INVALID_PACKET_FORMAT）。
- 長任務背景跑、log 寫檔、只帶 marker 回 chat；停 process 用 PID（不要 pkill -f 帶 script 名）。
- 結果、決策、下一步寫進 MONO_KERNEL_mi355-4.md（CONTINUE HERE + task 17 + append-only Log，每列帶 node + date）。
- 不要 commit/push，除非我要求；commit message 不加任何 AI attribution。
