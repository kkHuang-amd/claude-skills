# Prompt: continue the SGLang DSV4.1 mono kernel work on mi355-4 (2026-10-12)

Paste as the first message of a new session, after NEW_WORKSPACE_PROMPT.txt.

---

請用中文回答。

先 `git -C /workspace/claude-skills pull --rebase --autostash`（本地有未提交修改；有衝突就停下來回報），然後只讀：
1. /workspace/claude-skills/dsv41/MONO_KERNEL_mi355-4.md 的 CONTINUE HERE、Task list，以及 Design 一節（owner = mi355-4；這是 source of truth）
2. 需要細節時再讀同一份文件的對應小節（P0.1 layout 表、All 40 layers whole、V41 port plan、Remaining gap to ATOM），不要整份重讀

現況（細節以文件為準）：
- sglang：/sgl-workspace/sglang 分支 dsv41-mono-ffn（base HaiShaw perf/v41-mono），aiter：/sgl-workspace/aiter 分支 dsv41-opt-1008。全部未 commit。
- 40 層 DSV4.1 decoder layer 都能以 whole-layer mono 跑在 production KV layout（V41 + V41_FP4）：
  `SGLANG_MONO_DSV41_WHOLE=1 SGLANG_MONO_DSV41_INDEXER=1 SGLANG_MONO_DSV41_EDGE=1` + `--enable-mono-kernel --disable-shared-experts-fusion`。
  GSM8K 0.907；TPOT（sim AL）c1/c2/c4/c8 比 production -23/-21/-7/-6%。
- indexer / compressor 層是 hybrid（K1 -> SGLang forward_low_ratio_sources -> key-table kernel -> K2），不是 ATOM 的 in-kernel K2a。
- AgentX TP4（V4 layout）：mono 對同分支不開 mono 的 P90 c1 +34%、c2 +43%、c4 +47%、c8 +30%。使用者回報 c1 P90 仍落後 ATOM run
  37597985033 約 10%；40 層比 30 層在 c1 只多 +2.5%。

接下來（依序，先跟我確認優先順序再開始）：
1. Task 9：c1 step profile（長 context 的 AgentX 請求、40 層 production layout），把跟 ATOM 的差距拆成 indexer kernel / DSpark draft /
   層外 host 開銷，決定要不要移植 ATOM K2a。
2. Task 8：DSpark draft 層走 mono。
3. 在 production layout 上重跑 AgentX TP4 c1-c8（取代 V4 的數據），對 SA CI 37423021942 和 base_tp4_c* 比較。
4. Task 10-13（bounded wait / error word、逐 stage 數值測試、MiniMax-M3、GSM8K A/B x3）。

規則：
- 只用 GPU 4-7，不跟別人共用；用前 `rocm-smi --showmemuse` 確認空閒。
- AgentX sweep 一律從凍結的程式碼副本跑（`cp -a /sgl-workspace/sglang/python <snap>/` 然後 `SRC=<snap>/python`）：sweep 每個點都會重新載入
  程式碼，10-11 在 sweep 中改 code 讓兩個點作廢。
- 長任務背景跑、log 寫檔、只帶 marker 回 chat；停 process 用 PID（不要 pkill -f 或含 script 名的 pattern）；執行中的 script 不要原地改（寫新檔再 mv）。
- mi355-4 的 /shared_nfs 跟 m2m-255 不是同一個 NFS；CI 對照檔用 ref_ci_37423021942/agg_bmk_reconstructed.json。
- 結果、決策、下一步寫進 MONO_KERNEL_mi355-4.md（CONTINUE HERE + append-only Log，每列帶 node + date）。
- 不要 commit/push，除非我要求；commit message 不加任何 AI attribution。
