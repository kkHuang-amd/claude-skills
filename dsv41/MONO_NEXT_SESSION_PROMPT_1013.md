# Prompt: continue the SGLang DSV4.1 mono kernel work on mi355-4 (after 2026-10-12)

Paste as the first message of a new session, after NEW_WORKSPACE_PROMPT.txt.

---

請用中文回答。

先 `git -C /workspace/claude-skills pull --rebase --autostash`（本地有未提交修改；有衝突就停下來回報），然後只讀：
1. /workspace/claude-skills/dsv41/MONO_KERNEL_mi355-4.md 的 CONTINUE HERE 和 Task list（owner = mi355-4；這是 source of truth）
2. 需要細節時再讀對應小節（"Task 9: c1 gap vs ATOM"、"Task 9b A/B"、AgentX TP4 表格、Design），不要整份重讀

現況（細節以文件為準）：
- sglang：/sgl-workspace/sglang 分支 dsv41-mono-ffn，aiter：/sgl-workspace/aiter 分支 dsv41-opt-1008。全部未 commit。
  最新凍結副本：/sgl-workspace/snap_task8b_1012。
- 40 層 whole-layer mono（production KV layout）+ Task 9b nosync 修正（HIP 不再每步 publish_ready.synchronize）
  + Task 8 DSpark draft 走 mono（SGLANG_MONO_DSV41_DRAFT=1）。
- AgentX TP4 P90：c1 587.9 / c2 538.7 / c4 487.2 / c8 393.9，對 ATOM 37597985033 +1.8% / 無資料 / +13.6% / +19.2%；
  TPOT p50 1.58 / 1.57 / 1.59 / 1.82 ms（ATOM 1.52 / - / 1.53 / 1.94）。GSM8K 0.904，真實 accept len 3.398。
- 剩下：c1/c4 每 token 延遲仍比 ATOM 慢約 4%（ISL 越長越明顯）；估計還有約 0.6 ms/step 的 host 空泡（Task 9c）。

接下來（先跟我確認優先順序再開始）：
1. Task 9c：剩下的 host 空泡（模擬 AL 時 eager 的 accept / commit_hidden、host 端 draft/verify 準備）。
2. 是否移植 ATOM K2a（indexer 隨 context 的成本）。
3. Task 10-13（bounded wait / error word、逐 stage 數值測試、MiniMax-M3、GSM8K A/B x3）。
4. commit 前要先移除 SGLANG_MONO_DSV41_DRAFT_CHECK 的除錯程式碼。

規則：
- 只用 GPU 4-7，不跟別人共用；用前 `rocm-smi --showmemuse` 確認空閒。
- AgentX sweep 和 A/B 一律從凍結的程式碼副本跑（`cp -a /sgl-workspace/sglang/python <snap>/` 然後 `SRC=<snap>/python`）。
- 從 /tmp 啟動 server（不要在 sglang/srt/models 底下啟動：models/transformers.py 會蓋掉 transformers 套件）。
- profile 用 PROF_ACT=GPU；nosync 修正之後 profiler 會漏記 verify graph 的 kernel，空泡只能從乾淨的 TPOT 推算。
- 長任務背景跑、log 寫檔、只帶 marker 回 chat；停 process 用 PID（不要 pkill -f 或含 script 名的 pattern）；執行中的 script 不要原地改。
- mi355-4 的 /shared_nfs 跟 m2m-255 不是同一個 NFS；CI 對照檔用 ref_ci_37423021942/agg_bmk_reconstructed.json，
  ATOM 對照在 ref_atom_37597985033/tp4_c{1,4,8}/。
- 結果、決策、下一步寫進 MONO_KERNEL_mi355-4.md（CONTINUE HERE + append-only Log，每列帶 node + date）。
- 不要 commit/push，除非我要求；commit message 不加任何 AI attribution。
