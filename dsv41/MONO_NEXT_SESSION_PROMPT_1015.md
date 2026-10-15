# Prompt: mono after the RFC PR split, continue on mi355-4 (from 2026-10-15 14:20)

Paste as the first message of a new session, after NEW_WORKSPACE_PROMPT.txt.

---

請用中文回答。

先 `git -C /workspace/claude-skills pull --rebase --autostash`（有衝突就停下來回報），然後只讀：
1. /workspace/claude-skills/dsv41/MONO_KERNEL_mi355-4.md 的「## Task list -- TOP TABLE」整節（mono 總表，任務一變動就要更新）
2. /workspace/claude-skills/dsv41/MONO_FRAMEWORK_RFC_PLAN_mi355-4.md 的「CONTINUE HERE (mi355-4, 2026-10-14 21:55 +08)」那段
   （含 PR split / #44405 / V4-Pro 的紀錄）；Session log 只在需要細節時用 rg 找
3. `gh pr view <n> --repo sgl-project/sglang --json state,reviewDecision,statusCheckRollup` 看 #44405 #44406 #44407 #44408
   的狀態（只取需要的欄位，不要整份 dump）

現況：
- RFC #43949 重構已拆成 4 個 PR 並開出（base main，head HaiShaw:perf/v41-mono-pr1..4，疊加）：
  #44405 DSpark host sync（HIP only，多了 0a603c0670 的 _is_hip 限定）、#44406 框架 + common_hip、
  #44407 DSV4.1 + skip_metadata hook、#44408 DSpark draft stages。舊 PR #43984 已關閉並留言指向新 PR。
- #44406..#44408 還帶著 #44405 的舊 commit（沒有 _is_hip 那個 commit）；#44405 合併後要處理後面 PR 的 rebase。
  規則不允許 force-push，到時先跟我討論做法。
- 驗證：rebase 後 tip GSM8K 0.905、NIAH 60/60、TPOT 跟 h 一致；V4-Pro TP8 EAGLE 上 #44405 正確（0.951 / 0.948），速度在雜訊內。
- AgentX 10-14 夜跑：h（guard + hook）跟 s4（無 guard）、fwfinal 在 TP2 / TP4 c1-c8 都在雜訊內；仍落後 ATOM 1012（TP4 c8 P90 -8.9%）。
- 沒有背景工作在跑；GPU 全空。

這個 session：先回報 4 個 PR 的 CI / review 狀態（有失敗的 check 就找原因，有 review 意見就整理給我）。
然後跟我確認下一步要從總表的哪一項開始：A3（guard 剩下的 1-3% TPOT）、K2i deadline、capture sizes、
experimental/monokernels 搬移、step 6（測試位置 / e2e 切換點測試 / validate.py / skill）、或追 ATOM 1012（27b token-parallel、19 prefill opts）。

規則：
- 大目錄先 count 再 list；讀檔先 wc，再讀需要的範圍。
- GPU 預設只用 4-7（我另外同意時才用 0-7），用前確認空閒；同時只跑一條背景串接，用 process group 停止。
- commit 用 author wunhuang <wunhuang@amd.com>（-c 帶入），不加任何 AI attribution；commit / push 前先問我；永不 force-push。
- PR 標題要過 CI 的 pr-title.yml：`[Type] [AMD] scope: title (k/N)`；描述只改 Motivation..Checklist 之間（GET body -> 替換 -> PATCH）。
- 程式碼 worktree：/sgl-workspace/sglang_v41mono（perf/v41-mono-rfc，舊）、/sgl-workspace/sglang_v41mono_split（perf/v41-mono-pr，
  4 個 PR 的 commit）、/sgl-workspace/sglang_v41mono_pr1（pr1-hip，#44405 的最新內容）、/sgl-workspace/sglang_main_1015（origin/main d4daf52369）。
- skills repo（/workspace/claude-skills）10-14 / 10-15 的文件與腳本都還沒 commit；要 commit / push 前先問我。
