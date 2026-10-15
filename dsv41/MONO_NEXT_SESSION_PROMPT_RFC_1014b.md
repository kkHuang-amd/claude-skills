# Prompt: mono framework in the RFC #43949 shape, continue on mi355-4 (from 2026-10-14 17:10)

Paste as the first message of a new session, after NEW_WORKSPACE_PROMPT.txt.

---

請用中文回答。

先 `git -C /workspace/claude-skills pull --rebase --autostash`（有衝突就停下來回報），然後只讀：
1. /workspace/claude-skills/dsv41/MONO_FRAMEWORK_RFC_PLAN_mi355-4.md 的「CONTINUE HERE (mi355-4, 2026-10-14 17:10 +08)」
   整段，以及「## RFC #43949」「## Steps」兩節；Session log 只在需要細節時用 rg 找，不要整段讀
2. `git -C /sgl-workspace/sglang_v41mono log --oneline -4` 與 `git status --short | head`（確認 branch perf/v41-mono-rfc 與未 commit 的 A）

現況：
- step 0-5 與 kevin-mii 在 PR #43984 的 review（B 結構對齊、C srt 縮到 7 個檔案、K2i 搬進 plugin、DSpark 拆成獨立 commit）都完成，
  本地 commit 在 perf/v41-mono-rfc（未 push）。
- A（step 5 的 TPOT 成本：每 32 圈才讀時鐘 / STOP）在工作樹，驗證串接 scripts/rfc_a_validate_1014.sh 可能還在跑或已結束
  （log /shared_nfs/kk/rfc_a_validate_1014.log）。

這個 session：先讀 A 串接的結果並回報（fault injection、GSM8K / NIAH、s4 / c / a 的 TPOT 表）；通過的話問我是否把 A 併進第一個
commit。然後跟我確認 CONTINUE HERE 裡的待決事項（c1 +2% 的 srt hook、experimental/monokernels 搬移、rebase、capture sizes），
再往 step 6 走。

規則：
- 大目錄先 count 再 list；讀檔先 wc，再讀需要的範圍。
- GPU 只用 4-7，用前確認空閒；同時只跑一條背景串接，用 process group 停止。
- commit 用 author wunhuang <wunhuang@amd.com>（-c 帶入），不加任何 AI attribution；commit / push 前先問我。
- skills repo（/workspace/claude-skills）這次的文件與腳本都還沒 commit；要 commit / push 前先問我。
