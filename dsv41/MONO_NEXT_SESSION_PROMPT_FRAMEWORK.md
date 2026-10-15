# Prompt: mono framework -> RFC #43949 shape, on mi355-4 (from 2026-10-14)

Paste as the first message of a new session, after NEW_WORKSPACE_PROMPT.txt.

---

請用中文回答。

先 `git -C /workspace/claude-skills pull --rebase --autostash`（有衝突就停下來回報），然後只讀：
1. /workspace/claude-skills/dsv41/MONO_FRAMEWORK_RFC_PLAN_mi355-4.md（整份；這次的計畫，owner = mi355-4）
2. /workspace/claude-skills/dsv41/MONO_KERNEL_mi355-4.md 的 CONTINUE HERE 與 Log 最後 10 條；不要整份重讀
3. RFC 原文：`gh api repos/sgl-project/sglang/issues/43949 --jq .body`，以及它的 comments（kevin-mii 10-13 21:34 的回覆）

現況：
- 使用者決定先暫停 27b / S3 / 22 / 13 / 27a，自己把 mono 框架改成 RFC #43949 的形狀（不等 kevin-mii 的原型）。
- worktree /sgl-workspace/sglang_v41mono，branch perf/v41-mono-fw @ 53d0945468（未 push）；未 commit 的 S3 / 22 / 27a 存在
  patches/wip_s3_t22_t27a_1014.diff，也凍結在 /sgl-workspace/snap_t27a_1014。
- task 25（TP2 sweep）可能還在 GPU 4,5 上跑，結束後把結果整理進 MONO_KERNEL_mi355-4.md（section "AgentX night 10-13" 的表）。

這個 session：先跟我確認計畫的 step 0（開關名稱、MonoSpec 欄位是否完全照 RFC、M3 skeleton 去留、WIP 怎麼處理），再從 step 1 開始。
每一步：CPU tests、mono_isa_dump.py 確認 ISA 不變（step 5 除外）；GPU 驗證用凍結副本。

規則：
- 大目錄先 count 再 list；讀檔先 wc，再讀需要的範圍。
- GPU 只用 4-7，用前確認空閒；同時只跑一條背景串接，用 process group 停止。
- commit 用 author wunhuang <wunhuang@amd.com>（-c 帶入），不加任何 AI attribution；commit / push 前先問我。
