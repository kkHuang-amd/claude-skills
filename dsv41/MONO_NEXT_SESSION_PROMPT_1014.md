# Prompt: DSV4.1 mono on SGLang -- next tasks after task 20, on mi355-4 (from 2026-10-14)

Paste as the first message of a new session, after NEW_WORKSPACE_PROMPT.txt.

---

請用中文回答。

先 `git -C /workspace/claude-skills pull --rebase --autostash`（本地有未提交修改；有衝突就停下來回報），然後只讀：
1. /workspace/claude-skills/dsv41/MONO_KERNEL_mi355-4.md 的 CONTINUE HERE、Task list、"AgentX night 10-13" 小節
   （owner = mi355-4；source of truth），不要整份重讀
2. 需要時才讀 /workspace/claude-skills/dsv41/MONO_COMPARE_mi355-4.md（task 20 的比較、合併計畫、驗證結果）

現況（細節以文件為準）：
- perf/v41-mono = 134919476c，已 push 到 HaiShaw/sglang。worktree /sgl-workspace/sglang_v41mono（branch perf/v41-mono-fw
  同一個 commit），凍結副本 /sgl-workspace/snap_fw_f6_1013。開關只有 `--layer-backend mono`（不用任何 mono env）。
- kernels 在 python/sglang/kernels/ops/mono/gfx950/{common,deepseek_v41}，L1 在 python/sglang/srt/layers/mono/，
  DSV4.1 host 端在 python/sglang/srt/models/deepseek_common/amd/mono/。
- 驗證：GSM8K 1319 0.901（base 0.912）、NIAH 60/60、TPOT ±1.4%；AgentX TP4 c1 7,401 / 610.3、c4 9,561 / 516.0 與重跑
  10,021 / 487.1、c8 16,357 / 409.9、c16 29,078 / 272.5；0 Tracebacks。

這個 session：先把 Task list 的 Open 表給我看，依下面建議順序讓我選要做哪些，確認後再動手。
- 25：補 TP2 sweep（smoke + c1/c4/c8/c16，GPU 4,5，約 4.5 h；可選 base c4 同場重跑）
- 17 S3：index query 併進 K1（估 ~-100 us/step），以及 S1 的 512k TPOT A/B
- 22 + 13：check mode 加 relative L2 / cosine、逐 stage 找第 17 層的差異來源；GSM8K mono vs no-mono 各 3 次確認 -1 pt
- 26：dead code 清理。(a) 刪 DSV4.1 專屬沒用到的函式（約 37 行：back.py woa_loads、attention/device.py ld_i64 /
  atomic_add_dev、index_plan.py final_level / task_writes），用 mono_isa_dump.py 確認 84 個 kernel 不變；(b) FFN-only launch
  現在只是沒被測試的 fallback，要我決定：加強制走 FFN 的測試保留，或刪掉。共用 common 的 705 行跟 task 23 / M3 一起處理
- 19：把 HaiShaw perf/v41-* 的優化移植進 mono 分支
- 12：MiniMax-M3（要我先決定：我們做，或 zijie 基於我們的框架做；kernel 用對方的 aiter PR6047 版本）
- 10 / 11 / 15 / 16：fail-stop、逐 stage 數值測試、V4 vs V41 layout TPOT、高併發

規則：
- 大目錄先 count 再 list；讀檔先 wc，再讀需要的範圍。
- GPU 只用 4-7，用前 `rocm-smi --showmemuse` 確認空閒，並確認沒有殘留的 sweep/fw_ab/night 程序；同時只跑一條背景串接，
  用 process group 停止（見 CONTINUE HERE 的 Background jobs）。
- AgentX / A/B 一律從凍結副本跑；結果寫進 MONO_KERNEL_mi355-4.md（CONTINUE HERE + 表 + append-only Log，每列帶 node + date）。
- commit 用 author wunhuang <wunhuang@amd.com>（-c 帶入），不加任何 AI attribution；push 前先問我。
