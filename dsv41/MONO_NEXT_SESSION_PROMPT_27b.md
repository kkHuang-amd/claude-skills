# Prompt: DSV4.1 mono task 27b (token-parallel layers) on mi355-4 (from 2026-10-14)

Paste as the first message of a new session, after NEW_WORKSPACE_PROMPT.txt.

---

請用中文回答。

先 `git -C /workspace/claude-skills pull --rebase --autostash`（有衝突就停下來回報），然後只讀：
1. /workspace/claude-skills/dsv41/TOKEN_PAR_PORT_PLAN_mi355-4.md（整份，task 27b 的計畫；owner = mi355-4）
2. /workspace/claude-skills/dsv41/MONO_KERNEL_mi355-4.md 的 CONTINUE HERE、Task list，以及 Log 最後 15 條；不要整份重讀

現況（細節以文件為準）：
- worktree /sgl-workspace/sglang_v41mono：perf/v41-mono-fw 53d0945468（task 26，local commit，未 push），加上未 commit 的
  task 17 S3（`SGLANG_MONO_DSV41_K2A_IQ`，預設 0）、task 22 check-mode 統計、task 27a（ATOM #2545 非 token-parallel 的移植）。
  凍結副本：snap_s3_1014、snap_t22_1014、snap_t27a_1014。
- GPU 4-7 上依序排了 4 條背景串接：task 25 → S3 驗證 → 22/13 → 27a A/B（PGID 與 log 見 CONTINUE HERE）。先檢查它們的進度和結果，
  有結果就整理進 MONO_KERNEL_mi355-4.md。
- ATOM 參考：/tmp/mono_src/ATOM @ 5e8d9131（若 /tmp 被清掉，從 https://github.com/ROCm/ATOM.git 重新 fetch）。

這個 session：
- 先報告背景串接的結果（S3、22/13、27a 的 GSM8K / TPOT / check mode），再照 TOKEN_PAR_PORT_PLAN 的 step 1 開始做 27b。
- 每一步結束：非 tok_par 的 build 用 mono_isa_dump.py 確認 ISA 不變、CPU tests 通過；GPU 驗證用凍結副本跑。

規則：
- 大目錄先 count 再 list；讀檔先 wc，再讀需要的範圍。
- GPU 只用 4-7，用前 `rocm-smi --showmemuse` 確認空閒，同時只跑一條背景串接，用 process group 停止。
- AgentX 從 S3 驗證之後加 `--enable-encoder-swa-bounded-replay`（見 Decisions）。
- commit 用 author wunhuang <wunhuang@amd.com>（-c 帶入），不加任何 AI attribution；commit / push 前先問我。
