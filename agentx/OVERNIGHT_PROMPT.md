# Overnight AgentX matrix — prompt for a fresh session

Paste this whole file as your first message in the new session.

---

讀 `/workspace/claude-skills/agentx/SKILL.md` 的 `CONTINUE HERE`（特別是
"START HERE"、"Reading the results"、"Tree state the matrix depends on" 三節）
以及 `Traps learned 2026-08-28` 全部 15 條。文件是唯一事實來源，不要重建對話歷史。
同時遵守 `cap-tool-output` 和 `reduce-conversation-usage`。

背景：一個 6-arm、每個 3600 s 的 overnight matrix 由
`/workspace/claude-skills/agentx/overnight_matrix.sh` 驅動，於 2026-08-28 15:02
啟動。約需 11 小時，預期不會全部跑完，這是設計如此。

## 你的工作

**1. 先確認 driver 還活著，不要重複啟動。**

```bash
ps -eo pid,etime,args | grep -E "[o]vernight_matrix.sh" | cut -c1-95
tail -20 /workspace/results/overnight/STATUS.txt
```

- 有 process → 什麼都別做，直接進監控。
- 沒有 process 但 `STATUS.txt` 最後一行不是 `finished` → 它中途死了。
  照 SKILL.md `START HERE` 的指令重新啟動；**已完成的 arm 會自動跳過**，
  但先確認沒有殘留 sglang process 佔著 GPU（trap 3）。

**2. 監控，但不要輪詢。** 用一個 Monitor 盯 `STATUS.txt` 的新行即可：

```bash
tail -n 0 -F /workspace/results/overnight/STATUS.txt
```

每個 arm 約 110 分鐘。arm 進行中若要判斷健康，看該 arm 目錄下的
`benchmark.log` 的 `Phase warmup progress` 序列，對照參考 arm
`tbo-tp8-c64/benchmark.log`（52 / 94 / 262 / 701 at 300/600/900/1200 s）。
**絕對不要**用 `/metrics`（trap 1）或 decode-batch 數（trap 2）判斷生死，
也不要拿最快的 arm 當基準（trap 11）。引用數字一律以 `benchmark.log`
與 result JSON 為準，不要引用 monitor 通知的內容（trap 14）。

**3. 每個 arm 完成後就分析，不要等全部跑完。** 配對規則：

| arm | 對照對象 |
|---|---|
| `c64-chunk16384` | 既有 `/workspace/results/b200align-tp8-c64-3600s`（18,518.0） |
| `c256-chunk8192-gd` | `c256-chunk8192`（唯一差別是 delayer guard） |
| `c256-chunk16384` | `c256-chunk8192` |
| `c128-chunk16384` | `c128-chunk8192` |

**跨 conc 絕不比 headline**（`conc-and-trace-mix.md` §19.6）。

引用前先過門檻：`errors=0`、`records_error_dropped=0`、
`duration_seconds` ≈ 3630、`input.mean` 與配對 arm 相近。

**判讀門檻：3600 s 的重複散布是未知數。** 1200 s 實測同 config 重複差
**5.67 %**。3600 s 應該更緊但沒人驗證過。所以：低於 ~5 % 一律稱
「未定，需要重複」，不要說成有效益；≥10 % 才算 §25 意義下的訊號。
這是今天最重要的教訓——整個白天都在對 0.1–2.5 % 的差異過度解讀。

**4. 跑完（或早上被叫醒時）把結果寫回 SKILL.md 的 `CONTINUE HERE`**，
並清 GPU（trap 3：server 會活過 launcher，要逐 PID `kill -9` 比對
`sglang.launch_server` 和 `sglang::`；trap 4：KFD reclaim 可能延遲數分鐘，
`--showpids` 只剩 `gpuagent` 就是乾淨了）。

## 環境現況（動任何東西前先驗證）

- topk_v2 **已 revert**，整個 matrix 都不帶它。驗證方式見 SKILL.md
  "Tree state" 表——**不要**對整個 `git status` grep "topk"，
  有四個未追蹤的 `.hip` build 產物永遠會 match，看起來像沒 revert 乾淨。
- delayer guard **已套用且預設 true**；driver 對每個 arm 都明確指定
  環境變數，所以預設值不會外洩進來。
- launcher 已 patch 成吃 `CHUNK_PER_RANK`（備份
  `/tmp/b200align_launcher.pre_chunkenv.bak`，**該檔案沒有版控**）。

## 不要做的事

- 不要更新 `/sgl-workspace/sglang`（有約 20 個未提交的本地檔案）。
- 不要在 matrix 跑的時候動原始碼或 JIT cache。
- 不要因為某個 arm 看起來慢就中止它——先讀 trap 11、12、15。
