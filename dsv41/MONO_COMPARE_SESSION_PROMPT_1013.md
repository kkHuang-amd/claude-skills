# Prompt: SGLang mono -- task 20, compare with zijiecode/sglang-mono-amd m3-mono, on mi355-4 (from 2026-10-13)

Paste as the first message of a new session, after NEW_WORKSPACE_PROMPT.txt.

---

請用中文回答。

先 `git -C /workspace/claude-skills pull --rebase --autostash`（本地有未提交修改；有衝突就停下來回報），然後只讀：
1. /workspace/claude-skills/dsv41/MONO_COMPARE_mi355-4.md 全文（owner = mi355-4；這個 task 的 source of truth）
2. /workspace/claude-skills/dsv41/MONO_KERNEL_mi355-4.md 的 CONTINUE HERE、Task list，以及
   "Design: SGLang mono framework, multi-model from day one" 小節（含 Layers / L1 modules / Keeping L1 model-agnostic /
   Second model: MiniMax-M3），不要整份重讀

現況（細節以文件為準）：
- 我們的 mono：/sgl-workspace/sglang_v41mono（branch perf/v41-mono @3c5ea0778c，已 push 到 HaiShaw/sglang）。
  L1 框架在 python/sglang/srt/layers/mono/，DSV4.1 的 kernel + spec 在 python/sglang/srt/models/deepseek_common/amd/mono/，
  MiniMax-M3 只有 spec 骨架（python/sglang/srt/models/minimax_m3_mono/，task 12 未開始）。開關 --enable-mono-kernel。
  AgentX TP4（新 base，K2A=1）：c1 P90 617.1（vs ATOM 1010 +0.2%），c4 529.1（+22.1%）。
- 對方：https://github.com/zijiecode/sglang-mono-amd/tree/m3-mono @acf209105，在它的 main 上 3 個 commit（約 260 KB）：
  kernels 在 python/sglang/kernels/ops/mono/gfx950/（common + minimax_m3），runtime 在 python/sglang/srt/layers/mono/
  （contracts / families / dispatch / runner / peer_buffer / adapters / minimax_m3），開關 --layer-backend mono，
  MiniMax-M3 已在 Crusoe 264 驗證。clone 到 /tmp/mono_src/sglang-mono-amd。

這個 session 的目標：task 20
- 逐軸比較兩邊的設計（文件裡的表：package 分層、開關與 refusal、spec/contract API、dispatch 與 graph capture、runner 與
  scratch/epoch、peer buffer/all-reduce、weights、metadata、kernel 來源、MiniMax-M3 覆蓋、indexer、debug、測試、驗證），
  每格要有兩邊的 file:line 證據。
- 截長補短：每一軸給結論（採用對方 / 保留我們 / 合併），列出 adopt / keep / drop 清單和合併計畫。
- 特別看：對方 M3 的做法能不能直接補上我們的 task 12；我們的框架（多模型 spec、fail-stop、K2i）對方缺什麼。
- 這個 session 只讀不改：兩邊的 code 都不動，先把比較和計畫寫進 MONO_COMPARE_mi355-4.md，跟我確認後才動手。

規則：
- 大目錄先 count 再 list；讀檔先 wc，再讀需要的範圍；對方 repo 用 rg 找，不要整份 cat 大檔（post_attn.py 117 KB）。
- 需要跑 GPU 時只用 GPU 4-7，用前 `rocm-smi --showmemuse` 確認空閒；這個 task 原則上不需要 GPU。
- 結果、決策、下一步寫進 MONO_COMPARE_mi355-4.md（CONTINUE HERE + 表 + append-only Log，每列帶 node + date）；
  MONO_KERNEL_mi355-4.md 只更新 task 20 那一列和 Log。
- 不要 commit/push，除非我要求；commit message 不加任何 AI attribution。
