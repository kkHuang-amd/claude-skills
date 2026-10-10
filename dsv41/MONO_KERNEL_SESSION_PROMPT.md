# Prompt: start the SGLang mono-kernel work (MONO_KERNEL_EVAL.md P0 -> P1)

Paste as the first message of a new session, after NEW_WORKSPACE_PROMPT.txt.

---

請用中文回答。

先 `git -C /workspace/claude-skills pull --rebase`，然後讀（不要讀其他大檔）：
1. /workspace/claude-skills/dsv41/MONO_KERNEL_EVAL.md（整份；這是本任務的 source of truth，CONTINUE HERE 在最上面）
2. /workspace/claude-skills/dsv41/SKILL.md 的 Gotchas 與 Folder rules
3. /workspace/claude-skills/dsv41/OPT_SWEEP_1008.md 的 CONTINUE HERE（現行環境、flag、GSM8K/AgentX 方法）
4. /workspace/claude-skills/dsv41/DECODE_MULTISTREAM_1008.md 的 CONTINUE HERE（launch-gap 量測，只讀不改，owner 是 m2m-259）

任務：在 SGLang 上實作 DeepSeek-V4.1-Flash 的 mono（persistent fused decode-layer）kernel，先做 P0 再做 P1。
已決定（user, 2026-10-10）：
- Sourcing = option C：SGLang 自己的 contract（照 vLLM RFC #60904 的 MonoSpec 形狀）+ 外部 kernel
  （ROCm/ATOM #2479 的 `atom/mono` + `atom/models/deepseek_v41/mono`，或 vLLM #60397 的 FlyDSL stages），
  SGLang 只寫 adapter。不要把 ~8k 行 kernel 整包複製進 sglang，除非 P0 證明沒有別的路，並先問我。
- mono 開啟時 `--enforce-shared-experts-fusion` 關閉、`SGLANG_ROCM_MXFP8_AITER_PRESHUFFLE=0`（item 5 preshuffle 關），
  其他 opt1008 的 flag 照舊。

參考來源（只讀）：
- vLLM #60397（merged @193922d6）：`vllm/models/deepseek_v41/amd/mono/`、`amd/mono_decode.py`、tests
  `tests/models/test_dsv41_mono_*.py`。用 `gh api repos/vllm-project/vllm/contents/<path>?ref=193922d6 -H 'Accept: application/vnd.github.raw'`
  存到 /tmp/mono_vllm/。
- ATOM #2479：`/workspace/ATOM` 已有 `refs/remotes/pr/2479`（head 454d0e3，merge 0873517）。不要改 /workspace/ATOM 的
  working tree；用 `git -C /workspace/ATOM worktree add /tmp/atom-mono <ref>` 開唯讀 worktree。`gh` 讀不了 ROCm org
  （classic PAT 被拒），PR 頁面用 WebFetch。
- vLLM RFC #60904（open）：`gh api repos/vllm-project/vllm/issues/60904 -q .body`。

P0（先做完並寫進 doc 再進 P1）：
1. 逐一確認 MONO_KERNEL_EVAL.md 的 5 個 layout mismatch（KV record、dense 權重 layout、shared expert、routed MoE shuffle、
   DSpark rows），每項讀 SGLang 的 DSV4.1 HIP 程式（`python/sglang/srt/models/deepseek_v4.py`、
   `srt/models/deepseek_common/amd/*`、`srt/layers/attention/dsv4/*`、aiter MoE 路徑）與 ATOM/vLLM 的 `weights.py`，
   結論寫成表：一致 / 需 adapter / 需改 SGLang layout（附檔案:行號）。
2. 確認 ATOM 的 mono 能否以套件方式 import（`atom.mono` 與 V4.1 kernels 對 ATOM 其他模組的依賴、FlyDSL/aiter 版本需求
   vs 本機 aiter dsv41-opt-1008 與 flydsl 0.3.4.1）；列出 option C 需要的最小依賴。
3. 單獨跑 FFN-only launch 的 microbench（vLLM 或 ATOM 版本），用 SGLang TP4 的 shape，記錄 µs/layer 與 vs 現行
   SGLang 對應 kernel 序列（mHC AR+seam、router、sort、MoE、shared expert、AR）的時間。GPU 用前先確認空閒，
   只用 GPUs 4-7，不要和別人共用 GPU（persistent 256-CTA grid 會被 co-tenant 卡死）。

P1（P0 結論寫入 doc 後）：
- 在 SGLang 加最小的 mono runtime（`sglang/srt/layers/mono/`：attach / begin_step / forward_layer / peer memory /
  step epoch / health）與 DSV4.1 的 spec，先只接 FFN-only 模式（attention 照舊、其 output AR 關掉，kernel 做 AR+seam+MoE+AR），
  server arg 一個開關、預設 off；refusal 一次列完；不支援的 feature（TBO、DP attention、EP、PD、LoRA、hicache）啟動即拒絕。
- HIP graph：capture size <= 48 rows 的每個 width 都要先 build（compile 在 capture 前）。
- 驗證順序：per-stage numerics vs SGLang 原 ops → GSM8K TP4（EVAL_ONLY=true，確認 server log 沒有
  SGLANG_SIMULATE_ACC，>= 0.895）→ decode TPOT c1-c8（scripts/tp4_decode_profile.sh PROFILE=0）→ AgentX TP4 c1-c8
  （scripts/sweep_ci37423_opt.sh，CONCS/TSUF knob，一次只跑一個 server）並和 OPT_SWEEP_1008.md 的 opt1008 rows 比。

規則：
- sglang 在 /sgl-workspace/sglang 的新 branch（從 dsv41-opt-1008 開，例如 dsv41-mono-ffn），不要動 dsv41-opt-1008 本身。
- 結果、決策、下一步寫進 MONO_KERNEL_EVAL.md（CONTINUE HERE + append-only Log，每列帶 node + date）；
  log/trace 放 /shared_nfs/kk/results/DeepSeek-V4.1-Flash/mono/<tag>/。
- 不要 commit/push，除非我要求；commit message 不加任何 AI attribution（用 plumbing：write-tree / commit-tree）。
- 長任務背景跑、log 寫檔、只把 marker 帶回 chat；停 process 用 PID（不要 pkill -f 或含 script 名的 pattern）；
  執行中的 script 不要 in-place 改（寫新檔再 mv）；監看用輪詢（tail -F 在 NFS 上漏過事件）。
