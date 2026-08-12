# MegaMoE PR #876 — 新 chat handoff

把以下 prompt 貼到新的 Cursor chat：

---

請開始 DeepSeek-V4-Pro MegaMoE 新一輪效能評估。

目標是評估 [ROCm/FlyDSL PR #876 — MegaMoE on Gfx950](https://github.com/ROCm/FlyDSL/pull/876)
最新的 MegaMoE / MegaMoEV2 優化，在 8×MI355X 上是否改善 serving
效能，尤其是 8k/1k concurrency 256。

## 先讀文件

```text
/dockerx/var/amdsgl/kk/workspace/claude-skills/dsv4/FLYDSL_A2A_FINAL_REPORT_2026-07-30.md
/dockerx/var/amdsgl/kk/workspace/claude-skills/dsv4/megamoe/MEGAMOE_HANDOFF.md
/dockerx/var/amdsgl/kk/workspace/claude-skills/dsv4/megamoe/MEGAMOE_OPT_DIRECTIONS_HANDOVER.md
/dockerx/var/amdsgl/kk/workspace/claude-skills/dsv4/megamoe/FLYDSL_MEGAMOE_STAGE1_ANALYSIS.md
/dockerx/var/amdsgl/kk/workspace/claude-skills/dsv4/megamoe/EXPERIMENT_LOG.md
```

## Workspace isolation

MegaMoE 工作使用：

```text
FlyDSL: /sgl-workspace/FlyDSL
SGLang: /sgl-workspace/sglang
```

不要修改：

```text
/sgl-workspace/sglang-flydsl-a2a
```

該 branch 屬於已完成的 standalone FlyDSL-A2A backend。

## 必須先移植的 DP / PrefillDelayer fix

遠端 `HaiShaw/sglang:feat/mega-moe` 最新檢查點
`22faf9fef8048731863fb64c68cae2ab42b9fa4f` 尚未包含 FlyDSL-A2A branch
上的 PrefillDelayer mixed-slot guard。**在跑任何 DP、MegaMoE 或
PR #876 效能比較前，必須先移植並驗證此 fix。**

來源 commit：

```text
2d4cf9bd2  fix: preserve delayer slot protection after KV checks
```

舊 rebase 前 lineage 曾記為 `9b0eb8a55`；以目前 FlyDSL-A2A branch 的
`2d4cf9bd2` 為準。

涉及兩個檔案：

```text
python/sglang/srt/managers/prefill_delayer.py
test/registered/scheduler/test_prefill_delayer.py
```

Root cause：

- upstream `d03c8cee8` 將 PrefillDelayer negotiation 移到 KV-budget
  admission checks 之後；
- 高負載 DP step 因部分 ranks KV admission 失敗，從原本的 `all`
  branch 進入 `mixed` branch；
- generic mixed timeout 會在 request slots 仍不足時釋放 prefill，
  犧牲 decode protection，造成 TPOT / throughput regression；
- 這是 scheduler regression，不是 DP、MegaMoE 或 MoE kernel 本身
  的效能。

Fix contract：

- 將每個 scheduler 的 `max_running_requests` 加入 DP all-gather；
- 在 mixed state 直接使用所有 ranks 的
  `running_batch / max_prefill_bs / max_running_requests` 判斷 slot
  pressure；
- slot 不足時持續 delay，不進入 generic timeout；
- 保留 upstream post-KV admission correctness；
- rollback env：

```bash
export SGLANG_PREFILL_DELAYER_MIXED_SLOT_GUARD=0
```

正式 benchmark 必須使用：

```bash
export SGLANG_PREFILL_DELAYER_MIXED_SLOT_GUARD=1
```

移植時不要直接覆蓋 MegaMoE dirty worktree。先建立 clean branch /
isolated worktree，再 cherry-pick 或手動 port 這兩個檔案，處理與最新
PrefillDelayer API 的差異。

移植後至少驗證：

```bash
python3 -m pytest -q test/registered/scheduler/test_prefill_delayer.py
```

並確認新增 case
`mixed_slot_guard_does_not_timeout_under_slot_pressure` 通過。之後先跑一組
DP 8k/1k c256，確認數字回到約 30.5k–30.9k tok/s 的 matched runtime
範圍，才開始 MegaMoE old/new A/B。

## 背景基準

舊 compact-only MegaMoE matched serving：

```text
workload       8192 input / 1024 output
concurrency    256
MegaMoE        29,482 tok/s
DP             30,501 tok/s
MegaMoE / DP   96.7%
GSM8K          0.934
```

不要直接把不同 SGLang/Aiter/FlyDSL runtime 的數字當成 regression。
更新 PR #876 後，先在同一 source/runtime 上重跑 fresh-server matched
baseline。

## PR #876 新方向

PR 描述與近期 commits 顯示至少包含：

- MegaMoEV2；
- per-1x32 quantization；
- fused dispatch/sort/GEMM1；
- fused GEMM2 + weighted P2P combine；
- fixed-slot 與 compact dispatch；
- SBM-aware joint tuning；
- synchronized multi-rank autotuning；
- reusable tuning artifacts；
- FP8 persistent GEMM2 configs；
- stage1/stage2 tuning 與 tuned config path 修正。

舊結論「沒有低風險 perf lever」只適用於當時的 compact-only
implementation。PR #876 改變了 kernel architecture 與 tuning space，
必須重新量測，不能沿用舊結論。

## 第一階段：freeze 與 bring-up

1. 檢查 GPU、現有 terminal/process、兩個 workspace 的 branch/status。
2. 用 `gh` 讀 PR #876 最新 head、commits、changed files 與 CI。
3. 記錄 exact FlyDSL / SGLang / Aiter / Triton revisions。
4. 建立 isolated SGLang worktree，先移植並驗證 PrefillDelayer
   mixed-slot guard。
5. 不覆蓋未提交變更。
6. 驗證 PR #876 import path、MegaMoE API、tuning directory 與 runtime
   version。
7. 先跑最小 8-GPU correctness smoke，再啟動完整 server。

## 必須保留的 correctness gotchas

- DSV4 A8W4 `gate_mode=interleave`：
  w1 必須使用 `shuffle_weight_w4(gate_up=True)` 與
  `shuffle_scale_w4`。
- `MORI_SHMEM_MODE` 使用 STATIC_HEAP；不要設 ISOLATION。
- 先前 MTPR 8192 是安全 baseline；16384 曾有 int32 buffer sizing
  overflow。
- CUDA graph 必須保持開啟，除非只做明確標記的 diagnostic。
- 每個新 kernel/tuning arm 都要驗證 routing、mask、numerics 與
  successful request count。

## 第二階段：matched benchmark

先跑最小、可歸因的矩陣：

```text
Backend / arm
1. DP baseline
2. old compact-only MegaMoE（若可在同一 runtime 重現）
3. PR #876 default MegaMoE
4. PR #876 MegaMoEV2 / selected tuned path

Workload
8192 input / 1024 output
concurrency 256
2048 measured requests
512 warmups
fresh server per arm
```

若 PR #876 default 已完全取代舊 path，先以 exact commit worktree
保留 old/new A/B，不要在一個 arm 同時更換 SGLang、FlyDSL 與 tuning
artifact。

Primary metrics：

- total/output tok/s；
- TTFT；
- TPOT / ITL；
- successful requests；
- selected kernel/config signatures。

若單次差異小於約 1%，至少重跑 counterbalanced pairs，不要直接宣稱
regression 或 win。

## 第三階段：只有在 serving 結果需要時才 profile

先看 full-serving outcome，再決定 trace：

- stage1 dispatch/sort/GEMM1；
- activation/quant；
- stage2 GEMM2/combine；
- compact vs fixed-slot；
- persistent vs non-persistent GEMM2；
- prefill 與 decode 分開；
- exact interval union，不相加重疊 kernel duration。

任何「compute-bound / bandwidth-bound / occupancy-bound」結論都必須有
achieved utilization 或直接 outcome 證據。不要把 stall site 當成
root cause，也不要把允許更高 occupancy 當成實際達成。

## Decision gates

- correctness failure：先修 correctness，不做效能結論；
- serving uplift ≥1% 且重跑一致：進入擴展矩陣；
- microbench win、serving 無改善：不 promotion；
- 8k/1k c256 通過後，再跑 c128/c512 與 70k/300；
- 最終候選跑完整 GSM8K。

## Deliverables

1. exact revision / environment manifest；
2. old vs PR #876 matched results；
3. kernel/config signature proof；
4. regression 或 uplift 的 counterbalanced evidence；
5. 更新 MegaMoE handoff 與獨立效能 Canvas；
6. 清楚列出 promoted、rejected 與未驗證方向。

---

## 參考

- [ROCm/FlyDSL PR #876](https://github.com/ROCm/FlyDSL/pull/876)
- [SGLang FlyDSL-A2A PR #32726](https://github.com/sgl-project/sglang/pull/32726)
- [Aiter fused-A2 PR #4433](https://github.com/ROCm/aiter/pull/4433)
