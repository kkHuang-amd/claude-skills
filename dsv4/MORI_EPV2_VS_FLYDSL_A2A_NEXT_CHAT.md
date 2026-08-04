# MORI EPv2 FlyDSL A2A vs standalone FlyDSL-A2A — 新 chat handoff

把以下 prompt 貼到新的 Cursor chat：

---

請在 8×MI355X 上整合並比較兩種 FlyDSL A2A implementation：

1. 已完成的 SGLang standalone FlyDSL-A2A backend；
2. MORI EPv2 cco-LSA + FlyDSL dispatch/combine。

來源：

- [ROCm/mori PR #448](https://github.com/ROCm/mori/pull/448)
- MORI merge commit：`060c682f`
- SGLang standalone FlyDSL-A2A：
  `/sgl-workspace/sglang-flydsl-a2a`

目標不是只比較 dispatch/combine microbench，而是判斷在
DeepSeek-V4-Pro A8W4、EP8/DP8、8k/1k concurrency 256 serving workload
下，哪個 backend 的 end-to-end throughput、TTFT、TPOT 較好。

## 先讀文件

```text
/dockerx/home/wunhuang/tmp/claude-skills/dsv4/FLYDSL_A2A_FINAL_REPORT_2026-07-30.md
/dockerx/home/wunhuang/tmp/claude-skills/dsv4/A2A_EP_HANDOVER.md
/dockerx/home/wunhuang/tmp/claude-skills/dsv4/A2A_TBO_GAP_HANDOVER.md
/dockerx/home/wunhuang/tmp/claude-skills/dsv4/FLYDSL_KERNEL_OPT_PLAYBOOK.md
```

## 重要設計差異

不要把兩者描述成只有 kernel source 不同：

```text
standalone FlyDSL-A2A
  FlyDSL kernels
  mori-shmem symmetric heap / P2P provider
  SGLang backend: --moe-a2a-backend flydsl

MORI EPv2
  FlyDSL kernels
  mori-cco LSA flat symmetric VA
  peer pointer via cco.Window(...).lsa_ptr(...)
  currently needs SGLang adapter
```

因此：

- communication-only microbench 可比較 kernel/provider latency；
- serving 結果比較的是完整 backend/system design；
- 不可將差異只歸因於 FlyDSL codegen，除非有直接量測。

## Current MORI state

Local repository：

```text
/sgl-workspace/mori
```

PR #448 已 merge，但後續已有重要更新：

- `10febdf1`：將 `dispatch_combine_v2` 變成 installable package；
- `4890fd9d`：MI355X vec4 combine retune；
- `a238c2ad`：asymmetric fp8 dispatch + bf16 combine；
- `0abea937`：vec4 gather loads；
- `dafdcfcf`：inner-unroll + tuned geometry default；
- MORI `origin/cco_ep_opt` 另有：
  - `d2594c63` readlane tok_map decode；
  - `faf611a4` block-0-only xdb barrier；
  - `60384723` drop redundant acquire fence。

不要只 checkout 原始 merge commit 就宣稱代表目前 MORI。

第一輪使用最新 `origin/main` 作為可發布 baseline；將
`origin/cco_ep_opt` 三個額外 commits 視為獨立 candidate arm，不要混入
main baseline。

## Workspace isolation

現有 checkout 可能 dirty 或 detached。不要覆蓋任何工作：

```text
/sgl-workspace/sglang-flydsl-a2a
/sgl-workspace/mori
/sgl-workspace/aiter
```

建立 isolated worktrees：

```text
/tmp/sglang-mori-epv2-compare
/tmp/mori-epv2-main
/tmp/mori-epv2-opt              # 只有需要測 cco_ep_opt 時建立
```

SGLang integration 必須從 standalone FlyDSL-A2A branch 的最新 clean
commit 開始，因為它已包含：

- `--moe-a2a-backend flydsl`；
- PrefillDelayer mixed-slot guard；
- dynamic receive cap；
- TBO stream/event ordering；
- dispatch80/combine32 geometry；
- fused-A2 tuning contract。

記錄所有 exact commits，不要依賴 branch name。

## 必須保留的 DP scheduler fix

在所有 DP / MORI / FlyDSL benchmark 前確認存在：

```text
2d4cf9bd2  fix: preserve delayer slot protection after KV checks
```

涉及：

```text
python/sglang/srt/managers/prefill_delayer.py
test/registered/scheduler/test_prefill_delayer.py
```

正式測試：

```bash
export SGLANG_PREFILL_DELAYER_MIXED_SLOT_GUARD=1
```

先跑 unit test，並確認
`mixed_slot_guard_does_not_timeout_under_slot_pressure` 通過。沒有此 fix
時，scheduler regression 會污染 DP、MORI 與 FlyDSL serving 數字。

## Runtime freeze

開始前輸出 manifest：

- SGLang commit；
- MORI main / candidate commit；
- Aiter commit；
- FlyDSL Python package path、version、commit；
- PyTorch / Triton / ROCm；
- GPU architecture、CU count；
- model path；
- 所有 performance env。

standalone backend 已驗證的 historical runtime：

```text
SGLang  18accbfe7
Aiter   bee50d97a
FlyDSL  0.2.4
```

這些是 reference，不是強制 pin。MORI current main 若需要更新 runtime，
必須找一個兩個 backends 都能載入的 common runtime；若無法共用：

1. 明確標記 runtime version confound；
2. 先用 common-shape communication microbench；
3. end-to-end 結果只能稱為 system comparison，不能純歸因於 A2A kernel。

目前 merged Aiter main 在 host 上曾有 unrelated A8W8 CUDA-graph
`PassManager::run failed`。不要用關閉 CUDA graph 作為正式 workaround。

## Integration goal

新增獨立 backend identity，例如：

```text
--moe-a2a-backend mori-epv2
```

不要偷偷取代既有 `mori` 或 `flydsl` backend，因為需要同一 binary/source
下做三臂比較：

```text
mori          existing MORI v1
mori-epv2     cco-LSA + FlyDSL kernels
flydsl        standalone FlyDSL-A2A
```

建議做法：

1. 以 `moriep.py` 的 SGLang dispatcher contract 為外部 interface；
2. 使用 packaged
   `mori.ops.dispatch_combine_v2.EpDispatchCombineOp/Config`；
3. 保留獨立 backend enum、server args、dispatcher construction；
4. 不 vendor MORI EPv2 source 到 SGLang；
5. 使用 current MORI package/worktree PYTHONPATH；
6. 維持 routing handle lifetime、CUDA graph lifetime 與 symmetric window
   lifetime；
7. 先做 synchronous non-TBO path，再做 async/TBO。

若 current MORI package export 名稱與上面不同，讀 source 確認，不要猜 API。

## Routing / GEMM fairness gate

DeepSeek-V4-Pro production shape：

```text
hidden size       7168
routed topk       6
EP world size     8
local experts     48
expert inter dim  3072
```

PR #448 原始公開數字多為 topk=8；current MI355X tuning table已有
`(world=8, hidden=7168, topk=6)`，必須確認 runtime 實際命中該 row。

比較前驗證：

- routed topk 是否為 6；
- 是否有 fake-expert slot；
- dispatch output rows、indices、weights、scales；
- expert mask / routing invariance；
- receive cap 與 physical allocation；
- Aiter selected GEMM1/GEMM2 kernel pair；
- fused-A2 是否兩個 backends都實際生效。

不要全域硬設：

```text
AITER_FLYDSL_EP_NO_FAKE_EXPERT=1
```

除非已證明 MORI EPv2 與 standalone FlyDSL 都沒有 fake slot。若兩邊選到
不同 GEMM1/GEMM2，serving 結果同時改變了 A2A 與 expert GEMM，不能
歸因為 A2A backend win。此時：

1. 先修正/對齊 tuning contract；或
2. 將 communication-only 與 end-to-end system comparison 分開報告。

## Correctness gates

### MORI upstream

在 8 GPU 先跑：

```bash
cd /tmp/mori-epv2-main/tests/python/ops/dispatch_combine_v2
pytest -v test_dispatch_combine_v2_intranode.py
```

再以 DSV4 production shape 跑：

```text
EP8
hidden=7168
topk=6
tokens/rank=8,64,128,256,512,1024,2048,4096,8192
```

至少驗證：

- bf16 dispatch + bf16 combine；
- production candidate dtype；
- routing weights / indices；
- per-token scales；
- empty rank / skewed routing；
- max receive cap；
- replay / CUDA graph；
- repeated lifecycle create/destroy；
- gather combine。

FP4/FP8、scatter、blockwise quant 是額外 arms，不可與第一輪 production
baseline 混在一起。

### SGLang integration

- CPU/unit tests；
- 8-GPU identity-expert round trip；
- one-request skewed DP attention；
- prefill、decode、mixed batch；
- CUDA graph capture/replay；
- 512-request smoke；
- successful request count；
- routing/mask/kernel signature。

任何 correctness mismatch 先修正，不做效能結論。

## Phase 1：communication-only matched microbench

先比較 production-relevant non-TBO communication：

```text
Implementation
1. MORI v1
2. MORI EPv2 main
3. standalone FlyDSL-A2A

Shape
EP8 / hidden 7168 / topk 6
tokens per rank: 8,64,128,256,512,1024,2048,4096,8192
dtype: bf16 dispatch + bf16 combine first
mode: eager and CUDA graph, separately reported
```

每一臂使用自己的 validated production tuning，但要記錄實際：

- dispatch blocks / warps；
- combine blocks / warps；
- selected per-token schedule；
- actual received rows；
- dispatch/combine latency；
- correctness error；
- graph/eager mode。

PR #448 在 MI355X 的原始趨勢是 dispatch 約 1.02–1.27× faster、combine
大致 parity；這只是 hypothesis/reference，不是本機結論。

## Phase 2：non-TBO serving comparison

先做 non-TBO，才能隔離 backend 本身；不要拿 MORI EPv2 non-TBO 與
standalone FlyDSL-TBO 比較。

固定：

```text
model          DeepSeek-V4-Pro
hardware       8×MI355X
topology       TP8 / DP8 / EP8
workload       8192 input / 1024 output
concurrency    256
prompts        2048
warmups        512
CUDA graph     ON
radix cache    OFF
delayer        ON + mixed-slot guard
fresh server   every arm
```

Arms：

```text
A. DP non-TBO sanity baseline
B. MORI v1
C. MORI EPv2 main
D. standalone FlyDSL-A2A
```

建議 counterbalanced order：

```text
B -> C -> D
D -> C -> B
```

若差異 <1%，至少跑 3 pairs，報 paired CI；不要以單次 run 宣稱 win。

Primary metrics：

- total/output tok/s；
- TTFT；
- TPOT / ITL；
- successful requests；
- server errors；
- selected A2A geometry；
- selected GEMM pair；
- quant-call count。

Historical standalone reference：

```text
FlyDSL non-TBO  32,067.46 tok/s
```

只當 sanity range，正式結論使用本輪 matched runs。

## Phase 3：TBO comparison

只有 MORI EPv2 non-TBO correctness 與 serving 通過後才做 TBO。

需要為 MORI EPv2 補齊與 standalone backend 相同的：

- dedicated communication stream；
- ready/done event ordering；
- both TBO children 共用相同 comm stream；
- event + Python-reference lifetime，避免 `record_stream()` allocator
  fragmentation；
- per-child rank-consistent receive cap；
- dispatch/combine independent geometry；
- bounded telemetry。

比較：

```text
MORI EPv2 non-TBO
MORI EPv2 TBO
standalone FlyDSL non-TBO
standalone FlyDSL-TBO
```

standalone production TBO controls：

```bash
export GPU_MAX_HW_QUEUES=5
export SGLANG_FLYDSL_TBO_BLOCK_NUM=0
export SGLANG_FLYDSL_TBO_DISPATCH_BLOCK_NUM=80
export SGLANG_FLYDSL_TBO_COMBINE_BLOCK_NUM=32
export SGLANG_FLYDSL_TBO_USE_COMM_STREAM=1
```

這些 geometry 不可直接套到 MORI EPv2；先使用 MORI tuned schedule，再以
小範圍 screen 找 overlap-friendly geometry。

Historical standalone references：

```text
FlyDSL-TBO six-pair mean       33,772.24 tok/s
FlyDSL-TBO fused-A2 mean       34,664.82 tok/s
post-rebase single run         34,387.60 tok/s
```

仍只當 sanity range。

## Phase 4：optional MORI optimization branch

只有 EPv2 main 已有有效 baseline 後，才測：

```text
origin/cco_ep_opt
```

將 `d2594c63 / faf611a4 / 60384723` 視為一個 separate candidate，或逐一
cherry-pick 做單變因測試。不要把它們悄悄混進 main arm。

若 microbench win 沒有轉成 serving win，先檢查：

- prefill/decode token-size distribution；
- communication critical-path union；
- GEMM overlap inflation；
- host/API copy；
- symmetric-window allocation/lifecycle；
- receive padding；
- graph fixed overhead。

不要只加總重疊 kernel durations。

## Attribution rules

- stall site 不是 root cause；
- “允許更多 occupancy” 不等於實際達成；
- 改變 geometry 後要確認實際 selected schedule；
- 宣稱 bandwidth/compute/occupancy bound 前必須量 achieved utilization；
- peak hardware spec 不可代替 achieved result；
- microbench win 不等於 serving win；
- 每個實驗一次只改一個變因。

## Decision gates

- correctness failure：停止效能結論；
- MORI EPv2 main non-TBO ≥1% 且 pairs 一致：進 TBO integration；
- <1%：增加 pairs，不先 promotion；
- microbench win、serving null：做 component trace，不直接深度調 kernel；
- serving regression：先排除 GEMM、recv cap、dtype、scheduler confounds；
- c256 winner 再擴展 c128/c512 與 70k/300；
- 最終 candidate 跑完整 GSM8K。

## Deliverables

1. exact revision/runtime/environment manifest；
2. integration changes與 backend contract；
3. upstream MORI correctness結果；
4. DSV4 production-shape correctness；
5. MORI v1 / MORI EPv2 / standalone FlyDSL microbench；
6. non-TBO counterbalanced serving結果；
7. TBO結果與 overlap-aware trace；
8. promoted / rejected / blocked方向；
9. 獨立 Markdown handover；
10. 效能 Canvas。

完整工作期間不要 commit、push 或修改 PR，除非使用者明確要求。

---

## 參考

- [ROCm/mori PR #448](https://github.com/ROCm/mori/pull/448)
- `FLYDSL_A2A_FINAL_REPORT_2026-07-30.md`
- `A2A_EP_HANDOVER.md`
