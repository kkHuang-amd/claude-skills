# DeepSeek-V4-Pro FlyDSL-A2A 最終報告

日期：2026-07-30  
平台：8× AMD Instinct MI355X（gfx950）  
模型：DeepSeek-V4-Pro，A8W4 experts，FP8 KV  
拓撲：TP8 / DP8 / EP8，DP attention

## 1. 結論

FlyDSL-A2A 的本階段目標已完成。

最初在固定 8k input / 1k output、concurrency 256 workload 下，
FlyDSL-TBO 比 DP-TBO 慢 2.14%。完成 TBO overlap、geometry、shared eager
receive cap 與 fused-A2 後：

- 六組 counterbalanced fresh-server A/B 中，FlyDSL-TBO 比 DP-TBO 快
  1.913%，95% paired CI 為 1.494%–2.332%。
- fused-A2 再帶來 2.735% paired throughput uplift。
- fused-A2 最終點估計為 34,664.82 tok/s，比六組 DP-TBO 平均
  33,138.46 tok/s 高 4.61%；兩者不是同一組 interleaved A/B，因此
  4.61% 僅視為 point estimate。
- rebase 後四模式 regression screen 全數通過，沒有明顯效能退化。
- GSM8K 1,319 題 accuracy 0.937，invalid 0.000。

## 2. 效能演進

### 原始狀態

```text
DP-TBO        33,076.53 tok/s
FlyDSL-TBO    32,368.95 tok/s
FlyDSL gap        -2.14%
```

### Shared eager receive cap

兩組 counterbalanced OFF/ON pairs：

```text
cap OFF mean  32,737.22 tok/s
cap ON mean   33,751.42 tok/s
gain              +3.098%
TPOT              +2.225%
TTFT              +5.442%
```

### DP-TBO vs FlyDSL-TBO

六組 counterbalanced fresh-server pairs：

```text
DP-TBO mean              33,138.46 tok/s
FlyDSL-TBO mean          33,772.24 tok/s
FlyDSL advantage             +1.913%
95% paired CI      +1.494% to +2.332%
p-value                      3.93e-5
```

### Fused-A2

兩組 counterbalanced OFF/ON pairs：

```text
fused-A2 OFF mean        33,741.99 tok/s
fused-A2 ON mean         34,664.82 tok/s
throughput gain              +2.735%
TPOT improvement             +1.944%
TTFT improvement             +5.254%
```

### Workload matrix

```text
8k/1k c128     23,228.78 -> 23,597.17 tok/s   +1.59%
8k/1k c256     33,858.02 -> 34,772.93 tok/s   +2.70%
8k/1k c512     43,926.75 -> 45,154.43 tok/s   +2.79%
70k/300 c8     30,372.40 -> 31,266.21 tok/s   +2.94%
70k/300 c16    41,184.87 -> 42,797.90 tok/s   +3.92%
70k/300 c32    50,021.47 -> 51,806.45 tok/s   +3.57%
```

每個 matrix point 為單次 fresh-server run，沒有 confidence interval。

## 3. Rebase 後 regression screen

SGLang commit：

```text
18accbfe7fefb03adf2bc5791670db5c7e248813
```

為了與歷史數字保持相同 runtime family，測試使用相容的 Aiter feature
commit：

```text
bee50d97a7e74b796bdac8ef0247ecb6706132c3
```

固定 8k/1k、concurrency 256、每模式 2048 measured requests / 512 warmups：

```text
Mode          Current tok/s   Historical tok/s   Delta
DP               30,533.59          30,895.54   -1.17%
DP-TBO           33,005.46          33,138.46   -0.40%
FlyDSL           32,067.46          31,227.35   +2.69%
FlyDSL-TBO       34,387.60          34,772.93   -1.11%
```

四組皆完成 2048/2048 requests。約 1% 的下降仍在單次 run-to-run noise
範圍，沒有 confirmed regression。TBO 在本次 screen 中分別提升：

```text
DP       +8.10%
FlyDSL   +7.23%
```

Artifacts：

```text
/tmp/rebase_c256_regression/
```

## 4. 主要完成項目

### 4.1 FlyDSL-A2A backend

- 新增 ROCm `--moe-a2a-backend flydsl`。
- EP8 / DP8 local expert dispatch-combine。
- 支援 BF16 / FP8 / FP4 transport。
- 保持 symmetric-memory buffer 與 event lifetime。

### 4.2 TBO overlap

- FlyDSL dispatch/combine 移至 dedicated communication stream。
- 以 ready/done events 建立 compute/communication stream ordering。
- 兩個 TBO children 共用同一條 comm stream，避免錯序。
- 最終 geometry：

```text
GPU_MAX_HW_QUEUES=5
dispatch blocks=80
combine blocks=32
dedicated comm stream=enabled
```

### 4.3 Shared eager receive cap

每個 TBO child 使用所有 ranks 一致的 logical cap：

```text
recv_cap = min(
    physical_cap,
    next_pow2(sum(all-rank child padded rows)),
)
```

同一 cap 同時用於 dispatch encoding、Aiter views 與 combine decoding，
physical symmetric-memory allocation 不變。

### 4.4 Fused-A2

FlyDSL EP 實際 routed top-k 為 6，沒有 standard EP 的 fake-expert slot。
修正 tuning contract 後，Aiter 可選到 DSV4 production shape 的
FP8-output GEMM1，直接輸出 A2 FP8 與 e8m0 scale，移除第二次 dynamic
quant。

TBO trace hard signatures：

```text
dynamic quant calls       244 -> 122
A2 quant grid 294912      removed
A1 quant grid 114688      retained
activation/quant union    56.367 -> 18.066 ms
MoE union                 665.068 -> 615.521 ms
prefill stage span        902.095 -> 853.779 ms
```

Non-TBO trace：

```text
dynamic quant calls       122 -> 61
A2 quant grid 589824      removed
A1 quant grid 229376      retained
MoE union                 621.790 -> 569.707 ms
activation/quant/layer    915.45 -> 284.47 us
```

## 5. 主要 code paths

### SGLang

Repository / PR：

- [sgl-project/sglang PR #32726](https://github.com/sgl-project/sglang/pull/32726)

核心檔案：

```text
python/sglang/srt/layers/moe/token_dispatcher/flydslep.py
python/sglang/kernels/third_party/flydsl_a2a/
python/sglang/srt/batch_overlap/two_batch_overlap.py
python/sglang/srt/layers/moe/moe_runner/aiter.py
python/sglang/srt/managers/prefill_delayer.py
python/sglang/srt/model_executor/forward_batch_info.py
python/sglang/srt/server_args.py
python/sglang/srt/utils/bounded_telemetry.py
```

### Aiter

Repository / PR：

- [ROCm/aiter PR #4433](https://github.com/ROCm/aiter/pull/4433)

核心檔案：

```text
aiter/fused_moe.py
aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv
op_tests/test_dsv4_ep_fused_a2.py
op_tests/test_moe_tuning_topk.py
```

## 6. Correctness 與 validation

- GSM8K：accuracy 0.937，invalid 0.000。
- 8-GPU dynamic receive-cap full-cap parity：pass。
- 8-GPU two-child FlyDSL TBO dispatcher：pass。
- skewed routing 與 mixed empty/non-empty ranks：pass。
- dispatch/combine cap encoding consistency：pass。
- fused-A2 numerical coverage：

```text
M = 512 / 1024 / 4096 / 8192 / 16384 / 32768
```

- rebase 後四模式 serving regression screen：pass。
- pre-commit formatting hooks：pass。

## 7. Rejected experiments

以下方向經 correctness 與 production gates 後沒有 promotion：

- combine cross-PE waiter reduction；
- yielding remote poll；
- split dispatch data/coordinator kernels；
- comm priority；
- aiter-only receive slicing；
- dispatch 72 的細部 geometry；
-其他 completed geometry / HW queue screens。

短測改善不代表 serving critical path 改善；不要在沒有新證據時重跑。

## 8. Runtime controls

Production defaults：

```bash
export SGLANG_FLYDSL_DYNAMIC_RECV_CAP=1
export SGLANG_FLYDSL_DYNAMIC_RECV_CAP_EAGER=1
export SGLANG_FLYDSL_TBO_BLOCK_NUM=0
export SGLANG_FLYDSL_TBO_DISPATCH_BLOCK_NUM=80
export SGLANG_FLYDSL_TBO_COMBINE_BLOCK_NUM=32
export SGLANG_FLYDSL_TBO_USE_COMM_STREAM=1
export AITER_FLYDSL_EP_NO_FAKE_EXPERT=1
export GPU_MAX_HW_QUEUES=5
```

Rollback：

```bash
# Disable eager shared receive cap
export SGLANG_FLYDSL_DYNAMIC_RECV_CAP_EAGER=0

# Disable fused-A2 tuning contract
export AITER_FLYDSL_EP_NO_FAKE_EXPERT=0
```

## 9. 已知 runtime 注意事項

Aiter PR merge current main 後新增的 A8W8 decode-graph route，在目前 host
Triton/ROCm runtime 會於 CUDA graph capture 出現：

```text
RuntimeError: PassManager::run failed
```

這是與 FlyDSL-A2A / fused-A2 無關的 Aiter-main runtime blocker。為保持
與歷史 benchmark 可比，最終 regression screen 使用隔離的 premerge
Aiter `bee50d97a`，沒有關閉 CUDA graph，也沒有修改 kernel source。

## 10. 完整資料

- 詳細 handover：
  `A2A_TBO_GAP_HANDOVER.md`
- Backend umbrella：
  `A2A_EP_HANDOVER.md`
- 效能視覺化：
  `/root/.cursor/projects/sgl-workspace/canvases/flydsl-dp-tbo-performance.canvas.tsx`
- MoE trace 視覺化：
  `/root/.cursor/projects/sgl-workspace/canvases/moe-trace-timing.canvas.tsx`

## 11. 下一階段

FlyDSL-A2A branch 保持在 maintenance / review 狀態；新效能工作轉到
MegaMoE，評估 ROCm/FlyDSL PR #876 的新 MegaMoEV2、stage1/stage2
autotuning 與 fused transport 路徑。

MegaMoE 必須使用獨立 workspace 與 matched baselines，不要將實驗改動
放入 `/sgl-workspace/sglang-flydsl-a2a`。
