# Prompt: capture the MI355X decode trace for the cross-platform comparison

Paste the block below into an agent session on the MI355X node. It is
self-contained: paths, capture settings, the B200 reference numbers to compare
against, and every misreading that has already cost time on the B200 side.

Context for whoever is reading this file rather than the prompt: after matching
`prefill_decode_interval` on both platforms, the ITL gap is a stable ~1.6×
(MI355X 33.9 ms vs B200 20.7 ms at pdi=24; 50.6 vs 32.1 at pdi=10). The purpose
of this capture is to locate that 1.6× in the per-role kernel breakdown.

---

```text
在 MI355X 這台抓 AgentX c128 的 decode trace，目的是和 B200 已有的資料做逐角色對比,
找出配置對齊後仍存在的 1.6x ITL 差距落在哪裡。

## 先讀
/mnt/home/wunhuang/claude-skills/agentx/analysis/METHOD.md
這份是平台中立的分析方法，裡面記了所有已知陷阱。工具在同一個目錄：
trace_common.py（角色分類，跨平台比較的基礎）、trace_ranks.py、trace_summary.py、
decode_stats.py、show_result.py。四個 py 檔要放在一起，trace_common.py 是共用模組。

## 要抓什麼
c128、prefill_decode_interval=24（和 B200 對齊，也就是產生「B200-aligned」那一列的同一組設定）。
先確認 $RESULT_DIR/sglang_command.txt 裡真的是 --prefill-decode-interval 24，
不要只看腳本意圖 -- 那個檔案在 server 啟動「之前」就寫好了，它存在不代表 server 起來了。

## 擷取設定必須和 B200 完全一致，否則 profiler 干擾量不同就不能比
{"activities":["CPU","GPU"],"num_steps":40,"profile_by_stage":true,
 "record_shapes":true,"with_stack":false}

- SGLANG_TORCH_PROFILER_DIR 必須在 server 啟動前就 export 好。
- DP attention 開啟時 launcher 會把 router 放在 $PORT、server 放在 $PORT+1
  （dsv4_fp4_mi355x_sglang_mtp.sh:151）。/start_profile 只存在於 server，要打 $PORT+1。
- 觸發時機不要用固定 sleep。等 launcher log 出現 done=（warmup 結束、進 profiling 階段）
  再 settle 120 秒才觸發。warmup 長度會隨 page cache 冷熱變化，任何常數都會落在錯誤時機。
- 健康檢查用 /health_generate 而不是 /health。
- 風險：B200 上 num_steps=40 會讓 fused MoE kernel 內的 device-side grid sync 超時而殺掉
  server（deep_gemm barrier.cuh:45 "Grid sync timeout"）。trace 會成功寫出、但 run 之後會死。
  ROCm 的 aiter MegaMoE 是否有同樣的 device-side barrier 未知，請留意；
  trace 落地後就算 run 死掉也沒關係，agg 指標用先前完整那輪的即可。

## 分析（照這個順序）
python3 analysis/trace_ranks.py   <trace 目錄>          # 先看跨 rank 的 compute vs barrier
python3 analysis/trace_summary.py <trace 目錄>/*TP-0-*.gz
python3 analysis/decode_stats.py  <server.log>

## 要回報的數字
1. trace_summary 的 TARGET_VERIFY 區塊：n、step wall p50/mean、以及逐角色的
   ms/step 與 min/p50/max（gemm / moe / attn / other / quant / comm / norm_rope）。
   要 p50，不要只給平均 -- 分布是雙峰的，平均會被近乎空的 step 拉低。
2. trace_ranks 的 TARGET_VERIFY 跨 rank 表（compute / barrier / attn / gemm）。
3. step 標註的清單（TARGET_VERIFY / EXTEND / IDLE 各幾個）以及 bs 值。
   B200 那份全部是 bs=7、61 層、每 step 61-64 次 MoE 呼叫。
4. decode_stats 的 running-req/rank、kv pool usage、accept len、cuda graph 比例、
   以及 step_ms 依 batch 分桶的曲線。
5. 該輪的 ISL mean（B200 在 pdi=24 下是 114,401，pdi=10 是 109,796；吞吐比較對 ISL 敏感）。
6. verify step 內活躍的 GPU stream 數量（B200 multi-stream 是 132 條、single-stream 是 4 條）。

## B200 的對照基準（pdi=10 抓的，bs=7，TARGET_VERIFY 的 p50 ms/step）
gemm 21.1 / moe 13.7 / attn 5.2 / other 3.4 / quant 1.3 / comm 0.6
step wall p50 17.2 ms、每 step kernel 加總 45 ms（重疊 1.49x）、132 條 stream。
已驗證：多 stream 重疊不會因搶 SM 而拉長個別 kernel（關掉後 kernel 加總不變、
只有 wall 變長），所以 per-kernel 時間可以直接跨平台比，不需要為 stream 做正規化。

## 判讀上必須避免的錯誤（這些都是在 B200 上實際踩過的）
- 檔名不是證據。profile_by_stage 會把檔案命名成 -DECODE，內容卻可能只有一個 EXTEND step。
  一定先看 step[...] 標註。
- GPU step 標註是巢狀的：一個約 30 ms 的 step 底下有數百個同名的約 0.4 ms 子切片。
  不去巢狀就會把 62 個真實 step 數成 2077 個。trace_common.load() 已處理，
  並可用 CPU 側的 user_annotation 數量交叉驗證。
- kernel 時間加總除以 wall 不是 busy 比例。kernel 跨 stream 重疊，這個比值會超過 100%
  （實測 17 ms 的 step 裡有 25.7 ms 的 kernel）。真正的利用率只能從 achieved bytes
  或 FLOPs 對 peak 算。
- 單一 rank 的 top-kernel 排名在 DP 部署下不能當歸因。fused MoE a2a kernel 會吸收
  等待最慢 rank 的時間，所以輕載 rank 上它可以佔 90% 而幾乎沒做事。要跨 rank 互比。
- 不要拿部分 run 對完整 run 比，即使 batch 對齊；還要對齊 kv pool usage。
- 死掉的 server 會繼續回 200，warmup barrier 會無限等待而 errors=0。
  判斷存活看 returned= / done= 有沒有在動，crash 只寫進 server.log 不寫 launcher stdout。

## 收尾
launcher 結束後會留下一整串孤兒程序（launch_server、sglang::router、
data_parallel_controller、detokenizer、每個 rank 一個 tokenizer_worker、加上各 rank 的
scheduler），而且釋放 VRAM 不等於節點乾淨 -- 母程序會繼續佔住 dist-init port，
導致下一輪啟動 30 秒後死於 "port_base ... is not available"。
清完要同時驗程序數、VRAM 和 port 監聽三項。
```
