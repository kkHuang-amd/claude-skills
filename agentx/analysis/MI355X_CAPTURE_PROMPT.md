# Prompt: capture the MI355X decode trace for the cross-platform comparison

Paste the block below into an agent session on the MI355X node. It is
self-contained: paths, capture settings, the B200 reference numbers to compare
against, and every misreading that has already cost time on the B200 side.

Context for whoever is reading this file rather than the prompt: after matching
pdi, accept len, per-request KV working set and cuda-graph replay, the residual
is a stable 1.54-1.57× on *log-implied* step time. Log-implied includes
amortised prefill. On B200 only ~22 % of wall is inside decode steps
(TARGET_VERIFY p50 15.9 ms vs log-implied ~72 ms at batch 9). **The purpose of
this capture is first to report TARGET_VERIFY step wall p50** — that single
number decides whether the 1.55× is in decode kernels (~25 ms) or in the
prefill/waiting portion (~16 ms). Kernel roles are useful only after that.

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

## 要回報的數字（優先序就是這個，1 就能決定下一步）
1. **THE discriminator:** `TARGET_VERIFY` step wall **p50**, its `bs`, **and the
   KV working set of the capture window itself** (`#full token` ÷ batch from the
   same run's `server.log` at the capture timestamp). Always report the three
   together.
   **B200's 15.9 ms is RETRACTED** — it was captured mid-ramp at ~61k tok/req
   against a ~152k steady state, so it understates the real wall. Do not tune
   your capture to match it. Capture at steady state instead: per-request
   `#full token` ÷ batch ≥ ~130k and pool usage plateaued, verified from
   `server.log` *before* you trigger, not from a fixed sleep. B200 is
   re-capturing under the same rule and will publish a replacement.
2. The **unclassified kernel list** that `trace_summary.py` prints
   (`unclassified (add a ROLES pattern ...)`). ROCm names differ completely;
   without this, a large share of time may sit in `other`.
3. MoE kernel calls per step (B200: 61-64, one per layer).
4. Stream count active inside verify steps (B200 multi-stream 132, single-stream 4).
5. trace_summary TARGET_VERIFY roles, **p50** not mean (gemm / moe / attn /
   other / quant / comm / norm_rope), plus n / wall p50/mean.
6. trace_ranks TARGET_VERIFY cross-rank table (compute / barrier / attn / gemm).
7. decode_stats: running-req/rank, KV *absolute* working set
   (`#full token` / batch, not pool fraction), accept len, cuda-graph fraction,
   step_ms(batch) curve.
8. ISL mean of that arm (B200 pdi=24 is 114,401).

Write them into `claude-skills/agentx/exchange/mi355x-decode-trace.md` and
**commit + push** — the nodes have no shared filesystem.

## B200 的對照基準（pdi=24，TARGET_VERIFY bs=10，p50 ms/step）
**⚠ 這整組是在爬升期（~61k KV tok/req）抓的，僅供形狀參考，不是穩態基準。**
step wall 15.9 ms（已撤回）. Roles: gemm 19.39 / moe 12.22 / attn 6.58 / other 4.16 /
quant 1.82 / comm 0.77 / norm_rope 0.65. Summed kernel 24.55 ms inside a 15.9 ms
wall — that ratio is stream overlap, not utilisation. 132 streams.
`compute` is flat 15.05-15.06 ms across bs 2/5/10, so unequal-bs compute
comparisons are valid in this range.

Full table: `agentx/exchange/b200-decode-trace.md`.

已驗證：多 stream 重疊不會因搶 SM 而拉長個別 kernel（關掉後 kernel 加總不變、
只有 wall 變長 ~4-5 %），所以 per-kernel 時間可以直接跨平台比。

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
