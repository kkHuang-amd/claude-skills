Owner node: crsuse2-m2m-049

# sglang AgentX regression -- new image, c1 re-baseline (2026-10-02)

本目錄自成一套：runner、matrix、points、compare、CI baseline 都是從 `../` 複製來的，
`SKILL_DIR` 解析到本目錄，所以和 075 正在用 `../` 跑的 matrix 互不影響（凍結副本各自獨立）。
流程與陷阱見 `../REGRESSION_RUNBOOK.md`；075 的紀錄見 `../E2E_REBUILD.md`（owner 075，不要改）。

## CONTINUE HERE

**Status（17:10，完成）：** 回歸 = #41019 的 compressor wkv_gate GEMM（HIP 上 tgemm bf16 改成 `torch.mm(out_dtype=fp32)`，decode 小 M 時慢 1.3–2×）。
#41931（只改 compressor）完全補回：c1 配對 per-request ITL，HEAD 比 PA 慢 2.6 %，B1 比 PA 快 1.3 %；server decode 213.5 => 221.6（PA 219.0）。
c4 **不可 bisect**：同一個 c73f 跑兩次，tok/s 差 8.1 %、intvty 差 5.2 %（雙峰：約 590 vs 約 560 個請求），雙指標 bisect 17:08 以「DONE: no c4 gap」結束。
GPU 已釋放；沒有任何背景工作。fix branch `fix/dsv4-compressor-tgemm` 尚未 commit。
**雜訊檢查（14:35 加入）：** B0 之後先在 049 重跑 c73f c4（075 同 code 只有 3,425，而 049 是 3,640）；同 code 差距 ≥ gap/2 就停止 bisect，不做。
判準：tok/s/GPU ≥ mid+band 為 good、≤ mid-band 為 bad，中間就重跑一次、取平均和 mid 比；mid、band 由 Phase A c4 與 B0 c4 算出。
**決策紀錄：** `BISECT_C4.log`（本目錄）。**斷線恢復：** 直接重新啟動同一個指令（結果依 TAG 快取，已完成的會跳過）：
`cd /workspace && setsid nohup bash /workspace/claude-skills/agentx/regression/bisect_c4.sh < /dev/null > /workspace/results/bisect_c4.driver.log 2>&1 &`
（有 flock，不會重複執行；用 `flock -n /tmp/bisect_c4.lock true && echo idle` 檢查。**不要 `pkill -f bisect_c4`**：會連同自己的 shell 一起殺掉。）
**Done (crsuse2-m2m-049):** c1 `TAG=sglc73f-049`（舊複製腳本）、c4/c16 同 TAG（`e2e/matrix.sh` 快照）。結果 `/workspace/results/e2e-{c1,c4,c16}-sglc73f-049/`。
**Next:** 和使用者討論：(1) 同事的 c4 gap 是否就是請求數的雙峰；(2) 推動 #41931 merge（附 B1 c1 數據）；(3) 長期是否要做 aiter K=7168 bf16->fp32 tuned 列（方案 B）以保留 fp32 精度；(4) 合併 agentx 目錄（等 075）。
**Status file:** `/workspace/results/e2e-matrix-sglc73f-049.status`（用 wc -l 輪詢，NFS 不要 tail -F）
**Repro (resume; done points skipped):**
```bash
R=/workspace/claude-skills/agentx/regression; cd /workspace
TAG=sglc73f-049 POINTS=c1 PYTHONPATH=/sgl-workspace/sglang-ci/python SGL_DIR=/sgl-workspace/sglang-ci \
  setsid nohup bash $R/.agentx_e2e_matrix.frozen.sh < /dev/null \
  > /workspace/results/e2e-matrix-sglc73f-049.driver.log 2>&1 &
```
**Compare:**
```bash
R=/workspace/claude-skills/agentx/regression
python3 $R/compare_ci.py /workspace/results/e2e-c1-sglc73f/*_agentic.json /workspace/results/e2e-c1-sglc73f-049/*_agentic.json   # vs 075
python3 $R/compare_ci.py $R/ci_baseline_36401947630/conc1_kvnone_ep1-dpafalse/*.json /workspace/results/e2e-c1-sglc73f-049/*_agentic.json  # vs CI
```
**Pass criteria:** gates OK（errors 0、duration ≥95 %、ISL 差 ≤3 %）；P90 intvty 差 < 6.6 %
（c1 主看 intvty；c1 的 tok/s/GPU 對速度不敏感）。

**Restructure (user, 2026-10-02):** 已建 `../e2e/`（共用環境，matrix 改為自動快照到 `$RESULTS/e2e-matrix-<TAG>.code/`）與 `../legacy/`（空，舊檔先不搬）。
待 075 matrix 與本 c1 跑完：本目錄移到 `../tasks/regression/` 並刪掉複製的腳本；`E2E_REBUILD.md` 移到 `../tasks/rebuild-075/`；刪 `../` 的舊 e2e 腳本與 runbook 拆到 `../e2e/README.md`。

## Stack on crsuse2-m2m-049 (2026-10-02)

- sglang: worktree `/sgl-workspace/sglang-ci` @ `c73f7077eb`（detached），rust `_multimodal*.so`
  從 `/sgl-workspace/sglang` 複製。image 自帶 sglang 是 `0.5.21.dev20261001+g3b2ad1c6ae`
  （工作樹 HEAD `41cbe65de0`），本次不用。
- aiter: `e7d2453f25` + 6 個本地修改（075 是 `acf8fdf93` + 3 個）：
  - 與 075 相同的 3 個：`pa_mqa_logits_fp4_prefill.py`（`@cache`）、`csrc/cpp_itfs/torch_utils.py`、
    `hsa/gfx950/mla_v4/mla_a8w8_qh64_qseqlen1_gqaratio64_nm.co`（換過的 ASM）
  - 多出 3 個（dsv41 工作留下）：`dsv41_fp4_{tuned,untuned}_fmoe.csv`、
    `aiter/ops/triton/attention/pa_decode_sparse.py`（Triton <3.8 一律 unpeel）
- InferenceX `cb45b6da0b`、aiperf `754356e9a3`、venv `/workspace/agentx-runtime-e2e`（共用 NFS）。

## Results (append-only, node + date per row)

| node | date | point | stack | tok/s/GPU (ref) | P90 intvty (ref) | gates | verdict |
|---|---|---|---|---|---|---|---|
| crsuse2-m2m-075 | 2026-10-02 | c1 | sglang c73f7077eb, aiter acf8fdf93+3 | 2,079.0 | 220.0 | OK | 參考值（取自 ../E2E_REBUILD.md） |
| crsuse2-m2m-049 | 2026-10-02 | c1 | sglang c73f7077eb, aiter e7d2453f25+6（new image） | 2,093.2（075 +0.7 %，CI +0.7 %） | 219.3（075 -0.3 %，CI +1.2 %） | OK，244 succ、ISL +0.4 % | **aligned**：新 image/aiter 不影響 c1 |
| crsuse2-m2m-049 | 2026-10-02 | c4 | same（e2e/matrix.sh） | 3,640.3（CI 3,647.6，-0.2 %） | 191.7（CI 188.5，+1.7 %） | OK，589 succ（CI 592）、ISL +0.05 % | **aligned**；server 未掛（075 的 c4 掛過） |
| crsuse2-m2m-049 | 2026-10-02 | c16 | same（e2e/matrix.sh） | 11,187.8（CI 11,210.2，-0.2 %） | 95.9（CI 96.6，-0.7 %） | OK，2,304 succ（CI 2,303）、ISL -0.3 % | **aligned** |

## Phase B 計畫：PR #41019 的 GEMM 回歸（2026-10-02，crsuse2-m2m-049）

**線索（同事）：** 回歸從 sgl-project/sglang#41019（`effb752188`，2026-09-27 merged）開始。
#41019 在 `python/sglang/kernels/ops/gemm/bf16_fp32.py` 的 `linear_bf16_fp32` 拿掉了
`_use_aiter and bf16 -> tgemm.mm(x, y, otype=bf16).float()`，HIP 改走 `torch.mm(out_dtype=fp32)`。
修正 #41931（open，+7/-1）只在 `dsv4/compressor.py::_compute_wkv_gate` 把 tgemm 加回來。

**呼叫點（HEAD `41cbe65de0`）：** DSV4-Pro 只有 `compressor.py:457`（wkv_gate）受影響；MoE gate 在
HIP 走 `aiter_dsv3_router_gemm`，不經過這裡。`dsv41_sparse.py:158/159/164` 只給 V4.1 用，#41931 沒蓋到。
c73f7077eb 不含 #41019，HEAD 含。

**Microbench**（`bench_wkv_gate.py`，GPU0，CUDA graph，K=7168；us/call）：

| M | N=2048 tgemm / mm_f32 | N=1024 | N=512 |
|---|---|---|---|
| 7 | 8.6 / 15.4 | 7.8 / 13.7 | 7.8 / 10.2 |
| 16 | 8.9 / 11.5 | 7.2 / 11.0 | 7.0 / 10.7 |
| ≥64 | 持平 | 持平 | 持平 |

- decode 小 M 時 mm_f32 慢 1.3–2×；M≥64 與 prefill 持平。
- `tgemm.mm(otype=fp32)` 速度和 mm_f32 一樣（tuned 表只有 bf16 輸出），所以快的前提是 bf16 輸出
  （和 fp64 參考比 max abs err ~0.04；fp32 輸出 ~0）。#41019 拿掉它多半就是為了這個精度。
- eager（不在 graph 內）時 tgemm 每次呼叫有 ~90 us 的 CPU 分派開銷，所以只在 graph 內才佔優勢。
- 估算 c1（M=7）：30×N2048 + 31×N1024 + 30×N512 ≈ 91 次/forward，多出約 0.46 ms/verify step；
  step 約 3.77×4.4 ≈ 16.6 ms，所以預期 c1 P90 intvty 掉約 3 %。

**測試步驟**（每步 c1 3600 s ≈ 75 分；只比 049 Phase A 的列）：
1. B0：HEAD 原樣（`/sgl-workspace/sglang` editable，不設 PYTHONPATH），c1，確認回歸幅度。
2. B1：HEAD + #41931，c1。回到 Phase A 雜訊內 = 主因確認。
3. B2（B1 沒補滿才做）：`effb752188^` vs `effb752188` 的 c1，切出 #41019 本身佔多少，
   其餘部分在 c73f..HEAD 之間 bisect（#41019 的 fp4 indexer HIP、compress、kv layout 改動也在
   DSV4-Pro 路徑上，`--enable-deepseek-v4-fp4-indexer`）。
4. 定案後 c4/c16 各跑一次，確認沒有在其他點造成回歸。

### B0 啟動 + 修正方案調查（2026-10-02，crsuse2-m2m-049）

**B0 running：** `TAG=head41cbe-049 POINTS=c1 SKIP_SMOKE=1`，sglang `/sgl-workspace/sglang` @`41cbe65de0`
（editable，不設 PYTHONPATH），11:04:43 開始，預計 ~12:15。status `/workspace/results/e2e-matrix-head41cbe-049.status`。

**Alan 的 commit**（akao-amd/sglang `064155bc`）：在 `bf16_fp32.py` 加一條 gfx1250 專用分支，直接呼叫
aiter Triton/gluon `gemm_a16w16(x, y, dtype=fp32)`（fp32 累加、fp32 輸出，保住 #41019 要的精度）。
gfx950 刻意不動，理由是 aiter#5681 已在 gfx950 為這類 bf16->fp32 projection 經 `tgemm` tune 過。

**但對 DSV4-Pro（K=7168）不成立：**
- aiter#5681（`001cd7792`，已在 aiter HEAD）讓 `tgemm.mm(otype=fp32)` 真的輸出 fp32，但只新增了
  V4.1 的 K=5120 tuned 列。
- `dsv4_bf16_tuned_gemm.csv` 中 K=7168、N∈{512,1024,2048} 的列**全部是 bf16 輸出**（各約 90 列，
  flydsl/opus/asm/triton/torch）。所以 fp32 輸出會 fall back 到 torch，等於現在的 `torch.mm`
  （microbench 的 tg32_g ≈ mm_g 證實了這點）。
- `gemm_a16w16` 在 gfx950 走的是非 gluon 的 Triton kernel（`_GLUON_SUPPORTED_ARCHS=("gfx1250",)`），速度未知。

**修正選項（都在 `bf16_fp32.py`，一處涵蓋 compressor + dsv41_sparse）：**
- A. 照 #41931 的作法，在 compressor 用 tgemm 的 bf16 輸出：最快，等同 #41019 之前的行為，但輸出是 bf16 精度。
- B. `linear_bf16_fp32` 在 HIP 上走 `tgemm.mm(otype=fp32)`，並在 aiter 補 K=7168 的 bf16->fp32
  tuned 列（ASM `bf16gemm_fp32bf16_*` / OPUS 有 fp32 輸出）：速度和精度都保住，但要改 aiter 並重新 tune。
- C. 把 Alan 的 `gemm_a16w16(dtype=fp32)` 分支擴展到 gfx950：要先量 gfx950 上的 Triton 速度。

**下一步（B0 結束、GPU 空出後）：** microbench 加上 C（`gemm_a16w16` fp32），在 M=1..128 比較
A / 現況 / C；再依結果選方案，在 `/sgl-workspace/sglang` 開新 branch 實作，跑 c1（B1），並做 GSM8K 精度檢查。
| crsuse2-m2m-049 | 2026-10-02 | c1 | **B0** sglang HEAD 41cbe65de0（含 #41019），aiter 同上 | 2,058.0（049 Phase A -1.7 %） | 214.0（-2.4 %） | OK，242 succ、ISL -0.8 % | 雜訊內，但 ITL mean 4.35→4.50 ms（+3.4 %），和 microbench 估的 ~3 % 一致 |

### 選項 C microbench 結果（12:20，GPU0，CUDA graph，us/call）

C = `gemm_a16w16(dtype=fp32)` 在 gfx950 上走一般 Triton kernel（未 tune）：

| M | N=2048 A / 現況 / C | N=1024 | N=512 |
|---|---|---|---|
| 7 | 8.7 / 15.3 / 12.5 | 7.8 / 13.6 / 12.0 | 7.8 / 10.0 / 11.3 |
| 28 | 10.2 / 11.0 / 14.0 | 8.3 / 10.5 / 16.6 | 7.3 / 10.3 / 16.4 |
| 112 | 15.4 / 15.8 / 31.5 | 11.2 / 11.1 / 23.7 | 9.3 / 8.2 / 23.9 |
| 1024 | 41.6 / 40.4 / 156.6 | 28.0 / 29.5 / 157.2 | 21.9 / 19.4 / 157.6 |

**結論：C 在 gfx950 不可用。** 只有 M≤8、N≥1024 時比現況略快，仍比 A 慢 1.4–1.6×；M≥28 就比現況還慢，
M=1024 慢 4–8×（會拖慢 prefill／大 batch）。精度：C 的 err ~0（fp32），A 的 err ~0.04（bf16 輸出）。
剩下 A（#41931 的 bf16 輸出）或 B（aiter 補 K=7168 的 bf16->fp32 tuned 列，再走 `tgemm.mm(otype=fp32)`）。

### B1 running（2026-10-02，crsuse2-m2m-049）

使用者決定：只做 A（= #41931，只改 compressor；`bf16_fp32.py` 不動，因為 #41019 是 V4.1 PR，
它要的 fp32 不能被全域改掉）。branch `fix/dsv4-compressor-tgemm` @ `/sgl-workspace/sglang`
（base 41cbe65de0，套 `gh pr diff 41931`，**未 commit**；工作樹原本的 pyproject 修改保留）。
`TAG=fix41931-049 POINTS=c1 SKIP_SMOKE=1`，12:22:13 開始，matrix pid 40350 @049，預計 ~13:32。
判讀：B1 vs B0 看 ITL mean 與 P90 intvty；預期回到 Phase A（ITL ~4.35 ms、intvty ~219）。
**改為 c4（user, 12:25）：** B1 c1 已停（`e2e-c1-fix41931-049.void-stopped`）。HEAD 的 c4 還沒有數字，
所以兩個 c4 串在一起跑：`/workspace/results/run_c4_fix_then_head.sh`（chain pid 44515 @049，12:27:37 開始）
1. B1 c4：`TAG=fix41931-049`，fix branch（editable `/sgl-workspace/sglang`），預計 ~13:40
2. B0 c4：`TAG=head41cbe-049`，worktree `/sgl-workspace/sglang-head` @41cbe65de0（PYTHONPATH），預計 ~14:50
比較：三者都對 049 Phase A c4（3,640.3 / 191.7）。microbench 估這個 GEMM 在 c4（M≈28）只造成約 1 %；
c4 的 gap 若明顯更大，代表還有其他原因，要做 B2。
| crsuse2-m2m-049 | 2026-10-02 | c4 | **B1** fix branch（HEAD 41cbe65de0 + #41931） | 3,374.3（049 Phase A -7.3 %，rerun 區間） | 185.4（-3.3 %） | OK，但 560 succ（PA 589）、ISL -1.9 % | 延遲指標與 PA 相同（ITL mean +1 %、e2el -2 %、TTFT +1 %）；tok/s 的差來自完成的請求較少，server 負載分布與 idle gap（≥5 s 共 818 s vs 773 s）幾乎相同，不像 server 變慢；待 B0 c4 |
| crsuse2-m2m-049 | 2026-10-02 | c4 | **B0** HEAD 41cbe65de0（worktree sglang-head） | 3,683.3（049 PA +1.2 %） | 183.0（-4.5 %） | OK，594 succ、ISL +0.5 % | **tok/s 沒有 gap**；但 ITL mean 4.70→4.92 ms（+4.7 %）、ITL p90 +4.6 %。B1 的 ITL 是 4.75（+1 %）：修正補回大部分 decode 延遲。B1 的 tok/s -7 % 是雜訊（完成的請求較少），不是回歸 |

### B0 c4 結論（14:45）

c4 的 tok/s/GPU：PA 3,640 / B1 3,374 / B0 3,683 —— HEAD 沒有掉，B1 那次 -7 % 是雜訊（請求數 560 vs 589/594）。
**c4 上真正的訊號是 decode 延遲：** ITL mean PA 4.70 / B0 4.92（+4.7 %）/ B1 4.75（+1 %），c1 也一樣（PA 4.35 / B0 4.50）。
方向和幅度都符合 #41019 的 wkv_gate GEMM，#41931 補回大部分。所以 tok/s/GPU 不適合當 bisect 指標。
bisect 腳本仍在跑 c73f c4 重跑（`sglc73f-049-r2`，14:45 開始）。由於 gap 為負，雜訊檢查之後腳本會以「too noisy」的 ABORT 結束；
訊息措辭不準，實際原因是 c4 沒有 tok/s gap。這次重跑仍有用：它能量出同節點、同 code 下 ITL/intvty 的差距，用來判斷 B0 的 +4.7 % 是不是真的。

### intvty 調查（15:10）：server 速度 vs. 請求組成

**server 端**（server.log 的 `Decode batch` gen throughput，依 `#running-req` 分組）：
- c4 batch 1（約 75 % 的 decode 時間）：PA 225.9 / B0 218.7（-3.2 %）/ B1 225.6（-0.1 %）
- c4 batch 2：386.0 / 390.3 / 387.1（持平）；c1：PA 219.0 / B0 213.5（-2.5 %）
=> 修正後 server decode 速度回到 PA。

**每個請求配對**（`paired_itl.py`，依 conversation_id+turn_index 配對，同一 trace = 同樣的請求）：
- aiperf 的「intvty p90」是**慢尾**：1/(ITL p90)，等於 1/ITL 的 p10。
- c1 B0 vs PA：ITL 比值 median 1.026，p10–p90 落在 1.004–1.051，很集中 => c1 是乾淨的指標，B0 回歸是真的（+2.6 %）。
- c4 B0 vs PA：median 1.029，slow tail -4.6 %，p50 -2.8 %。
- c4 B1 vs PA：median **0.989**、p50 **+1.4 %**，但 slow tail **-3.0 %**；每個請求的比值很分散（p10 0.85、p90 1.16）。
  => c4 同一個請求在不同 run 之間的 ITL 可以差 ±15 %（受併發重疊、prefill 插隊影響）；慢尾由「哪些請求剛好被干擾」決定。
**判斷：** B1 的 slow tail -3 % 很可能是 c4 的時序雜訊，不是殘餘的回歸；用 c73f c4 重跑（`sglc73f-049-r2`）配對 PA 驗證：
PA 自己的 slow tail 若也會飄約 3 %，就是雜訊。另外排 B1 c1（乾淨的指標）直接確認修正：`/workspace/results/run_b1_c1_after_bisect.sh`（等 bisect lock 釋放後啟動，TAG=fix41931-049，預計 ~17:20 完成）。
| crsuse2-m2m-049 | 2026-10-02 | c4 | **c73f 重跑**（同 Phase A code） | 3,355.6（PA -7.8 %） | 182.0（-5.1 %） | OK，559 succ | **同 code 就差 8 %**：c4 的 tok/s 與 intvty 雜訊大於要找的 gap |

### c4 是雙峰分布（16:00）——tok/s「gap」來自 workload 路徑，不是 code

| run | code | succ | tok/s/GPU | intvty P90 |
|---|---|---|---|---|
| CI | c73f（image） | 592 | 3,647.6 | 188.5 |
| 049 PA | c73f | 589 | 3,640.3 | 191.7 |
| 049 B0 | HEAD | 594 | 3,683.3 | 183.0 |
| 075 | c73f | 562 | 3,425.0 | 184.9 |
| 049 PA-r2 | c73f | 559 | 3,355.6 | 182.0 |
| 049 B1 | HEAD+#41931 | 560 | 3,374.3 | 185.4 |

- 兩群：約 590 個請求 => 約 3,650 tok/s；約 560 個請求 => 約 3,370–3,425。**同樣是 c73f，兩群都出現過。**
  所以 c4 的 tok/s「-7 %」是 client 走了另一條 workload 路徑（請求數少約 5 %），不是 sglang 回歸。
- PA vs PA-r2 配對：每個請求的 ITL median 比值 **0.9996**（code 一樣，per-request 速度完全一樣），但 slow-tail intvty -4.7 %、tok -7.8 %。
- intvty 要在同一群裡比：590 群 PA 191.7 / CI 188.5 / HEAD 183.0（HEAD 低約 3–4 %，和 GEMM 回歸一致）；
  560 群 c73f 184.9 / 182.0，B1 185.4（修正版和 c73f 一樣）。
- 舊版 bisect 15:53 以 ABORT 結束（同 code 差距 8.1 %）。新版雙指標 bisect 會在 B1 c1 之後啟動，預期會立刻以
  「DONE: no c4 gap」結束：tok gap 是負的；intvty 的 good 平均是 186.9，bad 是 183.0，gap 2.1 % < 4 %。**c4 不可 bisect。**
- **結論：** 可量到的回歸只有 decode 速度（c1 per-request ITL +2.6 %、server decode -2.5～-3.2 %），#41931 可以補回（c4 server 端已確認）。
  B1 c1（約 17:20）是最後的確認。同事看到的 c4 gap 要確認是不是這種請求數的雙峰（比對 succ 數）。
| crsuse2-m2m-049 | 2026-10-02 | c1 | **B1** fix branch（HEAD + #41931） | 2,124.9（PA +1.5 %） | 218.2（-0.5 %） | OK，246 succ | **修正確認**：配對 ITL median 0.987（p10–p90 0.970–1.014），B0 是 1.026；server decode 221.6（PA 219.0 / B0 213.5） |
