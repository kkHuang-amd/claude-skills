# 階段 5：練習題（由簡到繁）

> 標籤說明：
> - 🟢 **離線**：純讀碼 / 紙上推導 / 寫原始碼，**不碰 GPU**，隨時可做。
> - 🔴 **需 GPU**：要編譯執行 / benchmark。**只在確認沒有效能測試在跑時批次做**。
>
> 建議：把所有 🟢 做完，累積一批 kernel 原始碼與設計，再找空檔一次驗證所有 🔴。

---

## Set A — Layout 代數（搭配階段 0）

- **A1** 🟢 給 `(8,16):(16,1)`、`(8,16):(1,8)`，各自寫出 `(i,j)` 的 offset 公式，並指出哪維連續。
- **A2** 🟢 寫一個 Python 函式 `crd2idx(crd, shape, stride)` 與 `idx2crd(idx, shape, stride)`（純 CPU、用一般 int），用幾組值自我驗證互為逆。
- **A3** 🟢 讀 `aiter/.../kernels/layout_utils.py`，解釋 `_div_pow2` / `_mod_pow2` 為何能取代 div/rem，以及它對 stride 的前提（2 的次方）。
- **A4** 🟢 在 `swiglu_and_mul.py` 裡找出所有 `// 和 %`，逐一標註它對應的 layout 座標分解（哪個是高維、哪個是低維）。

---

## Set B — FlyDSL 基礎與第一個 kernel（搭配階段 1）

- **B1** 🟢 默寫 `@flyc.kernel` + `@flyc.jit` 骨架（含 grid/block 與 `scf.IfOp` 邊界檢查）。
- **B2** 🟢 寫一個 elementwise kernel `out = x * 2 + 1`（f32），每 thread 一元素，含邊界檢查。用 `buffer_ops` 版與 `GTensor` 版各寫一次。
- **B3** 🟢 把 B2 改成向量化（每 thread 一次處理 4 個連續元素，`vec_width=4`），並寫出對應的 TV layout。
- **B4** 🔴 編譯執行 B2/B3，與 `torch` 的 `x*2+1` 比對數值（`torch.allclose`）。記錄首次編譯耗時與後續 dispatch 耗時差異。
- **B5** 🟢 解釋：若把 B2 的 `for i in range_constexpr(...)` 換成普通 runtime 迴圈，需要改用什麼（`scf.ForOp`）？兩者在 IR 上差在哪？

---

## Set C — Memory layout（搭配階段 2）

- **C1** 🟢 設計一個「把 `128×64` f16 tile 從 global 載入 LDS」的方案：256 thread、`LDG_VEC_SIZE=8`。寫出 `tid→(m,k)` 的 idx2crd、`LDG_REG_A_COUNT`，並用文字確認 coalesce。
- **C2** 🟢 手算 `swizzle_xor16(row, 0, k_blocks16=4)` 對 `row=0..7` 的結果，畫出哪些 row 共用同一 swizzle pattern。
- **C3** 🟢 寫一個 kernel：把一塊 tile 經 LDS「轉置」後寫回（global → LDS → 換 index 讀 → global）。先不 swizzle。
- **C4** 🟢 為 C3 加上 swizzle（寫與讀套同一函式），文字論證為何結果不變但 bank conflict 減少。
- **C5** 🔴 跑 C3/C4，用 profiler（`rocprof`/`omniperf`）看 LDS bank conflict 計數差異。
- **C6** 🟢 讀 `reduce.py` 的 `make_block_reduce_add`，畫出「intra-wave shuffle → LDS scratch → wave0 再規約」的資料流圖，標出每個 barrier 的必要性。

---

## Set D — MFMA layout（搭配階段 3，最重要）

- **D1** 🟢 默寫 `mfma_f32_16x16x16f16` 的三張 fragment 對應表（A: lane→(m,k)、B: lane→(n,k)、C: lane→(row,col)）。
- **D2** 🟢 反查練習：列出 lane = 0, 17, 33, 50, 63 各自持有的 C fragment 4 個元素的 `(row,col)`。
- **D3** 🟢 對照 `splitk_hgemm.py` 的 `ldmatrix_a_m_idx` / `ldmatrix_a_k_vec_idx` 與 `stmatrix_c_m_vec_idx` / `stmatrix_c_n_idx`，確認它們和你 D1 的表一致。
- **D4** 🟢 對照 `mfma_epilogues.py` 的 `default_epilog`，確認 `row = lane//16*4 + ii`、`col = lane%16`。解釋 `m_repeat` 與 `mi_base = mi*16` 的角色。
- **D5** 🟢 設計（不必跑）一個「單 warp、`16×16×16`、K 只有 16」的最小 MFMA：從 LDS 讀 A/B fragment → 一條 MFMA → 把 C fragment 寫回 global。寫出完整原始碼。
- **D6** 🔴 編譯執行 D5，與 `torch.matmul` 的 `16×16 @ 16×16` 比對。
- **D7** 🟢 把 D5 擴展到 `WARP_M=32, WARP_N=32, WARP_K=16`（4 個 atom 的雙層迴圈），寫出 `WARP_M_STEPS/WARP_N_STEPS` 迴圈。

---

## Set E — 整合與優化（搭配階段 4）

- **E1** 🟢 設計一個最小 single-block GEMM（`SPLIT_K=1, STAGES=2`, 同步 load, 無 scheduler hints）的參數推導：給定 `TILE_M=64, TILE_N=64, TILE_K=32`，算出所有 warp 分工與 load 參數，列出所有要滿足的整除約束。
- **E2** 🟢 把 E1 寫成完整 kernel（可大量參考 `splitk_hgemm.py` 的非 async 分支 L705–755，但簡化掉 split-K）。
- **E3** 🔴 編譯執行 E2，與 `torch.matmul` 比對（多種 M/N/K）。
- **E4** 🟢 在 E2 上「設計」multi-stage 改造：寫出 prologue 預載、主迴圈 `for...yield` 帶 `c_frags` 的結構（先不跑）。
- **E5** 🔴 把 E4 跑起來，比較 STAGES=2 vs 3 的效能（**確認無其它測試時**）。
- **E6** 🟢 為 E2 的 hot loop 設計 scheduler hints（`sched_vmem/dsrd/mfma` + `sched_barrier`），寫出順序並說明理由。
- **E7** 🔴 用 `@autotune` 對 E2 掃幾組 tile 形狀 / STAGES，記錄最佳 config（**獨佔 GPU 時**）。

---

## Set F — 讀懂 production kernel（綜合）

- **F1** 🟢 完整讀一遍 `splitk_hgemm.py`，畫出整體資料流圖（global→LDS→fragment→MFMA→epilogue→global），標註每段對應本計劃哪一章。
- **F2** 🟢 解釋 `split_k_barrier` 的 spin-wait + semaphore + atomic 機制，以及為何需要先 `zero_c`。
- **F3** 🟢 挑一個 attention kernel（`fused_compress_attn.py` 或 `flash_attn_func_gfx1201.py`），找出它的 MFMA 形狀、online softmax 的 running max/sum 維護方式。
- **F4** 🟢 比較 `splitk_hgemm.py`（手寫 layout）與高階 atom 路徑（`MFMA()` + `partition_*`），各列 3 個優缺點。

---

## 進度追蹤表

| Set | 主題 | 🟢 離線完成 | 🔴 GPU 驗證完成 |
|-----|------|:-----------:|:---------------:|
| A | Layout 代數 | ☐ | — |
| B | FlyDSL 基礎 | ☐ | ☐ |
| C | Memory layout | ☐ | ☐ |
| D | MFMA layout | ☐ | ☐ |
| E | 整合與優化 | ☐ | ☐ |
| F | 讀 production kernel | ☐ | — |

---

## 完成標準（畢業檢核）

當你能做到以下，就算掌握了 FlyDSL 的核心：

1. 看到任何 kernel 裡的 `//`/`%`/`*stride`，能立刻說出它對應哪個 layout 的 crd2idx/idx2crd。
2. 能默寫 `16×16×16 f16` MFMA 的 A/B/C fragment 對應。
3. 能從零寫出一個正確的 single-block GEMM（含 LDS、MFMA、epilogue）。
4. 能讀懂 `splitk_hgemm.py` 的每一段，並說明它屬於哪個優化階段。
5. 能規劃一條優化路線並用 autotune 驗證（在獨佔 GPU 時）。
