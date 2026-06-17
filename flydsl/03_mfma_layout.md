# 階段 3：AMD MFMA layout（難點 2）

> 目標：徹底搞懂 AMD CDNA 的 MFMA（Matrix Fused Multiply-Add）指令的 **fragment layout**——也就是一個 warp（64 lane）裡，**哪條 lane 持有矩陣的哪個元素**。這是 GEMM kernel 最難、也最關鍵的部分。
>
> 本章是純粹的「硬體規定 + 座標對應」，可完全離線推導。

---

## 3.1 為什麼 MFMA layout 特別難

和前面兩個難點不同：
- Layout 代數（難點 1）、memory layout（難點 3）是**你設計**的，你有自由度。
- **MFMA layout 是硬體寫死的**。指令 `mfma_f32_16x16x16f16` 規定了：A、B、C 三個矩陣的元素**必須**以特定方式分散在 64 條 lane 的暫存器裡。你沒有選擇，只能去**配合**它——把資料從 LDS 讀成它要的排列，計算完再把結果從它的排列寫回去。

所以難點 2 的本質是：**記住（或查表）MFMA 的固定 fragment layout，並寫出 lane → (row, col) 的對應公式。**

---

## 3.2 MFMA 指令命名與形狀

命名規則 `mfma_<accT>_<M>x<N>x<K><inT>`：

```
mfma_f32_16x16x16f16
      │    │  │  │ └ 輸入型別 f16
      │    │  │  └── K = 16
      │    │  └───── N = 16
      │    └──────── M = 16
      └───────────── 累加/輸出型別 f32
```

語意：`D[M,N] += A[M,K] × B[K,N]`，一個 warp（64 lane）一條指令算完一個 `16×16×16` 的小 tile。

工作區用到的（`splitk_hgemm.py`）：

| arch | 指令 | M,N,K | A/B frag | C frag |
|------|------|-------|----------|--------|
| gfx942 | `mfma_f32_16x16x16f16` | 16,16,16 | `vec(4,f16)` | `vec(4,f32)` |
| gfx942 | `mfma_f32_16x16x16bf16_1k` | 16,16,16 | `vec(4,i16)`（bf16 bitcast） | `vec(4,f32)` |
| gfx950 | `mfma_f32_16x16x32_f16` | 16,16,32 | `vec(8,f16)` | `vec(4,f32)` |
| gfx950 | `mfma_f32_16x16x32_bf16` | 16,16,32 | `vec(8,bf16)` | `vec(4,f32)` |

低階呼叫（`flydsl.expr.rocdl`）統一簽名：

```python
rocdl.mfma_f32_16x16x16f16(result_type, [a, b, c, cbsz, abid, blgp])
# a,b 是輸入 fragment 向量；c 是累加器；cbsz/abid/blgp 一般填 0
```

工作區把它包成 `WmmaHalf_m16n16k16` 等 class（`splitk_hgemm.py` L46），呼叫 `WMMA_IMPL(a_frag, b_frag, c_frag)`。

---

## 3.3 16×16×16 f16 的 fragment layout（最重要，背起來）

wave size = 64。對 `mfma_f32_16x16x16f16`：

### A 矩陣（16×16，每 lane 持 4 個 f16）
- `lane` 在 0..63
- A 的 layout：`lane` 拆成 `(m, k_group)`：
  - `m = lane % 16`（0..15，對應 A 的 row）
  - `k_group = lane // 16`（0..3）
  - 該 lane 的 4 個元素對應 `K` 維的 `k_group*4 + (0..3)`

對照 `splitk_hgemm.py` 的 `ldmatrix` 索引：

```python
ldmatrix_a_m_idx     = w_tid % WMMA_M                       # = lane % 16  → m
ldmatrix_a_k_vec_idx = w_tid // WMMA_M * WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K   # → k 起點
```

所以「為了餵 A 給 MFMA」，每條 lane 要從 LDS 讀回 `(row = m, col = k_group*4 ...)` 的 4 個連續 f16。這 4 個就是 A fragment。

### B 矩陣（16×16，每 lane 持 4 個 f16）
對稱地：

```python
ldmatrix_b_n_idx     = w_tid % WMMA_N                       # = lane % 16  → n
ldmatrix_b_k_vec_idx = w_tid // WMMA_N * WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K   # → k 起點
```

### C / D 矩陣（16×16 累加器，每 lane 持 4 個 f32）—— epilogue 的關鍵
這是寫回結果時要用的對應，工作區明確寫在 `mfma_epilogues.py` 的 `default_epilog`：

```
lane → (row, col):
  lane_div_16 = lane // 16   ∈ {0,1,2,3}
  lane_mod_16 = lane % 16    ∈ {0..15}    → 對應 C 的 column (n)
  該 lane 的 4 個 f32 (ii=0..3) 對應 C 的 row:
      row_in_tile = lane_div_16 * 4 + ii
```

`mfma_epilogues.py` L73–82 的 `default_epilog`：

```python
lane_div_16_mul4 = lane_div_16 * 4
for mi in range_constexpr(m_repeat):          # tile 內第幾個 16-row 區塊
    mi_base = mi * 16
    for ii in range_constexpr(4):             # 該 lane 的 4 個累加元素
        row_off = lane_div_16_mul4 + ii       # = lane//16 * 4 + ii
        row_in_tile = mi_base + row_off
        row = bx_m + row_in_tile              # 全域 row
        body_row(mi, ii, row_in_tile, row)
```

對照寫回 LDS 的 `splitk_hgemm.py` L758：

```python
stmatrix_c_m_vec_idx = w_tid // WMMA_N * WMMA_C_FRAG_VALUES   # = lane//16 * 4
stmatrix_c_n_idx     = w_tid % WMMA_N                          # = lane%16
# 第 kk 個累加元素寫到 LDS 的 (m = ...+kk, n = ...)
```

> **這張表（C fragment 的 lane→(row,col)）是 GEMM epilogue 的命脈**。`row = lane//16*4 + ii`、`col = lane%16` 要能默寫出來。

---

## 3.4 完整資料流：LDS ↔ fragment ↔ MFMA

把階段 2 和本章縫起來，一個 warp 算一個 tile 的流程：

```
        LDS (A tile, swizzled)              LDS (B tile, swizzled)
              │  ldmatrix 讀                      │  ldmatrix 讀
              │  lane→(m, k)                      │  lane→(n, k)
              ▼                                   ▼
        a_frag: vec(4,f16)                  b_frag: vec(4,f16)
              └───────────────┬───────────────────┘
                              ▼
              c_frag = MFMA(a_frag, b_frag, c_frag)   # vec(4,f32)
                              │  (在 K 迴圈中累加)
                              ▼
                  epilogue: lane→(row=lane//16*4+ii, col=lane%16)
                              ▼
                  寫回 LDS (CShuffle) 或直接寫 global
```

K 維大於 16 時，外層用 `WARP_K_STEPS`、`WARP_M_STEPS`、`WARP_N_STEPS` 迴圈，把多個 `16×16×16` MFMA atom 拼成 warp 負責的 `WARP_M × WARP_N` 大 tile（`ldmatrix_compute_tile_streaming`）。`MFMA_PER_WARP_K=2`（gfx942）時，一次讀進的 fragment 會 bitcast 拆成兩半，連續發兩條 MFMA（`splitk_hgemm.py` L609–642）。

---

## 3.5 兩條路徑：手寫 vs 高階 atom

### 路徑 A：手寫 fragment 索引（工作區 GEMM 的做法）
就是上面 `w_tid % 16`、`w_tid // 16 * 4` 這些。優點：完全可控、好做 swizzle 與 scheduler 微調。缺點：要自己記 layout、容易寫錯。

### 路徑 B：高階 MMA atom（CuTe 風格）
FlyDSL 提供 `MFMA(...)` atom + tiled MMA，自動處理 fragment layout：

```python
from flydsl.expr.rocdl.universal import MFMA
mma_ty  = MFMA(16, 16, 16, fx.Float16)        # 描述 MFMA 形狀/型別
atom    = make_mma_atom(mma_ty)
tiled   = make_tiled_mma(atom, atom_layout)   # 多 atom 拼大 tile
thr     = tiled.get_slice(thread_idx)         # 這條 thread 的分區
a_frag  = thr.partition_A(a_tensor)           # 自動算 fragment（= local_partition）
b_frag  = thr.partition_B(b_tensor)
c_frag  = tiled.make_fragment_C(...)
gemm(atom, c_frag, a_frag, b_frag, c_frag)
```

優點：layout 由代數自動推導，少出錯。缺點：抽象層較厚，極致優化時不如手寫靈活。

> 學習建議：**先用路徑 A 把 16×16×16 的 lane→(row,col) 三張表（A/B/C）親手推一遍**，理解硬體規定；之後再學路徑 B 的 atom 抽象，會發現它只是把這些表自動算出來。工作區 production kernel 多用路徑 A。

---

## 3.6 CDNA4 / scaled MFMA（進階，先知道有）

- `mfma_scale_f32_16x16x128_f8f6f4`、`wmma_scale_f32_16x16x128_f8f6f4`：FP8/FP6/FP4 含 per-block scale 的 MFMA（`rocdl/__init__.py`、`rocdl/cdna4.py` 的 `MFMA_Scale`）。
- WMMA（RDNA）是 wave32 的對應指令，fragment layout 與 CDNA 的 MFMA 不同，`WMMA(...)` 會依 arch 分派（gfx11 vs gfx12）。
- 入門先專注 gfx942 的 `16x16x16 f16`，把一個搞透，其它形狀只是「換 M/N/K 與 frag 長度」。

---

## 3.7 手算練習（離線，最重要的一章練習）

針對 `mfma_f32_16x16x16f16`，wave=64：

1. **C fragment 反查**：lane 37 持有的 4 個 f32，分別對應 `16×16` 輸出 tile 的哪 4 個 `(row, col)`？
2. **A fragment**：lane 20 的 4 個 f16 對應 A `16×16` 的哪些 `(m, k)`？
3. 一個 warp 要算 `WARP_M=32, WARP_N=32, WARP_K=16` 的 tile，需要幾個 `16×16×16` MFMA atom？`WARP_M_STEPS`/`WARP_N_STEPS` 各多少？
4. 把 epilogue 寫回的 `row = lane//16*4 + ii`、`col = lane%16`，用階段 0 的 layout 語言寫成一個 `(thread, value) → (row, col)` 的 TV layout（給出 shape 與 stride 的概念）。

<details>
<summary>參考答案</summary>

1. lane 37：`lane//16 = 2`，`lane%16 = 5`。col = 5；row = `2*4 + ii` = 8,9,10,11。即 `(8,5),(9,5),(10,5),(11,5)`。
2. lane 20：`m = 20%16 = 4`，`k_group = 20//16 = 1` → k = `1*4 + (0..3)` = 4,5,6,7。即 `(4,4),(4,5),(4,6),(4,7)`。
3. `32/16 * 32/16 = 2*2 = 4` 個 atom（K=16 一層）。`WARP_M_STEPS=2`, `WARP_N_STEPS=2`。
4. 座標 `(lane, ii)`：`row = (lane//16)*4 + ii`，`col = lane%16`。可拆成巢狀：lane = `(n_part=lane%16, m_part=lane//16)`；row 維 shape `(4 [m_part], 4 [ii])` stride `(4,1)`，col 維 shape `(16)` stride `(1)`。

</details>

---

## 3.8 自我檢核

- [ ] 我能默寫 `16×16×16 f16` 的 C fragment：`row = lane//16*4 + ii`、`col = lane%16`
- [ ] 我能寫 A/B fragment 的 `lane → (m,k)` / `(n,k)`
- [ ] 我理解 fragment layout 是硬體寫死的，我只能配合
- [ ] 我能說明「LDS swizzle 讀 → fragment → MFMA → 累加 → epilogue 寫回」整條資料流
- [ ] 我知道高階 atom（路徑 B）只是把這些對應自動化

讀完進入 [`04_kernel_design_and_opt.md`](./04_kernel_design_and_opt.md)。
