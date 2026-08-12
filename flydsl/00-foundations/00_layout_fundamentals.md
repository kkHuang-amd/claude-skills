# 階段 0：Layout 代數基礎（難點 1）

> 這是整個 FlyDSL 的地基。把這一章搞懂，後面的 memory layout 和 MFMA layout 都只是它的特例。
> 全章可以**純手算 / 文字推導**，完全不需要 GPU。

---

## 0.1 核心問題：一個座標 ↔ 一個 offset

GPU kernel 裡所有「資料排列」的問題，最後都歸結為一個函式：

```
給定一個多維座標 (i, j, k, ...)，它在記憶體裡的線性位址 offset 是多少？
```

CuTe / FlyDSL 用一個叫 **Layout** 的物件來精確描述這個函式。Layout 由兩個 tuple 組成：

- **shape**：每一維的大小，例如 `(4, 64)`
- **stride**：每一維「座標加 1」時 offset 要走多遠，例如 `(64, 1)`

寫法（CuTe 字串）：`(4,64):(64,1)`，FlyDSL API：`make_layout((4, 64), (64, 1))`。

### crd2idx：座標 → 線性 offset

公式就是內積：

```
idx = Σ coord_i × stride_i
```

例：layout `(4,64):(64,1)`，座標 `(2, 3)`：

```
idx = 2 × 64 + 3 × 1 = 131
```

這正是 **row-major 4×64 矩陣** 的排列：第 2 列第 3 行，offset = 2×64+3。

### idx2crd：線性 offset → 座標（crd2idx 的逆）

```
idx = 131,  shape=(4,64), stride=(64,1)
→ 先還原 stride 最大的維：i = 131 // 64 = 2，餘 131 % 64 = 3
→ j = 3 // 1 = 3
→ (2, 3)
```

> **關鍵心法**：stride 決定「memory 的真實排列」，shape 只決定「合法座標範圍」。
> 同一塊 memory，換 stride 就換了「視角」（view），資料本身沒動。

---

## 0.2 column-major vs row-major：只是 stride 不同

同一個 4×64 的邏輯矩陣：

| 排列 | layout | `(i,j)` 的 offset |
|------|--------|------------------|
| row-major（C 風格） | `(4,64):(64,1)` | `i*64 + j` |
| column-major（Fortran 風格） | `(4,64):(1,4)` | `i*1 + j*4` |

stride 為 `1` 的那一維就是「記憶體上連續」的維。這在向量化載入（一次讀 8 個連續元素）時極重要——只有連續維才能 coalesce / 向量化。

---

## 0.3 巢狀（hierarchical）layout

CuTe 的威力在於 shape/stride 可以是**巢狀 tuple**。例如：

```
((2,2), 4) : ((1,16), 4)
```

第一個 mode 是 `(2,2)`，自己又是個 2 維 layout。這用來表達「一個 tile 內又分 sub-tile」的階層結構。

座標也跟著巢狀：`((a,b), c)`，offset 公式照樣是「把每個葉節點 coord × 對應 stride 全加起來」：

```
idx = a×1 + b×16 + c×4
```

> 入門階段你**先用扁平（flat）layout**就好，例如 `(M,N):(sM,sN)`。巢狀 layout 在 MFMA fragment 與 tiled copy 才會大量出現，到階段 3 再回來看。

---

## 0.4 在 FlyDSL 裡實際長怎樣

### 路徑 A：fly dialect 的原生 layout（CuTe 等價物）

FlyDSL 的 `fly` MLIR dialect 直接實作了 CuTe 代數。主要 API 在 `flydsl/expr/primitive.py`：

| FlyDSL API | CuTe 對應 | 用途 |
|------------|-----------|------|
| `make_layout(shape, stride)` | `make_layout` | 建基本 layout |
| `make_coord(*coord)` | `make_coord` | 建座標（`None` = wildcard 該軸） |
| `crd2idx(crd, layout)` | `crd2idx` | 座標 → index |
| `idx2crd(index, layout)` | `idx2crd` | index → 階層座標 |
| `make_ordered_layout(shape, order)` | `make_ordered_layout` | 依 order 自動算 compact stride |
| `make_identity_layout(shape)` | identity | basis stride |
| `logical_divide / zipped_divide / tiled_divide` | `logical_divide` 等 | **tiling**（切 tile） |
| `composition / complement` | 同名 | layout 合成 |
| `slice(src, coord)` / `dice(src, coord)` | slice / dice | 固定某些軸、保留 `None` 軸 |

docstring 範例（`primitive.py`）：

```python
crd2idx((1, 2), make_layout((4, 8), (1, 4)))   # -> 9   （= 1*1 + 2*4）
idx2crd(9,      make_layout((4, 8), (1, 4)))   # -> (1, 2)
```

Python 包裝類 `Layout`（`flydsl/expr/typing.py`）讓你能用更自然的語法：

```python
ly = make_layout((4, 64), (64, 1))
ly.shape          # IntTuple (4, 64)
ly.stride         # IntTuple (64, 1)
ly(2, 3)          # -> 131   （__call__ = crd2idx）
ly(2, None)       # 含 None → 回傳 slice 後的子 layout
ly.get_hier_coord(131)   # -> idx2crd，得到階層座標
```

### 路徑 B：純 arith 手算 layout（工作區的做法）

工作區的 GEMM kernel **沒有**用 fly dialect 的 layout op，而是自己用整數算術算 offset，因為對「靜態、2 的次方」的 stride 可以用位移/遮罩取代除法/取餘，省掉 CDNA 上 10–15 cycle 的整數除法。

讀 `aiter/aiter/ops/flydsl/kernels/layout_utils.py`：

- `_parse_layout("(4,64):(64,1)")` → `([4,64], [64,1])`
- `idx2crd(idx, layout)`：靜態 layout 用 `shrui`/`andi`（2 的次方）算座標；動態才 fallback 到 `fx.idx2crd`
- `crd2idx(crd, layout)`：靜態就是 `Σ coord_i * stride_i`

關鍵優化片段（觀念）：

```python
# divisor 是 2 的次方時：
#   val // divisor  → arith.shrui(val, log2(divisor))   1 cycle
#   val %  modulus  → arith.andi(val, modulus - 1)      1 cycle
# 否則才用 divui / remui（10-15 cycle）
```

> **重點認知**：你在工作區看到的大量 `idx // X`、`idx % X`、`a * stride + b` 其實**就是手寫的 crd2idx**。理解了 layout 代數，這些散落的整數運算就不再是天書，而是「某個 layout 的 offset 公式被攤平寫出來」。

---

## 0.5 TV layout（Thread-Value layout）——通往 MFMA 的橋樑

這是最重要、也最容易卡住的概念，先建立直覺，階段 3 會再深入。

當一個 warp（AMD 上 64 條 thread）一起處理一個 tile 時，我們需要回答兩個問題：

1. **哪條 thread** 負責 tile 裡的哪些元素？
2. 每條 thread 自己手上的**第幾個 value** 對應 tile 的哪個座標？

把這兩件事合起來描述的 layout 就叫 **TV layout**：它的座標是 `(thread_id, value_id)`，輸出是「tile 內的線性 offset」。

```
TV_layout : (thread, value) -> offset_in_tile
```

FlyDSL helper（`flydsl/expr/derived.py`）：

- `make_layout_tv(thr_layout, val_layout)` — 把 thread layout 與 value layout 組成 TV layout
- `make_tiled_copy_tv(...)` / `make_tiled_copy_A/B/C(...)` — 給 copy 用的 TV 分區
- `ThrCopy.partition_S/D(tensor)` — 取得「**這條 thread** 負責來源/目的的哪一塊」
- `ThrMma.partition_A/B/C(tensor)` — MMA 版本

> CuTe 的 `local_partition` 在 FlyDSL 沒有同名函式，對應的就是上面這些 `partition_*`。
> CuTe 的 `local_tile` 對應 `logical_divide` / `zipped_divide` / `tiled_divide`。

工作區 GEMM 沒有用這些高階 helper，而是手寫 thread→座標（例如 `w_tid % WMMA_M`、`w_tid // WMMA_M * frag_values`）。這其實**就是把 TV layout 攤平手寫**。階段 3 我們會把這些手寫公式對回 MFMA 的官方 fragment layout。

---

## 0.6 手算練習（離線，零 GPU）

在紙上 / 文字裡推導，做完對照下面的答案。

1. layout `(8,16):(16,1)`，座標 `(3, 5)` 的 offset？反過來 offset `53` 的座標？
2. layout `(8,16):(1,8)`（column-major），座標 `(3, 5)` 的 offset？哪一維在記憶體連續？
3. 一個 `128×64` 的 row-major tile，要讓每條 thread（共 256 條）一次向量化讀 8 個連續元素，總共要幾次 load？每條 thread 的 `(row, col)` 起點用 `tid` 怎麼表示？（提示：對照 splitk_hgemm.py 的 `ldg_a`）
4. 給定 `idx // X` 與 `idx % Y` 散在程式裡，反推它對應的 layout shape/stride。

<details>
<summary>參考答案</summary>

1. `3*16 + 5 = 53`；`53 // 16 = 3` 餘 `5` → `(3, 5)`。
2. `3*1 + 5*8 = 43`；第 0 維（stride 1）在記憶體連續。
3. 總元素 `128*64=8192`，每次 256 thread × 8 = 2048 元素，需 `8192/2048 = 4` 次（對照 `LDG_REG_A_COUNT`）。每條 thread：`global_tid = 256*i + tid`；`row = global_tid // (BLOCK_K//8)`，`col = (global_tid % (BLOCK_K//8)) * 8`。這就是一個 TV layout 的攤平。
4. `idx // X` 表示「以 X 為 stride 的高維」，`idx % Y` 表示「大小為 Y 的低維」。若兩者搭配成 `(idx//N, idx%N)`，對應 shape 含一維大小 `N`、stride `1`。

</details>

---

## 0.7 自我檢核

- [ ] 我能口頭說出 `crd2idx` 的公式，以及 shape vs stride 的角色差異
- [ ] 給我一個 `(shape):(stride)`，我能立刻寫出 offset 公式
- [ ] 我知道「換 stride = 換 view，不動資料」
- [ ] 我能把工作區 kernel 裡散落的 `// 和 %` 認出來是「某個 layout 的 crd2idx/idx2crd」
- [ ] 我理解 TV layout 是 `(thread, value) -> offset`，且知道它連到 MFMA

讀完進入 [`01_flydsl_basics.md`](./01_flydsl_basics.md)。
