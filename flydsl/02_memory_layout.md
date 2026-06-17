# 階段 2：Memory read/write layout（難點 3）

> 目標：搞懂在 FlyDSL kernel 裡，**誰**（哪條 thread）在**什麼時候**讀寫 **global / LDS（shared）/ register** 的**哪個位址**，以及為什麼要向量化、要 swizzle、要 async copy。
>
> 本章可離線推導；只有最後驗證需要 GPU。

---

## 2.1 三層記憶體與三種 layout

| 記憶體 | FlyDSL 封裝 | 底層 op | layout 關注點 |
|--------|-------------|---------|---------------|
| **Global**（HBM/VRAM） | `GTensor` / `buffer_ops` | `buffer_load/store`（buffer descriptor） | coalesce（相鄰 thread 讀相鄰位址）、向量化 |
| **LDS / Shared** | `STensor` / `vector.load_op` | `vector.load/store` on `address_space=3` | bank conflict（swizzle 避開） |
| **Register** | `c_frags` 等 Python 變數 / `vector` 值 | SSA values | MFMA fragment layout（階段 3） |

「memory layout」就是把階段 0 的 layout 代數，分別套到這三層。

---

## 2.2 Global memory：coalesce 與向量化

### 核心原則
- 一個 warp（64 thread）同時發 load，硬體會把**相鄰 thread 讀相鄰位址**的請求合併成少數幾筆 memory transaction（coalescing）。
- 每條 thread 一次讀多個連續元素（向量化，例如 `dwordx4` = 128 bit）可大幅減少指令數與提高頻寬利用率。

### FlyDSL 寫法

`buffer_ops` 版（裸）：

```python
rsrc = buffer_ops.create_buffer_resource(tensor, max_size=True)   # buffer descriptor
v = buffer_ops.buffer_load(rsrc, offset, vec_width=8, dtype=T.f16)  # 一次讀 8 個 f16
buffer_ops.buffer_store(v, rsrc, offset)
# offset 是 element index（內部乘 element bytes）
```

`GTensor` 版（好讀）：

```python
A_ = GTensor(A, dtype=T.f16, shape=(-1, k))      # row-major, stride 自動算
vec = A_.vec_load((row, col), 8)                 # 從 (row,col) 讀 8 個連續元素
A_.vec_store((row, col), vec, 8)
off = A_.linear_offset((row, col))               # = row*k + col （= crd2idx！）
```

### 這就是 layout 代數的應用

看 `splitk_hgemm.py` 的 `ldg_a`（global → register 的載入）：

```python
for i in range_constexpr(LDG_REG_A_COUNT):       # 每條 thread 要載入幾次
    global_tid = BLOCK_THREADS * i + tid
    m_local_idx = global_tid // LDG_A_X_THREADS   # ← idx2crd 的「高維」
    k_local_idx = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE   # ← 「低維」× vec
    row_idx = m_offset + m_local_idx
    col_idx = k_offset + k_local_idx
    vec = A_.vec_load((safe_row_idx, col_idx), LDG_VEC_SIZE)    # 向量化讀
```

把 `BLOCK_THREADS * i + tid` 看成一個 flat thread-value index，`// LDG_A_X_THREADS` 與 `% LDG_A_X_THREADS` 把它拆成 `(m, k)` 座標——**這正是一個 TV layout 的 idx2crd**。設計 global load 就是設計這個 TV layout，讓：
- 同一 warp 內相鄰 tid 落在相鄰 `k`（連續維）→ coalesce
- 每條 thread 讀 `LDG_VEC_SIZE` 個連續元素 → 向量化

`safe_row_idx`（用 `arith.select` clamp 到 0）是邊界處理：越界就讀第 0 列再丟棄，避免越界存取。

### 設計 global load 的步驟（離線可做）
1. 算出 tile 總元素數 `BLOCK_M * BLOCK_K`。
2. 選 `LDG_VEC_SIZE`（f16 通常 8 = 128bit）。
3. 每次所有 thread 載入 `BLOCK_THREADS * LDG_VEC_SIZE` 個元素。
4. `LDG_REG_A_COUNT = (BLOCK_M*BLOCK_K) / (BLOCK_THREADS*LDG_VEC_SIZE)`，要整除。
5. 寫出 `tid → (m, k)` 的 idx2crd，確認連續維是 `k`。

---

## 2.3 LDS（shared memory）：bank conflict 與 swizzle

### Bank conflict 是什麼
LDS 分成 32 個 bank（每 bank 4 bytes 寬）。同一個 warp 的 thread 若同時存取**落在同一 bank 的不同位址**，硬體要序列化 → 變慢。

### Swizzle：用 XOR 打散位址

工作區用 `swizzle_xor16`（`splitk_hgemm.py` L32）：

```python
def swizzle_xor16(row, col_in_bytes, k_blocks16):
    return col_in_bytes ^ ((row % k_blocks16) * 16)
```

含義：寫入/讀取 LDS 時，根據 `row` 把欄位偏移 XOR 一段，使原本會撞同 bank 的存取被「錯開」。寫入（`sts_a`）與讀取（`ldmatrix`）用**同一個 swizzle 函式**，才能對得回原資料。

`sts_a`（register → LDS 寫入）：

```python
col_in_bytes = k_local_idx * DTYPE_BYTES
col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)   # swizzle
as_.vec_store((lds_stage, m_local_idx, col_in_bytes // DTYPE_BYTES), vecs[i], LDG_VEC_SIZE)
```

之後 MFMA 讀 LDS（`ldmatrix_compute_tile_streaming`）時，用**相同的** `swizzle_xor16` 算讀取位址：

```python
col_in_bytes = (warp_atom_k_idx + ldmatrix_a_k_vec_idx) * DTYPE_BYTES
col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
vec = as_.vec_load((s, row, col_in_bytes // DTYPE_BYTES), WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K)
```

> **心法**：swizzle 是一個「對 layout 再做一次可逆變換」的動作。寫和讀套同一個變換 → 資料不變，但 bank 衝突被打散。它不改變「邏輯座標 ↔ 值」的對應，只改變「邏輯座標 ↔ 實體 LDS offset」。

### LDS tensor 封裝

```python
as_ = STensor(smem_a_ptr, dtype_, shape=(STAGES, BLOCK_M, BLOCK_K))
as_.vec_store((stage, m, k), vec, 8)
v = as_.vec_load((stage, m, k), 8)
```

LDS 的配置用 `SmemAllocator`（`flydsl.utils.smem_allocator`），它管理 LDS 容量、對齊、各 buffer 的 offset（見 `splitk_hgemm.py` L200–212）。

---

## 2.4 Async copy（global → LDS 直送，跳過 register）

傳統路徑：global → register → LDS（佔用暫存器、兩段指令）。
Async 路徑：global → LDS **直接 DMA**，不經 register，且可與計算重疊。

工作區用 inline asm 發 `buffer_load_dwordx4 ... lds`（`splitk_hgemm.py` 的 `buffer_load_lds_inline` / `ldg_sts_a_async`）：

```python
asm = "s_mov_b32 m0, $0\n\tbuffer_load_dwordx4 $1, $2, 0 offen sc0 lds"
llvm.InlineAsmOp(None, [lds_ptr, global_offset, rsrc], asm, "s,v,s", has_side_effects=True)
```

FlyDSL 也有高階包裝（`flydsl/expr/rocdl/`）：
- `buffer_load_to_lds(rsrc, lds_ptr, voffset, size_bytes)` / `raw_ptr_buffer_load_lds(...)`
- `BufferCopy` / `BufferCopyLDS(bit_size)` atom（`rocdl/universal.py`）

搭配 `s_waitcnt vmcnt(N)`（透過 `__barrier(vmcnt)` 或 `rocdl.s_waitcnt`）控制「等到幾筆 DMA 完成」，做 multi-stage pipeline（階段 4 詳述）。

> 入門時**先不要碰 async copy**。先用同步的 `ldg → sts → barrier → load` 跑通，理解清楚後再升級到 async + 多 stage。

---

## 2.5 Reduction 的 LDS 用法（另一個好教材）

`aiter/.../kernels/reduce.py` 的 `make_block_reduce_add` 展示了 LDS 在 reduction 的典型用法，且**直接用 fly layout 代數**（不是手寫 `//`/`%`）：

```python
shape_red  = flir.make_shape(c_num_waves)
stride_red = flir.make_stride(c1)
layout_red = flir.make_layout(shape_red, stride_red)   # LDS scratch 的 layout
...
red_idx = flir.crd2idx(flir.make_coord(wave_idx), layout_red)  # 座標→offset
scratch_tv[red_idx] = w        # 每個 wave 把 partial 寫到 LDS[wave_id]
gpu.barrier()
```

流程（兩層 reduction）：
1. **Intra-wave**：用 `gpu.ShuffleOp(..., mode="xor")` 做 butterfly shuffle（`[32,16,8,4,2,1]`），warp 內 64 lane 規約成 1 值，**不需 LDS**。
2. **Cross-wave**：每個 wave 的 lane0 把 partial 寫進 `LDS[wave_id]`；barrier；wave0 讀回所有 partial，再 shuffle 規約，寫 `LDS[0]`；barrier；全 block 讀 `LDS[0]`。

> 這裡的重點：當 LDS 用量小（reduction scratch），直接用 `make_layout` + `crd2idx` 最清楚；當 LDS 用量大且效能關鍵（GEMM 的 A/B tile），才手寫 swizzle 的 offset。兩種風格你都會在工作區看到。

---

## 2.6 手算 / 設計練習（離線）

1. 一個 `128×64` f16 的 A tile，256 thread，`LDG_VEC_SIZE=8`。寫出 `tid → (m, k)` 的 idx2crd，並驗證 `LDG_REG_A_COUNT`。確認同 warp 相鄰 tid 在連續維。
2. 給 `swizzle_xor16(row, col_in_bytes, k_blocks16=4)`，列出 `row=0,1,2,3,4` 各自把 `col_in_bytes=0` swizzle 成多少。觀察 row 每 4 一循環。
3. 解釋：為什麼寫 LDS 和讀 LDS 要用「同一個」swizzle？若只在寫的時候 swizzle、讀的時候不 swizzle，會發生什麼？
4. 設計一個 block reduction（求 max）：用 reduce.py 的兩層結構，寫出 LDS scratch 的 `make_layout` 與 `crd2idx`。

<details>
<summary>參考答案要點</summary>

1. `LDG_A_X_THREADS = BLOCK_K/8 = 8`；`m = global_tid // 8`，`k = (global_tid % 8)*8`；`LDG_REG_A_COUNT = 128*64/(256*8) = 4`。同 warp 內 tid 0..63 → `k` 連續循環，符合 coalesce。
2. `0^((0%4)*16)=0`、`0^16=16`、`0^32=32`、`0^48=48`、`row=4` 時 `4%4=0` → 又回 0。
3. swizzle 是可逆映射；寫入時把邏輯 `(row,col)` 放到實體 `swizzle(row,col)`，讀取時必須用同一映射才能找回該值。只寫不讀（或反之）會讀到錯誤位置的資料。
4. shape `(NUM_WAVES,)`、stride `(1,)`；`crd2idx(make_coord(wave_idx), layout)` = `wave_idx`。

</details>

---

## 2.7 自我檢核

- [ ] 我能把 global load 的 `tid → 座標` 寫出來，並說明 coalesce / 向量化
- [ ] 我理解 bank conflict，以及 swizzle 為何寫讀要對稱
- [ ] 我能分辨同步 load 路徑 vs async DMA 路徑
- [ ] 我知道 reduction 用 LDS 的兩層（wave shuffle + cross-wave scratch）結構
- [ ] 我能把這三層記憶體存取都歸結回階段 0 的 layout 代數

讀完進入 [`03_mfma_layout.md`](./03_mfma_layout.md)。
