# 階段 1：FlyDSL 基礎 — 寫出第一個 kernel

> 目標：理解 FlyDSL 的「trace 出 MLIR」執行模型、`@flyc.kernel` / `@flyc.jit` 的分工、`expr` 模組與 `T` 型別系統，並能讀懂一個最簡單的 elementwise kernel。
>
> 讀程式碼與設計可離線；只有最後「實際編譯執行」需要 GPU。

---

## 1.1 心智模型：你在寫的是「IR 產生器」，不是運算

最關鍵、也最反直覺的一點：

```python
@flyc.kernel
def my_kernel(x: fx.Tensor):
    tid = fx.thread_idx.x
    a = arith.constant(1.0, type=T.f32)
    b = a + a          # 這行「不」算出 2.0
```

`b = a + a` 並沒有在 Python 裡算出 `2.0`。它**發射（emit）了一個 MLIR `arith.addf` 運算**到當前的 IR。整個 kernel body 是「描述一段計算圖」，最後由編譯器降到 GPU 機器碼。

推論：
- Python 的 `for` 迴圈是「**展開**（unroll）」——每次迭代都 emit 一份 IR。要 unroll 用 `range_constexpr`。
- 真正的 runtime 迴圈要用 `scf.ForOp` / `scf.WhileOp`（或 FlyDSL 包裝的 `for ... in range(...)` generator 形式，見階段 4）。
- 真正的 runtime 分支要用 `scf.IfOp` + `ir.InsertionPoint(if_op.then_block)`。
- `const_expr(x)` 標記「這是 compile-time 常數」，AST rewriter 會據此決定 unroll / 分支裁剪。

---

## 1.2 `@flyc.kernel` vs `@flyc.jit`

| 裝飾器 | 角色 | 在哪執行 |
|--------|------|----------|
| `@flyc.kernel` | **device 端** kernel body（GPU 上跑的程式） | 被 `@jit` 內呼叫時 trace 成 `gpu.func` |
| `@flyc.jit` | **host 端** launcher（決定 grid/block、呼叫 kernel） | 首次呼叫時觸發整段編譯 |

骨架（對照 `swiglu_and_mul.py`）：

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T

@flyc.kernel
def add_kernel(x: fx.Tensor, y: fx.Tensor, out: fx.Tensor, n: fx.Int32):
    # ... device 程式，跑在每條 thread 上 ...
    pass

@flyc.jit
def launch_add(x: fx.Tensor, y: fx.Tensor, out: fx.Tensor, n: fx.Int32,
               stream: fx.Stream = fx.Stream(None)):
    launcher = add_kernel(x, y, out, n)        # 不會立刻 launch，回傳 launcher
    launcher.launch(grid=(grid_x, 1, 1),
                    block=(256, 1, 1),
                    stream=stream)
```

呼叫 `launch_add(x, y, out, n)`：
- **第一次**：trace + 編譯 + 執行（慢，秒級）
- **之後**：走 cache 的 `CompiledFunction`，純 dispatch（微秒級）
- 也可手動 `cf = flyc.compile(launch_add, x, y, out, n)` 預先編譯（見 `tensor_shim.py` 的 `_run_compiled`）

`block > 256`（AMD）時要標 `@flyc.kernel(known_block_size=[BLOCK, 1, 1])`（見 splitk_hgemm.py L226）。

---

## 1.3 `expr` 模組速查

`import flydsl.expr as fx` 之後常用的東西：

| 來源 | 內容 | 範例 |
|------|------|------|
| `fx.thread_idx` / `fx.block_idx` | 內建索引 | `fx.thread_idx.x`, `fx.block_idx.x` |
| `fx.Tensor` / `fx.Pointer` | kernel 參數型別 | 簽名標註 |
| `fx.Int32` / `fx.Index` / `fx.Stream` | numeric / index / stream 型別 | `n: fx.Int32` |
| `fx.const_expr` / `fx.range_constexpr` | compile-time 標記 / unroll 迴圈 | `for i in range_constexpr(4):` |
| `arith` | 算術（add/mul/cmp/select/bitcast/index_cast…） | `arith.constant`, `arith.cmpi`, `arith.select` |
| `vector` | 向量操作 | `vector.load_op`, `vector.extract`, `vector.bitcast`, `vector.from_elements` |
| `gpu` | barrier、index | `gpu.barrier()` |
| `buffer_ops` | AMD global mem load/store | `buffer_load`, `buffer_store`, `create_buffer_resource` |
| `rocdl` | MFMA/WMMA、sched hints、buffer_load_lds | `rocdl.mfma_*`, `rocdl.sched_barrier` |
| `llvm` (`flydsl._mlir.dialects`) | inline asm、atomic、GEP | `llvm.InlineAsmOp`, `llvm.AtomicRMWOp` |
| `scf` (`flydsl._mlir.dialects`) | runtime 控制流 | `scf.IfOp`, `scf.WhileOp`, `scf.YieldOp` |

> 注意：`const_expr` / `range_constexpr` 不是獨立模組，定義在 `primitive.py`（`const_expr` 直接回傳輸入，是給 AST rewriter 看的標記；`range_constexpr` 就是 `range`）。

---

## 1.4 型別系統 `T`

`from flydsl.expr.typing import T`，`T` 是型別建構器（property 風格）：

| 類別 | 範例 |
|------|------|
| 整數 | `T.i8`, `T.i16`, `T.i32`, `T.i64`, `T.index` |
| 浮點 | `T.f16`, `T.bf16`, `T.f32`, `T.f64` |
| 向量（捷徑） | `T.f16x4`, `T.f32x4`, `T.bf16x8`, `T.i64x2` |
| 向量（動態） | `T.vec(n, elem)`，例如 `T.vec(8, T.f16)` |
| FP8 | `T.f8`, `T.f8x4` … `T.f8x16`（依 arch 自動選 E4M3FN / E4M3FNUZ） |

向量型別在 MFMA / 向量化 load 大量使用。例如 16×16×16 f16 MFMA 的 A/B fragment 是 `T.vec(4, T.f16)`（=`T.f16x4`），累加器是 `T.f32x4`。

---

## 1.5 讀懂第一個 kernel：`swiglu_and_mul`

完整檔在 `aiter/aiter/ops/flydsl/kernels/swiglu_and_mul.py`。它是很好的入門範例，因為**沒有 MFMA、沒有 LDS、沒有 layout 代數**，純粹是 elementwise + buffer load/store + runtime 分支。逐塊看：

### (a) grid / block 與索引

```python
bid = fx.block_idx.x          # 每個 block 處理一個 row
tid = fx.thread_idx.x         # block 內 256 條 thread
```

launcher 端 `grid=(num_rows, 1, 1)`, `block=(256, 1, 1)`。

### (b) runtime 分支（邊界檢查）

因為 `num_rows` 是 runtime 值，不能用 Python `if`，要用 `scf.IfOp`：

```python
row_valid = arith.cmpi(CmpIPredicate.ult, bid_i32, num_rows_i32)
_if_row = scf.IfOp(row_valid)
with ir.InsertionPoint(_if_row.then_block):
    ...  # 後續所有 IR 都 emit 進 then_block
    scf.YieldOp([])
```

> 模式記起來：建 `scf.IfOp(cond)` → `with ir.InsertionPoint(if_op.then_block):` → body → `scf.YieldOp([])`。這個「進 block 寫 IR」的寫法在工作區到處都是。

> **補充（建議閱讀）**：`InsertionPoint` 是 **MLIR Python** 的通用機制（不是 FlyDSL 專用語法）；何時需要、如何判斷、巢狀 `scf.if` 範例與官方文件連結見 [`01b_mlir_python_insertion_point.md`](./01b_mlir_python_insertion_point.md)。

### (c) compile-time unroll 迴圈

```python
for iter_idx in range_constexpr((out_dwords + DWORDS_PER_ITER - 1) // DWORDS_PER_ITER):
    dw_idx = thread_id + arith.constant(iter_idx * DWORDS_PER_ITER, type=i32)
    ...
```

`range_constexpr` → 編譯期完全展開，每個 `iter_idx` 都是 Python int 常數。

### (d) buffer load/store（global memory）

```python
in_rsrc = buffer_ops.create_buffer_resource(x, max_size=True)   # 建 buffer descriptor
gate_raw_dw = buffer_ops.buffer_load(in_rsrc, gate_dw_off, vec_width=1, dtype=i32)
...
buffer_ops.buffer_store(packed, out_rsrc, out_dw)
```

- `create_buffer_resource(tensor)` 把 tensor 包成 AMD 的 buffer descriptor（`!llvm.ptr<8>`），之後 `buffer_load/store` 都靠它。
- `buffer_load(rsrc, offset, vec_width, dtype)` 的 `offset` 是 **element index**（內部會乘 element bytes）。
- 這支 kernel 自己用 i32 dword 為單位搬資料（每條 thread 處理 2 個連續 bf16 = 1 dword），避免 read-modify-write 競爭。

### (e) 位元操作做 layout/dtype 轉換

interleave 定址、bf16↔f32 的 bitcast、把兩個 bf16 pack 成一個 dword：

```python
g_f32 = arith.bitcast(f32, (gate_shifted & mask16) << 16)   # bf16 放進 f32 高 16 bit
packed = results[0] | (results[1] << 16)                    # 兩個 bf16 pack 成 dword
```

這裡的 `block = col // NLane`、`lane = col % NLane`、`gate_col = block*2*NLane + lane` **就是手寫的 crd2idx**——把 interleave layout `(N0, 2, NLane)` 的座標轉成 offset。回顧階段 0 就懂了。

---

## 1.6 兩種 tensor 存取封裝（工作區慣用）

`aiter/.../kernels/tensor_shim.py` 提供更高階的封裝，比裸 `buffer_ops` 好讀：

- `GTensor(memref, dtype, shape, stride=None)` — global tensor，內建 buffer descriptor
  - `g[i, j]`、`g.vec_load((i, j), 8)`、`g.vec_store((i, j), vec, 8)`、`g.linear_offset((i, j))`
- `STensor(smem_ptr, dtype, shape)` — shared（LDS）tensor，底層用 `vector.load_op`/`vector.store`
- `TensorView` — 通用 view，支援 `local_tile(tile_shape, tile_idxs)`、`copy_(...)`
  - `local_tile` 就是階段 0 講的 tiling：固定 tile 起點、保留 tile 內 stride

`shape` 裡的 `stride` 預設用 `np.cumprod` 自動算成 **row-major compact stride**——這又回到階段 0 的 layout 代數。

> 入門時建議：先用 `GTensor`/`STensor` 寫，可讀性高；等要做極致優化（async copy、inline asm DMA）再下放到 `buffer_ops` + `llvm.InlineAsmOp`（見階段 2、4）。

---

## 1.7 編譯 / 執行（需要 GPU，找空檔再做）

```python
# 方式 1：直接呼叫 jit launcher（首次自動編譯）
launch_add(x, y, out, n)

# 方式 2：預編譯，拿到快取的 CompiledFunction
cf = flyc.compile(launch_add, x, y, out, n)
cf(x, y, out, n)   # 之後純 dispatch
```

> 在效能測試跑的時候，**不要**做這一步。先把前面的「讀懂 + 設計」做完。

---

## 1.8 練習（離線設計部分）

1. 不看 swiglu 範例，憑記憶寫出一個 `@flyc.kernel` + `@flyc.jit` 的骨架（含 grid/block、邊界 `scf.IfOp`）。
2. 設計一個 elementwise `out = x * 2 + 1`（f32）的 kernel：每條 thread 處理一個元素，含邊界檢查。先寫成「每元素一次 load」，再改成「向量化一次讀 4 個（`vec_width=4`）」。
3. 把第 2 題裡你寫的 `offset` 計算，用階段 0 的 layout 語言描述：它對應哪個 `(shape):(stride)`？

（這些都先**寫原始碼 + 紙上推導**，GPU 驗證留到 `05_exercises.md` 的批次驗證階段。）

---

## 1.9 自我檢核

- [ ] 我理解 kernel body 是在 trace IR，不是直接運算
- [ ] 我能分清 `range_constexpr`（unroll）vs `scf.ForOp`（runtime loop）
- [ ] 我會用 `scf.IfOp` + `ir.InsertionPoint` 寫 runtime 分支（細節與判斷準則見 [`01b_mlir_python_insertion_point.md`](./01b_mlir_python_insertion_point.md)）
- [ ] 我知道 `buffer_load` 的 offset 是 element index
- [ ] 我能把 kernel 裡的 `//` `%` 對回某個 layout 的 crd2idx

讀完進入 [`02_memory_layout.md`](./02_memory_layout.md)。若仍對 `InsertionPoint` 與巢狀 `scf.if` 不熟，建議先讀 [`01b_mlir_python_insertion_point.md`](./01b_mlir_python_insertion_point.md)。
