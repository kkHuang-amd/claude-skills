# 補充：MLIR Python 的 `InsertionPoint` — 何時需要、如何判斷

> 目標：補齊階段 1 裡「`scf.IfOp` + `ir.InsertionPoint`」背後的**通用 MLIR 概念**，讓你在讀 `aiter/.../swiglu_and_mul.py` 一類範例時，知道這段程式**為什麼要這寫**，以及**什麼情況下**才需要自己動手切插入點。
>
> 本文件**不**重複講 FlyDSL 的 `expr` / `T` / layout；它只處理：**用 Python 建 MLIR IR 時，新操作（op）要插進哪一個 block**。

---

## 1. 它屬於 MLIR，不是 FlyDSL 專用語法

`ir.InsertionPoint(...)` 來自 **LLVM MLIR 官方 Python bindings** 的 `mlir.ir` 模組。在工作區的 FlyDSL 範例裡，通常這樣 import：

```python
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
```

（實際路徑以你環境裡 FlyDSL 綁定的 MLIR 為準；概念上等價於上游的 `from mlir.ir import InsertionPoint`。）

**語意**：MLIR 在建立 op 時需要一個「**預設插入點**」（thread-local）。`with ir.InsertionPoint(某個 Block):` 會在 `with` 區塊內，把預設插入點暫時切到那個 block，離開 `with` 後還原。

因此它不是 FlyDSL 發明的關鍵字，而是 **任何** 用 Python 手搓帶 region 的 IR（`scf.if`、`scf.for`、自訂 dialect 的多 block region）時都會遇到的標準手法。

---

## 2. 官方文件（建議收藏）

讀這兩份就夠應付多數 kernel 內手建控制流的需求：

1. **[MLIR Python Bindings](https://mlir.llvm.org/docs/Bindings/Python/)**  
   搜尋章節 **「Insertion Points and Locations」**：說明 `with InsertionPoint(...):`、`ip=` 覆寫插入點、`InsertionPoint.at_block_begin(...)` 等。

2. **[MLIR Python bindings API documentation](https://mlir.llvm.org/python-bindings/)**  
   可查 `mlir.ir.InsertionPoint` 的建構子與靜態方法（版本會隨 LLVM 演進，以你本機為準）。

若你熟 C++ 的 `OpBuilder::setInsertionPointToStart/End/...`，Python 的 `InsertionPoint` 就是同一套概念，只是用 context manager 綁在「目前 thread」上。

---

## 3. 一句話判斷：我要不要 `InsertionPoint`？

問自己：

> **我接下來要 `emit` 的那一顆 op，應該出現在「哪一個 block」？**

- 若答案是 **「現在游標所在的那個 block」**（多數線性 kernel body），通常**不用**額外切換；維持在 `@flyc.kernel` trace 時預設的插入點即可。
- 若答案是 **某個 op 的 `then_block` / `else_block` / loop `body` 等巢狀 region 裡的 block**，而你的 API **沒有**用 callback 幫你自動「走進去」，那就需要：

  ```python
  _if = scf.IfOp(cond)
  with ir.InsertionPoint(_if.then_block):
      # 這裡建立的 op 都進 then_block
      scf.YieldOp(...)
  ```

**記憶口訣**：只要你在「**空的 region**」裡還要繼續塞 op，就要先讓插入點**進到那個 region 的 block**。

---

## 4. 例子（一）：外層 `scf.if` — 對照 `swiglu_and_mul.py`

**情境**：`num_rows` 是 runtime 值，要做「這個 row 是否存在」的邊界檢查。不能用 Python `if`（那是 trace 時就定死的），要用 `scf.IfOp`，並把「整段有意義的計算」放进 **then** 分支。

節錄自 `aiter/aiter/ops/flydsl/kernels/swiglu_and_mul.py`（略去與本主題無關的中間計算）：

```python
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr import arith, buffer_ops
from flydsl.expr.arith import ArithValue, CmpIPredicate

row_valid = arith.cmpi(CmpIPredicate.ult, bid_i32, num_rows_i32)
_if_row = scf.IfOp(row_valid)
with ir.InsertionPoint(_if_row.then_block):
    in_rsrc = buffer_ops.create_buffer_resource(x, max_size=True)
    out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)
    # ... 其餘只在「合法 row」時才執行的 IR ...
    scf.YieldOp([])
```

**若沒有** `with ir.InsertionPoint(_if_row.then_block):`，後面的 `buffer_ops.create_buffer_resource(...)` 之類的 op 會留在 **if 外層**的當前 block，`then_block` 可能是空的或結構非法——這就是為什麼範例一定要切插入點。

---

## 5. 例子（二）：巢狀 `scf.if` — 外層 row、內層 dword

同一支 kernel 裡，在「row 合法」之後，還要對 **每個 dword 索引** 做 `dw_idx < out_dwords` 的檢查。於是出現 **巢狀** if：內層的 body 必須再切一次插入點。

```python
_if_row = scf.IfOp(row_valid)
with ir.InsertionPoint(_if_row.then_block):
    # ... 已在「合法 row」區塊內 ...

    dw_valid = arith.cmpi(CmpIPredicate.ult, dw_idx, out_dwords_i32)
    _if_dw = scf.IfOp(dw_valid)
    with ir.InsertionPoint(_if_dw.then_block):
        # 只有「合法 dword」才執行的 load / 計算 / store
        # ...
        scf.YieldOp([])

    scf.YieldOp([])   # 閉合外層 _if_row.then_block
```

重點：**每一層**「要在分支裡繼續建 IR」，就對應 **一層** `with ir.InsertionPoint(該層的 block):`。離開內層 `with` 後，插入點回到外層 `then_block`，才能再接外層的 `scf.YieldOp` 等。

完整可執行脈絡仍以倉庫內 `swiglu_and_mul.py` 為準。

---

## 6. 例子（三）：`@flyc.jit` 裡把東西插進 module body（進階、知道即可）

同一檔案 launcher 端有：

```python
ctx = CompilationContext.get_current()
with ir.InsertionPoint(ctx.gpu_module_body):
    pass
```

這裡示範的是：**host 端 trace / 註冊**時，有時需要暫時把插入點切到「GPU module 的頂層 body」，以便在正確的 IR 層級插入宣告或輔助 op。日常寫 elementwise kernel 時較少自己寫這段；看到時知道「是在切全域/module 級插入點」即可。

---

## 7. 什麼情況通常**不需要**自己寫 `InsertionPoint`？

- 整段邏輯是 **單一 block 的線性序列**（只有 `arith` / `vector` / `buffer_ops` 等，沒有手建 `scf.IfOp` / `scf.ForOp` 的空 region）。
- 你用的是 **FlyDSL 已封裝好的高階 API**（例如某個 helper 在內部幫你填好 region），你自己沒有呼叫 `scf.IfOp(...)` 再去塞子區塊。
- 迴圈是 **`range_constexpr`**：那是 **Python 層 unroll**，每次迭代都在**同一個**當前 block 裡重複 emit，不涉及 `scf` 的巢狀 block（對照階段 1 主文件）。

---

## 8. 與 `ip=` 參數的關係（進階）

MLIR Python 允許在**單次** op 建立時用關鍵字參數覆寫插入點，例如 `ip=ir.InsertionPoint.at_block_begin(block)`，而不必包一整段 `with`。適合只插一顆 op、或與官方文件範例對照時使用。詳見上一節官方文件。

---

## 9. 常見錯覺與除錯線索

| 現象 | 可能原因 |
|------|----------|
| `then` / `body` 裡面是空的，但外層有一大段 op | 建立 `IfOp`/`ForOp` 後**沒有**切 `InsertionPoint` 到子 block，op 全建在外層 |
| region 結構 verifier 報錯（缺 terminator、yield 不對） | 在錯的 block 建 `YieldOp`，或子區塊沒塞滿就結束 |
| 報「需要 default insertion point」之類錯誤 | 在沒有作用中的 `InsertionPoint` / `Context` 脈絡下建 op（較少見於正常 `@flyc.kernel` trace） |

---

## 10. 自我檢核

- [ ] 我分得清：`InsertionPoint` 是 **MLIR Python** 的行為，不是 FlyDSL 獨有語法。
- [ ] 我能用「下一顆 op 應該落在哪個 block」判斷要不要 `with ir.InsertionPoint(...)`。
- [ ] 我看得懂 `swiglu_and_mul.py` 裡外層 `_if_row` 與內層 `_if_dw` 兩層 `InsertionPoint` 的對應關係。
- [ ] 我知道若要查 API 細節，應去 **MLIR 官方 Python 文件**，而不是只在 FlyDSL 文件裡找。

讀完可回到 [`01_flydsl_basics.md`](./01_flydsl_basics.md) 的 §1.5，並繼續 [`02_memory_layout.md`](./02_memory_layout.md)。
