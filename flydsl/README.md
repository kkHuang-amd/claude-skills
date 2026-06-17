# FlyDSL 學習計劃

> 目標讀者：懂 GPU 程式設計（CUDA / HIP），但 FlyDSL 與 CuTe 風格的 layout 概念是新的。
>
> 核心難點（依你自己的判斷）：
> 1. **FlyDSL 的 layout 定義**（CuTe 風格的 `shape:stride`、`idx2crd`/`crd2idx`、divide/partition）
> 2. **AMD 內部 MFMA layout**（thread ↔ fragment ↔ matrix element 的對應）
> 3. **Memory read/write layout**（global / LDS / register，向量化、swizzle、async copy）
>
> 本計劃以這三大難點為核心，由原理到實作逐步拆解。

---

## 0. 重要：本計劃的離線原則

整份計劃刻意設計成**不需要佔用 GPU**也能學習大部分內容，因為工作區常有 GPU 效能測試在跑。

- **可離線做的事**（CPU only，全程安全）：
  - 讀程式碼、讀本計劃、在紙上 / 文字推導 layout 座標
  - 用 `python -c "import flydsl"` 之類確認 API 路徑（純 import，不編譯）
  - 寫 kernel 原始碼、設計 layout、推導 thread-to-element 對應
- **會用到 GPU 的事**（要先確認沒有效能測試在跑才做）：
  - `flyc.compile(...)`、`launcher.launch(...)`、跑任何 `test_flydsl_*.py`
  - `@autotune` 的 benchmark（會跑 kernel）

> 練習題（`05_exercises.md`）標註了哪些步驟是「離線設計」、哪些是「需要 GPU 驗證」。先把離線部分全部完成，再找空檔批次驗證。

---

## 1. 環境與座標

- FlyDSL 安裝路徑：`/opt/venv/lib/python3.10/site-packages/flydsl`（版本 `0.2.0`）
- 工作區裡的真實 kernel 範例（最佳教材）：
  - `aiter/aiter/ops/flydsl/kernels/` — splitk_hgemm、swiglu_and_mul、reduce、preshuffle_gemm、mfma_* 等
  - `aiter/aiter/ops/flydsl/kernels/layout_utils.py` — layout 字串 parse 與 idx2crd/crd2idx 的純 arith 實作（很好的入門讀物）
  - `aiter/aiter/ops/flydsl/kernels/tensor_shim.py` — `GTensor`/`STensor`/`TensorView`（封裝 global/shared 存取）
- `mori/python/mori/ir/flydsl/` — 用 FlyDSL 包裝 shmem device function 的範例（進階，先略過）

慣用 import：

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T
from flydsl.expr import arith, vector, gpu, buffer_ops, const_expr, range_constexpr
from flydsl.expr.rocdl import mfma_f32_16x16x16f16
from flydsl.expr.rocdl.universal import MFMA, BufferCopy
from flydsl.autotune import Config, autotune
```

---

## 2. 學習路線圖

建議**按順序**讀，每份文件結尾都有「自我檢核問題」與對應練習題。

| 階段 | 文件 | 主題 | 對應難點 |
|------|------|------|----------|
| 0 | [`00_layout_fundamentals.md`](./00_layout_fundamentals.md) | CuTe 風格 layout 代數：shape/stride、idx2crd/crd2idx、logical_divide、partition | 難點 1 |
| 1 | [`01_flydsl_basics.md`](./01_flydsl_basics.md) | `@flyc.kernel`/`@flyc.jit`、expr/arith/vector、`T` 型別、第一個 elementwise kernel | 基礎 |
| 2 | [`02_memory_layout.md`](./02_memory_layout.md) | global / LDS / register layout、向量化 load/store、swizzle、async copy | 難點 3 |
| 3 | [`03_mfma_layout.md`](./03_mfma_layout.md) | AMD MFMA/WMMA fragment layout、thread↔element 對應、rocdl intrinsics | 難點 2 |
| 4 | [`04_kernel_design_and_opt.md`](./04_kernel_design_and_opt.md) | 完整 HGEMM 拆解、software pipeline、scheduler hints、autotune | 整合 + 優化 |
| 5 | [`05_exercises.md`](./05_exercises.md) | 由簡到繁的練習題（標註離線 / 需 GPU） | 實作 |

### 心智模型（一句話版本）

> FlyDSL = **MLIR 的 Python 前端** + **CuTe 風格的 layout 代數** + **AMD CDNA/RDNA 的 MFMA/buffer intrinsics**。
>
> 你寫的 Python 不是「執行」運算，而是在「**描述（trace）一段 MLIR IR**」；`@flyc.jit` 把它編譯成 GPU 機器碼。所有 layout 計算都是 compile-time 的座標代數，最後變成 thread 怎麼算自己該讀寫哪個位址。

---

## 3. 三大難點的關聯（為什麼要照這順序學）

```
                    難點 1: Layout 代數
                  (shape:stride, crd2idx)
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
   難點 3: Memory layout         難點 2: MFMA layout
  (誰讀寫 global/LDS 的哪格)    (誰持有 fragment 的哪個元素)
            │                           │
            └─────────────┬─────────────┘
                          ▼
                   完整 GEMM kernel
              (把資料從 global 搬到能餵給
               MFMA 的 register/LDS 排列)
```

- **Layout 代數是地基**：它回答「一個多維座標 ↔ 一個線性 offset」這個唯一的核心問題。
- **Memory layout** 是「把 layout 代數套到 global/LDS/register 三種記憶體」。
- **MFMA layout** 是「硬體規定的、固定的 fragment layout」——你不能改它，只能去**配合**它：把資料排成 MFMA 指令期望的樣子。
- GEMM 把三者縫合：global → (向量化載入) → LDS（swizzle 避 bank conflict）→ register（MFMA fragment）→ MFMA → 累加 → 寫回。

---

## 4. 如何使用本計劃

1. 每個階段先讀文件，理解概念與工作區真實程式碼的對照。
2. 用文件裡的「手算練習」在紙上 / 文字推導座標（離線、零 GPU）。
3. 做 `05_exercises.md` 對應的離線設計題。
4. 累積到一定程度後，找沒有效能測試的時段，批次跑「需 GPU 驗證」的題目。

預估投入：每階段 0.5～1.5 天（視熟悉度），總計約 1～2 週可達到「能讀懂並改寫工作區 kernel」的程度。
