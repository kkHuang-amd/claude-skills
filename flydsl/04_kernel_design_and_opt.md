# 階段 4：Kernel 設計與優化

> 目標：把前三章縫成一個完整的 HGEMM，理解 software pipeline（multi-stage）、split-K、scheduler hints、autotune，並建立一套可重複的優化方法論。
>
> 設計與讀碼可離線；benchmark / autotune 需要 GPU（找空檔再做）。

---

## 4.1 一個 GEMM kernel 的解剖（以 `splitk_hgemm.py` 為主軸）

完整檔：`aiter/aiter/ops/flydsl/kernels/splitk_hgemm.py`。建議開著它對照本節。

### 整體結構

```
compile_hgemm_kernel(dtype, n, k, TILE_M, TILE_N, TILE_K, STAGES, SPLIT_K, ...)
  │  （host 端、編譯期）算出所有 layout 參數、LDS 配置、kernel 名稱
  │
  ├─ @flyc.kernel hgemm_kernel(...)      ← device 程式
  │     1. 建 GTensor (A,B,C)、STensor (LDS as_/bs_/cs_)、SmemAllocator
  │     2. 算 thread/warp 座標 (wid, w_tid, warp_m_idx, ...)
  │     3. zero_c / split_k_barrier（SPLIT_K 用）
  │     4. 主迴圈：prologue → K-loop（load + MFMA pipeline）→ epilogue
  │     5. 寫回 global（含 SPLIT_K 的 atomic add）
  │
  └─ @flyc.jit launch_hgemm_kernel(...)  ← 算 grid，finalize LDS，launch
```

### 編譯期參數推導（L107–224）
這一大段全是 Python int 算術（編譯期），把使用者給的 tile 形狀，推導出：
- warp 分工：`BLOCK_M_WARPS × BLOCK_N_WARPS × BLOCK_K_WARPS`
- 每 warp 的 MFMA 步數：`WARP_M_STEPS / WARP_N_STEPS / WARP_K_STEPS`
- global load 的 TV 參數：`LDG_REG_A_COUNT`、`LDG_A_X_THREADS`（= 階段 2 的 idx2crd）
- LDS 用量與 offset（`SmemAllocator`）
- 一堆 `assert` 確保形狀整除、LDS 不超容量、`s_waitcnt` 計數合法

> **設計 kernel 的第一步就是這個推導**。tile 形狀決定了後面所有 layout。把它當成「解一組整除約束 + 算 TV layout」的數學題。

### Device 端的座標建立（L264–290）

```python
tid   = fx.thread_idx.x
wid   = tid // WARP_SIZE          # 第幾個 warp
w_tid = tid % WARP_SIZE           # warp 內 lane (0..63)
wid_mn = wid % BLOCK_MN_WARPS     # 在 MN 平面的 warp 座標
wid_k  = wid // BLOCK_MN_WARPS    # 在 K 切片的 warp 座標
warp_m_idx = wid_mn // BLOCK_N_WARPS * WARP_M
warp_n_idx = wid_mn %  BLOCK_N_WARPS * WARP_N
```

這就是把 `tid` 用 idx2crd 拆成 `(wid_k, wid_mn) → (warp_m, warp_n)` 再到 lane——**全部是 layout 代數**。

---

## 4.2 Software pipeline（multi-stage）

GEMM 的核心優化：讓 **memory load** 與 **MFMA 計算** 重疊，藏住 HBM 延遲。

### 概念
- LDS 開 `STAGES` 份 buffer（double/triple buffering）。
- Prologue：先預載 `STAGES-1` 個 K-block 到 LDS。
- 主迴圈第 `bki` 次：一邊發 **下一個** K-block 的 async load（寫 stage `write_stage`），一邊用 **當前** stage 的資料做 MFMA。
- Epilogue：把剩下還在 flight 的 stage 算完。

`splitk_hgemm.py` 的 `B_TO_LDS` 分支（L659–703）：

```python
for s in range_constexpr(STAGES - 1):           # prologue 預載
    ldg_sts_b_async(ks_begin + s*BLOCK_K, s)
    ldg_sts_a_async(ks_begin + s*BLOCK_K, s)

for bki, state in range(0, BLOCK_K_LOOPS-(STAGES-1), 1, init=init_state):  # runtime loop
    __barrier((STAGES-2) * LDG_WAIT_COUNT)        # 等夠多 DMA 完成
    ldg_sts_b_async(k_offset + (STAGES-1)*BLOCK_K, write_stage)  # 預取下一塊
    ldg_sts_a_async(k_offset + (STAGES-1)*BLOCK_K, write_stage)
    c_frags_new = ldmatrix_compute_tile_streaming(current_stage, c_frags)  # 算當前
    hot_loop_scheduler()                          # scheduler hints
    results = yield [k_offset_next, next_stage] + c_frags_new
```

> 注意這裡的 `for ... in range(start, stop, step, init=...)` + `yield` 是 FlyDSL 把 **runtime loop 帶 loop-carried values**（`scf.ForOp` 的 iter_args）包成 generator 的語法。`c_frags`（累加器）就是 loop-carried，跨迭代帶著走。

### `s_waitcnt` 與 stage 數
`__barrier(vmcnt=N)` 發 `s_waitcnt vmcnt(N) + s_barrier`，意思是「等到還剩 N 筆 memory op 未完成時就放行」。stage 越多，能容忍越深的 in-flight load，但耗更多 LDS 與暫存器。`LDG_WAIT_COUNT` 與 `(STAGES-2)*LDG_WAIT_COUNT < 63` 的 assert 就是在管這個。

---

## 4.3 Scheduler hints（指令排程微調）

CDNA 上，VMEM（load）、DS（LDS read/write）、MFMA、VALU 走不同的硬體 pipe。手動提示編譯器交錯這些指令可大幅提升 overlap。

`splitk_hgemm.py` 的 `hot_loop_scheduler`（L666–680）：

```python
for i in range_constexpr(LDG_REG_B_COUNT_AS):
    rocdl.sched_vmem(1)       # 安排 1 個 vmem（global/DMA load）
for i in range_constexpr(LDG_REG_A_COUNT_AS):
    rocdl.sched_vmem(1)
for ki in range_constexpr(WARP_K_STEPS):
    for i in range_constexpr(WARP_N_STEPS):
        rocdl.sched_dsrd(1)   # 安排 1 個 LDS read
    for i in range_constexpr(WARP_M_STEPS):
        rocdl.sched_dsrd(1)
    for i in range_constexpr(WARP_M_STEPS):
        rocdl.sched_mfma(WARP_N_STEPS)   # 安排 MFMA
rocdl.sched_barrier(0)        # 鎖定這段排程，禁止跨界重排
```

常用 hints：
- `rocdl.sched_barrier(0)`：排程屏障，禁止指令越過它重排（劃定 region）。
- `rocdl.sched_vmem(n)` / `sched_dsrd(n)` / `sched_mfma(n)`：在當前位置「指定接下來放 n 個某類指令」。
- `rocdl.s_waitcnt(0)`、`rocdl.sched_barrier(...)` 也用於正確性（確保 DMA 完成）。

> 這是**最後階段**的優化，需要看 ISA / profiler。入門時先把功能寫對，scheduler hints 之後再加。

---

## 4.4 Split-K（多 block 協作算同一塊 C）

當 K 很大、M×N 小，單一 block 算完整 K 會讓 SM 閒置。Split-K 把 K 切成 `SPLIT_K` 份，由 `grid.y` 個 block 各算一份 partial，再用 **atomic add** 累加到 global C。

`splitk_hgemm.py` 的機制：
- `zero_c()`：第一個 block 把 C 清零並發 signal。
- 其它 block 在 `split_k_barrier()` spin-wait 該 signal（確保 C 已清零才開始 atomic add）。
- 寫回時用 `llvm.AtomicRMWOp(fadd, ...)` 累加（L817）。
- 用 `semaphore` 計數，最後一個 block 清掉 signal/semaphore。

> Split-K 引入了 grid 間同步與 atomic，是進階主題。先把 `SPLIT_K=1` 的單 block 版跑通。

---

## 4.5 Autotune

`flydsl.autotune` 自動 benchmark 多組 config 選最快：

```python
from flydsl.autotune import Config, autotune

@autotune(
    configs=[
        Config(TILE_M=128, TILE_N=128, TILE_K=64, STAGES=2, num_warps=4, waves_per_eu=2),
        Config(TILE_M=256, TILE_N=128, TILE_K=64, STAGES=3),
        Config(TILE_M=128, TILE_N=256, TILE_K=64, STAGES=2, maxnreg=128),
    ],
    key=['m', 'n', 'k'],     # 這些值不同就重新調
    warmup=5, rep=25,
)
@flyc.jit
def launch_gemm(..., TILE_M: fx.Constexpr[int], ...):
    ...
```

- `Config(**kwargs)`：`kwargs` 注入 jit 的 constexpr/runtime 參數；`num_warps`、`waves_per_eu`、`maxnreg` 是特殊的 compiler hints。
- `key`：cache key 的依據（通常是 shape）。同 key 只調一次，結果存 `~/.flydsl/autotune/<fn>.json`。
- `do_bench` 用 torch CUDA/HIP event 計時 → **這一步會跑 kernel，需要 GPU**。
- `compiler_opts`（waves_per_eu/maxnreg）經 `CompilationContext.compile_hints()` 注入。

> autotune 一定要在沒有其它效能測試時跑，否則 (a) 互相干擾結果不準 (b) 佔用 GPU。

---

## 4.6 優化方法論（可重複的流程）

1. **先求對，再求快**：`SPLIT_K=1`、`STAGES=2`、同步 load、無 scheduler hints，先過數值驗證。
2. **量測 roofline**：算這個 shape 是 memory-bound 還是 compute-bound，決定優化方向。
3. **向量化 global load**：確保 128-bit（`dwordx4`）載入、coalesce（階段 2）。
4. **LDS swizzle**：消除 bank conflict（階段 2）。
5. **Multi-stage pipeline**：藏 HBM 延遲（4.2）；調 `STAGES`。
6. **Async copy（buffer_load_lds）**：省暫存器、增 overlap（階段 2.4）。
7. **Scheduler hints**：交錯 vmem/ds/mfma（4.3）。
8. **Split-K / tile 形狀**：針對 shape 調（4.4）。
9. **Autotune**：把 3–8 的旋鈕交給 autotuner 掃（4.5）。

每步都用 profiler（`rocprof` / `omniperf`）確認瓶頸有移動，避免盲調。

---

## 4.7 其它值得讀的工作區 kernel（依難度）

| 難度 | 檔案 | 學什麼 |
|------|------|--------|
| 入門 | `kernels/swiglu_and_mul.py` | elementwise、buffer load/store、interleave layout |
| 入門 | `kernels/reduce.py` | LDS reduction、shuffle、fly layout 代數 |
| 中 | `kernels/silu_and_mul_fq.py`、`kernels/quant_utils.py` | 量化、fp8 |
| 中 | `kernels/mfma_epilogues.py` | MFMA C-fragment 對應、CShuffle epilogue |
| 中高 | `kernels/splitk_hgemm.py` | 完整 GEMM、pipeline、split-K、scheduler |
| 高 | `kernels/mfma_preshuffle_pipeline.py`、`kernels/preshuffle_gemm.py` | preshuffle 權重、進階 pipeline |
| 高 | `kernels/fused_compress_attn.py`、`kernels/flash_attn_func_gfx1201.py` | attention、online softmax |
| 高 | `kernels/chunk_gated_delta_h.py`、`linear_attention_*.py` | linear attention |

---

## 4.8 自我檢核

- [ ] 我能從 tile 形狀推導出 warp 分工與 global load 的 TV 參數
- [ ] 我理解 multi-stage pipeline 如何用 LDS double-buffer 重疊 load 與 MFMA
- [ ] 我看得懂 `for ... yield` 是 runtime loop 帶 loop-carried 累加器
- [ ] 我知道 scheduler hints / s_waitcnt 的作用，但會留到最後再調
- [ ] 我能描述 split-K 的清零 + signal + atomic add 流程
- [ ] 我會用 autotune，且知道它需要 GPU

讀完進入 [`05_exercises.md`](./05_exercises.md)。
