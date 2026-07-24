# HANDOVER — aiter `gemm_a8w8_blockscale_preshuffle` Triton compile failure (gfx950)

Paste-ready handover for another agent that hit the same
`gemm_a8w8_blockscale_preshuffle` compile crash.

---

## 問題：aiter `gemm_a8w8_blockscale_preshuffle` Triton kernel 在 gfx950 編譯失敗

環境：ROCm、8×MI355X（gfx950）、DeepSeek-V4-Pro、aiter 拉到 `origin/main`
（如 `874840aef`）、flydsl 0.2.4、`triton-custom` 3.6.0
（`/sgl-workspace/triton-custom`，`triton.__version__==3.6.0`）。啟動 ATOM 或
SGLang 的 DSV4 服務、在 forward / cuda-graph capture 階段崩潰。

症狀（traceback 末段）：
```
File ".../ATOM/atom/model_ops/linear.py", line 881, in forward
  y = gemm_a8w8_blockscale_preshuffle_impl(...)
File ".../aiter/ops/gemm_op_a8w8.py", line ~921, in gemm_a8w8_blockscale_bpreshuffle
  return _gemm_a8w8_blockscale_preshuffle_triton(...)
File ".../aiter/ops/triton/gemm/basic/gemm_a8w8_blockscale.py", line 390, in gemm_a8w8_blockscale_preshuffle
  impl[grid](...)
File ".../triton-custom/python/triton/backends/amd/compiler.py", line 269, in make_ttgir
  pm.run(mod, 'make_ttgir')
RuntimeError: PassManager::run failed
```

根因：升級後的 aiter 帶了新的 **Triton 版 a8w8 blockscale GEMM kernel**，這顆
kernel 用這個環境 pin 的 `triton-custom 3.6.0` 在 `make_ttgir`（TTGIR pass）
階段編不過。也就是 **新 aiter 的 triton kernel 與舊的 triton-custom 不相容**。
是否走 triton 由 aiter tuned CSV（`AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE`，
merge 後在 `/tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv`）裡
該 (gfx,cu,M,N,K) 條目的 `libtype` 欄位決定；DSV4 的形狀被 tune 成
`libtype=triton`，沒有現成 env 可覆寫成 CK/asm。

分派邏輯在 `aiter/ops/gemm_op_a8w8.py`：`get_CKGEMM_config()` 讀 CSV，若
`config["libtype"]=="triton"` 就走 triton；否則（`cktile/ck/asm/opus` 或
`config is None`）走 CK/asm，最後還有 `try: gemm_a8w8_blockscale_bpreshuffle_ck(...)`
這個預設 CK fallback。

## 兩條解法

### 1. 繞道（快，已驗證可用）
把該 triton 分支 env-gate 掉，讓它 fall through 到預設 CK 路徑。
在 `aiter/ops/gemm_op_a8w8.py` 頂部加 `import os`，並把
```python
if config is not None and config["libtype"] == "triton":
```
改成
```python
if (config is not None and config["libtype"] == "triton"
    and os.environ.get("AITER_DISABLE_BLOCKSCALE_TRITON", "0") != "1"):
```
然後啟動時設環境變數 `AITER_DISABLE_BLOCKSCALE_TRITON=1`。
- env-gated、預設關、不影響未設該 env 的執行。
- 對效能 A/B 公平（該顆 linear 改走 CK，ON/OFF 都一樣）。
- 缺點：那顆 GEMM 不是最佳 triton kernel（用 CK 取代）。

### 2. 正解
把 `triton-custom` 升級到與新 aiter 相符的版本（讓 `make_ttgir` 能編這顆
blockscale GEMM kernel），或改用官方 `rocm/atom-dev:latest` docker（內含相符的
aiter/flydsl/triton），就不需要繞道。

## 驗證
套繞道後 DSV4 服務可正常起、跑完整 benchmark；不設該 env 則回到原本的
`PassManager::run failed`。

## 相關背景
本問題是「把 ATOM coalescer 就地跑起來」升級 aiter→main 後冒出的最後一層相依
不相容（aiter → flydsl 0.2.4 → triton-custom）。完整串接見同目錄
`PROBLEMS.md` 第 6 節。
