# MORI EPv2 新 Docker Image 環境建立 Handover

更新日期：2026-08-03  
目標平台：單機 8× AMD Instinct MI355X（gfx950）  
目標模型：DeepSeek-V4-Pro A8W4，TP8 / DP8 / EP8

## CONTINUE HERE

在新 image 內依序完成：

1. 確認 ROCm / PyTorch / Triton runtime；
2. 安裝 FlyDSL 0.2.4；
3. checkout Aiter `bee50d97a`，且必須初始化 submodules；
4. 安裝 patched MORI `e491aea0`；
5. checkout SGLang EPv2 integration `cf2df6266`；
6. 跑 import、CPU、8-GPU graph correctness；
7. 最後才啟動 DeepSeek-V4-Pro server。

不要使用 MORI upstream `main` 或 SGLang upstream `main` 取代下列 pinned
commits。SGLang branch 依賴 patched MORI API：

- `prepare_recv_cap()`；
- `dispatch(..., recv_cap=..., clone_routing=False)`；
- dispatch payload publication fence。

## 1. Exact revisions

### SGLang

```text
Repository  https://github.com/HaiShaw/sglang
Branch      feat/mori-epv2
Commit      cf2df6266ce4b69b771d304801f5a13333796159
```

### MORI

```text
Repository  https://github.com/kkHuang-amd/mori
Branch      feat/epv2-sglang-stability
Commit      e491aea0de6c351468d6f11d5fa1dd391aff51c2
```

### Aiter

```text
Repository  https://github.com/ROCm/aiter
PR ref      refs/pull/4433/head
Commit      bee50d97a7e74b796bdac8ef0247ecb6706132c3
```

`bee50d97a` 沒有 remote branch 指向它；先 fetch PR ref，ancestor commit
才會出現在本機 object database。不要直接使用 PR head `8525fc016`：
它 merge 了較新的 A8W8 decode route，在此 ROCm/Triton runtime 曾觸發
`TritonAMDGPUConvertToBufferOps` / `PassManager::run failed`。

### FlyDSL

```text
Package  flydsl==0.2.4
```

不要讓 dependency resolver 升級到 FlyDSL 0.3.x。

## 2. 已驗證 runtime

```text
Python    3.10.12
PyTorch   2.9.1 + ROCm 7.2
HIP       7.2.26015
Triton    3.6.0 (ROCm build)
GPU       MI355X / gfx950 / 256 CU
```

先確認 base image：

```bash
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, p.gcnArchName, p.multi_processor_count)
PY

hipcc --version
rocm-smi --showproductname --showmeminfo vram --showpids
```

預期看到 8 GPUs、`gfx950`、每張 256 CU。若 PyTorch 不是 ROCm build，
先修正 base image，不要在後續步驟 workaround。

## 3. 建議系統套件

MORI source build 至少需要：

```bash
apt-get update
apt-get install -y \
  build-essential cmake ninja-build git pkg-config \
  python3-dev libpci-dev libibverbs-dev ibverbs-utils libnuma-dev
```

使用 virtualenv：

```bash
python3 -m venv /opt/venv
source /opt/venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install pybind11
```

若 base image 已經有正確 ROCm PyTorch，不要讓之後的 `pip install`
重新安裝 PyPI CUDA/NVIDIA torch。

## 4. 安裝 FlyDSL

```bash
source /opt/venv/bin/activate
python -m pip install --no-deps "flydsl==0.2.4"

python - <<'PY'
import importlib.metadata
import flydsl
print("flydsl:", importlib.metadata.version("flydsl"), flydsl.__file__)
assert importlib.metadata.version("flydsl") == "0.2.4"
PY
```

## 5. Checkout Aiter

```bash
mkdir -p /workspace
cd /workspace

git clone https://github.com/ROCm/aiter.git
cd aiter
git fetch origin refs/pull/4433/head:refs/remotes/origin/pr-4433
git checkout bee50d97a7e74b796bdac8ef0247ecb6706132c3
git submodule update --init --recursive

git rev-parse HEAD
git submodule status
```

`git submodule status` 的 `3rdparty/composable_kernel` 前面不可是 `-`。
已驗證的 CK revision 為：

```text
af7118e342580ecd3f71edce7b1d0ba465012ecf
```

本環境以 source + JIT 方式使用 Aiter：

```bash
export AITER_REPO=/workspace/aiter
export PYTHONPATH="${AITER_REPO}:${PYTHONPATH:-}"
```

第一次 server startup 會 build 多個 Aiter modules，可能需要數分鐘。
不要在 build 中途 kill container；中斷可能留下
`aiter/jit/build/lock_module_*` 或 nested `build/lock`。

## 6. Checkout 並安裝 MORI

```bash
cd /workspace
git clone \
  --branch feat/epv2-sglang-stability \
  https://github.com/kkHuang-amd/mori.git \
  mori-epv2

cd /workspace/mori-epv2
git checkout e491aea0de6c351468d6f11d5fa1dd391aff51c2
git submodule update --init --recursive

python -m pip install . --no-build-isolation
```

驗證安裝來源和 API：

```bash
cd /tmp
python - <<'PY'
import inspect
import mori
from mori.cco import Communicator
from mori.ops.dispatch_combine_v2 import (
    EpDispatchCombineConfig,
    EpDispatchCombineOp,
)

print("mori:", mori.__version__, mori.__file__)
print("dispatch:", inspect.signature(EpDispatchCombineOp.dispatch))
assert hasattr(EpDispatchCombineOp, "prepare_recv_cap")
assert "recv_cap" in inspect.signature(EpDispatchCombineOp.dispatch).parameters
assert "clone_routing" in inspect.signature(EpDispatchCombineOp.dispatch).parameters
print("MORI EPv2 API: OK")
PY
```

不要把另一份 `/workspace/mori/python` 放在 `PYTHONPATH` 前面，否則可能
shadow 掉剛安裝、含 CCO extension 的 patched package。

## 7. Checkout SGLang integration

```bash
cd /workspace
git clone \
  --branch feat/mori-epv2 \
  https://github.com/HaiShaw/sglang.git \
  sglang-mori-epv2

cd /workspace/sglang-mori-epv2
git checkout cf2df6266ce4b69b771d304801f5a13333796159
```

此 SGLang revision 的 `pyproject.toml` 會 pin 較新的 generic torch。對已準備
好的 ROCm image，建議沿用 image 內 SGLang dependencies，並只做 no-deps
editable install：

```bash
python -m pip install -e ./python --no-deps
```

或完全不 install，直接：

```bash
export SGLANG_REPO=/workspace/sglang-mori-epv2
export PYTHONPATH="${SGLANG_REPO}/python:${AITER_REPO}:${PYTHONPATH:-}"
```

驗證 backend registration：

```bash
python - <<'PY'
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.server_args import MOE_A2A_BACKEND_CHOICES

b = MoeA2ABackend("mori-epv2")
assert b.is_mori_epv2()
assert not b.is_mori()
assert b.supports_aiter()
assert "mori-epv2" in MOE_A2A_BACKEND_CHOICES
print("SGLang MORI EPv2 registration: OK")
PY
```

## 8. Network interface

選擇 container 內可用的 host interface，不要硬抄舊機器的
`enp81s0f1`：

```bash
ip -br link
ip route

export IFACE=<實際介面名稱>
export MORI_SOCKET_IFNAME="${IFACE}"
export GLOO_SOCKET_IFNAME="${IFACE}"
export NCCL_SOCKET_IFNAME="${IFACE}"
```

單機仍需要 Gloo/CCO bootstrap 可正常連線。

## 9. Correctness gates

### CPU tests

```bash
cd /workspace/sglang-mori-epv2

SGLANG_USE_AITER=1 \
PYTHONPATH="${SGLANG_REPO}/python:${AITER_REPO}" \
pytest -q test/registered/unit/layers/moe/test_aiter_runner.py

SGLANG_USE_AITER=0 \
PYTHONPATH="${SGLANG_REPO}/python" \
pytest -q \
  test/registered/scheduler/test_prefill_delayer.py::TestPrefillDelayerNegotiate::test_negotiate
```

預期：

```text
test_aiter_runner.py: 8 passed
TestPrefillDelayerNegotiate::test_negotiate: 1 passed
```

### 8-GPU adapter + graph replay

```bash
cd /workspace/sglang-mori-epv2

export SGLANG_USE_AITER=0
export SGLANG_MORI_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_MORI_EPV2_PER_RANK_VMM_GB=4
export FLYDSL_RUNTIME_CACHE_DIR=/tmp/flydsl-epv2-validation-cache

PYTHONPATH="${SGLANG_REPO}/python:${AITER_REPO}" \
GRAPH_REPLAYS=1000 \
torchrun --standalone --nproc_per_node=8 \
  test/manual/dsv4/test_mori_epv2_dispatcher.py

PYTHONPATH="${SGLANG_REPO}/python:${AITER_REPO}" \
GRAPH_REPLAYS=1000 SKEWED=1 EMPTY_LAST_RANK=1 \
torchrun --standalone --nproc_per_node=8 \
  test/manual/dsv4/test_mori_epv2_dispatcher.py

PYTHONPATH="${SGLANG_REPO}/python:${AITER_REPO}" \
torchrun --standalone --nproc_per_node=8 \
  test/manual/dsv4/test_mori_epv2_graph_tiers.py
```

預期：

```text
MORI-EPV2-SGLANG-IDENTITY: PASS
MORI-EPV2-SGLANG-GRAPH: PASS replays=1000
EPV2-GRAPH-TIERS: failures=0 on all ranks
EPV2-GRAPH-LIFECYCLE: failures=0 on all ranks
```

完成後確認 GPU cleanup：

```bash
rocm-smi --showpids --showmeminfo vram
```

預期沒有 KFD PIDs，idle VRAM 約 0.3 GB/GPU。

## 10. Production server environment

```bash
source /opt/venv/bin/activate

export MODEL=/dockerx/data/deepseek-ai/DeepSeek-V4-Pro
export PORT=8000
export SGLANG_REPO=/workspace/sglang-mori-epv2
export AITER_REPO=/workspace/aiter

export PYTHONPATH="${SGLANG_REPO}/python:${AITER_REPO}:${PYTHONPATH:-}"
export FLYDSL_RUNTIME_CACHE_DIR=/tmp/flydsl-epv2-serving-cache

export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
export SGLANG_OPT_USE_FUSED_COMPRESS=true
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export SGLANG_OPT_FP8_WO_A_GEMM=false
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=false
export SGLANG_OPT_USE_TOPK_V2=false
export SGLANG_OPT_USE_AITER_INDEXER=true
export SGLANG_OPT_USE_TILELANG_INDEXER=false
export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
export SGLANG_OPT_USE_TILELANG_MHC_POST=false
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false
export SGLANG_ROCM_USE_MULTI_STREAM=false
export AITER_BF16_FP8_MOE_BOUND=0
export AITER_FLYDSL_EP_NO_FAKE_EXPERT=1
export SGLANG_EAGER_INPUT_NO_COPY=true
export SGLANG_PREFILL_DELAYER_MIXED_SLOT_GUARD=1

export SGLANG_SHARED_EXPERT_TP1=0
export SGLANG_DP_SHARED_EXPERT_LOCAL=0
export SGLANG_DP_USE_GATHERV=0
export SGLANG_DP_USE_REDUCE_SCATTER=0

export SGLANG_MORI_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=8192
export SGLANG_MORI_EPV2_PER_RANK_VMM_GB=4
```

啟動：

```bash
python3 -m sglang.launch_server \
  --model-path "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --tp 8 \
  --ep-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend mori-epv2 \
  --deepep-mode normal \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --load-balance-method round_robin \
  --attention-backend dsv4 \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 256 \
  --swa-full-tokens-ratio 0.15 \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 65536 \
  --cuda-graph-max-bs 1024 \
  --max-running-requests 1024 \
  --disable-radix-cache \
  --disable-shared-experts-fusion \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --enable-prefill-delayer
```

Startup 期間應看到：

```text
MORI EPv2 init ... world=8 ... hidden=7168 ... topk=6 ... recv_cap=65536
MORI EPv2 graph cap ... recv_cap=512/1024
Application startup complete
The server is fired up and ready to roll!
```

第一次 Aiter JIT build 可能使 startup 延長數分鐘。不要關閉 CUDA graph
作為正式 workaround。

## 11. Smoke benchmark

```bash
PYTHONPATH="${SGLANG_REPO}/python" \
python3 -m sglang.bench_serving \
  --backend sglang-oai \
  --base-url "http://127.0.0.1:${PORT}" \
  --model "${MODEL}" \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --random-range-ratio 1.0 \
  --num-prompts 512 \
  --max-concurrency 256 \
  --request-rate inf \
  --warmup-requests 0 \
  --output-file /tmp/mori_epv2_512.jsonl
```

本輪參考值（只作 sanity range，非正式 CI）：

```text
Successful requests       512 / 512
Total token throughput    11,569.56 tok/s
Output token throughput    1,285.51 tok/s
Median TTFT                   75.63 s
Median TPOT                  126.89 ms
```

Standalone FlyDSL 同輪約 30,970 tok/s。EPv2 目前可正確 serving，但效能
不應視為 production winner。

## 12. 已知限制

1. MORI EPv2 目前只整合 intranode `world_size <= 8`。
2. 第一輪只支援 synchronous non-TBO。
3. Transport baseline 為 BF16 dispatch + BF16 gather combine。
4. Eager prefill 的 physical receive cap 為 65,536 rows；combine 仍有大型
   full-buffer staging/copy 成本。
5. Decode graph 使用 DP-synchronized 512/1024 logical caps。
6. EPv2 必須使用 op-owned live routing map；不要改回 allocator-managed
   routing clone。
7. 上游 optional `bf16-scatter-fp8blockwise-scaled` correctness arm仍有
   hidden mismatch，不屬於第一輪 BF16 gather baseline。
8. 不要啟用 TBO，直到 non-TBO combine staging / critical path 已改善。

## 13. 常見失敗

### `Unsupported flydsl version`

```text
expected >=0.2.4, got 0.2.2
```

修正：

```bash
python -m pip install --force-reinstall --no-deps flydsl==0.2.4
```

並確認舊 package path 沒有排在 `PYTHONPATH` 前方。

### `No module named mori.cco.cco`

代表 Python 正在讀 MORI source tree，但 CCO extension沒有安裝。移除 stale
`PYTHONPATH=/path/to/mori/python`，重新：

```bash
cd /workspace/mori-epv2
python -m pip install . --no-build-isolation
cd /tmp
python -c 'from mori.cco import Communicator; print(Communicator)'
```

### Aiter JIT 永久等待 baton

通常是先前 build 被 kill，留下 zero-byte lock。先確認沒有任何 Aiter build
process，再刪除對應 stale lock：

```text
/workspace/aiter/aiter/jit/build/lock_module_*
/workspace/aiter/aiter/jit/build/<module>/build/lock
```

不要在仍有 builder process 時刪 lock。

### CUDA graph replay stale output / GPU memory fault

先確認 MORI 與 SGLang exact commits。若 SGLang 調用的 MORI API沒有
`clone_routing=False`，或 MORI dispatch沒有每-warp release publication，
代表載入了錯誤 package。

```bash
python - <<'PY'
import inspect
from mori.ops.dispatch_combine_v2 import EpDispatchCombineOp
print(inspect.signature(EpDispatchCombineOp.dispatch))
print(inspect.getsourcefile(EpDispatchCombineOp))
PY
```

## 14. 最終 manifest

環境建好後保存：

```bash
{
  echo "=== git ==="
  git -C /workspace/sglang-mori-epv2 rev-parse HEAD
  git -C /workspace/mori-epv2 rev-parse HEAD
  git -C /workspace/aiter rev-parse HEAD
  git -C /workspace/aiter submodule status

  echo "=== python ==="
  python - <<'PY'
import importlib.metadata
import torch
import mori
import flydsl
import aiter

print("torch", torch.__version__, "hip", torch.version.hip)
print("mori", mori.__version__, mori.__file__)
print("flydsl", importlib.metadata.version("flydsl"), flydsl.__file__)
print("aiter", aiter.__file__)
PY

  echo "=== gpu ==="
  rocm-smi --showproductname --showmeminfo vram --showpids
} | tee /tmp/mori_epv2_environment_manifest.txt
```

## 15. Related artifacts

```text
SGLang branch:
https://github.com/HaiShaw/sglang/tree/feat/mori-epv2

MORI branch:
https://github.com/kkHuang-amd/mori/tree/feat/epv2-sglang-stability

Original integration/performance handover:
/tmp/sglang-mori-epv2-compare/MORI_EPV2_INTEGRATION_HANDOVER.md
```

