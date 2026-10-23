# MI455 proxy MoE: 1K1K / 8K1K / 70K300

Host: `ctheliosmc-rck-g02-h13-16.rck.dcgpu` (gfx1250, 8 GPUs).

The 1K1K / 8K1K / 70K300 numbers were measured on:

| | |
|---|---|
| Image | `rocm/sgl-dev:v0.5.22-rocm101-mi45x-20261020-test` |
| Container | `cca_scaled_llm_mi455_6720` |
| Model | `/work/proxy_12l` (12-layer dummy MXFP4 checkpoint, already generated) |
| Parallelism | TP8 EP8 |
| AITER | `#6720`, plus `#6719` with commit `89dd89da` reverted |
| `mem-fraction-static` | `0.80` |

`rocm/sgl-dev:v0.5.22-rocm101-mi45x-20261022` was used only for single-GPU synthetic MXFP4 GEMM tuning (`cca_shared_mxfp4_tune_1022`). That container did not get `#6719` / `#6720`, and it did not serve the 12-layer model or run 1K1K.

`0.88` ran out of VRAM on the 8192-token prefill (RCCL allreduce left with about 16 MB). Use `0.80`.

## Login

```bash
ssh chienach@ctheliosmc-rck-g02-h13-16.rck.dcgpu
```

Paths below are under `/home/chienach`. Stop only containers named `cca_scaled_llm_*`.

## Container

```bash
docker run -d --name cca_scaled_llm_mi455_6720 \
  --network=host --ipc=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 --pids-limit=-1 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v /home/chienach/scaled_llm/proxy_moe_sglang:/proxy_moe_sglang:ro \
  -v /home/chienach/scaled_llm/work:/work \
  -v /home/chienach/scaled_llm/patches:/patches:ro \
  rocm/sgl-dev:v0.5.22-rocm101-mi45x-20261020-test \
  sleep infinity
```

If the container already exists: `docker start cca_scaled_llm_mi455_6720`.

## AITER patches

AITER in the image is `2bb549102`. Apply the two diffs, then restore the unshuffled GEMM config that `89dd89da` rewrites:

```bash
docker exec cca_scaled_llm_mi455_6720 bash -lc '
  set -e
  cd /sgl-workspace/aiter
  patch -p1 < /patches/aiter_pr6720.diff
  patch -p1 < /patches/aiter_pr6719.diff
  git checkout -- aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_afp4wfp4/DEFAULT.json
'
```

What remains from `#6719`:

- `dynamic_mxfp4_quant` falls back to Triton when `gl.amd.cdna5.scaled_downcast` is missing (Triton 3.8).
- `configs/gfx1250/triton/normalization/rmsnorm_large_m_small_n/DEFAULT.json` with `{"num_warps": 4, "num_stages": 2}`.

`#6720` sends `group_size == 32` and `N > 8192` down the split quant path. Do not also set `AITER_LARGE_HIDDEN_WORKAROUND=1`; that applies the same edit again.

## Environment

Inside the container, before either command:

```bash
cd /tmp
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1
export SGLANG_USE_AITER=1
export PYTHONUNBUFFERED=1
```

`cd /tmp` is required. The image `WORKDIR` is `/sgl-workspace`, which shadows `sglang.benchmark`.

`ROCR_VISIBLE_DEVICES` is unset, so the server uses all 8 GPUs. Port is `18030`.

## Server

One server per `(shape, concurrency)`. `MAX_BS` below is that concurrency.

1K1K and 8K1K (`CONTEXT=32768`):

```bash
python3 -m sglang.launch_server \
  --model-path /work/proxy_12l \
  --tp-size 8 --ep-size 8 \
  --kv-cache-dtype fp8_e4m3 \
  --context-length 32768 \
  --mem-fraction-static 0.80 \
  --max-running-requests "$MAX_BS" \
  --cuda-graph-max-bs-decode "$MAX_BS" \
  --chunked-prefill-size 8192 \
  --attention-backend triton \
  --watchdog-timeout 3600 \
  --host 127.0.0.1 --port 18030
```

70K300 uses the same command with `--context-length 131072`. Chunked prefill stays `8192`.

Wait until `curl -sf http://127.0.0.1:18030/health_generate` returns.

## Benchmarks

`random-ids`, exact lengths (`--tokenize-prompt`, `--random-range-ratio 1.0`). `ignore_eos` is left at the bench default.

| Shape | Input | Output | Context | Prompts | Warmup |
|---|---:|---:|---:|---|---|
| 1K1K | 1024 | 1024 | 32768 | concurrency × 8 | concurrency × 2 |
| 8K1K | 8192 | 1024 | 32768 | concurrency × 8 | concurrency × 2 |
| 70K300 | 70000 | 300 | 131072 | concurrency × 4 | concurrency × 2 |

Concurrencies that were swept: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 for 1K1K and 8K1K; 2, 4, 8, 16, 32 for 70K300. The commands below are concurrency 2, the first finished point of each shape. For another concurrency, set `MAX_BS` on the server to that value and substitute prompts / warmup from the table.

1K1K, concurrency 2 (16 prompts, warmup 4):

```bash
python3 -m sglang.bench_serving --backend sglang \
  --host 127.0.0.1 --port 18030 --tokenizer /work/proxy_12l \
  --dataset-name random-ids --tokenize-prompt \
  --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0 \
  --num-prompts 16 --max-concurrency 2 --warmup-requests 4 \
  --request-rate inf \
  --output-file /work/artifacts/isl1024-osl1024-c2.jsonl --output-details
```

8K1K, concurrency 2 (same server as 1K1K; 16 prompts, warmup 4):

```bash
python3 -m sglang.bench_serving --backend sglang \
  --host 127.0.0.1 --port 18030 --tokenizer /work/proxy_12l \
  --dataset-name random-ids --tokenize-prompt \
  --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \
  --num-prompts 16 --max-concurrency 2 --warmup-requests 4 \
  --request-rate inf \
  --output-file /work/artifacts/isl8192-osl1024-c2.jsonl --output-details
```

70K300, concurrency 2 (context 131072; 8 prompts, warmup 4):

```bash
python3 -m sglang.bench_serving --backend sglang \
  --host 127.0.0.1 --port 18030 --tokenizer /work/proxy_12l \
  --dataset-name random-ids --tokenize-prompt \
  --random-input-len 70000 --random-output-len 300 --random-range-ratio 1.0 \
  --num-prompts 8 --max-concurrency 2 --warmup-requests 4 \
  --request-rate inf \
  --output-file /work/artifacts/isl70000-osl300-c2.jsonl --output-details
```

Kill the server before the next point. Results from the sweep are under `/home/chienach/scaled_llm/work/artifacts/fixed_seq_len_12l_tp8ep8/` on the host (`/work/artifacts/fixed_seq_len_12l_tp8ep8/` in the container). 1K1K c=2 completed 16/16 at 154.5 output tok/s, median TTFT 114 ms, median ITL 12.85 ms.

## Difference from the `run_stage.sh` smoke bench

The older one-shot command (1024 in / 256 out, 64 prompts, concurrency 16, then `summarize_run.py`) is a different benchmark:

| | `run_stage.sh` smoke bench | 1K1K / 8K1K / 70K300 |
|---|---|---|
| Lengths | 1024 in / 256 out, once | 1024/1024, 8192/1024, 70000/300 |
| Prompts | 64 | concurrency × 8, or × 4 when input ≥ 32768 |
| Warmup / request rate / details | unset | warmup = concurrency × 2, `--request-rate inf`, `--output-details` |
| `mem-fraction-static` | script default 0.88 | 0.80 |
| Context | 32768 | 32768, or 131072 for 70K300 |
| `--max-running-requests` | 64 | the point's concurrency |
| `--cuda-graph-max-bs-decode` | unset | the point's concurrency |
| Smoke test | yes | no |
| AITER | `AITER_LARGE_HIDDEN_WORKAROUND=1`, including the GEMM config rewrite | `#6720` already applied; `89dd89da` GEMM config reverted |

Shared settings: `cd /tmp`, `SGLANG_USE_AITER=1`, `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1`, TP8 EP8, KV `fp8_e4m3`, `--attention-backend triton`, chunked prefill 8192, watchdog 3600, `random-ids`, `--tokenize-prompt`, range ratio 1.0, port 18030.
