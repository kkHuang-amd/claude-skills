# Kimi-K3 handover

> Current handover for the next FP8 KV / AITER DCP investigation:
> `HANDOVER_2026-08-07.md`.
>
> This file retains the earlier public-weight, DCP-port, KVV, and Docker
> transition history. New chats should start with the 2026-08-07 handover.

## Current state

- The old weights at `/shared_nfs/huggingface_models/kmd` were deleted.
- Latest public weights were downloaded from
  `moonshotai/Kimi-K3` to `/shared_nfs/huggingface_models/moonshotai/Kimi-K3`.
- Download verification: about 1.6 TB, `config.json`,
  `model.safetensors.index.json`, and all 96 safetensors shards are present.
- The benchmark checkout is
  `/sgl-workspace/kvv-bench/kvv-k3-0727-update`.
- The original ad-hoc launcher is `/sgl-workspace/sglang/run_kmd.sh`.
- Reusable handover scripts are in
  `/workspace/useful-scripts/benchmarking/kimi-k3`.
- No server or benchmark should be running at handover time. Verify before
  switching Docker images.

## AMD decode DCP port validation (2026-08-04/05)

- Target code: `/sgl-workspace/sglang`, `kimi-k3@e6311f7559`.
- Port status: the six `k3_dcp` commits have been semantically ported,
  including AITER Gluon decode/prefill paging, DCP8 correctness fixes, and
  AITER DCP + DSpark static target-verify.
- Targeted CPU tests: `9 passed`.
- TP8 non-DCP baseline (`dcp_size=1`, Triton, 200 GSM8K questions, 5-shot,
  temperature 0): score `0.985`, latency `33.095 s`, output throughput
  `606.373 token/s`.
- Baseline log: `/tmp/kimi-k3-gsm8k-tp8-baseline.log`.
- DCP8 AITER server reached `ready to roll` with `page_size=32`,
  `dcp_comm_backend=ag_rs`, replicated Q disabled, 655,584 token capacity,
  and memory profile weights/KV/graph = `194.4/16.88/0.96 GB` per GPU.
- DCP8 chat smoke and `/generate` batches 1 and 8 passed.
- First DCP8 launch exposed an existing ROCm JIT failure from the unconditional
  `cuda_fp8.h` include in `situ_and_mul.cuh`; the include is now CUDA-guarded,
  and a standalone ROCm compile/run parity test passed.
- DCP8 GSM8K (same 200 questions, 5-shot, temperature 0): score `0.985`,
  latency `117.757 s`, output throughput `170.988 token/s`. Accuracy exactly
  matches the TP8 non-DCP baseline and exceeds the `0.95` acceptance threshold.
- DCP8 GSM8K log: `/tmp/kimi-k3-gsm8k-dcp8.log`.
- Successful DCP8 server log:
  `/tmp/kimi-k3-dcp8-aiter-server-next.log`.
- With `CUDA_GRAPH_MAX_BS_DECODE=4` and `MAX_RUNNING_REQUESTS=8`, graph batch 2
  and eager batch 5 both completed successfully.
- DCP8 + AITER + DSpark static verify reached full warmup. Chat smoke and an
  8-request generation batch passed; per-request observed accept lengths ranged
  from `2.13` to `4.57` on the short synthetic prompts.
- DSpark server log: `/tmp/kimi-k3-dcp8-aiter-dspark-server.log`.
- All validation servers were stopped. No benchmark is active.

### CONTINUE HERE

- Status: port and requested validation are complete.
- No continuation is required unless the changes should be committed or a
  longer DSpark acceptance/performance benchmark is requested.

## Why the Docker image is being replaced

The newly downloaded public weights loaded and served, but the current image
printed many AITER warnings for Kimi-K3 GEMM shapes such as `M=4913`:

```text
not found tuned config in /tmp/aiter_configs/bf16_tuned_gemm.csv,
will use default config! using torch solution:0
```

The new image should be checked for compatible Kimi-K3 support and tuned AITER
configs. Record the exact image name and digest before benchmarking.

## Docker mount checklist

The container needs:

```text
/shared_nfs/huggingface_models/moonshotai/Kimi-K3
/sgl-workspace/kvv-bench/kvv-k3-0727-update
/workspace/useful-scripts/benchmarking/kimi-k3
```

Recommended Docker runtime flags:

```text
--ipc=host
--shm-size 32g
all 8 GPUs/devices exposed
port 8000 published if the client runs outside the container
```

The server recipe uses ROCm/AITER-specific environment variables. Confirm they
exist in the replacement image instead of silently dropping them.

## Model facts relevant to the benchmark

- Architecture: 2.8T MoE, 104B activated, native multimodal.
- Quantization: MXFP4 weights / MXFP8 activations.
- Context length: 1,048,576 tokens.
- Kimi-K3 always thinks. The model card documents top-level
  `reasoning_effort=low|high|max`.
- The 2026-07-27 KVV checkout sends open-source effort through
  `chat_template_kwargs`:

  ```json
  {
    "thinking": true,
    "preserve_thinking": true,
    "thinking_effort": "max"
  }
  ```

Validate that the replacement SGLang image supports this request format before
running the full suite.

## Validated server recipe on the old image

```bash
SGLANG_USE_AITER=1 \
SGLANG_KDA_FUSED_DECODE=0 \
SGLANG_AITER_K3_OPT=1 \
AITER_FLYDSL_FORCE=1 \
AITER_SITUV2_A8W4=1 \
sglang serve \
  --model-path /shared_nfs/huggingface_models/moonshotai/Kimi-K3 \
  --trust-remote-code \
  --enable-multimodal \
  --tp 8 \
  --attention-backend triton \
  --dtype bfloat16 \
  --mem-fraction-static 0.90 \
  --max-running-requests 128 \
  --cuda-graph-max-bs 128 \
  --host 0.0.0.0 \
  --port 8000
```

Keep Radix Cache enabled. It is essential for BEAM because each of 35
conversations is reused by 20 probing questions.

## Startup sequence

Cold loading 96 shards took roughly 5–6 minutes in prior runs. SGLang then
starts Uvicorn and performs a vision warmup.

`Application startup complete` is not sufficient. Sending 1M-token BEAM
requests at that point blocked the internal warmup for 600 seconds, after which
SGLang killed itself with exit code 137. Wait for:

```text
The server is fired up and ready to roll!
```

Then verify:

```bash
curl -sS http://localhost:8000/v1/models
```

The expected ID is `/shared_nfs/huggingface_models/moonshotai/Kimi-K3` when that local path is passed to
the launcher.

## KVV benchmark setup

```bash
cd /sgl-workspace/kvv-bench/kvv-k3-0727-update
uv sync
uv pip install -e .

export KIMI_BASE_URL=http://localhost:8000/v1
export KIMI_API_KEY=EMPTY
```

Use `run_kvv.sh` for the exact commands. Parameters selected from the 0727 K3
README:

| Benchmark | Max tokens | Temperature | Top-p | Effort | Connections |
|---|---:|---:|---:|---|---:|
| OCRBench | 16384 | 1.0 | 0.95 | max | 50 |
| MMMU-Pro | 98304 | 1.0 | 0.95 | max | 50 |
| Tool-call | 32768 | 1.0 | 0.95 | max | 50 |

Run order: OCRBench, MMMU-Pro, Tool-call, then BEAM.

## Baseline results from the deleted pre-release weights

These results are historical references only. Do not report them as results
from `/shared_nfs/huggingface_models/moonshotai/Kimi-K3`.

| Benchmark | Measured | Screenshot reference |
|---|---:|---:|
| OCRBench | 0.902 (stderr 0.009) | API 0.89 / open source 0.891 |
| MMMU-Pro | 0.782 (stderr 0.010) | API 0.82 / open source 0.818 |
| Tool-call | 0.965 (stderr 0.017) | API 0.83 / open source 0.752 |

The first attempt to retest OCRBench with the public weights was interrupted
before producing a score. There is no valid public-weight result yet.

## BEAM 1M

BEAM uses two stages:

1. `beam_generate.py` generates 700 answers.
2. `beam_judge.py` scores them with a separate judge endpoint.

Generation requirements:

- Keep Radix Cache enabled.
- Use `/shared_nfs/huggingface_models/moonshotai/Kimi-K3` as `--tokenizer`.
- Do not use the bundled Kimi-K2.6 tokenizer. It undercounted some K3 prompts
  by about 22K tokens and produced context-length 400 errors.
- Wait for full server warmup before starting.
- Use a new output filename for each Docker image/weight combination.
- The previous old-weight output
  `beam/answers_kmd_max.jsonl` must not be reused.

With working Radix Cache, logs showed about 980K cached tokens for repeated
questions in the same conversation. The rough generation estimate was
10–12 hours. Judging time is additional and requires judge model credentials.

## DeepSWE

DeepSWE was not run. The KVV README only says:

- 113 DeepSWE v1.1 tasks.
- Pier >= 0.3.0.
- Kimi Code >= 0.23.6 registered as a Pier agent.

The workspace did not contain Pier, Kimi Code, or DeepSWE. General setup:

```bash
git clone https://github.com/datacurve-ai/deep-swe
uv tool install datacurve-pier
```

Kimi Code agent registration and container-to-model networking still need to
be configured.

## Result hygiene

For every run, save:

- Docker image name and digest.
- SGLang version/commit.
- AITER version/commit.
- `/v1/models` output.
- Full server command and server log.
- Full benchmark command and `.eval`/JSONL output.
- Any retries, HTTP errors, context truncation, or fallback-kernel warnings.

Never merge partial outputs across different weights, images, sampling
parameters, or tokenizer settings.

