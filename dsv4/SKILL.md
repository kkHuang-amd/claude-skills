---
name: dsv4-atom-serving-perf-sweep
description: Benchmark DeepSeek-V4-Pro serving performance on ATOM (ROCm, 8xMI355X, TP8) and compare clients/engines. Use when measuring tok/s, TTFT, TPOT at fixed (ISL, OSL, concurrency) points, when driving an ATOM OpenAI-compatible server with either ATOM's own benchmark_serving or SGLang's bench_serving, or when hitting the two interop gotchas: (1) sglang import crash from cohere2_moe.py @strict on huggingface_hub>=1.x, and (2) sglang.bench_serving KeyError 'choices' against ATOM's usage-only final SSE chunk.
---

# DeepSeek-V4-Pro serving perf sweep on ATOM (MI355X, TP8)

Playbook for measuring DeepSeek-V4-Pro serving perf at fixed
**(ISL, OSL, concurrency)** points on 8xMI355X (gfx950), and comparing the
**SGLang bench client** vs **ATOM's native bench client** against the *same*
running server. Mirrors the InferenceX-style gpt-oss sweep so numbers are
cross-engine comparable.

Scripts live in `useful-scripts/benchmarking/dsv4/`:
`run_atom_dsv4.sh` (server), `bench_dsv4.py` (sglang client wrapper),
`sweep_dsv4_atom_client.sh`, `sweep_dsv4_sglang_client.sh`, `lm-eval.sh`.

## 1. Launch the ATOM server (TP8, FP8 KV)

```bash
ATOM_DISABLE_MMAP=true ATOM_MOE_GU_ITLV=1 AITER_BF16_FP8_MOE_BOUND=0 \
python3 -m atom.entrypoints.openai_server \
  --model /dockerx/data/deepseek-ai/DeepSeek-V4-Pro/ \
  --server-port 8000 -tp 8 --kv_cache_dtype fp8 \
  --trust-remote-code --enable-dp-attention
```

- DeepSeek-V4-Pro is 64 safetensors shards → cold load ~9-10 min. Poll
  `http://127.0.0.1:8000/health` (200) before benching; do NOT use the
  gpt-oss `wait_server.sh` (it greps the sglang-specific "server is fired up").
- `/v1/models` returns the model id = the **path** you passed; use that exact
  string as `--model` for the bench clients (tokenizer + request "model" field).
- **Accuracy caveat (important):** the official ROCm/ATOM recipe says
  `ATOM_USE_TRITON_MOE=1` is **required** for V4-Pro, else it silently falls
  back to a numerically wrong MoE path (GSM8K ~0.95 → ~0.6). The launch above
  (PD-disagg env set, no triton-moe) is fine for *relative* client/engine perf
  comparison on one server, but absolute numbers / accuracy may be off. Verify
  with `lm-eval.sh` (gsm8k, flexible-extract should be ~0.95).

## 2. Two interop gotchas (both fixed without editing the bench logic)

### 2a. sglang import crash: cohere2_moe.py `@strict`
`python -m sglang.bench_serving` (and `sglang.launch_server`) fail at import:
```
StrictDataclassDefinitionError: Class 'Cohere2MoeConfig' must be a dataclass before applying @strict
```
Cause: `srt/configs/cohere2_moe.py` applies huggingface_hub's `@strict` (present
in hf_hub >=1.x, e.g. 1.18.0) to a class that is not a `@dataclass`; newer
hf_hub also treats inherited `validate_*` methods (e.g. `PreTrainedConfig.
validate_rope`) as strict validators and rejects them. **Repo fix** (preferred):
drop `@strict`, add `@dataclass`:
```python
from dataclasses import dataclass
def strict(cls):  # no-op; @strict only adds runtime field validation
    return cls
@strict
@dataclass
class Cohere2MoeConfig(PreTrainedConfig):
    ...
```
Runtime-only fallback (no repo edit): monkeypatch
`huggingface_hub.dataclasses.strict` to identity *before* importing sglang
(see `bench_dsv4.py`).

### 2b. sglang.bench_serving KeyError 'choices' vs ATOM
ATOM's streaming `/v1/completions` ends with a usage-only SSE chunk that has
**no `choices` key**:
```
data: {"id": ..., "usage": {"prompt_tokens":..,"completion_tokens":..}}
data: [DONE]
```
`async_request_openai_completions` assumes `data["choices"][0]["text"]` always
exists → `KeyError: 'choices'` during warmup. Fix without touching the repo:
wrap `json.loads` so usage/summary chunks get an empty `choices=[{"text":""}]`;
the existing falsy-text guard then skips them. Output length is unaffected
(comes from `--random-output-len` with ignore_eos on). This is an **ATOM-only**
quirk — not needed against an SGLang server. Implemented in `bench_dsv4.py`.

## 3. Run the sweep (client-only; server must already be up)

Two clients, same grid, same server → isolate client-vs-server effects.

```bash
# SGLang bench client (InferenceX-style) via the wrapper
WORKLOADS="8192:1024" CONCS="128 256" bash sweep_dsv4_sglang_client.sh
# ATOM native bench client (ROCm recipe, num-prompts=CONC*8)
WORKLOADS="8192:1024" CONCS="128 256" bash sweep_dsv4_atom_client.sh
```

Fixed params (keep constant for comparability): `--random-range-ratio 1.0`
(fixed lengths, InferenceX default), `--request-rate inf` (closed loop),
`--ignore-eos` (exact OSL — verified: total generated == prompts*OSL),
`num_prompts = conc*8`, `warmups = conc*2`. Against an SGLang server set
`BENCH="python3 -m sglang.bench_serving"` to skip the wrapper.

ATOM native client equivalent (recipe + the *8 fix + matched warmup):
```bash
python -m atom.benchmarks.benchmark_serving \
  --model=$MODEL --backend=vllm --base-url=http://localhost:8000 \
  --dataset-name=random --random-input-len=$ISL --random-output-len=$OSL \
  --random-range-ratio=1.0 --num-prompts=$((CONC*8)) --max-concurrency=$CONC \
  --num-warmups=$((CONC*2)) --request-rate=inf --ignore-eos \
  --save-result --percentile-metrics="ttft,tpot,itl,e2el"
```

## 4. Reference numbers captured (2026-06-09, this server config)

DeepSeek-V4-Pro, TP8, FP8 KV, ISL=8192/OSL=1024, same ATOM server both rows.
(NOTE: launched WITHOUT `ATOM_USE_TRITON_MOE=1` — see §1 caveat.)

| conc | client | total tok/s | tok/s/gpu | Med TTFT (ms) | Med TPOT (ms) | Med E2E (ms) |
|---:|---|---:|---:|---:|---:|---:|
| 128 | ATOM   | 21,526 | 2,691 |  9,392 | 43.5 | 54,280 |
| 128 | SGLang | 20,937 | 2,617 |  9,087 | 45.9 | 55,898 |
| 256 | ATOM   | 30,942 | 3,868 | 18,921 | 55.4 | 75,764 |
| 256 | SGLang | 29,781 | 3,723 | 15,747 | 61.9 | 78,549 |

Takeaways: the two clients agree on throughput within ~3% (ATOM client reports
slightly higher) → SGLang-client numbers are trustworthy for cross-engine
comparison. TTFT diverges more at conc=256 (ATOM higher) due to client-side
dispatch differences (ATOM does an extra single-prompt probe + burstiness
wrapper); TPOT goes the other way. 8k prefill at high concurrency is
prefill-queue bound (TTFT 9s→19s from 128→256).

## 5. Cleanup (frees ALL VRAM)

Killing the launcher PID + the port listener is NOT enough — ATOM spawns 8 DP
EngineCore subprocesses (multiprocessing-fork, reparented) that keep ~293 GB/GPU
resident. Kill the whole tree:
```bash
PID=$(lsof -ti :8000); kill -9 $PID
# then kill the DP engine cores + their parents (find via rocm-smi --showpids
# and `ps -eo pid,ppid,cmd | grep multiprocessing-fork`)
rocm-smi --showmeminfo vram   # confirm back to ~0.3 GB/GPU
```

## Key takeaways (transferable)

1. ATOM is OpenAI-compatible → any `/v1/completions` bench client works, but its
   streaming tail chunk omits `choices`; guard client parsers for that.
2. Compare bench *clients* against one server first to validate the client; only
   then compare *engines*. Here the two clients agreed within ~3%.
3. `--ignore-eos` is mandatory for apples-to-apples OSL; verify
   `total_generated == num_prompts * OSL`.
4. For V4-Pro correctness on ATOM, `ATOM_USE_TRITON_MOE=1` is required — perf
   sweeps without it are only valid as relative comparisons.
5. ATOM DP servers leave EngineCore subprocesses holding VRAM; always kill the
   whole process tree and confirm with `rocm-smi`.
