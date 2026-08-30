# DeepSeek-V4-Pro @ 8x B200 (SGLang) — Benchmark Results

## Results — Lane A (random ISL=8192 / OSL=1024, ratio=1.0)

Bench settings: `num-prompts = conc*8`, `warmup = conc*2`, `request-rate=inf`, `--backend sglang`.

| workload | TP, DP | conc | total tok/s | tok/s/gpu | out tok/s | Med TTFT (ms) | Med TPOT (ms) | Med ITL (ms) | interact (tok/s/u) | Med E2E (ms) |
|----------|--------|------|-------------|-----------|-----------|---------------|---------------|--------------|--------------------|--------------|
| 8k/1k    | 8,8    | 128  | 29,860      | 3,732     | 3,318     | 6,976         | 31.89         | 26.12        | 31.4               | 39,444       |
| 8k/1k    | 8,8    | 256  | 41,858      | 5,232     | 4,651     | 6,898         | 48.17         | 29.82        | 20.8               | 56,306       |

Notes:
- `tok/s/gpu` = total tok/s / 8. `interact (tok/s/u)` = 1000 / Med TPOT.
- conc256 is throughput-saturated (~42k tok/s total); larger prompt counts raise TTFT stability, not peak throughput.
- MTP / speculative decoding was **NOT** enabled (server `speculative_algorithm=None`). Peak InferenceX dashboard points use MTP + full FP4.

## Test date

- 2026-07-06 (Asia/Taipei, UTC+8). Lane A rerun executed ~20:30–20:45.

## Versions

| component | version |
|-----------|---------|
| SGLang    | `0.0.0.dev1+gda802ddca` (commit `da802ddcafe55e25b3e1db86b1e0444afc3e05bc`, 2026-06-27) |
| Model     | `deepseek-ai/DeepSeek-V4-Pro` at `/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro` (config quant `fp8`, routed experts auto-detected `fp4`) |
| GPU       | 8x NVIDIA B200 (183 GB each) |
| KV cache dtype | `fp8_e4m3` |
| Attention backend | `dsv4` (DeepseekV4AttnBackend), page_size 256 |
| MoE       | `megamoe` (auto from `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1`), EP=8 |
| aiperf (agentic lane) | 0.8.0 (InferenceX submodule `cquil11/aiperf-agentx-v1.0`) |

## Commands

### Server launch (dp8 + megamoe + HiCache DRAM offload)

Launched via `useful-scripts/benchmarking/dsv4/run_sgl_dsv4_pro_b200.sh`:

```bash
MODE=dp8 CONC=256 HICACHE=on HICACHE_RATIO=8 bash run_sgl_dsv4_pro_b200.sh
```

Effective server command:

```bash
python3 -m sglang.launch_server \
  --model-path /shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro \
  --served-model-name DeepSeek-V4-Pro \
  --host 0.0.0.0 --port 8000 --trust-remote-code \
  --tp 8 --dp 8 --tokenizer-worker-num 8 \
  --enable-dp-attention --enable-dp-attention-local-control-broadcast \
  --incremental-streaming-output --stream-interval 20 \
  --dist-init-addr 127.0.0.1:10000 \
  --ep-size 8 --moe-a2a-backend deepep \
  --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
  --mem-fraction-static 0.88 --swa-full-tokens-ratio 0.1 \
  --max-running-requests 512 --cuda-graph-max-bs 64 --chunked-prefill-size 32768 \
  --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
  --watchdog-timeout 1800 --weight-loader-prefetch-checkpoints --enable-metrics \
  --enable-hierarchical-cache --hicache-ratio 8 --hicache-write-policy write_through \
  --hicache-io-backend direct --hicache-mem-layout page_first_direct
```

Env exported by the launch script:

```
PYTHONNOUSERSITE=1  TORCH_CUDA_ARCH_LIST=10.0  SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1  SGLANG_OPT_USE_JIT_NORM=1
SGLANG_OPT_USE_JIT_INDEXER_METADATA=1  SGLANG_OPT_USE_TOPK_V2=1
SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1  SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1
SGLANG_OPT_FIX_HASH_MEGA_MOE=1  SGLANG_OPT_USE_FAST_MASK_EP=1
SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1  SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096
SGLANG_OPT_FIX_NEXTN_MEGA_MOE=1  SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=0
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1   # (HiCache path)
```

Note: with `--enable-dp-attention` the server auto-adjusts `chunked-prefill-size` to 4096.

### Lane A benchmark (per row)

```bash
# conc 128
python3 -m sglang.bench_serving \
  --backend sglang --base-url http://localhost:8000 --model DeepSeek-V4-Pro \
  --dataset-name random --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \
  --num-prompts 1024 --max-concurrency 128 --warmup-requests 256

# conc 256
python3 -m sglang.bench_serving \
  --backend sglang --base-url http://localhost:8000 --model DeepSeek-V4-Pro \
  --dataset-name random --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \
  --num-prompts 2048 --max-concurrency 256 --warmup-requests 512
```

## Related artifacts

- Launch script: `/workspace/useful-scripts/benchmarking/dsv4/run_sgl_dsv4_pro_b200.sh`
- Raw bench outputs: `/workspace/useful-scripts/benchmarking/dsv4/bench_results/laneA_c*_np*.jsonl`
- Agentic lane (aiperf inferencex-agentx-mvp) driver + results:
  `run_agentic_replay.sh`, `agentic_results/` (see `agentic_results/FINDINGS.md`)

## Multi-stream A/B: SGLANG_ROCM_USE_MULTI_STREAM (Lane A, same settings)

NOTE: `SGLANG_ROCM_USE_MULTI_STREAM` is a ROCm/AMD-only flag; this is an NVIDIA
B200/CUDA box, so it is expected to be a no-op. Confirmed on the server process
env. Run 2026-07-06 ~20:55–21:20.

| conc | metric        | multi-stream ON (default) | SGLANG_ROCM_USE_MULTI_STREAM=0 |
|------|---------------|---------------------------|--------------------------------|
| 128  | total tok/s   | 29,860                    | 29,974                         |
| 128  | out tok/s     | 3,318                     | 3,330                          |
| 128  | Med TPOT (ms) | 31.89                     | 31.85                          |
| 128  | Med TTFT (ms) | 6,976                     | 6,732                          |
| 128  | Med E2E (ms)  | 39,444                    | 39,316                         |
| 256  | total tok/s   | 41,858                    | 39,922                         |
| 256  | out tok/s     | 4,651                     | 4,436                          |
| 256  | Med TPOT (ms) | 48.17                     | 50.45                          |
| 256  | Med TTFT (ms) | 6,898                     | 6,873                          |
| 256  | Med E2E (ms)  | 56,306                    | 58,796                         |

Conclusion: no meaningful difference. conc128 identical (~30k tok/s); conc256
delta (~5%) is within run-to-run variance (conc256 is throughput-saturated).
As expected, the ROCm multi-stream flag has no effect on the CUDA path.
Raw: bench_results/laneA_noms_c128_np1024_*.jsonl, laneA_noms_c256_np2048_*.jsonl
