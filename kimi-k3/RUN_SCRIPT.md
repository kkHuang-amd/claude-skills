# Kimi-K3 run script

Launcher:

```text
/workspace/useful-scripts/benchmarking/kimi-k3/launch_server.sh
```

## Merged defaults

- Model: `/shared_nfs/huggingface_models/moonshotai/Kimi-K3`
- TP: 8
- Decode DCP: 8 with the AITER attention backend
- Listen: `0.0.0.0:8000`
- Static memory fraction: `0.93`
- Maximum running requests: `8`
- Decode CUDA Graph maximum batch size: `8`
- Prefill context parallelism: disabled
- Radix Cache: disabled for the long-context run
- Chunked prefill and maximum prefill tokens: `8192`
- Mamba full-memory ratio: `0.3`
- Mamba SSM dtype: `bfloat16`
- Mamba tracking interval: `1024`
- INT8 Mamba checkpoint and cache reporting: enabled
- Kimi-K3 reasoning and tool-call parsers: enabled
- AITER K3, FlyDSL, SituV2 A8W4, and FlyDSL AR norm optimizations: enabled

## Decisions from the two supplied commands

- Use the current canonical `--tp-size` spelling instead of `--tp`.
- Use `--cuda-graph-max-bs-decode`; `--cuda-graph-max-bs` is a deprecated
  alias in the checked SGLang source.
- Keep the graph batch at 8 because the tested request cap is 8. Raising the
  request cap also enlarges Kimi-K3 Mamba state/checkpoint pools.
- For the `ISL≈68k`, `OSL≈350` case, enable prefill CP with `zigzag` sequence
  splitting and disable Radix Cache. This favors one long prefill over
  shared-prefix reuse.
- Keep the initial prefill chunk at `32768`. DSV4's documented global-to-rank
  chunk division applies to DP-attention and is not evidence that Kimi-K3
  prefill CP divides this setting the same way.
- BEAM still benefits from Radix Cache. Restore its previous behavior with
  `ENABLE_PREFILL_CP=0 RADIX_CACHE=1`.
- Keep `0.0.0.0` for container access. Port 8000 preserves compatibility with
  the companion benchmark scripts; use `PORT=30000` or `PORT=8200` when needed.

Example throughput override:

```bash
MAX_RUNNING_REQUESTS=256 \
CUDA_GRAPH_MAX_BS_DECODE=256 \
./launch_server.sh
```

Current DCP validation defaults are equivalent to:

```bash
DCP_SIZE=8 \
ATTENTION_BACKEND=aiter \
ENABLE_PREFILL_CP=0 \
RADIX_CACHE=0 \
./launch_server.sh
```

## Validation note

The 2026-08-04 8x MI355X A/B proved the path is correct but not performant:
throughput regressed 74-82% at concurrency 1-8. Explicit attention CP increased
per-GPU weights from 194.38 GB to 253.20 GB and reduced cache capacity to one
68k request, serializing the workload. Use `ENABLE_PREFILL_CP=0` for production.
See `PREFILL_CP_EXPERIMENT.md` for commands, results, and artifacts.
