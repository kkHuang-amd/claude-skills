# Kimi-K3 TP8 ROCm trace analysis

## Scope

- 8x MI355X, TP8
- DCP disabled
- DSPARK disabled
- Triton attention backend
- Radix Cache disabled
- Fixed 8,192 input / 1,024 output, one concurrency-32 wave
- Local Kimi-K3 branch includes PR 33599 attention-residual fusion

Two traces were captured:

- CUDA Graph production trace for representative kernel composition and
  production step timing.
- CUDA Graph disabled, GPU+CPU-stack trace for reliable per-kernel duration and
  Python callsite attribution.

## Production decode step

Server logs report approximately 842 generated tok/s at batch 32, equivalent to
about 38.0 ms per decode step. The rank-median representative graph step is
40.1 ms, consistent with the server measurement.

Rank-median GPU breakdown:

- Dense GEMMs: 8.48 ms, 21.1%, 698 launches.
- MoE stage 1: 6.95 ms, 17.3%, 92 launches.
- Full-attention Triton decode: 6.35 ms, 15.8%, 24 launches.
- MoE stage 2: 4.11 ms, 10.2%, 92 launches.
- MoE route/pack/quant: 2.97 ms, 7.4%, 368 launches.
- PyTorch-native kernels: 2.86 ms, 7.1%, 622 launches.
- AITER custom all-reduce: 2.33 ms, 5.8%, 187 launches.
- KDA packed decode: 1.84 ms, 4.6%, 69 launches.
- Attention-residual aggregation: 1.66 ms, 4.1%, 186 launches.
- Activation/add: 0.80 ms, 2.0%, 185 launches.
- Other and sampling: 1.75 ms, 4.4%.

MoE route plus its two GEMM stages is about 14.0 ms (34.9%). Dense GEMMs and
full attention add another 14.8 ms (36.9%). Most of the step is therefore
already in specialized AITER/FlyDSL/Triton/HIP kernels, not naive PyTorch.

## PyTorch-native callsites

The eager CPU-stack trace maps the largest native GPU materializations to:

- `kimi_k3.py::_forward_routed`: fallback `latent.copy_(expert_output)`,
  about 0.34 ms/step and 92 copies.
- `topk.py::biased_grouped_topk_gpu`: per-layer correction-bias dtype copy,
  about 0.32 ms/step and 92 copies.
- `fused_norm_gate.py::LayerNormGatedFunction.forward`: output/residual copy,
  about 0.27 ms/step and 69 copies.
- `forward_mla.py::forward_absorb_core`: `torch.cat` of MLA Q/K components,
  about 0.28 ms/step and 48 concatenations.
- `memory_pool.py::set_kv_buffer`: indexed KV write, about 0.13 ms/step and
  24 indexed writes.
- `forward_mla.py` weight scaling/materialization: repeated `to(...)*scale`
  and copies, about 0.30 ms/step across 24 MLA layers.

The directly attributable set is about 1.6 ms/step. The full production trace
contains 2.86 ms of PyTorch-native kernels, so replacing every native operation
has a hard ceiling near 7%; a realistic first pass is about 3-5%.

## Optimization experiments

All experiments used TP8, DCP/DSPARK disabled, Triton attention, Radix Cache
disabled, and fixed 8,192 input / 1,024 output at concurrency 32. Early KDA and
MoE measurements were 32-request single-wave screens. MLA and the final
combined decision used the stricter matched methodology: 64 warmups and 256
measured requests.

### Retained: repeated MoE copies

Commit `e9d8cb9472` caches the AITER top-k correction-bias cast and makes the
routed MoE path honor its caller-provided output buffer.

- PyTorch-native launches: 622 to 530 (-92).
- PyTorch-native GPU time: 2.86 to 2.45 ms/step (-0.40 ms).
- Output throughput: +0.64%.
- Median TPOT: -0.54%.
- Median ITL: -0.47%.

The production graph trace in `copy-optim-ab/profile` confirms the launch and
GPU-time reductions; this change is retained.

### Screened: ROCm KDA fused decode

The kernel oracle passed. The initial single-wave screen improved output
throughput by 0.82%, median TPOT by 0.78%, and median ITL by 0.71%. Because this
screen used a different short-run launch and workload length, its percentages
are provisional and cannot be added directly to the full MLA result.

### Screened: Triton MLA preparation fusion

The AITER fused RoPE/QK concat/cache-write operation was safely dispatched for
the plain and static hybrid MLA pools. Unit oracles covered BF16/FP8 cache bits,
batch 1/32, fallback gates, and graph address stability. A 200-example GSM8K
run scored 0.990.

The matched full serving A/B result was:

- Fused: 524.16 output tok/s, 46.98 ms median TPOT, 37.90 ms median ITL.
- Unfused: 523.41 output tok/s, 47.07 ms median TPOT, 37.99 ms median ITL.
- Delta: +0.14% output throughput and about -0.19% TPOT/ITL.

The gain is below normal serving variance. Raw results remain in
`/sgl-workspace/aiperf-results/tp8-triton-mla-{fused,unfused}-8192-1024-c32`.

### Final combined decision

KDA and MLA were restored together on top of the retained MoE change. Eighteen
tests and ten subtests passed. The combined 200-example GSM8K run scored 0.980
(196/200), within two samples of the prior 0.990 run.

The authoritative matched 64-warmup/256-request comparison against the same
MoE-only baseline was:

- MoE-only: 523.41 output tok/s, 47.069 ms median TPOT, 37.988 ms median ITL.
- MoE + KDA + MLA: 524.61 output tok/s, 47.042 ms median TPOT,
  37.938 ms median ITL.
- Delta: +0.23% output throughput, -0.06% TPOT, and -0.13% ITL.

The combined result demonstrates that the earlier standalone percentages were
not additive. KDA and MLA remain below measurement noise after controlling the
launch and sample count, so both experiments were removed again. The MoE copy
optimization remains retained. Combined artifacts are in
`/sgl-workspace/aiperf-results/tp8-triton-moe-kda-mla-combined-8192-1024-c32`.

## Recommended optimization order

1. Cache the fallback BF16-scaled `w_kc` tensor instead of rebuilding
   `w_kc.to(bfloat16) * w_scale` every decode step.
2. Inspect the gated norm custom-autograd wrapper and return the Triton output
   directly during inference to remove its copy.

These are lower-risk launch/materialization reductions. They should be tested
as one change at a time with a kernel oracle and a short concurrency-32 serving
A/B.

The larger remaining opportunities require kernel work rather than replacing
naive PyTorch:

- Tune the Triton MLA grouped decode kernel (15.8% of step).
- Improve or fuse MoE routing and the FlyDSL stage-1/stage-2 path (34.9%).
- Port the K3 AR/norm fusion strategy to MI355X; the existing implementation is
  SM100/SM103-only, so ROCm keeps a multi-stage MoE front/tail.
- Reduce or overlap the 187 custom all-reduce launches (5.8%), while avoiding
  interpreting collective spin-wait as transfer time.
- Dense GEMMs are 21.1%; gains require better GEMM selection/fusion rather than
  a generic Triton rewrite.

## Final reproduction commands

Launch the retained TP8 configuration:

```bash
DCP_SIZE=1 ATTENTION_BACKEND=triton MAX_RUNNING_REQUESTS=32 \
CUDA_GRAPH_MAX_BS_DECODE=32 RADIX_CACHE=0 \
ENABLE_INT8_MAMBA_CHECKPOINT=1 ENABLE_CACHE_REPORT=1 \
bash /dockerx/var/amdsgl/kk/workspace/useful-scripts/benchmarking/kimi-k3/launch_server.sh
```

Run the standard serving workload:

```bash
python -m sglang.benchmark.serving \
  --backend sglang-oai --base-url http://127.0.0.1:8000 \
  --dataset-name random --model /dockerx/data/models/Kimi-K3 \
  --tokenizer moonshotai/Kimi-K3 --num-prompts 256 \
  --random-input-len 8192 --random-output-len 1024 \
  --random-range-ratio 1.0 --max-concurrency 32 \
  --warmup-requests 64 --seed 42
```

## Trace caveats

- CUDA Graph traces expose one representative captured kernel set. Kernel
  counts and the representative production step are useful; whole-trace wall
  spans are not.
- Eager decode is about 256 tok/s (roughly 125 ms/step), versus 842 tok/s with
  graphs. Eager aggregate time includes thousands of host launches and should
  not be used as the production performance target.
- Custom all-reduce durations vary strongly by rank because of barrier/spin
  time. The production graph result and rank-median interval should be used,
  not the slowest-rank raw sum.

## Artifacts

- Graph traces: `/sgl-workspace/kimi-k3-traces/tp8-graph`
- Eager GPU traces: `/sgl-workspace/kimi-k3-traces/tp8-eager`
- Eager stack traces: `/sgl-workspace/kimi-k3-traces/tp8-eager-stack`
- Machine-readable summary:
  `/sgl-workspace/kimi-k3-traces/trace_summary.json`
- Structured graph-on/stack-trace mapping:
  `/sgl-workspace/kimi-k3-traces/structured_analysis.txt`
