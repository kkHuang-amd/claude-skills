# [AMD] Fix the dsv4 performance of MoE issue

## Motivation

On AMD (MI355X / gfx950), the DeepSeek-V4 FP4 MoE path runs the aiter
`fused_moe` kernels (`mfma_moe1_silu_mul`, `mfma_moe2_afp8_wfp4_bf16_cshuffle`)
noticeably slower than the reference ATOM engine for the same model/workload.

Root cause: the aiter FP4 MoE kernel requires the per-partition intermediate
size to be 256-aligned, so during weight loading we pad it
(`moe_intermediate_size_per_partition` 384 → 512 for DSV4 at TP8). The padded
weight/scale tensors are produced correctly, **but the pad amount was never
propagated to `fused_moe`**. `AiterMoeQuantInfo.intermediate_pad` defaulted to
`0`, so the kernel treated the whole padded 512 region as real and computed the
128 zero-padded channels every step. ATOM passes `intermediate_pad = 128` and
only computes the real 384, hence the gap.

A controlled microbench (same `aiter.fused_moe`, identical weights/routing,
only `intermediate_pad` varied) isolates the cost:

| run                              | moe1     | moe2    |
|----------------------------------|----------|---------|
| `intermediate_pad=0`   (current) | 127.1 µs | 98.3 µs |
| `intermediate_pad=128` (fixed)   | 104.4 µs | 79.7 µs |

i.e. ~18% on the gate/up GEMM and ~19% on the down GEMM, purely from skipping
the padded computation. With matched routing + pad, the SGLang and ATOM kernels
are then identical (104 µs vs 104 µs, 55 µs vs 55 µs at low/high active-expert
counts respectively), confirming there is no remaining kernel/layout gap.

While here, the FP4 weight/scale shuffling was also cleaned up: the legacy
`shuffle_*_a16w4` helpers (with a hard-coded gate/up-interleave flag) are
replaced by the unified `shuffle_scale` / `shuffle_weight` driven by
`SGLANG_USE_AITER_MOE_GU_ITLV` (default `True`, so default behavior is
unchanged), removing dead code and making the interleave layout configurable.

## Modifications

`python/sglang/srt/layers/quantization/fp8.py`:

1. **Propagate the intermediate padding to `fused_moe`**
   - In `process_weights_after_loading_block_quant` (FP4 expert path), record
     `layer.intermediate_pad = padded_inter - inter_per_part` and
     `layer.hidden_pad = 0`.
   - In `maybe_get_hip_aiter_quant_info`, pass
     `hidden_pad` / `intermediate_pad` into `AiterMoeQuantInfo` so
     `fused_moe(..., intermediate_pad=128, hidden_pad=0)` computes only the real
     intermediate channels (was implicitly `0`).

2. **Unify FP4 weight/scale shuffle (remove `*_a16w4`, honor gu-interleave env)**
   - Imports: drop `shuffle_scale_a16w4` / `shuffle_weight_a16w4`, keep
     `shuffle_scale` / `shuffle_weight`.
   - Add `self.gu_intv = envs.SGLANG_USE_AITER_MOE_GU_ITLV.get()` in
     `Fp8MoEMethod.__init__`.
   - Scale shuffle: `shuffle_scale(scale_2d, num_experts, self.gu_intv,
     is_w13_scale)`.
   - Weight shuffle: `shuffle_weight(..., is_guinterleave=self.gu_intv,
     gate_up=True/False)` for w13/w2.

No API or config changes for users; defaults preserve existing behavior aside
from the (faster) correct `intermediate_pad`.

## Accuracy Tests

The padded intermediate channels are zero-filled weights/scales, so the
gate/up outputs for those channels are zero and contribute nothing to the
down-projection. Computing them (`pad=0`) vs skipping them (`pad=128`) is
therefore **numerically equivalent by construction** — the fix removes wasted
compute, not real contributions.

GSM8K (5-shot, 200 questions, DSV4-Pro, TP8, same server config; only
`intermediate_pad` toggled):

| build | Accuracy | Invalid |
|-------|---------:|--------:|
| before (`intermediate_pad=0`)   | 0.970 | 0.000 |
| after  (`intermediate_pad=128`) | 0.965 | 0.000 |

The 1-question delta (0.970 vs 0.965) is within run-to-run FP/scheduling noise
(the two paths use different GEMM M-tiling, so borderline tokens can flip); no
accuracy regression. The shuffle unification is a no-op at the default
`SGLANG_USE_AITER_MOE_GU_ITLV=True` (same interleave as the previous hard-coded
path).

**Reproduce** (`before` = base branch, `after` = this PR; launch the server, then
run GSM8K against it):

```bash
# Server (TP8)
SGLANG_USE_AITER=1 AITER_BF16_FP8_MOE_BOUND=0 \
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4-Pro --trust-remote-code --tp 8 \
  --attention-backend dsv4 --kv-cache-dtype fp8_e4m3 --page-size 256 \
  --chunked-prefill-size 16384 --cuda-graph-max-bs 512 --max-running-requests 512 \
  --mem-fraction-static 0.90 --disable-radix-cache --disable-shared-experts-fusion \
  --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 --port 8000

# GSM8K accuracy
python3 -m sglang.test.few_shot_gsm8k \
  --num-questions 200 --num-shots 5 --max-new-tokens 1024 --parallel 64 --port 8000
```

## Speed Tests and Profiling

**Kernel microbench** (MI355X, `aiter.fused_moe`, T=64 decode, DSV4 FP4,
identical weights/routing, only `intermediate_pad` varied):

| metric | pad=0 (before) | pad=128 (after) | Δ      |
|--------|---------------:|----------------:|--------|
| moe1 (`mfma_moe1_silu_mul`) | 127.1 µs | 104.4 µs | −17.9% |
| moe2 (`mfma_moe2_..._cshuffle`) | 98.3 µs | 79.7 µs | −18.9% |

**End-to-end** (SGLang, TP8, DSV4-Pro, cuda graph on, random ratio 1.0,
conc 64; pad value baked at graph capture):

| workload | metric | before (pad=0) | after (pad=128) | Δ |
|----------|--------|---------------:|----------------:|------|
| 8k/1k | output tok/s | 1346 | 1413 | +5.0% |
| 8k/1k | TPOT (ms) | 37.30 | 35.09 | −5.9% |
| 8k/1k | median ITL (ms) | 27.98 | 26.02 | −7.0% |
| 1k/1k | output tok/s | 2139 | 2274 | +6.3% |
| 1k/1k | TPOT (ms) | 28.22 | 26.36 | −6.6% |
| 1k/1k | median ITL (ms) | 27.66 | 25.48 | −7.9% |

Net: ~5–6% decode throughput and ~6–8% TPOT/ITL improvement at conc 64, with
the MoE kernels now on par with the ATOM reference at matched routing.

**Reproduce** — end-to-end (`before` = base branch, `after` = this PR; launch the
TP8 server as in the Accuracy section, then run `bench_serving`):

```bash
# 8k/1k, conc 64
python3 -m sglang.bench_serving --backend sglang-oai \
  --base-url http://127.0.0.1:8000 --model deepseek-ai/DeepSeek-V4-Pro \
  --dataset-name random --random-input-len 8192 --random-output-len 1024 \
  --random-range-ratio 1.0 --num-prompts 128 --max-concurrency 64 \
  --request-rate inf --warmup-requests 32

# 1k/1k, conc 64
python3 -m sglang.bench_serving --backend sglang-oai \
  --base-url http://127.0.0.1:8000 --model deepseek-ai/DeepSeek-V4-Pro \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 \
  --random-range-ratio 1.0 --num-prompts 192 --max-concurrency 64 \
  --request-rate inf --warmup-requests 32
```

**Reproduce** — kernel microbench: call `aiter.fused_moe` directly on a single
captured decode step (T=64) of DSV4 FP4 weights/scales/routing, varying only the
`intermediate_pad` argument (`0` vs `128`) and timing
`mfma_moe1_silu_mul` / `mfma_moe2_..._cshuffle` via `torch.profiler`:

```python
import torch
from aiter.fused_moe import fused_moe
from aiter import QuantType, ActivationType
from aiter.ops.flydsl.moe_common import GateMode
import torch.profiler as P

d = torch.load("moe_step.pt")  # hidden,w13,w2,w13_scale,w2_scale,topk_ids,topk_weights
g = lambda k: d[k].cuda()
def call(pad):
    return fused_moe(
        hidden_states=g("hidden"), w1=g("w13"), w2=g("w2"),
        topk_weight=g("topk_weights"), topk_ids=g("topk_ids"),
        quant_type=QuantType.per_1x32, activation=ActivationType.Silu,
        w1_scale=g("w13_scale"), w2_scale=g("w2_scale"),
        gate_mode=GateMode.INTERLEAVE.value, swiglu_limit=10.0, intermediate_pad=pad)

for pad in (0, 128):                       # 0 = before, 128 = after
    for _ in range(15): call(pad)
    torch.cuda.synchronize()
    with P.profile(activities=[P.ProfilerActivity.CUDA]) as prof:
        for _ in range(50): call(pad)
        torch.cuda.synchronize()
    for e in prof.key_averages():
        if "mfma_moe1_silu" in e.key or "mfma_moe2" in e.key:
            print(pad, e.key, e.device_time_total / e.count, "us")
```

Run with `AITER_BF16_FP8_MOE_BOUND=0` to select the flydsl MoE kernels.

## Checklist

- [x] Format code with pre-commit.
- [ ] Add unit tests (numerical-equivalence test for `intermediate_pad`).
- [ ] Update documentation (n/a — internal perf fix).
- [x] Provide accuracy and speed benchmark results (above): GSM8K 0.965 vs 0.970
  (no regression); moe1 −18% / moe2 −19%; +5–6% decode throughput.
- [x] Follow the SGLang code style guidance.
