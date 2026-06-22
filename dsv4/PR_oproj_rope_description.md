# [AMD] Improve DeepSeek-V4 prefill: CK o-proj GEMM + fused attention-output RoPE

## Motivation

On AMD (MI355X / gfx950), DeepSeek-V4-Pro prefill (tp8 + dp-attention) lagged the
reference ATOM engine. After matching the comm path (gatherv + reduce_scatter) and
confirming MoE GEMM / collectives are already at parity (shared aiter kernels), a
symmetric per-rank pure-prefill profile (OSL=1, both engines single-stream, same
16384-tokens/rank chunk) localized the residual ~16% prefill gap to **two
engine-specific kernels that SGLang ran slower than ATOM**, both in the MLA
attention block:

1. **MLA output / q / kv projection w8a8-block FP8 GEMM uses the Triton kernel.**
   `apply_w8a8_block_fp8_linear` picks `triton_gemm_a8w8_blockscale` for the MLA
   projection shapes (they are in the hard-coded
   `use_aiter_triton_gemm_w8a8_tuned_gfx950` list). On gfx950 + HIP≥7.2 the CK
   bpreshuffle kernel (`gemm_a8w8_blockscale_bpreshuffle`, which aiter dispatches to
   the `ck_tile QuantGemm`) is faster for these shapes — this is also what ATOM uses
   (ATOM explicitly treats the Triton FP8 blockscale GEMM as slower).

2. **Attention-output inverse RoPE falls back to a per-token strided kernel on HIP.**
   `fused_rope_inplace(o[..., -rd:], ...)` uses a single fused CUDA kernel on NV but
   on HIP falls back to `apply_rotary_emb_triton`, which launches one program per
   token and does strided (2i / 2i+1) interleaved loads. ATOM's `inverse_rope_gptj`
   batches tokens and uses coalesced contiguous loads.

Per-layer trace (pa_prefill-windowed) confirmed these were the only two non-shared
kernels materially slower than ATOM; everything else (pa_prefill MLA attn,
moe1/moe2/reduce, all-gatherv/reduce-scatterv comm, kv/q projections, mhc norms) was
already at parity.

## Modifications

`python/sglang/srt/layers/quantization/fp8_utils.py`:
- Add module toggle `_FORCE_CK_W8A8` (default `False`) + `set_force_ck_w8a8()`.
  `use_aiter_triton_gemm_w8a8_tuned_gfx950(n,k)` returns `False` when the toggle (or
  the `SGLANG_FORCE_CK_W8A8` env override) is set, so the dense w8a8-block GEMMs go
  through the CK bpreshuffle path instead of Triton.

`python/sglang/srt/layers/deepseek_v4_rope.py`:
- Add `apply_rotary_emb_triton_kernel_batched` (BLOCK_M tokens/program) and
  `apply_rotary_emb_contig_kernel` (contiguous-load GPT-J rope via reshape+flip,
  mirrors ATOM's `inverse_rope_gptj`; supports forward + inverse).
- Add module toggle `_USE_BATCHED_ROPE` (default `False`) + `set_batched_rope()`.
  `apply_rotary_emb_triton` uses the contiguous kernel for the 3D (attention-output)
  case and the batched kernel otherwise when enabled (or `SGLANG_ROPE_BATCHED`).

`python/sglang/srt/models/deepseek_v4.py`:
- `DeepseekV4ForCausalLM.__init__` calls `set_force_ck_w8a8(True)` and
  `set_batched_rope(True)`, so DeepSeek-V4 gets both by default (no env needed). The
  env vars remain as overrides; other models are unaffected (defaults stay OFF).

## Accuracy (gsm8k 5-shot, lm_eval local-completions, tp8+dp8, MI355X)

| build | flexible-extract | strict-match |
|---|---:|---:|
| this PR (default-on) | 0.9477 | 0.9477 |

≈ the pre-PR baseline (~0.95); both kernels are numerically equivalent.

## Speed (DeepSeek-V4-Pro, MI355X, tp8 + dp-attention, ATOM bench client, ratio1.0)

Pure-prefill (ISL/OSL = N/1, conc 512, isolates prefill compute):

| ISL | baseline tok/s | this PR tok/s | gain | vs ATOM |
|---:|---:|---:|---:|---:|
| 1024 | 48,291 | 52,560 | +8.8% | 84% → 91% |
| 8192 | 47,013 | 51,049 | +8.6% | 84% → 91% |

End-to-end c512 (ISL/1024, conc 512):

| workload | baseline tok/s | this PR tok/s | gain | vs ATOM |
|---|---:|---:|---:|---:|
| 1k/1k | 17,233 | 17,809 | +3.3% | 86.9% → 90% |
| 8k/1k | 32,254 | 33,946 | +5.2% | 82.1% → 86% |

(End-to-end gain is smaller than pure-prefill because c512 is decode-bound; the
matched prefill only speeds up the prefill share of each step.)

### Commands

Server (DeepSeek-V4-Pro, tp8 + dp-attention; the two opts are default-on):
```bash
sglang serve --model-path <DeepSeek-V4-Pro> --trust-remote-code \
  --tp 8 --dp 8 --enable-dp-attention --enable-prefill-delayer \
  --disable-radix-cache --attention-backend dsv4 --page-size 256 \
  --mem-fraction-static 0.90 --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 131072 --cuda-graph-max-bs 512 --max-running-requests 512
```
Bench (per point):
```bash
python -m sglang.bench_serving --backend sglang-oai --model <DeepSeek-V4-Pro> \
  --dataset-name random --random-input-len {1024,8192} --random-output-len {1,1024} \
  --random-range-ratio 1.0 --num-prompts 4096 --max-concurrency 512 --request-rate inf
```
Accuracy:
```bash
lm_eval --model local-completions --tasks gsm8k --num_fewshot 5 \
  --model_args model=<DeepSeek-V4-Pro>,base_url=http://localhost:8000/v1/completions,num_concurrent=64
```

## Checklist

- [x] Correctness verified (gsm8k 5-shot 0.9477 ≈ baseline).
- [x] Default behavior for non-DSV4 models unchanged (toggles default OFF).
- [x] Env overrides (`SGLANG_FORCE_CK_W8A8`, `SGLANG_ROPE_BATCHED`) retained.
- [ ] CI / broader shape & shorter-context validation.
- [ ] Reviewers.
