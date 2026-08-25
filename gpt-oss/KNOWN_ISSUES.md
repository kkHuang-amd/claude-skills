# Known Issues — Plan A SHUFFLE 5D KV + `pa_decode_gluon` (gpt-oss-120b, ROCm AITER)

Branch: `feat/plan-a-5d-kv-pa-decode-gluon`

Scope: issues that affect the SHUFFLE 5D vectorized KV cache layout
(`SGLANG_KV_CACHE_LAYOUT=vectorized_5d`) + `pa_decode_gluon` decode path
added in this branch on top of SGLang's AITER backend.

---

## Correctness

### A1. TP=2 and TP=4 GSM8K is broken (P0, blocker)

`gpt-oss-120b` GSM8K accuracy collapses at TP=2 and TP=4 on the AITER
backend. TP=1 (this branch) is healthy and TP=8 has been verified
locally as correct. The breakage is **not** introduced by Plan A — the
NHD baseline regresses to the same degree at TP=2, and rolling back to
commit `104cb74761` (the last known-good pre-Plan-A reference) does not
fix it either.

| TP | Layout | KV dtype | extras                  | flexible | strict | status |
|---:|--------|----------|--------------------------|---------:|-------:|--------|
| 1  | 5D     | fp8_e4m3 | —                        | 0.8802   | 0.3306 | ✓      |
| 2  | 5D     | fp8_e4m3 | `--enable-aiter-allreduce-fusion` | 0.1092 | 0.0462 | ✗ |
| 2  | 5D     | fp8_e4m3 | —                        | 0.1766   | 0.0895 | ✗      |
| 2  | 5D     | bf16     | —                        | 0.0804   | 0.0379 | ✗      |
| 2  | NHD    | bf16     | —                        | 0.3230   | 0.1630 | ✗      |
| 2  | NHD    | fp8_e4m3 | `--enable-aiter-allreduce-fusion`, commit `104cb74761` | 0.0326 | 0.0121 | ✗ |
| 4  | —      | —        | (not run; expected to track TP=2) | — | — | ✗ |
| 8  | —      | —        | —                        | (user local) | — | ✓ |

Repro (TP=2):

```bash
SGLANG_KV_CACHE_LAYOUT=vectorized_5d HIP_VISIBLE_DEVICES=0,1 \
python3 -m sglang.launch_server \
  --model-path /path/to/gpt-oss-120b/ --tp 2 --trust-remote-code \
  --chunked-prefill-size 131072 --max-running-requests 128 --mem-fraction-static 0.85 \
  --prefill-attention-backend aiter --decode-attention-backend aiter --page-size 64 \
  --disable-radix-cache --host 127.0.0.1 --port 8000 --kv-cache-dtype fp8_e4m3 \
  --enable-aiter-allreduce-fusion
```

then

```bash
lm_eval --model local-chat-completions --apply_chat_template \
  --model_args model=/path/to/gpt-oss-120b/,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=65,max_retries=3,max_gen_toks=2048,tokenized_requests=False \
  --tasks gsm8k --num_fewshot 3
```

Suspected root cause: a FlyDSL-compiled **MoE** kernel (not attention)
whose tile / stride selection is correct at TP=1 (full expert set) and
at TP=8 (clean expert sharding), but breaks at TP=2 / TP=4 where the
per-rank expert / hidden shape lands on a different code path. This is
consistent with the observation that the failure is independent of the
KV cache layout (NHD baseline TP=2 also breaks), the KV dtype (bf16 and
fp8 both break), and the SGLang commit (pre-Plan-A `104cb74761` also
breaks) — i.e. the only common factor is the MoE path used at TP=2/4.

Next steps:
1. Run the TP=2 repro with the MoE backend swapped out (e.g.
   `--moe-runner-backend triton`, or any non-FlyDSL aiter MoE path)
   while leaving everything else identical. If accuracy comes back
   the FlyDSL MoE kernel suspicion is confirmed.
2. `grep -rni "flydsl\|fly_dsl" /sgl-workspace/aiter/` and focus on
   the MoE entry points; bisect by disabling them one at a time on
   TP=2.
3. File the reproducer upstream to AITER once narrowed.

Mitigation in this branch: none yet. Consider emitting a warning when
`SGLANG_KV_CACHE_LAYOUT=vectorized_5d` is combined with TP ∈ {2, 4}
until the upstream fix lands.

### A4. `forward_extend` fallback path on the 5D pool is broken (FIXED, commit `275d95bd5`)

The original fallback `mha_batch_prefill_func` paged call has been
replaced with a gather-and-linearize path: a new
`launch_gather_shuffle_5d_to_linear` triton kernel
(bit-exact inverse of `launch_reshape_and_cache_shuffle_5d`) pulls the
full per-request K/V from the 5D pool into a contiguous (T, H, D)
buffer in `store_dtype` (uint8 bytes for fp8, bf16 for bf16), then
runs `mha_batch_prefill_func` in 3D LINEAR mode (page_size=1) — which
*does* have a working kernel. fp8 K/V are forwarded raw to aiter
together with the per-tensor descales (aiter's LINEAR-mode prefill
supports fp8 K/V/Q natively), so no host-side dequant happens.

GSM8K with the original repro command (TP=1, page=64, chunked-prefill
still active):
  bf16 KV: 0.8863 / 0.3442
  fp8  KV: 0.8848 / 0.3374
  pre-A4 single-chunk baseline: 0.8802 / 0.3306

Original (now historical) description follows for context.

#### Original description (pre-fix)

The fast `_extend_no_prefix` shortcut in `AiterAttnBackend.forward_extend`
only fires when every request in the batch has zero
`extend_prefix_lens_cpu` (fully-fresh prompts). Whenever that condition
fails the code falls through to the legacy `mha_batch_prefill_func`
path that reads `k_cache` / `v_cache` from the 5D pool with an extra
`kv_last_page_lens` argument (lines ~2418-2466 in `aiter_backend.py`).

Two distinct issues live on that fallback path today:

1. **Runtime error**: the call crashes outright on this path. The user
   has a reproducer that triggers it even with `--disable-radix-cache`
   (i.e. it is **not** gated on prefix-cache reuse — chunked-prefill /
   the scheduler will still produce batches with non-zero
   `extend_prefix_lens_cpu` and land on the fallback path).
2. **Precision regression on SWA layers**: separately, when the
   fallback path does run to completion accuracy degrades on
   sliding-window layers — `swa_page_table` is selected but the
   `kv_last_page_lens` / page-table indexing semantics expected by
   `mha_batch_prefill_func` for the 5D SWA pool aren't right. This
   was observed during initial Plan A bring-up and was the reason
   the no-prefix shortcut was added in the first place.

Reproducer (crash):

```bash
# server
SGLANG_KV_CACHE_LAYOUT=vectorized_5d \
SGLANG_USE_AITER_MOE_GU_ITLV=False \
python3 -m sglang.launch_server \
  --model-path /path/to/gpt-oss-120b/ \
  --tp 1 --trust-remote-code \
  --mem-fraction-static 0.85 \
  --prefill-attention-backend aiter --decode-attention-backend aiter \
  --page-size 64 --disable-radix-cache --port 8000

# client
lm_eval --model local-chat-completions --apply_chat_template \
  --model_args model=/path/to/gpt-oss-120b/,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=65,max_retries=3,max_gen_toks=2048,tokenized_requests=False \
  --tasks gsm8k --num_fewshot 3
```

Workaround:
- `--disable-radix-cache` is **not** sufficient. Until this is fixed,
  any workload whose scheduler produces batches with non-zero
  `extend_prefix_lens_cpu` (chunked prefill mid-prompt, multi-turn,
  some disaggregation modes, …) will hit it.
- Single-shot fresh-prompt benchmarks where every request fits in one
  prefill chunk are still safe (they keep firing the no-prefix
  shortcut).

Next steps:
1. Capture the exact crash signature (stack + offending tensor shapes)
   from the GSM8K repro above.
2. Decide between (a) properly fixing the 5D fallback for SWA +
   non-zero prefix (correct `kv_last_page_lens` derivation, correct
   page-table indexing under the 5D layout) or (b) widening the
   no-prefix shortcut so it also handles batches with prefix by
   gather-concatenating cached pool K/V with the freshly arrived K/V
   into a 3D linear buffer and calling `mha_batch_prefill_func` in
   LINEAR mode.

### A5. GSM8K accuracy collapses at small decode batch on gpt-oss MXFP4 — `GPTOSS_SWIGLU_MXFP4_BF16_BOUND` (ROOT CAUSED, P1 — upstream AITER MoE)

**TWO compound issues. The second one is the real upstream bug.**

#### Issue 1 (P3, harness): `benchmark/gsm8k/bench_sglang.py` defaults `--max-new-tokens` to 512

This is too small for gpt-oss-120b on GSM8K 5-shot (the model
frequently produces longer chain-of-thought answers). The 512 cap
silently truncates those answers and the truncated string never
matches the GSM8K answer regex, so they count as wrong (Invalid).
Workaround: always run the harness with `--max-new-tokens 2048`.

#### Issue 2 (P1, upstream AITER): MoE bf16-activation path is buggy when decode batch < 256

`/sgl-workspace/aiter/aiter/fused_moe.py:326` selects the activation
quant dtype for the gpt-oss MXFP4 + SwiGLU + SEPARATED gate path
based on a hard threshold:

```python
q_dtype_a = dtypes.bf16 if M < _SWIGLU_MXFP4_BF16_BOUND else dtypes.fp4x2
```

where `M = topk_ids.shape[0]` = current decode batch size and
`_SWIGLU_MXFP4_BF16_BOUND = int(os.environ.get("GPTOSS_SWIGLU_MXFP4_BF16_BOUND", "256"))`.
SGLang forces `gate_mode = SEPARATED` for gpt-oss MXFP4 (via
`SGLANG_USE_AITER_MOE_GU_ITLV=False`, which is auto-set in
`server_args.py:2157`), so this dispatch hits exactly once per layer
per forward.

The bf16-activation path picked when `M < 256` produces measurably
wrong logits on gpt-oss-120b MXFP4. The error is small per layer,
but compounds across 36 layers and is amplified by TP all-reduce, so
the visible damage scales hard with TP:

| TP | `max_running` | `GPTOSS_SWIGLU_MXFP4_BF16_BOUND` | GSM8K (5-shot, `--max-new-tokens 2048`) | Avg tokens / req |
|---:|---:|---:|---:|---:|
| 1 | 128 | 256 (default) | 0.854 | ~650 |
| 1 | 256 | 256 (default) | 0.941 | ~630 |
| 8 | 128 | 256 (default) | **0.485** | **~1800** (model loops) |
| 8 | 256 | 256 (default) | 0.930 | ~630 |
| 8 | 128 | **128** (workaround) | **0.938** | ~580 |

Setting `GPTOSS_SWIGLU_MXFP4_BF16_BOUND=128` restores TP=8 +
`max_running=128` to baseline accuracy (0.938 vs the 0.930 of the
"OK" max_running=256 case).

The catastrophic TP=8 drop is consistent with a reasoning model
getting subtly wrong logits early in a chain, going off into
repetitive loops, and burning the entire `--max-new-tokens` budget
without ever reaching a parseable answer (avg tokens/req jumps from
~630 to ~1800).

Workaround:

```bash
# Force the fp4 MoE-activation path at small decode batch.
# Pick a value ≤ the smallest decode batch you expect at steady state.
export GPTOSS_SWIGLU_MXFP4_BF16_BOUND=128
```

**Don't set `BOUND=0` (always-fp4).** Verified to crash the server
on startup at TP=8 — `Rank 1 scheduler died during initialization
(exit code: -6)`. Root cause: SGLang's warmup / CUDA-graph capture
phase issues forwards at very small `M` (1, 2, …). AITER FlyDSL's
fp4-activation MoE kernel does not ship pre-tuned configs for those
tiny `M` values, so `get_2stage_cfgs` returns no dispatch and the
worker aborts. Any small positive bound that still lets warmup-time
small-`M` calls fall into the bf16 branch is safe (`128` is a
known-good value for our gpt-oss-120b setup; warmup output is not
used for accuracy).

Verified matrix:

| `GPTOSS_SWIGLU_MXFP4_BF16_BOUND` | Server start | TP=8 mrr=128 GSM8K |
|---:|---|---:|
| 0 | **rank-1 SIGABRT at init** | — |
| 128 | OK | **0.938** ✓ |
| 256 (default) | OK | 0.485 ✗ |

Long-term fix (upstream): the bf16-activation MoE path in
AITER FlyDSL needs to be debugged for gpt-oss MXFP4 + SwiGLU +
SEPARATED gate; until then SGLang could either (a) plumb the bound
down via an `envs.SGLANG_*` mirror with a saner default for
gpt-oss, or (b) auto-set the bound based on `max_running_requests`
when the gpt-oss MXFP4 path is detected.

#### Why this *looked* like a `--max-running-requests == 256` cliff during bisection

The bf16 / fp4 dispatch is *purely* a function of decode batch `M`.
When `--max-running-requests = 256`, steady-state decode batch
reaches 256 and the fp4 (correct) path takes over. Every smaller
value keeps decode batch ≤ N < 256, so the bf16 (buggy) path runs
every step. Hence the apparent "anything < 256 fails" boundary —
it is the *current* decode batch, not a CUDA-graph bucket, that
selects the kernel.

Original (now historical) sub-investigation text follows for
context.

#### Original (pre-root-cause) bisection

The reason `--max-running-requests` looked like the discriminator: at
higher concurrency the long-tail answers complete before the harness
prints (or were just slightly different — small batch-numerics noise
on top of truncation), masking the truncation effect. It is *not* a
real CUDA-graph or scheduler bug — same server, same prompts, just
the harness's hard token cap.

Same server (TP=1, SHUFFLE 5D, fp8 KV, `--max-running-requests 128`),
two harnesses:

| Harness | Config | Accuracy |
|---|---|---:|
| `lm_eval` (3-shot, chat template, `max_gen_toks=2048`) | unchanged | **0.8878** |
| `bench_sglang.py` (5-shot, `--max-new-tokens 512` default) | unchanged | **0.755** |
| `bench_sglang.py` (5-shot, `--max-new-tokens 2048`) | bumped | **0.854** |

And the harness-cap interaction, holding `bench_sglang.py` fixed:

| `--max-running-requests` | `--max-new-tokens` | Accuracy | Invalid |
|---:|---:|---:|---:|
| 128 | 512 (default) | 0.755 | 0.060 |
| 256 | 512 (default) | 0.833 | 0.013 |
| 128 | 2048 | 0.854 | 0.044 |
| **256** | **2048** | **0.941** | **0.008** |

Fix / workaround:
- Pass `--max-new-tokens 2048` (or higher) to
  `benchmark/gsm8k/bench_sglang.py` when running gpt-oss-120b.
- Consider raising the default in the harness itself for reasoning
  models, or making it model-dependent.

The earlier hypothesis (256-bucket CUDA-graph capture being the
discriminator) was ruled out by a control test: `--max-running-requests
256 --cuda-graph-max-bs 128` (captures only up to bs=128, same as the
"failing" config) still scores 0.801, while `--max-running-requests
128` alone scores 0.763 — i.e. the captured graphs are not the cause.

(Original investigation matrix preserved below for context.)

#### Original (pre-root-cause) bisection

`gpt-oss-120b` GSM8K accuracy drops below 0.80 (flexible-extract)
whenever the server is launched with **any** `--max-running-requests`
value below 256. Bisection on TP=1, SHUFFLE 5D, fp8 KV (the original
A5 repro):

| `--max-running-requests` | accuracy | status |
|---:|---:|---|
| 128 | 0.763 | ✗ |
| 192 | 0.757 | ✗ |
| 224 | 0.769 | ✗ |
| 240 | 0.781 | ✗ |
| 248 | 0.761 | ✗ |
| 252 | 0.784 | ✗ |
| 254 | 0.784 | ✗ |
| 255 | 0.779 | ✗ |
| **256** | **0.833 / 0.845** | ✓ (re-confirmed) |
| (flag omitted) | 0.827 | ✓ |

The cutoff is exact: every value <256 fails, ≥256 (or default) passes.
This lines up with SGLang's CUDA-graph `cuda_graph_bs` bucket list,
which has 256 as a standard bucket boundary
(`[..., 240, 248, 256, 272, ...]`). Most likely some scheduler /
metadata path is conditioned on having the 256-bucket captured (the
auto-tuned default ends at ≥256 in our setup), and truncating the list
to `< 256` leaves it without a bucket that the runtime later expects.

The failure is independent of:

* layout (NHD or SHUFFLE 5D)
* TP (TP=1 *and* TP=8 both fail with 128 and pass without it)
* KV dtype (bf16 / fp8 both affected)
* this branch (almost certainly an upstream scheduler / batch-grouping
  bug, not introduced by Plan A)

Reproducer:

```bash
# server (fails)
SGLANG_KV_CACHE_LAYOUT=vectorized_5d SGLANG_USE_AITER_MOE_GU_ITLV=False \
python3 -m sglang.launch_server \
  --model-path /path/to/gpt-oss-120b/ \
  --tp 1 --trust-remote-code \
  --mem-fraction-static 0.85 \
  --prefill-attention-backend aiter --decode-attention-backend aiter \
  --chunked-prefill-size 131072 \
  --max-running-requests 128 \
  --page-size 64 --disable-radix-cache --port 8000 \
  --kv-cache-dtype fp8_e4m3

# client (sglang's own gsm8k harness)
python3 benchmark/gsm8k/bench_sglang.py \
  --num-questions 1319 --parallel 1319 --num-shots 5 --port 8000
```

Workaround: drop `--max-running-requests` entirely (rely on the auto-
tuned default) or set it to **≥256**.

Next steps:
1. Compare the captured `cuda_graph_bs` list between
   `--max-running-requests 255` and `--max-running-requests 256` and
   diff which bucket(s) the 256-case captures that the 255-case
   doesn't, then check whether scheduler / model-runner code-paths
   reference any bucket >= last-captured-N.
2. Bisect SGLang main to find the introducing commit (this is upstream,
   not Plan A).

### A2. fp8 quantization in the SHUFFLE 5D writer (NOT A BUG — verified)

Initial concern was that `launch_reshape_and_cache_shuffle_5d` in
`utils.py` doesn't take `k_scale` / `v_scale` arguments and would
silently miscompute fp8 if the fused write path ever fell back to it.

Verified to be a non-issue: `MHATokenToKVPool.set_kv_buffer`
(`memory_pool.py:1141-1147`) already does the per-tensor fp8 quant on
the host before invoking the kernel:

```python
if cache_k.dtype != self.dtype:
    if k_scale is not None:
        cache_k.div_(k_scale)
    ...
    cache_k = cache_k.to(self.dtype)
if self.store_dtype != self.dtype:
    cache_k = cache_k.view(self.store_dtype)
```

By the time `launch_reshape_and_cache_shuffle_5d` is called the source
tensor is already fp8 bytes (viewed as uint8 — fp8 store_dtype maps to
uint8 because `Tensor.index_put` is not implemented for fp8). The
kernel just byte-copies into the SHUFFLE 5D cache slots — no per-tensor
scaling needed.

End-to-end verification: forced the standalone writer path via a debug
toggle on the fused-set-kv hook, with `--kv-cache-dtype fp8_e4m3 +
SGLANG_KV_CACHE_LAYOUT=vectorized_5d + TP=1`, and confirmed GSM8K
(3-shot, chat template, lm_eval) lands at **0.8848 flexible / 0.3480
strict** — same band as the fused-write baseline (~0.88 / 0.34).

No code change required.

### A2 (historical concern, kept for context). fp8 quantization is missing from the SHUFFLE 5D writer (P2)

`fused_qk_rope_reshape_and_cache` is enabled for the 5D pool and
handles fp8 quantization correctly (`apply_scale=True`, `k_scale`,
`v_scale`), so the on-the-wire fp8 path works. The standalone
`launch_reshape_and_cache_shuffle_5d` writer in `memory_pool.py` is
therefore dead code in normal operation, but it does **not** apply
the per-tensor fp8 scales itself. If a future change disables the
fused kernel (or routes a layer through the standalone writer for
any reason) fp8 KV will silently be stored without quantization and
GSM8K will collapse the same way the decode-side scale bug did before
the `pa_decode_gluon` `key_scale`/`value_scale` fix.

Action: either (a) add fp8 quant to `launch_reshape_and_cache_shuffle_5d`
matching `launch_reshape_and_cache_flash`, or (b) `assert` in the
5D writer that the input dtype already matches `store_dtype` so the
silent miscompute becomes a loud failure.

---

### A6. `sink_ptr` without `sink_size` selects a `_nsink` kernel — prefill aborts (FIXED locally in aiter, NOT committed, NOT upstreamed)

Affects every gpt-oss launch on the AITER backend, both the SHUFFLE 5D path
and the legacy NHD path. The server loads weights, captures CUDA graphs, then
dies on the first prefill (the startup warmup `/generate`):

```text
RuntimeError: invalid argument for batch_prefill: no matching kernel found.
              page_size=1, num_pages=6, dtype=bf16.
              If KV cache exceeds 2GB (INT32_MAX byte offset) with page_size < kN0,
              CDNA3+ GPU (MI300/MI350) is required.
```

**The error message is misleading.** It blames KV-cache size and GPU
generation; neither is the cause. `num_pages=6` is just the 6-token warmup
batch, and this reproduced on MI355X (gfx950), which is CDNA4.

Traceback tail:

```text
sglang/srt/layers/attention/aiter_backend.py:2454  forward_extend
sglang/srt/layers/attention/aiter_utils.py:97      forward_extend_vectorized_5d
aiter/aiter/ops/mha.py:3935                        mha_batch_prefill_func
```

#### Root cause — two bugs in `aiter/aiter/ops/mha.py`

gpt-oss uses a learned per-head **sink logit**, passed as `sink_ptr`. It does
NOT use `sink_size` (aiter's StreamingLLM-style "first N KV tokens always
attended"), so `sink_size` stays at its default `0`. That exposes:

1. **Parameter-order mismatch (the dominant bug).** `_mha_batch_prefill` calls
   the op with all-positional args in the order

   ```text
   ..., kv_block_descale, kv_last_page_lens, block_table, seqlen_k, sink_ptr, gen
   ```

   but `cmdGenFunc_mha_batch_prefill` — the function that picks which JIT
   module to build/load — declares

   ```text
   ..., kv_block_descale, sink_ptr, gen, kv_last_page_lens, block_table, seqlen_k
   ```

   so inside the generator `sink_ptr` actually receives `kv_last_page_lens`.

2. **Inconsistent sink predicate.** The generator computed
   `has_effective_sink` from `sink_size > 0` only, while the C++ side
   (`csrc/cpp_itfs/mha_fwd_batch_prefill.cu:52`) uses

   ```cpp
   bool has_sink = args.sink_size > 0 || args.sink_ptr != nullptr;
   ```

Net effect: Python names and loads the `..._nsink.so` module, C++ then asks for
a `has_sink=true` kernel arm that module does not contain,
`fmha_batch_prefill` returns `t < 0`, and the `TORCH_CHECK` above fires.

Fixing only (2) is **not** sufficient — verified; the generator still reads the
wrong slot and keeps selecting `_nsink`. Both changes are required.

#### Affected call sites (all pass `sink_ptr` and never `sink_size`)

```text
python/sglang/srt/layers/attention/aiter_utils.py:112    5D fresh-prompt path
python/sglang/srt/layers/attention/aiter_utils.py:199    5D gather-and-linearize path
python/sglang/srt/layers/attention/aiter_backend.py:2502 legacy NHD path
```

#### Why this cannot be worked around from the SGLang caller

SGLang already passes `sink_ptr` **by keyword** and is not doing anything
wrong. The corruption happens two layers deeper, inside aiter: `_mha_batch_prefill`
re-invokes the decorated op `mha_batch_prefill` with **all-positional** args, and
`compile_ops` forwards that same arg tuple to `cmdGenFunc_mha_batch_prefill`.
Nothing SGLang passes can change that binding.

Tail misalignment (positions 25-29 of the generator's signature):

```text
generator expects     actually receives
sink_ptr           <- kv_last_page_lens
gen                <- block_table
kv_last_page_lens  <- seqlen_k
block_table        <- sink_ptr
seqlen_k           <- None
```

Three caller-side workarounds exist. None is acceptable:

**A. Pass `sink_size > 0` from SGLang.** Mechanically works — `sink_size` sits
at position 15 and *is* bound correctly, so `has_effective_sink` becomes true
and the `_sink` module is selected. But `sink_size` in aiter means
StreamingLLM-style "the first N KV tokens are always attended", and it really
does change the mask:

```python
k_start_window = torch.clamp(abs_q - window_left, min=sink_size)
is_sink = i_k < sink_size
```

gpt-oss wants a learned per-head sink **logit**, not a preserved KV prefix, and
gpt-oss runs with a sliding window (128) so this masking path is live. The
result is silently **wrong attention output** instead of a crash — strictly
worse than the current failure.

**B. Monkeypatch `cmdGenFunc_mha_batch_prefill`.** Ineffective after import:
`compile_ops` calls `gen_func(*args, **kwargs)` through a closure variable
captured at decoration time (`aiter/jit/core.py:1618`), not via a module
attribute lookup. Patching would have to win a race before aiter is imported.

**C. Call the low-level op `mha_batch_prefill` directly with keyword args.**
Keyword binding sidesteps the misordering, but it also skips everything
`mha_batch_prefill_func` does first: contiguity normalization, 5D/4D/3D layout
validation, head-size divisibility checks, `sink_ptr` dtype/device coercion,
and the final `out[..., :head_size_v_og]` slice. That copies aiter's internal
contract into SGLang and breaks whenever aiter changes it.

Conclusion: this must be fixed in aiter. Any model using a learned sink logit
(`sink_ptr` without `sink_size`) hits it.

#### Fix

Local, uncommitted, in `aiter/aiter/ops/mha.py`:

```text
1. Reorder cmdGenFunc_mha_batch_prefill's trailing parameters to match the op:
     kv_block_descale, kv_last_page_lens, block_table, seqlen_k, sink_ptr, gen
2. has_effective_sink = (sink_size > 0 or sink_ptr is not None) and (
        causal or not (window_size_left == -1 and window_size_right == -1))
```

After the fix aiter JIT-builds (~10 min, one time)

```text
aiter/jit/mha_batch_prefill_bf16_nlogits_nbias_mask_nlse_ndropout_nqscale_sink.so
```

and the server reaches `The server is fired up and ready to roll!`.

This is an upstream aiter bug and has NOT been submitted. Note (2) is a
judgement call — upstream may instead intend callers to pass `sink_size`; (1)
is unambiguously a bug either way.

#### Minimal repro (no model required, ~20 s)

```python
import torch, aiter
dev="cuda"; B,S,HQ,HKV,D = 2,3,64,8,64; T=B*S
q=torch.randn(T,HQ,D,dtype=torch.bfloat16,device=dev)
k=torch.randn(T,HKV,D,dtype=torch.bfloat16,device=dev)
v=torch.randn(T,HKV,D,dtype=torch.bfloat16,device=dev)
ind=torch.tensor([0,S,2*S],dtype=torch.int32,device=dev)
kvi=torch.arange(T,dtype=torch.int32,device=dev)
sinks=torch.randn(HQ,dtype=torch.float32,device=dev)
def run(**kw):
    return aiter.mha_batch_prefill_func(q,k,v,ind,ind,kvi,S,S,causal=True,
        window_size=(128,0),return_lse=False,return_attn_probs=False,**kw)
run()                  # OK   -> loads ..._nsink.so
run(sink_ptr=sinks)    # FAIL -> same "no matching kernel found" error
```

#### Verification after fix

```text
short prompt          "Paris is the capital of France."   correct
8,093-token needle    retrieved "ZQ-7741"                 correct
```

Numerically correct, not merely non-crashing — so the `_sink` kernel arm is
producing right answers on both the fresh-prompt and SWA paths.


## Stability

### B1. Sporadic TTFT spike at ISL=8192, OSL=1024, concurrency=64 (P2)

A single sweep run produced median TTFT 6755 ms (output throughput
3469 tok/s) at this exact `(8K, 1K, c=64)` point. Two clean re-runs
on the same server gave 510-513 ms TTFT and 4574-4592 tok/s, in line
with c=32 and c=128 neighbours. The original outlier appears to be
a scheduler / chunked-prefill corner case at this batch size that is
not reproducible after a fresh server start. No fix needed yet, but
worth profiling once if it ever resurfaces in CI.

### B2. GPU cleanup between server restarts requires manual PID kills (P3)

`pkill -f "sglang.launch_server.*"` and similar patterns can match the
parent shell's own command line on this container and kill the wrong
process. After a crashed or aborted launch, leftover sglang scheduler
workers continue to hold VRAM and the next launch OOMs even though
`rocm-smi` initially reports the GPU as free (the held memory only
shows up under `rocm-smi --showpids`). Operational workaround: kill
workers explicitly by PID from `rocm-smi --showpids` after each
restart. This is not a code bug, just a doc-it item.

---

## Performance (TP=1)

### C1. Decode TPOT scaling is worse than ATOM as batch grows (P0)

At ISL=1024 / OSL=1024 SGLang TPOT degrades monotonically vs ATOM
as concurrency rises:

| conc | SGL TPOT (ms) | ATOM TPOT (ms) | Δ      |
|-----:|--------------:|---------------:|-------:|
|    4 |          3.96 |           3.96 |   0.0% |
|   16 |          5.37 |           4.81 | +11.8% |
|   32 |          7.05 |           5.76 | +22.3% |
|   64 |          9.48 |           7.42 | +27.8% |
|  128 |         14.24 |          11.36 | +25.4% |

Total token-throughput follows the same trend; SGLang is -16.3 %
behind ATOM at c=128. ISL=8192 / OSL=1024 is unaffected at low
concurrency (SGLang is +18-26 % at c≤8) and roughly tied at c≥16.

Likely contributors identified in earlier profiling:
- per-layer `seq_lens.to(int32)` and the surrounding metadata
  rebuild fire 36× per decode step;
- decode metadata tensors are reallocated on every forward pass
  instead of being captured into the CUDA graph the way ATOM does.

Both require some graph-safe metadata caching in
`init_forward_metadata` before they can be removed.

### C2. SGLang wins at long-context low-concurrency (informational)

For ISL=8192 / OSL=1024 with c ≤ 8 SGLang outperforms ATOM by
+18-26 % throughput, -12-20 % TPOT, and -62 % TTFT. No action.

### C3. Prefill TTFT is consistently ~10 ms slower than ATOM at small
batch (P3)

At ISL=1024 with c ≤ 32 SGLang's median TTFT is ~50 ms versus ATOM's
~40 ms (+17-24 %). The absolute gap is small and only matters if
P50 TTFT becomes a hard SLA target.

---

## Sweep data

All raw JSONs that back the tables above live under
`/workspace/bench_results_v2/` on the host the original sweep was run
on, named `sglang_5d_fp8_isl{ISL}_osl{OSL}_c{CONC}.json` and
`atom_fp8_isl{ISL}_osl{OSL}_c{CONC}.json`. The summarizer is at
`/workspace/summarize_table.py`.
