---
name: aiter-pa-decode-gluon-design
description: Explains why AITER's `pa_decode_gluon` paged-attention decode kernel beats SGLang's legacy `unified_attention` by ~46% on gpt-oss-shape decode, and the design levers that produce the win. Use when investigating decode-attention performance on ROCm, when porting another model to the SHUFFLE 5D KV layout, or when reviewing/optimizing other AITER gluon kernels.
---

# AITER `pa_decode_gluon` — why it's fast

`pa_decode_gluon` is the paged-attention decode kernel SGLang's AITER
backend dispatches to when the KV pool is allocated in the SHUFFLE 5D
layout (`SGLANG_KV_CACHE_LAYOUT=vectorized_5d`). On gpt-oss-120b @
MI300 it is **~46 % faster than `unified_attention`** at the same
shapes, and end-to-end gives ~14 % TPOT and ~15 % output throughput.

This skill explains the four design levers that buy that speedup. The
goal is that when you next look at an AITER decode kernel — or want
to apply the same techniques elsewhere — you know what to look for.

Source: `/sgl-workspace/aiter/aiter/ops/triton/gluon/pa_decode_gluon.py`
(entry point `paged_attention_decode_v2_gluon_large_block_dot_kernel`,
~5.6k lines of Gluon).

## Lever 1 — SHUFFLE 5D KV layout matches MFMA tile shape

The pool is allocated as:

```
K cache: (num_blocks, H_kv, head_dim / X, page_size, X)
V cache: (num_blocks, H_kv, page_size / X, head_dim, X)
where X = 16 / store_dtype.itemsize  (8 for bf16, 16 for fp8)
```

The trailing `X` dimension is exactly **16 bytes wide** for any
supported dtype. The MI300 / MI350 LDS coalesces global→LDS loads on
16-byte boundaries and the V_MFMA_F32_16x16x16/32 instructions
consume operands as 16-element vectors along the K-axis. So:

- K reads: a single `buffer_load` along the inner `X` lane delivers
  one MFMA operand row directly — no LDS shuffle, no in-kernel
  transpose.
- V reads: the K-axis is the *page offset* (page_size/X), and the
  inner `X` is the contraction lane. The same coalesced load feeds
  the second MFMA without any layout fix-up.
- bf16 vs fp8 share the same kernel and the same memory traffic — only
  the MFMA `K` widens from 16 to 32 (see Lever 4).

By contrast, `unified_attention` consumes the NHD (`(T, H_kv, D)`)
layout. The kernel has to permute / strided-load to assemble each
16-element MFMA lane, costing extra LDS round-trips and limiting
issue rate.

**Implication for new kernels**: if your kernel uses MFMA on a paged
KV cache, allocate the pool in SHUFFLE 5D from day one. The SGLang
side that enables it is gated by `SGLANG_KV_CACHE_LAYOUT` and lives
in `python/sglang/srt/mem_cache/memory_pool.py` (search for
`"vectorized_5d"`).

## Lever 2 — Gluon DSL gives direct MFMA + buffer-load control

`pa_decode_gluon` is written in **Gluon**, AITER's superset of Triton
that exposes AMD-specific intrinsics as first-class:

```python
import aiter.gluon as gl

qk_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
    instr_shape=[16, 16, MFMA_INSTR_K],   # 16 (bf16) or 32 (fp8)
    ...,
)

gl.amd.cdna3.buffer_load(...)   # explicit CDNA3 vmem load
gl.amd.cdna3.buffer_store(...)  # explicit CDNA3 vmem store
```

Two things this buys over plain Triton:

1. **Hand-picked MFMA tile**. Triton's auto-scheduler often falls back
   to MFMA `16x16x16` or `32x32x8` regardless of dtype. Gluon's
   `instr_shape=[16, 16, MFMA_INSTR_K]` with `MFMA_INSTR_K`
   computed from the cache dtype matches MI300's peak-throughput
   `V_MFMA_F32_16x16x16_BF16` (bf16) or `V_MFMA_F32_16x16x32_FP8`
   (fp8) directly.
2. **Buffer-load/store with explicit bounds masks**. `gl.amd.cdna3.buffer_*`
   uses VMEM `buffer_*` (not the slower `global_*`) which gives
   per-lane out-of-bounds masking in hardware — no need for branchy
   software masks. For variable-length context this is a big deal: a
   single instruction handles the irregular last partition.

**Implication for new kernels**: when a Triton kernel is bottlenecked
on memory bandwidth or on the MFMA pipeline, rewriting the inner loop
in Gluon and pinning `AMDMFMALayout` typically claws back 1.5–2×.
Reach for `gl.amd.cdna3.buffer_load/store` whenever you need
hardware-masked loads.

## Lever 3 — Split-K across context, tuned to SM count

Decode is "skinny" — one query per request — so the K dimension
(context length) dominates the workitem. `pa_decode_gluon` splits the
context into partitions of size `context_partition_size` (256 in our
config) and launches one CTA per `(sequence, kv_head, partition)`:

```python
def get_recommended_splits(num_sequences, num_kv_heads, split_kv_blocks=1):
    num_sm = props.multi_processor_count * get_occupancy()
    max_context_partition_num = triton.cdiv(
        num_sm, num_sequences * num_kv_heads * split_kv_blocks
    )
    return min(max_context_partition_num, 8)
```

So at `bs=1, num_kv_heads=8` on a 304-CU MI300X, the heuristic asks
for ~38 partitions and clamps to 8 — exactly enough to saturate the
machine without producing more partial sums than the reduction kernel
can chew through. At `bs ≥ 38` it falls to 1 and we run one CTA per
(seq, kv_head) with no split.

The partial results land in pre-allocated `exp_sums`, `max_logits`,
and `temporary_output` buffers, and a separate small reduction
kernel (`paged_attention_v2_reduce_*`) combines them into the final
output. This is the classic "FlashDecoding"/"Split-K paged attention"
pattern, but with the split count chosen per-call from the *real*
shape rather than a fixed heuristic.

`unified_attention` is FlashAttention-style — one CTA per (seq,
kv_head) — and gives up the split-K win whenever `bs` is small (≤8),
which is exactly the regime where Plan A wins biggest (we measured
+18-26 % throughput vs ATOM at c=4-8).

**Implication for new kernels**: any decode kernel for a model with
modest `bs × num_kv_heads` should expose a split-K knob and use a
recommended-splits function similar to the snippet above. Hardcoding
the split count loses throughput as the deployment scales up or down.

## Lever 4 — Compile-time constexpr branching keeps one source file fast for every dtype

The kernel body is a single Gluon function but dispatches on
`gl.constexpr` flags for:

- KV dtype (`bf16` vs `fp8` vs `int8`) — picks
  `MFMA_INSTR_K` (16 vs 32) and the right `define_layout` overload
- query length (`q_len == 1` decode vs spec-decode draft batches)
- sliding window (`SLIDING_WINDOW > 0`)
- KV / query quant mode (per-tensor vs per-token descales)
- sinks (gpt-oss-style logit sink token) — `sinks=None` path is
  cheaper

Because these are `gl.constexpr`, the compiler **specialises one
kernel per shape combo** and dead-code-eliminates every branch we
don't take. There is no runtime `if` in the inner loop. The
`@gluon.jit` cache keeps the compiled kernels around between calls,
so the per-call cost after first warm-up is just a kernel launch.

`unified_attention` is the opposite: a generic Triton kernel that
handles many of those cases at *runtime* via boolean masks and
conditional reads, which costs both registers and issue slots.

**Implication for new kernels**: when you write a kernel that has to
serve several dtypes / mask modes / window sizes, prefer `gl.constexpr`
flags over runtime branches even if it looks like more code. The
binary cost is one specialised kernel per used combo, which the JIT
cache makes essentially free.

## Putting it together

For the gpt-oss-120b @ MI300 decode case (`bs=64..256, num_kv_heads=8,
head_dim=128`, bf16 or fp8 KV):

| Mechanism | Where it lives in `pa_decode_gluon.py` | What it costs `unified_attention` |
|-----------|---|---|
| 1. SHUFFLE 5D layout | layout descriptors at top + `define_layout` | extra LDS shuffles per inner-loop tile |
| 2. Gluon + AMD MFMA | `qk_mfma_layout` / `pv_mfma_layout` constexprs + `gl.amd.cdna3.buffer_*` | suboptimal MFMA tile + slower `global_load` |
| 3. Context split-K | 3D launch grid `(seq, kv_head, output_partition_idx)` + `get_recommended_splits` | empty SMs at small batch |
| 4. constexpr dispatch | `MFMA_INSTR_K`, `SLIDING_WINDOW`, `KV_QUANT_MODE`, `SINK*` `gl.constexpr` | runtime branches in the inner loop |

Stack them and you get the ~46 % per-kernel speedup we observe, and
the ~14 % end-to-end TPOT win once you compose with the SHUFFLE 5D
writer + the fused RoPE+set_kv path (see `aiter_utils.py` and
`PR #27063` for the SGLang plumbing).

## When to apply this skill

- **Optimizing a non-gpt-oss model on the AITER backend**: check
  whether the model's decode kernel already uses SHUFFLE 5D + a Gluon
  kernel. If not, replicating levers 1+2 typically gets you ≥30 %.
- **Reviewing a new AITER decode kernel**: use the table above as a
  checklist — anything missing is a perf bug.
- **Investigating a perf regression on the decode path**: confirm the
  kernel is still being dispatched (check
  `AiterAttnBackend.forward_decode` in
  `python/sglang/srt/layers/attention/aiter_backend.py`) and that
  `get_recommended_splits` is still returning >1 at small batch.
- **Porting to a new GPU arch**: lever 2 is CDNA3-specific
  (`gl.amd.cdna3.buffer_*`, `AMDMFMALayout`). CDNA4 / RDNA4 need
  different intrinsics; lever 1 (layout) and lever 4 (constexpr
  dispatch) port unchanged.

## Anti-patterns / things to avoid

- **Don't try to make `unified_attention` faster by changing its
  layout.** It is a Triton kernel that assumes NHD; the perf ceiling
  is bounded by the layout mismatch (lever 1), not by Triton itself.
- **Don't pad `M` to feed a different MoE kernel choice** without
  also confirming the decode kernel still picks the
  `pa_decode_gluon` path. The two dispatches are independent (the
  MoE bound is `GPTOSS_SWIGLU_MXFP4_BF16_BOUND`, the attention
  dispatch is `SGLANG_KV_CACHE_LAYOUT`).
- **Don't expose `context_partition_size` as a user knob.** It is
  intentionally fixed at 256 to match MI300 LDS sizing. Changing it
  invalidates the layout descriptors.

## References

- AITER kernel:
  `/sgl-workspace/aiter/aiter/ops/triton/gluon/pa_decode_gluon.py`
- SGLang dispatch (decode):
  `python/sglang/srt/layers/attention/aiter_utils.py`
  `forward_decode_vectorized_5d`
- SGLang dispatch (prefill gather):
  `python/sglang/srt/layers/attention/aiter_utils.py`
  `forward_extend_vectorized_5d`
- Pool layout: `python/sglang/srt/mem_cache/memory_pool.py`
  (`MHATokenToKVPool`, `"vectorized_5d"` branch)
- SHUFFLE writer / reader kernels:
  `python/sglang/srt/layers/attention/utils.py`
  (`reshape_and_cache_shuffle_5d`, `gather_shuffle_5d_to_linear`)
- E2E numbers + bring-up history: PR #27063 description, plus
  `KNOWN_ISSUES.md` A1/A4/A5 entries (local doc, not in repo).
