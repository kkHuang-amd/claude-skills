# Enabling #34624's compress fusion under the FP4 indexer — trace + plan

## CONTINUE HERE — next action

**Do this:** awaiting direction. Correctness work is finished on both variants; the open items
are (a) a perf measurement through the aiperf harness, which is the only thing that can support
a throughput claim, and (b) optionally an extend variant of the **head_dim 512 flashmla**
epilogue, which is now the one compressor still falling back under MTP.

**State:** both variants **done and verified end to end.**

  - decode epilogue: 99.74 % identical codes; GSM8K 0.951, fusion proven ACTIVE, MTP off.
  - extend (TARGET_VERIFY / DRAFT_EXTEND) epilogue: 99.53 % identical codes and a bit-exact
    ring buffer; **GSM8K 0.947 with `ACTIVE, fp4 extend epilogue` x8 on a real MTP arm** — the
    first arm on which this fusion has ever bound. Sections below.

**Nothing is benchmarked.** GSM8K latency is unusable as a perf signal here (~2.2x replicate
spread on identical config — see the MTP-ON section).

**Where the code is:** worktree `/shared_nfs/kk/tmp/combined` (branch `combined-dsv4-decode`,
base `7233d46bdf`), changes **uncommitted**. The worktree's git metadata lives in
`/sgl-workspace/sglang/.git`, which does **not** survive an image swap, so the work is also
exported as patches under `/shared_nfs/kk/docs/` (base SHA in the matching `.base`):

  - `fp4-compress-fusion-A.patch` — decode only, the state that scored GSM8K 0.951.
  - `fp4-compress-fusion-B.patch` — **current**, cumulative: decode + extend.

Re-apply with `git apply` on that base if the worktree is gone. `/sgl-workspace/sglang` itself
is untouched at `efaeb6f664`.

**Repro the kernel checks** (decode first — it is the regression test for the shared epilogue):
```bash
cd /shared_nfs/kk/tmp/combined && PYTHONPATH=$PWD/python python3 /shared_nfs/kk/docs/diff_fp4_fused.py \
  && PYTHONPATH=$PWD/python python3 /shared_nfs/kk/docs/diff_fp4_fused_extend.py
```
**Repro a GSM8K arm** (label, gate, ref). The no-MTP ref `/shared_nfs/kk/tmp/ref_nomtp` is only
needed for the decode variant; for the extend variant use a normal MTP ref:
```bash
bash /shared_nfs/kk/docs/run_gsm8k_fp4fuse.sh mylabel 1 /shared_nfs/kk/tmp/ref_nomtp
```
**Pass criteria:** `[compress-fusion] ACTIVE, fp4 extend epilogue` must appear in
`/tmp/agentx_debug_server.log`, and GSM8K must land in **0.946–0.955** (the measured replicate
band). A silent layout error scores 0, not 0.95.

**Node etiquette:** shared node, gates refuse rather than kill — keep it that way. Post-arm
VRAM residue is ~120 GB and takes **~16 min** to drain here (the older note says 13); wait,
never kill blindly.

## Why the fusion could not fire on MTP arms — READ THIS FIRST

**Addressed 2026-09-04 by the extend variant** (see "Extend variant result" below); the gate
now admits extend plans for the FP4 indexer epilogue. Keep reading anyway — this is why the
`SGLANG_OPT_FUSE_COMPRESS_NORM_ROPE` flag alone still cannot be trusted as evidence, and it is
why the fp8 and flashmla epilogues remain unreachable under MTP.

**The fp4 epilogue works, and it still cannot fire on any of our arms.** The compress
fusion is gated on `plan.is_decode`, and **with MTP on there is no decode plan**: the
steady-state forward modes are `TARGET_VERIFY` and `DRAFT_EXTEND`
(`deepseek_v4_backend_hip_radix.py:422-428`), so the compressor builds a *prefill/extend*
plan and `CompressorPrefillPlan.is_decode` is `False`. Measured, not inferred — see
"Instrumented run" below. Every arm on the board runs `--speculative-algorithm EAGLE
--speculative-num-steps 3`.

So relaxing the FP4 guard was **necessary but not sufficient**. To get this on an MTP arm
the fused kernel needs a prefill/extend-shaped variant; the decode-only kernel is
unreachable. Note #34624's own `+2.95 %` was measured with **no** `--speculative-algorithm`
in the server command, i.e. on real decode steps where the plan does exist.

**Historically: do not spend an arm on `SGLANG_OPT_FUSE_COMPRESS_NORM_ROPE` while MTP is on.**
It was a no-op, and without the marker below you would score the baseline and call it a win.
That is fixed for the FP4 indexer path only. The rule that survives: **score nothing without
the marker line.** The gate now prints one of three things per TP worker —
`ACTIVE, fp4 epilogue` (decode), `ACTIVE, fp4 extend epilogue`, or a `SKIPPED` / `no epilogue:`
line naming the condition that failed.

## Handoff notes (superseded by CONTINUE HERE above)

**Status:** **IMPLEMENTED, JIT-COMPILES, AND THE DIFFERENTIAL CHECK PASSES.** Not yet run
end-to-end in a server; no GSM8K yet; nothing benchmarked.
**Where:** worktree `/shared_nfs/kk/tmp/combined` (branch `combined-dsv4-decode`), changes
uncommitted. `/sgl-workspace/sglang` untouched at `efaeb6f664`.
**Next:** GSM8K against this worktree with `SGLANG_OPT_FUSE_COMPRESS_NORM_ROPE=1`. The open
question is plumbing, not correctness: `gsm8k_arm.sh` → `agentx_debug.sh serve` launches
from `/sgl-workspace/sglang`, so it needs `PYTHONPATH` pointed at the worktree (or the
changes applied to the benchmark tree).
**GSM8K trap:** `gsm8k_arm.sh` already sets `NO_ACC_PIN=1`, which makes `agentx_debug.sh:66`
`sed -i '/SGLANG_SIMULATE_ACC/d'` the replayed env. Without it every arm scores ~0.72
instead of ~0.93 — this is what invalidated all four gsm8k arms of 2026-08-29.

### Differential check result (2026-09-04)

`/shared_nfs/kk/tmp/diff_fp4_fused.py`, 12 decode steps, 3 compress boundaries, page 64:

```
identical codes: 99.74%      max abs diff: 0.500000     mean abs diff: 0.001302
p99 rel diff   : 0.0000      beyond one code step: 0.260%      RESULT: PASS
```

`max abs diff 0.5` is exactly one e2m1 grid step at the low end (the grid is
{0, .5, 1, 1.5, 2, 3, 4, 6}), and the reference wrote real data (`|sum| = 330.25`, so the
harness is not comparing two empty caches). This is the expected difference class: the fused
kernel keeps the compressed row in fp32 registers where the chain rounds it to bf16 between
launches. The same tolerance shape the fp8 fusion documents.

**Harness gotcha, already fixed:** aiter asserts `cos/sin dtype must match input dtype`
(`dsv4_rotate_quant.cu:1415`). The reference chain must allocate the compressed row as
**bf16** via `compress_forward(out=...)`, the way `compressor_v2.py:209` does in production;
passing the default fp32 row aborts the process.

### Extend variant result (2026-09-04)

`/shared_nfs/kk/docs/diff_fp4_fused_extend.py`, one prefill-shaped step `(seq 8, extend 8)`
followed by three verify-shaped steps of 4 tokens, page 64, 5 compress boundaries:

```
slots=[3, 7, 11, 15, 19]
ring-buffer max abs diff: 0.000000      identical codes: 99.53%
max abs diff: 0.500000                  mean abs diff: 0.001953
p99 rel diff: 0.0000                    beyond one code step: 0.313%     RESULT: PASS
```

Same difference class as decode: `max abs diff 0.5` is one e2m1 grid step at the low end, and
`|sum| = 490.5` on the reference side says the harness is not comparing two empty caches. The
schedule deliberately mixes shapes — the first step puts several compress plans in one grid,
the later steps exercise the ring-buffer overlap *across* launches, which is the only thing
the copied staging kernel can get wrong.

**The ring buffer is checked separately and must be bit-exact** (`pool_max == 0.0` is part of
the pass criterion). It is a plain memcpy in both paths, and it is an *input* to the next
step's compress, so a layout error there would otherwise surface only as a value drift several
steps later.

**Re-ran the decode check after the refactor:** 99.74 % / 0.500000 / 0.001302 / 0.260 %, i.e.
identical to the pre-refactor numbers, so factoring the epilogue out is value-preserving.

**JIT rebuild confirmed, not assumed.** `run_extend_fp4` is a new wrapper in the same
`load_jit` call, so a stale `.so` would fail to bind the symbol rather than silently run old
code — and
`/root/.cache/sglang/jit/gfx950/sgl_kernel_jit_dpsk_v4_fused_compress4_norm_rope_128_fp32_t_fp32_t_fp32_t_64_false_16`
was rewritten during the run. This is the positive form of the evidence the 2026-08-29 lesson
asked for.

### End-to-end confirmation with MTP **ON** — the extend variant (2026-09-04)

Arm `fp4ext1`, ref `/workspace/results/hicache-fp4-int20-c128-fuse-mf090` (EAGLE, num-steps 3,
topk 1, num-draft-tokens 4), `SGLANG_OPT_FUSE_COMPRESS_NORM_ROPE=1`:

```
[compress-fusion] ACTIVE, fp4 extend epilogue                                x8
[compress-fusion] gate set but SKIPPED: hip=True ratio=128 decode=False ...  x8   (no fused c128 kernel — by design)
[compress-fusion] gate set but SKIPPED: no epilogue: decode=False fp4=False head_dim=512   x8
```

`GSM8K 0.947, Invalid 0.000` — inside the 0.946–0.955 replicate band, so the extend epilogue is
correct in a real server and not only in the kernel test. **This is the first arm on which the
fusion has ever bound**; every earlier MTP arm scored the unfused baseline.

The third line is the new skip log earning its keep: the head_dim 512 flashmla c4 compressor
has **no extend variant**, so under MTP it still falls back. With MTP off that same compressor
logged `ACTIVE, flashmla epilogue`. That is the remaining gap, and it is now visible in the log
instead of silent.

**Do not read GSM8K latency as a perf signal on this node.** Measured on the *identical* MTP-on
config with the fusion proven **inactive**: 50.8 / 51.7 / 110.1 s. The extend arm's 96.2 s sits
inside that ~2.2x spread, so it says nothing either way. Any perf claim needs the aiperf
harness, not this.

### End-to-end confirmation with MTP OFF (2026-09-04)

Ref `/shared_nfs/kk/tmp/ref_nomtp` is the c128 fusion arm with the four `--speculative-*`
flags stripped, so decode is a real decode plan. `SGLANG_OPT_FUSE_COMPRESS_NORM_ROPE=1`:

```
[compress-fusion] ACTIVE, fp4 epilogue        x8   (one per TP worker)
[compress-fusion] ACTIVE, flashmla epilogue   x8   (the head_dim 512 c4 compressor)
[compress-fusion] gate set but SKIPPED: ratio=128 decode=True   x8   (no fused c128 kernel — by design)
```

`GSM8K 0.951, Invalid 0.000` (latency 85.0 s vs ~51 s with MTP, as expected without
speculative decode). That is inside the 0.946–0.955 band measured with the fusion inactive,
and a wrong FP4 layout does not score 0.951 — #37423's precedent for a bad layout is
**GSM8K 0**. So the fp4 epilogue is correct in a real server, not only in the kernel test.

**Not run:** a no-MTP, gate-off baseline. The comparison above is against the MTP-on band, so
it bounds "not broken" rather than "identical". Worth one run if an accuracy claim is ever
made from this.

### Instrumented run — how this was established

Three GSM8K runs, all from the worktree via `PYTHONPATH`, all with
`SGLANG_OPT_FUSE_COMPRESS_NORM_ROPE=1`:

| label | accuracy | invalid | fusion active? |
|---|---|---|---|
| fp4fuse | 0.948 | 0.000 | unknown (no instrumentation yet) |
| fp4fuse2 | 0.955 | 0.000 | unknown (`logger.info` never reached the log) |
| fp4fuse3 | 0.946 | 0.000 | **NO — proven by marker** |

`fp4fuse3` printed, 8 times each (once per TP worker):

```
[compress-fusion] gate set but SKIPPED: hip=True ratio=4   decode=False online=False
[compress-fusion] gate set but SKIPPED: hip=True ratio=128 decode=False online=False
```

and never printed `ACTIVE`. `decode=False` on **every** call is the finding.

**Two lessons about verification, both of which cost a run:**

1. `logging.getLogger(__name__).info(...)` from `srt/layers/attention/dsv4/compressor_v2.py`
   **does not reach** `/tmp/agentx_debug_server.log`, even though other worker-side INFO
   lines (e.g. `Init Unified Radix Cache`) do. Use `print(..., file=sys.stderr, flush=True)`
   for anything an arm must be scored against.
2. A JIT-cache directory is not evidence: production used the same template arguments as the
   standalone test (`..._128_fp32_t_fp32_t_fp32_t_64_false_16`), so it was a cache hit and
   left no new artifact. Absence of a new build proves nothing either way.

**Free by-product — GSM8K replicate spread on this node:** 0.946 / 0.948 / 0.955 on identical
config, i.e. a spread of **0.9 points**. The board previously had no GSM8K replicate band, so
score future accuracy deltas against this, not against zero. (All three are the healthy
~0.95 band; the `SGLANG_SIMULATE_ACC_LEN` trap would have shown ~0.72.)

### What was changed

| File | Change |
|---|---|
| `kernels/jit/csrc/deepseek_v4/fused_compress4_norm_rope_hip.cuh` | `kvcache_scale` added to the params struct; local `quant_fp4_e2m1`; `wave_reduce_max` given a lane-width template (default unchanged); `kFp4Store` template flag on `flash_c4_decode_norm_rope_indexer_w64` with the new part 4a; `kernel_fp4` + `run_decode_fp4` on the dispatch struct, `#ifdef USE_ROCM` |
| `kernels/ops/attention/dsv4/compress.py` | HIP registers a second wrapper `decode_fp4`; new `compress_forward_norm_rope_store_fp4` |
| `kernels/ops/attention/dsv4/__init__.py` | export it |
| `srt/layers/attention/dsv4/compressor_v2.py` | guard split into a shared `fuse_gate`; FP4 + head_dim 128 routes to the new op instead of being excluded |

### What the extend variant added (2026-09-04)

| File | Change |
|---|---|
| `fused_compress4_norm_rope_hip.cuh` | `PlanC`/`PlanW` aliases; `FusedCompress4NormRopeExtendParams`; the w64 decode kernel's parts 1–4 factored into `c4_norm_rope_indexer_epilogue_w64` (templated on the params type) and called from both entry points; new `flash_c4_extend_norm_rope_indexer_w64`; `write_c4_extend` (a copy of `write_c4_prefill`) + `kFusedWriteBlockSize`; `kernel_extend_fp4` / `kernel_extend_write` / `WriteTrait` and `run_extend_fp4` on the dispatch struct |
| `kernels/ops/attention/dsv4/compress.py` | `extend_fp4` wrapper registered; `compress_forward_norm_rope_store_fp4` now takes either plan shape and dispatches on `plan.is_decode` |
| `srt/layers/attention/dsv4/compressor_v2.py` | `fuse_gate` no longer requires `is_decode`; `fuse_gate_decode_only` keeps the fp8/flashmla epilogues decode-only; marker says `fp4` vs `fp4 extend`; a third skip log fires when the gate passes but no epilogue matches |
| `/shared_nfs/kk/docs/diff_fp4_fused_extend.py` | new differential test (see result above) |

Two decisions worth not re-litigating:

- **The epilogue is shared, not copied.** Decode and extend differ only in how a wavefront
  finds its work; everything from the compressed row onwards is identical. The "deliberate
  copy" rule applies to `c4_v2.cuh`, the common all-platform file — not to two kernels inside
  the same HIP-only file, where a copy would just be two things to keep in step. The decode
  differential test re-verifies the refactor and did so bit-for-bit.
- **`out_loc[plan.ragged_id]` is bounds-checked.** `c4_out_loc` is allocated over *write*
  tokens, which CP-v2 padding can make fewer than the metadata rows `ragged_id` counts
  (`metadata_kernel.py:112`). aiter's host-side metadata guards the same way
  (`fp4_indexer_hip.py`, `ragged_ids < out_loc.shape[0]`), so the kernel carries a
  `num_out_loc` field and early-outs — wave-uniform, one compare. The JIT non-aiter reference
  does the unchecked load; do not copy that.

## Plan for the extend (TARGET_VERIFY) variant — what MTP arms need

**IMPLEMENTED 2026-09-04** — kept below because it is the design record and the indexing is
still the thing to check against if the extend kernel ever misbehaves.

Traced 2026-09-04. The pieces mostly exist:

- **The compress core is already shared.** `c4_v2.cuh` calls the same
  `c4_forward<...>` from both `flash_c4_decode` (:284) and `flash_c4_prefill` (:315). The
  only difference is the plan and the source row: prefill uses `PlanC` and
  `kv_src = kv_input + plan.ragged_id * kElementSize` (:308).
- **The store already has an extend mode.** `fused_norm_rope_v2.cuh` templates on
  `ForwardMode kMode`, and `fused_norm_rope_indexer_fp4` handles `CompressExtend` (:260):
  `plan = PlanC[work_id]`, skip on `plan.is_invalid()`, `out_loc = out_loc[plan.ragged_id]`
  (decode instead skips on `seq_len % ratio` and indexes `out_loc[work_id]`).

**No grid-wide barrier is needed** — check the launch order before assuming one is.
`run_prefill` launches `prefill_c_kernel` (compress) **first** and `prefill_w_kernel`
(the ring-buffer staging) **second** (`c4_v2.cuh:480-487`). So the compress never depends on
rows staged by this launch: the current token comes straight from
`kv_src = kv_input + plan.ragged_id * kElementSize`, and the ring buffer only supplies the
overlap from *previous* steps via `plan.read_page_0/1`. Decode is the same shape, it just
stages first because a token stages its own row.

So the extend fusion replaces **2 of 3 launches**: fuse `flash_c4_prefill` with the
norm/rope/rotate/fp4 store, and keep the staging write as its own launch. That removes the
compressed-row HBM round-trip, which is the entire point of #34624.

**Wiring constraint:** `run_prefill` launches both kernels together, so there is no existing
entry point for "stage only". Do **not** add one to `c4_v2.cuh` — that is the common
all-platform file the #34624 rework deliberately restored byte-for-byte, and touching it
re-opens the reviewer's objection. Copy `write_c4_prefill` into the HIP-only fused file
instead, the same deliberate-copy pattern already used for the compress core, and have the
fused extend entry point launch both.

Indexing for the extend kernel, from `flash_c4_prefill` (`c4_v2.cuh:301-316`):

```
plan       = plan_c[plan_id];  if (plan.is_invalid()) return;   // wave-uniform: 1 plan/wavefront
kv_src     = kv_input + plan.ragged_id * kElementSize
kv_buf_0/1 = kv_buffer + plan.read_page_0/1 * kPageElementSize
compress   = c4_compress_core(..., plan.seq_len > 4, plan.buffer_len)   // decode passes 8
position   = plan.seq_len - ratio
out_loc    = out_loc[plan.ragged_id]                                    // decode: out_loc[work_id]
```

Work count is `num_compress`, not `batch_size`. Note the decode kernel's wave-uniform
early-out is the `seq_len % ratio` check; for extend it becomes `plan.is_invalid()`, which is
also wave-uniform at one plan per wavefront — but re-verify that if the mapping changes.

Concretely: a `flash_c4_extend_norm_rope_indexer_w64` alongside the w64 decode kernel, reading
`PlanC[token_id]`, `kv_src` via `plan.ragged_id`, `out_loc[plan.ragged_id]`, calling the
existing `fused_c4::c4_compress_core`, then the part 4a fp4 epilogue unchanged. Skip on
`plan.is_invalid()` instead of the `seq_len % ratio` check — and note that check is currently
what makes the decode early-out wave-uniform, so re-verify uniformity for the extend variant.

## Why not aiter

The compress cannot move into aiter. `FusedCompress4NormRopeParams` takes
`const PlanD* plan_d` (sglang's compressor plan struct) and the kernel both **reads and
writes** `kv_buffer` — `c4_write_decode` stages the current step's kv/score row into the
ring buffer before the window softmax. The Trait also encodes sglang's buffer geometry
(`kOverlapOffset = kHeadDim`, `kScoreOffset = 2*kHeadDim`, `kElementSize = 4*kHeadDim`,
`kPageElementSize = 4*kElementSize`). aiter's ops take plain tensors; putting sglang cache
state and plan structs behind an aiter entry point is the wrong boundary.

## What the trace found — the gap is much smaller than it looks

The fused kernel already runs the **whole** FP4 transform pipeline. Its indexer variants are
documented as `compress -> RMSNorm -> RoPE -> il -> fp8 quant -> paged store`, and "il" is
the same **128-point Hadamard** aiter applies, with the same `rsqrt(kHeadDim)` scaling
(`fused_compress4_norm_rope_hip.cuh` part 3, and aiter's `do_rotate_act` butterfly with
`rotate_dim_rsqrt<dim>()`). So norm, rope and rotate all already match.

**Only the quantise-and-store step differs**, and sglang already has an fp4 version of it —
just writing a different cache layout. `fused_norm_rope_v2.cuh` (the non-aiter FP4 indexer
path, `forward_fp4`) already does exactly the right grouping:

```350:374:python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh
    local_max = warp::reduce_max<8>(local_max);
    const auto scale_raw = fmaxf(1e-4f, local_max) / 6.0f;
    const auto scale_ue8m0 = static_cast<uint8_t>(cast_to_ue8m0(scale_raw));
    const auto inv_scale = inv_scale_ue8m0(scale_ue8m0);
    const uint8_t packed0 = quant_fp4_e2m1(data[0] * inv_scale) | (quant_fp4_e2m1(data[1] * inv_scale) << 4);
    // ... 4 elems/lane, 8 lanes per group => group_size 32, 4 groups per 128-dim row
    if ((lane_id & 7) == 0) static_cast<uint8_t*>(scale_ptr)[lane_id >> 3] = scale_ue8m0;
```

`group_size 32`, 4 groups per row, UE8M0 scales — identical to what aiter is called with
(`_GROUP_SIZE = 32`, `_HEAD_DIM = 128`, `_ROPE_DIM = 64`, `_KV_BLOCK_SIZE = 64`).
`quant_fp4_e2m1` lives in `fused_norm_rope_v2.cuh:25`; `cast_to_ue8m0` / `inv_scale_ue8m0`
in `include/sgl_kernel/deepseek_v4/fp8_utils.cuh`.

## The three cache layouts (this is what actually differs)

| path | buffer | bytes/token |
|---|---|---|
| fp8 | one fused `(num_pages, page_bytes)` | 132 = 128 codes + 4B fp32 scale |
| JIT fp4 (non-aiter) | one fused buffer | 68 = 64 codes + 4B (4× UE8M0) |
| **aiter fp4 (HIP, ours)** | **split**: payload `(num_pages, 1, 4, page_size, 16)` as `float4_e2m1fn_x2` + scale `(num_pages, 1, 4, page_size)` uint8 | 64 + 4, preshuffled |

`deepseek_v4_memory_pool.py:312-330`. `c4_page_size = page_size // 4`, so with the arms'
`--page-size 256` the indexer pool's page is **64**, matching aiter's `_KV_BLOCK_SIZE`.

## The two offset formulas to reproduce (from `aiter/csrc/kernels/dsv4_rotate_quant.cu`)

`k_tiles = dim/128 = 1`, `kv_block_size = 64`, `block_idx = out_loc >> kPageBits`,
`pos_in_block = out_loc & mask`.

Payload (`kv_fp4_preshuffle_offset`, line 127), with `k_tiles == 1` so `k_tile == 0`:

```
group4 = packed_byte_idx / 16
sub16  = packed_byte_idx % 16
offset = block_idx*4*kv_block_size*16 + group4*kv_block_size*16 + pos_in_block*16 + sub16
```

Scale (`kv_scale_preshuffle_offset`, line 143) — **the scale axis is interleaved**, and this
is the part with no analogue in the fp8 epilogue:

```
tiles_per_block = kv_block_size / 16          # 4 at page 64; 1 at page 16
sflat  = (pos_in_block % 16) * tiles_per_block + (pos_in_block / 16)
offset = block_idx*4*kv_block_size + group4*kv_block_size + sflat
```

The interleave exists so `pa_mqa_logits_fp4`'s packed-dword load of NTPW e8m0 bytes is
contiguous (aiter's own comment, lines 151-159). Compute `tiles_per_block` from `kPageSize`
— do not hardcode 4.

## Which kernel to extend

**The wave-mapped one: `flash_c4_decode_norm_rope_indexer_w64`.** The file's own comment
(lines 517-533) says the 32-lane warp-mapped variant *loses* to the two kernels it replaces
on ROCm: two tokens share a wavefront so the `seq_len % ratio` early-out is not
wave-uniform (44 % of wavefronts run the full tail for one live token), and 4 elems/lane
keeps `score_fp32[4][8]` live at 85 VGPRs / 5 waves per SIMD. The w64 variant is
`FusedC4Trait<128, 2>`: 2 elems/lane, 64 lanes per token.

Mapping falls out cleanly at 2 elems/lane — **1 byte of fp4 per lane**:

```
packed_byte_idx = wave_lane          # 64 lanes -> 64 bytes -> 128 fp4 codes
group4          = wave_lane / 16     # 16 lanes * 2 elems = 32 dims = group_size
sub16           = wave_lane % 16
scale reduce    = max over 16 lanes; written by (wave_lane & 15) == 0 at group4 = wave_lane >> 4
```

## Known numerical differences to expect in the differential test

- aiter's absmax floor is `fp4_max * FLT_MIN` (`eps_amax`); sglang's is `fmaxf(1e-4f, ...)`.
  Differs only for near-zero rows.
- aiter takes **bf16** `cos`/`sin` tables; the fused kernel does RoPE from fp32
  `freqs_cis`.
- The fused kernel keeps the compressed row in registers instead of rounding it to bf16 on
  the way out — already documented as up to ~1e-4 of fp8 codes differing by one code.

So the test must allow a small code-level delta, not demand bit-equality. Do **not** promote
this to GSM8K until the differential test passes: a wrong layout here is silent, and
#37423's precedent is a permuted weight that passed every shape/dtype/stride check and
scored **GSM8K 0**.
