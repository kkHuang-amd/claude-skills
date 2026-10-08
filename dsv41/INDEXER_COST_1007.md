# V4.1 HIP indexer cost in prefill: optimization backlog

Owner node: crsuse2-m2m-259 (only the owner edits CONTINUE HERE; other nodes append to the Log or take a row in Backlog)
Created: 2026-10-07. Source data: FP4_INDEX_PLANE_PORT.md P0.3b, results/fp4_index_scorer.md "P0.3b" table.

## CONTINUE HERE

**Status:** I1 is done and pushed. It is commit 0bac6d8607 on HaiShaw/sglang `perf/v41-index-q-prefill-fuse`, on top of
aa5551d9b6 (main of 10-07). The bitwise check was re-run on the rebased tree (15/15, num_warps=2).
- bitwise equal;
- prefill q -53..-60%, prefill fwd -1.2..-1.6%;
- GSM8K 0.908/0.912 vs main 0.907;
- decode TPOT neutral.
**Next:**
- Open the upstream PR from that branch if the user wants it ([AMD][V4.1] title, template per NEW_WORKSPACE_PROMPT).
- I4 microbenched (see "I4 microbench"): B dropped, A ~1% at 131k only, C (no logits rectangle, ~4-5% at 131k, effort L)
  is the only big lever. The user chose to try bf16 logits first.
- I4 bf16 logits: prototype is exact (bitwise = rounded fp32). Microbench ~1.6% of fwd at 131k, 0.7% at 66k; the scorer is
  compute-bound, so the C estimate dropped to ~2-3%. Next: fix the block-max kernel, then in-server spans, then a long-context A/B.
  See "I4 bf16 logits".
**Repro (measurement):**
```bash
bash /workspace/claude-skills/dsv41/scripts/fp4idx_prefill_probe.sh
```
This needs the instrumented worktree `/sgl-workspace/sglang-fp4idx` (env `SGLANG_DSV41_IDX_TIMING`). It uses GPUs 0-3 and
writes `summary.txt` per forward.
**Pass criteria:** an item ships if it cuts prefill forward time by >= 1% with GSM8K unchanged (TP4 bar ~0.90), or if it
removes a stall seen on live requests.

## What was measured (2026-10-07, crsuse2-m2m-259, TP4 EP1 V4.1-Flash)

Setup:
- SGLang main 16a23a672b, aiter e7d2453f2, the TP4_GAP_1006.md lane config, chunk 16384.
- Each span is rank-0 GPU stream time: event pairs with no sync.
- Values are warm medians.
- The indexer runs on 8 layers: 2, 8, 14 (r=2) and 20, 24, 28, 32, 36 (r=1). They are replicated per TP rank,
  32 heads x 128.
- "other" = indexer - score - q_inputs: top-k transform, candidate select/apply, identity fills and metadata.

| new tok | ctx | fwd ms | indexer ms (%) | q_inputs ms | score ms | other ms |
|---|---|---|---|---|---|---|
| 4096 | 4k | 75.0 | 3.10 (4.1) | ~2.0 | 0.36 | ~0.7 |
| 4096 | 33k | 77.4 | 4.84 (6.3) | ~2.0 | 1.52 | ~1.3 |
| 4096 | 131k | 77.4 | 10.16 (13.1) | ~2.0 | 5.40 | ~2.8 |
| 16384 | 16k | 213.2 | 10.05 (4.7) | ~6.9 | 1.40 | ~1.8 |
| 16384 | 66k | 225.8 | 21.84 (9.7) | ~6.9 | 9.06 | ~5.9 |
| 16384 | 131k | 243.4 | 37.99 (15.6) | ~6.9 | 20.42 | ~10.7 |

**JIT stalls:**
- The first forward at a new page-table width stalls 0.2-5.4 s in the FlyDSL prefill scorer. Widths are bucketed at
  64 pages = 4096 keys.
- The first workspace build at a new shape stalls 0.7-3.9 s.
- One run produced 15 new compiles.
- Results persist in `aiter/jit/flydsl_cache`, so this happens once per node or container. Every fresh deployment still
  hits it on live requests.

**Host side:** not exposed in prefill. Every span's GPU time is at or below its host time, so the stream stays backlogged.

## Backlog (expected value is an estimate until measured)

| ID | Item | Evidence | Idea | Est. gain | Effort | Owner | Status |
|---|---|---|---|---|---|---|---|
| I1 | q path in prefill | q_inputs ~6.9 ms per 16k fwd (3.2%); split: pack 4.55 + rope_fq 1.54 + GEMM 0.58 + weights 0.21 | fused rows-tiled RoPE+fq+pack prefill kernel, bitwise equal | measured: q 7.10 -> 3.31 ms per 16k fwd; fwd -1.2..-1.6% | S | crsuse2-m2m-259 | implemented; GSM8K OK (0.908/0.912 vs 0.907); decode TPOT neutral | ready to upstream (ask) |
| I2 | Per-width JIT stalls (scorer + workspace build) | server: 0.2-5.4 s and 0.7-3.9 s on first use of each width. Microbench (P0.4): A compiles 1 kernel per 4096-key bucket, ~0.16 s cold / ~42 ms warm-disk; the Triton workspace build compiles per row-count shape, 0.55-1.9 s cold. Row-group B does NOT help (also per width, 1.18 s cold) | (a) explain the server-vs-microbench gap first (4-rank concurrent compile? several variants per fwd?); (b) aiter: make the key width a runtime arg of `flydsl_pa_mqa_logits_fp4_prefill`; (c) prewarm widths at startup (~256 buckets to 1M; ~11 s warm-disk / ~42 s cold per process [inferred]); (d) coarser bucket | removes TTFT spikes on fresh nodes and on first long requests | S-M | | todo |
| I3 | Scorer at long context | 8.4% of fwd at 131k (16k chunk) | row-group B, 1.3-1.4x at keys >= 4k (FP4_INDEX_PLANE_PORT.md P1, parked) | ~2.3% at 131k, ~0.5% at 33k | M | | parked |
| I4 | select (top-k / candidates) at long context | MEASURED, see "I4 measured": select = 0.75% / 1.4% / 2.5% / 4.2% of a 16k fwd at 16k / 33k / 66k / 131k ctx. Memory-bound passes over the fp32 logits rectangle | A: fold layer-20 publish (block maxima) into its top-k pass, ~2-2.5 ms at 131k. B: tune topk_v2 toward roofline (~57% now), ~3-4 ms. C: top-k inside the scorer, no logits rectangle (aiter), ~8-12 ms | 1-5% at 131k only; <1.5% at <=33k | M / M / L | crsuse2-m2m-259 | microbenched: B has ~no headroom, A ~1% at 131k only, C is the only >1% lever (see "I4 microbench") |
| I5 | DSpark target-verify scoring | verify scorer = 0.5-2.3% of the decode step (c1..c64; 6 rows/req); row-group 1.5-2.4x at bs>=8 | P3 in FP4_INDEX_PLANE_PORT.md | 0.4-1.3% of the decode step (c32 best) | M | | sized, low priority |

### I4 measured (2026-10-07, crsuse2-m2m-259, run i4_select_259, extra spans in select)

**Per layer at ctx 131k, 16k new tokens, fwd 244 ms (ms, median of 3 forwards):**

| layers | indexer | score | select |
|---|---|---|---|
| 2 / 8 / 14 (r=2, full) | ~7.4 each | ~4.0 each | ~1.5 each |
| 20 (r=1, candidate source, full) | 15.4 | 8.05 | 5.57 (top-k + publish) |
| 24 / 28 / 32 / 36 (consumers) | 0.22 each | 0.08 each | 0.03 each |

**The consumer layers do NOT score the full context.** Earlier text in this repo's chat history inferred they did; the
measurement refutes it. So "score only candidate blocks" (ATOM page-8, FP4_INDEX_PLANE_PORT.md P2) has little left to
save.

**Select split per 16k forward (ms):**

| ctx | select | sel_topk | sel_publish | sel_consume | rest (fills/metadata) |
|---|---|---|---|---|---|
| 16k | 1.61 | 1.25 | 0.18 | 0.09 | 0.32 |
| 33k | 3.00 | 2.09 | 0.67 | 0.10 | 0.29 |
| 66k | 5.70 | 3.94 | 1.47 | 0.10 | 0.41 |
| 131k | 10.31 | 7.10 | 2.81 | 0.10 | ~0.4 |

**Roofline (5 TB/s) [inferred]:**
- The scorer writes the [rows, lc] fp32 logits, and top-k (plus publish on layer 20) reads them again.
- At 131k: r=2 layers have a ~4.3 GB rectangle, so top-k is >= 0.86 ms against 1.5 ms measured (~57%).
- Layer 20 has ~8.6 GB, read twice (top-k + publish), so >= 3.4 ms against 5.57 ms measured.

### I4 microbench (2026-10-07, crsuse2-m2m-259, 1x MI355X, `scripts/i4_select_bench.py`)

16384 causal rows, fp32 logits, ms (median of 5). `amax(dim=1)` over the same rectangle is the one-read baseline:

| lc | amax (1 read) | top-k v2 | block-max (publish pass) | publish total |
|---|---|---|---|---|
| 16k | 0.21 (5.1 TB/s) | 0.20 | 0.17 | 0.21 |
| 33k | 0.38 (5.7 TB/s) | 0.51 | 0.44 | 0.62 |
| 66k | 0.73 (5.9 TB/s) | 1.01 | 1.10 | 1.33 |
| 131k | 1.50 (5.7 TB/s) | 2.28 | 2.26 | 2.69 |

**The "57% of roofline" above is wrong.** Rows longer than 8192 take `TopKStreaming` (topk_impl.cuh), which reads the row
TWICE: histogram, then collect. Counted as two reads it runs at 7.1 TB/s, above HBM, so the second pass partly hits
L2/MALL. Top-k is ~1.5x one read; only a single-pass top-k could save more, and a 512 KB row does not fit LDS.
**B is dropped.**
- **Block-max** (`_candidate_block_scores_kernel`) runs at ~3.5 TB/s, ~60% of one read. Fixing it alone would save
  ~0.8 ms on layer 20 (~0.3% of the 244 ms fwd).
- **A** (block maxima computed inside top-k's first pass, or in the layer-20 scorer epilogue) removes the whole pass:
  ~2.3 ms in microbench, ~1% of fwd at 131k, <0.3% at <=33k. That is right at the 1% gate, and only at 131k.
- **C** is the only lever above 1%. Server bytes at 131k: the scorer writes ~21.5 GB of logits (3 x 4.3 + 8.6), then
  top-k and publish read them back. Score + select there is ~31 ms (12.6% of fwd). Not writing the rectangle saves the
  write and every re-read [inferred: ~10-12 ms, 4-5%]. That needs a top-k fused into the FlyDSL scorer: per key-tile
  partial top-k plus a merge. Effort L.
- **Cheaper variant of C** [untested]: bf16/fp16 logits would halve every byte above, but it changes ties, so it needs a
  top-k overlap check and GSM8K.

### I4 bf16 logits (2026-10-07, crsuse2-m2m-259, in progress)

**Accuracy proxy** (`scripts/i4_bf16_overlap.py`): synthetic logits `sum_h w_h relu(q_h . k)`, H=32, D=128, random
q/k, top-512 of 16k / 131k. Random data has dense near-ties, so this is a pessimistic case.
- bf16 storage changes 0.4-0.7% of today's selected set. Regret against exact logits does not change (2.0e-2).
- FP4 q/k quantization (today's path) already changes 17-22% of the exact set, so bf16 adds noise two orders of
  magnitude smaller than what is already there.
- fp16 changes only 0.1%, but the real logit range is unknown, so overflow is a risk. bf16 is the choice.
- **GSM8K does NOT test this.** Its prompts compress to fewer than 512 positions, so top-k selects everything.
  Use a long-context eval.

**top-k v2 bf16 variant** (worktree `/sgl-workspace/sglang-i4`, branch `dsv41-i4-bf16-logits`, base 16a23a672b +
timing patch 0002):
- Compile-time `SGL_TOPK_BF16` builds a second JIT module (`topk_v2_bf16`). It reads 16-byte vectors of 8 bf16 and does
  every compare, histogram bin and tie value in fp32.
- `topk_transform_paged_v2` picks the module by `scores.dtype`.
- `candidate_block_scores` now accepts bf16 and still outputs fp32 (exact).

Microbench (`scripts/i4_bf16_topk_bench.py`): 16384 causal rows, ms. Check: value multiset equals torch.topk, 0/512 bad
at every lc.

| lc | amax fp32 / bf16 | top-k fp32 / bf16 | publish fp32 / bf16 |
|---|---|---|---|
| 4k | 0.04 / 0.03 | 0.06 / 0.06 | 0.08 / 0.08 |
| 16k | 0.21 / 0.18 | 0.21 / 0.20 | 0.21 / 0.19 |
| 33k | 0.36 / 0.26 | 0.51 / 0.35 | 0.60 / 0.55 |
| 66k | 0.71 / 0.44 | 1.05 / 0.69 | 1.33 / 1.10 |
| 131k | 1.50 / 0.82 | 2.25 / 1.25 | 2.66 / 2.28 |

- Top-k is -31..-44% at >= 33k.
- Publish barely moves. The block-max Triton kernel is not bandwidth-bound; it ran at ~60% of one read even in fp32.

**bf16 scorer epilogue** (2026-10-08). aiter worktree `/sgl-workspace/aiter-i4` (e7d2453f2 + the local
`lru_cache` -> `cache` edit, branch `dsv41-i4-bf16-logits`): `build_pa_mqa_logits_fp4_prefill_module(out_bf16=)`
converts the fp32 sum to bf16 (RNE) and stores it with BufferCopy16b. The wrapper picks this from `out.dtype`. SGLang
`_alloc_logits` returns bf16 prefill logits when `SGLANG_DSV41_PREFILL_LOGITS_BF16=1` (default 0; decode stays fp32),
viewing the same pooled block.
- Patches: `patches/sglang_local_i4_bf16_logits_0001.patch` (on top of timing patch 0002) and
  `patches/aiter_local_i4_bf16_logits_0001.patch`.
- Bench: `scripts/i4_bf16_scorer_bench.py`. One request, 16k new tokens, SGLang path with the prefill workspace, ms.
- **Bitwise:** bf16 logits == round-to-bf16(fp32 logits) on every valid column, 0 mismatches in every case.

| ctx, ratio | lc | score fp32 / bf16 | top-k fp32 / bf16 |
|---|---|---|---|
| 16k, r=2 | 8k | 0.43 / 0.40 | 0.15 / 0.16 |
| 33k, r=2 | 16k | 0.99 / 0.94 | 0.25 / 0.26 |
| 66k, r=2 | 33k | 2.00 / 1.88 | 0.54 / 0.37 |
| 131k, r=2 | 66k | 3.97 / 3.78 | 1.08 / 0.65 |
| 131k, r=1 | 131k | 8.02 / 7.47 | 2.30 / 1.13 |

**The scorer is compute-bound, not write-bound.** It writes 8.6 GB in 8 ms, ~1 TB/s, and halving the bytes only saves
5-7%.
- **This corrects the C estimate above.** Not writing the rectangle saves the write share (~0.2-0.55 ms per layer)
  plus top-k's reads, not 10-12 ms. C is now ~2-3% at 131k at best, with effort L.
- **bf16 total, per 16k fwd** (microbench, layers 2/8/14 at r=2 + layer 20 at r=1): 131k ctx ~4 ms (score 1.1 +
  top-k 2.5 + publish 0.4), ~1.6% of 244 ms. 66k ~1.5 ms (0.7%). <=33k ~0.3%.
- **Next, in order:**
  1. Fix the block-max kernel (bf16 publish 2.28 ms against a 0.82 ms one-read floor at 131k): ~1.4 ms more at 131k,
     taking bf16 + block-max to ~2.2%.
  2. In-server spans at 131k with the env on/off (`fp4idx_prefill_probe.sh` pointed at sglang-i4 + aiter-i4).
  3. A long-context accuracy A/B (not GSM8K).

### I1 measured split (2026-10-07, crsuse2-m2m-259, in-server spans, run i1_qsplit_259)

Sums over the 8 index layers per prefill forward (ms):

| new tok | q_wqb (MXFP8 GEMM) | q_rope_fq | q_pack | q_weights | total |
|---|---|---|---|---|---|
| 4096 | 0.26 | 0.44 | 1.14 | 0.21 | ~2.05 |
| 16384 | 0.58 | 1.54 | 4.55 | 0.21 | ~6.9 |

**Root cause:** both elementwise kernels launch one tiny program per (token, head) row of 128 elements.
- **`pack_fp4_query_flydsl`** uses grid (T, 64) while V4.1 has 32 heads, so half of its ~1M programs only write
  zero scale bytes. Per layer it moves ~170 MB, roughly a 0.04 ms roofline, but measures 0.57 ms (~14x off).
- **`rope_tail_fake_quant_fp4`** uses grid (T*H,). It measures 0.19 ms per layer against a ~0.06 ms roofline.
- **The GEMM is already fast.** `q_lora` arrives pre-quantized (`Mxfp8Activation`), so `wq_b` has no activation-quant step.

**Plan:**
- One prefill kernel does RoPE + fake-quant + pack. Each program covers several heads of a token and reuses
  `rope_tail_fake_quant_fp4_row` + `quantize_fp4_indexer_row`, the same row functions as the decode fused kernel
  `_index_q_pack_weights_kernel`, so it is bitwise identical.
- Head weights stay as they are.
- Target: 6.1 -> <1 ms per 16k forward, which is ~2.4% of prefill fwd. At 4k: ~1.3 ms of 77 ms, ~1.7%.

### I1 implementation and results (2026-10-07, crsuse2-m2m-259)

**Code:**
- Branch `dsv41-index-q-prefill-fuse` in worktree `/sgl-workspace/sglang-i1`, based on main 16a23a672b. Uncommitted,
  saved as `patches/sglang_local_index_q_prefill_fuse_0001.patch`; apply it with `git apply`.
- New `index_q_rope_pack_flydsl` / `_index_q_rope_pack_kernel` in `kernels/ops/attention/dsv4/fp4_indexer_hip.py`.
  Its grid is (T, H/8), each program walks 8 heads, and it reuses the decode fused kernel's row functions.
- `_indexer_inputs` (low_ratio_backend_hip.py) uses it for prefill rows when H % 16 == 0, H <= 64, D == 128 and freqs
  are complex64. The weights path is unchanged.

**Bitwise:** `scripts/i1_q_rope_pack_check.py` and `i1_q_rope_pack_timing.py`.
- With num_warps=2 or 4: identical q_fp4 + q_scale for H=16/32/64, T=1..16384, including out-of-range positions, plus a
  20-seed stress at T=16384 H=32 with 0 mismatched bytes.
- num_warps=1 is ~1.9x faster again but flips about 1 tie in 4M codes, e.g. -2.0 vs -1.5 where the RoPE value sits on a
  rounding boundary. The difference is codegen (FMA contraction per lane layout), not logic. **Shipped with
  num_warps=2.**

**Microbench (1 GPU, GPU time):** T=16384 H=32 goes from 1420 us (rope 360 + pack 1057) to 545 us with nw2 (293 us
with nw1). Microbench absolute times are ~1.9x the in-server ones for both kernels [inferred: idle-GPU clocks]; the
ratios hold.

**In-server (TP4, spans, runs i1_qsplit_259 -> i1_fused_259):**

| new tok | q_inputs ms per fwd (8 layers) | fwd ms, vs P0.3b baseline | TTFT ms (client, prefix 0) |
|---|---|---|---|
| 16384 | 7.10 -> 3.31 (-53%) | 213.2 -> 210.7 (-1.2%) | 226.1 -> 222.4 (-1.6%) |
| 4096 | 2.40 -> 0.96 (-60%) | 75.0 -> 73.8 (-1.6%) | 87.9 -> 87.5 |

Forward deltas are within ~1-2 ms of run-to-run noise; the q_inputs saving (3.8 ms per 16k fwd) is stable.
**Left in q at 16k:** fused ~2.5 ms + wq_b 0.58 + weights 0.21. num_warps=1 would take another ~1.1 ms if the tie
flips are acceptable.

**GSM8K A/B** (`scripts/tp4_gsm8k_ab.sh`, EVAL_ONLY TP4, 5-shot 1319; rows in results/gsm8k.md):

| run | accuracy | eval latency |
|---|---|---|
| fused, 1st run | 0.908 | 64.6 s |
| main | 0.907 | 32.2 s |
| fused, rerun on warm caches | 0.912 | 36.2 s |

- **Accuracy:** equal within run-to-run noise (~1 pt), as expected from bitwise-equal q codes.
- **The 64.6 s first run was a cold effect.** Startup was ~13.5 min on both sides; the warm rerun took 36.2 s.
- **Latency is unresolved.** The fused rerun's 36.2 s vs main's 32.2 s (decode gen throughput mean 3675 vs 3801 tok/s)
  is one sample each and not resolved.
- **Why decode is in scope:** the change also routes target-verify rows above the GEMV row limit to the fused kernel,
  so decode is touched.
- **Decode TPOT A/B, done.** `tp4_decode_profile.sh` PROFILE=0 CONC=32 LOAD_CONC=16,32,16,32 ISL=32768 OSL=1024
  MEM=0.80, one server per side, 2 samples per conc; logs `/shared_nfs/kk/dsv41/profile_tp4/i1_tpot_{base,i1}/`.
  Clean TPOT ms:

  | conc | main | fused (i1) |
  |---|---|---|
  | 16 | 3.878, 4.017 | 3.946, 3.876 |
  | 32 | 6.208 (one slow straggler; min 5.322), 5.320 | 5.214, 5.321 |

  **No decode regression.** The same-config spread is ~3.5% (c16 main 3.878-4.017) and the fused side is equal or
  slightly faster. The +12% GSM8K eval latency on the warm rerun was noise.

### I1 detail (original hypothesis)

The prefill path is `indexer.queries(q_lora, freqs_cis, pos)`. That is the `wq_b` GEMM plus RoPE plus FP4
fake-quant (`_rope_fq4`).
- Then `pack_fp4_query_flydsl(q)` quantizes and packs again, and `_indexer_head_weights(indexer, x)` is a separate GEMM.
- **The GEMM alone should not take 0.86 ms.** `wq_b` is 16384 x 1280 -> 4096, about 172 GFLOP; weights_proj is
  16384 x 5120 -> 32, memory-bound at ~170 MB. Both together should be ~0.15-0.25 ms per layer [inferred, no roofline
  measured yet], so most of the 0.86 ms is probably RoPE / fake-quant / pack elementwise work.
- **The fused kernel already exists, for decode only.** `index_q_rope_pack_weights_flydsl`
  (`low_ratio_backend_hip.py:_indexer_inputs`) does RoPE + FP4 pack + head-weight reduce in one launch, but it is gated
  to small row counts (`_gemv_head_weight_rows`).
- **ATOM's version:** #2479 does q RoPE + FP4 quant for prefill with one aiter `rope_rotate_activation(...,
  shuffle_scale=True, round_rope=True, do_rotate_act=False)` call.
- **First step:** time the sub-steps of `_indexer_inputs` at T=16384 (GEMM / RoPE+fq / pack / weights). Then try
  the fused kernel at prefill row counts. Numerics must stay bit-identical to `_rope_fq4` and the pack, or be A/B'd on
  top-k and GSM8K.

## Log (append-only; date, node, what)

- 2026-10-07 crsuse2-m2m-259: created from P0.3b data (`/shared_nfs/kk/dsv41/fp4_index_port/p03b_259_1007_0337/`).
- 2026-10-07 crsuse2-m2m-259: I2 updated with P0.4. Row-group also compiles per key width; I3 is parked.
- 2026-10-07 crsuse2-m2m-259: I5 sized; see results/fp4_index_scorer.md "P3 sizing".
- 2026-10-07 crsuse2-m2m-259: I1 split measured. The pack and rope kernels are launch-bound (one program per head row). Started the fused prefill kernel.
- 2026-10-07 crsuse2-m2m-259: I1 implemented; bitwise check and in-server q -53% / fwd -1.2..-1.6%. GSM8K A/B running.
- 2026-10-07 crsuse2-m2m-259: I1 GSM8K A/B: 0.908 / 0.912 (warm rerun) vs main 0.907. Eval latency +12% on the warm rerun (n=1); a decode TPOT A/B is queued as next.
- 2026-10-07 crsuse2-m2m-259: I1 decode TPOT A/B neutral (c16 3.95 vs 3.91, c32 ~5.32 vs ~5.27 ms). I1 is ready to upstream.
- 2026-10-07 crsuse2-m2m-259: I1 pushed to HaiShaw/sglang perf/v41-index-q-prefill-fuse (0bac6d8607).
- 2026-10-07 crsuse2-m2m-259: I4 measured. select is memory-bound over the logits rectangle (4.2% at 131k); the consumer layers are already cheap, so P2 page-8 has little value. Options A/B/C are listed.
- 2026-10-07 crsuse2-m2m-259: I4 microbench. Streaming top-k reads twice (already ~1.5x one read), so B is dropped. Block-max runs at 60% BW; A ~1% at 131k only; C (no logits rectangle) is the only >1% lever.
- 2026-10-07 crsuse2-m2m-259: I4 bf16. Accuracy proxy OK (0.4-0.7% set change vs 17-22% from FP4). top-k v2 bf16 variant is exact and -44% at 131k; publish block-max is not BW-bound. Next: bf16 scorer epilogue.
- 2026-10-08 crsuse2-m2m-259: I4 bf16 scorer epilogue is bitwise = rounded fp32. The scorer is compute-bound (-5..-7%), top-k -40..-51% at >=66k. Microbench total ~1.6% at 131k. C estimate revised to ~2-3%.
