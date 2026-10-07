# V4.1 HIP indexer cost in prefill: optimization backlog

Owner node: crsuse2-m2m-259 (only the owner edits CONTINUE HERE; other nodes append to the Log or take a row in Backlog)
Created: 2026-10-07. Source data: FP4_INDEX_PLANE_PORT.md P0.3b, results/fp4_index_scorer.md "P0.3b" table.

## CONTINUE HERE

**Status:** findings recorded, nothing implemented. This doc is a planning backlog.
**Next:** pick an item from Backlog. I1 (q path) and I2 (JIT stalls) have the best expected value per effort.
P0.4 showed row-group does not fix I2; see I2 options.
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
| I1 | q path in prefill | q_inputs ~0.86 ms per layer per 16k tokens, ~6.9 ms per fwd (3.2%) at EVERY context length | see "I1 detail" | maybe 1.5-2.5% of prefill fwd [inferred] | S-M | | todo |
| I2 | Per-width JIT stalls (scorer + workspace build) | server: 0.2-5.4 s and 0.7-3.9 s on first use of each width. Microbench (P0.4): A compiles 1 kernel per 4096-key bucket, ~0.16 s cold / ~42 ms warm-disk; the Triton workspace build compiles per row-count shape, 0.55-1.9 s cold. Row-group B does NOT help (also per width, 1.18 s cold) | (a) explain the server-vs-microbench gap first (4-rank concurrent compile? several variants per fwd?); (b) aiter: make the key width a runtime arg of `flydsl_pa_mqa_logits_fp4_prefill`; (c) prewarm widths at startup (~256 buckets to 1M; ~11 s warm-disk / ~42 s cold per process [inferred]); (d) coarser bucket | removes TTFT spikes on fresh nodes and on first long requests | S-M | | todo |
| I3 | Scorer at long context | 8.4% of fwd at 131k (16k chunk) | row-group B, 1.3-1.4x at keys >= 4k (FP4_INDEX_PLANE_PORT.md P1, parked) | ~2.3% at 131k, ~0.5% at 33k | M | | parked |
| I4 | "other" (top-k / candidates) at long context | ~10.7 ms per fwd at 131k (4.4%), grows with ctx | Profile first: split `_select_topk_extend_hip` into top-k / candidate select / consumer apply. Candidate layers 24..36 may score the full rectangle and then mask | unknown, up to ~4% at 131k | M | | todo (measure) |
| I5 | DSpark target-verify scoring | verify scorer = 0.5-2.3% of the decode step (c1..c64; 6 rows/req); row-group 1.5-2.4x at bs>=8 | P3 in FP4_INDEX_PLANE_PORT.md | 0.4-1.3% of the decode step (c32 best) | M | | sized, low priority |

### I1 detail

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
