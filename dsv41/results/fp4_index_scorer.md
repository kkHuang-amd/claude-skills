# FP4 index scorer microbench (FP4_INDEX_PLANE_PORT.md)

Append-only. Each row has a node and a date. The FP4 plane is E2M1 + E8M0/32, head_dim 128. The `rowgroup` column
uses the ragged `query_start_loc` path. `old` is `pa_mqa_logits_fp4` with `next_n=qlen` (one workgroup per row).
Neither is SGLang's prefill kernel (`flydsl_pa_mqa_logits_fp4_prefill`). Time is µs, from `run_perftest` with 20 iters,
on 1x MI355X.

| date | node | aiter | page | batch | ctx<= | qlen | heads | rowgroup us | old us | err (both) | log |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10-07 | crsuse2-m2m-259 | 5b2f7d1d1 | 64 | 8 | 8192 | 1 | 64 | 4.6 | 4.8 | 3.8e-6 | p02_rowgroup_page64.log |
| 10-07 | crsuse2-m2m-259 | 5b2f7d1d1 | 64 | 32 | 8192 | 1 | 64 | 5.7 | 6.5 | 3.8e-6 | 〃 |
| 10-07 | crsuse2-m2m-259 | 5b2f7d1d1 | 64 | 8 | 8192 | 4 | 64 | 7.3 | 6.6 | ≤7.6e-6 | 〃 |
| 10-07 | crsuse2-m2m-259 | 5b2f7d1d1 | 64 | 8 | 8192 | 6 | 64 | 9.2 | 8.2 | ≤7.6e-6 | 〃 |
| 10-07 | crsuse2-m2m-259 | 5b2f7d1d1 | 64 | 1 | 8192 | 128 | 64 | 9.1 | 15.2 | 3.8e-6 | 〃 |
| 10-07 | crsuse2-m2m-259 | 5b2f7d1d1 | 64 | 2 | 8192 | 256 | 64 | 19.9 | 46.9 | ≤7.6e-6 | 〃 |
| 10-07 | crsuse2-m2m-259 | 5b2f7d1d1 | 64 | 1 | 16384 | 512 | 64 | 24.2 | 88.8 | ≤7.6e-6 | 〃 |
| 10-07 | crsuse2-m2m-259 | 5b2f7d1d1 | 64 | 4 | 4096 | 1 | 32 | 3.9 | 3.8 | ≤1.9e-6 | 〃 |

Logs: /shared_nfs/kk/results/DeepSeek-V4.1-Flash/fp4_index_port/. Script: scripts/fp4_rowgroup_page64_check.py.

## P0.3: SGLang production scorer (A) vs row-group (B), V4.1 shapes (heads 32, dim 128, page 64)

A is `aiter_fp4_paged_mqa_logits` with a prepared workspace and bucket 64:
- prefill: `flydsl_pa_mqa_logits_fp4_prefill` with a per-row table. Same kernel file in e7d2453f2 and 5b2f7d1d1.
- decode: `flydsl_pa_mqa_logits_fp4`.

B is `flydsl_pa_mqa_logits_fp4_rowgroup` on the request table with a prebuilt plan.

"GPU" is CUDA-graph replay (20 calls per graph); "eager" is back-to-back Python calls. Rel err A vs B ≤ 1.1e-7 everywhere.
The A schedule build (eager), which B does not need, costs 45-64 µs per step per ratio.
Node crsuse2-m2m-259, 2026-10-07, aiter 5b2f7d1d1. Log: `/shared_nfs/kk/results/DeepSeek-V4.1-Flash/fp4_index_port/p03_scorer_vs_sglang_graph.log`.
Script: scripts/fp4_scorer_vs_sglang_bench.py.

| case | ratio | rows | keys<= | GPU A us | GPU B us | A/B | eager A | eager B |
|---|---|---|---|---|---|---|---|---|
| 1x4k fresh | 2 | 4096 | 2048 | 24.7 | 29.2 | 0.84 | 132.4 | 30.5 |
| 1x8k chunk2 (4k on 4k) | 2 | 4096 | 4096 | 67.5 | 56.0 | 1.21 | 127.8 | 55.7 |
| 1x32k last chunk (4k on 28k) | 2 | 4096 | 16384 | 259.2 | 207.4 | 1.25 | 304.0 | 232.6 |
| 1x16k fresh chunk | 2 | 16384 | 8192 | 294.1 | 251.4 | 1.17 | 340.0 | 283.9 |
| 1x128k last chunk (4k on 124k) | 2 | 4096 | 65536 | 1011.8 | 743.2 | 1.36 | 1076.5 | 820.5 |
| 8x512 fresh | 2 | 4096 | 256 | 9.6 | 12.8 | 0.75 | 124.5 | 18.7 |
| 4x1k on 30k prefix (agentic) | 2 | 4096 | 15872 | 260.6 | 208.4 | 1.25 | 291.5 | 234.9 |
| decode bs1 ctx8k | 2 | 1 | 4096 | 2.6 | 3.5 | 0.74 | 126.8 | 18.8 |
| decode bs8 ctx8k | 2 | 8 | 4096 | 2.7 | 3.6 | 0.75 | 128.3 | 18.0 |
| decode bs32 ctx32k | 2 | 32 | 16384 | 6.5 | 7.9 | 0.83 | 126.0 | 19.0 |
| decode bs64 ctx8k | 2 | 64 | 4096 | 4.1 | 5.1 | 0.80 | 125.5 | 18.0 |
| verify bs8 ctx8k q5 | 2 | 40 | 4096 | 3.6 | 6.3 | 0.57 | 132.1 | 19.0 |
| verify bs32 ctx32k q5 | 2 | 160 | 16384 | 25.1 | 15.9 | 1.57 | 126.5 | 19.4 |
| 1x4k fresh | 1 | 4096 | 4096 | 47.3 | 47.0 | 1.01 | 124.8 | 47.9 |
| 1x8k chunk2 (4k on 4k) | 1 | 4096 | 8192 | 117.8 | 97.5 | 1.21 | 133.0 | 94.5 |
| 1x32k last chunk (4k on 28k) | 1 | 4096 | 32768 | 493.5 | 372.7 | 1.32 | 564.0 | 434.2 |
| 1x16k fresh chunk | 1 | 16384 | 16384 | 537.0 | 419.1 | 1.28 | 570.2 | 449.8 |
| 1x128k last chunk (4k on 124k) | 1 | 4096 | 131072 | 2071.4 | 1505.1 | 1.38 | 2142.4 | 1537.1 |
| 8x512 fresh | 1 | 4096 | 512 | 12.0 | 16.3 | 0.74 | 124.4 | 19.6 |
| 4x1k on 30k prefix (agentic) | 1 | 4096 | 31744 | 542.3 | 381.3 | 1.42 | 575.7 | 447.0 |
| decode bs1 ctx8k | 1 | 1 | 8192 | 2.6 | 3.5 | 0.75 | 126.7 | 22.4 |
| decode bs8 ctx8k | 1 | 8 | 8192 | 2.8 | 3.7 | 0.75 | 126.2 | 18.6 |
| decode bs32 ctx32k | 1 | 32 | 32768 | 13.6 | 12.7 | 1.07 | 127.1 | 23.1 |
| decode bs64 ctx8k | 1 | 64 | 8192 | 6.8 | 7.8 | 0.87 | 126.2 | 18.5 |
| verify bs8 ctx8k q5 | 1 | 40 | 8192 | 4.9 | 6.5 | 0.76 | 129.8 | 23.4 |
| verify bs32 ctx32k q5 | 1 | 160 | 32768 | 47.5 | 27.2 | 1.75 | 131.7 | 29.1 |

## P0.3 cold-cache rerun (aiter #6145 method: 2 GiB read before each call, per-call events, median of 20)

Node crsuse2-m2m-259, 2026-10-07. `COLD=1` in the same script, with 3 more decode/verify cases.
Log: `/shared_nfs/kk/results/DeepSeek-V4.1-Flash/fp4_index_port/p03_scorer_vs_sglang_cold.log`.

| case | ratio | rows | keys<= | cold A us | cold B us | A/B |
|---|---|---|---|---|---|---|
| 1x4k fresh | 2 | 4096 | 2048 | 28.8 | 33.2 | 0.87 |
| 1x8k chunk2 (4k on 4k) | 2 | 4096 | 4096 | 60.5 | 57.6 | 1.05 |
| 1x32k last chunk (4k on 28k) | 2 | 4096 | 16384 | 250.6 | 196.0 | 1.28 |
| 1x16k fresh chunk | 2 | 16384 | 8192 | 275.6 | 250.8 | 1.10 |
| 1x128k last chunk (4k on 124k) | 2 | 4096 | 65536 | 1032.9 | 752.6 | 1.37 |
| 8x512 fresh | 2 | 4096 | 256 | 13.7 | 18.0 | 0.76 |
| 4x1k on 30k prefix (agentic) | 2 | 4096 | 15872 | 263.4 | 196.4 | 1.34 |
| decode bs1 ctx8k | 2 | 1 | 4096 | 6.2 | 7.5 | 0.83 |
| decode bs8 ctx8k | 2 | 8 | 4096 | 6.4 | 6.9 | 0.92 |
| decode bs32 ctx32k | 2 | 32 | 16384 | 11.4 | 12.6 | 0.91 |
| decode bs64 ctx8k | 2 | 64 | 4096 | 8.5 | 9.4 | 0.91 |
| verify bs8 ctx8k q5 | 2 | 40 | 4096 | 6.9 | 10.0 | 0.69 |
| verify bs32 ctx32k q5 | 2 | 160 | 16384 | 27.0 | 19.0 | 1.42 |
| decode bs16 ctx64k | 2 | 16 | 32768 | 11.6 | 12.2 | 0.95 |
| decode bs64 ctx100k | 2 | 64 | 51200 | 51.7 | 47.0 | 1.10 |
| verify bs64 ctx64k q5 | 2 | 320 | 32768 | 67.6 | 48.0 | 1.41 |
| 1x4k fresh | 1 | 4096 | 4096 | 45.2 | 50.1 | 0.90 |
| 1x8k chunk2 (4k on 4k) | 1 | 4096 | 8192 | 105.8 | 93.5 | 1.13 |
| 1x32k last chunk (4k on 28k) | 1 | 4096 | 32768 | 494.4 | 387.6 | 1.28 |
| 1x16k fresh chunk | 1 | 16384 | 16384 | 507.6 | 419.0 | 1.21 |
| 1x128k last chunk (4k on 124k) | 1 | 4096 | 131072 | 2067.6 | 1492.0 | 1.39 |
| 8x512 fresh | 1 | 4096 | 512 | 17.0 | 21.4 | 0.79 |
| 4x1k on 30k prefix (agentic) | 1 | 4096 | 31744 | 514.9 | 378.0 | 1.36 |
| decode bs1 ctx8k | 1 | 1 | 8192 | 6.3 | 7.5 | 0.84 |
| decode bs8 ctx8k | 1 | 8 | 8192 | 6.4 | 7.3 | 0.88 |
| decode bs32 ctx32k | 1 | 32 | 32768 | 18.6 | 18.4 | 1.01 |
| decode bs64 ctx8k | 1 | 64 | 8192 | 11.6 | 12.4 | 0.94 |
| verify bs8 ctx8k q5 | 1 | 40 | 8192 | 8.4 | 10.2 | 0.83 |
| verify bs32 ctx32k q5 | 1 | 160 | 32768 | 48.4 | 29.8 | 1.62 |
| decode bs16 ctx64k | 1 | 16 | 65536 | 18.5 | 18.6 | 0.99 |
| decode bs64 ctx100k | 1 | 64 | 102400 | 100.5 | 89.8 | 1.12 |
| verify bs64 ctx64k q5 | 1 | 320 | 65536 | 129.8 | 83.6 | 1.55 |

## P0.3b: indexer share of SGLang V4.1 prefill steps (TP4, in-server spans)

Node crsuse2-m2m-259, 2026-10-07. Config = the TP4_GAP_1006.md lane, served from worktree `/sgl-workspace/sglang-fp4idx`
(main 16a23a672b plus env-gated spans) on aiter e7d2453f2.
- Load: `prefill_cost_probe.py`, max_new_tokens=1, chunk 16384.
- Script: `scripts/fp4idx_prefill_probe.sh`. Summary: `/shared_nfs/kk/results/DeepSeek-V4.1-Flash/fp4_index_port/p03b_259_1007_0337/summary.txt`.
- Values are medians over warm (no JIT) forwards. "score" is the FP4 logits call; "indexer" also covers q, top-k
  and candidates. All values are rank-0 GPU stream time.

| new tokens | max_seq | fwd ms | indexer ms (%) | score ms (%) | q_inputs ms | score host ms | ws_build us |
|---|---|---|---|---|---|---|---|
| 4096 | 4k | 75.0 | 3.10 (4.1) | 0.36 (0.5) | ~2.0 | 1.60 | ~370 |
| 4096 | 33k | 77.4 | 4.84 (6.3) | 1.52 (2.0) | ~2.0 | 1.69 | ~360 |
| 4096 | 131k | 77.4 | 10.16 (13.1) | 5.40 (7.0) | ~2.0 | 1.73 | ~390 |
| 16384 | 16k | 213.2 | 10.05 (4.7) | 1.40 (0.7) | ~6.9 | 1.75 | ~490 |
| 16384 | 33k | 218.3 | 13.88 (6.4) | 3.86 (1.8) | ~6.9 | 1.80 | ~510 |
| 16384 | 49k | 222.4 | 18.12 (8.1) | 6.54 (2.9) | ~6.9 | 2.18 | ~480 |
| 16384 | 66k | 225.8 | 21.84 (9.7) | 9.06 (4.0) | ~6.9 | 2.14 | ~500 |
| 16384 | 131k | 243.4 | 37.99 (15.6) | 20.42 (8.4) | ~6.9 | 3.73 | ~540 |

**JIT stalls (cold forwards):** the first forward at a new page-table width spent 0.19-5.4 s in score, e.g. fwd at
82k = 5.4 s and at 131k = 1.55 s. The first `ws_build` at a new shape spent 0.7-3.9 s. 15 new
`launch_pa_mqa_logits_fp4_prefill_*` entries appeared in `aiter/jit/flydsl_cache` during the run, even though the cache
already held 6201.

## P0.4: JIT compiles and first-call cost across shapes (1x MI355X, aiter 5b2f7d1d1)

Node crsuse2-m2m-259, 2026-10-07. Script: `scripts/fp4_scorer_jit_sweep.sh`. Log: `/shared_nfs/kk/results/DeepSeek-V4.1-Flash/fp4_index_port/p04_jit_sweep.log`.

Setup:
- Each mode runs in its own process: first with EMPTY FlyDSL/Triton caches (cold), then a new process on the same
  caches (warm disk, as after a server restart).
- Width sweep: rows=2048, keys 4k..256k. Rows sweep: keys 32k, rows 512..16k.
- The 4k row is the first call in each process, so it also carries one-time init.

| | A = SGLang prefill scorer | B = row-group |
|---|---|---|
| new FlyDSL kernel per new key width | yes (1 kernel per 4096-key bucket) | **yes (1 per width)** |
| first call at a new width, cold cache | ~160-167 ms | **~1180-1190 ms** |
| first call at a new width, warm disk cache | ~41-46 ms | ~73-75 ms |
| new kernel per new row count | no | no |
| workspace build (A only, Triton) per new row-count shape | cold 0.55-1.9 s, warm 2-5 ms | not needed |
| steady at 2048 rows, 128k keys | 1.32 ms | 0.85 ms |

## P3 sizing: DSpark target-verify scorer share of the decode step (TP4)

Node crsuse2-m2m-259, 2026-10-07. Verify has 6 rows per request (`speculative_num_draft_tokens`=6) and accept length 3.50.

How it was computed:
- **Step time:** clean TPOT x AL. c1/c4/c8 use ctx 64k from m255_tp4_d64k_bscurve (crsuse2-m2m-255, 10-06).
  c16/c32/c64 use ctx 32k from p03v_259_* on this node.
- **A (SGLang scorer):** the microbench at the same shapes (`scripts/fp4_verify_shapes_bench.py`), summed as
  3 r=2 layers + 5 r=1 layers.
- **Check:** it matches the in-server trace kernel `pa_mqa_logits_fp4_kernel_0`, which was c1/c4/c8 = 53/134/179 us
  per step (that includes ~2.7 us/call profiler inflation), against a prediction of 36/118/181.
- **Missing trace:** profiling c16 hung the server (NCCL BROADCAST watchdog timeout, the TP rank desync from
  TP4_GAP_1006.md item 5), so c16+ has no trace.

| conc | ctx | step ms | A us/step | A share | B us/step | saving us | saving % of step |
|---|---|---|---|---|---|---|---|
| 1 | 64k | 7.6 | 36 | 0.5% | 51 | -15 | -0.2% (B slower) |
| 4 | 64k | 10.0 | 118 | 1.2% | 70 | 48 | 0.5% |
| 8 | 64k | 11.2 | 181 | 1.6% | 111 | 70 | 0.6% |
| 16 | 32k | 15.0 | 176 | 1.2% | 112 | 64 | 0.4% |
| 32 | 32k | 20.5 | 449 | 2.2% | 190 | 259 | 1.3% |
| 64 | 32k | 25.6 | 586 | 2.3% | 364 | 222 | 0.9% |

Logs: `/shared_nfs/kk/results/DeepSeek-V4.1-Flash/fp4_index_port/p03v_verify_shapes.log` and `p03v_run*.out`;
`/shared_nfs/kk/results/DeepSeek-V4.1-Flash/profile_tp4/p03v_259_*`.
