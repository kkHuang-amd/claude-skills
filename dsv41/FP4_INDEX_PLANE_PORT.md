# FP4 index plane port (ATOM #2479 -> SGLang)

Owner node: crsuse2-m2m-259 (only the owner edits CONTINUE HERE; other nodes append to Progress log / own a task row)
Created: 2026-10-07

## CONTINUE HERE

**Status:** P0.1 done. The kernel is aiter #6145 (`5b2f7d1d1`, 2026-10-06), and it serves page 8 AND page 64.
Page 64 reads the SAME preshuffle layout that SGLang's pool already writes, so P1 needs no layout change.
P0.2 done on crsuse2-m2m-259 (results/fp4_index_scorer.md).
**P0.3b done** (results/fp4_index_scorer.md, "P0.3b" table; warm medians, TP4 V4.1 prefill, chunk 16k).
- **Score share of a prefill forward:** 0.5-2% at <=33k context, 4% at 66k, 7-8.4% at 131k.
  With B's 1.3-1.4x, steady-state P1 saves <=~2.3% at 131k (about 5.6 of 243 ms), ~0.5% at 33k, and nothing at short
  context, where B is slower. **That misses the 5% gate.**
- **Host overhead is NOT exposed.** The score spans' GPU time is at or below their host time (16k@16k: 1.4 vs 1.75 ms),
  so the stream had a backlog. ws_build is 0.4-0.5 ms per forward (0.2%). The A-vs-B host saving is ~0.
- **New finding: JIT stalls.** SGLang's prefill scorer compiles one FlyDSL kernel per page-table width bucket
  (`LOW_RATIO_PAGE_TABLE_BUCKET`=64 pages = 4096 keys). The first request reaching a new width stalls 0.2-5.4 s
  inside its forward, and a first ws_build adds another 0.7-3.9 s. The run made 15 new compiles.
  - The stall is persisted in `aiter/jit/flydsl_cache`, so each width pays it once per node or container. Every fresh
    deployment still pays it on live requests, as TTFT spikes.
  - The ATOM test says one row-group compile serves every length. If that holds, this is P1's main value, not the 1.3x.
- **Other indexer costs, now tracked in INDEXER_COST_1007.md:** q_inputs is ~6.9 ms per 16k tokens (3.2%), and top-k/candidates is
  ~2-10 ms. Neither is part of this port.

**P0.4 done** (results "P0.4" table). The JIT hypothesis is REFUTED.
- **Row-group B also compiles one kernel per key width,** and it is slower to compile:
  - cold: ~1.18 s per width vs A's ~0.16 s;
  - warm disk cache, per new process: ~74 ms vs ~42 ms.
- Neither kernel recompiles per row count. ATOM's "one compile serves every band length" means row bands, not key
  width. ATOM probably fixes the width per server [inferred].
- **Server stalls are bigger than the microbench and not explained.** P0.3b saw 0.2-5.4 s stalls in the server, against
  ~0.16 s per width here. Untested candidates: 4 TP ranks compiling at once (CPU contention), and several widths/variants
  per forward (r=2 and r=1, row chunks). A's Triton workspace build adds 0.55-1.9 s cold per new row-count shape.
**Decision proposal: P1 is PARKED.**
- The only remaining gain is <= ~2.3% of prefill at 131k context; it is slower at short context, and it changes nothing
  about JIT.
- The JIT stall moves to INDEXER_COST_1007.md I2: make key width a runtime arg in aiter's kernel, or prewarm, or use
  coarser buckets.
- P3 (DSpark verify, 1.4-1.75x at bs>=32 / ctx>=32k) is the one open port candidate. It needs a decode-side
  in-server measurement first.
**P3 sized (10-07):** verify scorer = 0.5-2.3% of the decode step; B would save 0.4-1.3% (c4-c64, best c32), with
-0.2% at c1 (results "P3 sizing"). Verify has 6 rows per request. This is modest, so P3 is low priority.
- Implementing it needs request-level `query_start_loc` + the request page table in the decode/verify metadata.
  The plan is shape-only, so it is capture-friendly.
- Dispatch rule: B only when bs >= 4.
**Overall verdict for the row-group port:** small gains everywhere measured.
- prefill: <= 2.3% at 131k;
- verify: <= 1.3% at c32;
- JIT: no help.
- page-8 / candidate-only scoring (P2): measured in INDEXER_COST_1007.md I4. The consumer layers 24..36 already cost
  only ~0.2 ms each at 131k, so P2 has little value.
**Test env (node-local, crsuse2-m2m-259):**
- aiter worktree `/sgl-workspace/aiter-6145` @5b2f7d1d1; `3rdparty/composable_kernel` is a symlink to `/sgl-workspace/aiter`'s,
  which is the same CK commit.
- ATOM worktree `/tmp/atom-2479` @08735170.
- Run with `PYTHONPATH=/sgl-workspace/aiter-6145[:/tmp/atom-2479]`.
- **P0.3b instrumented SGLang** (`/sgl-workspace/sglang-fp4idx`, detached at 16a23a672b), to rebuild on another node:
  `git -C /sgl-workspace/sglang worktree add --detach /sgl-workspace/sglang-fp4idx 16a23a672b && git -C /sgl-workspace/sglang-fp4idx apply /workspace/claude-skills/dsv41/patches/sglang_local_fp4idx_timing_0001.patch`.
  The patch adds the env-gated spans (`SGLANG_DSV41_IDX_TIMING=<dir>`) and is local only. The editable `/sgl-workspace/aiter` (e7d2453f2) stays
  untouched.
**Files (SGLang, paths under python/sglang/):**
- `srt/mem_cache/deepseek_v4_memory_pool.py`: `DeepSeekV4IndexerPool` (:509)
- `srt/layers/attention/dsv4/low_ratio_backend_hip.py`: `low_ratio_index_topk_hip_{decode,extend}` (:227/:327)
- `kernels/ops/attention/dsv4/fp4_indexer_hip.py`: `aiter_fp4_paged_mqa_logits` (:310)
- `kernels/ops/attention/dsv4/fp4_indexer_schedule_hip.py`: `build_prefill_schedule` (:128)
- `deepseek_v4_backend_hip_radix.py`: `init_forward_metadata_indexer` (:996)
**Repro (ATOM reference test, once aiter is new enough):**
```bash
cd /workspace/ATOM && git -c gc.auto=0 -c maintenance.auto=false worktree add /tmp/atom-2479 08735170 \
  && cd /tmp/atom-2479 && python -m pytest -q tests/models/deepseek_v41/test_index_fp4.py 2>&1 | tail -5
```
**Pass criteria:**
- Top-k is the same as the current SGLang path, except at near-ties (tol 1e-4*max, same rule as the ATOM test).
- GSM8K 5-shot 1319, TP4/EP4: >= ~0.90. The PR #39857 bar is 90.45 with DSpark off and 90.22 with it on.
- No regression in AgentX or bench_serving TTFT/TPOT. The expected win is prefill indexer time.

Repo rule: never let git auto-gc run in /workspace/ATOM. The NFS checkout breaks it (`.nfs…` bad sha1). Always pass
`git -c gc.auto=0 -c maintenance.auto=false`.

## Assessment (2026-10-07, crsuse2-m2m-259; SGLang main 16a23a672b, ATOM 08735170)

**The main point:** SGLang already has an FP4 index plane on HIP.
- The low-ratio pools (ratio 1/2) are always FP4 (`force_fp4=True`). The c4 pool is FP4 when
  `--enable-deepseek-v4-fp4-indexer` is set.
- Payload and scale are split (68 B/token at dim 128, E2M1 + ue8m0/32), and aiter FlyDSL `flydsl_pa_mqa_logits_fp4{,_prefill}`
  does the scoring.

So the 68 B vs 132 B memory gain that ATOM gets from this PR is *already in SGLang*. Before #2479, ATOM V4.1 was FP8-only.
What is left to port is the plane's **geometry and scoring structure**, not the FP4 format.

| Aspect | SGLang today | ATOM #2479 | Port value |
|---|---|---|---|
| Element format | E2M1 + E8M0/32, split planes | same | none |
| Page | 64 slots (`_KV_BLOCK_SIZE=64`) | 8 rows = 1 candidate block, MFMA-lane-interleaved (4 keys per dword) | unknown, P0.3 |
| Writer | JIT HIP `index_k_norm_rope_pack_store_split` (k_norm+RoPE+pack, RNE) | Triton norm+RoPE+aiter `_mxfp4_quant_op` (SCALING_MODE=1) | none (rounding mode differs!) |
| Q quant | Triton `index_q_rope_pack_weights_flydsl` (RoPE+pack+head-weight reduce, one launch) | aiter `rope_rotate_activation` | none (SGLang fuses more) |
| Prefill rows | one row per token, page table expanded per row; `row_to_batch`=arange, `local_starts`=0 | request-level ragged rows off the request PAGE table (FULL layers); one compile serves every length | **main candidate** |
| Decode rows | 1 row/req, schedule capture-safe | ragged plan from shapes only (graph stable) | low |
| Candidate table | `[rows,max_seq_len]` fp32 rectangle + Triton candidate kernels | candidates translated through PAGE table + batch_ids, no tile table | coupled to page-8 |
| Flag | `--enable-deepseek-v4-fp4-indexer` (c4 only) | `--index_cache_dtype {bf16,fp8,fp4}` | only needed if page-8 is optional |

**Numerics contracts to keep apart:**
- The c4 FP4 path uses Hadamard (aiter `rope_rotate_activation`). The V4.1 low-ratio path does not.
- SGLang's index writer rounds with RNE; ATOM uses aiter scale mode 1.

Any change must be A/B'd on top-k selection and GSM8K.

**Obstacles:**
- Page 64 is baked in at several places:
  - FlyDSL defaults and `_KV_BLOCK_SIZE`
  - the JIT writer's `page_size` template
  - `LOW_RATIO_PAGE_TABLE_BUCKET` / `expand_index_page_table`
  - the PD/HiCache rule that index pages tile a full KV page
- Page-8 makes the per-row page table 8x longer unless scoring goes ragged first.
- The prefill schedule is refreshed outside CUDA graph capture.
- Prefill CP and the torch oracle are unsupported on HIP.

## Plan

Phases are gated: do not start a phase until the gate before it says the win is real. Per the
perf-bottleneck-attribution skill, the gate needs a measured number, not an inference.

| ID | Task | Gate / output | Owner | Status |
|---|---|---|---|---|
| P0.1 | Find an aiter commit with `make_fp4_mqa_plan` + `pa_mqa_logits_fp4_rowgroup`; note whether it is compatible with sglang main's other aiter ops | aiter #6145 `5b2f7d1d1`, see "aiter #6145" below | crsuse2-m2m-259 | done 10-07 |
| P0.2 | Run ATOM `test_index_fp4.py` on gfx950 with that aiter (+ aiter op test, + page-64 ragged check) | ATOM 11/11 pass; aiter kv64/kv8 PASS; page-64 rowgroup err ≤7.6e-6 | crsuse2-m2m-259 | done 10-07 |
| P0.3 | (done 10-07 crsuse2-m2m-259: microbench, see results) Microbench at real shapes (64 heads? check model, dim 128, ISL 4k/8k/32k prefill; decode bs 1..64 ctx 4k-128k): SGLang per-token page-64 vs ATOM ragged page-8. Also time the same per layer in an SGLang prefill trace, to see if the indexer matters at all | us/layer table in results/ | | todo |
| P0.3b | Indexer share in real prefill (TP4 server, spans) | score 0.5-8.4% of fwd, host not exposed, JIT stalls 0.2-5.4 s per new width | crsuse2-m2m-259 | done 10-07 |
| P0.4 | Row-group vs current: compiles and first-call cost across widths | B also compiles per width (1.18 s cold vs A 0.16 s); P1 parked | crsuse2-m2m-259 | done 10-07 |
| P1 | Request-level ragged prefill on the EXISTING page-64 plane. The local aiter prefill kernel already accepts `row_to_batch`/`local_starts`; SGLang fills them with identity. Change `build_prefill_schedule` + `low_ratio_index_topk_hip_extend` to feed request rows and the request page table. No pool/layout change | top-k identical, prefill indexer us down, GSM8K | | todo |
| P2 | Only if P0.3 shows a page-8 layout gain over P1: page-8 interleaved plane behind a new arg (e.g. `--dsv41-index-page-size 8`). Needs the pool, writer template, candidate kernels, expand_index_page_table, PD/HiCache tiling and pool_configurator sizing | GSM8K + AgentX | | blocked on P0.3 |
| P3 | Target-verify via ragged plan (graph-stable shapes); B only when rows/req > 1 and bs >= 4 | sized 10-07: saves 0.4-1.3% of the decode step at c4-c64 (best c32, 1.3%); B is slower at c1 | crsuse2-m2m-259 | sized; low priority |
| P4 | Upstream PR: title `[AMD][V4.1][k/N] ...`, format per NEW_WORKSPACE_PROMPT | PR link | | todo |

Out of scope: the mono-only FP4 parts (`mono/kernels/index_score_fp4.py`, `index_query.stage_iquant_fp4`,
`mono/index_plan.py`) belong with a mono-decode port, not this one.

## aiter #6145 (`5b2f7d1d1`), P0.1 findings

**The new API** (in `aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_rowgroup.py`, exported from `aiter.ops.flydsl`):
- `make_fp4_mqa_plan(num_seqs, max_qlen, num_rows, heads, page_size, max_seq_len, pages_per_block=1, ...)` returns an
  `Fp4MqaPlan`.
  - The plan depends only on the batch shape, so a captured CUDA graph keeps it.
  - Then call `flydsl_pa_mqa_logits_fp4_rowgroup(plan, q_fp4, q_scale, kv_cache, kv_scale, block_tables, weights,
    query_start_loc, row_ends, *, weight_scale, out)`.
- One wave scores `rows_per_wave` query rows off each key load, so the rows of a tile read each key once.
  The old `pa_mqa_logits_fp4` uses one workgroup per row, which makes every speculative/prefill row read the whole
  context.
- `SUPPORTED_PAGE_SIZES = (8, 64)`.
  - **Page 64** is the existing `pa_mqa_logits_fp4` preshuffle, which is what SGLang's split FP4 pool already writes.
  - **Page 8** is the V4.1 lane-major layout: `kv_cache [pages, K_TILES, 4, 4, P/4, 16]`, `kv_scale [pages, K_TILES, 4, P]`.

**Compatibility with SGLang main:**
- **The old entry point:** `flydsl_pa_mqa_logits_fp4` keeps its signature; `context_lens` is now Optional.
- **`rope_rotate_activation`:** only gains `round_rope=False`. Both changes are backward compatible.
- **FlyDSL pin:** unchanged, `flydsl==0.3.4.1`, the same as installed.

**Cost of bumping:**
- `5b2f7d1d1` is 132 commits ahead of the local `e7d2453f2`.
- That range also has #5761, which unified the gfx950/gfx1250 MXFP4 paged MQA-logits, plus #5600 and #4221 in the same area.

**Recommended approach:**
- P0 tests run in a separate worktree. An isolated JIT dir is needed because the C++ `dsv4_rotate_quant` changed.
- A full aiter bump is decided at P1, with a GSM8K regression run.

**Local state on crsuse2-m2m-259:** `/sgl-workspace/aiter` is at detached `e7d2453f2`, editable install, and dirty:
- the FMoE CSVs
- `pa_mqa_logits_fp4_prefill.py` (`lru_cache` → `cache` only)
- `pa_decode_sparse.py` (staged)
- `torch_utils.py`
- an mla `.co`

`origin/main` was fetched 2026-10-07; the worktree did not change.

**Plan impact:**
- P1 can call the rowgroup kernel with `page_size=64` on the existing SGLang plane, giving request-level ragged rows.
  The current per-token `flydsl_pa_mqa_logits_fp4_prefill` would be replaced.
- P2 (page 8) uses the same kernel, so it needs no new aiter work. Only SGLang-side layout changes remain.

## ATOM reference map (commit 08735170)

- **Layout:** `atom/model_ops/deepseek_v41/index_plane.py` (39 lines);
  `atom/model_ops/attentions/pool_layout/v41_pool_geometry.py` (`FP4_INDEX_PAGE_ROWS=8` :62, `index_planes` :157, `paged_extents` :267).
- **Cache:** `atom/model_ops/attentions/deepseek_v41/cache.py` (`_index_plane`, `index_units`, `write_index_fp4` :494).
- **Writer:** `atom/model_ops/deepseek_v41/index_write.py` (`_write_index_fp4_kernel` :151, `write_index_rows_fp4` :216).
- **Scoring:** `atom/model_ops/deepseek_v41/paged_scoring.py`
  - `quantize_query_fp4` :59
  - `score_topk_quantized` :149
  - `_band_logits` :268
- **Ragged rows:** `atom/model_ops/fp4_mqa_ragged_metadata.py`, `score_workspace.py`,
  `candidate_table.py` (PAGE-table translation), and `models/deepseek_v41/attention.py` (`_ragged` :162).
- **Tests:** `tests/models/deepseek_v41/test_index_fp4.py` checks byte-exact writer output, top-k vs the dequant reference, ragged ==
  per-row, and a single compile.
- **Logit:** `sum_h w_h * relu(q_h.k) * weights_scale`, where `weights_scale = head_dim^-0.5 * heads^-0.5`.
- **ATOM flag:** V4.1 still defaults to fp8. CI configs (DSpark TP4 accuracy, nightly agentic) switched to fp4.
  The PR has no FP4-specific end-to-end accuracy or perf numbers.

## Progress log (append-only; every entry: date, node, what, link to result)

- 2026-10-07 crsuse2-m2m-259: assessment above. Local aiter e7d2453f2 lacks `make_fp4_mqa_plan` / rowgroup kernel.
- 2026-10-07 crsuse2-m2m-259: P2 has little value. The candidate consumer layers are already cheap (INDEXER_COST I4).
- 2026-10-07 crsuse2-m2m-259: P3 sized.
  - **Method:** existing c1/c4/c8 traces (crsuse2-m2m-255) plus clean TPOT at c16/c32/c64 on this node, with the
    microbench at verify shapes (6 rows/request), which is validated against the trace kernel times.
  - **Result:** B saves 0.4-1.3% of the decode step; c1 is slower.
  - **Server hang:** profiling at c16 hung the server (NCCL watchdog).
- 2026-10-07 crsuse2-m2m-259: P0.4 done.
  - **JIT:** row-group also compiles per key width, and is slower to compile, so JIT is not a reason to port.
  - **P1:** proposed parked. Its only steady-state gain is <= ~2.3% at 131k.
  - **JIT stall follow-up:** moved to INDEXER_COST_1007.md I2.
- 2026-10-07 crsuse2-m2m-259: P0.3b done.
  - **Setup:** worktree `/sgl-workspace/sglang-fp4idx` (env `SGLANG_DSV41_IDX_TIMING`, spans in low_ratio_backend_hip.py +
    deepseek_v4.py; local only, never upstream). The run was `scripts/fp4idx_prefill_probe.sh`, log dir
    `p03b_259_1007_0337`.
  - **Steady state:** the score share is too small for the 5% gate.
  - **JIT:** found per-width JIT stalls of 0.2-5.4 s, which set up P0.4.
- 2026-10-07 crsuse2-m2m-259: P0.3 microbench done (V4.1 heads 32 / dim 128; index layers 2,8,14 r=2 and 20..36 r=1).
  - **Bug in the first run:** it used eager timing, and A showed a constant ~125 us floor. That floor is host overhead, so the
    first run's "4-14x" was an artifact. It was redone with CUDA-graph replay.
  - **Graph results:** prefill at long context, B 1.2-1.4x; short context and decode, A faster; big verify, B 1.6-1.75x.
  - **OOM in the 16k x 131072 rows case:** rows x width = 2^31, the int32 limit. SGLang chunks rows by its logits budget anyway,
    so the case was changed to 4k on 124k.
  - **Open:** P0.3b, a prefill trace.
  - **Cold-cache rerun** (#6145 method), same conclusions:
    - long prefill: B 1.28-1.39x;
    - short prefill: A faster;
    - 1-row decode: about equal (bs64 ctx100k 1.1x);
    - verify q5 at bs>=32 / ctx>=32k: B 1.4-1.6x.
  - **PR scope:** neither #6145 nor ATOM #2479 says prefill-only. ATOM moved V4/V3.2 decode to the same ragged kernel,
    and #6145 claims decode gains vs aiter's default schedule.
- 2026-10-07 crsuse2-m2m-259: P0.2 done.
  - **ATOM** `tests/models/deepseek_v41/test_index_fp4.py`: 11 passed (45.7 s).
  - **aiter** `op_tests/test_flydsl_pa_mqa_logits_fp4.py`: PASS at `--kv_block_size 64` and 8. It is a script, not pytest; under
    pytest you get "fixture 'batch' not found".
  - **Page-64 rowgroup gap:** neither test covers the ragged rowgroup kernel at page 64, the P1 case, so it got its own check,
    `scripts/fp4_rowgroup_page64_check.py`. All 8 configs pass (err ≤ 7.6e-6).
  - **Speed vs the old per-row kernel:**
    - qlen 128/256/512 (prefill-like): 1.7x / 2.4x / 3.7x faster.
    - qlen 1: equal.
    - qlen 4/6 (spec verify): 10-12% SLOWER, so keep decode/verify on the old kernel unless P0.3 says otherwise.
  - **Caveat:** "old" is the decode kernel with next_n, NOT SGLang's prefill kernel. That comparison is P0.3.
- 2026-10-07 crsuse2-m2m-259: P0.1 done. The kernel is aiter #6145 `5b2f7d1d1` (page 8 + 64). It is API-compatible, and the flydsl pin is unchanged.
