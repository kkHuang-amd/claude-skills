# K2a port plan: ATOM V4.1 mono decode indexer scoring -> SGLang mono kernels

Owner node: mi355-4

Read-only study (2026-10-12). No code edits, no GPU runs. Sources:
- ATOM `origin/main` at 68506823, which contains #2479 (08735170). Paths below are under `atom/models/deepseek_v41/mono/` unless they are given in full.
- SGLang `/sgl-workspace/sglang/python/sglang/` (`srt/...` and `kernels/...`).
- aiter `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/mqa_logits/`.

Model facts used throughout:
- 32 index heads x 128 dims, top-k 512, candidate top 2048 blocks of 8 rows.
- Layers 2/8/14/20 own index K and run a compressor (`kv_source_layer_ids`). Layer 20 is the candidate source (ratio 1).
- Layers 24/28/32/36 are candidate consumers (ratio 1). They score layer 20's index plane (`dsv41_sparse.py:203-205`).
- Layers 2/8/14 are ratio 2 and score their full context.
- A DSpark K5 verify step has 6 rows per request.

## 1. What ATOM K2a does on an indexer layer (decode / verify)

Launch order per layer (`runner.py:396-434`):
1. **K1** (`attn_pre`) also runs the index projections (`kernels/index_query.py:1-27`, args at `runner.py:492-504`):
   - wq_b FP8 GEMV of q_lora, then bf16, then GPT-J RoPE on dims 64..127.
   - weights_proj GEMV of x.
   - Output: E2M1 q `iq [S,32,64]`, E8M0 scales `iqs [S,4,16,4]`, and f32 weights `iw [S,32] = rn(w/64)` (`stage_iquant_fp4`, `index_query.py:204-244`).
2. **Compressor**, run by the original module between the launches (`runner.py:411-412`). It writes the index-plane row of the step.
3. **K2a** (`kernels/attn_post.py`), with the indexer first in the launch (`attn_post.py:537-541`): `run_indexer`, then a barrier, then attention, wo_a, wo_b, all-reduce, FFN seam.
4. Only after the candidate producer (layer 20): `candidate_block_table` runs (one Triton kernel, `runner.py:424-434`, `model_ops/deepseek_v41/candidate_table.py:111-141`). It turns `icand` block ids into physical 8-row page ids plus a bound, for the REINDEX layers.

Step metadata (`step_meta.py:13-50`, one Triton launch a step) gives `kmeta [S,4]`: first window position, window count, block-table offset `req*stride`, and slot row base. The FP4 FULL-layer walk plan `iplan [256,4]` (lead, span, ch0, ch1 for each CTA) is built **on the host** by `index_plan.fill_walk_plan` (`index_plan.py:222-259`). The `WalkPlanner` step planner publishes it as a pinned CpuGpuBuffer (`walk_plan.py:25-57`). Per-layer arguments are listed at `runner.py:630-680`:
- `itab`: the request PAGE table, with `ishift = log2(index_blocks_per_page)`. On REINDEX layers it is the candidate table instead, with `ishift 0`.
- `ibound`: `step.visible[ratio]` on FULL layers, the candidate context on REINDEX layers.
- `ibatch`: batch ids.
- `ilift`: 1 on REINDEX layers. `iprod`: 1 on layer 20.

`run_indexer` stages (`kernels/index_score.py:370-412`). Every per-step cell decision is made in `index_plan.py` and evaluated **traced on device** from `ibound`:

| stage | what | inputs -> outputs | refs |
|---|---|---|---|
| iscore (FP4) | A CTA takes one piece (up to WALK_ROWS=6 adjacent rows of one request) and a run of 128-col chunks. The piece's q, scale and weight rows go to LDS. 8 waves walk 64-key steps, 2 steps of keys in flight. Per row, head tile and key tile it issues one 16x16x128 scaled FP4 MFMA. Then relu as `x+abs(x)` times w/2, the lane's head fma chain, and 2x permlane32 + 1x permlane16 swaps. One dword store of the logit per lane. The arithmetic is aiter rowgroup's, bit for bit. | iq/iqs/iw, plane+pscale through itab -> `ilog [S,bound_max]` f32 (device scope). The producer also writes `ibmax` 8-row block maxima (NaN ignored, capped at FLT_MAX, newest block +inf). Then ISF flags for each (token, chunk). | `index_score_fp4.py:133-185` (rows), `:216-241` (page/keys), `:254-302` (step), `:312-355` (block max), `:407-486` (task, plan) |
| ilist sel | Exact top-512 tree. Level 0: one 8192-col segment per CTA (polls its chunks' ISF). LDS radix select (4 x 8-bit passes, ties to lowest index, output ascending). Level l>0: merges `8192/k` lists from ISEL<l-1>. The final list goes to global. | `ilog` -> `isel [S,512]` (compressed row; on REINDEX lifted through `icand`), `irow [S,512]` (pool row via kmeta word 2 + block table), then IRDY[t] | `index_score.py:241-327`, `:330-367`; `topk.py:58-165`; tree `index_plan.py:95-160` |
| ilist blk (layer 20 only) | Same tree with k=2048 over `ibmax` | -> `icand [S,2048]` block ids ascending | `index_score.py:382-383` |

How attention consumes it: `await_selection` polls IRDY[t] (`kernels/attention.py:154-161`). `_key_row` then reads `sel=irow` for keys < `selected(bound)=min(bound,512)`, and window rows after that (`attention.py:164-177`). There is no separate key-table stage. A bound of 512 or less skips scoring: the selection is every column in order (`index_plan.py:57-62`). LDS goes in the union member `k2a_sel`: 8192 keys plus 3x320 histogram words (`attn_post.py:462-500`). The mailbox regions `isf`, `irdy`, `isel<l>` and `iblk<l>` are sized from `index_bound_max = roundup(max_model_len,128)` (`index_score.py:69-77`, `runner.py:132`). `ilog` is one shared `[S,bound_max]` f32 buffer (`runner.py:317-330`).

ATOM FP4 plane (`model_ops/deepseek_v41/index_plane.py:1-17`, `kernels/index_score_fp4.py:1-12`):
- 8-row pages. Values `[page][t%4][chunk c][t//4][16B]` (512 B). Scales `[page][c][t]` E8M0 (32 B).
- A lane reads keys `4l+nt` (one page per lane per step) and one scale dword.
- The PAGE table names `block_size/ratio/8` pages for each entry (`v41_pool_geometry.py:247-254`, `cache.py:68-91`).

## 2. What SGLang does today for the same work

Our hybrid (`SGLANG_MONO_DSV41_WHOLE=1` + `SGLANG_MONO_DSV41_INDEXER=1`, `mono/spec.py:125-150,235-267`):
1. K1 exports bf16 `xn` and `ql` (`mono/kernels/runner.py:198-200,316-358`).
2. `between()` runs SGLang's `forward_low_ratio_sources` (`srt/layers/attention/deepseek_v4_backend.py:2729-2786`).
3. A separate `kt_kernel` launch rebuilds the key table from the fresh top-k (`runner.py:359-376`, `layer.py:428-490`, `front.stage_kt` `front.py:192-227`).
4. K2 (`attention/back.py:903-`) does split, combine, wo_a and wo_b. The MoE launch follows.

The K2 key table uses `KT[t][k] = (comp_bt[req][lc//e]*e + lc%e) | 1<<31` for `k < ntopk=min((pos+1)//ratio,512)`, then window slots. It reads `core.sparse_raw_indices(ratio)` (compressed positions, -1 padded, unordered; `spec.py:337-340`).

Kernels per indexer layer inside `between()`, verify step:

| part | kernels (launches) | refs |
|---|---|---|
| compressor (2/8/14: c2, 20: c1) | `compressor.project_fused`/`wkv` GEMM; `c2_decode_norm_rope_store` / `c1_decode_norm_rope_store` (flash_c2/c1); `indexer.forward_wk` GEMM; `index_k_norm_rope_pack_store_split` | `deepseek_v4_backend.py:2928-3018`, `low_ratio_backend_hip.py:594-609` |
| index q (all 8) | `indexer.wq_b` GEMM; `rocm_router_gemv_split_k` (weights_proj); `index_q_rope_pack_weights_flydsl` (RoPE + tail fake-quant + FP4 pack + weight reduce) = "idx gemm/quant" + "index_q_pack 33" | `low_ratio_backend_hip.py:70-98`, `kernels/ops/attention/dsv4/fp4_indexer_hip.py:820-860` |
| logits (all 8) | `flydsl_pa_mqa_logits_fp4`, `next_n=1`. This is the **one-row-per-workgroup** kernel: each of the 6 verify rows reads the whole context. **Consumer layers 24-36 score the full ratio-1 context.** | `low_ratio_backend_hip.py:265-278`; `fp4_indexer_hip.py:411-433`; aiter `pa_mqa_logits_fp4.py:550-620` |
| top-k | `plan_topk_v2` + `topk_transform_paged_v2` (unordered, -1 padded) -> `sparse_page_indices`/`sparse_raw_indices` | `candidate_blocks_hip.py:19-37`, `low_ratio_backend_hip.py:304-311` |
| layer 20 candidates | `candidate_block_scores`, `_candidate_lengths_kernel`, plan + topk v2 over blocks | `candidate_blocks_hip.py:60-96,295-342` |
| 24-36 two-level | `gather_candidate_blocks` (compact 16384 cols), plan + topk v2, `_map_compact_selection_kernel` | `candidate_blocks_hip.py:345-434` |
| mono glue | `kt_kernel` (8 a step) | `runner.py:363-376` |
| per step (pre) | `prepare_fp4_decode_workspace` for each ratio: page-table pad + device schedule (graph safe) | `low_ratio_backend_hip.py:101-112`, `fp4_indexer_hip.py:211-242` |

SGLang FP4 index plane (`srt/mem_cache/deepseek_v4_memory_pool.py:566-583`, writer `fp4_indexer_hip.py:546-613`):
- 64-row pages, one buffer each for payload and scales.
- Payload `[page,1,4 chunk,64 slot,16B]`. Scales `[page,1,4 chunk,64]`, with slot `o` stored at `(o%16)*4 + o//16`.
- aiter's rowgroup kernel reads this layout as its "page 64" variant: lane key `16nt+l`, scale dword (`pa_mqa_logits_fp4_rowgroup.py:1-41`).
- The rowgroup kernel also has the page-8 variant, which is ATOM's layout.

## 3. Layout mismatches and bridges

| item | ATOM | SGLang | bridge | pick |
|---|---|---|---|---|
| FP4 index plane | 8-row pages; values `[t%4][c][t//4]`; scales `[c][t]` | 64-row pages, split buffers; values `[c][slot]`, scales transposed 16x4 | (a) **adapter in `_keys` only**: use virtual page `P = phys*8 + sub`. Value of key t at `phys*4096 + c*1024 + (8sub+t)*16`; the lane's 4 keys are contiguous 64 B. Scale bytes at `phys*256 + c*64 + 32(sub%2) + 16(l%2) + 4nt + sub//2`: one 16 B load + 4 byte extracts. (b) Switch FULL to the native page-64 mapping (`16nt+l`, one scale dword, as rowgroup does); this changes swaps, `key` and block maxima (~60 lines). (c) Change SGLang to page 8: pool, writer, prefill scorer, candidate kernels, PD/HiCache. Too wide. | (a) first (keeps ATOM arithmetic and every other stage unchanged); (b) later as an optimization |
| page table / `ishift` | request PAGE table, each entry `2^ishift` 8-row pages, row = batch id | `indexer_metadata.page_table`, one row per verify row (expanded), each entry one 64-row page | FULL: `itab = page_table`, `ishift = 3`, `ibatch = arange(rows)`. The walk-plan **owner** must be the request id (`row // 6` at verify), so a piece's 6 rows share key loads. | adapter (args only) |
| candidate table | `candidate_block_table` maps `icand` to physical 8-row page ids + bound | `CandidateBlocks.ids` unordered + compact mask | Vendor ATOM's `_candidate_table_kernel` with `UNITS=8`: entry = `page_table[row][b//8]*8 + b%8`, which is the same virtual page id as the adapter. Runner-owned buffers, graph safe. | port ATOM |
| selection output | `isel` (compressed row, ascending, valid prefix) + `irow` (pool row) | `sparse_raw_indices` / `sparse_page_indices [rows,512]` (stride 512) | `_final` writes `isel` into `sparse_raw_indices(ratio)` (consumer layers read it) and `irow` as `page_table[row][lc//ps]*ps + lc%ps` into `sparse_page_indices` (non-mono readers). ATOM's ascending order is valid for ours. | adapter (in `_final`) |
| key table | none (attention reads `irow`) | K1 `stage_kt` -> `kt`/`klen`; K2 reads `kt` | Stage 1: keep `kt_kernel`. Stage 2: `_final` also writes `kt[t][pos] = c_slot|1<<31` (or -1). The window part and `klen` come from K1 unchanged, since `ntopk` equals `selected(bound)` when `bound = (pos+1)//ratio`. K2 split polls IRDY[t] before reading `kt`. | adapter |
| query | iq E2M1 `[S,32,64]`, iqs `[S,4,16,4]`, iw **f32** | q_fp4 int8 `[T,32,64]`, q_scale `[T,1,4,16,4]` (same order), weights **bf16** (already /64) | q and scale are the same bytes. Weights: read bf16 and widen in `_stage_rows` (the w/64 scaling is exact, so the values are identical), or have the q-pack write f32. | adapter |
| compressed KV (attention) | bf16 pool rows (`kmeta`) | fp8_ds_mla / V41(_FP4) records via `comp_bt` | not touched: we keep our K2 attention and only the indexer moves | none |

## 4. CUDA/HIP-graph compatibility

- Device, traced from `ibound` (graph safe): everything in `index_plan` used by the kernel, i.e. chunk counts, segment/list counts, tree levels, REINDEX split (`reindex_split` uses only the CTA index), and `score_tasks` (FP8 path).
- Build constants: `S`, `index_bound_max`, `index_fp4`. Mailbox sizes are static for each build.
- Host computed: `fill_walk_plan` (numpy, step planner). ATOM can do this because its step builder has the visibility rows and batch ids on the host before publishing.
  - In SGLang a replayed graph cannot run host code. With the Task 9b no-sync change, `seq_lens_cpu` is not even ready in time.
  - **Bridge:** a 1-CTA Triton `walk_plan` kernel (~80 lines) launched once per step in the metadata pass, for each ratio (1, 2), from the device `compressed_seq_lens` and request ids, into a persistent `[256,4]` buffer. S <= 48, so pieces <= 48.
  - The width search spans about `[ceil(total/256), ceil(total/(256-pieces))]`, a few candidates, so it is a short loop.
  - The CPU property test checks it against `fill_walk_plan`.
- `candidate_block_table`: one Triton kernel. Graph safe once its outputs are persistent buffers (ATOM allocates fresh tensors).
- Mailbox flags (ISF, IRDY, lists) use our L1 epoch-tagged mailbox, so they are replay safe in the same way as K2's existing `srdy/crdy/ardy`.
- `ilog`/`ibmax` are persistent, sized `[48, bound_max]`.

## 5. Staged implementation plan

Env switches: `SGLANG_MONO_DSV41_K2A=0|1|2`, which requires `SGLANG_MONO_DSV41_INDEXER=1`.
- 1 = standalone indexer launch.
- 2 = fused into K2.

Also `SGLANG_MONO_DSV41_K2A_IQ=1` (index query in K1) and `SGLANG_MONO_DSV41_K2A_CHECK=1` (eager only: run SGLang's indexer too and compare).

| stage | content | lines (vendored + new) | numerics test |
|---|---|---|---|
| S0 measure (no tree change) | On the ATOM container, ATOM K2a timeline (`tl`, stage "index") at c1 for 8k/128k/512k. Gives the real cost of the in-launch indexer that we would add. | ~60 (script) | none |
| S1 standalone "K2i" launch (`K2A=1`) | Vendor `index_plan.py` (FP4 parts, ~260), `topk.py` (165), `index_score_fp4.py` (487, plus ~40 for the `_keys`/weights adapter), and the tree half of `index_score.py` (~250, no FP8 scorer) into `mono/kernels/indexer/`. Wrap them in a 256x512 persistent launch with its own LDS, scratch and mailbox. Add the device walk-plan kernel and the vendored candidate-table kernel. `between()` keeps the compressor and SGLang's q path (wq_b, gemv, q-pack). K2i replaces logits, top-k and candidates, and writes `sparse_raw_indices`/`sparse_page_indices`. `kt_kernel` and K2 are unchanged. | ~1,200 vendored + ~450 new | (a) **logits**: K2i `ilog` vs `flydsl_pa_mqa_logits_fp4` on captured real q/K at 8k/128k/512k: max rel err (expect <= 1e-5; not bitwise, because the one-row kernel's sum order differs) and against an fp64 reference. (b) **top-k given equal logits**: feed SGLang's logits into the tree (test hook prefilling `ilog`). Sets must equal `topk_transform_paged_v2`'s except exact ties. (c) **layer**: per-row Jaccard of `sparse_raw_indices` vs SGLang path (target median 1.0, p1 >= 0.99); layer 20 `icand` vs `CandidateBlocks.ids` as sets; K2 `out` rel err <= bf16 noise. (d) GSM8K 1319 >= 0.895, plus a 128k needle check. (e) `walk_plan` kernel == `fill_walk_plan` (CPU property test). |
| S2 fuse into K2 (`K2A=2`) | `run_indexer` before `run_back`; a `sel` LDS member in `back_lds` union; indexer regions in `back_scratch`; ~22 extra kernel args on `ix=1` builds only; `_final` also writes `kt`; `stage_split` polls IRDY[t]. Drops the K2i and `kt_kernel` launches. | ~250 new | K2 `out` and `sparse_raw_indices` **bitwise** equal to S1 (same scorer and tree, same kt). Run at S=6/12/48. |
| S3 index query in K1 (`K2A_IQ=1`) | Vendor `index_query.py` (244) into front.py as stages iw/iq/iquant_fp4 (~120 wiring); pack `indexer.wq_b` + `weights_proj` in spec.py (~40). Drops wq_b GEMM, split-K gemv and q-pack from `between()`. | ~250 vendored + ~160 new | q_fp4/q_scale bytes and weights vs `index_q_rope_pack_weights_flydsl` on real rows. Expect bitwise; check SGLang's tail fake-quant (`fp4_rope_fake_quant.py`) against ATOM's single `quantize_fp4`. Count any mismatching bytes, then repeat S1 (c)/(d). |
| S4 optional | Native page-64 lane mapping for FULL layers (option (b) in section 3) | ~60 | logits bitwise equal to S1 |

Total: about **1.45k vendored + 0.9k new**, about 2.3k lines, plus ~400 lines of tests. S1 alone already captures most of the context-dependent saving.

## 6. Expected per-step saving and risks

Today at c1, 128k: about 500-550 us/step over the 8 layers (Task 9 table).

| | removed (us/step) | added (estimate, to be confirmed by S0) |
|---|---|---|
| S1 | logits 81 + top-k 123 + candidate/compact 40 + decode workspaces ~10-20 = **~255-265** | K2i: FULL ratio-2 (3 layers) ~12-20 each; layer 20 ~20-30; REINDEX (4 layers) ~8-12 each (16384 cols, context independent); launch ~3 each -> **~110-170** |
| S2 | kt + K2i launches/tails 8 x ~6-8 = **~50-60** | ~0 (the indexer overlaps nothing, but the launch boundary goes away) |
| S3 | index-q GEMM + gemv (~70-90 of the 140; the compressor GEMMs stay) + q-pack 33 = **~105-125** | K1 index-query stages ~3-5 per layer, **~30-40** |

Net at 128k:
- S1: ~0.10-0.15 ms/step.
- S1-S3: **~0.25-0.33 ms/step**. At AL 3.51 that is about 0.07-0.09 ms/token, ~4-5% of ITL. This matches the Task 9 estimate of ~0.3 ms.

Context growth:
- Today we grow ~0.9 us/step per 1k context (8k->128k: +107 us).
- Most of that comes from the 4 REINDEX layers scoring the full ratio-1 context, from layer 20, and from each of the 6 verify rows re-reading every key.
- With K2a, REINDEX is fixed at 16384 columns, and FULL layers read each key once per 6-row piece. Only the scan of layers 2/8/14/20 and the top-k tree depth grow.
- Expected at 512k: ~0.5-0.7 ms/step saved (~0.15-0.2 ms/token). That should close the 1.048x gap to ATOM at >=512k.
- This is extrapolated, not measured: S0 and S1 measure it.

Risks:
1. **Selection not bit-identical to SGLang's.**
   - The logits come from rowgroup arithmetic, not the one-row kernel's sum order.
   - Ties: ATOM goes to the lowest index; topk v2's rule is unordered.
   - Block-max NaN handling differs.
   - Accuracy must be shown by tests (b)-(d), not assumed.
2. **K2 resource pressure on `ix=1` builds.**
   - About 22 more kernel args; ATOM already notes SGPR limits and reads only `STAGE_ARGS`.
   - The 36 KB `sel` LDS member and the larger scratch could slow attention on these 8 layers.
   - S1 isolates this: if S2 regresses, stop at S1.
3. **Device walk plan vs ATOM's host plan.**
   - The device kernel must reproduce the host plan exactly, including the request-id owners for verify rows.
   - Load imbalance shows up as a slow last CTA.
   - All 256 CTAs must be resident (persistent mailbox waits).
4. **Memory.** `ilog [48, 1M]` f32 is 192 MB/rank, plus `ibmax` 24 MB. Mailbox ISF needs 48 x 8192 flags at 1M.
5. **S3 q-path mismatch.** SGLang's RoPE-tail fake-quant and its bf16 weight rounding may not match ATOM's chain byte for byte.
6. **The win is bounded.** After 9b nosync + task 8 + 9c fold, the 10-12 py-spy trace shows no host bubble left at c1 (host ~3.9 ms/step ahead): the remaining gap is GPU time, and our c1 P90 tail is long-ISL, so the indexer is now the main lever. Still measure (S0, or S1 directly if no ATOM container on this node) before committing to S2/S3.
