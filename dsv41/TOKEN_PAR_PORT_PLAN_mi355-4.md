# Task 27b: token-parallel mono layers (port of ATOM #2545) -- plan

Owner node: mi355-4. Parent: MONO_KERNEL_mi355-4.md (task 27, section "Task 27"). Status 2026-10-14: step 1 done.

## CONTINUE HERE

**Status (mi355-4, 10-14 11:50 +08):** steps 1-3 coded in the worktree /sgl-workspace/sglang_v41mono (uncommitted, on top
of perf/v41-mono-fw 53d0945468 + S3 / task 22 / 27a = snap_t27a_1014). Frozen: /sgl-workspace/snap_t27b_s2_1014 (steps 1-2),
/sgl-workspace/snap_t27b_s3_1014 (steps 1-3). Non-tok_par ISA isa_task27b_s3/ == isa_task27a/ (91 kernels, diff -r empty);
tok_par K2 builds isa_task27b_s3tok/ compile, no spill (VGPR 240-254); CPU tests 44 pass. GPU bitwise checks QUEUED:
scripts/t27b_check_1014.sh (s2 then s3, each TP4 then TP2; waits for the 27a chain PGID 1364030), PGID 1396879, log
/shared_nfs/kk/t27b_check_1014.log; stop: `kill -- -1396879` + GPU 4-7 sglang.
**Next:** read the check log (pass: 0 "MISMATCH" lines per step / TP, per-200 summaries on every rank, GSM8K sane, no
Traceback; a mismatch names tensor, bytes, rows; the first step showing it is the culprit). Then step 4 (K1 seam + wqkv_a
+ qkv on own tokens, QG / QGS / KG, wq_b from QG; drop OG: layer contract = own rows; host final gather + chain breaks).
Each step ends with the ISA dump (non-tok_par identical) + CPU tests; tok_par builds: `mono_isa_dump.py OUT --tok-par-only`.
**ATOM reference:** /tmp/mono_src/ATOM @ 5e8d9131 (`git show 5e8d9131:<path>`), files
atom/models/deepseek_v41/mono/kernels/{token_par,peer_layout,attn_pre,attn_post,layer_post,moe}.py and
mono/runner.py (`_tok_par`, `_gather_rows`).
**Pass criteria:** tok_par outputs bitwise equal to non-tok_par (same rank order; ATOM: "bits unchanged") on real rows
at S = 12 / 24 / 48, TP2 + TP4; GSM8K 1319 >= 0.895; TPOT c4 / c8 lower than snap_t27a_1014; no hang in 1000 steps.

## What ATOM does (S >= 12, S % TP == 0; rank r owns tokens r n .. r n + n - 1, n = S / TP)

- FFN end (MoE `down_combine` / `stage_reduce`): a token's down partial goes to its owner only; `reduce` sums only the
  owner's tokens (`own_view(c, ("out",))`) -> the layer output holds this rank's rows only. AR -> reduce-scatter.
- K1 (`attn_pre`): slice .. qkv on own tokens (`own_view(c, K1_OWN)`: S = n, per-token args shifted by `t0` rows,
  `TOKEN_ROW_BYTES`). wqkv_a runs on n tokens (fewer MFMA token tiles: S = 48 TP4 -> 1 tile instead of 3). stage_qkv
  pushes the q latent's MXFP8 words / codes to every rank's QG / QGS (peer), and the quantized window row to every
  other rank's KG; every rank then writes all tokens' window rows into its own cache (own from registers, others
  from KG, `ring_row_from`). wq_b reads QG / QGS instead of the local QX8 (all S tokens). Indexer layers: the normed
  row to every other rank's NBG (`normed_row_from` for the compressor) and the head weights to IWG.
- K2 (`attn_post`): attention on all S tokens (heads split as before); wo_b pushes a token's partial to its owner
  only; FFN-seam slice / gate / norm on own tokens (`own_view(c, K2A_OWN)`); the norm publishes its row locally and
  pushes it to every other rank's XG (`publish_x_tok_par`); the MoE starts with `gather_rows` (others' rows from XG
  into `normed`, then XRDY); the MoE runs on all S tokens (experts / intermediate TP-split as before).
- Runner: `_tok_par(s)` = not check and not split_k2 and s >= tok_par_from (12) and s % tp == 0; off when the custom
  all-gather is unavailable. After the last layer `_gather_rows` all-gathers hidden + the drafter's taps (one custom
  AG). The DSpark draft runner never runs token parallel.
- One peer layout (`peer_layout.py`): attn, ffn, xg, qg, qgs, kg, nbg, iwg. A poll batch may mix scratch and peer
  regions (ATOM sync.py change; 27a ported only ops.py first_rounds / batched_rounds -- port the poll change in step 1).

## Mapping onto ours

| ATOM | ours | note |
|---|---|---|
| peer_layout.py | new `deepseek_v41/peer_layout.py`; today attn at 0 (`seam.attn_peer_bytes`), ffn at `moe.ffn_offset` | runner allocates `2 * peer_half_bytes(tp)` (two parities): keep parity halves, grow each half |
| token_par.py | new `deepseek_v41/token_par.py` with OUR per-token arg names / row bytes | K1 args: res_in, pend, post_in, comb_in, pre_in, res_out, post_out, comb_out, pre_out, pos, slot, q? (q is head-split: NOT per-token-owned), xn_out, ql_out; K2 args likewise + normed |
| attn_pre seam (slice, norm, gate) | stages/seam.py (+ layer.py K1 build) | seam code is ATOM's: own_view applies directly |
| attn_pre wqkv_a / qkv | attention/front.py (vLLM's) stage_wqkv / stage_qkv | ours: qkvp partials over 2 K halves, `stage_qkv` writes QX8 / QRDY and the V41 record (`_store_v41_record`) at the token's slot. Needs: run on n tokens; push QX8 to QG of every rank (+ flag); push the record bytes (or the bf16 kv row) to KG; a new stage writing others' records into the local SWA cache. Our cache is SGLang's paged V41 fp8 record (512 B data + 16 B scales), replicated per rank |
| attn_pre wq_b on QG | front.py wq_b loop reads `c["qx8"]` / `qx8s` + `qrdy` | point qx8 at QG (peer, system scope) and poll its flags; LDS load path unchanged |
| attn_post wo_b push to owner | attention/back.py wo_b + seam.stage_push / stage_reduce (`attn` region) | push only to owner, reduce over own tokens |
| layer_post publish_x_tok_par / gather_x | layer.py K2 build (store_normed / publish), stages/moe.py run_moe | add XG push + `gather_rows` before xq |
| moe down_combine / reduce | stages/moe.py down_combine / stage_reduce | the ATOM diff applies almost verbatim (seen in 27a review) |
| index_query iw -> IWG, NBG | attention/index_query.py (S3) + seam xn export | iw on own tokens' rows needs every token's xn: either gather xn (NBG) first or keep iw on all tokens from NBG |
| runner `_gather_rows` | kernels runner + host (deepseek_v4_hip.py layer loop, fused_mhc `_mono_whole`) | see host contract |

## Host contract (ours differs from ATOM)

SGLang's layer loop reads the residual stream on the host at (a) Engram layers (`model.layers[nxt].engram`), (b) DSpark
capture layers (`dspark_layers_to_capture`), (c) after the last layer; `defer_post` is False there and `hc_post` runs
on the host. Under token parallel only own rows are valid, so each of these points needs an all-gather of the pending
tuple (x, residual, post, comb) or of the collapsed hidden. Options: (1) gather at each such point (custom AG, ~10-20
us each; count the Engram + capture layers of V4.1-Flash first); (2) break the token-parallel chain there (that
layer's K2 runs non-tok_par: full reduce). Start with (2) for simplicity -- per-layer `tok_par` flag decided at pack
time from the host's read points -- and measure.

## Steps (each: ISA dump of non-tok_par builds identical; CPU tests; then GPU)

1. Peer layout module + `tok_par` build flag in MonoBuild / FrontBuild / BackBuild / MoeBuild (default False) +
   sync.poll mixed spaces (ATOM sync.py diff) + token_par.own_view with our arg names. No behaviour change.
2. FFN reduce-scatter only, gathered back at once (temporary): MoE push-to-owner + own reduce, then a gather stage so
   the layer output is full again. Bitwise test vs non-tok_par at S = 12/24/48. Proves the RS + owner mapping.
3. K2 seam on own tokens + XG push / gather before the MoE; wo_b push to owner. Layer output still gathered (step 2's).
4. K1 seam + wqkv_a + qkv on own tokens; QG / QGS push; KG push + others' records written locally; wq_b from QG.
   Drop step 2's gather: the layer contract becomes own rows. Host: final gather + chain breaks at Engram / capture
   layers (option 2).
5. Indexer layers: xn / NBG for the compressor and SGLang's indexer between launches (they read full xn / ql rows),
   S3's iw / iquant on gathered rows (IWG).
6. Runner gating (`tok_par_from = 12`, s % tp == 0, not check mode, custom AG available), DSpark draft off,
   CUDA-graph capture of tok_par widths; AgentX TP4 c4 / c8 (+ encoder replay) vs ATOM 1012.

## Step 1 (done 10-14, mi355-4)

- `deepseek_v41/peer_layout.py`: peer buffer = [half 0: ATTN | FFN][half 1: ATTN | FFN][gather 0][gather 1]. The partial
  halves keep their old offsets and size (K2's parity stride `half` is a compiled constant, so growing the halves would
  change every non-tok_par K2's ISA); the gathers XG / QG / QGS / KG / NBG / IWG (ATOM's sizes; KG sized for a bf16 row,
  also fits our 528 B V41 record) sit at `gather_base(tp) = 2 * peer_half_bytes(tp)`, parity p at
  `+ p * gather_half_bytes()` (2,214,912 B at MAX_TOKENS 48). Runner allocates `peer_buffer_bytes(tp)` (TP4 15.7 -> 20.2
  MB). seam.attn_peer_bytes, moe.peer_bytes / ffn_offset, layer.peer_half_bytes now come from peer_layout (same values).
- `deepseek_v41/token_par.py`: TOKEN_ROW_BYTES with our K1 / K2 arg names (res_in/out, pend, out, pre/post/comb_in/out,
  pos, slot (int64), xn_out, ql_out, iq_out, iqs_out, iw_out; `q` is head-split, not owned); own_count; own_view(c, names,
  tp, rank) -- tp and rank explicit because our K1 context has no `d` / `rank` (K1 gets a rank arg in step 4).
- `tok_par: bool = False` in MonoBuild (asserts s % tp == 0 when set), FrontBuild, BackBuild, MoeBuild; `_front_key` /
  `_back_key` / `_moe_key` pass it; K1 / K2 symbols get `tok=True` only when set (other symbols unchanged).
- common/device/sync.py `Mailbox.poll`: per-spec scope (ATOM's sync.py diff), a batch may mix scratch and peer pairs.
- test_mono_specs.py TestDsv41PeerLayout (CPU): partial halves unchanged, gathers disjoint and past both halves.

## Step 2 (coded 10-14, mi355-4; GPU check queued with step 3's)

- stages/moe.py: `down_combine` under `c["tok_par"]` pushes a token's partial to its owner only, at its own token index
  there (`region_pair(rank, t - p n, n, col)`, same FFN offset); `run_moe` runs `stage_reduce` on
  `own_view(c, ("out",), tp, rank)` (n tokens, `out` from the rank's first row). Our `sum_partials` already sums in rank
  order 0..TP-1, so the owner's sum equals every rank's non-tok_par sum bitwise. Temporary (steps 2-3): `reduce_pair` also
  `put_bf`s the sum to every other rank's OG (new gather region, [token][hidden / 2], step row t0 + t; `put_bf` rounds as
  `.to(BFloat16)`), then `gather_out` polls OG for the other ranks' tokens and stores the word into `out` -> full rows.
- layer.py: K2 under `key.tok_par` passes `og` = (sym + gpar, peers + gpar, gather_base + OG offset), gpar = (ep & 1) x
  gather_half_bytes(); run_ffn sets `cb["tok_par"]`, `cb["og"]`. Non-tok_par: python branches only, ISA unchanged.
- tok_par K2 VGPR / LDS (isa_task27b_s2tok/): TP4 S=12 252 (non-tok 250), S=24 235-236 (232), S=48 256 (246); TP2 250 /
  235-237 / 254-255; scratch 0 everywhere. S=48 sits at the limit: watch step 3.
- runner.py check mode `SGLANG_MONO_DSV41_TOK_PAR_CHECK=1` (eager only; skipped under graph capture; tp > 1; M >= 12,
  M % tp == 0): K2 tok_par into fresh buffers, sync + cpu_group barrier, restore epoch buffer (epoch + MoE counters) and
  this width's scratch (K1's mailboxes live there), zero the peer buffer, barrier, then the regular K2; the 5 outputs
  compared bytewise. Logs "mono tok_par check ... MISMATCH <tensor>: N B differ, rows [..], max|d|" (first 20) and a
  per-200 summary per rank. The DSpark draft runner (same class) is checked too.
- fw_ab.sh: FW_TP / FW_GPUS (defaults 4 / 4,5,6,7) and the tok_par summary line; mono_isa_dump.py `--tok-par[-only]`.

## Step 3 (coded 10-14, mi355-4; GPU check queued)

- layer.py K2 under `key.tok_par`: wo_b's hook `push_attn_owner` sends a token's partial to its owner alone, at
  `attn_region_pair(rank, t - owner n, n, col)` (owner base picked by selects, one store). run_ffn runs the FFN seam on
  `own_view(ca, K2_SEAM_OWN, tp, rank)` (token_par.K2_SEAM_OWN: res_in/out, pre/post/comb_in/out): slice_prefetch /
  stage_reduce (ATTN at n rows) / stage_slice on the CTA's slices, stage_norm on n tokens with `store_normed_tok_par`
  (normed row at `normed + t0 HIDDEN 2`, plus the bf16 words from registers to every other rank's XG at step row
  t0 + t), XRDY published at t0 + t; the gate (before_ug) on own tokens. stages/moe.py run_moe: `gather_rows` first
  (other ranks' tokens on their xq CTA: `gather_x` polls XG, stores into `normed` at device scope, publishes XRDY).
- K2's res_out / post_out / comb_out / pre_out now hold this rank's rows only (the final contract); `out` is still full
  through OG. Until step 6 tok_par runs only in check mode, whose real step is the regular K2, so K1 (still full) is
  unaffected; the check compares `out` on all rows and the other four on this rank's rows.
- Spill found and fixed: holding the OG / XG bases (computed at K2 entry) across the kernel spilled two 64-bit values at
  S = 48 (scratch 20 / 36 B, reloaded at the reduce); `og` / `xg` are now callables computing the addresses where used.
  Result: VGPR TP4 S=12 254 / S=24 240 / S=48 240; TP2 254 / 240-242 / 245-248; scratch 0 (step 2 was 252 / 235 / 256).
  (Moving the slice weights' early load off made no difference; `/tmp/k2tok_probe.py tp,t,r ...` compiles single builds.)

## Risks

- VGPR: K2 S >= 12 builds already 246-250 (27a); ATOM fits under the same limit, but our vLLM-derived K1 / K2 differ.
  Watch spills per step.
- Our V41 KV record writer for others' tokens (KG) is new code (ATOM writes a bf16 ring).
- Mailbox regions in the peer buffer: system scope, epoch tags; parity halves must keep two layers in flight apart.
- Host read points multiply gathers; option (2) loses tok_par on those layers.
