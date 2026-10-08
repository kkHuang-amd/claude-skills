> ARCHIVE (frozen 2026-10-07): full bring-up / debug history of DEP_1005.md. Live doc = DEP_1005.md.

# DSV4.1-Flash DEP (DP attention + MegaMoE) bring-up on MI355X (2026-10-05)

Owner node: mi355-4

## CONTINUE HERE

**GSM8K (2026-10-07 09:20 +08, mi355-4): DP2 + TP MoE + replay PASSES** -- fix tree b70264d7bc, EVAL_ONLY=true
SERVER_ONLY=1 (real DSpark acceptance), c32 config (PDI 32), 5-shot 1319 on backend port 8889: 0.901 / 0.907 (baseline
TP 0.894-0.906), 0 crashes. Earlier 0.359 / 0.366 and the garbled sample outputs were INVALID: server started with
SERVER_ONLY but without EVAL_ONLY=true keeps SGLANG_SIMULATE_ACC_LEN (benchmark-only simulated acceptance), same trap
as env1001_r1. Router 8888 cannot proxy /model_info -> run GSM8K against 8889. The A/B conclusions drawn from those
garbled samples ("pre-existing DP accuracy bug") are void.
**PDI defaults updated (2026-10-07 08:00 +08, user):** DP + TP MoE on the fix tree now defaults c32 -> PDI 32,
c64 -> PDI 16 (P90-oriented; c128 and others unchanged = PDI 4 for CONC >= 32).
**QUEUE DONE (2026-10-06 21:35 UTC, all 6 PASS 0 err; rows in results table):** DP2 + TP MoE + replay, fix tree,
GPUs 0,1, one server at a time, `scripts/dp2tp_queue.sh` RUNS="32:32 64:4 64:16 64:32 128:4 128:16" (conc:PDI), tags
`dp2tp_replay_c<C>_pdi<P>`, 105 min timeout per run. Progress + one RESULT line per run (TTT, P90, p50, TTFT, requests,
crash count) in `/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/dp2tp_queue.txt`; ends with "QUEUE DONE" (~7 h, ~20:45 UTC). Next session:
read that file, append rows to the results table below, compare c64 vs tp2r_c64 142,243.2 / 52.0. c128 has no TP2
reference in results/agentx.md. Cleanup kills only port holders / sglang:: / aiperf / router in this container.
**Script defaults (2026-10-06 21:40 +08, user request):** agentx_colleague_mi355x_sglang.sh detects the DP late-layer
resize in SRC (`grep _late_layer_dp_counts`) -> DP_TP_REPLAY_OK=1 -> DP_MOE=tp defaults to REPLAY=1 and c32 to PDI 16
(dp2tp_replay_c32_pdi16 102,786 / 98.8 > PDI4 102,631 / 88.9). Without that tree (default SRC=/sgl-workspace/sglang) it
keeps REPLAY=0 / PDI 4. c64+ under DP unmeasured -> TP rule (PDI 4). Run the best DP2 TP-MoE point with
`SRC=/sgl-workspace/sglang-dsv41-dp-fault/python DP_ATTENTION=true DP_MOE=tp TP=2 CONC=32 ...`.
**DONE (2026-10-06 20:10 +08): bounded replay for DP2 + TP MoE PASSED 3600 s, 102,631 / 88.9 (-3.0% / -5.1% vs tp2r_c32)** (user: "catch up with TP"; TP2 c32 BCG 96,768 /
69.9 vs Replay 105,800 / 93.7, i.e. replay = +9.3% / +34%; our fixed DP2 TP-MoE 96,931 / 70.7 ~= TP2 without replay).
PUSHED as b70264d7bc on HaiShaw/sglang `fix/dsv41-dp-fault` (on top of cb5b0b1cdc, no trailer). Contents:
- deepseek_v4_hook: replay allowed with DP attention iff moe_a2a_backend none and dp == tp (a2a/DEP still rejected).
- deepseek_v4 `_late_layer_dp_counts`: eager forwards only (skipped under capture), every rank all-gathers (tp_group,
  int64 -> pynccl) its late-layer row count = tail rows if it extends, else all its rows; None if unchanged.
  `_enter_late_layer_dp` at i == late_layer_start on EVERY rank: global_num_tokens cpu/gpu/padded/for_logprob = those
  counts, buffer len = sum, SUM_LEN, reset dp_local cache, set_dp_buffer_len_from_batch; input_ids_global = None (no
  late hash layer in V4.1). `_exit_late_layer_dp` restores after the loop (logits path sees full rows again).
- Run: `SRC=/sgl-workspace/sglang-dsv41-dp-fault/python REPLAY=1 DP_ATTENTION=true DP_MOE=tp TP=2 CONC=32 GPUS=0,1
  TAG=dp2tp_replay_c32`. Pass: 3600 s, 0 err; target tp2r_c32 105,800 / 93.7.
  First launch rejected by my own guard: with DP attention cfg.dp_size is normalised to 1, the value is
  cfg.attn_dp_size -> guard now checks `attn_dp_size == tp_size` (logs dp2tp_replay_c32_guardfail/). Relaunched:
  ready 11:00:33 UTC, replay flag confirmed in sglang_command, 11 min clean, both schedulers alive; ends ~12:01 UTC.
**PUSHED (2026-10-06 18:50 +08):** problem-3 fix only (4 files, +55/-4: parallel_state COPY_IN, forward_batch_info
`_dp_tp_moe_masks_pad_rows`, decode_cuda_graph_runner `_set_dp_real_num_tokens`, deepseek_v4 `_mask_dp_pad_rows`) as
commit cb5b0b1cdc on HaiShaw/sglang branch `fix/dsv41-dp-fault` (/sgl-workspace/sglang-dsv41-dp-fault, base
f3b5a28f43, no attribution trailer). NOT included (still local in /sgl-workspace/sglang only): problem-1 bounded-replay
patch, engram IDLE zero-hash patch. aiter-side repro paused by user.
**Status (2026-10-06 18:35 +08, mi355-4): clean fix PASSED 3600 s, 96,931 / 70.7 (-8.4% / -24.5% vs tp2r_c32, +15.0% / +35.7% vs DEP2). Next: aiter minimal repro (not reproduced yet with 50k replays). Nothing running.**
**Prev status (16:50): BEST FIX = custom AG/RS kept ON with SGLANG_AITER_CAPTURE_COPY_IN=1 (capture stages AG/RS inputs through the pre-registered IPC pool) + pad-row mask; dp2tp_copyin passed 3600 s, 79,913 / 49.9. Root cause sits in aiter's registered graph-buffer path (peer reads a graph-pool tensor over IPC); kernel barriers verified correctly paired. Still TODO: remove debug probes, make COPY_IN the default for DP TP-MoE, clean perf rerun, aiter-side minimal repro. Nothing running.**
**Prev status (2026-10-06 14:45 +08, mi355-4): PROBLEM 3 SOLVED (workaround config passes 3600 s, no clamp): two aiter custom-collective bugs under decode graph replay -- custom all-gather (all_gather_reg) delivers garbage peer rows, custom equal-chunk reduce_scatter turns clean partials into NaN. Fix config: SGLANG_USE_AITER_AG=0 + SGLANG_DP_USE_REDUCE_SCATTER=0 + pad-row mask (default on). Nothing running, GPUs free. Older status below.**
**Prev status (2026-10-06 12:10 +08, mi355-4): carrier of the problem-3 NaN found = MAX_LEN PAD ROWS holding stale
graph-pool memory with HUGE FINITE values (bf16-max scale), not NaN. Clamp A/B in flight (TAG dp2tp_clamp).
User: focus on DP (DP2 attention + TP MoE, DP_MOE=tp), no DEP.**
- **New evidence (TAG dp2tp_idprobe2, decode graph ON, crashed as usual ~6 min):** the per-layer absmax probe now
  shows the gathered buffer's OTHER-rank slice at **layer 0** holding `3.3895313892515355e+38` — exactly bf16 max,
  i.e. random bits read as bf16 — while this rank's own slice is 1.0. Second rank: 1.688e38 on the other slice at
  layer 2. So the MoE input is NOT clean after all: it is finite but ~1e38, which overflows the fp8/fp4 MoE and
  produces the inf/NaN partials seen at slots 160/400. **This is why every earlier `nan_to_num` experiment failed:
  nan_to_num only maps NaN/inf, never huge finite values.** The 10-06 09:50 absmax run reporting "other slice
  absmax < 5" was read too early / from a step whose garbage sat in the OWN slice ([3,1] crash) — treat it as
  refuted, not as evidence.
- **Mechanism:** the pad rows of each rank's local hidden buffer are never refreshed by the decode graph replay
  (`decode_cuda_graph_runner` copies only `[:raw_num_token]`; `cuda_graph_buffer_registry.py:547` FOREACH_COPY,
  "padded tokens aren't read"), and they do get read here: `_dp_gather_via_all_gather` with attn_tp_size==1 is a
  plain `all_gather_into_tensor`, so the whole MAX_LEN chunk — pad rows included — lands in the global buffer and
  goes through the MoE. Pad rows also pass through layer-0 attention with sentinel seq_lens, so their values are
  whatever the graph pool last left there. Fits [1,0] (idle peer: all 12 rows stale), [2,1] (peer 6 real + 6 stale)
  and [3,1] (the rank with 1 of 3 slots real has stale rows in its OWN slice -> own rows NaN, peer not idle).
- **Refuted along the way (this session):**
  - *input_ids pad slots carry garbage token ids* -> NO. New `id_probe_record` shows the gathered ids are all valid
    and small (`[mine_max, mine_bad, other_max, other_min, other_bad, len]` = `[21017,0,21017,16,0,24]`,
    `[0,0,223,5,0,12]`): bad counts are 0 and an idle rank's own slot is all zeros. The ids are stale-but-in-range,
    so hash routing / engram lookup stay valid. Keep the probe, it is cheap and it closes this line for good.
  - *aiter custom all-reduce* (the aiter-custom-allreduce-nan-crash skill signature is very close) -> NO. Probes
    240/400 sit on the `self.mlp` output with `mlp_reduce_scatter=True`, i.e. upstream of every collective, and
    aiter e7d2453f2 does carry the PR 3514 `end_sync` in `cross_device_reduce_1stage`.
- **Problem 3 chain (proven):** MoE output NaN -> DSpark verify logits rows all NaN -> chain_speculative_sampling
  returns vocab_size 129280 -> DSpark draft embed index_select OOB -> HSA 0x1016.
- **Key result:** the NaN ONLY happens with the decode CUDA graph. `--cuda-graph-backend-decode disabled` (+ local
  engram IDLE patch) = dp2tp_nograph2 ran the full 3600 s, 2508 ok, 0 errors. With the graph: 12+ repros all crash
  ~5-6 min after ready.
- **What the probes showed under the graph:** the gathered MoE input is finite, normal magnitude and identical on both
  ranks; yet each rank's self.mlp partial is clean on its OWN slice and NaN only on OTHER ranks' slices; combine sums
  that into the victim's rows. Usually the peer is idle (global_num_tokens [1,0]/[0,1]), also seen [4,1], [3,1].
- **Ruled out:** dp buffer init (SGLANG_DP_BUFFER_ZERO), NaN/inf in gathered rows (nan_to_num incl. inf), huge
  finite garbage (absmax < 5), aiter reduce_scatter (SGLANG_DP_USE_REDUCE_SCATTER=0 still crashes), graph bucket
  mismatch (DPGEOM: idle and active pick the same bs/tokens, non-ragged), side-stream race (CUDA-only stream),
  cross-row quant scale (per_1x32), mHC post fusion (TP4/Blackwell only).
- **Why DEP2 (MegaMoE) does not hit it:** _use_tp_moe_gather is False -> no global dp buffer, no dp_gather /
  reduce_scatter, no rank ever computes a partial for other ranks' rows that is then summed (a2a returns each token's
  result to its own rank); pad rows are masked (moe_num_token_non_padded local, MegaMoE _fill_padded_rows, idle 0-token
  branch); script sets SHARED_EXPERT_LOCAL/GATHERV/REDUCE_SCATTER=0 for megamoe; different MoE kernels. DEP also runs
  the decode graph + DSpark + engram + idle ranks (dep2_c32 3600 s, 0 err), so none of those alone triggers it ->
  suspects = TP-gather-path-only pieces inside the graph: dp buffers (symmetric memory under MAX_LEN decode,
  get_global_dp_buffer use_symmetric_memory), dp_gather/dp_scatter collectives, aiter fused_moe on the gathered buffer,
  shared expert LOCAL/TP1.
- **11:02 RESULT dp2tp_clamp: STILL crashes** (ready 02:55:40 UTC, "bonus out of" 03:02:18, [0,1]). Clamp was live
  (other-slice absmax pinned at 9984 = bf16(1e4) on layers 2-9 -> the pad rows still hold huge garbage every layer).
  But now the MoE side is CLEAN: mine_vs_peer {}, mlp_out_mine [], other_partial [], shared_local []; yet moe_out
  (post-combine, own rows) NaN from layer 3, local_pre_gather from 4. => with clean finite self.mlp output, NaN
  appears in the COMBINE (default = aiter custom equal-chunk reduce_scatter, dp_reduce_scatter_tensor) or the
  shared_local add. Caveat: probes test NaN only, not huge finite; the other-row partials (fed 1e4 inputs) may be
  huge, though only own rows should reach this rank. The old "REDUCE_SCATTER=0 still crashes" (dp2tp_nors) was with
  unclamped garbage, so it does not clear the combine.
- **DONE (PASSED 3600 s) 11:05 `TAG=dp2tp_clamp_nors`**: clamp ON + `SGLANG_DP_USE_REDUCE_SCATTER=0` (AR + dp_scatter combine),
  one variable vs dp2tp_clamp. Survives -> aiter custom reduce_scatter is the NaN source; crashes -> look at the
  shared_local add / AR and add absmax probes on the partial and post-combine.
  **11:27 interim: NO crash** — ready 03:09:04 UTC, at 03:27 (18.5 min, ~3x the usual crash point) both schedulers
  alive, no "bonus out of"/HSA/OTHERPARTIAL, aiperf progressing (prefix hit 95%). FULL RUN PASSED 04:10 UTC: 3660 ok, 0 errors, no crash (results row below).
  Reading so far: clamp alone crashes (dp2tp_clamp), RS=0 alone crashes (dp2tp_nors), clamp + RS=0 survives ->
  BOTH the pad-row garbage and the aiter custom reduce_scatter path are involved. Still to explain why the
  equal-chunk reduce_scatter turns clean, clamped partials into NaN on own rows.
- **12:25 PROPER FIX implemented (local, uncommitted), run IN FLIGHT `TAG=dp2tp_maskpad`** (probes on, NO clamp,
  REDUCE_SCATTER default=1, decode graph ON -> one variable vs the crashing baseline):
  - `forward_batch_info.enable_num_token_non_padded()` now also True for attn_dp_size > 1 + moe_a2a none
    (`_dp_tp_moe_masks_pad_rows`, env `SGLANG_DP_MASK_PAD_ROWS`, default true). That turns on the existing
    capture-safe `num_token_non_padded` graph slot (per-replay copy of this rank's real count) — same path DEP2
    (EP>1) already uses with DSpark verify. `moe_num_token_non_padded()` still returns None under attn DP, so MoE
    topk masking is unchanged.
  - `deepseek_v4._mask_dp_pad_rows`: `masked_fill_` rows >= num_token_non_padded of local hidden to 0 right before
    `dp_gather_replicate` in `_run_moe_ffn_dp_sync`.
  - Pass: 3600 s, 0 errors, accept len ~3.5 (proves real verify rows are not masked). If it crashes, the aiter
    custom reduce_scatter has its own bug -> rerun with REDUCE_SCATTER=0 to separate.
  - **RESULT dp2tp_maskpad: crashed** (ready 04:27:43 UTC, crash 04:33:46, [1,0]). The plumbing works on the active
    rank (slice probe col 3 = num_token_non_padded = 6 = its real rows), but the IDLE peer's pad rows were NOT zeroed
    (other-slice absmax 2.9e38 from layer 4): the idle rank's replay keeps a stale count in the slot (its fabricated
    batch does not reach the registry post_fill / pre-planned branch skips fill_from). Also seen: own rows NaN
    after the layer-1 combine while self.mlp output was NaN-free -> RS path still suspect.
  - **v2 (13:00, IN FLIGHT `TAG=dp2tp_maskpad2`)**: `decode_cuda_graph_runner._set_dp_real_num_tokens` does
    `buffers.num_token_non_padded.fill_(raw_num_token)` before EVERY replay (both branches; eager fill_, legal outside
    capture), gated by the same `_dp_tp_moe_masks_pad_rows()`. Same run config as dp2tp_maskpad.
  - **RESULT dp2tp_maskpad2: crashed** (ready 04:41:04, crash ~04:47, **global_num_tokens [1,1]** = 6 real rows per
    rank, n_real = 6 = all rows, i.e. NO pad rows at all). Yet the other slice is NaN (5-6/6 rows) from layer 1 with
    absmax ~3.4e38 on layers 1-8. => pad-row zeroing alone is NOT sufficient; the peer's REAL rows arrive bad too
    (the peer's own state went bad first — KV-cache poisoning from an earlier step is a candidate, since NaN in KV
    persists across steps). Matrix so far: clamp only X, RS=0 only X, mask only X, clamp + RS=0 PASS.
  - **12:48 IN FLIGHT `TAG=dp2tp_maskpad_nors`**: mask (v2) + `SGLANG_DP_USE_REDUCE_SCATTER=0`, no clamp. Pass ->
    formal fix = pad masking + AR/dp_scatter combine (and the aiter custom reduce_scatter needs its own bug hunt);
    crash -> clamp is doing something mask does not (huge values in REAL rows), look at what writes them.
  - **RESULT dp2tp_maskpad_nors: crashed** (ready 04:53:53, crash 04:59:57, [1,0]). DECISIVE probe pair:
    the IDLE rank (OTHERPARTIAL, mode IDLE) reads n_real = 0 (mask works) and its OWN slice is clean, yet sees NaN in
    the ACTIVE rank's real rows at layer 0; the ACTIVE rank sees 3.39e38 in the idle rank's (zeroed) slot at layer
    0. Each rank's own slice (local copy) is clean, the slice that arrives via the collective is garbage => **the
    DP gather collective itself corrupts data under graph replay**. Under capture `_all_gather_into_tensor`
    (parallel_state.py:1292) uses the **aiter custom all-gather `all_gather_reg`** (registered IPC input). The
    allgather_naive/vec kernels do have `end_sync`, so not a literal PR-3514 copy; suspect reg/IPC-buffer reuse.
  - **13:15 IN FLIGHT `TAG=dp2tp_mask_noag`**: mask + `SGLANG_USE_AITER_AG=0` (gather via RCCL/pynccl), RS default,
    no clamp — one variable vs dp2tp_maskpad2. Pass -> aiter custom all-gather is the root cause.
  - **RESULT dp2tp_mask_noag: crashed** (ready 05:14:40, crash 05:20:23, [2,1]; env confirmed in log) but it SPLITS
    the two bugs: with AG=0 the gathered buffer is clean on both slices (absmax <= 1.0, no 3.4e38 anywhere) ->
    **aiter custom all-gather (all_gather_reg under graph) is bug #1**. self.mlp output fully NaN-free
    (mlp_out_mine / other_partial / shared_local empty) yet moe_out NaN from layer 0 on own rows -> **aiter custom
    equal-chunk reduce_scatter (dp_reduce_scatter_tensor, RS=1) is bug #2**. Explains clamp+RS=0 passing: clamp made
    AG garbage finite, RS=0 dodged bug #2.
  - **13:28 IN FLIGHT `TAG=dp2tp_mask_noag_nors`**: mask + `SGLANG_USE_AITER_AG=0` + `SGLANG_DP_USE_REDUCE_SCATTER=0`,
    no clamp. Pass -> formal fix = both custom collectives off for DP2 TP-MoE (+ pad mask); aiter bugs to report.
    **13:51 interim: NO crash** — ready 05:30:07 UTC, 21 min clean (no bonus/HSA/OTHERPARTIAL = no NaN anywhere,
    both schedulers alive, prefix hit 95.9%). FULL RUN PASSED 06:31 UTC (results row below).
- **Next (older plan), decode graph ON throughout:**
  1. IN FLIGHT `TAG=dp2tp_clamp` (first launch died on port 9123 held by the previous crash run's leftover
     tokenizer/router/python3 -> logs dp2tp_clamp_portfail/; after a crash, kill port holders by PID from
     `ss -ltnp | rg ':9123|:8888|:8889'` before relaunching): `SGLANG_DP_ZERO_PAD_ROWS=1` now means `nan_to_num_(0,0,0).clamp_(-1e4, 1e4)` on
     the gathered buffer (deepseek_v4.py `_run_moe_ffn_dp_sync`), i.e. it finally neutralises huge FINITE garbage.
     Real activations are < ~10 so the clamp is a no-op for them. PASS = survives well past the 5-6 min mark that
     12+ earlier decode-graph repros died at -> carrier confirmed.
  2. If it passes, the proper fix is to stop feeding pad rows to the MoE instead of clamping: plumb the real
     per-rank token count to the GPU capture-safely (the graph runner knows `raw_num_token` on the host at replay
     time, so an eager copy into a small persistent int32 buffer before replay is legal) and zero / mask the pad
     rows after the gather. `PaddingPolicy.ZERO` on the hidden-state graph slot, mirroring what `positions` already
     has (issue #24361), is the cheaper variant worth trying first.
  3. Then the perf arm: 3600 s, 0 errors, no "bonus out of", and TTT / P90 vs tp2r_c32.
  `SGLANG_DP_ZERO_PAD_IDS=1` (new in decode_cuda_graph_runner, default off, zeroes `input_ids[raw_num_token:]`
  before every replay) is now only a tidiness knob — the id probe showed the ids were never the problem.
- **Superseded candidate list** (kept in case run 1 refutes the above): diff graph replay vs eager. Candidates: (a) buffers fixed at capture time that the
  MoE path reads (get_global_dp_buffer / get_local_dp_buffer allocated during capture vs replay; aiter fused_moe
  moe_buf / sorting workspaces; topk buffers); (b) per-replay metadata: decode_cuda_graph_runner writes
  global_num_tokens_gpu = [num_tokens]*dp (_global_num_tokens_for_graph) and dp buffer len; anything the MoE uses to
  pick rows (rank offset, local len) baked at capture; (c) SGLANG_DP_SHARED_EXPERT_LOCAL / SGLANG_SHARED_EXPERT_TP1
  under capture. Cheap A/Bs first, one variable each, keep the decode graph ON: SGLANG_DP_SHARED_EXPERT_LOCAL=0,
  SGLANG_SHARED_EXPERT_TP1=0, SGLANG_DP_USE_GATHERV=0 (script now honors env overrides for all four).
  Repro (decode graph on, probes on); watch: `bash scripts/watch_crash_or_done.sh <tag>/server.log 'bonus out of|HSA_STATUS|Fatal Python' 4200`
  (exit 2 = crash, 0 = launcher finished, 1 = timeout):
  `cd /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx && SGLANG_DSPARK_DEBUG_IDS=1 DP_ATTENTION=true DP_MOE=tp TP=2 CONC=32 GPUS=0,1 PORT=8888 TAG=<tag> setsid nohup bash /workspace/claude-skills/dsv41/scripts/agentx_colleague_run.sh > <tag>.nohup 2>&1 < /dev/null &`
  Eager control: add `EXTRA_ARGS="--cuda-graph-backend-decode disabled"`.
  Pass criteria: decode graph ON, 3600 s, 0 errors, no "bonus out of"; then compare TTT/P90 with tp2r_c32.
- **Local uncommitted SGLang changes** (/sgl-workspace/sglang, base affa261e3d):
  KEEP: problem 1 patch (arg_groups/deepseek_v4_hook.py guard; _tail_rows in layers/attention/deepseek_v4_backend.py,
  _scatter_tail_rows in models/deepseek_v4.py); layers/engram.py IDLE -> zero hash ids (needed for eager idle).
  DEBUG, remove after the fix (all default off): SGLANG_DSPARK_DEBUG_IDS -> dspark_draft._debug_check_draft_ids;
  dspark_worker_v2 _debug_check_prefill_ids / _debug_check_bonus / _debug_report_other_partial ("OTHERPARTIAL") /
  _nan_probe_layers + forward_batch_generation wrapper; models/deepseek_v4 nan_probe_* / nan_slice_* / absmax_* and
  their calls in the layer loop and _run_moe_ffn_dp_sync; model_executor/runner/decode_cuda_graph_runner "DPGEOM" log.
  SGLANG_DP_BUFFER_ZERO (layers/dp_attention.py); SGLANG_DP_ZERO_PAD_ROWS (deepseek_v4 _run_moe_ffn_dp_sync,
  nan_to_num_ on the gathered buffer). Probe slots: 0+ layer entry, 80+ local pre-gather, 160+ moe_out (post-combine),
  240+ mlp_out own slice, 320+ shared_local, 400+ other-slice partial.
- Problem 1 (bounded replay under DP) patched locally for the a2a (DEP) case only, untested on GPU.
- Logs: /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/failed_dep2_1005/dp2tp_* (crash repros), dp2tp_nograph2/ (eager, completed).
**Only completed DP number:** dep2_c32 (DEP2 MegaMoE, option D, prefill graph off, no replay) 84,320.5 / 52.1 =
-20.3% / -44.4% vs TP2 Replay tp2r_c32 (105,800.1 / 93.7), -28.5% / -65.2% vs vLLM B200 DEP2 c32 (117,908.2 / 149.5).

**Open DP problems (SGLang affa261e3d, all on DSv4.1 + DSpark + engram host table):**
1. Bounded replay blocked under any DP attention (arg_groups/deepseek_v4_hook.py:317-335, "input_ids_global is a
   DP-wide gather"). vLLM DEP2/DEP4 run it (eager prefill steps, layers 21-39 on last 128 tokens). Likely biggest
   prefill/TTFT gap. Needs a per-rank tail slice before / independent of the DP gather.
2. Breakable prefill graph replay fails when a rank is padded to the DP max: "Tensor match failed ... c2.cuh:476 ...
   expected 7587 but got 16384" in kernels/ops/attention/dsv4/low_ratio_compress.py c2_prefill_norm_rope_store
   (dp2tp_c32_bcg). Workaround: prefill graph disabled under DP (script default). vLLM pads alike and its PIECEWISE
   graphs cope (DEP4 captures prefill up to 8190; DEP2 only to 576, longer prefill eager).
3. DP2 + TP MoE (DP_MOE=tp) hard crash ~6 min in: HSA 0x1016 in at::native indexSelectSmallIndex<BFloat16>
   (grid 5120 = hidden), "Fatal Python error: Aborted", no Python stack (dp2tp_c32_hsa1016). Next: repro with
   HIP_LAUNCH_BLOCKING=1 (or AMD_SERIALIZE_KERNEL=3) to get the caller (DP gather/scatter, DSpark, dp-lm-head?).
4. Engram DP lookup needs global_dp_buffer_len, unset with MoE a2a + dp==tp (layers/engram.py:829). Worked around by
   option D (no --moe-dense-tp-size -> require_mlp_tp_gather True). Proper fix: compute the slot from
   global_num_tokens like vLLM (engram_gathered_num_tokens).
5. MegaMoE at EP2 decode 1.2-1.7x slower than TP fused_moe (microbench below; same for DSv4-Pro shape) -> config rules
   are EP8-oriented; tile_n=256 / bounded-class overrides GPU-fault. aiter-side, lower priority for DEP2.
6. V4.1 vision rejects MoE a2a -> --json-model-override-args '{"vision_n_layers": 0}' (works; text path unchanged).
7. Node: rank 1 (NUMA node 1) loads 260-490 s vs ~120 s; two servers at once trip the hardcoded 480 s
   UNBALANCED_MODEL_LOADING_TIMEOUT_S. Run ONE server at a time. Never `pkill -f` a pattern contained in your own
   command (killed the agent shell once).
**Health check while a DP run is "running":** `ps -eo comm | rg -c sglang::sch` must be 2 (TP2) and both GPUs hold
VRAM; tokenizer/router survive a dead scheduler, so aiperf just hangs without errors.
**Uncommitted:** all of 2026-10-05 after f49640f (scripts, DEP_1005.md, REL_REGRESS_1005.md, results/agentx.md,
bench_megamoe_dsv41.py); f49640f itself not pushed.
**D (chosen 2026-10-05 by user):** drop `--moe-dense-tp-size 1` (knob DP_MOE_DENSE_TP_SIZE re-adds it) so
require_mlp_tp_gather() returns True -> DP ranks padded alike, global_dp_buffer_len set, engram DP lookup works. Same
idea as vLLM engram (pads ids to the DP max from dp_metadata.num_tokens_across_dp_cpu, all-gather, head-sharded lookup,
all-gather rows + select own tokens; vllm models/deepseek_v41/common/engram.py @ac9126e58). Prefill graph still disabled.
**Goal:** SGLang counterpart of B200 vLLM DEP2/DEP4 (results/agentx.md "Reference: vLLM on B200 NEW").
**Script:** `scripts/agentx_colleague_run.sh` with `DP_ATTENTION=true` (path aligned to agentx/sa-script/
dsv4_fp4_mi355x_sglang_mtp.sh, DP + ENABLE_MEGAMOE arm; differences: DSv4.1 per-CONC PDI/mem/chunk, MTPR = per-rank chunk,
stream-interval 1, DSpark + SGLANG_RAGGED_VERIFY_MODE=static, vision_n_layers=0).
**Repro (DP_MOE=megamoe default = DEP2; DP_MOE=tp = DP2 + TP MoE; SERVER_ONLY=1 keeps the server up; problem 3
needs HIP_LAUNCH_BLOCKING=1 added):**
```bash
cd /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx && DP_ATTENTION=true DP_MOE=tp TP=2 CONC=32 GPUS=0,1 PORT=8888 TAG=dp2tp_dbg \
  setsid nohup bash /workspace/claude-skills/dsv41/scripts/agentx_colleague_run.sh > dp2tp_dbg.nohup 2>&1 < /dev/null &
```
Pass criteria for a DP arm: completes 3600 s with 0 errors, then TTT / P90 vs tp2r_c32 and vLLM DEP2 c32.

## aiter custom AG / RS bug hunt (mi355-4 2026-10-06 afternoon; user: "can the two aiter bugs be fixed?")

- Under capture SGLang calls AG via `all_gather_reg` (always registered), RS via `reduce_scatter(registered=True)`, AR
  via `registered_input=enable_register_for_capturing` (True on gfx950). All three read the PEER's graph-pool input
  directly over IPC. RS=0 (AR) and RS=1 (RS) consume the SAME MoE-partial tensor, yet only RS breaks.
- Kernel code (csrc/include/custom_all_reduce.cuh): allgather_naive/vec and reduce_scatter_split_first_dim use the
  same start_sync / end_sync<final=true> as cross_device_reduce_1stage; no missing barrier (not a PR-3514 copy).
- Graph-buffer registration (`flush_graph_buffers` / C++ `register_graph_buffers`) pairs addresses purely by
  position, no cross-rank count/order check; `get_buffer_RD` reuses an existing map entry instead of pushing.
  Counts in logs are symmetric (2640/2640, 96/96) -> no count misalignment seen (order not verified).
- Hypothesis under test: **barrier-counter desync**. `Signal._flag[80]` is one monotonic per-block counter shared by
  ALL custom kernels; waits are `peer_flag >= my_flag`. If one rank runs one more/less custom kernel touching
  block b (or same op with a different grid), block b pairs kernel n on one rank with kernel n-1 on the other
  forever -> silent garbage reads, no hang (RCCL would hang instead). Only big collectives use high blocks ->
  intermittent.
- Probe: `dspark_worker_v2._debug_report_ca_flags` (SGLANG_DSPARK_DEBUG_IDS=1, every
  SGLANG_DEBUG_CA_FLAGS_EVERY=200 steps): synchronize, hipMemcpy this rank's `_flag[80]` (meta + 5120 B), log
  `CAFLAGS step= ... sum= flags=`. Lockstep DP ranks must show identical arrays at the same step.
  IN FLIGHT 14:58 `TAG=dp2tp_caflags` (crashing config: AG/RS custom default, pad mask on).
- **RESULT: desync REFUTED.** dp2tp_caflags (every 200 steps) crashed before decode samples; dp2tp_caflags2 (every
  5 steps, synchronize each sample) compared 3536 same-step pairs incl. decode: **0 mismatches** -> barriers are
  always correctly paired. Side finding: with a sync every 5 steps the run did NOT crash in 20 min (4 OTHERPARTIAL
  NaN events still) -> host syncs mask it = a timing/visibility race, like the PR-3514 case.
- Also ruled out: expandable_segments (log: `expandable_segments=False`, so AR is not on a forced copy-in path);
  kernel code AR vs RS/AG uses the same start_sync/end_sync and plain loads.
- Remaining mechanism: barriers paired, but the data read over IPC from the peer's GRAPH-POOL tensor is stale /
  belongs to another tensor (graph pool addresses are reused across buckets; start_sync is RELAXED/device-scope,
  no acquire before peer reads). Fix candidate needing no kernel change: under capture stage AG/RS inputs through the
  pre-registered IPC pool (copy-in / "unreg" path, which SGLang already uses under SGLANG_MEMORY_SAVER_CUDA_GRAPH).
  New env `SGLANG_AITER_CAPTURE_COPY_IN=1` (parallel_state.py `_all_gather_into_tensor` + `_maybe_aiter_reduce_scatter`).
  IN FLIGHT 15:37 `TAG=dp2tp_copyin`: crashing config (AG/RS custom, mask on) + COPY_IN=1, flag probe off.
  Pass -> bug is in the registered graph-buffer path; fix = copy-in (one <=1 MB memcpy per call).
  **15:55 interim: NO crash and 0 OTHERPARTIAL** for 17 min after ready (07:38:11 UTC), both schedulers alive. Every
  earlier custom-AG/RS run crashed by ~6 min and the sync-masked one still logged OTHERPARTIAL.
  **FULL RUN PASSED 08:45 UTC** (results row): copy-in fixes both aiter custom ops; +1.1% / +1.6% vs RCCL fallback.

- **17:00 cleanup + perf rerun + aiter repro (mi355-4):**
  - Debug probes removed; clean diff = 7 files / 70 lines, saved `dsv41/patches/dp2tp_fix_clean_1006.patch`
    (full debug diff kept at /shared_nfs/kk/results/DeepSeek-V4.1-Flash/patches/dp2tp_debug_full_1006.patch). `SGLANG_AITER_CAPTURE_COPY_IN`
    now defaults to 1. Clean perf run `TAG=dp2tp_fix_c32` PASSED: 96,931 / 70.7 (results row).
  - AR was never on the registered path: the copy-in run logs ZERO "Registering N cuda graph addresses", the crashing
    run 3200 (target) + 120 (draft) per rank -> all registrations came from AG + RS. Slots advance per flush
    (`d_rank_data_base_ += n`), so sessions do not overwrite each other.
  - Repro `/shared_nfs/kk/results/DeepSeek-V4.1-Flash/aiter_repro/repro_ag_rs_graph.py` (2 GPUs, buckets 6/12/24/48 in one pool, input made
    in-graph, AG reg -> RS reg, junk 3e38 scratch, random replay order, device-side check): **50k replays, 0 errors**
    -> not reproduced yet. `--mismatch` (rank1 eager/unreg while rank0 replays reg) HANGS instead of corrupting, so
    "one rank graph, other eager" is not the SGLang mechanism (SGLang never hung).
  - Lane note: the repro ran on GPUs 2,3 during the perf run's first minutes (09:05-09:08 UTC, the hung mismatch case
    spun ~1.5 min); killed. Do not run the repro concurrently with a perf run.

## Bring-up log (mi355-4, sglang affa261e3d, logs agentx/failed_dep2_1005/ and agentx/dep2_c32/)

1. `--enable-decoder-swa-bounded-replay cannot be combined with DP attention yet` -> DP defaults REPLAY=0.
2. `V4.1 vision currently supports TP/EP/DP without CP, PP or MoE A2A` -> `--json-model-override-args
   '{"vision_n_layers": 0}'` (loader skips vision.* / *_vl weights; text gate bias unchanged). Weights then load:
   150.88 GB/rank, avail 132 GB.
3. **Blocker:** engram `_dp_sharded_lookup` (layers/engram.py:829) `torch.empty((None, ...))`: with dp == tp and
   moe_a2a_backend megamoe, require_mlp_tp_gather() and require_attn_tp_gather() are both False, so runners never set
   global_dp_buffer_len, but engram's DP path needs it for dp_gather_replicate. Hit in breakable prefill-graph capture
   and, with the prefill graph disabled, in decode-graph capture. Eager forward has the same dependency.
   => SGLang engram under DP only works with `--moe-a2a-backend none` today.

## DP2 + TP MoE (DP_MOE=tp: --tp 2 --dp 2, ep 1, moe_a2a none; 2026-10-05)

- Bounded replay still blocked (deepseek_v4_hook.py rejects any DP attention: input_ids_global DP-wide gather).
- First try with breakable prefill graph (dp2tp_c32_bcg, failed_dep2_1005/): ran ~4 min then both schedulers died:
  "Tensor match failed ... c2.cuh:476 ... expected 7587 but got 16384" in c2_prefill_norm_rope_store -- a rank
  padded to the DP max prefill batch breaks the breakable replay. Tokenizer/router stayed up, so aiperf hung at 30/354
  with no error (check `ps -eo comm | rg -c sglang::sch`, not only the log). DP now always defaults the prefill
  graph to disabled.
- Second try dp2tp_c32 (prefill graph disabled; failed_dep2_1005/dp2tp_c32_hsa1016/): ready 10:35 UTC, warmup 219/354
  at 300 s, then at 10:40:57 one scheduler hard-crashed: HSA_STATUS_ERROR_EXCEPTION 0x1016 in
  at::native indexSelectSmallIndex<BFloat16> (grid 5120 = hidden size, i.e. an out-of-range row index_select on
  hidden states), "Fatal Python error: Aborted", no Python stack. Previous step: both DP ranks prefilling 16384-token
  chunks (DP0 #new-seq 3). The other scheduler spun in a collective; aiperf hung at done=2 for 13 min, GPU1 VRAM 0.3 GB.
  Caller unknown (async). Next for root cause: HIP_LAUNCH_BLOCKING=1 / AMD_SERIALIZE_KERNEL=3 repro for a Python stack.

## Results (append-only; node + date per row)

| date (node) | point | TTT / P90 | vs TP2 Replay tp2r_c32 | vs vLLM DEP2 c32 | notes |
|---|---|---|---|---|---|
| 2026-10-05 (mi355-4) | DEP2 c32 dep2_c32 (option D, prefill graph off, no replay) | 84,320.5 / 52.1 | -20.3% / -44.4% | -28.5% / -65.2% | 3861/4215 ok, 0 err, intvty p50 103.4 (tp2r 166.6); prefix hit ~96% |
| 2026-10-06 (mi355-4) | DP2 + TP MoE c32 dp2tp_nograph2 (decode CUDA graph OFF, local engram IDLE patch, debug probes on) | 44,788.7 / 23.6 | -57.7% / -74.8% | n/a | diagnostic: first DP2 TP-MoE run to finish 3600 s, 2508 ok, 0 err; eager decode is slow |
| 2026-10-06 (mi355-4) | DP2 + TP MoE c32 dp2tp_clamp_nors (decode graph ON, SGLANG_DP_ZERO_PAD_ROWS=1 clamp, SGLANG_DP_USE_REDUCE_SCATTER=0, debug probes on) | 78,621 / 48.2 | -25.7% / -48.6% | n/a | first DECODE-GRAPH DP2 TP-MoE run to finish 3600 s: 3660 ok / 4014, 0 err, no NaN; TTFT p50 0.61 s, intvty p50 82.0; vs dep2_c32 -6.8% / -7.5%. Workaround, not the fix |
| 2026-10-06 (mi355-4) | DP2 + TP MoE c32 dp2tp_mask_noag_nors (decode graph ON, pad-row mask, SGLANG_USE_AITER_AG=0, SGLANG_DP_USE_REDUCE_SCATTER=0, no clamp, debug probes on) | 79,080 / 49.1 | -25.3% / -47.6% | n/a | PASS 3600 s: 3659 ok / 4013, 0 err, no NaN (0 OTHERPARTIAL); TTFT p50 0.65 s, intvty p50 81.9; vs dep2_c32 -6.2% / -5.8%; candidate formal config |
| 2026-10-06 (mi355-4) | DP2 + TP MoE c32 dp2tp_copyin (decode graph ON, pad-row mask, custom AG+RS ON with SGLANG_AITER_CAPTURE_COPY_IN=1, no clamp, debug probes on) | 79,913 / 49.9 | -24.5% / -46.7% | n/a | PASS 3600 s: 3710 ok / 4064, 0 err, 0 OTHERPARTIAL; TTFT p50 0.59 s, intvty p50 85.1; vs RCCL fallback dp2tp_mask_noag_nors +1.1% / +1.6%; vs dep2_c32 -5.2% / -4.2%; RECOMMENDED config |
| 2026-10-06 (mi355-4) | DP2 + TP MoE c32 dp2tp_fix_c32 (CLEAN: probes removed, script defaults, custom AG+RS ON, COPY_IN default 1, pad-row mask default on, decode graph ON) | 96,931 / 70.7 | -8.4% / -24.5% | -17.8% / -52.7% | PASS 3600 s: 4177 ok / 4531, 0 err; TTFT p50 0.58 s, intvty p50 139.4; vs dep2_c32 +15.0% / +35.7%; only Tracebacks = SIGTERM teardown; repro lane overlapped 09:05-09:08 UTC |
| 2026-10-06 (mi355-4) | DP2 + TP MoE c32 dp2tp_replay_c32 (fix cb5b0b1cdc + DP bounded replay, uncommitted at run time; REPLAY=1, prefill graph off, clean) | 102,631 / 88.9 | -3.0% / -5.1% | -13.0% / -40.5% | PASS 3600 s: 4337 ok / 4691, 0 err; TTFT p50 0.50 s, intvty p50 151.8; vs no-replay dp2tp_fix_c32 +5.9% / +25.7%; vs DEP2 +21.7% / +70.6% |
| 2026-10-06 (mi355-4) | DP2 + TP MoE c32 dp2tp_replay_c32_pdi16 (b70264d7bc, REPLAY=1, PDI 16, chunk 32768, mem 0.80) | 102,786 / 98.8 | -2.8% / +5.4% | -12.8% / -33.9% | PASS 3600 s: 4334 ok / 4688, 0 err; TTFT p50 0.65 s, intvty p50 156.9; vs PDI4 dp2tp_replay_c32 +0.2% / +11.1% (TTFT 0.50 -> 0.65 s) |
| 2026-10-06 (mi355-4) | DP2 + TP MoE c32 dp2tp_replay_c32_pdi32 (fix tree b70264d7bc, REPLAY=1, PDI 32) | 103,742 / 109.5 | -1.9% / +16.9% | -12.0% / -26.8% | PASS, 4343 ok, 0 err; TTFT p50 0.88 s, intvty p50 161.3 |
| 2026-10-06 (mi355-4) | DP2 + TP MoE c64 dp2tp_replay_c64_pdi4 (PDI 4) | 149,400 / 48.6 | vs tp2r_c64 142,243.2 / 52.0: +5.0% / -6.5% | n/a | PASS, 8339 ok, 0 err; TTFT p50 0.70 s, intvty p50 68.1 |
| 2026-10-06 (mi355-4) | DP2 + TP MoE c64 dp2tp_replay_c64_pdi16 (PDI 16) | 144,746 / 78.5 | vs tp2r_c64: +1.8% / +51.0% | n/a | PASS, 7983 ok, 0 err; TTFT p50 2.73 s, intvty p50 88.9 |
| 2026-10-06 (mi355-4) | DP2 + TP MoE c64 dp2tp_replay_c64_pdi32 (PDI 32) | 123,457 / 98.3 | vs tp2r_c64: -13.2% / +89.0% | n/a | PASS, 6777 ok, 0 err; TTFT p50 11.33 s, intvty p50 113.6 |
| 2026-10-06 (mi355-4) | DP2 + TP MoE c128 dp2tp_replay_c128_pdi4 (PDI 4) | 143,738 / 43.4 | no TP2 c128 ref | n/a | PASS, 9393 ok, 0 err; TTFT p50 28.93 s (saturated), intvty p50 50.5 |
| 2026-10-06 (mi355-4) | DP2 + TP MoE c128 dp2tp_replay_c128_pdi16 (PDI 16) | 116,894 / 73.8 | no TP2 c128 ref | n/a | PASS, 7793 ok, 0 err; TTFT p50 49.33 s (saturated), intvty p50 83.2 |

## aiter MegaMoEV2 tuning mechanism (aiter e7d2453f2, read 2026-10-05)

- No tuned CSV / tuner: kernels/mega_moe/mega_moe_config.py = static "MegaMoEV2 configuration rules for MI355X",
  keyed by token bucket (1..32768), MTPR class (mtpr > 1024 -> MAX_MTPR_CLASS large path with fp8 p2p quant), and only
  two shape thresholds (inter_dim >= 2048 -> tile_n 512; model_dim < 4096 -> block_n 128). Defaults model_dim 7168,
  inter_dim 3072, world_size 8 = DSv4-Pro EP8.
- Fast fixed-slot dispatch only when mtpr <= 255 AND world_size == 8 AND experts_per_rank == 48 (DSv4-Pro EP8).
  DSv4.1 DEP2 = world 2, 192 experts/rank, d5120 i2304 -> never fixed-slot; MTPR 16384 -> large class even at decode.
- AOT bundles (aiter/aot/flydsl/mega_moe.py) only for DSv4-Pro: w8, epr 48/52/56, d7168, i3072, mtpr 8192/16384/32768;
  DSv4.1 compiles online.
=> Rules were derived on DSv4-Pro EP8; other models / EP sizes get the same heuristic, not a tuned choice.

## Decode profile DEP2 (mi355-4 2026-10-05, ctx 65536, bs 8, 40 steps, simulated AL 3.51)

decode_bs_sweep via router (clean, start spread 0.2 s): tok/s per req bs4 198.4, bs8 179.8, bs16 153.0 -- pure decode is
NOT slow (AgentX dep2_c32 live per-req 117, tp2r_c32 182) -> the AgentX gap comes from prefill interaction, not decode.
Router does not proxy /start_profile (404); profile taken on backend 8889 (traces prof_dep2/dep2_be_ctx65536_bs8/).
GPU busy 748 ms / 40 steps = 18.7 ms/step: moe 52% (megamoe_stage1 30.5%, stage2 11.9%, prepare 9.7%, ep_combine 3.7%),
gemm 19%, sparse attn 8%, norm/rope/quant 7%, allreduce 0.4%. Both DP ranks identical.
TP2 (non-DP, same recipe, GPUs 2,3, profiled run): tok/s per req bs4 235.0, bs8 195.4, bs16 157.8 -> DEP2 -16% / -8% / -3%.
Per 40 steps at bs8: TP2 busy 540.7 ms (13.5 ms/step) vs DEP2 748.3 ms (18.7): moe 161.5 + custom AR ar_ll128 45.4 = 207
vs megamoe 389.4 (stage1 228.3 vs TP2 moe1 137.2; stage2 88.7 vs gemm2 71.9; megamoe_prepare 72.4 vs sorting+quant 24);
gemm 145 vs 167, sparse attn 61 vs 63 -> the whole decode gap is MegaMoE (~+180 ms / 40 steps = ~4.6 ms/step).
Suspects: MTPR 16384 puts decode in the large-MTPR class (fp8 p2p quant, compact path); tile_n 512 picked by
inter_dim >= 2048 but DSv4.1 inter 2304 = 4.5 x 512 (ragged last tile); rules never validated for w2 / epr 192.
Load-barrier note: rank 1 (NUMA node 1) loads 260-490 s vs rank 0 ~120 s; node1 MemFree only 183-242 G (page cache
1.18 T) -> THP pre-fault of the ~96 G per-rank engram shard compacts; two servers at once tripped the 480 s barrier
(UNBALANCED_MODEL_LOADING_TIMEOUT_S hardcoded in load_model_utils.py). Run one server at a time on this node.

## MegaMoE microbench (mi355-4 2026-10-05, scripts/bench_megamoe_dsv41.py, EP2 GPUs 0,1, MTPR 16384, uniform route)

ms per MoE call, rank-mean. TP = all 2x tokens through aiter fused_moe with inter/2-sharded experts + RCCL AR
(pessimistic for TP: SGLang uses aiter custom AR). Absolute times ~2x the in-server trace for both, ratios comparable.

| tok/rank | v4.1 mega | v4.1 TP+AR | ratio | v4-Pro-shape mega | v4-Pro TP+AR | ratio |
|---|---|---|---|---|---|---|
| 8 | 0.304 | 0.178 | 1.71 | 0.439 | 0.272 | 1.61 |
| 24 | 0.461 | 0.361 | 1.28 | 0.744 | 0.565 | 1.32 |
| 48 | 0.615 | 0.507 | 1.21 | 1.019 | 0.943 | 1.08 |
| 96 | 0.726 | 0.624 | 1.16 | 1.182 | 1.089 | 1.09 |
| 192 | 1.045 | 0.705 | 1.48 | 1.948 | 1.217 | 1.60 |
| 1024 | 1.209 | 1.167 | 1.04 | 1.932 | 1.936 | 1.00 |
| 4096 | 3.104 | 3.277 | 0.95 | 4.644 | 4.995 | 0.93 |

=> Same relative loss for the DSv4-Pro shape at EP2 -> not DSv4.1-shape tuning; MegaMoE at EP2 decode is 1.2-1.7x the
TP path (designed/fast-pathed for EP8), break-even ~1k tokens/rank, small win at 4k (prefill). Bucket-256 cliff
(sbm 64, s2_bn 128) on both shapes. Overrides tile_n=256 and the bounded-class rule set both GPU-fault (memory access
fault) on a MTPR-16384 instance -> configs are not freely swappable; rule changes need kernel-side support.
Dispatch/combine: tuned_dispatch_combine_intranode.csv has EP8-only rows; DSv4.1 (fp8_ocp, hidden 5120, 192 local
experts) logs "no rule matching shape ... using static geometry defaults" (affects combine, 3.7% of decode).

## Bounded replay under DP (checked 2026-10-05)

vLLM DEP2 runs WITH it: server log DEP2 c16 "Decoder SWA bounded replay: in eager prefill steps, layers 21-39 run on
each request's last 128 tokens only" + "Capturing decoder replay CUDA graphs (PIECEWISE) 57"; CacheConfig
swa_bounded_replay defaults True (vllm/config/cache.py @ac9126e58). SGLang blocks it: arg_groups/deepseek_v4_hook.py:317-335,
"input_ids_global is a DP-wide gather, so the tail slice cannot apply" (also blocks it with any prefill CUDA graph).
Live dep2_c32 at 26 min vs tp2r_c32: TTT -16%, intvty p50 95 vs 149, TTFT p50 612 vs 410 ms, per-req decode 117 vs 182
tok/s, DP load balanced (running 5.3 / 5.0) -> missing replay (long prefills stall both DP ranks) is a prime suspect.

## Problem 1 analysis (mi355-4 2026-10-05 19:10, SGLang affa261e3d + local uncommitted patch)

- Late layers 21-39 (after kv_source 20) hold no engram (engram 1, 14) and no hash-routed MoE, so input_ids_global is
  unused there. The only real DP coupling is the MoE: DP_MOE=tp does dp_gather_replicate / dp_scatter /
  reduce_scatterv in every layer sized by full-extend global_num_tokens (+ global_dp_buffer_len), which the per-rank
  tail breaks. With an a2a backend (MegaMoE) each rank routes its own rows: should_use_mega_moe uses global counts
  (same decision every layer/rank), num_token_non_padded masking is harmless with fewer rows, idle rank -> 0-token
  branch, DSpark prefill only injects tail hidden into draft KV via token_indices (no collective).
- Patch: deepseek_v4_hook.py guard now rejects replay only for DP attention with moe_a2a_backend == none;
  _tail_rows / _scatter_tail_rows bound the single-request contiguous slice by len(token_indices) (DP MAX_LEN padding
  may follow real rows; eager DP prefill is SUM_LEN = no padding). Script: DP_MOE=megamoe now defaults REPLAY=1,
  DP_MOE=tp REPLAY=0. DEP2 replay server passed arg validation; stopped during weight load (user: no DEP for now).
- For DP_MOE=tp: needs per-layer tail token counts across DP ranks (each rank knows only its own tail = sum(min(ext,128)));
  would have to be added to the scheduler's DP all-gather (prepare_mlp_sync_batch) and a second dp buffer len for
  late layers. Not done.

## Problem 3 analysis (mi355-4 2026-10-05)

- Faulting kernel indexSelectSmallIndex<BFloat16,int64,uint,2,2,-2>, 5120 threads / wg 512 = 10 blocks: 2D bf16 source
  row width 5120, <= 16 indices. F.embedding == weight.index_select(0, ids) -> prime suspect: target embed_tokens
  (129280 x 5120, unsharded under DP attn-tp 1, no id masking) hit with an out-of-range token id. DSpark block 5: DP0
  with 3 reqs -> 15 tokens per verify/draft step (fits <= 16). Crashed rank = DP0 (rank1 survived); main thread was in
  process_batch_result_decode synchronize -> decode/verify step, not prefill. aiter e7d2453f2 already has PR 3514
  (custom-AR end_sync), so not that known NaN bug; still check for NaN/garbage sampled ids.
- HIP_LAUNCH_BLOCKING=1 repro (failed_dep2_1005/dp2tp_dbg_blocking, crash again ~6 min): fault surfaces at
  dspark_draft.py resolve_greedy_mask right after draft_model_runner.forward (graph replay) -> the index_select is the
  draft embed (forward_embed, SGLANG_DSPARK_EMBED_IN_GRAPH default on) of draft_block_ids = [bonus, mask x4].
- Debug check SGLANG_DSPARK_DEBUG_IDS=1 (local, dspark_draft._debug_check_draft_ids; failed_dep2_1005/dp2tp_dbg_draftids):
  "bs=1 ids=[[129280, 128799 x4]] bonus=[129280] mode=decode global_num_tokens=[0, 1]" on DP1 -> bonus token ==
  vocab_size (129280), i.e. a sampler "not found" value (NaN/invalid probs), mask 128799 is fine.
- All 3 crashes are preceded by the same trace request "#new-token: 1683, #cached-token: 688640" (~690K ctx) prefill on
  the crashing rank -> suspect that prefill's logits (first sampled token = the bonus). Next check: SGLANG_DSPARK_DEBUG_IDS
  also checks prefill next_token_ids in dspark_worker_v2 (_debug_check_prefill_ids: logits NaN/Inf, prefix/extend lens).
- (19:30-21:55) Prefill ids never bad; bad bonus comes from DSpark VERIFY: target logits rows all NaN (775680 = 6 x
  129280) -> chain_speculative_sampling returns vocab_size. Per-layer probes (local, SGLANG_DSPARK_DEBUG_IDS): the
  OTHER DP rank's rows in the TP-MoE gather buffer are NaN first (layer 0-1), this rank's rows go NaN 1-5 layers later
  (cross-row contamination inside the MoE). Usually the peer is idle (global_num_tokens [0,1]/[1,0]), once [4,1].
  Ruled out: SGLANG_DP_BUFFER_ZERO=1 (zero-init dp buffers) still crashes; count mask on num_token_non_padded was a
  no-op because forward_batch.num_token_non_padded is None on this path (probe col = -1 every layer).
  Why V4-Pro is not hit: same code (models/deepseek_v4.py), but only reached with DP attention + moe_a2a none, which
  V4-Pro recipes never use; forward_batch.moe_num_token_non_padded() returns None by design under DP gather, so this
  path has no padded-row masking at all. V4.1 amplifiers: DSpark (NaN -> vocab_size id -> embedding OOB fault), engram.
  Not verified on V4-Pro.
- 21:56: SGLANG_DP_ZERO_PAD_ROWS=1 = nan_to_num on local rows (<= 2048) before the gather, TAG dp2tp_nan0; stopped at
  warmup 90 s (user shutdown), NO verdict (logs failed_dep2_1005/dp2tp_nan0_stopped).
- 10-06 07:20 rerun (failed_dep2_1005/dp2tp_nan0_local): STILL crashes; every rank's local rows clean at the gather,
  yet the idle peer's slot in the gathered buffer is still 6 NaN rows -> the NaN is not what the peer sends; it is the
  MAX_LEN pad slot itself. This rank's rows then go NaN inside the MoE (moe_out NaN on many layers).
- Existing upstream guard for exactly this: layers/dp_attention.py mask_dp_pad_moe_topk_ids (pad rows -> topk_id -1),
  called only from token_dispatcher/standard.py when NOT use_aiter_moe_runner AND an EP local_expert_mapping exists AND
  SGLANG_OPT_MASK_DP_PAD_MOE -> never on our path (aiter fused_moe, TP-sharded, no EP). Also its counts come from
  global_num_tokens_gpu, which is the PADDED list in both eager (prepare_mlp_sync_batch copies [max]*n) and decode graph
  replay (_global_num_tokens_for_graph) -> real per-rank counts exist only on CPU (ScheduleBatch.global_num_tokens).
  Proper fix needs real counts plumbed to GPU capture-safely, then zero / drop pad rows after the gather.
- 10-06 07:35: SGLANG_DP_ZERO_PAD_ROWS now = nan_to_num_ on the GATHERED buffer (rows <= 4096) after
  dp_gather_replicate (local variant removed). TAG dp2tp_nang -> STILL crashes, and the gathered buffer is fully clean
  (mine_vs_peer empty) while this rank's moe_out goes NaN at layer 4 -> NaN is produced INSIDE the MoE/combine from
  clean input; pad-slot NaN was a co-symptom. DP_MOE=tp sets SGLANG_DP_USE_REDUCE_SCATTER=1 (decode combine = aiter
  custom reduce_scatter, MoE-internal AR skipped), SGLANG_DP_USE_GATHERV=1, SGLANG_DP_SHARED_EXPERT_LOCAL=1,
  SGLANG_SHARED_EXPERT_TP1=1 (script now lets env override each). Suspect: aiter custom reduce_scatter (same class as
  the PR 3514 missing end_sync bug) when the peer is idle.
- 10-06 07:55: A/B one variable: SGLANG_DP_USE_REDUCE_SCATTER=0 (no nan_to_num), TAG dp2tp_nors -> STILL crashes
  (env confirmed in log) -> aiter custom reduce_scatter is not the root cause.
- 10-06 08:10: default config + probes split inside the MoE: shared_local (slot 320+, SHARED_EXPERT_LOCAL output on
  local rows), mlp_out_mine (240+, self.mlp output on this rank's slice, pre-combine), moe_out (160+, post-combine).
  TAG dp2tp_mlp -> shared_local [] and mlp_out_mine [] at every layer, moe_out NaN from layer 4 -> this rank's partial
  is clean; the NaN comes from the PEER's partial for this rank's rows, summed in the combine (reduce_scatter).
  Peer idle again ([1,0]). Hypothesis: idle rank's forward (likely decode graph replay) uses a different gather
  layout / buffer length than the active rank, so the summed partials do not line up.
- 10-06 08:30: A/B one variable: EXTRA_ARGS="--cuda-graph-backend-decode disabled", TAG dp2tp_nograph -> inconclusive:
  dies at once with "AssertionError: engram serves extend, target-verify and decode, not 4" (4 = IDLE) in
  layers/engram.py forward -> eager IDLE forward is unsupported by engram; with the graph on, the idle rank always runs
  a decode-graph replay of a fabricated batch. Suspect geometry mismatch: replay picks bs from _max_dp_batch_size
  (non-ragged) but from the per-rank ragged_layout.graph_num_tokens in ragged verify mode.
- 10-06 08:45: DPGEOM log (decode_cuda_graph_runner replay, same env, dedup per (mode, ragged, raw_bs, bs,
  padded_tokens, global_num_tokens_cpu)), TAG dp2tp_geom -> geometry mismatch REJECTED: idle rank replays the
  TARGET_VERIFY graph with the same bucket as the active rank every time (e.g. active raw_bs=2 bs=2 [12,0], idle
  raw_bs=0 bs=2 padded 12 [12,0]), non-ragged. Idle rank dies only from gloo "connection closed" in the next DP
  all_gather after the active rank raises. aiter fused_moe moe_buf is torch.empty + atomic stage2; num_local_tokens is
  only passed for mori.
- 10-06 09:10: every rank records NaN in its own MoE partial for OTHER ranks' rows (slot 400+); DSpark worker logs
  "OTHERPARTIAL" (outside graphs, max 3x) when set -> shows the idle rank's side. TAG dp2tp_other -> idle DP1
  (mode IDLE): its own pad rows have NaN INPUT but CLEAN partial output (mlp_out_mine []), while its partial for the
  active rank's rows (clean input) has NaN (other_partial) -> rows' outputs look mis-mapped / mixed inside the MoE,
  not plain propagation. moe_routed_quant_stream is CUDA-only (None on ROCm) -> not a side-stream race.
  CAVEAT: the dp2tp_nang "clean buffer" test used nan_to_num_(nan=0) only, which maps +-inf to bf16 max -> not clean.
- 10-06 09:30: rerun with nan_to_num_(nan=0, posinf=0, neginf=0) on the gathered buffer, TAG dp2tp_nang2 -> STILL
  crashes ([4,1] and [1,0]); gathered buffer has no NaN/inf on either rank, yet on BOTH ranks the MoE partial is clean
  on the rank's OWN slice and NaN only on OTHER ranks' slices -> position-dependent, not data-dependent. Rejects
  "NaN router -> bad topk ids -> moe_sorting corruption". MoE quant is per_1x32 (no cross-row scale).
  should_use_dp_reduce_scatterv() is False here (moe_ep_size 1), so the nors run really used AR + dp_scatter.
  New suspect: the gather delivers finite-but-huge garbage in other ranks' slices (own slice is a local copy, others
  come via the collective; hidden is 3D mHC [T, 4, 5120]).
- 10-06 09:50: probe absmax of own vs other slice per layer after the gather (default config), TAG dp2tp_absmax ->
  REJECTED huge-garbage idea: other slice absmax < 5 (normal). Crash here [3,1], DP1 own rows NaN after layer 0 MoE
  -> peer not even idle. MHC post fusion (Blackwell/TP4 only) and HIP moe_mhc_fusion (tp_size==4) are off on TP2.
  So: finite, normal, identical MoE input on both ranks, yet self.mlp output is NaN only on rows outside the
  computing rank's own slice. All decode MoE so far ran inside the decode CUDA graph.
- 10-06 10:05: engram hasher returns zero hash ids for IDLE (local patch, layers/engram.py) so eager idle works; rerun
  with EXTRA_ARGS="--cuda-graph-backend-decode disabled", TAG dp2tp_nograph2 (reading: no crash -> graph-related;
  crash -> MoE path itself, then capture inputs eagerly and replay the single MoE call offline).
  RESULT (10-06 09:30): NO crash. Ready 01:01:53 UTC, warmup 354 done, profiling 21+ min, errors=0, accept len ~3.5,
  both schedulers alive (every previous decode-graph run crashed at ~5-6 min, 12+ repros). => the NaN is tied to the
  decode CUDA graph replay path under DP2 + TP MoE (idle / unbalanced ranks), not the MoE math. Full run COMPLETED:
  profiling 3640 s, 2508 ok / 0 errors / 5 cancelled (grace timeout), no NaN/HSA; the only Traceback is the normal
  SystemExit(0) at teardown. TTT 44,788.7 tok/s/GPU, P90 intvty 23.6 (p50 30.3), TTFT p50 0.68 s -- eager decode is
  slow, this is a diagnostic, not a perf candidate (tp2r_c32 105,800.1 / 93.7).
- Next: find what the decode-graph replay does differently from eager under DP2 + TP MoE: buffers fixed at capture
  (global/local dp buffers, MoE workspace / moe_buf, topk buffers), the global_num_tokens / dp_padding metadata
  written at replay ([num_tokens]*dp), and the idle rank's fabricated IDLE->graph batch.

## Options

- A: `--moe-a2a-backend none` (DSpark-under-DP supported "built-in TP MoE"): DP attention + gather/reduce-scatter MoE.
  No code change; not A2A like vLLM DEP.
- B: patch engram DP lookup to gather ids itself when global_dp_buffer_len is None (all-gather over the TP group
  using global_num_tokens), keeping MegaMoE.
- C: patch engram for the 'shared' host layout under DP: full table in /dev/shm, each rank looks up its own tokens
  locally, no collective (vLLM dp_shared_memory equivalent). Keeps MegaMoE.
