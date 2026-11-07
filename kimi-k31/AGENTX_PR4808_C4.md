# K3.1-Pro AgentX c4: InferenceX PR 4808 recipe vs + SkyOpt flags (MI355X)

Owner node: crsuse2-m2m-v3-131. Started 2026-11-05.

## CONTINUE HERE

**NEXT SESSION TASK (set 2026-11-07 12:05 UTC by the user; owner node crsuse2-m2m-v3-131, re-check `hostname`,
GPUs free):** continue optimizing K3.1 c1 on upstream aiter (mxmoe). State as of now:
- Result: on aiter main 0fa23206b + flydsl 0.4.2 + gate-mode overlay, point `agentx-k31/points/c1-mxmoe-p16-fix`
  (presort16 + fix B + fix C) reaches AgentX c1 P90 intvty 191.7 / 191.5 (old c1-wide-opt 189.4 / 191.4), gsm8k
  within noise (same-config spread ~12/1319; use >= 3 runs). Details: blocks below dated 2026-11-07.
- UNCOMMITTED local work (nothing pushed; ask the user before commit/push/PR):
  (1) sgl-amd-mono `/sgl-workspace/sgl-amd-mono` branch `k31/wide-launch-presort16` (5 files, on top of #111's
      k31/wide-launch-topk): launch BM16 presort "aux16" -> `aiter.presorted_moe.register_presorted_moe`.
  (2) aiter worktree `/sgl-workspace/aiter-main` branch `k31/adaptive-sort-output` (fix B, `aiter/fused_moe.py`
      +21/-6: adaptive aux sort uses a 4-byte-aligned caller output). Local test copies in op_tests/
      (`test_moe_2stage_k31out.py`, `_k31numeric.py`) are scratch, not for commit.
  (3) fix C = kimik3 a8w4 token-16/inter-384 row stage2 -> `flydsl_moe2_layout_afp8_wfp4_bf16_t16x128x128_atomic_nt_sbm16`
      (csv in the point dir; not in aiter yet).
  (4) SGLang bug (not fixed in tree): K3 situ a8w4 w13 is GU-interleaved but moe_runner/aiter.py passes
      gate_mode=SEPARATED -> wrong output with aiter >= #6464 mxmoe rows. Worked around only by the PYTHONPATH
      overlay `/sgl-workspace/k31-gu-overlay/sitecustomize.py` (K31_GU_ITLV_OVERLAY=1; K31_NO_ADAPTIVE_OUTPUT=1
      turns fix B off).
  Pending user decisions: where (1)-(4) go (PRs to fork/upstream, #111 or new PR).
- Next optimization: (A) mxmoe BM16 stage1 under real routing is +1.8 us/call vs c6 (~+0.1 ms/bs2 step, the last
  gap). Must be measured on REAL AgentX load (synthetic random-token traces do not show A): capture with
  `fixBC-20261107/prof.sh`-style runs (points `*-prof`), compare with `agentx-k31/verify_diff.py`; candidates are
  BM16 mxmoe g1 names (tile 16x128/16x256 x256, flags f16in/hpf/nt/fp8out/situv2, xcd0/2/4). Then the earlier
  list: shrink the wide launch's latency stages, bs1 step (B300 gap).
- Tools (all in `/shared_nfs/kk/agentx-k31/`): itl_classes.py (AgentX per-step classes), verify_diff.py,
  copy_ctx.py, prev_kernel.py, stream_overlap.py, gaps.py (one-off 14-29 ms host stalls exist in every trace: compare
  kernel busy, not span), capture_trace_bsN.sh, driver_matrix.sh (AgentX). Never `pkill -f` a pattern that appears
  in your own command line; kill by PID.

**gsm8k1319 noise study: B and C cost no accuracy (2026-11-07 12:00 UTC, crsuse2-m2m-v3-131; GPUs free;
`gsm-recheck-20261107/`; overlay switch K31_NO_ADAPTIVE_OUTPUT=1 turns fix B off):**
```text
config (all new env + overlay + presort16)     correct / 1319 per run      mean
p16 (no B, no C)                               1289, 1281, 1277            1282.3
p16 + B                                        1281                        -
C only (B off)                                 1282                        -
B + C                                          1279, 1274                  -
```
Same-config spread is ~12 questions (p16: 1289..1277), so B and C are within noise. C (nt stage2) is the one kernel
that differs in the server logs and changes greedy paths (answer-text overlap clusters with/without C), but the
aiter op test gives token-16 logits_diff 1.48e-5 with nt vs 1.56e-5 without. Use >=3 runs per config for gsm8k
comparisons at this level.

**AgentX c1 with B+C = c1-wide-opt level (2026-11-07 10:50 UTC, crsuse2-m2m-v3-131; GPUs free;
`agentx-fixBC-20261107/`, point c1-mxmoe-p16-fix):**
```text
                       P90 intvty     ITL p90 ms   concurrent-decode med / mean ms     solo med   tok/s/GPU
c1-wide-opt (old)      189.4 / 191.4  5.28 / 5.22  18.25/20.94, 18.24/20.80            13.91/13.90 2188.7/2193.3
mxmoe before fixes     186.3 / 186.0  5.37 / 5.38  18.59/21.05, 18.65/21.03            13.92/13.93 2187.7/2189.8
mxmoe + B + C          191.7 / 191.5  5.22 / 5.22  18.38/20.65, 18.32/20.62            13.91/13.86 2185.5/2194.3
```
r1 had 1/249 InvalidInferenceResultError (session 91 turn 16: a request whose normal output is 9 reasoning-only
tokens; the same request also fails in old-env c1-draft8; numerically fragile, not a bug); r2 0/249.
gsm8k1319 (graphs) on B+C: 0.9697, trunc 0 (new-env off 0.9735, p16 before fixes 0.9773, old baseline 0.9788).
Per-question (sgl_eval outputs under /root/.sgl_eval/sglang_run_eval/): off->B+C lost 17 gained 12; p16->B+C lost
18 gained 8 (sign test p~0.08); wrong answers are normal `stop` + boxed numbers (reasoning slips, no garbage /
truncation); identical answer text vs off 988/1319 (p16: 1087) -> C's different stage2 kernel changes greedy paths.
Not significant yet; next: rerun gsm8k1319 on B+C (and C-only if it stays ~0.970).

**B fixed + B+C validated (2026-11-07 08:25 UTC, crsuse2-m2m-v3-131; GPUs free):** B = aiter-main branch
`k31/adaptive-sort-output` (local worktree `/sgl-workspace/aiter-main`, uncommitted; `aiter/fused_moe.py` +21/-6:
`_adaptive_moe_sort(output=)` uses a caller output as moe_buf when shape/dtype/device/contiguous/4-byte aligned,
`_moe_sorting_impl` passes it). aiter op test with every test_fmoe call given a -7.0-filled output= buffer
(`op_tests/test_moe_2stage_k31out.py`, local only): kimik3 a8w4 tok16/32 rows 144/144 in-place, 0 copy-back,
logits_diff 1.55e-5 / 1.57e-5 (= before); a4w4 mxmoe row 128/128 in-place, strict accuracy pass; original test
incl. 7 presort contracts pass. Point `agentx-k31/points/c1-mxmoe-p16-fix` (+`-prof`) = c1-mxmoe-p16 + token-16
stage2 atomic_nt csv (C) + B (via PYTHONPATH aiter-main) (`fixBC-20261107/`): quick 8/8 (token counts differ
slightly from before: greedy text changes, answers right); synthetic bs2 196k: copy-after-gemm2 0, VERIFY kernel
busy 17.36 ms (C only 17.44, old c6 17.81). NOTE: every trace has ~1 one-off 14-29 ms host stall (old env too,
e.g. old cap1 14 ms at set_mla_kv_buffer -> sparse_mla); compare kernel busy, not span (`agentx-k31/gaps.py`).
Real AgentX load cap1 (same step mix): kernel busy old 16.09 / mxmoe 16.40 / B+C 16.19 ms -> gap 0.31 -> 0.10 ms;
what is left is A = stage1 mxmoe BM16 under real routing (+163 us/step, +1.8 us/call x 92); stage2 nt now equal
(1455 vs 1447 us). Est. AgentX p90 ~189-190. Next (user to choose): AgentX c1 x2 + gsm8k on c1-mxmoe-p16-fix, or
first real-load stage1 BM16 candidates.

**B and C root-caused (2026-11-07 07:40 UTC, crsuse2-m2m-v3-131; GPUs free):**
- B (copyBuffer +100 us/step): 24 x ~4.2 us per step, one after every DSA layer's routed gemm2
  (`agentx-k31/copy_ctx.py <trace>`). aiter `_moe_sorting_impl` takes the BM16 adaptive (aux) sort with
  "`output` not threaded here" -> `_adaptive_moe_sort` allocates its own moe_buf and `_return_output` copies it
  into the caller's output. KDA layers (presort16, in-place output) do not copy. Fix = aiter: let the adaptive
  sort use a suitable caller `output` (shape/dtype/contiguous/4-byte aligned), as the presort path already does.
- C (non-MoE kernels +5-15 %, ~+150 us/step): same kernels/counts/predecessors, no stream overlap, NOT lamport
  (AITER_CUSTOM_AR_DISABLE_LAMPORT=1 same), NOT xcd4 (stage1 xcd4/none/xcd2/16x128 all same), NOT env (new env +
  c6 row = normal). Cause: the token-16/inter-384 row's stage2 `flydsl_moe2_layout_afp8_wfp4_bf16_t16x128x128_
  atomic_sbm16` lacks `nt` -> expert weights evict L2/MALL and the following DSA-layer attention kernels slow
  down. With `..._t16x128x128_atomic_nt_sbm16` (`xcd-cands-20261107/`, bs2 196k synthetic): DSA kernels back
  to old (Cijk MT128 8.70 -> 8.10, sparse_mla 17.38 -> 16.64 us), stage2 18.56 -> 17.66 us, VERIFY 17.72 ->
  17.44 ms (old c6 env 17.81). Stage1 xcd2 + nt: 17.57 (worse). Only tok 2 and tok 16 inter-384 rows lack nt
  in aiter main's kimik3 a8w4 table. Next: AgentX c1 with token-16 stage2 = atomic_nt (+ quick/gsm check).

**c1 gap mxmoe vs c1-wide-opt is REAL (2026-11-07 05:20 UTC, crsuse2-m2m-v3-131; GPUs free):** AgentX c1 x2 each
(`agentx-rerun-20261107/`): wide-opt P90 intvty 189.4 / 191.4, ITL p90 5.28 / 5.22; mxmoe-p16 186.3 / 186.0, 5.37 / 5.38.
Per-step classes (`agentx-k31/itl_classes.py <profile_export.jsonl>`): solo steps equal (med 13.90 vs 13.92 ms);
concurrent-decode steps med 18.25 / 18.24 (old) vs 18.59 / 18.65 (new) = +0.37 ms -> the p90 gap.
Synthetic bs2/bs3 traces at ISL 300k (`bsN-300k-20261107/`, random tokens) show NO gap (bs2 VERIFY 18.08/18.04 vs
18.10/18.06, bs3 21.27/21.11 vs 21.05/21.10). Traces inside real AgentX load (`agentx-prof-20261107/`, points
`c1-*-prof`, 2 s windows at a bs2 step; `agentx-k31/verify_diff.py old new`): VERIFY 16.22 -> 16.61 (cap1),
15.36 -> 15.59 (cap3). cap1 per step, new - old: (A) routed MoE BM16 under real routing +157 us (stage1 +1.0,
stage2 +0.7 us/layer vs c6 BM32); (B) `__amd_rocclr_copyBuffer` 29 -> 128 us (+100; presumed DSA-layer fused_moe
copying a misaligned atomic output, NOT verified); (C) many non-MoE kernels +5-15 % (Cijk hipBLASLt +26/+19,
sparse_mla +19, hgemm +15, quantize_k +14 ...; also in synthetic traces) ~ +150 us; allreduce lamport -60;
DSA-layer sort/topk -60; GPU idle +0.08 ms. Next: (1) verify/fix B (aligned MoE output for DSA layers),
(2) find why C (new aiter tuned-GEMM table picks?), (3) retune the token-16 mxmoe BM16 row on real routing.

**c1 on upstream aiter (2026-11-07 00:53 UTC, crsuse2-m2m-v3-131; GPUs free):** point `agentx-k31/points/c1-mxmoe-p16`
= c1-base + aiter-main/flydsl 0.4.2/gate-mode overlay PYTHONPATH + `K31_GU_ITLV_OVERLAY=1` + launch top-k + sort
(aux16), default upstream table. bs1 trace (`bs1-mxmoe-20261106/`, same mix): old c1-wide-opt 13.339 ms/step
(DRAFT 1.172 + VERIFY 12.167) vs new 13.407 (1.166 + 12.241), +0.5 %. AgentX c1 (`agentx-mxmoe-p16-20261107/`):
```text
                     tok/s/GPU  ITL mean/p90 ms  P90 intvty  TTFT p50/mean s  hit    errors
c1-wide-opt (old)    2188.7     4.25 / 5.28      189.4       0.49 / 0.69      0.971  0/249
c1-mxmoe-p16 (new)   2187.7     4.24 / 5.37      186.3       0.50 / 0.70      0.971  0/249
```
=> moving to upstream aiter (mxmoe + launch presort) is performance-neutral for c1 (n=1 each; p90 -1.6 % is within
run-to-run spread, bs1 +0.5 %, bs2 step equal 17.8 ms). No c1 gain from the aiter upgrade itself; the value of the
presort16 code is that it keeps c1-wide-opt's level once SGLang moves to aiter >= #6464 (without it: +9.5 us/layer).

**RESULT (2026-11-06 19:09 UTC, crsuse2-m2m-v3-131; GPUs free, nothing running): launch BM=16 presort works on
upstream aiter once the gate layout is right.** All on aiter 0fa23206b + flydsl 0.4.2 + the gate-mode overlay
below, point c1-base, same decode mix every trace (7 x bs2, 2 x bs3); `presort16-20261106/status.txt`:
```text
per KDA layer, bs2                          routed seg us   VERIFY ms/step   gsm8k1319 greedy (graphs)
base: upstream mxmoe row, aiter's own sort  65.3, 65.5      18.29, 18.38     knobs off: 0.9735, trunc 0
p16:  + launch BM16 presort (aux16)         55.9, 56.0      17.95, 17.71     0.9773, trunc 0, 68 x aux16 on
c6 (BM32 pair, no launch sort on new aiter) 63.1, 62.1      18.10, 18.10     -
```
p16 vs base: -9.5 us/layer (-14.5 %), -0.5 ms/bs2 step (-2.8 %); vs c6 -6.6 us/layer. Kernels with the right layout:
stage1 `..._iq_il_...` 33.8 + stage2 18.6 us; the separate `moe_sort_quant` (5.3) and the per-layer copy are gone.
quick.py (8 questions): overlay off-knobs 8/8, p16 8/8 (same token counts as the old env).
**Next (needs the user):** (1) SGLang fix: pass GateMode.INTERLEAVE for situ when w13 is GU-interleaved (else any
aiter >= #6464 table with mxmoe rows breaks K3) -- upstream PR or ask SGLang owners; (2) commit `k31/wide-launch-
presort16` (5 files, uncommitted) and add it to #111 or a follow-up PR once SGLang moves to aiter >= d4defd456;
(3) optional aiter issue: fused_moe should reject mxmoe a8w4 rows under gate_mode SEPARATED instead of computing
garbage. c6 csv row is now moot (p16 beats it on upstream aiter).

**Status (2026-11-06 ~18:20 UTC, crsuse2-m2m-v3-131; superseded by RESULT above; driver `drive2.sh`):** launch BM=16 presort
("aux16") for upstream aiter's mxmoe row implemented on sgl-amd-mono branch `k31/wide-launch-presort16` (local,
UNCOMMITTED, on top of k31/wide-launch-topk): `moe_front.sort_layout/sort_init/sort_rows(kind)` (bm 16, pad m,
m_indices, dest = reverse_sorted, no quant), `route_row` quant decoupled from zeroing, `kda_layer(sort=None|"sgl"|
"aux16")`, `presort.registered()` -> `aiter.presorted_moe.register_presorted_moe` (+ moe_buf_zeroed), auto kind =
"aux16" iff `aiter.presorted_moe` exists, `presort.ref_aux16/check_aux16`, `layer_bench --sort [sgl|aux16]`.
- Verified: layer_bench `--sort aux16` (new aiter env) all PASS incl. vs aiter's own BM16 sort (nv, eids, rows per
  expert, reverse_sorted); `--sort` (sgl, old env) PASS; launch time unchanged (93.8 / 100.4 us). Eager SORT_CHECK
  on served data: 68 x `launch sort aux16 on`, 1000 calls 0 bad, out_rel 5.01e-3 = own-sort self spread 4.98e-3.
- **BLOCKER FOUND: upstream aiter's kimik3 a8w4 mxmoe row is WRONG under SGLang 380ee3f** (gsm8k 0 %, all
  truncated; quick 8-question check `quick.py` 0/8; old 285579b62 table on the same new aiter 8/8; old table + only
  the new token-16 row 1/8). Root cause: SGLang loads K3 situ a8w4 w13 GU-interleaved (`mxfp4.py`
  shuffle_weight_a16w4(.., True)) but passes `gate_mode=SEPARATED` (moe_runner/aiter.py situ branch). Old "gui"
  kernels ignore gate_mode; the mxmoe stage1 honors it (`interleave=(gate_mode == INTERLEAVE)`). aiter's own test of
  that row: INTERLEAVE logits_diff 1.6e-5, forced SEPARATED 0.979. Fix belongs in SGLang (pass INTERLEAVE when w13
  is GU-interleaved) and/or aiter (reject SEPARATED for mxmoe a8w4). => all earlier upstream-row numbers (c6 vs
  upstream 57-64 us, presort16 54.1 us) were measured on a WRONG-layout kernel; redo them.
- Overnight rerun uses an experiment-only overlay `/sgl-workspace/k31-gu-overlay/sitecustomize.py` (PYTHONPATH +
  K31_GU_ITLV_OVERLAY=1; rewrites Situv2 gate_mode SEPARATED -> INTERLEAVE in aiter.fused_moe.fused_moe; nothing
  installed or modified): quick checks off / p16, bs2 traces base(top-k)/p16/c6 x2, gsm8k1319 p16 / off.

**Status (2026-11-06 ~15:00, crsuse2-m2m-v3-131): the commit/push task below is DONE.**
- sgl-amd-mono: branch `k31/wide-launch-topk` pushed to fork, 2 commits on fc61d69: 53339e2 (launch top-k bit-exact,
  `..._WIDE_LAUNCH_TOPK`) + a818fe2 (launch sort/quant/zero + `presort.py`, `..._WIDE_LAUNCH_SORT`). User's call:
  kept `..._WIDE_TOPK_CHECK` / `..._WIDE_SORT_CHECK`; removed `..._WIDE_DUMP_IDS` and `layer_bench --sort-cut`
  (`kda_layer(sort=)` is a bool again, `sort_rows` has no `cut`). Formatted with ruff 0.15.1 / isort (pre-commit pins).
  Re-check after the edits: layer_bench `--rows 12,24 ... --sort` 6/6 sort checks PASS, layers PASS (93.4 / 99.5 us);
  commit-1 tree alone (no --sort) PASS too. **PR: sgl-project/sgl-amd-mono#111** (base main, stacked on #108 which is
  still OPEN -> its 2 commits show in #111 until #108 merges; then rebase/retarget if needed).
- aiter: tuner patch reverted (`gemm_moe_tune.py`), other pre-existing dirty files untouched. c6 row committed on
  fork branch kkHuang-amd/aiter `kimik3-a8w4-m16-inter384-c6` (473fbea98, base 285579b62 = what was tested; no PR).
  CAUTION: upstream main 0fa23206b retuned that row to `flydsl_mxmoe_g1_a8w4_16x256x256_f16in_nt_fp8out_situv2_xcd4`
  (block_m 16, fused). **Measured 2026-11-06 15:33 (crsuse2-m2m-v3-131), both on upstream aiter 0fa23206b
  (worktree `/sgl-workspace/aiter-main` + flydsl 0.4.2 in `/sgl-workspace/pydeps-flydsl042`, PYTHONPATH only; the
  installed aiter/flydsl untouched), c1-base + launch top-k/sort, bs2 trace each; results + `run.sh`/`run2.sh` in
  `/shared_nfs/kk/results/Kimi-K3.1-Pro/moetune-m12-upstream-20261106/`:**
  ```text
  per KDA layer, bs2      routed seg us   VERIFY ms/step   stage1 / stage2 / sort us
  c6   (2 runs)           57.0, 58.3      17.78, 17.80     35.1-36.1 / 18.3-18.6 / (no separate kernel seen)
  upstream (2 runs)       64.3, 61.4      18.44, 18.11     31.8-33.2 / 16.7-17.8 / sort_quant 5.3-5.5
  ```
  (run `up` discarded: its trace window had 10 bs1 steps.) c6 wins by ~5 us/layer, ~0.5 ms/step (-2.6 %), but the
  upstream GEMMs themselves are faster; it loses on aiter's separate `moe_sort_quant` (block_m 16).
  NEW FINDING: on upstream aiter the launch sort is NOT used for either row (`launch sort NOT used` x68):
  `presort.py` wraps `aiter.fused_moe._moe_sorting_impl`, which the new a8w4 path no longer calls. So PR #111's
  sort lever is a no-op once aiter is bumped -> needs a hook for the new path (and block_m 16) before/when SGLang
  moves to aiter >= #6464. With that, upstream row - sort_quant ~= 57 us ~= c6.
  **Feasibility of feeding upstream mxmoe (2026-11-06, crsuse2-m2m-v3-131): yes, via aiter's official presort API**
  `aiter/presorted_moe.py` `register_presorted_moe(aux)` (d4defd456, 2026-11-06): BM=16 only; keys sorted_token_ids,
  sorted_weights, sorted_expert_ids (1 per 16 rows), num_valid_ids[2], m_indices (sorted row -> token, pad = M),
  reverse_sorted (M*topk: (token,slot) -> sorted row) + topk_ids_ptr/topk_weights_ptr/M/block_m/num_experts/topk;
  `moe_buf_zeroed` + `moe_buf_ptr` skip fused_moe's zeroing (graph-safe if our zeroing is in the captured launch).
  Launch changes: `sort_rows` block 32 -> 16 (SORT_BM param), emit m_indices (ids & 0xffffff) and reverse_sorted
  (= our `dest`), drop `_quant8` for this path (stage 1 `iq_sep_native_scale` quantizes inline); wide.py registers
  instead of presort.py's wrapper; keep `out` 4-byte aligned. Upstream seg kernels: stage1 31.8 + stage2 16.7 +
  sort_quant 5.3 + copyBuffer 4.1 (x1.11/layer; presumed atomic-GEMM2 misaligned-output copy, not verified).
  Estimate 61.4 -> ~52 us/layer (-5.5 vs c6, ~-0.37 ms/bs2 step). Blocker: SGLang must move to aiter >= d4defd456
  (+ flydsl 0.4.2). Posted on PR #111 (issuecomment-6688843616).
- Next: optimization candidates (i)/(ii) in the 14:00 Status paragraph below.

**[DONE] NEXT SESSION TASK (set 2026-11-06 14:05 by the user): commit + push the lever-(a) work below, then continue
optimizing (candidates in the Status paragraph). Owner node crsuse2-m2m-v3-131; re-check `hostname` first.**
1. sgl-amd-mono (`/sgl-workspace/sgl-amd-mono`, editable install = this tree), branch `k31/wide-launch-topk`, on top of
   fc61d69 = the PR #108 branch `k31/fix-warmup-bank-oob` (base e52803f). Remote `fork` = github.com/kkHuang-amd/sgl-amd-mono.
   Uncommitted: M `kimi_k31/gfx950/layer.py`, `kimi_k31/gfx950/stages/moe_front.py`, `kimi_k31/wide.py`,
   `test/kernels/kimi_k31/layer_bench.py`; new `kimi_k31/presort.py`. Suggested 2 commits:
   (a) launch top-k bit-exact to SGLang + routed experts on the launch's ids/w (`_tie_rank`, `_sigmoid_aiter`, div_rn
       weights in route_row; `_routed` / `_experts` / `_launch_topk_ok`; knob `SGLANG_AMD_MONO_K31_WIDE_LAUNCH_TOPK`);
   (b) launch sort + mxfp8 quant + output zeroing (`sort_layout`/`_quant8`/`sort_rows`/`place_codes`/`sort_init`,
       `kda_layer(sort=)`, `sort_buf` arg, `presort.py`, `layer_bench --sort`; knob `..._WIDE_LAUNCH_SORT`).
   Debug knobs inside (default off): `..._WIDE_TOPK_CHECK`, `..._WIDE_SORT_CHECK`, `..._WIDE_DUMP_IDS`,
   `layer_bench --sort-cut` -- ask the user whether to keep them in the PR. Before committing: quick re-check
   `torchrun --nproc-per-node 8 layer_bench.py --rows 12,24 --L 6 --nvb-a 4 --nvb-m 4 --up-norm --front-map --sort`
   (in test/kernels/kimi_k31; expect every "sort check ... PASS", no FAIL).
   Rules (NEW_WORKSPACE_PROMPT): NO "Co-authored-by" or any AI trailer (if injected, build with
   `git write-tree` / `git commit-tree` / `git update-ref`; check `git log -1 --format=%B`); `git fetch` before push;
   never force-push; ask before opening a PR (title `[AMD][K3.1][k/N] ...`, template sections only).
2. aiter (`/sgl-workspace/aiter`, detached 285579b62): ONLY local change of this work is the unused tuner patch in
   `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` (TUNE_MOE_TOPK_FILE) -> revert (`git checkout -- <file>`);
   the tree had other unrelated dirty files before this work -- do not touch them.
   Optional upstream: the c6 kernel pair for the kimik3 a8w4 token-16 Situv2/inter-384 row (ask the user); today it
   is served from `agentx-k31/points/c1-wide-opt/kimik3_a8w4_tuned_fmoe_m12c6.csv` via AITER_CONFIG_FMOE.
3. claude-skills: commit this doc (main is ahead 2 of origin and `git pull --rebase` refused with this file dirty):
   commit -> `git pull --rebase` -> push. Other untracked agentx/ files are not from this work.
Resume prompt: "讀 /workspace/claude-skills/NEW_WORKSPACE_PROMPT.txt 並遵守，用中文回答。讀
/workspace/claude-skills/kimi-k31/AGENTX_PR4808_C4.md 的 CONTINUE HERE，做 NEXT SESSION TASK。"

**Status (2026-11-06 14:00, crsuse2-m2m-v3-131): GPUs free, nothing running. Lever (a) done and measured, NOT
committed** (sgl-amd-mono branch `k31/wide-launch-topk`, 4 files + new `presort.py`; aiter local tuner patch only):
launch top-k made bit-exact to SGLang (`..._WIDE_LAUNCH_TOPK=1`), launch sort/quant/zero (`..._WIDE_LAUNCH_SORT=1`),
both default off, plus the M=12 kernel pair c6 via `AITER_CONFIG_FMOE` (point `c1-wide-opt`). AgentX c1: P90 intvty
171.2 -> 189.4, ITL p90 5.84 -> 5.28 ms, bs2 step 20.5 -> 19.0 ms mean; gsm8k1319 0.9765 vs 0.9788 (noise).
B300 is 241 / 4.15 ms: bs2 is still +31 % over bs1 (wide launch ~96 us/layer with ~45 % latency stages: KDA
recurrence 11.6, MoE tail 13.2, seams/RS/route ~18 us; stage-1 MoE ~34 us at ~4.7 TB/s). Next candidates:
(i) shrink the wide launch's latency stages (AttnRes seam prefetch, KDA recurrence CTA parallelism);
(ii) bs1 step 13.9 ms itself (the B300 gap is also ~8 % at bs1);
(iii) upstream: c6 row for aiter's kimik3 table (tuned on served routing), launch top-k/sort PR to sgl-amd-mono.**

**Status (2026-11-06 04:40, crsuse2-m2m-v3-131): GPUs free, nothing running.** Mono warmup-bank fix found,
verified and upstreamed (sgl-project/sgl-amd-mono PR #108; fork branch kkHuang-amd `k31/fix-warmup-bank-oob`,
checked out in `/sgl-workspace/sgl-amd-mono`, editable install = the fixed code). c1 after the fix:
6 tok 2186.4 / 8 tok 2222.7 tok/s/GPU (tie). c4: `--max-mamba-cache-size 150` +9.8 % (n=1).
**NEXT TASK (new session): close the c1 decode per-step gap to B300 SGLang.**
```text
c1 (AgentX, CI-simulated acceptance)  tok/s/GPU  P90 intvty  ITL mean/p90 ms  SIM acc  step = ITL mean x acc
MI355X 6 tok, fixed (this node)       2186.4     171.2       4.32 / 5.84      3.43     14.8 ms
MI355X 8 tok, fixed (this node)       2222.7     170.3       4.23 / 5.87      3.68     15.6 ms
B300 SGLang c1 (CI 41932307764)       2370.0     241         3.71 / 4.15      3.68     13.7 ms
```
- Cache hit is already equal (0.971 vs 0.974): the gap is decode. Per step ~+8 % (6 tok) / +14 % (8 tok) vs
  B300 at mean, and much worse in the tail: ITL p90 +41 %, P90 intvty -29 %. The tail is the bigger lever.
- Known per-step breakdown at 6 tok, c1, 196 k ctx (trace, TP0, section "Why 8 draft tokens cost more"):
  DRAFT 1.17 ms + VERIFY 12.27 ms; VERIFY = mono k2 launches 7.53 (MoE) + KDA 2.33 + sparse MLA 1.27 +
  DSA indexer 0.52 + gemm 0.33 + rest. Ruled out for the tail: `--prefill-decode-interval 16` / lpm (null).
- B300 c1 flags that differ (CI job log `/shared_nfs/kk/agentx-k31/ci/b300_job_*.log`): no pdi / lpm,
  `--mamba-radix-cache-strategy extra_buffer`, `--mamba-track-interval 256`, `--mamba-ssm-dtype float32`,
  `--mamba-max-states-per-path 1`, HiCache 230 GB, max-running 2, cuda-graph-max-bs-decode 2.
- Suggested first steps: (1) split ITL into steady decode vs steps that share a GPU slot with prefill chunks
  (per-request ITL series from `aiperf_artifacts/profile_export.jsonl` of `agentx-fix-20261106/c1-base`, and
  server.log batch timestamps) to see what the p90 tail is; (2) per-step roofline of the 7.5 ms mono MoE part
  (weight bytes / HBM BW) to see how far from bound it is; (3) apply skill perf-bottleneck-attribution before
  calling anything "bound".
**ITL p90 tail = concurrent decode, not prefill (2026-11-06, crsuse2-m2m-v3-131, c1-base 6 tok, aiperf per-chunk
series `aiperf_artifacts/profile_export.jsonl`, 248 profiling reqs):** "c1" is not bs1 -- 43/249 requests are
subagents (`agent_depth=1`), so 40 requests overlap another. ITL p90 is per-request; the >=p90 bucket (25 reqs)
has 22 overlapping requests, chunk median 20-21.5 ms vs 13.8 ms (<p90). Chunk intervals (1 chunk = 1 spec step,
~3.45 tok) split by what else is active:
```text
solo (bs1 decode)        n=123737  sum 1728 s  med 13.86  p90 14.34  mean 13.96 ms
2 requests decoding      n= 11904  sum  244 s  med 20.06  p90 21.70  mean 20.53 ms   (+47 % per step)
other request prefilling n=   529  sum   34 s  med 19.22  p90 184    mean 63.4 ms
```
=> lever #1 for the tail is the **bs2 decode step (20.5 ms vs 14.0 ms bs1)**; prefill stalls are only 34 s of 2006 s.
bs2 is inside the captured decode graphs (`--cuda-graph-max-bs-decode 4`; B300 uses 2 / max-running 2), so it is
not an eager fallback. B300 p90/mean ITL = 1.12 vs ours 1.35 -> their bs2 penalty is much smaller. Likely suspect: MoE
at 12 vs 6 tokens touches ~2x distinct experts (weight-bytes bound), plus the KDA/MLA kernels not scaling with bs.
**intvty p90 target feasibility** (intvty p90 = 1/ITL p90; replay of the measured chunk series with per-class
step scaling, "now" reproduces 5.86 ms): bs2 17 ms -> 206; bs2 15.5 ms -> 226; bs2 = bs1 13.96 ms -> 240;
bs1 -7 % and bs2 = bs1 -> 259. Prefill stalls do not matter (capping them at 30 ms changes nothing).
B300 (acc 3.68): mean step 3.71x3.68 = 13.7 ms, p90 4.15x3.68 = ~15.3 ms -> their concurrency penalty is ~+10-25 %, ours +47 %.
So 241 at 6 tok needs bs2 ~= bs1, which is unlikely for a weight-bound MoE (12 vs 6 tokens). A realistic target is
bs2 ~15.5 ms -> ~225, and the rest has to come from a cheaper bs1 step.
**bs2 trace DONE (2026-11-06 04:58, crsuse2-m2m-v3-131; TP0 summary in `trace-bs2-20261106/summary.txt`):**
bs2 step = DRAFT 1.22 + VERIFY 19.20 = 20.4 ms (matches AgentX 20.5). At R>=2 requests mono does NOT run the
fused narrow path: `kimi_k31/spec.py` sends `rows > wide.NARROW` to `wide.Wide.backbone` (persistent
`k31_kda_layer_m12` per KDA layer + SGLang routed experts `mfma_moe1_*` / `opus_moe_stage2_a8w4` / `moe_sorting`).
```text
VERIFY ms/step         bs1 (narrow mono)   bs2 (wide)   delta
KDA                    2.33                6.57         +4.24  <- biggest (2.8x for 2x rows)
MoE + o_proj layer     ~6.6 (k2 launches)  ~9.1 (moe 6.11 + gemm 1.53 + comm/attn_res/norm ~1.2 + other)  +2.5
DSA indexer            0.52 (+logits 0.49) 1.25 (+0.86) +1.1  (2 contexts, expected ~2x)
sparse MLA             1.27                0.72         -0.55 (category split differs, check)
```
CORRECTION: the table above compares different scopes. The wide `k31_kda_layer` is a WHOLE-layer launch (MoE tail ->
AttnRes -> in_proj -> KDA ring -> o_proj -> RS -> MoE front/shared/latent up; only routed experts are SGLang),
while narrow `k3_mono_kda_pre` is in_proj+conv+KDA only (o_proj..MoE is in `k3_mono_k2`).
**KDA-layer roofline (2026-11-06, crsuse2-m2m-v3-131; bytes/rank from safetensors headers, TP8, MI355X HBM peak
8 TB/s; per-call times from TP0 traces, script `/shared_nfs/kk/agentx-k31/kfam.py <trace.gz> <n_verify_steps>`):**
```text
per KDA layer, per rank            bytes (MB)                         us/call   achieved   % of 8 TB/s
bs1 kda_pre (in_proj q,k,v,g,f,b)  ~89                                34.5      2.6 TB/s   32 %
bs1 k3_mono_k2 (o_proj..MoE)       ~270 (22 o_proj + 47.5 shared/latent/gate + ~201 routed*)  75.0  3.6 TB/s  45 %
bs2 k31_kda_layer (whole layer)    ~145 (94 attn + 47.5 non-routed + ~3 KDA state x2 req)     96.5  1.5 TB/s  19 %
bs2 routed MoE (moe1+stage2+sort)  ~383 (~175 experts* x 2.19 MB)     66.4      5.8 TB/s   72 %
* distinct experts = uniform-routing estimate 896*(1-(1-16/896)^n), n = 6 / 12 tokens -> 92 / 175; correlated
  draft tokens give fewer, so the MoE % is an upper bound. Not measured.
```
Per layer bytes: KDA attn bf16 751.6 MB/8, shared experts 264 + latent up/down 103 + gate 13 MB /8, one routed
expert 17.5 MB/8. KDA state per rank 12 heads x 128 x 128 x fp32 = 786 KB per request (read+write; more if the
L verify intermediates are written).
=> The wide KDA launch moves ~145 MB in 96.5 us = 19 % of peak HBM. It is NOT bandwidth-bound and the KDA state is
too small to matter, so "state-bound" is ruled out. Cause not measured yet (candidates: in-launch cross-rank sync
(RS / seam / push stages), GEMV tiling at 12 rows). The routed MoE is already close to the limit.
[SUPERSEDED by the stage timeline below: narrow is not faster for this work] Target: at the narrow path's efficiency (~3.3 TB/s) the launch would take ~45 us -> -51 us x 68 = -3.5 ms/bs2 step
-> bs2 ~16.9 ms -> intvty p90 ~210 (replay table above). At ~6 TB/s (~25 us) -> -4.9 ms -> bs2 ~15.5 -> ~226.
**Wide-launch stage timeline (2026-11-06, crsuse2-m2m-v3-131; logs `/shared_nfs/kk/results/Kimi-K3.1-Pro/
wide-timeline-20261106/nvb{1,4,8}.log`):** standalone bench matches production (bench 86.7 / 93.6 / 104.1 us at
nvb 1/4/8 vs trace 89 / ~96 / 106). Repro (in `sgl-amd-mono/test/kernels/kimi_k31`, 8 GPUs, no server):
`torchrun --nproc-per-node 8 layer_bench.py --rows 12 --L 6 --nvb-a 1 --nvb-m 1 --up-norm --front-map --timeline`.
Critical path = "last" CTA column deltas, nvb 1 (nvb 8 in brackets); bytes/rank estimated:
```text
stage            us (nvb8)    bytes       note
t1 push + up     13.2         ~6 MB       previous layer's MoE tail (latent down + push to peers)
t2 seam           5.6 (13.5)  ~0          AttnRes block reads: +~1.1 us per block -> latency, not bytes
t3 in_proj       16.7 (18.3)  ~90 MB      ~5.4 TB/s = efficient
kda recurrence   11.6         ~3 MB       L=6 serial recurrence, latency
t4 o_proj         7.1         ~22 MB      ~3.1 TB/s
t5 rs + seam      7.4 (15.1)  ~0          cross-rank reduce-scatter + MLP AttnRes
t6 front         10.8         ~35 MB      ~3.2 TB/s (shared gate_up + latent up + gate)
route + sdown    11.1         ~11 MB      routing, shared down
total            83.5 (102)   ~145 MB
```
=> Only in_proj runs near HBM speed. About 45 % of the path is latency-type stages with almost no bytes:
KDA recurrence 11.6, MoE tail 13.2, seams/RS/route ~18 us (nvb 8: ~33). The small GEMVs reach ~3 TB/s.
**Re-attribution of the bs2 penalty (per KDA layer):** bs1 narrow = kda_pre 34.5 + k2 75.0 = ~110 us, and that
INCLUDES routed experts (fused into k2). bs2 = wide launch ~96 + routed MoE 66.4 (moe1 39.6 + stage2 19.4 +
sorting 7.4) = ~162 us. The wide launch is NOT slower than narrow for the same non-routed work. The +52 us/layer
(x68 = 3.5 ms/step, matches) is mostly the routed experts running serially as separate kernels after the
launch, with nothing overlapping them.
Levers, roughly ordered by payoff/effort:
  (a) overlap or fuse routed experts with the wide launch (prefetch next in_proj weights during MoE, or a fused
      wide routed stage as narrow k2 does): up to ~-50 us/layer -> bs2 ~17 ms -> intvty p90 ~206-215; big job.
  (b) drop `_moe_sorting` (7.4 us x 92 = 0.68 ms/step) if the launch's own `route` stage output can feed moe1.
  (c) AttnRes seams scale +1.1 us/block (latency): prefetch blocks earlier -> up to ~-8 us/layer at avg nvb ~4.5.
  (d) KDA recurrence 11.6 us on 12 heads x 2 req: check CTA parallelism (latency-bound, few CTAs?).
**User chose (a); plan `/root/.claude/plans/a-vivid-wreath.md` (gated: measure ceiling -> step 1 -> step 2 fusion).**
**Step 0 MEASURED (2026-11-06, crsuse2-m2m-v3-131): distinct routed experts per layer, real prompts.** Eager server
(`--disable-cuda-graph`) + env-gated hook `/shared_nfs/kk/agentx-k31/moehook/sitecustomize.py` (wraps
`KimiK3MoE._forward_routed`, re-runs `self.topk`, skips while a stream is capturing: the prefill breakable graph
still captures with --disable-cuda-graph). Runner `moehook/run_count.sh`, client `moehook/client.py`: 2 concurrent
chat requests over sglang source files, 1024 out. Data `/shared_nfs/kk/results/Kimi-K3.1-Pro/moe-distinct-20261106/count.csv`.
```text
rows 12 (bs2 verify), 88136 calls, 92 layers: distinct mean 114.2 (p10 86, p90 139); per 6-row request 62.2
uniform-routing estimate was 175 / 92 -> draft tokens share experts heavily
```
Corrected routed MoE roofline per KDA layer at bs2 (whole segment between adjacent wide launches = 72.9 us, all
kernel time, zero idle under graph):
```text
aiter topk_reg_kernel   6.5 us   duplicate of the launch's own route stage
_moe_sorting            7.4 us
mfma_moe1 (w13 + fp8q) 39.6 us   ~166 MB -> 4.2 TB/s
opus stage2 (w2)       19.4 us   ~83 MB  -> 4.3 TB/s     (bytes = 114 experts x 2.19 MB/rank, split 2:1)
```
=> Routed MoE is at ~47 % of peak HBM, not 72 %. Ceiling of the routed segment ~28 us/layer (topk 6.5 + sort 7.4 +
MoE kernels to ~5.5 TB/s ~14) -> x68 ~1.9 ms/bs2 step -> intvty p90 ~171 -> ~190. Full fusion (step 2) cannot beat
that floor: the routed inputs need the route and the next launch needs the output, so nothing overlaps. Step 2
DOWNGRADED in favour of: step 1 (launch topk), sort from the launch (or aiter PR 6484, NOT in aiter 285579b62:
`fused_moe` has no precomputed-sort arg), aiter MoE M=12 kernel tuning (moe1 uses a 32-row tile).
**Step 1 (launch top-k) IMPLEMENTED, not committed (2026-11-06, crsuse2-m2m-v3-131):** sgl-amd-mono branch
`k31/wide-launch-topk` (off `k31/fix-warmup-bank-oob`), `kimi_k31/wide.py`: `_routed()` replaces
`pk.moe._forward_routed(...)` in the backbone loop; with `SGLANG_AMD_MONO_K31_WIDE_LAUNCH_TOPK=1` (default 0) it feeds
the launch's `bufs["ids"]/["w"]` through `build_precomputed_topk_output` to `moe.experts` (same a8w4 / route-quant
handoff / zero-copy steps as `_forward_routed`). Eligibility `_launch_topk_ok()` logs `launch top-k on|off (layer N)`
once per layer with the failing checks.
PITFALL hit: SGLang's `precomputed_topk_postprocess_is_noop` starts with `_is_cuda` -> always False on ROCm, so the
first A/B ran the knob as a silent no-op (gsm8k "on" was really off). Replaced by explicit HIP checks (no shared
slots, no EPLB remap, no megamoe, no routing sim, STANDARD/None format); on HIP `_post_process_topk_ids` is a no-op
for K3's call (num_token_non_padded None). ALWAYS grep `launch top-k on` (expect 68 lines) before reading an A/B.
Results `/shared_nfs/kk/results/Kimi-K3.1-Pro/step1-launch-topk-20261106/` (runner `agentx-k31/run_step1.sh`,
KNOBS / NUM / TAGSUF / NOTRACE env; segment tool `agentx-k31/kda_seg.py <trace>`):
```text
                         knob 0 (baseline trace)   knob 1
KDA->KDA routed segment  76.2 us                   68.7 us   (-7.5; aiter topk_reg_kernel 6.4 us gone)
VERIFY GPU span (bs2)    19.20 ms                  18.91 ms  (-0.29, one trace each)
gsm8k200 greedy          0.99                      0.98 (knob verified on: 68 lines)
gsm8k1319 greedy         0.9788 (max out 3049)     0.9742, 1 req hit 32768 max_tokens (truncated 0.00076),
                                                   out tokens 290k vs 219k; knob verified (68 lines)
```
OPEN RISK: -0.46 pt (~0.8 sigma, 2 runs) + one runaway gen with knob on; not proven equal. Proposed check (deferred by
the user 2026-11-06, go straight to sort removal): eager hook comparing launch ids/w vs `self.topk` on the same logits
at bs2 (expect 0 mismatched rows, w diff <~1e-6). Sort removal also consumes the launch's routing, so do it before
trusting any accuracy number of the sort work.
**Sort removal, investigated (read-only, 2026-11-06):** the 7.4 us `_moe_sorting_small_kernel_sorted` is SGLang's
Triton kernel (`sglang/kernels/ops/moe/moe_sorting_small.py:384`, installed into aiter by
`apply_aiter_small_moe_sort_patch` in `moe_runner/aiter.py:172`). For this shape it does THREE jobs:
  (1) the BM=32 sort: experts ascending, pair order kept, each expert padded to x32. `sorted_ids` (slot<<24)|token,
      pad (16<<24)|M; `sorted_weights` (pad 0); `sorted_expert_ids` per block; `num_valid_ids` = [blocks*32, M];
      max_padded = M*16 + 896*32 - 16;
  (2) EMIT_MX: the stage-1 mxfp8 quant of `lat` (fp8 [M,3584] + swizzled e8m0 scales per sorted row, :309, :366-369);
  (3) zeroing the MoE output (stage2 `opus_moe_stage2_a8w4` accumulates with atomics).
Stage1 (FlyDSL `mixed_moe_gemm_2stage_common.py`) does no sorting ("_fp8q_sort" = it writes its fp8 output and
scales in sorted-row layout). aiter `fused_moe` has no presorted arg; `fused_moe_2stages` (:4552) takes the sort
tensors + `prequantized_a1` directly. PR 6484 not local (and it is BM=16). Cheapest hook: a ContextVar in SGLang's
`sorting_impl_wrapper` (it already short-circuits the quant via `_emitted_quant`) that returns launch-made buffers.
Replacing it means the wide launch, after `route`, must: grid-sync, count per expert, write the 4 sort tensors,
the fp8 latent + swizzled scales, and zero `out`. Ceiling 7.4 us/layer (~0.5 ms/bs2 step); net ~5-6 us after the
moved work. This is a FlyDSL kernel + integration change whose bugs show as accuracy loss.
**Step 2 (launch sort) IMPLEMENTED, not committed (2026-11-06, crsuse2-m2m-v3-131; plan
`/root/.claude/plans/a-vivid-wreath.md`):** knob `SGLANG_AMD_MONO_K31_WIDE_LAUNCH_SORT=1` (needs `_LAUNCH_TOPK=1`,
default 0), buckets m <= 64 only (SGLang small-sort range), used only when the step fills its bucket (rows == b;
else SGLang's sort runs, correct either way). Same branch `k31/wide-launch-topk`. Pieces:
- `gfx950/stages/moe_front.py`: `route_row` (sort builds) also mxfp8-quantizes its row (`_quant8`, bit-exact copy
  of SGLang `_sorted_mx_quant`), writes token-major codes, zeroes the row of the MoE output (= `sh_out - m*3584*2`,
  wide's flat ping layout), stores ids/w at device scope, counts the row in a `cnt` word. `sort_rows` (one CTA,
  bid BLOCKS-m-1): resets the previous launch's pair rows to pad, waits for m rows, 64-bit token mask an expert
  (LDS atomic_or) -> counts -> block scan -> expert ids, real pairs (stable rank = popcount of lower tokens),
  publishes `dest`, puts mailbox `srtmk`. `place_codes` (32 idle CTAs below it): scatter the e8m0 codes.
  Buffer resting state = all pad (`sort_init`), so no pad pass.
- `gfx950/layer.py`: `kda_layer(sort=...)` (key + name `_srt`), new kernel arg `sort_buf`, counters +2 words,
  plan region `srtmk`. `sort` int: 7 = full, 1..5 = debug cut points (`layer_bench --sort-cut`).
- `wide.py`: `_launch_sort(b)`, `bufs["sort"]/["sortv"]`, `_routed(..., sortv)`; `presort.py` wraps
  `aiter.fused_moe._moe_sorting_impl` (re-wraps if SGLang re-applies its patch), checks shapes / output / ids
  pointer, else falls back and logs `launch sort skipped: <why>`. Debug `SGLANG_AMD_MONO_K31_WIDE_TOPK_CHECK=1`
  (eager): launch ids/w vs `self.topk`, summary every 2000 calls.
PITFALLS: (1) one CTA issuing ~20 K scattered byte stores cost +11 us (and the pad pass +8 us): moved the codes to
32 helper CTAs and replaced the pad pass by the reset invariant; (2) an LDS storage object cannot enter a dynamic
`if` (peek outside); (3) qout's i32 element index is byte/4.
Unit check `layer_bench.py --sort` (bit for bit vs SGLang `_run_small_sort` on the launch's own ids/w/latent:
num_valid, sorted ids/w/expert ids, every fp8 byte, real rows' codes, output zeroed; plus a 2nd pass with rotated
buffers = the reset path): ALL PASS at m 12/24/48 (L6) and 16/32/64 (L8). Launch cost (nvb 4, us):
```text
L6  m12 92.1 -> 93.1   m24 98.5 -> 99.1   m48 111.1 -> 112.7
L8  m16 95.6 -> 96.5   m32 104.0 -> 104.5 m64 120.7 -> 122.4   (vs ~7.4 us SGLang sort kernel removed)
```
Logs `/shared_nfs/kk/results/Kimi-K3.1-Pro/step2-launch-sort-20261106/` (mat_L*_*.log, cut*.log, tl_*.log).
Server checks (`agentx-k31/run_step2.sh`, SKIP_A / SKIP_B):
- A (eager, `SGLANG_AMD_MONO_K31_WIDE_TOPK_CHECK=1`): launch top-16 vs `self.topk` on the same logits over 49.6 M rows:
  **1513 rows (0.003 %) pick different experts**, w diff <= 1.8e-7 elsewhere -> the step-1 launch top-k is NOT
  exactly SGLang's (likely near-tie order; explains its small gsm8k delta). Open item. Eager gsm8k is far too slow
  (33 q in 79 min): stopped A by hand.
- First B (graphs, gsm8k1319): **acc 37 %, 56 % truncated (runaway gens, accept 6.00)**. Cause: `int(True) == 1`: wide
  passed sort=True and the debug cut level made the stage stop after its wait -> sort buffers never written. The
  bench never hit it (it passed `--sort-cut 7`). Fixed: `kda_layer` normalizes `sort is True -> 7` before the build
  key; layer_bench now passes True like wide.
- Production A/B (`SGLANG_AMD_MONO_K31_WIDE_SORT_CHECK=1`, eager, `agentx-k31/run_sortcheck.sh`): on served data the
  launch's sort / quant / codes are bit-identical to SGLang's kernel, output zeroed, and the experts' output with the
  launch sort vs SGLang's: out_rel 5.07e-3 == SGLang vs itself 5.06e-3 (stage-2 atomics), 200 calls, 0 bad.
- LESSON: the unit test compared buffers only; the bug was in how production builds the kernel. Always run the
  production-path A/B check (same call path, both outputs) before an accuracy gate.
- B rerun (graphs, knobs on, gsm8k1319 greedy): **0.9742**, truncated 0, max out 5859, 68 x `launch sort on`, 0 skipped
  -- identical to the step-1 (launch top-k only) score 0.9742; baseline (both off) 0.9788. The sort adds nothing; the
  -0.46 pt sits with the launch top-k's 0.003 % routing mismatches (open item above).
- C bs2 trace (`step2-launch-sort-20261106/trace`, `kda_seg.txt`): `_moe_sorting` / `topk_reg_kernel` gone from the
  68 KDA layers (24/step left = the DSA layers' MoE, SGLang path).
```text
bs2, per KDA layer / step        baseline   + launch top-k   + launch sort
KDA->KDA routed segment (us)     76.2       68.7             64.3   (moe1 41.7, stage2 19.0)
wide launch (us/call)            96.5       97.2             95.5   (one trace each: ~1 us noise)
VERIFY GPU span (ms)             19.20      18.91            18.65  (-0.55 ms, -2.9 %); DRAFT 1.24 unchanged
```
  bs2 step ~20.4 -> ~19.9 ms -> intvty p90 ~171 -> ~177 by the replay table (not an AgentX run).
- Debug knobs left in on purpose (default off): `..._WIDE_TOPK_CHECK`, `..._WIDE_SORT_CHECK`, `layer_bench --sort-cut`.
**Launch top-k made EXACT (2026-11-06 ~11:50, crsuse2-m2m-v3-131; `moe_front.route_row`).** Three differences vs
SGLang's aiter `topk_reg_kernel<float,14,true,false,true>` (csrc/kernels/topk_softmax_kernels_group.cu, module_moe_asm:
no fast-math, denormals flushed):
  (1) exact-score ties: aiter picks the lower `legacy_topk_tie_rank(id)` (an old DPP-table order), the launch picked the
      lower id -> `_tie_rank()` now used in the per-lane sort and in the pop (rank + id packed in one wave_umax word);
  (2) `C_LOG2E` is a DOUBLE literal: aiter's exp2 argument is an fp64 product rounded once -> `_sigmoid_aiter()`
      (fx.Float64 multiply); the f32 product of the rounded constant was off by an ulp now and then;
  (3) weights use an IEEE 1/sum (`ops.div_rn`), not hw_rcp.
Eager check (`agentx-k31/run_topkcheck.sh`, 4 conc x 8 x 1024 tokens): after (1)+(3) 10 / 4.25 M rows still differed,
w diff 2.2e-7; after (2) **0 / 4.40 M rows, w diff 0**. layer_bench: launch time unchanged, sort check PASS.
gsm8k1319 greedy, graphs, both knobs on: **0.9773** (baseline 0.9788 = 2 questions; before the fix 0.9742 = 6).
bs2 trace: routed segment 64.4 us, launch 95.7 us (unchanged); VERIFY mean 20.17 ms is ONE 128.7 ms
cross_device_reduce stall in that trace (medians equal) -> ~18.76 ms without it, same as 18.65.
**Routed MoE kernels at M=12, tuned on the SERVED path (2026-11-06 12:00-12:50, crsuse2-m2m-v3-131).** M=12 looks up
the token-16 row (get_padded_M) of `aiter/configs/model_configs/kimik3_a8w4_tuned_fmoe.csv`, tuned on uniform random
routing (~173 experts at 12 tokens); served bs2 routing touches ~110 (dump: `SGLANG_AMD_MONO_K31_WIDE_DUMP_IDS`,
`step2-launch-sort-20261106/ids_bs2.pt`). aiter's `gemm_moe_tune.py` is NOT usable for this as is: its a8w4 path draws
its own random routing (:4954, my `TUNE_MOE_TOPK_FILE` patch at :1722 never reached) and adds the activation quant to
us1; its pick (75.8 us) was slower than the current pair. Instead: one bs2 trace per kernel pair
(`agentx-k31/run_moe_cands.sh`: a copy of the kimik3 csv with the token-16 Situv2/inter-384 row swapped, served with
`AITER_CONFIG_FMOE=<aiter>/configs/tuned_fmoe.csv:<copy>` -- a set env var reads ONLY its files, so the default
table goes first). Results `/shared_nfs/kk/results/Kimi-K3.1-Pro/moetune-m12-20261106/` (cands.txt, seg_*.txt):
```text
pair (stage1 + stage2)                                             s1 us  s2 us  routed seg us
current: t32x128x256_w4_gui_xcd4 + opus atomic b3_ws2              41.9   19.0   64.4
c1 t32x64x256_w3_gui_xcd4_kw2 + opus b2_ws2                        44.8   19.4   67.8
c2 t32x64x256_w2_gui_kw2 + moe2_layout t32x128x128_atomic_nt       37.6   17.7   58.8
c3 t32x128x256_w3_gui + moe2_layout t32x128x128_atomic             34.5   19.0   57.1
c4 t32x64x256_gui_kw2 + moe2_layout t32x128x256_atomic             FAILS: stage2 BK 256 must divide inter 384
c5 t32x128x256_w4_gui + opus b3_ws2                                34.2   18.9   56.6
c6 t32x128x256_w4_gui + moe2_layout t32x128x128_atomic_nt          33.6   17.7   54.9   <- best (-9.5 us/layer)
c7 t32x128x256_w3_gui + moe2_layout t32x128x128_atomic_nt          34.2   17.8   55.6
```
The xcd4 stage-1 variant is the slow part at real expert counts. c6 gsm8k1319 greedy (graphs, launch top-k + sort on):
**0.9765** (baseline 0.9788, exact top-k with the old pair 0.9773: within 1-3 questions); log 32 x `using 2stage
... w4_gui_fp8`, sort-on 68, skipped 0; trace routed segment 55.2 us. All candidates kept the launch sort (accumulate
mode). Point `agentx-k31/points/c1-wide-opt` = c1-base + both knobs + c6 csv (copied into the point dir).
**AgentX c1-wide-opt (2026-11-06 12:50-13:57, crsuse2-m2m-v3-131; `/shared_nfs/kk/results/Kimi-K3.1-Pro/agentx-wideopt-20261106`),
0/249 errors, hit 0.971 (= base):**
```text
                     tok/s/GPU  ITL mean/p90 ms  P90 intvty  TTFT p90 s | step by concurrency (med / mean ms, ~12 K bs2 steps)
c1-base (fix-1106)   2186.4     4.32 / 5.84      171.2       1.28       | bs1 13.86/13.96   bs2 20.06/20.53
c1-wide-opt          2188.7     4.25 / 5.28      189.4       1.01       | bs1 13.84/14.04   bs2 18.21/19.01 (-9.2 % / -7.4 %)
```
The single-arm P90 gain (+10.6 %) is at the edge of the 5-10 % arm noise, but the per-step split is not noise:
bs1 unchanged (narrow path untouched), bs2 -1.5 ms mean, as predicted (~19.2 ms).
Scripts: `trace_bs2.sh` +
`capture_trace_bs2.sh` (in `/shared_nfs/kk/agentx-k31/`), out `/shared_nfs/kk/results/Kimi-K3.1-Pro/trace-bs2-20261106/`
(status.txt, summary.txt). Compare per kernel with `trace-tok68-20261105/summary.txt` (c1-base = bs1).
**Next:** roofline `k31_kda_layer_m12` (KDA state bytes x 2 req / HBM BW) vs 6.57 ms, then wide MoE path; old note: and diff it per kernel against the bs1
breakdown above (`trace_breakdown.py`); also count the distinct experts per layer at bs1 vs bs2 to get the MoE
byte roofline. No B300 per-request series locally (only `ci/b300_job_*.log`), so the B300 bs2 cost is inferred.
**Files (node-local):** `/shared_nfs/kk/agentx-k31/` (points/, `run_point_k31.sh`, `driver_matrix.frozen.sh`
<point...> with `OUT=`, `watch_matrix.sh <driver pid from status.txt>`, `trace_tok68.sh`, `topk_*.sh`, ci/);
`/tmp/k31v/` (env.sh, capture_trace_k31.sh, bench_kscatter.py, replay_topk.py, eval deps; env-gated debug hook
`dumphook/sitecustomize.py`). /tmp is lost on restart.
**Repro (one AgentX arm, ~65 min):**
```bash
OUT=/shared_nfs/kk/results/Kimi-K3.1-Pro/<run-tag>; mkdir -p $OUT; cd /workspace
OUT=$OUT setsid nohup bash /shared_nfs/kk/agentx-k31/driver_matrix.frozen.sh c1-base < /dev/null > $OUT/driver.log 2>&1 &
```
Trace: `bash /shared_nfs/kk/agentx-k31/trace_tok68.frozen.sh <pid to wait for>` then
`python3 /workspace/claude-skills/kimi-k31/scripts/trace_breakdown.py <TP0 trace>`.
**Pass / read:** single 3600 s arm resolves ~5-10 % only (agentx skill: replicate spread 5.67 %); quote
total tok/s/GPU and P90 interactivity; overall cache hit, not device-tier. Scripts must use `grep`, not `rg`
(or `export -f rg`); keep `AIPERF_RUNTIME_DIR` on /shared_nfs (the /workspace quota breaks the venv rebuild).

## Setup

- Client: InferenceX `pr-4808` @ 7ad0ba79c (`/workspace/InferenceX-latest`, main was 2ad2c9f97 2026-11-04).
  New layout: client is `python3 -m infx.bench agentic` (`srt_agentic.sh`), aiperf = PyPI `agentx-harness==1.0.13`
  in a uv venv under `AIPERF_RUNTIME_DIR` (no aiperf submodule any more; `AIPERF_WARMUP_REQUESTS_PER_LANE` is not read).
- Point = CI job 127116611069 (run 41977056700, PR 4808 TP8 c4, passed 1h40m) via `agentx/e2e/extract_point.sh`
  (raw extract kept at `agentx/e2e/points/ci41977056700/k31pro-c4`). Local edits only: model / draft local paths,
  `--speculative-draft-model-revision` dropped, split `--model-loader-extra-config` JSON rejoined.
  CI injects `SGLANG_SIMULATE_ACC_LEN=3.43` (`match-expected`, `real-draft-token`) -- not in the recipe yaml.
- Draft: `RadixArk/Kimi-K3.1-Dflash2-Preview` @ d0d8bc2e -> `/shared_nfs/RadixArk/Kimi-K3.1-Dflash2-Preview-d0d8bc2e`
  (sha256 1e07d235..., NOT the same weights as step55000 e883bba6...).
- Server stack: image sglang 0.5.23.dev0+k31pr48490.380ee3f, aiter 285579b62, sgl-amd-mono e52803f (CI image is
  `lmsysorg/sglang:dev-kimi-k31-pro-mi35x@sha256:6af91676...`; not checked to be identical).
- opt arm = base + `SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_{FRONT,O_PROJ,STREAM}=1` + mem 0.90 (0.85 leaves no KV with
  these switches, see `DFLASH_STEP55000_VERIFY.md`). Everything else identical.

## References (CI artifacts)

```text
c4                               tok/s/GPU  P90 intvty  TTFT p50/mean s  ITL p90 ms  ISL     hit    dur
MI355X CI (PR 4808, 41977056700)  3005.9     129.6       0.41 / 2.40      7.72        176637  0.887  3617
B300 SGLang CI (41932307764)      3424.8     189.0       0.40 / 1.19      5.29        174932  0.973  3595
```
B300 c4 runs HiCache (kvdram); not the same config. (User called 41932307764 "vLLM"; its title and artifacts are
SGLang on B300.)

## Results

(append rows: node, date)

### base (PR 4808 c4 as CI ran it), crsuse2-m2m-v3-131, 2026-11-05 08:58-10:05, rc=0, 0 errors

```text
                    tok/s/GPU  P90 intvty  TTFT p50/mean s  ITL p90 ms  ISL     hit    dur
this node, base     2973.7     133.8       0.42 / 2.54      7.47        168937  0.883  3620
MI355X CI c4        3005.9     129.6       0.41 / 2.40      7.72        176637  0.887  3617
delta vs CI         -1.1 %     +3.2 %                                    -4.4 %
```
Reproduces CI within noise (single-arm spread ~5.7 %); ISL 4.4 % lower (trace mix at a fixed window).
Server accept len in decode lines ~3.42 (SIM 3.43).

Trap (new client): `Runtime.active()` is only true when already running the venv python, so EVERY run
`rmtree`s and rebuilds the uv venv under `AIPERF_RUNTIME_DIR`. The opt arm's rebuild on /workspace hit
"Disk quota exceeded (os error 122)" with df still showing 309 G free (quota, not df) -> client exit 2 at
start. Fix: `AIPERF_RUNTIME_DIR=/shared_nfs/kk/agentx-runtime-latest`; opt rerun 10:18.

### Gap to B300 SGLang (CI artifacts, run 41977056700 vs 41932307764) -- analysis, crsuse2-m2m-v3-131, 2026-11-05

```text
        tok/s/GPU  P90 intvty  ITL mean/p90 ms  overall hit (gpu/cpu)    SIM acc  draft tok
MI c1   2191.9     175         4.26 / 5.71      0.971                    3.43     6
B300 c1 2370.0     241         3.71 / 4.15      0.974 (0.974/0.000)      3.68     8
MI c4   3005.9     130         5.66 / 7.72      0.887                    3.43     6
B300 c4 3424.8     189         4.40 / 5.29      0.973 (0.661/0.312)      3.68     8
```
- c1: cache hit already equal. ITL mean gap 14.8 %, of which ~7.3 % is the simulated acceptance alone
  (3.68 / 3.43, B300 runs 8 draft tokens); per-step time gap ~7 %. ITL p90 gap (+38 %) is larger than mean.
- c4: hit 0.887 vs 0.973 is the KDA-state (mamba) pool, not KV. c4 base server metrics (TP0, profiling):
  `mamba_available_tokens` p50 0 / max 4 of `--max-mamba-cache-size 60`, `mamba_evictable` 50-59 (all slots
  hold cached prefix states and get evicted); KV evicted only 2.10 M tokens vs 75.5 M cached (2.8 %), kv
  evictable p50 1.44 M of 1.76 M. A hybrid prefix hit needs the KDA state at that position, so evicted
  states = misses. B300 keeps them in host DRAM (HiCache); mono cannot (spec has no KV_OFFLOAD).
- Mamba slot cost here: ssm 3.08 GB + conv 0.11 GB for 60 slots = ~53 MB / slot / rank.
- B300 c1 flags of note: `--speculative-num-draft-tokens 8`, `--mamba-radix-cache-strategy extra_buffer`,
  `--mamba-track-interval 256`, `--mamba-ssm-dtype float32`, `--mamba-max-states-per-path 1`, no
  `--prefill-decode-interval` / lpm, `--max-running-requests 2`, HiCache 230 GB.

### Gap matrix results (crsuse2-m2m-v3-131, 2026-11-05, OUT=/shared_nfs/kk/results/Kimi-K3.1-Pro/agentx-gap-20261105)

```text
c4                tok/s/GPU  P90 intvty  TTFT p50/mean s  ITL mean/p90 ms  ISL     hit    errors
base (08:58)      2973.7     133.8       0.42 / 2.54      5.78 / 7.47      168937  0.883  0/506
mamba150 (12:15)  3265.7     143.1       0.37 / 1.05      5.45 / 6.99      172089  0.951  1/545
delta             +9.8 %     +7.0 %      -12 % / -59 %    -5.7 % / -6.4 %  +1.9 %  +0.068
B300 c4 (CI)      3424.8     189.0       0.40 / 1.19      4.40 / 5.29      174932  0.973
```
mamba150: hit 0.883 -> 0.951 (B300 0.973); gap to B300 tok/s/GPU 13.2 % -> 4.6 %. Mamba slots no longer pinned
(available p50 0 -> 7, p90 43; min still 0); KV evicted 2.10 M -> 4.18 M tokens (KV 1.76 M -> 1.44 M), harmless.
The one error: InvalidInferenceResultError (a response with no content, only usage/[DONE]); no server fault.
+9.8 % is above the 5.67 % replicate spread but below the 10 % bar; every secondary metric moved the same way.

```text
c1                tok/s/GPU  P90 intvty  TTFT p50/mean s  ITL mean/p90 ms  ISL     hit    errors
base (13:22)      2186.0     174.7       0.49 / 0.70      4.30 / 5.72      251408  0.971  0/249
MI355X CI c1      2191.9     175         0.49 / 0.71      4.26 / 5.71      251408  0.971
B300 c1 (CI)      2370.0     241         0.56 / 0.77      3.71 / 4.15      260660  0.974
```
c1-base reproduces CI c1 to 0.3 % (same ISL to the token: same traces at c1).

```text
c1                tok/s/GPU  P90 intvty  TTFT p50/mean s  ITL mean/p90 ms  ISL     hit    errors  SIM acc
base (13:22)      2186.0     174.7       0.49 / 0.70      4.30 / 5.72      251408  0.971  0/249   3.43
draft8 (14:29)    2028.2     165.7       0.49 / 0.71      4.76 / 6.03      242287  0.969  0/240   3.68
delta             -7.2 %     -5.2 %                       +10.7 % / +5.4 %
```
draft8 LOSES at c1: per decode step (ITL mean x acc) 14.7 -> 17.5 ms (+19 %) for +7.3 % acceptance. 8-token
verify costs more than it returns on this stack at the CI-simulated acceptance. Break-even needs ~+19 % real
acceptance (gsm8k step55000 6 -> 8 tokens measured +17 %, 4.35 -> 5.07: close, not over). Server warmup with
8 tokens took ~20 min (first-time mono / FlyDSL builds for L=8), ~3 min with 6.

```text
c4                tok/s/GPU  P90 intvty  TTFT p50/mean s  ITL mean/p90 ms  ISL     hit    errors  SIM acc
base (08:58)      2973.7     133.8       0.42 / 2.54      5.78 / 7.47      168937  0.883  0/506   3.43
draft8 (15:53)    2889.8     123.0       0.40 / 2.59      6.13 / 8.13      168248  0.894  0/495   3.68
delta             -2.8 %     -8.1 %                       +6.1 % / +8.8 %
```
draft8 loses at c4 too: per step 19.8 -> 22.6 ms (+14 %). Driver done 17:00, GPUs free.

### Summary of the gap matrix (2026-11-05)

- KEEP `--max-mamba-cache-size 150` at c4: +9.8 % tok/s/GPU, +7.0 % P90 intvty, hit 0.883 -> 0.951; gap to
  B300 c4 13.2 % -> 4.6 % (tok/s/GPU). n=1; replicate before landing in the recipe. Untried: 200+ slots.
- DROP 8 draft tokens: c1 -7.2 %, c4 -2.8 % tok/s/GPU at CI-simulated acceptance (3.68 vs 3.43).
- c1: hit already equals B300; remaining gap is decode per-step time (ITL p90 5.72 vs 4.15 ms). Not yet tried:
  B300 runs without `--prefill-decode-interval 16` / `--schedule-policy lpm`.

```text
c1                tok/s/GPU  P90 intvty  TTFT p50/mean s  ITL mean/p90 ms  ISL     hit    errors
base (13:22)      2186.0     174.7       0.49 / 0.70      4.30 / 5.72      251408  0.971  0/249
no pdi, no lpm    2189.5     172.8       0.48 / 0.66      4.31 / 5.79      251407  0.971  0/249   (11-06 00:41)
```
Null (+0.2 % / -1.1 %): `--prefill-decode-interval 16` and `--schedule-policy lpm` are not the c1 ITL-tail gap to B300.

InferenceX PR 4846 (2026-11-05, zijiecode) independently sets `max-mamba-cache-size 150` at c1 / c4 (same as our
c4 finding), adds `SGLANG_ROCM_SPARSE_MLA_DECODE_HIP=1` (already default-on in our sglang) and bumps the image to
`lmsysorg/sglang-rocm:dev-kimi-k31-pro-mi35x-1105` (split-K sparse MLA decode/verify, DFLASH fc row-shard in
prefill, MoE small sort to 1024; the last two not found in our sglang build).

### Why 8 draft tokens cost more: c1 decode trace (crsuse2-m2m-v3-131, 2026-11-06 00:41-00:48)

`/shared_nfs/kk/results/Kimi-K3.1-Pro/trace-tok68-20261105` (points c1-base / c1-draft8 + SGLANG_PROFILE_* env,
bench_serving random ISL 196608 / OSL 512 c1, profile 2 s from the last prefill chunk; `trace_breakdown.py` TP0):
```text
per step GPU ms        6 tok    8 tok    delta
DRAFT                  1.174    1.217    +0.04
VERIFY                 12.268   14.104   +1.84
  dsa_indexer          0.517    1.586    +1.07   k_scatter<0,true,16> x24/step: 17.4 -> 61.9 us/call (x3.6 for x1.33 rows)
  other (mono k2)      7.528    8.215    +0.69   k3_mono_k2_s6_* -> k3_mono_k2_s8_* (+9 %)
  kda / sparse_mla     3.604    3.682    +0.08
total                  13.44    15.32    +1.88 (+14 %)
```
- The DSA indexer top-k scatter scales super-linearly 6 -> 8 rows: the main lever. Linear (~23 us/call) would
  save ~1 ms/step -> 8-token step ~+7 % = break-even with the +7.3 % simulated acceptance.
- AgentX c1 bs1 (no profiler) measured +3.3 ms/step; GPU explains ~1.9 ms, ~1.4 ms unexplained (host?).
  The synthetic bench's TPOT (8 tok 7.52 ms) is inflated by the profiler; do not use its wall clock.
- Next: microbench `sglang::dsa_gfx950::topk::k_scatter` at ctx ~196 k, rows 6 vs 8 (launch config / occupancy).

#### CORRECTION (2026-11-06): k_scatter is NOT super-linear in rows -- one call per step takes a slow path

- Isolated microbench (`/tmp/k31v/bench_kscatter.py`, logits_hist -> topk_transform, ctx 196608, table 16384 pages
  so the server's `k_scatter<0,true,16>` variant runs, random data): 6 rows 11.6 us, 8 rows 11.6 us, 16 rows
  12.2 us at G=64 (G=48 / 32 no better). Flat in rows.
- In the traces the per-call MEDIAN is equal (6 tok 17.1 us, 8 tok 16.8 us, p90 22.7 / 22.8), no other-stream
  overlap. The 8-token mean (61.9 us) comes from exactly 35 calls of 1.1-1.5 ms = one per verify step, always the
  22nd DSA layer of the step (index 21 of 24), preceded by a normal 19.6 us logits_hist. 38.5 ms of the 52.0 ms
  total. 6 tokens: max 47 us, none slow.
- Likely a data-dependent fallback in topk_transform.cuh: refinement over > REF_CAP (2048) candidates in the
  threshold bin (radix fallback, one block per row from global) or HIER_BAD -> flat histogram rescan. Not yet
  confirmed: needs the layer-21 inputs of an 8-token verify step dumped and replayed in the microbench.
- So of the +1.88 ms/step: ~1.1 ms is this one call (fixable in principle), +0.69 ms mono k2 (rows 6 -> 8).

#### ROOT SYMPTOM (2026-11-06 01:00-01:15): with mono + 8 draft tokens, layer 87's DSA index-K cache is garbage

Debug hook (node-local `/tmp/k31v/dumphook/sitecustomize.py`, env-gated `K31_TOPK_DUMP` / `K31_TOPK_STATS`, put on
the server's PYTHONPATH only; wraps `dsa_indexer_hip.hist_topk_paged` and the loader's `topk_transform`), eager
server (`--disable-cuda-graph`, point `c1-draft8-dump`), one 196608-token request:
- The slow call is reproduced eagerly (79 of 82 steps, always DSA index 21 = model layer 87, ~1.1 ms) and offline
  from the dump (`/tmp/k31v/replay_topk.py`: 1125 us).
- Layer 87's indexer logits are EXACTLY 0 for every position of every row (196,522 equal values, 0 above the
  threshold) -> all candidates in one bin -> the > REF_CAP single-block fallback. Layers 83 / 92: ~195 k distinct.
  Top-k over all-equal logits is arbitrary (two replays pick different slots): layer 87's sparse attention reads
  ~random tokens. A CORRECTNESS bug, found through its latency.
- Inputs at layer 87 (8-row verify, seqlen 364-372): q absmax 448 and gate weights normal; the index-K pages the
  request reads have absmax 0.75 (layers 83 / 91: 448), all 6 pages including prompt pages written by prefill.
  -> layer 87's index-K region is overwritten (or never written) on the mono 8-token path.
- 6 draft tokens: no slow call in the trace (layer 87 fine). Mono off: the fused indexer path is not used at all
  (`_mono_k31` gates `hist_topk_paged` and the index-K store), so it is not a clean control.
- gsm8k cannot see this: with <= 2048 context tokens DSA top-k keeps every token, so the indexer is moot. Earlier
  "8 tokens gsm8k 0.985 with wide" (MONO_KERNEL 10.32) does not clear long-context accuracy at 8 tokens.
- Class of bug seen before: MONO_KERNEL 10.23 (mono K1 overran its scratch into the KV free list under DFLASH).

#### ROOT CAUSE + FIX (2026-11-06 01:40-02:15, crsuse2-m2m-v3-131): mono K1 warmup bank one block short

Localization (all eager, 8 tokens, env-gated hook `/tmp/k31v/dumphook/sitecustomize.py`):
1. No torch tensor overlaps layer 87's index-K storage (own 222 MiB segment; neighbours = layers 83 / 91 index-K).
2. Raw pages: layer 87 keys are ALL ZERO, scales 2^-22 (the writer's floor for amax 0); layer 83 normal. The
   legitimate writer wrote zero keys, in prefill and verify -> the indexer INPUT was wrong before any request.
3. Layer 87 `indexer.k_norm.weight` and `.bias` (fp32 [128]) are all zero (layer 83 normal) -> LayerNorm output 0.
4. Bisect: `SGLANG_AMD_MONO_K3_K1=0` restores k_norm and index-K; MLA_PRE=0 and K31_WIDE=0 do not.
5. `sgl_amd_mono/kimi_k3/spec.py` `_warmup`: the throwaway bank was `torch.zeros(rows, nb, HIDDEN)` with
   nb = key.nblocks = prev_valid_blocks, but a block-write layer's K1 (and mla_pre) has write_idx = nb and stores
   its row at `t*stride0 + write_idx*stride1` (`stages/seam.py:163`): for the last token that is one HIDDEN row
   (14 KiB) past the allocation, written with zeros. At 8 draft tokens the allocator had placed layer 87's k_norm
   there; at 6 it landed somewhere harmless. Bug is config-independent (same lesson as MONO_KERNEL 10.23).
   Served banks are `(T, block_num, H)` with write_idx <= block_num - 1: not affected.

Fix (sgl-amd-mono branch `k31/fix-warmup-bank-oob` off e52803f, uncommitted, +20/-1): warmup bank gets nb + 1
rows; asserts `write_idx < blocks.shape[1]` in `k1.warmup` and `mla_pre_launch`.
Check (8 tokens, eager, 196 k request): layer 87 k_norm 1.078 / 0.087, all 24 DSA layers index-K absmax 352-448,
no slow topk_transform call, no assertion. Pending: kernel tests, gsm8k 6 / 8 tokens with CUDA graphs, c1 trace,
AgentX c1-draft8 rerun.
Fix checks (2026-11-06 02:20-02:50): mono kernel tests `test_kimi_k3_k1.py`, `test_kimi_k3_mla_pre.py`,
`test_kimi_k31_k1_ring.py`: 19 passed (+8 subtests). gsm8k200 greedy, CUDA graphs on, REAL acceptance
(`gsm_check.sh`): 8 tokens 0.985, accept 5.080 req-weighted; 6 tokens 0.99, accept 4.458. Real acceptance gains
+14 % at 8 tokens (CI's simulated values 3.68 / 3.43 assume +7.3 %). No assertion fired.
Upstream: sgl-project/sgl-amd-mono PR #108 "[Fix] [MI355X] kimi_k3: Size the warmup bank for the block-write row",
from kkHuang-amd fork branch `k31/fix-warmup-bank-oob` (fa79409 fix + fc61d69 test, base e52803f), opened 2026-11-06.

### AgentX c1 after the fix (crsuse2-m2m-v3-131, 2026-11-06, OUT=/shared_nfs/kk/results/Kimi-K3.1-Pro/agentx-fix-20261106)

```text
c1                      tok/s/GPU  P90 intvty  ITL mean/p90 ms  ISL     hit    errors
draft8, before fix      2028.2     165.7       4.76 / 6.03      242287  0.969  0/240
draft8, after fix       2222.7     170.3       4.23 / 5.87      253910  0.971  1/251   (02:24-03:30)
base (6 tok), before    2186.0     174.7       4.30 / 5.72      251408  0.971  0/249
B300 c1 (CI)            2370.0     241         3.71 / 4.15      260660  0.974
```
Fix: draft8 +9.6 %. Fixed draft8 vs pre-fix 6 tok: +1.7 % tok/s, -2.5 % P90 intvty: a tie within noise (CI's
simulated acceptance only grows 7.3 %; real gsm8k acceptance grows 14 %). Gap to B300 c1: -6.2 % tok/s (6 tok:
-7.8 %), P90 intvty -29 %, ITL p90 +41 %. c1-base after the fix running (03:30).
c1-base after the fix (03:30-04:36): 2186.4 tok/s/GPU, P90 intvty 171.2, ITL 4.32 / 5.84 ms, hit 0.971, 0/249
errors = the pre-fix 2186.0 (fix has no effect at 6 tokens). Same session 8 vs 6 tokens: +1.7 % tok/s, P90 intvty
170.3 vs 171.2, ITL mean -2.1 %, ITL p90 +0.5 %: a tie at CI's simulated acceptance. Gap to B300 c1: -6.2 % tok/s,
-29 % P90 intvty (decode per-step time). If CI's golden acceptance for 8 tokens tracked the real +14 %, 8 tokens
should win. GPUs free 04:37.
