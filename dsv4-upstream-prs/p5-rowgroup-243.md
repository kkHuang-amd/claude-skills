Owner node: crsuse2-m2m-243

# P5 rowgroup crash (SGLANG_HIP_DSV4_C4_INDEXER_ROWGROUP=1) + takeover of the PR split from m2m-214

Parent plan: `PLAN.md` (owner m2m-214, node expected to be shut down; its /tmp branches are lost).

## CONTINUE HERE

**Status (2026-10-13 06:35, m2m-243):** all 8 PR branches PUSHED to HaiShaw/sglang (new branches, no force), remote SHAs verified.
P1 9dff916e6b, P2 32682bf3fb, P3 ed173ab691, P4 fcb88999f8, P5 a1c314cd41 (incl. guard), P6 ae64191992 (incl. guard),
P7 07034c5f6d, P8 c88e6f5238. Base 4d5692b6cd. PRs NOT opened yet.
**P5 rowgroup crash:** NOT reproduced on 243 in 16 GSM8K rounds (P5 alone, P5 + all model commits on new main, P5 on old
base). 214's 2 failures were its two earliest rowgroup runs (04:47, 05:08); 214's later P5 runs passed. Root cause unknown;
suspect 214 node state, not P5 code. Not a push blocker.
**Update 2026-10-13 ~07:10 (243):** format script (`/workspace/useful-scripts/sglang-format.sh`) passes on all 8 branches.
New commits pushed: P5 edd0292e39 (V4.1 `SGLANG_HIP_FP4_INDEXER_ROWGROUP`, default on, now falls back without aiter #6145),
P8 439cdd170c (resolved #6042 patch without trailing whitespace; git apply result identical). PR drafts: `pr-bodies.md`
(co-author AMD-yanfeiwang on every body, user request). P5 C4 microbench: speed/P5_c4_rowgroup_bench_243.log.
**PRs opened 2026-10-13 (243, by kkHuang-amd, head HaiShaw):** [2/8] #44003, [3/8] #44004, [4/8] #44005, [5/8] #44006,
[6/8] #44007, [7/8] #44008. Opener: /tmp/p5w/open_pr.py <Pkey> (body = pr-bodies.md block inside the upstream template;
numbers in /tmp/p5w/pr_numbers.json).
**Next:** long-ctx P1 split plan A/B (/tmp/p5w/speed_long.sh: CTX=200000 ISL=64000 c8/c32) -> update P1 speed line -> open [1/8]
-> open [8/8] ({P1}/{P5}/{P6} filled) -> PATCH [1/8],[5/8],[6/8] bodies: "[8/8]" -> its PR number.
**Unit (243, 2026-10-13):** P5 old 53p/33s, P5 pin 86p, P6 old 6s, P6 pin 6p -- same counts as 214's guarded runs.
**Repro (P5 rowgroup e2e, 5 GSM8K rounds on one server):**
```bash
REPS=5 /workspace/claude-skills/dsv4-upstream-prs/scripts/e2e_rep.sh <tag> /tmp/prwt/dsv4-fp4-indexer-rowgroup/python \
  /sgl-workspace/aiter_pin SGLANG_HIP_DSV4_C4_INDEXER_ROWGROUP=1
```
**Pass criteria:** READY (not READY_BUT_ERR), every round GSM8K >= 0.945 (main_pin 0.953).

## PR map (243; "table #" = user's original item numbers as recorded in PLAN.md)

| PR | link | head (HaiShaw/sglang) | table # | source commits |
|----|------|------------------------|---------|----------------|
| [1/8] | sgl-project/sglang#44016 | amd/dsv4-fp8-unified-kv-decode | #1 #8 #12 | e144f84055 1e32269bf0 3795b59ec3 |
| [2/8] | sgl-project/sglang#44003 | amd/dsv4-fp8-verify-swa-ring | #2 | 5e256859ef |
| [3/8] | sgl-project/sglang#44004 | amd/moe-small-batch-sort | #3 | 04627b429a |
| [4/8] | sgl-project/sglang#44005 | amd/dsv4-c4-indexer-qlora-quant-reuse | #4 | 6eccfd9b35 |
| [5/8] | sgl-project/sglang#44006 | amd/dsv4-fp4-indexer-rowgroup | #5 #9 | 19fbe52377 89cf0ba9a1 + a1c314cd41 edd0292e39 |
| [6/8] | sgl-project/sglang#44007 | amd/dsv4-dense-fp8-mxscale | #10 | 6069a730c5 + ae64191992 |
| [7/8] | sgl-project/sglang#44008 | amd/dsv4-intree-gemm-tune | #13 | dc7fc758e6 |
| [8/8] | sgl-project/sglang#44035 | amd/rocm-dockerfile-aiter-pin | #11 | 4c4a688866 + 439cdd170c |

## Rebuild on m2m-243 (2026-10-13) -- node-local, /tmp

- Script `/tmp/p5w/build_prs.sh` (copy below). Base 4d5692b6cd, branches `amd/<name>`, worktrees `/tmp/prwt/<name>`,
  committer kkHuang-amd, authors kept. P6/P7 conflicts resolved exactly as PLAN.md table says.
  P1 9dff916e6b, P2 32682bf3fb, P3 ed173ab691, P4 fcb88999f8, P5 fd75faf6fa, P6 8ed49ff5cb, P7 07034c5f6d, P8 c88e6f5238.
- P6 guard REDONE on 243: ae64191992 (kkHuang-amd) `fp8_utils.aiter_has_dense_mxscale()` probes
  `aiter.ops.gemm_op_a8w8.get_mxscale_bpreshuffle_config`; `set_dense_mxscale(True)` warns + stays off without it; test class
  skips. Old aiter: 6 skipped (unit/P6_old_guard_243.log), runtime stays off. Pin: probe True; GPU unit run queued after e2e
  (/tmp/p5w/after_queue.sh -> unit/P6_pin_guard_243.log).
- P5 guard REDONE on 243: a1c314cd41 (kkHuang-amd). Runtime: SGLANG_HIP_DSV41_INDEX_K_PAGE8 (V4.1 page8 pool, from #43073)
  was env-only -> ImportError mid-serving on old aiter; now falls back to page-64 pool with a warning
  (`aiter_has_fp4_rowgroup()`; C4 switch already checked). Tests: `requires_aiter_rowgroup` mark on the 10 row-group/page8
  tests (both missing symbols come from aiter #6145 5b2f7d1d1). Old aiter, marked tests only: 33 skipped (= 214's count).
  Full old/pin unit runs queued after e2e -> unit/P5_{old,pin}_guard_243.log.
- LOST with 214 (now redone above): kkHuang guard-fix commits 4fce24df5d (P5) and 0011479d5b (P6) (unit tests skip/guard on old aiter).
  Must be redone; see unit logs `.../upstream-prs-1013/unit/P5_old_guard.log`, `P6_old_guard.log`.
- aiter_pin: `git worktree` of /sgl-workspace/aiter at 02575f1c3a + CK submodule 96afae1d +
  `docker/patches/aiter_pr6042_resolved.patch` from commit 4c4a688866 (same as Dockerfile).

## Findings

- 214 logs (P5_pin_rg, P5_pin_rg2): the gloo `Connection reset by peer` traceback is secondary. Only one rank
  raises; the others report "TCPStore server has shut down too early" -> rank 0 (store host) died first with
  no Python traceback, no fault text, ~2 s after `ready to roll` (rg2) / ~1 min into GSM8K (rg).
- 214 e2e passes with rowgroup: duyi_tip_rg 0.951 (Duyi branch, base d1357bf87b), bisectA_rg 0.948 (content unknown).
  Fails: P5 branch alone on base 4d5692b6cd. P5_pin_rg_nograph = FAIL_START.
- Two hypotheses: (a) 89cf0ba9a1 depends on another Duyi commit (1e32269bf0 / e144f84055 / 6eccfd9b35 touch the same
  files); (b) main drift d1357bf87b -> 4d5692b6cd. A/B: x_allnew (all 7 model commits on 4d5692b6cd) vs x_p5old
  (only 19fbe52377+89cf0ba9a1 on d1357bf87b).

- 214 main_pin_nograph = FAIL_START too (05:32) -> the nograph start failure is NOT P5-specific; ignore it for P5.
- 214 P5_pin_rg4 (05:37, same env as rg2) READY 0.951 -> P5 rowgroup crash is INTERMITTENT (2 fail / 1 pass on 214;
  rg3 aborted). Single-run A/B is not evidence. Switched to scripts/e2e_rep.sh: one server, REPS GSM8K rounds,
  stop at first crash. Queue /tmp/p5w/queue.sh: P5_rep_243, x_allnew_rep_243, x_p5old_rep_243 (REPS=5).

- From 214's PLAN.md (written 05:31, before its rg4/rg5 passes): bisectA = P5+P1 commits 0.948, bisectB = P5+#4 0.949;
  it counted P5+rowgroup as 3/3 crash. Later 214 runs P5_pin_rg4 0.951, P5_pin_rg5 0.951 -> 214 total 2 fail / 2 pass
  (rg3 aborted). The --disable-cuda-graph start failure is aiter inverse_rope_group_quant.cu:1181 "o must be bf16/fp16,
  got fp8" from deepseek_v4.py:2478 (fused wo_a invrope); main_pin_nograph fails the same way -> pre-existing on main.
- NOTE: /workspace is NFS shared with 214 (PLAN.md changed by 214 at 05:31 while 243 was working). The dsv4-upstream-prs/
  dir is untracked in git (never committed).

- LONG-CONTEXT FAULT (243, 2026-10-13 07:47): P1 branch, CTX=200000, ISL=64000 (speed/P1_*_long_243): server READY, then the
  bench warmup (1 request, 64K in) -> "Memory access fault by GPU node-2", both with split plan on and off. Prefill chunks
  finished; fault at/after the first decode/verify. Python stack only shows the sync point (overlap_utils resolve_seq_lens_cpu).
  main 4d5692b6cd + aiter pin, same config, 64K in: SAME fault (speed/main_long64k_243) -> not P1 (see correction below).
  P1 at 32K in: OK (speed/P1_long32k_243). Not yet checked: main + old aiter (e7d2453f2) at 64K.
  Split plan A/B at ISL=32000 CTX=200000: c8 ITL on 6.22/6.20 vs off 6.10/6.21 ms (same). c32 (3-4 x 32K per DP rank) FAULTS
  in BOTH arms (same memory access fault, KV usage 1%). Pattern so far: 1x32K/rank OK, 1x64K or ~3x32K per rank -> fault;
  hypothesis (unverified): something sized for 65536 tokens per rank.
  CORRECTION: main + OLD aiter e7d2453f2 at 64K = NO fault (speed/main_old_long64k_243). So the fault comes with the aiter pin
  bump (02575f1c3a + resolved #6042 patch) -> HOLD [8/8]. Suspects: #6042 patch (pa_decode_sparse.py, 64-bit sparse MLA loop)
  and aiter #6159 (gfx950 sparse MLA opt). Note: local old aiter /sgl-workspace/aiter already has #6042 staged
  (identical to 042a2891 / the resolved patch), i.e. it matches main's Docker image (old pin + #6042 cherry-pick).
  Pin WITHOUT #6042 at 64K: STILL faults (speed/main_pin_no6042_long64k_243) -> #6042 is NOT the cause; the regression is in
  aiter e7d2453f2..02575f1c3a. aiter_pin restored (patch re-applied).
  Pin minus #6159 (3 python files back to 41630d0e7^, #6042 removed) at 64K: STILL faults (speed/main_pin_no6159_long64k_243)
  -> #6159 NOT the cause. aiter_pin restored. Remaining: 208 aiter commits (49 attention/MLA/decode/sparse). Proposed next:
  identify faulting kernel (no cuda graph + serialized launches, FUSED_INVROPE off) before a full bisect. [8/8] on HOLD.
  No graph + HIP_LAUNCH_BLOCKING + AMD_SERIALIZE_KERNEL=3 + FUSED_INVROPE=0 at 64K: NO fault (3/3 ok,
  speed/main_pin_blocking_long64k_243) -> not localized (3 variables changed). In the aiter range only #5896 (40d524b61, also
  [6/8]'s dependency) touches csrc/kernels/inverse_rope_group_quant.cu (fp32 cos table support). Running graph ON +
  FUSED_INVROPE=0 only (main_pin_noinvrope_long64k_243).
  Graph ON + FUSED_INVROPE=0: STILL faults -> not invrope / not #5896. Running graph OFF (+FUSED_INVROPE=0, no blocking):
  main_pin_nograph_long64k_243.
  Graph OFF only: NO fault -> graph-related. ROOT CAUSE (likely): the only asm decode .co loaded is
  hsa/gfx950/mla_v4/mla_a8w8_qh64_qseqlen1_gqaratio64_nm.co. The Docker image (main and P8, rocm.Dockerfile COPY) overwrites
  aiter's copy with sglang's python/sglang/kernels/ops/attention/dsv4/asm/gfx950/mla_v4/...nm.co (aiter#5195 build,
  md5 88b2f8e4). /sgl-workspace/aiter (old) HAS it (dirty file); my aiter_pin did NOT (stock 4fdce754, same as old commit).
  So the 64K fault = stock aiter nm .co + cuda graph + >~64K ctx per rank; NOT the aiter pin bump. [8/8] Dockerfile keeps the COPY.
  aiter_pin now has sglang's .co copied in. main_pin_sglco_long64k_243 (sglang .co loaded, md5 88b2f8e4): STILL faults
  -> .co alone is NOT it. aiter_pin also lacked 2 more Docker steps (all stages): pa_mqa_logits_fp4_prefill.py
  @lru_cache(maxsize=32) -> @cache (old local aiter has it), and torch_utils.py torch.Stream patch. Suspect: plan cache
  eviction (32) with >32 captured graphs -> dangling buffers. Both applied to aiter_pin (= Docker image now);
  verifying main_pin_docker_long64k_243.
  RESULT: main + Docker-equivalent aiter_pin at 64K, graph ON: NO fault (3/3). => the 64K fault was MY incomplete aiter_pin
  (missing Docker steps), NOT an aiter regression. [8/8] Dockerfile has all steps -> OK. Not isolated which of the last two
  steps (lru_cache->cache vs torch.Stream patch) is the fix; cache eviction is the likely one.
  Hypothesis (unverified): 214's P5 rowgroup silent rank death may be the same missing-step issue (214's aiter_pin was built
  the same way, per PLAN.md "02575f1c3a + resolved #6042 patch").
  ALL earlier aiter_pin results on 243 (GSM8K, speed, P1 split-plan numbers in #44016) ran WITHOUT these Docker steps.
  2026-10-13 ~09:50: [8/8] opened #44035; #44016/#44006/#44007 bodies PATCHed ("[8/8]" -> #44035). Re-run queue on the
  Docker-equivalent aiter_pin: /tmp/p5w/redo_dk.sh (P5 rowgroup 5 rounds; P1 def/nosplit 8K; P6 off/mx 8K; P1 64K long),
  tags *_dk_243. Then update #44016 / #44007 speed lines.
  DONE (speed/parsed_dk_243.md, median ITL avg of 2 reps): P1 split on vs off 8K c16 6.71/6.76, c64 9.32/9.32; 64K
  (CTX 200000) c8 6.26/6.26, c32 7.60/7.53 -> no measurable difference (old "+6.1% slower at c64" did NOT reproduce:
  artifact of the incomplete aiter_pin). P6 mxscale off->on c16 6.78->6.68 (-1.5%), c64 9.80->9.16 (-6.5%).
  #44016 and #44007 speed lines PATCHed (live body verified == intended); pr-bodies.md synced.
  CAVEAT: all earlier 243 e2e/speed runs with aiter_pin used the stock .co (short-ctx GSM8K / speed; P1 split-plan asm path).
  [1/8] opened as #44016 with the 8K and 32K numbers.

## Results (append-only; node + date per row)

| date | node | tag | status | GSM8K | note |
|------|------|-----|--------|-------|------|
| 2026-10-13 05:59 | crsuse2-m2m-243 | P5_pin_rg_243 | READY | 0.954 | first run on fresh aiter_pin (JIT compiled during startup); no crash |
| 2026-10-13 06:08 | crsuse2-m2m-243 | P5_rep_243 | READY x5 rounds | 0.946/0.950/0.954/0.947/0.948 | P5 branch only, no crash in 5 rounds |
| 2026-10-13 06:18 | crsuse2-m2m-243 | x_allnew_rep_243 | READY x5 rounds | 0.951/0.950/0.946/0.951/0.948 | 7 model commits on 4d5692b6cd, no crash |
| 2026-10-13 06:27 | crsuse2-m2m-243 | x_p5old_rep_243 | READY x5 rounds | 0.945/0.948/0.951/0.950/0.952 | P5 commits only on d1357bf87b, no crash |
| 2026-10-13 09:48 | crsuse2-m2m-243 | P5_rg_rep_dk_243 | READY x5 rounds | 0.948/0.950/0.946/0.954/0.948 | Docker-equivalent aiter_pin, no crash (cannot confirm 214 cause) |
