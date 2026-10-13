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
**Next:** PLAN.md T7 -- PR bodies (template rule), user review, open PRs; then `git merge origin/main` to sync (no force-push).
**Unit (243, 2026-10-13):** P5 old 53p/33s, P5 pin 86p, P6 old 6s, P6 pin 6p -- same counts as 214's guarded runs.
**Repro (P5 rowgroup e2e, 5 GSM8K rounds on one server):**
```bash
REPS=5 /workspace/claude-skills/dsv4-upstream-prs/scripts/e2e_rep.sh <tag> /tmp/prwt/dsv4-fp4-indexer-rowgroup/python \
  /sgl-workspace/aiter_pin SGLANG_HIP_DSV4_C4_INDEXER_ROWGROUP=1
```
**Pass criteria:** READY (not READY_BUT_ERR), every round GSM8K >= 0.945 (main_pin 0.953).

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

## Results (append-only; node + date per row)

| date | node | tag | status | GSM8K | note |
|------|------|-----|--------|-------|------|
| 2026-10-13 05:59 | crsuse2-m2m-243 | P5_pin_rg_243 | READY | 0.954 | first run on fresh aiter_pin (JIT compiled during startup); no crash |
| 2026-10-13 06:08 | crsuse2-m2m-243 | P5_rep_243 | READY x5 rounds | 0.946/0.950/0.954/0.947/0.948 | P5 branch only, no crash in 5 rounds |
| 2026-10-13 06:18 | crsuse2-m2m-243 | x_allnew_rep_243 | READY x5 rounds | 0.951/0.950/0.946/0.951/0.948 | 7 model commits on 4d5692b6cd, no crash |
| 2026-10-13 06:27 | crsuse2-m2m-243 | x_p5old_rep_243 | READY x5 rounds | 0.945/0.948/0.951/0.950/0.952 | P5 commits only on d1357bf87b, no crash |
