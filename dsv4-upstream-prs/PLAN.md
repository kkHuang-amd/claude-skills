Owner node: crsuse2-m2m-243 (taken over 2026-10-13 from crsuse2-m2m-214, which went silent at 05:53 and is being shut down)

# DSv4 AMD opt -> upstream sglang PR split

## CONTINUE HERE

**2026-10-13 06:40 (m2m-243): current state is in `p5-rowgroup-243.md` CONTINUE HERE.** The 8 branches were rebuilt on 243
(same base/commits/conflict resolutions; guard fixes redone) and pushed to HaiShaw/sglang. The 214-local notes below
(/tmp paths, 4fce24df5d/0011479d5b, queue) are history only.

**Status:** 8 PR branches built LOCALLY in `/tmp/pr_probe` (node-local, crsuse2-m2m-214) on origin/main
4d5692b6cd (2026-10-13). Not pushed, not tested.
**Decisions (user, 2026-10-13):** include unmerged deps in the branch; in-tree aiter code goes into
sglang; do NOT reuse existing PRs #41383/#41905/#41900/#41903/#43073 -- open new PRs; exclude
5a25286422 and router commits (#42027/#43483).
**Unit tests (2026-10-13, crsuse2-m2m-214):** logs /shared_nfs/kk/results/DeepSeek-V4-Pro/upstream-prs-1013/unit/.
aiter_pin = /sgl-workspace/aiter_pin (02575f1c3a + resolved #6042 patch); old = /sgl-workspace/aiter (e7d2453f2 = main's pin).
P1 old 10p/6s, pin 232p | P3 old 4p | P5 old 53p/33s, pin 86p (after guard fix 4fce24df5d) |
P6 old 6s, pin 6p (after guard fix 0011479d5b) | P7 old 10p. Fixes = new commits by kkHuang-amd on P5/P6.
**User decisions 2 (2026-10-13):** test fully BEFORE push; sync with main by `git merge origin/main` (no force-push).
**E2E:** scripts/e2e_one.sh + scripts/e2e_queue.sh; results append to
/shared_nfs/kk/results/DeepSeek-V4-Pro/upstream-prs-1013/e2e/results.md. Baseline main_pin GSM8K 0.953 (2026-10-13).
Queue (8 runs, ~15 min each) started 2026-10-13 ~04:15 on crsuse2-m2m-214.
**E2E results so far:** main_old 0.945, main_pin 0.953, P1_pin 0.948, P1_old 0.951, P2 0.951, P4 0.948,
P5_pin (default) 0.948, P6_pin_mx 0.951, P7 0.948 -- all OK.
**OPEN ISSUE:** P5 + SGLANG_HIP_DSV4_C4_INDEXER_ROWGROUP=1 (DP8+EAGLE+UNIFIED_KV_FP8) crashes 3/3 (silent
rank death, gloo reset). duyi_tip + rowgroup OK (0.951); bisectA (P5+P1 commits) OK 0.948; bisectB (P5+#4) OK 0.949
-> ambiguous, rerunning matrix. --disable-cuda-graph run aborted in aiter inverse_rope_group_quant.cu:1181
"o must be bf16/fp16, got fp8" (deepseek_v4.py:2478 fused wo_a invrope) -- checking if main has it too.
**Next:** GSM8K e2e (DeepSeek-V4-Pro TP8) baseline main vs each model branch, then push.
**(old) Next:** user approves push -> `git push haishaw <branch>` x8; then per-branch tests (table below).
Committer: kkHuang-amd <43161300+kkHuang-amd@users.noreply.github.com>; authors kept (Duyi-Wang).
**Rebuild a branch:** `/tmp/build_prs.sh <branch> <sha>...` (stops on conflict).

| PR | branch | commits (table #) | conflict resolution |
|----|--------|-------------------|---------------------|
| P1 | amd/dsv4-fp8-unified-kv-decode | e144f84055 1e32269bf0 3795b59ec3 (#1 #8 #12) | none |
| P2 | amd/dsv4-fp8-verify-swa-ring | 5e256859ef (#2) | none |
| P3 | amd/moe-small-batch-sort | 04627b429a (#3) | none |
| P4 | amd/dsv4-c4-indexer-qlora-quant-reuse | 6eccfd9b35 (#4) | none |
| P5 | amd/dsv4-fp4-indexer-rowgroup | 19fbe52377 89cf0ba9a1 (#5 #9) | none |
| P6 | amd/dsv4-dense-fp8-mxscale | 6069a730c5 (#10) | environ.py: dropped #1/#8 env vars, kept the 2 MXSCALE vars |
| P7 | amd/dsv4-intree-gemm-tune | dc7fc758e6 (#13) | deepseek_v4.py: dropped set_dense_mxscale lines (belong to P6) |
| P8 | amd/rocm-dockerfile-aiter-pin | 4c4a688866 (#11) | none |

## Facts (checked 2026-10-13 on crsuse2-m2m-214)

- Source: `Duyi-Wang/sglang@amd_mori_1012_dsv4opt2`, 14 linear commits over origin/main
  (merge-base d1357bf87b, origin/main 40f5d9411f).
- Upstream PRs #41383 #41905 #41900 #41903 #43073 (and router #42027 #43483) are all OPEN, not merged,
  on other people's forks (AMD-yanfeiwang, RolaoDenthu).
- aiter main has #6126, #6145, #5967, #5862, #6259; 02575f1c3a (Dockerfile pin) contains #6126/#6145.
  NOT merged in aiter: #5882, #6004, #6029, #6030, #6042.

Conflict probe on origin/main 40f5d9411f:
- clean alone: e144f84055 5e256859ef 04627b429a 6eccfd9b35 19fbe52377 4c4a688866 5a25286422
- 1e32269bf0 needs e144f84055 (#41383) -> clean when stacked
- 3795b59ec3 needs e144f84055 + 1e32269bf0 -> clean when stacked
- 89cf0ba9a1 needs 19fbe52377 (#43073) -> clean when stacked
- 6069a730c5 conflicts only in environ.py (env-var block, expected trivial)
- dc7fc758e6 conflicts in models/deepseek_v4.py (to check: likely adjacent to 6069a730c5 edits)

## Task table

| T | commit(s) | branch (HaiShaw/sglang) | depends on | aiter needed | test |
|---|-----------|-------------------------|------------|--------------|------|
| T0 | - | setup: worktree per branch, `git fetch origin main` | - | - | - |
| T1 | 1e32269bf0 | amd/dsv4-fp8-asm-decode-split-plan | #41383 | #6126 (merged) | test_dsv4_unified_fp8_decode.py + GSM8K + decode bench |
| T2 | 89cf0ba9a1 | amd/dsv4-c4-indexer-rowgroup | #43073 | #6145 (merged) | test_fp4_indexer_hip.py + bench_dsv4_c4_indexer_rowgroup_hip.py + GSM8K |
| T3 | 6069a730c5 | amd/dsv4-dense-fp8-mxscale | - | aiter main | test_dsv4_dense_mxscale_gfx95.py + GSM8K + bench |
| T4 | 4c4a688866 | amd/rocm-dockerfile-aiter-pin | - | 02575f1c3a | docker build (gfx950) |
| T5 | 3795b59ec3 | amd/dsv4-flydsl-v4nm-intree (HOLD?) | #41383, T1 | replaces aiter#5882 | test_flydsl_v4nm_decode.py |
| T6 | dc7fc758e6 | amd/dsv4-intree-gemm-tune (HOLD?) | T3? | replaces aiter#6004/6029/6030 | test_aiter_intree_tune.py |
| T7 | - | write PR bodies (template rule), user review, then open PRs | T1-T6 | - | - |

## Open questions for user
- Q1: stacked deps (T1 on #41383, T2 on #43073, both unmerged): include the dependency commit in our branch
  (PR shows both diffs, note "depends on #N") or wait for the dependency to merge?
- Q2: T5/T6 vendor aiter code in-tree (5271-line generated kernel; tuned csv/json). Upstream or wait for aiter?
- Q3: rows 1-5 (existing PRs): leave to their owners, or also mirror rebased branches to HaiShaw?
  Commit 5a25286422 (MegaMoE stage1 ready-order fix) and router 6/7 are not in the table: excluded?
