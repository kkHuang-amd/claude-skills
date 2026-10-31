---
name: kimi-k31
description: >-
  Bring up and optimize Kimi-K3.1 (KimiK31ForConditionalGeneration: KDA + DSA hybrid, 896-expert
  MXFP4 MoE, DSPARK spec decode) with SGLang on 8x MI355X (gfx950 only). Use when launching
  Kimi-K3.1, rebuilding aiter for it, triaging aiter PRs for K3.1, or profiling/tuning its
  prefill/decode. Repo: kevin-mii/sglang-k31-amd branch dev_amd.
---

# Kimi-K3.1 on MI355X (gfx950)

Entry point: `CONTINUE_HERE.md`. Read it first; open other files only for the task at hand.

| File | What |
|------|------|
| `CONTINUE_HERE.md` | status, next action, repro, owner node |
| `PLAN.md` | task items (bring-up -> accuracy -> perf), acceptance gates |
| `AITER_PRS.md` | aiter open-PR triage for K3.1 (fetched as `pr/<N>` refs in aiter-k31) |
| `env.sh` | `source` before every run — fixes PYTHONPATH shadowing, sets gfx950 arch |

Upstream docs inside the sglang repo (read narrowly, they are large):
`DEV_AMD.md` (4 KB, branch rules, env switches, validation) and
`docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx` §"Kimi-K3.1 on MI350X / MI355X"
(`rg -n` for the heading, then `sed -n` ~40 lines).

Related: `../kimi-k3/` (K3 predecessor; DEV_RULES and trace-capture method still apply).

Optimization principles (apply to every op; detail in `OPT_TRACKER.md`): per-layer -> per-step,
no unnecessary host sync, fuse small kernels, overlap/merge independent ops, compare to roofline.

Artifacts: `/shared_nfs/kimi-k31/{logs,results,traces}`. Results go under
`results/<run-tag>/`; raw traces are capture -> analyze -> summarize -> delete.
