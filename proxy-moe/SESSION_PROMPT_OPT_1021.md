# Session prompt: continue proxy MoE optimization on crsuse2-m2m-231 (paste as the first message)

```text
Read /workspace/claude-skills/NEW_WORKSPACE_PROMPT.txt and follow it, answer in Chinese.
The main checkout /workspace/claude-skills is blocked by 2 unrelated unmerged files (not ours; do
not resolve them). Use the worktree instead: cd /data/proxy_moe/wt/skills && git fetch origin &&
git merge --ff-only origin/main. Edit, commit and push docs from there (no AI attribution lines).
Read ONLY (rg is not installed; use grep -n '^## ' first, then sed ranges):
  proxy-moe/SKILL.md
  proxy-moe/OPT_crsuse2-m2m-231.md   sections "CONTINUE HERE" (status, prioritized Next, Rules,
                                     Repro), "TTT/GPU + P90 interactivity", "T11", "Item A",
                                     "Items B (moe1 streaming) and D", "T3 notes", "T3 results"
  proxy-moe/fixed_seq_len/RESULTS.md sections "Observations", "70000/300: --chunked-prefill-size",
                                     "Comparing with other platforms", "Geomean MI355X / GB300"
Read perf-bottleneck-attribution/SKILL.md before claiming any bound.
Do not reconstruct prior chat history; the docs are the source of truth.
Node: crsuse2-m2m-231 (check `hostname`). Owner of OPT_crsuse2-m2m-231.md = this node.
Environment is already set up on this node: new base via `source /data/proxy_moe/newbase_env.sh`
(worktrees /data/proxy_moe/wt/{aiter,sglang} = pushed feat/proxy-model-1020 branches, FlyDSL 0.4.0
in /data/proxy_moe/pyenv_new), checkpoints /data/proxy_moe/proxy_24l and proxy_12l, drivers
/data/proxy_moe/repro.sh and proxy-moe/fixed_seq_len/sweep_proxy.sh, tools /data/proxy_moe/tools
(backup /shared_nfs/kk/proxy_moe_node_backup_c231). If /data is gone (container reclaimed),
rebuild per OPT doc ("Environment on this node", T11) before running anything.
First: check GPUs are free (rocm-smi) and nothing else runs (ps for sglang).

Task: continue the optimization from "CONTINUE HERE → Next" in priority order. Start with item 1
(F / T3: prefill AR/compute overlap) by writing a short design + measurement plan into the OPT doc
(where the micro-batch split goes in models/laguna.py, how overlap is verified in a trace, expected
gain from the measured EP1 prefill comm share), then ask me before large code changes. Item 2
(CONC=1024 ITL jump) can be measured in parallel planning, not in parallel runs (run_stage.sh and
sweep teardown kill every sglang server: experiments run sequentially).
One variable per experiment; record every run (node + date) in the OPT doc Results table; ask
before pushing code branches or opening upstream PRs.
```
