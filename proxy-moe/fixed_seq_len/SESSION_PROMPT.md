# Session prompt: proxy MoE fixed_seq_len sweep (paste as the first message)

```text
Read /workspace/claude-skills/NEW_WORKSPACE_PROMPT.txt and follow it, answer in Chinese.
Then: cd /workspace/claude-skills && git fetch && read ONLY (from origin/main if the main checkout
is still blocked by unrelated unmerged files; the editable checkout is the worktree
/data/proxy_moe/wt/skills, commit + push from there):
  proxy-moe/SKILL.md
  proxy-moe/fixed_seq_len/README.md          (CONTINUE HERE, shapes, recipes, traps)
  proxy-moe/OPT_crsuse2-m2m-231.md           sections "CONTINUE HERE" and "TTT/GPU + P90 interactivity"
Do not reconstruct prior chat history; the docs are the source of truth.
Node: crsuse2-m2m-231 (check `hostname`). Owner of fixed_seq_len/README.md = this node.

Task: run the proxy MoE fixed_seq_len sweep (1024/1024, 8192/1024: CONC 2..1024; 70000/300:
CONC 2..32; recipes tp8ep1 (24L, 8 GPUs) and tp4ep1 (12L, GPUs 0-3), 50 points, ~4-6 h) exactly as README.md "Repro" says:
background nohup, record the PID, monitor by DONE count + log mtime (one channel; a quiet monitor
is not proof of progress). Check GPUs are free first (rocm-smi). Do not change server flags
mid-sweep. When all points are DONE: `python3 report.py --md`, write fixed_seq_len/RESULTS.md
(config table + per-recipe tables + warnings + notes), commit + push (no AI attribution lines),
and tell me the headline numbers. Optimizations are discussed only after the sweep.
```
