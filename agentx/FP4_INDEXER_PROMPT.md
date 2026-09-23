# Session prompt — integrate sglang #37353 (AMD FP4 indexer) and test at c128 on DP+TBO

Paste everything between the markers as the first message of a new session.

--------------------------------- BEGIN ---------------------------------

Read `/workspace/claude-skills/NEW_WORKSPACE_PROMPT.txt` and adopt its rules for
this whole session (token discipline, capped tool output, batched calls, docs
as durable state). Then read these two, in one batched message, and nothing
else to start with:

- `/workspace/claude-skills/agentx/AGENTX_20260901.md` — the node's source of
  truth. Read the `CONTINUE HERE` block and the section
  `FP4 indexer (#37353) — integration survey`.
- `/workspace/claude-skills/agentx/mori_mxfp8_recvbound_c64.sh` — the shape a
  working arm script has on this node, including the process-cleanup preamble.

Do NOT reconstruct prior chat history; the doc is the source of truth. Do not
read `/workspace/claude-skills/agentx/SKILL.md` — it describes a destroyed node
and its results section is historical only.

## The task

Integrate [sglang#37353](https://github.com/sgl-project/sglang/pull/37353) (AMD
FP4 indexer for DeepSeek V4 on gfx95) into this environment, then benchmark it
at **concurrency 128 on DP+TBO** against a matching baseline.

Nothing has been applied yet. The survey section of the doc lists the four
pieces that must come in (the PR itself, #36581, aiter#5034 = `8578af1`,
aiter#5126) and why the PR's `docker/rocm.Dockerfile` route is inert here.

## Answer these before touching any code

1. **Does the FP4 indexer work with TBO?** The PR's first commit says TBO is
   not supported; no later commit mentions it. The target is DP+TBO, so this
   decides whether the task is even possible as stated. Answer from the source,
   not from the PR description. If TBO is genuinely unsupported, say so and
   propose the alternative (e.g. c128 DP without TBO) rather than running
   something that silently takes a wrong path.
2. **How far behind is aiter?** Its HEAD is 163 commits behind main. Determine
   whether the two cherry-picks are enough or whether the adapter needs a full
   aiter upgrade. A full upgrade means a rebuild and invalidates the three arms
   already measured, so it needs the user's approval first.
3. **What actually conflicts** between the fork branch and sglang HEAD
   `52e1c24744`.

## Ground rules for this node

- Apply upstream changes to the **working tree** with `git show <sha> | git apply`,
  not cherry-pick: both `/sgl-workspace/aiter` and `/sgl-workspace/mori` carry
  uncommitted patches (aiter#4954, aiter `dsv4_ep_tune`, mori#600) plus other
  people's local edits. A `git checkout`/`stash` in either repo destroys them.
- **The launcher never kills its own server.** Before every arm, kill three
  process shapes — `python3 -m sglang.launch_server`, `sglang::tokenizer_worker:*`,
  `sglang::router` — or the next arm dies with `Address already in use` or
  `port_base at 9123 is not available`. A clean `rocm-smi` is not evidence the
  node is free: the workers hold ports without holding VRAM. Never use a bare
  `pkill -f` with a pattern that matches your own command line.
- Full log to a file, only markers to the chat. Verify a running arm early
  rather than at the end.
- **Show the launch command to the user for review before starting any arm.**
  The user asked for this explicitly.

## What a valid result looks like

- Gates first: `errors=0`, `records_error_dropped=0`, `duration_s>=900`,
  aiperf coverage ok. Report with
  `python3 /workspace/claude-skills/agentx/arm_report.py <arm> [<baseline>]`.
- ISL matched within 3 % between the pair, cache hit flat.
- The PR reports **+5.8 %** at conc 48. That is below the 10 % bar and near the
  5.67 % replicate spread, so a single unreplicated pair cannot resolve it.
  Plan for a replicate and say so up front.
- Add finished arms to `ROWS` in
  `/workspace/claude-skills/agentx/summary_table.py` and regenerate the table.
- Write results and decisions into `AGENTX_20260901.md` as you go, keeping its
  `CONTINUE HERE` block current.

Start by reading the two files above and answering the three questions.
Acknowledge briefly, then work.

---------------------------------- END -----------------------------------
