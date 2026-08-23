---
name: reduce-conversation-usage
description: >-
  Reduce agent conversation / context / token usage on long, multi-turn, or expensive tasks —
  large command or GPU/build logs, repeated file reads, kernel/debug/benchmark loops. Keep durable
  state in project docs (not the chat), filter command output, read narrowly, batch tool calls,
  prune debug noise, and hand off to a fresh chat via docs. Apply proactively whenever a session is
  long-running or token-heavy (many tool calls, big outputs, repeated iteration).
---

# Reduce Conversation Usage

Six methods to keep a session's context/token usage low. Apply them proactively on long or
expensive tasks; the goal is to keep the working context small so the session stays fast and cheap.

## 1. State lives in docs, not the chat — hand off to a fresh chat
- Write durable results/decisions/next-steps into a **project doc** (e.g. a `README.md` /
  `HANDOFF.md` / design doc), not just the conversation.
- When history grows large, **start a new chat** whose first message points at the doc, e.g.
  "read `path/to/DESIGN.md` §N and continue". A fresh chat + one doc read ≈ tiny context vs a huge
  thread.
- Keep a short **"CONTINUE HERE"** block at the top of the relevant doc (status + exact next step +
  which files/functions + the repro command) so resuming costs one read.

### CONTINUE HERE template

```markdown
## CONTINUE HERE

**Status:** <one line — done / blocked / in progress>
**Next:** <exact action, e.g. rerun C4 endpoint after JIT warmup>
**Files:** `path/to/file.py` (`function_name`), env vars if any
**Repro:**
\`\`\`bash
<single copy-paste command>
\`\`\`
**Golden / pass criteria:** <numbers or thresholds>
```

### Fresh-chat resume prompt

```text
Read <path/to/HANDOFF.md> and continue. Apply skill reduce-conversation-usage.
Do not reconstruct prior chat history — the doc is the source of truth.
```

## 2. Filter command / log output — never dump full logs into context
- Always post-process noisy commands: `... 2>&1 | rg "PASS|FAIL|relL2|Error|<marker>"` or
  `| tail -n 30`. Redirect the full log to a file (`> /tmp/run.log`) and `rg` only what you need.
- Especially for GPU/build/test runs and servers: grep for result/error lines, not the whole output.

### Common rg patterns (GPU / server / benchmark)

| Task | Pattern |
|------|---------|
| SGLang server | `ready to roll\|Initialization failed\|Traceback\|max_total_num_tokens` |
| bench_serving | `Successful requests\|Total token throughput\|Mean TPOT\|Benchmark duration` |
| GSM8K / accuracy | `Accuracy\|gsm8k\|invalid\|PASS\|FAIL` |
| Build / test | `PASS\|FAIL\|Error\|relL2\|FAILED` |

Example — server startup (full log to file, only markers in context):

```bash
python -m sglang.launch_server ... > /shared_nfs/kk/server.log 2>&1 &
rg "ready to roll|Initialization failed|Traceback|max_total_num_tokens" /shared_nfs/kk/server.log
```

Example — endpoint sweep (per-case log files, grep summaries only):

```bash
for c in 2 4 8 16 32; do
  ... --output-file "$RESULT_DIR/c${c}.jsonl" > "$RESULT_DIR/c${c}.log" 2>&1
done
rg "Successful requests|Total token throughput|Mean TPOT" "$RESULT_DIR"/*.log
```

## 3. Read narrowly; don't re-read
- Prefer `Grep`/`Glob` for specific symbols over reading whole large files.
- Read targeted line ranges (`offset`/`limit`), not entire 500+ line files.
- Don't re-read a file already in context.

## 4. Fewer, more decisive turns; batch tool calls
- Make independent reads/greps/edits in **one message** (parallel tool calls) instead of many
  sequential round-trips.
- Decide and act rather than re-deliberating across turns; when the user gives a clear directive,
  execute it directly. Ask a question only when genuinely blocked.

## 5. Prune temporary debug noise
- Remove temporary `printf`/debug prints and disable verbose per-step logging (env flags) once
  they've served their purpose — they bloat every subsequent log you read back.

## 6. Compact or restart when the thread is large
- Use the client's compact/summarize when available; otherwise start a fresh chat and resume via
  the doc handoff (method 1). Treat the docs as the source of truth, the chat as scratch.

## Quick check before a long loop
- [ ] Results/next-step captured in a doc (not only chat)?
- [ ] Command output filtered (rg/tail), full log to a file?
- [ ] Reads are narrow (grep / line ranges), no re-reads?
- [ ] Independent tool calls batched in one message?
- [ ] Temp debug prints/verbose logging removed?
- [ ] Thread very large → hand off to a fresh chat via the doc?

## Supplement: artifact paths (this workspace)

Keep chat lean; put bulky artifacts on disk:

| Kind | Path |
|------|------|
| Durable docs / handoffs | `/workspace/claude-skills/<project>/` |
| Large logs, traces, run artifacts | `/shared_nfs/kk/` |
| Models | `/shared_nfs/models/` |

## Supplement: long-running jobs

- Redirect stdout/stderr to a file under `/shared_nfs/kk/` or `/tmp/`.
- Poll with `rg` on the log file — never read the whole log back into context.
- For background shells, check readiness via filtered patterns, not full terminal dumps.
- If one benchmark case is an outlier (e.g. first-run JIT), **rerun that case alone** and record
  both runs in the doc — don't paste both full logs into chat.

## Supplement: GPU cleanup (after server/bench)

After server or benchmark runs, verify VRAM is released before starting the next experiment:

```bash
rocm-smi --showmeminfo vram | rg "GPU\[|Used"
# Kill by numeric PID from rocm-smi --showpids; avoid broad pkill -f (can hit self)
for p in $(rocm-smi --showpids | awk '/^[0-9]/{print $1}'); do kill -9 "$p"; done
```

## Changelog

- **2026-08-18:** Synced to `~/.cursor/skills/`; added CONTINUE HERE template, common rg patterns,
  artifact paths, long-running job notes, GPU cleanup snippet.
