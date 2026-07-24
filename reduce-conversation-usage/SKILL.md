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

## 2. Filter command / log output — never dump full logs into context
- Always post-process noisy commands: `... 2>&1 | rg "PASS|FAIL|relL2|Error|<marker>"` or
  `| tail -n 30`. Redirect the full log to a file (`> /tmp/run.log`) and `rg` only what you need.
- Especially for GPU/build/test runs and servers: grep for result/error lines, not the whole output.

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
