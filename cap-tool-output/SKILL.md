---
name: cap-tool-output
description: >-
  Hard mechanical caps on how many bytes each tool call is allowed to return. Use alongside
  reduce-conversation-usage (which covers session/doc strategy); this one covers the per-command
  layer: width caps, byte caps, count-before-list, scoped find/ls/git, one-channel logging, and
  filtering INSIDE background commands, and capping the length of your own replies. Apply on
  every command whose output size you cannot predict — logs, find/ls/grep over build trees, git
  status/diff, JSON/API responses, server startup — and before sending any reply.
---

# Cap Tool Output

`reduce-conversation-usage` tells you *what to keep out of the chat*. This skill is the mechanical
layer under it: **every tool call must have a hard upper bound on bytes returned, chosen before you
run it.** "I'll grep for the interesting lines" is a plan, not a bound.

## The cost model — why early dumps are the expensive ones

Context is re-sent on **every subsequent turn**. A byte printed on turn 3 of a 40-turn session is
paid ~37 more times. So:

```
cost ≈ bytes_printed × turns_remaining
```

Consequences that change behaviour:

- **A dump early in a long session is far worse than the same dump at the end.** Be strictest at the
  start, when you're exploring and output size is least predictable.
- **Two cheap calls beat one unbounded call.** Measuring first (`| wc -c`) then fetching narrowly is
  almost always cheaper than one dump you then ignore.
- **You cannot un-print.** There is no way to remove bytes from context short of compaction. The
  only control point is *before* the call.

## Rule 1 — cap width, not just line count

`tail -30`, `head -20`, `-n 50` bound the number of lines and say **nothing** about bytes. One line
can be enormous.

Measured, from a real SGLang startup log in this workspace:

| What | Chars | ≈ Tokens |
|------|-------|----------|
| The single `server_args=ServerArgs(...)` line | **16,980** | ~4,200 |
| Entire rest of that 42 KB log, as `tail -30` | 2,696 | ~670 |

One line was **6× the cost** of the 30-line tail it arrived in. `tail -30 server.log` looked like a
bounded command and was not.

**Always compose both bounds:**

```bash
tail -n 30 run.log | cut -c1-200          # ≤ 30 lines AND ≤ 200 cols
rg "PASS|FAIL|Error" run.log | cut -c1-200
```

Structured-config / argparse dumps (`ServerArgs`, `TrainingArguments`, `argv`, JSON on one line) are
the usual offenders. If you need one field, extract that field — don't print the record:

```bash
rg -o "mem_fraction_static=[^,]*" server.log        # not: rg "server_args" server.log
```

## Rule 2 — an unconditional byte cap on anything unfamiliar

Width+line caps require you to predict output *shape*. When you can't, cap bytes directly:

```bash
<any command> 2>&1 | head -c 2000
```

This is the highest-leverage habit in this skill because it needs no prediction. Make it the default
suffix for any command you have not run before in this session. If 2000 bytes turns out to be too
few, you spend one more cheap call — versus one 100 KB dump you cannot take back.

## Rule 3 — count before you list

Ask for the **answer**, not the data. Most exploratory questions are counts or booleans:

```bash
find . -name "*batch_prefill*" | wc -l        # 40 → decide how to narrow
rg -c "sink_ptr" aiter/ops/mha.py             # count, not the matches
test -f "$MODEL/config.json" && echo yes      # boolean, not ls -la
```

Measured in this workspace: `find . -name "*batch_prefill*"` in the aiter tree returns
**104,671 chars (~26k tokens)** because JIT build directories hold very long generated filenames.
`| head -40` still cost ~8 KB and answered nothing. `| wc -l` costs 3 bytes and tells you to narrow
the pattern — which is the actual next step.

Only list once the count says listing is safe (say, < 30 short entries).

## Rule 4 — scope every search; never search from `/`

`find / ...` is both a token risk and a wall-clock risk — it hit the 120 s tool timeout in this
workspace and returned nothing useful.

```bash
# bad
find / -name "fmha_batch_prefill_api.cpp" 2>/dev/null

# good — scope, depth-limit, prune generated trees, cap
find /sgl-workspace/aiter -maxdepth 4 -name "fmha_*_api.cpp" \
  -not -path "*/jit/build/*" -not -path "*/3rdparty/*" | head -20
```

Prune `build/`, `jit/`, `.git/`, `3rdparty/`, `__pycache__/`, `node_modules/` by default — generated
trees are where the pathological filename lengths live. Prefer `rg --files -g 'pattern'` over `find`
in a repo; it respects `.gitignore` for free.

## Rule 5 — bounded git

`git status --short` in a dirty monorepo is ~1.2 KB here and can be far worse; `git diff` with no
path is unbounded.

```bash
git diff --stat -- <path>          # shape first, always
git diff -- <specific/file.py>     # then the one file
git status --short | head -20
git log --oneline -5               # never bare git log
```

Never run bare `git diff` / `git show` on a repo you haven't sized with `--stat` first.

## Rule 6 — one channel per log

Watching one file through several mechanisms multiplies the *same* bytes into context. Pick exactly
one:

- a `Monitor` on a filtered `tail -f`, **or**
- a periodic filtered `tail`, **or**
- reading the file at the end.

Do not run a Monitor and also poll the same log — you pay for every matching line twice, and the
duplicate adds no information.

## Rule 7 — filter INSIDE the background command

You cannot filter a background task's output after the fact: `TaskOutput` returns what the command
printed. So the filter has to be part of the command.

```bash
# bad — the filter comes too late; TaskOutput replays everything
<long job> > /tmp/job.log 2>&1        # then TaskOutput → full dump

# good — full log to disk, only markers ever reach context
<long job> > /tmp/job.log 2>&1
echo "exit=$?"; rg "PASS|FAIL|Error|Traceback" /tmp/job.log | tail -20 | cut -c1-200
```

Write the whole log to a file **and** print only the verdict. The file stays available for a
targeted follow-up query; the chat only ever sees the verdict.

## Rule 8 — extract fields from structured responses

Never dump a JSON body. Parse it and print the fields:

```bash
curl -s "$URL" | python3 -c "import json,sys; d=json.load(sys.stdin); \
  print(d['choices'][0]['message']['content'][:300]); print('usage:', d['usage'])"
```

Same for `nvidia-smi` / `rocm-smi` / `pip list` / `env` — filter to the rows you asked about.

## Rule 9 — don't pay for confirmation you already have

- After a successful `Edit`/`Write`, **do not re-read the file**. A failed edit is an error; silence
  is success.
- Don't re-`ls` a directory you just created, or re-run a command to "make sure".
- Edit large files in place (`Edit`, `sed -i`, small `python3 -` scripts) — never `Write` a whole
  large file back to change a few lines.

## Rule 10 — put throwaway scripts in files

A heredoc you run inline is echoed back to you in the tool result. For anything you'll run more than
once, write it to `/tmp/x.py` once, then re-run it by path — subsequent runs cost only their output.

## Rule 11 — your own replies are context too

Every rule above caps what *tools* put into context. Your own output lands in
the same place, under the same multiplier — and it is billed twice: once at the
higher output-token rate when you generate it, then as input on every later
turn. A long reply is often the largest single cost in a session, and it is the
one you control most directly.

**Prose by default; tables only to compare.** A markdown table costs roughly
2-3x the same content written as sentences — borders, padding, a header row,
and one line per item whether or not the item needs one. Use a table only when
the reader must compare 3+ items across 2+ dimensions. Two numbers belong in a
sentence.

**Do not restate the tool result.** The user saw the command and its output.
Report what it *means* and what changed, not a transcription of it.

**One line for success; detail only for failure.** "All 11 frontmatter blocks
parse" beats eleven `OK` rows. If something failed, then show the failure —
that is when detail earns its cost.

**Do not re-list the backlog every turn.** Outstanding items belong in a
message when they change, or when asked. Repeating an unchanged to-do list on
every turn pays for it again each time, and trains the reader to skip it.

**No closing recap.** If the body already said what you did, a summary section
that says it again is pure duplication.

**Match length to the decision.** Ask what the user has to decide next. A
yes/no answer is a sentence. Headings, tables and sections on a three-line
answer are noise — and noise you re-send forever.

Symptom to watch for: you are capping every command at 200 columns and then
writing a 60-line reply about it. The discipline is not real until it applies
to your own output.

## Pre-flight checklist

Before **every** command whose output size you can't state as a number:

- [ ] Do I need the data, or just a count / boolean? → `wc -l`, `-c`, `test`
- [ ] Is there a width cap? → `| cut -c1-200`
- [ ] Is there an outer byte cap? → `| head -c 2000`
- [ ] Is the search scoped and pruned? → no `/`, `-maxdepth`, `-not -path`
- [ ] Long job → full log to a file, only markers printed?
- [ ] Am I already watching this log through another channel?
- [ ] Am I re-reading something I just wrote or already have?

And before sending the reply itself:

- [ ] Would prose say this more cheaply than a table?
- [ ] Am I restating output the user already saw?
- [ ] Is success collapsed to one line, with detail reserved for failures?
- [ ] Am I re-listing an unchanged backlog, or adding a recap of my own body?

## Escalation ladder

Go **down** this ladder only when the step above genuinely didn't answer the question:

1. `wc -l` / `-c` / `rg -c` — how much is there?
2. `rg -o "<narrow pattern>"` — just the token you need
3. `rg "<pattern>" | head -20 | cut -c1-200` — the matching lines
4. `sed -n 'A,Bp' file` — one targeted range
5. Read the whole file — rare, and only for small files

Most exploration should terminate at step 2 or 3.

## Relationship to `reduce-conversation-usage`

Complementary, not overlapping:

| Skill | Layer |
|-------|-------|
| `reduce-conversation-usage` | Session strategy — docs as source of truth, handoff to fresh chat, prune debug noise, compact |
| `cap-tool-output` (this) | Per-call mechanics — width/byte caps, count-first, scoped search, one channel, filter-in-background |

Apply both. This one is what you check immediately before pressing enter on a command.
