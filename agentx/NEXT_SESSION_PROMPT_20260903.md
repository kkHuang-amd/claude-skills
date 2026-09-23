# Session prompt — 2026-09-03 (paste everything below the line)

---

Read `/workspace/claude-skills/NEW_WORKSPACE_PROMPT.txt` and adopt its rules for
this whole session (token discipline, capped tool output, batched calls, docs as
durable state). Then read these two, in one batched message, and nothing else to
start with:

- `/workspace/claude-skills/agentx/ITL_GAP_FINDINGS.md` — read the
  **`CONTINUE HERE`** block at the top in full, plus §3 and §4b.
- `/workspace/claude-skills/agentx/DATA_AND_ANALYSIS_20260902.md` — read **§1**
  (the arm table), **§6 item 1**, **§9** and **§10**. Skip the rest for now.

Do NOT read `SKILL.md` (129 KB, describes a destroyed node) and do not
reconstruct prior chat history.

`AGENTX_20260901.md` belongs to a **concurrent session** working the OOR/mem-frac
track. Read its `CONTINUE HERE` only if you touch that track, and never rewrite
the file — an earlier session's edits to it were silently overwritten. Put your
own results in `ITL_GAP_FINDINGS.md`.

## Where the investigation stands

The question was: why is SGLang's ITL p90 ~27 % worse than ATOM's on DP
attention at matched concurrency? **It is answered.** One scheduler flag,
`--prefill-decode-interval 10 → 20`, moved c128 ITL p90 from 78.39 to
**58.78 ms**, past ATOM's 61.5 ms, with throughput +4.43 % in the same direction.
The cost is TTFT: 8.50 → 13.22 s, which overshoots ATOM's 10.9 s.

So the gap was an **operating point on a prefill-versus-decode trade-off**, not a
decode-execution deficit. Both pre-registered criteria were tested: ITL passed,
TTFT failed, and the optimum is between 10 and 20.

## Task 1 (priority): find the interval that matches ATOM on both axes

`cp interval20_c128.sh interval15_c128.sh`, change two things only —
`EXTRA_SERVER_ARGS="--prefill-decode-interval 15"` and
`RESULT_DIR=/workspace/results/interval15-c128` — then

```bash
mkdir -p /workspace/results/interval15-c128
nohup bash /workspace/claude-skills/agentx/interval15_c128.sh \
  > /workspace/results/interval15-c128/run.log 2>&1 &
```

~92 min wall clock (3 min server start, ~28 min aiperf dataset config + warmup,
60 min measurement, ~4 min export). The script self-verifies at the end and runs
`arm_report.py interval15-c128 dptbo-c128`.

**Pass criterion, scored as a pair:** ITL p90 ≤ 61.5 ms **and** TTFT avg ≤ 10.9 s
**simultaneously**. That is the reportable result — "SGLang matches ATOM on both
axes once the prefill/decode split is tuned". If 15 lands ITL under and TTFT
over, try 12; if both under, the interval is not the binding constraint above 15
and you can stop.

Verify in the arm's own output that the override took: the resolved command
carries the launcher's hardcoded `10` **and** your value, and argparse keeps the
last. `server.log`'s `server_args` must show `'prefill_decode_interval': 15`.

## Task 2 (user's, deferred from 2026-09-02): c192 + hicache

Separate problem from task 1. At c192 throughput regresses 27 % and TTFT avg
hits 59.71 s. §9 has the measured mechanism; the short version is that the
**miss rate rose 36 %** (8.60 → 11.70 % of context tokens — "only 3 pp of hit" is
36 % of the quantity prefill cost is proportional to), total prefill work rose
16 %, and c160 had no spare capacity, so admission fell below the offered load
and TTFT became queue wait. Per-batch prefill work is **flat** (+0.9 %) — the
prefills did not get bigger, there are just more of them needed.

The one lever with evidence is the CPU cache tier: a c256 run on the previous
node had cache hit 94.4 % with the **GPU tier at only 66.3 %**, i.e. a third of
all prefix reuse came from CPU. hicache has been off in **every** arm on this
node.

1. **Smoke first, `DURATION=300`.** hicache was skipped because #37353's rust
   `DeepseekV4C4IndexerScale` pool-name change was not applied. That pool is
   FP4-specific, so running with **FP4 off should avoid needing the rust
   rebuild** — that is an inference, not a verified fact. 30 min to find out
   beats losing 1.5 h.
2. Then the full arm: `CONC=192`, `KV_OFFLOADING=hicache`, FP4 off. Launcher
   support is at `b200align_mtp.sh:113-124` (`HICACHE_RATIO` 1.5,
   `write_through`, `direct`, `page_first_direct`); it also needs
   `TOTAL_CPU_DRAM_GB` set.
3. **Pass criterion:** recompute `Σnew / Σ(new + cached)` from the `Prefill
   batch` lines in `server.log`; it must fall from **11.70 %** below **8.6 %**
   (c160's level), and TTFT must follow. If the miss rate falls but TTFT does
   not, prefill demand is not the driver and the mechanism in §9 is wrong.

## Settled — do not spend an arm re-litigating these

1. **TBO stays on.** Measured: TBO off is worse on ITL (83.93 vs 78.39 ms), TTFT
   (+19 %) and tok/s (−3.23 %). Both engines' TBO is prefill-only by
   construction, so it was never a decode-side confounder.
2. **Do not profile a decode step.** It was the planned fallback, but a scheduler
   flag moved ITL by 20 ms — more than the whole gap. The pure-decode step is
   *faster* at c192 than c160 (92.5 vs 109.8 ms) because the batch is smaller.
   Decode is not slow; it was being interrupted.
3. **Do not re-run the chunk-size diagnostic.** The previous node's c256 pair
   already measured it: 16,384 → 8,192 gives ITL −25 % and TTFT +147 %.
4. **The c192 collapse is not eviction and not memory pressure.** `full token
   usage` median is 0.36 at c192, *lower* than c160's 0.40. Occupancy went down.
5. **ATOM's `run 33074134043` is unreachable** — ROCm blocks all classic PATs
   (403, two tokens tried). It only decides whether to quote +27 % or +36 %.

## Traps this node has actually sprung

- **The node is shared.** `interval20_c128.sh` waits for three consecutive idle
  checks and **refuses** rather than killing. Keep that. A blind kill preamble
  destroyed another session's c96 arm at 05:12:25 on 09-02, and the giveaway was
  misread as "my own leftover processes". Check `ps -eo pid,lstart,args` and
  compare against your own arm's end time before killing anything.
- **Never `pkill -f` a pattern matching your own command line.** `pkill -f
  'sglang::'` killed the shell running it, mid-command.
- **The tree carries 19 uncommitted files belonging to other sessions**
  (`pyproject.toml` ×2, a deletion, 16 untracked `kernels/aot/csrc` HIP files).
  `git reset --hard` destroys them. To undo the two local commits use
  `git reset --soft 52e1c24744 && git reset`, or `git revert 33979a814b` for just
  the fix. Backup at `/workspace/tree-backup-20260902-072044/`.
- **Every FP4 number in §1 is stale** — `33979a814b` changed the FP4 scoring
  path. Re-measure before quoting, and state the SHA with new comparisons. The
  non-FP4 path is behaviourally unchanged (verified: `should_use_topk_v2()` and
  the old `envs.SGLANG_OPT_USE_TOPK_V2` condition are same-valued here).
- **`ITL p90` is per-request TPOT**, p90 taken across requests — not the p90 of
  token gaps. It cannot show a bursty tail.
- **Metric traps that have already produced wrong conclusions here:** "% of
  windows containing a prefill" is arithmetic on `decode_log_interval=40` vs
  `interval=10`, not a finding; the 0-prefill bucket in §4 is load-biased, so use
  it for direction only; a *conditional* median (cached-token over non-zero
  batches) moved +60 % while the mean moved −14.5 % — always check which
  population a median is over.
- **A dead server keeps answering `GET /metrics` with 200** and aiperf then waits
  forever. Verify an arm early, not at the end.
- **Every arm carries `vram_sampler.sh`** and the md5 + SHA snapshot. The OOR
  cliff is still live on the fp8 path: `interval20-c128` ran with min free VRAM
  **0.98 GB** and 6 late Triton device loads. Root cause and the outstanding fix
  (mirror `b6e3728`'s bounded buffer into `_aiter_fp8_paged_mqa_logits`,
  `dsv4/indexer.py:160`) are in `ITL_GAP_FINDINGS.md` §3.

## What a valid result looks like

- Gates first: `errors=0`, `records_error_dropped=0`, `duration_s` 3500–3750,
  aiperf coverage ok. Report with
  `python3 /workspace/claude-skills/agentx/arm_report.py <arm> [<baseline>]`.
- ISL matched within 3 % between a pair; cache hit flat.
- **Replicate spread on this node is 5.67 %** — a smaller throughput delta is
  null. ITL deltas of 25 % are far outside it and are the robust signal.
- Keep `ITL_GAP_FINDINGS.md`'s `CONTINUE HERE` current as you go.

## Node state at handoff (verified 2026-09-03 06:53 local)

Clean and idle: 0 matching processes, 2.2 GB across 8 GPUs. `sglang` HEAD
`33979a814b`, 19 files uncommitted (not yours). Scripts to copy:
`interval20_c128.sh`, `vram_sampler.sh`, `tbo_debug_probe.sh`. Tools:
`arm_report.py`, `summary_table.py`, `cache_tier_gate.py`,
`decode_stall_split.py`, `wait_and_launch.sh`.

Start by reading the two documents above, then state your plan for task 1.
