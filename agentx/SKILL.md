---
name: agentx-inferencex-test-env
description: >-
  Build and run the InferenceX "AgentX" agentic trace-replay benchmark (aiperf
  --scenario inferencex-agentx-mvp) on a single node, using
  benchmarks/single_node/agentic/*.sh. Use when standing up AgentX from scratch,
  when a launcher aborts before the server starts (missing aiperf venv, empty
  utils/aiperf submodule, unresolved traces, INFMAX_CONTAINER_WORKSPACE wrong),
  or when wiring DeepSeek-V4-Pro FP4 + SGLang MTP on MI355X.
---

# AgentX single-node test environment

AgentX is InferenceX's *agentic coding* scenario: instead of a synthetic
fixed-ISL/OSL sweep, it replays recorded Claude-Code trajectories (multi-turn,
tool-heavy, with subagent fan-out) from a HuggingFace dataset against a live
OpenAI-compatible server, driven by a **forked AIPerf** running the
`inferencex-agentx-mvp` scenario.

The launcher scripts under `benchmarks/single_node/agentic/` are labelled
**MVP / experimental** by the repo itself — they are a reference implementation
of the plumbing, not a published benchmark.

## CONTINUE HERE

**Status (2026-08-30, 09:00 UTC): NEW MAIN IS +7.52 % AT c64 -- suggestive, n=1,
needs one replicate.** The node was restarted since the notes below: `/tmp` is
EMPTY (topk_v2 patch, all three launcher backups, `acc_driver.sh`, the gsm8k
logs, the `/tmp/*_ref` dirs and the JIT caches are all gone) and the tree is now
HEAD `cdbfe90b4a` (sglang main, `0.5.18.dev20260829`), NOT `a1f9508dd4`.
`/workspace/results/*` and `/workspace/claude-skills/agentx/*` survived.
The tree also carries someone else's uncommitted work (pyproject edits +
untracked `*.hip` kernels) -- do not revert or stash it without asking.

**topk_v2 (#36684) is now upstream and ON by default here** -- no patch needed:
`arg_groups/model_hook.py:355` does `SGLANG_OPT_USE_TOPK_V2.set(True)` in the
`DeepseekV4ForCausalLM` + `is_hip()` branch, which is exactly what the old manual
`server_args.py:5891` patch did.

| arm | tok/s/GPU | TTFT | ITL p90 | intvty p90 | cache |
|---|---|---|---|---|---|
| `c64-chunk16384-newmain` (main `cdbfe90b4a`) | **20,131.6** | 4.90 s | 49.74 ms | 20.10 | 0.955 |
| `c64-chunk16384` (old `a1f9508dd4`) | 18,724.0 | 5.32 s | 54.51 ms | 18.35 | 0.954 |
| delta | **+7.52 %** | -7.82 % | -8.75 % | +9.57 % | flat |

Both certified, ISL matched to 1.36 %, duration 3628 s both sides, **cache hit
flat** so the delta is not a prefix-cache artefact. +7.52 % is above the 5.67 %
replicate spread but under §25's 10 % bar. What makes it more than the topk_v2
non-result: **all four secondary metrics move the same way**, and ITL p90's
-8.75 % clears its own 7.01 % noise floor. Still n=1 -- replicate before quoting.
**Attribution is "new main as a whole"** (spec v2 default + upstream topk_v2 +
everything between the two commits), never spec v2 alone.

### TRAP: `EP_SIZE=1`, not 8 -- EP8 OOMs and the launcher default lies

Every arm on file runs `ep_size=1`. `agentx_b200align.sh` defaults `EP_SIZE` to
**8**, so the reproduce line's explicit `EP_SIZE=1` is load-bearing, not
decoration. An EP8 c64 run (2026-08-30, `VOID-c64-ep8-oom-20260830/`) died 11 min
into warmup: `HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Available Free mem : 388 MB`,
`scheduler_0` aborted with exit -3, every other DP rank then raised gloo
`Connection closed by peer` out of `dp_attn.py:182 all_gather`. It is not a
capacity problem and conc is not the variable -- the startup budget was
*roomier* than the arms that pass:

| arm | slots/rank | free VRAM after startup | outcome |
|---|---|---|---|
| `c128-chunk16384` (ep1) | 32 | 38.98 GB | ran |
| `c256-chunk16384-2x` (ep1) | 64 | 36.67 GB | ran 2 h |
| c64 ep8 | 16 | 41.53 GB | dead in 11 min |

EP8 eats 41 GB of headroom at runtime while KV usage sits at 0.09. If you ever
want the EP8 number, that unbounded growth is the thing to fix first.

### TRAP: a dead server keeps answering 200, and aiperf waits forever

When the schedulers die, `launch_server`'s HTTP layer **stays up**: `/health`
returns 200, the router keeps accepting, and the agentic warmup barrier (no
`--agentic-warmup-grace-period`, §"Next, in order" item 4) waits **indefinitely**
with `errors=0`. The EP8 crash above burned 95 minutes looking exactly like a
healthy quiet arm. **`errors=0` plus a frozen `returned=` is a corpse, not
health**, and the crash is written to `server.log`, NOT to the launcher stdout --
watch both.

Two monitoring tools were silently broken by the same class of bug (a phase
change and a stall look identical), both fixed 2026-08-30:
- `warmup_check.sh` stall check anchored `grep -oE "^returned="` against lines
  that begin `Phase warmup progress | ...`, so it matched nothing and printed
  **STALLED for every arm, healthy or not**. Backup: `warmup_check.sh.bak`.
- New `watch_arm.sh <arm>`: emits ONLY on stall (~20 min no progress), server
  death, completion, phase change, plus a 30-min heartbeat. Two ways to get it
  wrong, both hit on the day it was written: it must track `returned=` (warmup)
  **and** `done=/ok=` (profiling), or every phase boundary reads as a stall; and
  it must exit when the result json appears, or **every finished arm ends in a
  false STALL**. A detector with no terminating condition trains you to ignore it.


**Status (2026-08-29, ~12:00 local): ALL-OPT ARM IN FLIGHT. HEAD `a1f9508dd4`, but
the tree is NO LONGER CLEAN and that is deliberate -- topk_v2 (#36684) was
re-applied from `/tmp/topkv2_36684.patch` (3 files: `topk_v2.cuh`,
`topk_impl.cuh`, `server_args.py:5891` `is_hip()` -> `set(True)`) and the JIT
caches were swapped so `~/.cache/sglang/jit` is now the topkv2 build, with the
pre-36684 one preserved as `jit.pre36684` (both exist this time; the earlier
revert had `mv`-ed its backup away).**

**Why: the ask is the ALL-OPTIMISATIONS-IN number, so fusion and topk_v2 run
together.** That arm cannot attribute fusion on its own -- its baseline is
`c128-chunk16384-topkv2` (**26,943.3** tok/s/GPU, certified), not a clean
baseline, and topk_v2's own +3.75 % is itself inside the noise floor, so the
subtraction carries more error than either term.

**`--enforce-shared-experts-fusion` DOES work on the DP+TBO path** -- previously
untested (`agentx_ladder.sh:16`, launcher comment at line 172). V4's gate
(`deepseek_v4.py:3294`) is stricter than V2's: fusion happens ONLY with
`--enforce-...`, so the `--disable-shared-experts-fusion` we have always passed
was redundant. V4's gate has no TBO exclusion, unlike V2's.

**CAVEAT that changes how to read any fusion result: on this checkpoint fusion is
not a pure throughput change, it is a PRECISION change.** The loader prints
`Loading FP8 shared expert weights into FP4 fused MoE weights` -- routed experts
are FP4, the shared expert is stored FP8 (`config.json`'s `quant_method: "fp8"`
is not the whole story), and fusion silently down-quantises it to FP4.
`quant_blocks_shared_experts_fusion` exists to veto exactly this, but it is
implemented as `can_fuse_shared_expert` on the **Quark** config only, so it never
fires here. Any throughput delta is confounded with an accuracy delta -- pair
every fusion perf number with a gsm8k run on the same server.

Also verified from source (not assumed): `SGLANG_SHARED_EXPERT_TP1` and
`SGLANG_DP_SHARED_EXPERT_LOCAL` are inert once fusion is on. Both DSV4 call sites
(`deepseek_v4.py:2291`, `:2566`) require `self.mlp.shared_experts is not None`,
and that module is never constructed when `num_fused_shared_experts > 0`
(`deepseek_v2.py:722-727` is skipped). The one reader outside that block,
`moe_ep_setup.py:141`, is a `raise ValueError` validation only and cannot fire
here (`_use_aiter` is true, and 3072/8 % 128 == 0 anyway). Side effect worth
remembering: fusion therefore **loses** the DP-local shared-expert optimisation,
so the two arms differ by more than "fused or not".

**New launcher knob:** `FUSE_SHARED_EXPERTS=1` opts the DP branch into fusion
(default 0 = baseline). Backup: `/tmp/b200align_launcher.pre_fusion.bak`.

**To restore the pre-topkv2 tree:** `git apply -R /tmp/topkv2_36684.patch`, then
swap `~/.cache/sglang/jit` back with `jit.pre36684`.

Seven arms on disk in `/workspace/results/overnight/`. Read the summary table with
`python3 /workspace/claude-skills/agentx/summary_table.py`; compare any two arms
with `arm_report.py <arm> <baseline>`.

**Results that stand:**

1. **`CHUNK_PER_RANK` 16384 beats 8192 at c256 by +13.15 %** (30,745.5 vs
   27,171.7, both certified, ISL matched to 0.95 %) -- the only result in this
   file clearing §25's >=10 % bar. Load-dependent: null at c64, decisive at c128
   (8192 could not certify at all), +13.15 % at c256.
2. **The `MAX_RUNNING_REQUESTS` 256 clamp caused a server deadlock at c256.**
   3/3 arms hung with `errors=0` and 95 % of the KV pool free; removing the clamp
   (now 2*CONC, `MAX_RUNNING_CAP` knob, backup `/tmp/b200align_launcher.pre_maxrun.bak`)
   made c256 run clean. Single-variable attribution.
3. **topk_v2 (#36684) is NOT resolvable** -- +3.75 % headline, -3.11 % ITL mean,
   every delta under its own noise floor. Tree already reverted.

**The number that reframes everything: our headline is 18x our real compute.**
94 % of `tok/s/chip` is prefix-cache hits. At c256 we do **1,714** real
tok/s/chip; ATOM does **2,900** -- **we are 69 % behind on actual work** while
only 31 % behind on the headline. ATOM's TTFT is flat across the DP band
(10.5 -> 10.9 -> 13.4 s); ours explodes (5.32 -> 10.63 -> 38.15 s). **The c256
problem is a scaling failure in real prefill throughput and TTFT under load.**

**Three lines of attack were measured and killed** -- do not restart them without
new evidence: cache tiering (the DRAM spill moves 80 MB/s per rank = **0.32 % of
one PCIe link**), the SWA reserve (frees 82 GB but a prefix cache refills
regardless), and `cuda-graph-max-bs` (SGLang already prunes the ladder to
bs=64 = `max_num_reqs`; the 128 never took effect). See the RETRACTIONS section.

**The instrument cannot resolve small effects.** Per-metric replicate spreads:
headline 5.67 %, ITL mean 5.51 %, ITL p50 4.54 %, ITL p90 7.01 %, TTFT 26.3 %.
ITL is **not** the sensitive instrument it looks like. Resolving a 4 % effect needs
~8 runs per arm (~30 GPU-h). **Measure kernel changes with rocprof or a
microbenchmark; reserve 2-hour end-to-end arms for changes expected to exceed
~10 %.**

### ACCURACY REGRESSION — the blocker as of 2026-08-29 14:00 (overnight autonomous task)

**DP+TBO + fusion + topk_v2 scores gsm8k 0.737** (1319 q, max-new-tokens 8192,
parallel 64; `Invalid: 0.001`, so not a parse/truncation artefact -- the model
is genuinely wrong). Reference band from an earlier arm is 0.931/0.939. The
**c128 perf run is deliberately NOT launched** until this is understood; a fast
number from a broken model is worthless.

**Ordered task from the user (asleep):** make DP+TBO+fusion+topk_v2 pass gsm8k,
then run the c128 agentx arm. If the cause turns out to be the FP8->FP4
shared-expert down-quantisation itself, "making it pass" may be impossible
without a checkpoint that stores the shared expert in FP4 -- say so rather than
fudge the criterion.

**Arm matrix** (harness: `gsm8k_arm.sh <ref_dir> <label>`, results appended to
`/workspace/results/accuracy/STATUS.txt`, full logs `/tmp/gsm8k_<label>.log`;
driver `/tmp/acc_driver.sh` runs them back-to-back and restores the tree):

| # | arm | ref dir | answers |
|---|---|---|---|
| 1 | dp+tbo+fusion+topkv2 | `/tmp/allopt_c128_ref` | **0.737 (DONE)** |
| 2 | dp+tbo+NOfusion+topkv2 | `c128-chunk16384-topkv2` | is fusion implicated? |
| 3 | tp-only+fusion+topkv2 | `/tmp/tp_fusion_ref` | is it DP/TBO or the quantisation? |
| 4 | dp+tbo+fusion+NOtopkv2 | `/tmp/dptbo_fusion_notopk` | is topk_v2 implicated? |

Arm 3 is the sharp one (user's idea): if TP-only+fusion is clean, the FP8->FP4
down-quantisation is NOT the cause and the DP/TBO path mishandles the fused
shared expert. Arm 3 collapses DP and TBO together -- if it passes, a fifth arm
(DP on, TBO off) is needed to say which.

**Live hypothesis to check first:** `topk_sigmoid` (`topk.py:962`) is the kernel
that injects the fused shared expert into `topk_weights/topk_ids`, and topk_v2
(#36684) rewrites exactly those topk kernels. The patch text contains no
`shared`-expert logic (only a shared-memory comment), which is weak evidence for
innocence, not proof. Arm 4 settles it.

### Next, in order

1. **`fused-shared-expert`** (user-requested, not started). We currently run
   `--disable-shared-experts-fusion`. **Decide the instrument first** -- per the
   paragraph above, a single 3600 s arm will return "not resolvable" unless fusion
   moves >6 %.
2. **Validate ATOM's recipe as written** (`/workspace/ATOM`, cloned;
   `recipes/DeepSeek-V4-Agentic-InferenceX.md`). Confirm 44,722 tok/s/chip and
   13.4 s TTFT at c256 reproduce on this node before bisecting anything. If they
   do not, the head-to-head is void and that is the first thing to know. Needs
   ATOM's `atom.entrypoints.openai_server` and image; aiperf pin already matches
   (`754356e9`, both sides).
3. **Fix our client env** -- five AIPerf vars ATOM sets and we do not. Only
   `AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300` actually pays: the 10 s default is
   what invalidated `c128-chunk8192`. **Doing this changes the baseline**, so
   re-baseline afterwards; it was deliberately NOT applied before the topk_v2 arm.
4. **Report to ATOM** that their `--warmup-grace-period` observation is right:
   confirmed in aiperf source, the agentic barrier waits **indefinitely** without
   `--agentic-warmup-grace-period`. They asked to be told.

### What is NOT answered, and why — read before planning the next run

- **The delayer mixed-slot guard is still unmeasured, and now we know why.**
  `#running-req` at c256/512 slots: median 2, p90 24, p99 36, max 45 against a
  **64/rank cap** -- slots never get tight, so the guard's condition never arises.
  This is the same reason c64 measured +0.13 %. The guard-on arm was aborted
  (`rc=137`, deliberate) rather than spend 2.5 h on a predicted null.
  **The bind:** tight slots are what the guard needs, and 1x headroom is exactly
  what deadlocks the server. The prerequisite work item is finding the
  tight-but-alive regime -- bracket it with `MAX_RUNNING_CAP` 384 (48/rank) then
  320 (40/rank); 512 works, 256 hangs.
- **Everything is n=1 per cell.** The c256 chunk pair is now load-bearing and has
  never been replicated. Replicate it before anyone lands a default change.
- `c128-chunk8192` is **INVALID** (aiperf coverage 94.1 % < 95 %). Kept as evidence
  that 8192 cannot certify at c128; never quote its numbers.

### Three exit codes, three different meanings — do not conflate them in STATUS.txt

| rc | meaning |
|---|---|
| **124** | killed by the driver's `timeout` -- a hang, **or a healthy arm killed by too tight a cap** (the old 150 min was only ~5 min above what a healthy c128 arm needs; now 200 min) |
| **137** | 128+9 = SIGKILL, a deliberate abort by an operator |
| **1** | aiperf refused to certify its own run (coverage < 95 %) -- the server was fine |

**Tree state re-verified 15:15, all four rows of the table below PASS:** HEAD
`a1f9508dd4`; `git status --porcelain` on the three topk_v2 paths prints nothing;
`_mixed_slot_guard` -> 3 hits; launcher honours `CHUNK_PER_RANK` at lines 147/182
with the backup still at `/tmp/b200align_launcher.pre_chunkenv.bak`. One
correction to the JIT row: the active cache is `~/.cache/sglang/jit` and the only
sibling left is `jit.topkv2` -- there is **no `jit.pre36684` directory any more**,
it was moved into place rather than copied. Nothing to fix (the topk_v2 *source*
is reverted, so a stale entry would rebuild), but do not go looking for a backup
that no longer exists.

Confirmed at line 233 of the launcher, since the c256 note below depends on it:
`CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS` is then clamped `>128 -> 128`.

**Two analysis tools now exist -- use them instead of hand-reading JSON:**

```bash
python3 /workspace/claude-skills/agentx/arm_report.py <arm> [<baseline-arm>]
bash    /workspace/claude-skills/agentx/warmup_check.sh <arm>   # health of a RUNNING arm
```

`arm_report.py` prints the sanity gates (`errors`, `records_error_dropped`,
`duration_seconds`) **before** the headline, refuses `ls *.json` in favour of the
exact `dsv4_..._c<N>.json` name, flags an ISL move >3 %, refuses cross-conc
comparison, and grades any delta against the 5.67 % replicate spread and the
10 % bar. Both were validated against arms already on file: `arm_report.py`
reproduces 18,518.0 / ITL p90 50.03 / intvty p90 19.99 for
`b200align-tp8-c64-3600s` and +6.16 % / -32.29 % against `tbo-tp8-c64`;
`warmup_check.sh` reproduces the reference series 52/94/262/701.

### FIRST — check before you touch anything

```bash
ps -eo pid,etime,args | grep -E "[o]vernight_matrix.sh" | cut -c1-95
tail -20 /workspace/results/overnight/STATUS.txt
```

- **process alive** -> do nothing, go to monitoring. Do **not** start a second one.
- **no process, last STATUS line is not `finished`** -> it died. Clear leftover
  servers first (trap 3), then relaunch — completed arms are skipped
  automatically:

```bash
mkdir -p /workspace/results/overnight
setsid nohup bash /workspace/claude-skills/agentx/overnight_matrix.sh \
  < /dev/null > /workspace/results/overnight/driver.log 2>&1 &
```

**Monitor, don't poll** — one watch on the status file is enough
(`tail -n 0 -F /workspace/results/overnight/STATUS.txt`). Each arm takes ~110 min
and logs START / DONE-with-tok-s / FAILED. To judge a running arm's health, read
its `benchmark.log` `Phase warmup progress` series against reference arm
`tbo-tp8-c64/benchmark.log` (52 / 94 / 262 / 701 at 300/600/900/1200 s).

**Analyse each arm as it lands, not at the end.** Pairings:

| arm | compare against |
|---|---|
| `c64-chunk16384` | existing `b200align-tp8-c64-3600s` (**18,518.0**) |
| `c256-chunk8192-gd` | `c256-chunk8192` (differs only by the delayer guard) |
| `c256-chunk16384` | `c256-chunk8192` |
| `c128-chunk16384` | `c128-chunk8192` |

**Six arms, 3600 s each, ~110 min per cycle => ~11 h.** It will not finish in 8 h
and that is by design — the order is chosen so **every completed arm answers a
question by itself**, and the two that spill over (the c128 pair) are the least
important and form a clean unit to finish later:

| # | arm | conc | chunk/rank | guard | answers | cum. ETA |
|---|---|---|---|---|---|---|
| 1 | `c64-chunk16384` | 64 | 16384 | off | chunk effect @c64 | ~1 h50 |
| 2 | `c256-chunk8192` | 256 | 8192 | off | c256 anchor | ~3 h40 |
| 3 | `c256-chunk8192-gd` | 256 | 8192 | **on** | **the delayer question** | ~5 h30 |
| 4 | `c256-chunk16384` | 256 | 16384 | off | chunk effect @c256 | ~7 h20 |
| 5 | `c128-chunk8192` | 128 | 8192 | off | c128 anchor | ~9 h10 |
| 6 | `c128-chunk16384` | 128 | 16384 | off | chunk effect @c128 | ~11 h |

Arm 1 needs no partner — `b200align-tp8-c64-3600s` (**18,518.0** tok/s/GPU,
3630 s, succ 4561) is already a valid chunk-8192 anchor: same four B200 flags,
delayer+TBO, `--chunked-prefill-size 65536` (= 8192/rank), and it predates both
of today's patches, so guard-off / topk-off. Arm 2 is the anchor for **both**
arm 3 and arm 4, which is why it comes second.

**Why c256 is where the delayer can finally bite.** The launcher sets
`MAX_RUNNING_REQUESTS=$((2 * CONC))` **capped at 256**, so c64->128, c128->256,
c256->512 **clipped to 256**. Only at c256 does concurrency equal
max_running_requests — the 2x headroom is gone and slots actually get tight. At
c64 the guard measured +0.13 % because `#running-req` was median 2, p90 8 against
a per-rank cap of 16 (see "Closed 2026-08-28").

#### c256 runs at 1x headroom **on purpose** — and what to do if it looks bad

The launcher's own comment asks for **2x** headroom ("AgentX concurrency counts
live session trees, not individual requests. Subagent fan-out can push
instantaneous request concurrency above CONC, so leave 2x headroom rather than
clipping those bursts at the scheduler"). The `>256 -> 256` clamp gives c64 and
c128 their 2x but leaves **c256 at 1.0x**, so fan-out bursts *are* clipped there.

This was a deliberate decision (2026-08-28, agreed with the user), because
loosening it would very likely reproduce the c64 null: the guard can only matter
when slots are tight, and raising the cap to 512 restores the slack that made the
c64 measurement uninformative. Consequences to carry into any write-up:

- **Within-c256 pairs stay controlled** — chunk 8192 vs 16384 and guard on vs off
  both run at max_running=256, so it is not a confound for those comparisons.
- **c256 absolute numbers are a *saturation* point, not a clean c256 operating
  point.** Never present them as "DSV4 at concurrency 256".
- `CUDA_GRAPH_MAX_BS` is clamped to **128** while max_running is 256, so decode
  batches above 128 fall out of CUDA graphs and run eager. Controlled inside the
  c256 pairs; one more reason cross-conc headline comparison is invalid.
- Memory is not the constraint: pool is 56.1 M tokens, 256 running x ISL 117 k
  ~= 29.9 M = 53 %.
- **Risk, untested:** c256 has never run in this setup. At 1x headroom TTFT will
  stretch and queueing grows; if failures exceed
  `AIPERF_FAILED_REQUEST_THRESHOLD=0.10` aiperf aborts that arm. The driver logs
  `FAILED` and moves on, so it cannot take down the night.

**Escalation path — if the c256 numbers come back bad, switch the cap to 512:**

```bash
# 1. stop the driver FIRST (see the hazard note below)
pkill -f '[o]vernight_matrix.sh'; # then clear servers per trap 3
# 2. edit the launcher
#    benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_b200align_mtp.sh:231
#      [ "$MAX_RUNNING_REQUESTS" -gt 256 ] && MAX_RUNNING_REQUESTS=256
#    -> raise 256 to 512 (or make it ${MAX_RUNNING_CAP:-256})
# 3. move the finished c256 dirs aside so they re-run, then relaunch the driver
```

**HAZARD — never edit that launcher while an arm is running.** bash reads a
script lazily by byte offset, so editing it mid-execution can shift the offset
and make the running shell execute garbage. This is why the cap was **not**
pre-patched to an env var the way `CHUNK_PER_RANK` was: `CHUNK_PER_RANK` was
patched *before* the matrix started, the cap would have to be patched *during*
it. Stop the driver, edit, relaunch — completed arms are skipped automatically.

When the matrix ends (or you are woken), write the results into this section and
clear the GPUs (trap 3 + trap 4).

### Matrix results as they land

#### Arm 1/6 `c64-chunk16384` — DONE 16:46, ran 104 min. **NULL on every metric.**

Gates all pass: `errors=0`, `records_error_dropped=0`, `duration_seconds` 3628.5,
ISL within 1.23 % of the anchor. Quoted from the result JSON, not the monitor
notification (trap 14 — the notification said 18724.0 and the JSON agrees, but
check it every time).

| | chunk 16384 (arm 1) | chunk 8192 (`b200align-tp8-c64-3600s`) | delta | same-config replicate swing |
|---|---|---|---|---|
| tok/s/GPU | **18,724.0** | **18,518.0** | +1.11 % | 5.67 % |
| TTFT mean | 5.32 s | 7.21 s | -26.3 % | **26.3 %** |
| ITL p90 | 54.51 ms | 50.03 ms | +9.0 % | **7.0 %** |
| intvty p90 | 18.35 | 19.99 | -8.2 % | **6.6 %** |
| succ | 4668/5375 | 4561/5268 | +2.35 % | — |
| ISL | 115,480 | 116,923 | -1.23 % | — |

**Answer to "chunk effect @c64": there is none that this setup can measure.**
Doubling `CHUNK_PER_RANK` 8192 -> 16384 moves throughput +1.11 %, a fifth of the
replicate spread. Do not carry the +1.11 % anywhere as a win.

**The TTFT / ITL / intvty moves are also null, and the right-hand column is why.**
Resist the tempting mechanistic story (bigger chunks -> fewer prefill
interruptions -> faster first token, longer decode stalls -> worse inter-token).
It is plausible, the signs line up, and it is **not supported**: every one of the
three deltas is the same size as the swing the *same configuration* produced
between the 07:13 and 14:41 1200 s runs with nothing changed.

#### 2x-headroom rerun, arm 1/5 `c128-chunk8192` — result produced, but **it FAILS aiperf's own validity bar. Do not quote 24,539.6.**

Ran 23:18 -> 01:15 (117 min), `rc=1`. It did **not** hang — the server was healthy
the whole way, `errors=0`, `records_error_dropped=0`, 6852/8274 successful. A
**third** failure mode, distinct from both the c64 nulls and the c256 hangs:

```
ERROR Timeout waiting 10.0s for cancelled credits to return. Some credits may be stuck.
ERROR Profiling metric coverage below the required 95.0% for phase 'profiling':
      TTFT=94.1%, inter-token latency=94.6% over the configured 3600.0s duration.
ERROR Received fatal profile-results validation error   -> exit 1
```

**aiperf rejected its own run.** Coverage 94.1 % / 94.6 % against a 95 % bar, and
`duration_seconds` came out **3406.8 s** instead of ~3630 because the phase was
force-completed with credits still stuck.

**Why, and why it matters more than the exit code:** TTFT at c128 is **27.35 s**
against 5.32-7.21 s at c64. Requests were still in flight when the window closed,
so their TTFT/ITL were never recorded. The missing 5.9 % are therefore **the
slowest requests**, and every latency percentile in this arm is computed over the
subset that finished — **biased optimistic by construction**. `TTFT 27.35 s`,
`ITL p90 66.39`, `intvty p90 15.06` are all better than the truth by an unknown
margin.

**The headline is not rescued by the shorter window.** tok/s/GPU divides by the
measured duration, so it self-normalises — but it is still computed over a run
aiperf declared invalid, and the tail of the window is exactly where the
slow-request backlog sat. Treat 24,539.6 as **not quotable**.

And it must **never** be set against c64's 18,518 / 18,724: cross-concurrency
headline comparison is invalid (§19.6), and this arm demonstrates why in one
number -- **ISL is 96,760 here vs ~116,000 at c64**. It is a different workload,
not a faster one.

**Expect this again, and read the pair carefully:**

- Arm 2 `c128-chunk16384` is the same concurrency and will very likely trip the
  same gate. If both arms fail coverage, the pair is **not** automatically still
  comparable: the truncation bias scales with each arm's own TTFT, so the arm
  with worse TTFT loses more of its slow tail. A chunk-size delta read across two
  differently-biased subsets is confounded. Check both coverage percentages
  before comparing, and if they differ materially, say so instead of quoting the
  delta.
- The three c256 arms will have **worse** TTFT still, so coverage there is likely
  to be worse, not better. A c256 arm that survives the hang may still be
  unquotable for this reason. Surviving the hang and producing a valid result are
  two separate bars.

**The real fix is a longer drain at end of phase, not a longer benchmark.** The
requests are not failing, they are simply not finishing inside the window. Nothing
was changed here unattended -- flagged for a decision.

`arm_report.py` now gates on this automatically (`aiperf_coverage=FAILED 94.1/94.6`).
Parsing note: the message appears **3x** in `benchmark.log` and the last instance
is inside a word-wrapped summary box with the percentages on a different line, so
a parser that keeps the last match reads `?/?`. Keep the first match that carries
numbers.

#### 2x rerun arm 3/5 `c256-chunk8192-2x` — **THE HANG IS FIXED. Clean single-variable attribution.**

Ran 03:12 -> 05:45 (153 min), `rc=0`, **all gates pass**: `errors=0`,
`records_error_dropped=0`, `duration_seconds` 3629.2, aiperf coverage ok.
Headline 27,171.7 tok/s/GPU, succ 7234/10079 (the 2,845 shortfall is exactly the
c256 warmup set).

**This is a clean attribution, not a correlation.** Against the three hung arms
the *only* thing that changed is `MAX_RUNNING_REQUESTS` 256 -> 512. Same
concurrency 256, same `CHUNK_PER_RANK` 8192, same guard=0, same tree, same
launcher. The old arms hung 3/3; this one ran a full window.

> **The 256 clamp caused the hang. c256 at 1x headroom deadlocks the server; at
> 2x it is healthy.** The launcher's own comment was right -- AgentX concurrency
> counts session trees, and clipping subagent fan-out at the scheduler is not a
> safe economy.

#### Two of my own predictions were wrong here — record both

1. **"c256 will fail aiperf coverage because its TTFT is higher."** It did not.
   c256 has **TTFT 94.13 s** -- 3.4x c128's 27.35 s -- and still certified,
   while c128 at chunk 8192 **failed** coverage. So coverage is **not** a simple
   function of TTFT. The c128 failure came with a force-terminated window
   (3406.8 s) and stuck credits; this one ran its full 3629.2 s. Whatever drives
   the credit-return stall, it is not "high TTFT" on its own. **Do not reuse that
   reasoning** -- the mechanism is unestablished.
2. The delayer pair was moved to chunk 16384 on the strength of prediction (1).
   With the premise dead, the move was reverted before it ran: the 8192 pair is
   cheaper (its anchor already exists) and equally valid.
   `matrix_2x_b.sh` is **superseded and was never launched** -- keep it only as a
   template.

#### The delayer question is still blocked, and now it is QUANTIFIED

`#running-req` across the whole c256 run, against a per-rank cap of 64
(512 / 8 ranks), n=20,967 samples:

| median | p90 | p99 | max | cap |
|---|---|---|---|---|
| 2 | 24 | 36 | 45 | **64** |

**Slots never got tight.** This is the same picture that made c64 a null
(median 2, p90 8, cap 16) and is relatively *slacker*: p90/cap is 37.5 % here vs
50 % at c64. So `c256-chunk8192-gd-2x` is expected to be **another null, for the
c64 reason** -- not because the guard does nothing, but because the condition it
guards against does not arise. Chunk size is irrelevant to this; 16384 would not
have helped.

**The tight-but-alive regime, if it exists, is bracketed:**

| MAX_RUNNING_CAP | per-rank | outcome |
|---|---|---|
| 512 | 64 | works; p90 24/64, slack |
| 384 | 48 | untested; p99 36 would start to bind |
| 320 | 40 | untested |
| 256 | 32 | **hung 3/3**; p90 24 and p99 36 would bind often |

Finding that boundary is the prerequisite for ever measuring the guard. It is
also, on its own, the answer to "how much headroom does AgentX actually need",
which is worth more than the guard question.

#### 2x rerun arm 5/5 `c256-chunk16384-2x` — **+13.15 %. The first result in this file to clear the >=10 % bar.**

Both arms valid, both gates clean, **only `CHUNK_PER_RANK` differs** (conc 256,
max-running 512, guard 0, same tree). ISL matches to 0.95 %, so it is the same
workload.

| | chunk 16384 | chunk 8192 | delta | noise floor |
|---|---|---|---|---|
| tok/s/GPU | **30,745.5** | 27,171.7 | **+13.15 %** | 5.67 % |
| TTFT mean | 38.15 s | 94.13 s | **-59.5 %** | 26.3 % |
| TTFT p50 | **21.50 s** | 77.58 s | **-72.3 %** | — |
| ITL p90 | 145.08 ms | 109.42 ms | +32.6 % | 7.0 % |
| intvty p90 | 6.89 | 9.14 | -24.6 % | 6.6 % |
| succ | 8266/11111 | 7234/10079 | +14.3 % | — |
| cache hit | 0.663 | 0.661 | +0.3 % | — |
| ISL | 107,225 | 108,252 | -0.95 % | — |

**This is the cleanest positive result on file.** Same concurrency (so §19.6 does
not apply), both runs certified by aiperf, one variable, ISL controlled, and the
delta is 2.3x the replicate spread and above §25's >=10 % bar. Contrast the
+6.2 % b200align claim, which has no error bar and sits below that bar.

**The chunk effect is strongly load-dependent — that is the real finding:**

| conc | chunk 8192 -> 16384 | verdict |
|---|---|---|
| 64 | +1.11 % | null (1/5 of noise) |
| 128 | 8192 could not certify at all | categorical |
| 256 | **+13.15 %** | **real, above the bar** |

At c64 the doc was right to call it a null; it was not a missed effect, the effect
genuinely is not there at low load. It grows monotonically with prefill pressure,
which is what a prefill-scheduling-granularity knob should do.

**The trade is real and it is not free.** ITL p90 +32.6 % and intvty p90 -24.6 %,
both several times their noise floors. Bigger chunks buy first-token latency and
throughput by making decode wait longer behind each prefill chunk. At c256 that
trade is clearly worth taking; at c64 it buys nothing and still costs.

**Control that fell out for free: cache hit is 0.661 vs 0.663 across the pair.**
The collapse from ~95 % to ~66 % is therefore a function of **concurrency, not
chunk size**, which supports the "256 session trees exceed the prefix cache"
reading and rules out chunking as its cause.

**c256 is still not a serving point.** TTFT p50 improves 77.58 -> 21.50 s, a 3.6x
gain, but 21.5 s to first token is not a product. The honest summary is that
16384 makes c256 *measurable and much less bad*, not usable. The usable operating
point remains c128 or below.

#### Memory knobs are CONSTANT across every arm — and HiCache has ~3x unused DRAM

Checked in `sglang_command.txt` for all six arms, identical in every one, with no
`CONC` branching in the launcher:

| flag | value | where |
|---|---|---|
| `--mem-fraction-static` | **0.90** | line 154, `${MEM_FRACTION_STATIC:-0.90}` |
| `--swa-full-tokens-ratio` | **0.15** | line 297, `${SWA_FULL_TOKENS_RATIO:-0.15}` |
| `--hicache-ratio` | 1.5 | line 114 |
| write policy / io backend / layout | write_through / direct / page_first_direct | 113-124 |

So neither is a confound in any comparison in this file. Both are env-overridable
and **neither has ever been swept** -- `mem-fraction-static 0.90` in particular is
inherited, not tuned.

**TRAP — the JSON reports the two KV tiers in DIFFERENT UNITS.**
`gpu_total_tokens` is the **aggregate** over 8 ranks; `cpu_total_tokens` is
**per-rank**. Computing `cpu/gpu` straight from the result JSON gives **0.187**
and the false conclusion that the DRAM tier is 5x undersized. The true figures:

| | per rank | aggregate |
|---|---|---|
| GPU KV pool | 6.92 M tokens (`max_total_num_tokens`) | 55.4 M |
| HiCache DRAM | 10.38 M tokens | 83.0 M |
| ratio | **1.50** = exactly `--hicache-ratio` | |

HiCache is sized correctly. Confirm against the server log, which is unambiguous:
`Allocating 79.70 GB host memory for V4 paged pool 'deepseek_v4_c4'` plus
10.27 GB for the indexer, **per rank** (pages=40540, 256 tokens/page).

**The lever: 720 GB of 2,399 GB host DRAM is in use -- 30 %.**

| HICACHE_RATIO | host DRAM | % of 2,399 GB |
|---|---|---|
| **1.5 (current)** | **720 GB** | **30 %** |
| 2.0 | 960 GB | 40 % |
| 3.0 | 1,440 GB | 60 % |
| 4.0 | 1,919 GB | 80 % |

**This matters most at c256, where BOTH tiers run full:** `gpu_usage_pct` 1.00 and
`cpu_usage_pct` 1.00, i.e. the DRAM tier is evicting. Overall hit rate is 94.2 %,
so ~5.8 % of prompt tokens are true misses that must be prefilled from scratch at
ISL ~107 k. Since **99.1 % of headline throughput is prefill**, hit-rate gains
convert into throughput almost one-for-one -- a percentage point of hit rate is
worth roughly a percentage point of tok/s.

Untested, and the launcher's own comment warns that ratio 4 oversubscribed DRAM in
a *different* recipe (spec-none, smaller device pool, mem-fraction 0.85), so start
at **2.0 or 3.0**, not 4.0, and watch `cpu_usage_pct` and host RSS. This is the
cheapest remaining lever on the table and it has never been tried here.

### topk_v2 (#36684) at c128, 3600 s — NOT RESOLVABLE. And the noise floor is now known per metric.

Single-variable arm against the valid `c128-chunk16384` baseline: same conc, chunk
16384, guard 0, max-running 256, **client env deliberately left unfixed** so the
drain-timeout change would not become a second variable. Both arms certified
(`errors=0`, coverage ok, duration 3629.9 / 3628.7).

**topk_v2 was verified ACTIVE at runtime** -- 8/8 schedulers had
`sgl_kernel_jit_dpsk_v4_topk_v2.so` mapped (see trap 18; the log check is useless).

| metric | topk_v2 | baseline | delta | **replicate spread** | resolvable? |
|---|---|---|---|---|---|
| tok/s/GPU | 26,943.3 | 25,968.4 | +3.75 % | 5.67 % | **no** |
| ITL mean | 62.350 ms | 64.350 ms | -3.11 % | **5.51 %** | **no** |
| ITL p50 | 61.110 ms | 63.920 ms | -4.40 % | **4.54 %** | **no** |
| ITL p90 | 83.520 ms | 85.220 ms | -1.99 % | 7.01 % | **no** |
| computed tok/s/chip | 1,444 | 1,383 | +4.41 % | — | — |
| succ | 7,814 | 7,570 | +3.22 % | — | — |
| TTFT mean | 8.98 s | 10.63 s | -15.47 % | 26.3 % | no |

**Every delta is smaller than its own metric's noise floor.**

#### New: ITL noise floors, derived from the same replicate pair

Computed from `b200align-tp8-c64-1200s` vs
`rebaseline-notopkv2-noguard-c64-1200s` (identical config, nothing changed):

| metric | spread |
|---|---|
| headline | -5.67 % |
| **ITL mean** | **+5.51 %** |
| **ITL p50** | **+4.54 %** |
| ITL p90 | +7.01 % |

**This kills the idea that ITL is the sensitive instrument.** The reasoning was
sound -- decode has no cache-hit dilution, so a MoE-routing kernel should show up
undiluted in ITL -- but ITL's own run-to-run spread (5.5 %) is as wide as the
headline's (5.67 %), so it buys no resolving power. Worth knowing before designing
the next kernel A/B.

#### What can and cannot be said

- **The MTP confounder is clean, and it argues slightly FOR the result.** accept
  len 2.4855 vs 2.4969 (-0.46 %) -- lower acceptance means more forward passes per
  token, which should make ITL *worse*, yet ITL improved. The gain is not an MTP
  artefact.
- **All seven metrics moved favourably.** That is suggestive, but they are not
  seven independent observations: headline / succ / computed measure the same
  "work completed", and the ITL family is internally correlated. The replicate
  pair shows noise here is **correlated too** -- when nothing changed, headline
  fell 5.67 % *and* ITL rose 5.51 % together. One correlated draw looks exactly
  like this.
- **Verdict: consistent with a real ~3-4 % gain, indistinguishable from one noise
  draw.** Same standing as the earlier c64 test (-2.5 % / +3.3 %), just with a
  better-controlled baseline.

#### The instrument is wrong for this class of change — read before the next kernel A/B

To resolve a 4 % effect against a ~5.5 % pairwise spread (single-run sd ~3.9 %)
needs roughly **8 runs per arm** for a 2 % standard error -- ~30 GPU-hours for one
kernel flag. That is not a sensible way to evaluate kernel optimisations.

**Recommendation: measure kernel changes at the kernel, not end-to-end.** Profile
the top-k kernel directly (rocprof / a microbenchmark) where a 3 % change is
trivially resolvable, and use one end-to-end arm only to confirm no regression and
no accuracy change. Reserve the 2-hour end-to-end arm for changes expected to
exceed ~10 % (like `CHUNK_PER_RANK`, which delivered +13.15 %).

**This applies directly to the queued `fused-shared-expert` test.** Current runs
carry `--disable-shared-experts-fusion`. Unless fusion is expected to move more
than ~6 %, a single 3600 s arm will return exactly this verdict again.

**Tree restored after the run:** patch reverted (`git status` clean on all three
paths), JIT caches swapped back (`jit` = pre-36684 active, `jit.topkv2` set aside).

### ATOM head-to-head — we are 69 % behind on REAL compute, and we have a confirmed client bug

`/workspace/ATOM` cloned; the recipe is `recipes/DeepSeek-V4-Agentic-InferenceX.md`
(on main, not just the PR). It reports the same scenario, dataset, hardware and
duration we run, so the rows are directly comparable.

| conc | ATOM tok/s/chip | ours | ATOM TTFT | ours TTFT |
|---|---|---|---|---|
| 64 | 21,888 | 18,724 (-14 %) | 10.5 s | **5.32 s (we win)** |
| 128 | 30,709 | 25,968 (-15 %) | 10.9 s | 10.63 s (tie) |
| 256 | **44,722** | 30,745 (**-31 %**) | **13.4 s** | **38.15 s (2.8x worse)** |

**ATOM's TTFT is nearly flat across the DP band (10.5 -> 10.9 -> 13.4 s); ours
explodes (5.32 -> 10.63 -> 38.15 s).** We are competitive at c64/c128 and fall
apart only at c256. This is a scaling failure, not a tuning gap.

**The real-compute gap is WORSE than the headline gap.** ATOM's own recipe states
that at c256 "roughly **2,900** of the 44,722 tok/s/chip actually went through
prefill" -- independently confirming the cache-hit accounting analysis above.
Ours is **1,714** tok/s/chip.

> **ATOM does 69 % more real prefill work per chip at c256, while we lag 31 % on
> the headline. And their DP rows run WITHOUT `--enable-tbo`** (the recipe says
> the measured table predates it) -- we run *with* TBO and are still behind.

#### CONFIRMED BUG (ours): our agentic warmup barrier waits forever

ATOM's recipe flags that `--warmup-grace-period` is inert for agentic runs and
invites us to check. **They are right.** From aiperf's own source,
`utils/aiperf/src/aiperf/config/flags/cli_config.py:2305`:

> `agentic_warmup_grace_period` ... "The agentic warmup is synthesized from the
> profiling phase rather than a user-declared warmup phase, so it does **NOT**
> honor `--warmup-grace-period` (which requires `--warmup-duration`). **If not
> set, the warmup barrier waits indefinitely until every primed trajectory
> returns.**"  — default `None`.

Our launchers `export AGENTIC_WARMUP_GRACE_PERIOD=3600`, which `build_replay_cmd`
passes as **`--warmup-grace-period 3600`**, and we never set
`--warmup-duration`. So the value is discarded and **we run with no grace period
at all.** ATOM passes `--agentic-warmup-grace-period 1800`.

**This is the amplifier behind the c256 losses.** The server stall was real (its
scheduler was silent for 1 h 53 m), but with no barrier timeout the client could
never give up, so each arm burned its full 150-minute timeout and produced no
JSON. A 1800 s grace would have forced the phase and yielded *something*.
Fix: pass `--agentic-warmup-grace-period` in `build_replay_cmd`. **Tell ATOM --
they asked to be told if the plain flag turned out to be right.**

#### The client-env gap, ranked by what it actually buys

We set **none** of ATOM's five AIPerf env vars. Same aiperf commit on both sides
(`754356e9`, verified via `git submodule status utils/aiperf`), so this is pure
configuration, not version skew.

| var | ATOM | ours | what it buys |
|---|---|---|---|
| `AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT` | **300** | default **10 s** | **the one that pays -- see below** |
| `AIPERF_HTTP_TCP_USER_TIMEOUT` | 900000 | unset | robustness on long stalls |
| `AIPERF_DATASET_CONFIGURATION_TIMEOUT` | 1800 | unset | startup only |
| `AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT` | 1800 | unset | startup only |
| `AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES` | 0 | unset | **no-op on this pin** -- the field does not exist in `754356e9`. Not a workload difference |

**`AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300` is the fix for the c128 invalidation.**
The arm died on exactly this:

```
Timeout waiting 10.0s for cancelled credits to return. Some credits may be stuck.
  -> coverage TTFT=94.1% < 95%  -> fatal validation error  -> exit 1
```

**That 10.0 s is the default.** At the end of the profiling phase aiperf cancels
outstanding requests and waits for credits; with ~100 k-token prefills in flight,
10 s cannot drain them, their TTFT/ITL are never recorded, coverage falls under
the bar and the run is thrown away. ATOM allows 300 s. This also finally supplies
the mechanism that was left "unestablished" above -- coverage failure is about
**the drain window at phase end**, not about TTFT directly, which is why c256 at
TTFT 94 s certified while c128 at TTFT 27 s did not.

**It does not make runs faster -- it makes them up to ~5 min slower and valid
instead of invalid.** 1 of our 6 arms was destroyed by this; that is the payoff.

**By contrast `--agentic-warmup-grace-period` buys little now.** Healthy warmups
end on their own (c64 advanced at 1200 s), and the stall it would have bounded was
caused by the `MAX_RUNNING_REQUESTS` clamp, already fixed. Keep it as insurance and
for recipe parity, not as a performance or runtime item.

#### `ATOM_DP_SESSION_AFFINITY` is mandatory in their recipe

> "Without it a conversation's turns land on different DP ranks, so the prefix KV
> written by one turn sits on a rank the next turn never reaches — and an agentic
> trace is nothing but multi-turn sessions, so **the whole workload degrades to
> cold prefill**."

We run `USE_SGLANG_ROUTER=false`, i.e. exactly that degraded state. **Our HiCache
DRAM tier masks it** -- overall hit stays 94 % because misses fall to DRAM instead
of recomputing -- which is why our on-chip is 66 % and theirs is 93 %. Their
router also reads `x-dynamo-parent-session-id`, which carries **the lineage of a
forked agent tree**, so a whole subagent fan-out lands on one rank; plain
per-session hashing is weaker than that. Client side they set
`AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true`; we set only
`AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true`.

#### RETRACTIONS — three of my own conclusions above were wrong

1. **"SWA ratio is the biggest lever."** No. It frees 82 GB but a prefix cache
   fills regardless; the payoff is bounded and indirect.
2. **"Session affinity is the largest lever (multiples, not percentages)."**
   Overstated *as a bandwidth argument*. The DRAM spill moves 2.11 TB over the
   run = **80 MB/s per rank = 0.32 % of one PCIe link**. Tiering is nearly free
   in bandwidth terms, so "get the hits back on chip" is not by itself a
   performance case. Affinity may still matter via scheduling locality -- but the
   bandwidth argument I made for it does not hold.
3. **"cuda-graph-max-bs 128 is over-provisioned."** SGLang already prunes the
   ladder to bs=64 (`max_num_reqs`); the flag never took effect at 128.

**What survives:** the c256 gap is in *real prefill throughput* and *TTFT under
load*, not in cache tiering, not in SWA reserve, not in graph memory. Anything
that does not move `computed tokens / s / chip` is not addressing it.

### NEXT: validate ATOM's recipe directly — our knowledge here is insufficient

Agreed with the user, 2026-08-29. The levers reachable from our current frame are
exhausted and measured near-zero. The gap is structural and needs ATOM's design
understood, then their recipe reproduced on this node as a reference point.
Order:

1. **Fix `--agentic-warmup-grace-period`** in `build_replay_cmd` (cheap, confirmed
   bug, independent of everything else).
2. **Run ATOM's recipe as written** at c256 to confirm 44,722 / 13.4 s reproduces
   on this hardware. Needs ATOM's own `atom.entrypoints.openai_server`, image and
   aiperf pin `754356e9`. If it does not reproduce, the comparison is void and
   that is the first thing to know.
3. Only then bisect: their `ATOM_DP_SESSION_AFFINITY` + `ATOM_DP_LB_REQ_EQUIV`
   against our router-off path, and `--index_cache_dtype fp4` against our
   `deepseek_v4_fp4_indexer=False`.

### ATOM recipe comparison (ROCm/ATOM#2068) — where our on-chip cache capacity goes

ATOM reaches **93 % on-chip** hit at these concurrencies; we reach 66 % on-chip at
c256 (94 % overall, the rest served from the DRAM tier). Cloned `ROCm/ATOM` to
`/workspace/ATOM` and compared. `gh` cannot read the ROCm org with a classic PAT
(403); the PR diff came from the web UI, the code from the clone.

| | ATOM #2068 | ours | note |
|---|---|---|---|
| session affinity | **`ATOM_DP_SESSION_AFFINITY=1`** | **router disabled** (`USE_SGLANG_ROUTER=false`, launcher:140) | biggest suspect |
| DP load balance | `ATOM_DP_LB_REQ_EQUIV=512` | none | see §23 |
| mem fraction | `--gpu-memory-utilization 0.9` | `--mem-fraction-static 0.90` | same |
| max seqs | `--max-num-seqs $((CONC*2))`, **no clamp** | now 2*CONC (clamp removed tonight) | **independent corroboration** |
| prefill chunk | `--attn-prefill-chunk-size 16384` | 16384 measured best tonight | **independent corroboration** |
| cuda graph | `--cudagraph-mode FULL`, no max-bs | `--cuda-graph-max-bs 128` | **not the problem -- see below** |
| index cache | **`--index_cache_dtype fp4`** | `deepseek_v4_fp4_indexer=False` | real gap |
| SWA ratio | not set | `--swa-full-tokens-ratio 0.15` | **the big one** |

#### 1. CUDA graph size is NOT over-provisioned — SGLang already prunes it

The hypothesis was that `--cuda-graph-max-bs 128` wastes memory because a DP rank
can never see that batch size. **It does not, because the flag never took
effect at 128.** The actual captured ladder in the c256 run is

```
bs=[2,4,6,...,30,32,40,44,48,52,56,60,64]
```

topping out at **64 = `max_num_reqs`** (max-running 512 / 8 ranks). SGLang applies
the same bound ATOM does in `model_runner.py:4031` --
`min(max_num_seqs, max_num_batched_tokens // (mtp_k+1))` -- and drops the rest.

Cost of what *is* captured: 3.25 + 1.09 + 1.78 = **6.12 GB/rank** (target verify /
draft decode / draft extend), 49 GB aggregate. Observed `#running-req` peaked at
45, so only the bs 48-64 rungs are dead weight -- 1-2 GB/rank at most. **Real, but
the smallest lever on this page.**

#### 2. `--swa-full-tokens-ratio 0.15` is ~3x over-provisioned — 116 GB idle

Measured over the whole c256 run, n=12,305 scheduler samples:

| pool | median | p90 | p99 | **max** |
|---|---|---|---|---|
| **SWA** | 0.010 | 0.030 | 0.030 | **0.050** |
| full | 0.380 | 0.880 | 0.990 | **1.000** |

**The SWA pool never exceeded 5 % utilisation while the full pool saturated.**
Per rank the KV pool is 102 GB / 6.92 M tokens, of which `swa_size=1,037,824`
tokens = **15.0 % = 15.3 GB is reserved for SWA and at most 0.8 GB is ever used**.

> **~14.5 GB/rank = 116 GB aggregate sits idle — 2.4x the memory of every CUDA
> graph combined — while the full-attention pool is pinned at 100 % and spilling
> 28 % of its cache hits to host DRAM.**

Setting `SWA_FULL_TOKENS_RATIO=0.05` frees **10.2 GB/rank = 82 GB aggregate** into
the full/prefix pool and still leaves **6.7x headroom** over the observed peak.
Untested; the launcher (lines 93-96) notes SWA slots are proactively released
during chunked prefill, so behaviour at a smaller pool needs checking, not
assuming. **This is the cheapest, largest single lever identified so far.**

#### 3. fp4 index cache is off for us and on for ATOM

`deepseek_v4_fp4_indexer=False` in every arm; ATOM ships `--index_cache_dtype fp4`.
The host tier gives the rough scale: the indexer pool is 10.27 GB against 79.70 GB
for the main pool, **~11 % of the KV footprint**, so fp4 plausibly returns ~5 % of
the pool. This file already parks `--enable-deepseek-v4-fp4-indexer` "behind an
accuracy gate" -- that gate is now worth paying for, but it **is** an accuracy
question, not a free win.

#### 4. Session affinity — the largest suspect, and the one we already failed at

Without affinity a multi-turn session's turns land on arbitrary DP ranks, so its
prefix is rebuilt on up to **8** ranks; effective on-chip prefix capacity divides
accordingly. That fits the observed progression exactly (`gpu_usage_pct`
0.42 -> 0.70 -> 1.00). ATOM pins sessions with `ATOM_DP_SESSION_AFFINITY=1`.

**But §23 already tried routing here and it went badly** -- "routing fixes cache,
imbalance kills it", 2.20-3.12 req/s. The difference is that ATOM pairs affinity
with a load balancer (`ATOM_DP_LB_REQ_EQUIV=512`, plus `ATOM_NUMA_BIND=1`).
**Not verified in ATOM's source** -- the mechanism behind `LB_REQ_EQUIV` was not
read, and whether ATOM's `--max-num-seqs` is global or per-rank under DP was not
confirmed either. Treat this paragraph as a lead, not a finding.

**Testable prediction for any of these:** GPU-tier hit rate rises and the CPU-tier
share falls from 28 %. Watch `gpu_cache_hit_rate` **and** `overall_cache_hit_rate`
together -- overall barely moves when a tier shift reverses, so the device-tier
number is the one that shows the win here.

**Suggested order (cheapest and safest first):** SWA ratio 0.05 -> fp4 indexer
(accuracy-gated) -> trim cuda-graph-max-bs to 48 -> session affinity + LB.

### RECOMMENDATION: make `CHUNK_PER_RANK` 16384 the default above c64

8192 is the launcher default and it is the worse choice at every concurrency
tested above 64: it failed certification outright at c128 and costs 13 % of
throughput plus 3.6x the median TTFT at c256. At c64 the two are indistinguishable,
so 16384 is never worse in these six arms. Before landing it: this rests on n=1
per cell -- replicate the c256 pair, since it is now the load-bearing result.

#### Delayer arm ABORTED 05:5x by user decision — c256 is past the knee, cut losses

`c256-chunk8192-gd-2x` was killed ~10 min into its run (STATUS records
`FAILED rc=137` at 05:54:33 -- **137 = 128+9 = SIGKILL, i.e. this deliberate
abort, NOT a hang and NOT the 124 of a timeout**) and the driver advanced to
`c256-chunk16384-2x` (chunk 16384, guard 0, max-running 512 = 2x conc). Only the
arm's processes were killed, not the driver, so it logged `FAILED` and moved on
by itself -- no restart, no script edit, no lazy-read hazard.

**Why, in one line: the arm would have cost 2.5 h to produce an expected null at
an operating point nobody would ship.** Two independent reasons, both already
established above:

- `#running-req` p90 24 against a 64/rank cap -- the guard's condition never
  arises, so the measurement was predicted null before it started;
- c256's TTFT distribution makes the whole point untenable as an operating
  point (below).

#### The c256 TTFT distribution — this is why c256 is not a serving point

| | mean | p50 | p75 | p90 | p95 |
|---|---|---|---|---|---|
| **TTFT (s)** | 94.13 | **77.58** | 140.16 | **206.32** | 254.48 |
| e2el (s) | 159.48 | 128.94 | 207.85 | 310.15 | 392.12 |

**The p50 is 77.6 s.** This is not a tail problem that a mean is exaggerating --
*half* of all requests wait over a minute for their first token, p90 waits 3.4
minutes, p95 over 4. Against the other valid arms, TTFT mean is **8.9x** c128's
10.63 s and **13-18x** c64's 5.32-7.21 s.

`e2el` p50 128.94 s with TTFT p50 77.58 s means **60 % of a request's life is
spent waiting to start**, not generating.

**The throughput knee arrives long before the latency knee.** c128 -> c256 buys
+4.6 % headline (and that comparison is itself invalid across concurrency) for a
**8.9x TTFT penalty**. The usable operating point is at c128 or below. Any future
c256 arm is a saturation probe, never a serving recommendation.

#### CORRECTION — the c256 "cache collapse" is NOT a collapse. It is a tier shift, and it closes an old open question.

An earlier draft of this section called 0.661 a cache collapse and reached for
"256 session trees exceed the prefix cache". **That was wrong.** §15's warning
that `gpu_cache_hit_rate` is **device-tier only** is exactly right, and was walked
into anyway. Read `overall_cache_hit_rate`:

| arm | GPU tier | CPU tier | **overall** | `gpu_usage_pct` |
|---|---|---|---|---|
| c64 | 0.954 | — | **0.954** | 0.42 |
| c128 | 0.943 | 0.003 | **0.946** | 0.70 |
| c256 / 8192 | 0.661 | 0.281 | **0.942** | **1.00** |
| c256 / 16384 | 0.663 | 0.281 | **0.944** | **1.00** |

**Cache effectiveness is flat at ~94-95 % across every concurrency.** What moves
is *which tier serves the hit*. The GPU KV pool fills (0.42 -> 0.70 -> **1.00**);
once saturated, prefix blocks demote to the CPU/HiCache DRAM tier and 28 % of
hits are served from host memory. HiCache is doing its job.

**This closes "Next actions" item 4 — the TP8 0.689-vs-91.8 % contradiction.**
`armB-tp8-c64` reads GPU tier **0.689**, overall **0.950**, `gpu_usage_pct`
**1.00**. Identical mechanism: it was GPU-saturated, so its device-tier number
looked terrible while true cache effectiveness was 95 %. **The "opposite
ordering" was an artefact of comparing a device-tier number against an overall
number.** No experiment is needed; the question is answered. §22/§23's 91.8 % and
this file's 0.689 were never in conflict.

**Rules that follow, and they are not optional:**

- **Quote `overall_cache_hit_rate`. Never quote `gpu_cache_hit_rate` alone.**
  `summary_table.py` now reports overall, with the device tier and
  `gpu_usage_pct` as separate diagnostic columns so a tier shift is visible
  instead of disguised as a regression.
- `gpu_usage_pct` reaching **1.00** is the signal that a tier shift is under way.
  It is a legitimate saturation indicator; the device-tier hit rate is not.

**The real cost is a transfer cost, not a miss cost.** 220 M prompt tokens
(c256/8192) and 249 M (16384) were served from host DRAM rather than GPU. That is
plausibly a contributor to c256's TTFT, but it is **not** the explanation for the
chunk result: both c256 arms sit at 28.1 % CPU-tier share, so the +13.15 % is
cleanly controlled for cache tiering.

#### 2x rerun arm 2/5 `c128-chunk16384` — **VALID (rc=0)**. The pair is split, and that split IS the result.

| | chunk 16384 | chunk 8192 | |
|---|---|---|---|
| aiperf verdict | **PASS** | **FAIL** 94.1 / 94.6 % | |
| duration_s | 3628.7 | 3406.8 (force-completed) | |
| tok/s/GPU | 25,968.4 | 24,539.6 | +5.82 % |
| TTFT mean | **10.63 s** | **27.35 s** (optimistic) | -61 % |
| ITL p90 | 85.22 ms | 66.39 ms (optimistic) | +28.4 % |
| intvty p90 | 11.73 | 15.06 (optimistic) | -22.1 % |
| ISL | 98,730 | 96,760 | +2.04 % |

**Do not quote the +5.82 % as a chunk effect.** It is confounded, and the
mechanism is measurable: **99.14 % of the headline is INPUT tokens**, i.e. the
number is essentially prefill throughput counted only over requests that
*completed* inside the window. The 8192 arm was force-terminated 222 s early with
credits stuck, so the prefill work in flight at cutoff (ISL ~97 k per request) was
performed but never counted. An early-terminated arm **systematically undercounts
its own throughput**, and the size of that effect is the same order as the 5.82 %
being claimed.

**The defensible finding is categorical, not a percentage:**

> At c128, `CHUNK_PER_RANK` 16384 produces a **valid** measurement and 8192 does
> **not**. The mechanism is TTFT -- 10.63 s vs 27.35 s (and the 27.35 is itself
> optimistic, measured only over requests that finished). 8192 pushes first-token
> latency high enough that too few requests complete inside the window for aiperf
> to certify the run.

That is a 2.6x TTFT difference against a **26.3 % TTFT noise floor** -- one of the
few effects in this whole file that is unambiguously outside noise, and it has a
categorical consequence (run validity) rather than only a numeric one.

**It also rescues the c64 result from looking like a contradiction.** At c64,
chunk 16384 moved TTFT -26.3 % -- exactly at the noise floor, correctly called a
null. Same *direction*, magnitude growing sharply with load: null at c64, 2.6x at
c128. Load-dependent, not absent. Still n=1 per config; the c128 pair deserves a
replicate before this becomes doctrine.

**The ITL / intvty penalties are real in direction, unknown in size.** Both are
measured on the 8192 arm's fast subset, so its 66.39 ms is better than the truth
and the true penalty is **smaller** than the +28.4 % shown. Direction matches c64
(+9.0 %, at noise). The trade is genuine: bigger chunks buy first-token latency
and pay inter-token smoothness.

#### Forward hazard for the three c256 arms — the delayer arm uses chunk 8192

If 8192 cannot certify a run at c128, it is **less** likely to at c256, where TTFT
is higher still. Two of the three queued c256 arms use chunk 8192 --
`c256-chunk8192-2x` and, critically, **`c256-chunk8192-gd-2x`, the delayer arm.**
So the delayer question may come back invalid for a second reason, unrelated to
the hang.

The delayer pair is at least internally consistent (both 8192, guard the only
difference), but **the c128 pair just demonstrated that two arms sharing a config
can land on opposite sides of the validity bar**, and equal invalidity cannot be
assumed. Nothing was changed unattended -- flagged as a decision: consider running
the delayer pair at chunk 16384 instead, where a valid measurement is at least
achievable.

#### Arm 2/6 `c256-chunk8192` — **FAILED 19:19, rc=124. The server HUNG. c256 may be unusable.**

**rc=124 is the driver's own `timeout $PER_RUN_TIMEOUT` (150 min), not an aiperf
abort.** The predicted failure mode did not happen: `errors=0` throughout,
`AIPERF_FAILED_REQUEST_THRESHOLD` was never approached. There is **no result
JSON** and never will be — the arm never reached the benchmark phase at all.
`benchmark.log` holds 257 `Phase warmup progress` lines and **zero**
`Phase benchmark` lines.

**The warmup set scales with concurrency: 707 requests at c64 -> 2,845 at c256.**
That alone lengthens warmup ~4x and is worth knowing before sizing any timeout.

Client-side progression, then a flat line:

| elapsed | returned | sent | in_flight | errors |
|---|---|---|---|---|
| 30 s | 15/2,845 | 285 | 270 | 0 |
| 600 s | 173/2,845 | 285 | 112 | 0 |
| 1800 s | 223/2,845 | 285 | 62 | 0 |
| 3600 s | **223** | 285 | 62 | 0 |
| 5400 s | **223** | 285 | 62 | 0 |
| 7711 s | **223** | 285 | 62 | 0 |

**98 minutes of exactly zero progress, no errors.** Note `sent` never moved past
285 either: aiperf sent its first warmup round and was still waiting on it.

**The hang is server-side, and it is not saturation.** The server's last
scheduler line is `17:26:21`, with several DP ranks logging a prefill in the
same second and then nothing for 1 h 53 m until the driver's terminate at
19:19:17. The router's last request is the same second.

```
[17:26:21 DP7 TP7] Prefill batch, #new-seq: 1, #new-token: 8192, ...
    #running-req: 0, #queue-req: 8, #pending-token: 2,556,452, full token usage: 0.03
[17:26:21 DP0 TP0] Prefill batch, ... #running-req: 0, #queue-req: 8,
    #pending-token: 2,713,995, full token usage: 0.05
[17:26:21 DP6 TP6] Prefill batch, ... #running-req: 0, #queue-req: 1,
    #pending-token:   674,872, full token usage: 0.11
```

Read those three lines carefully, because they rule out both obvious stories:

- **Not memory / saturation.** Token usage is **0.03-0.11**. ~95 % of the KV
  pool is free. The 56.1 M-token pool was never the constraint, as predicted.
- **Not scheduler slot exhaustion, so the doc's own escalation path would NOT
  have fixed it.** Slot exhaustion would show `#running-req` pinned near
  256/8 = 32 per rank. It reads **0**. Raising the `MAX_RUNNING_REQUESTS` cap
  256 -> 512 addresses a condition that is not occurring — **do not spend the
  night on that edit.**
- **`#pending-token` is wildly imbalanced across ranks** (0.67 M vs 2.56 M vs
  2.71 M, a 4x spread) and every rank shows `#running-req: 0` while holding
  queued work. All ranks entered a prefill in the same second and none came
  back. That is the signature of a **DP-attention collective deadlock** under
  imbalanced load, not of an overloaded server.
- The client believed **62** requests were in flight while the server knew of
  only **8** queued. ~54 requests are unaccounted for between the two.

**This is the case traps 12 and 15 say is benign — and it is not.** Those traps
are right that a mid-warmup `in_flight` collapse usually clears. Distinguish
them by two checks, not by the collapse itself:

1. does `returned=` advance at all over ~12 minutes? (arm 2: no, for 98 min)
2. how old is the **server's** last log line? (arm 2: 1 h 53 m)

Benign barrier: `returned` still creeps, server still logging. Real hang: both
frozen. Amend traps 12/15 with this before trusting them again.

#### Arm 3/6 `c256-chunk8192-gd` — **HUNG TOO. The guard is not the variable. c256 is a bug, not an arm.**

Detected 20:36 by the stall detector, 12 min with `returned=` frozen. Same
signature as arm 2, confirmed on the **live** server:

| | arm 2 (guard OFF) | arm 3 (guard ON) |
|---|---|---|
| froze at | `returned=223/2,845` (**8 %** of warmup) | `returned=2,076/2,845` (**73 %** of warmup) |
| in_flight at freeze | 62 | 169 |
| server last scheduler line | 17:26:21, then silent 1 h 53 m | 20:23:42, then silent 13 min+ |
| `#running-req` / `#queue-req` | 0 / 8 | 0 / 43 |
| token usage | 0.03-0.11 | 0.13 |
| errors | 0 | 0 |

**The mixed-slot guard did not prevent the hang.** It was the one thing that
differed between these two arms and the one subsystem plausibly implicated, so
arm 3 was deliberately allowed to run rather than cancelled on arm 2's evidence.
It hung anyway. **The delayer guard question cannot be answered at c256.**

**The freeze point is wildly non-deterministic — 8 % vs 73 % of the same warmup.**
That rules out a deterministic limit (a fixed queue depth, a fixed token count)
and points to a **race**: a DP-attention collective that a rank can miss under
imbalanced load. It also means "it got further" is **not** evidence of a fix —
any future c256 attempt must be judged by reaching `Phase benchmark`, never by
progressing further into warmup than last time.

**Checked and ruled out as a confound:** a stale aiperf client from arm 2
surviving into arm 3. At 20:36 exactly one set of aiperf managers was live
(`system_controller`/`dataset_manager`/`timing_manager`/`worker_manager`/
`records_manager`, one each) with start times matching arm 3's own server. So
arm 3 hung on its own, not because two clients were driving one server. **But
the gap is real** -- see the driver bug below.

#### Arm 4/6 `c256-chunk16384` — hung too. **3 of 3 c256 arms. Guard and chunk size are both ruled out.**

Stalled 22:44 at `returned=253/2,845`, `in_flight=32`; server's last scheduler
line 22:31:22 (`#new-token: 16384`, `#running-req: 0`, token usage 0.11), then
13 min of silence, 0 batches in the last 3 min. Verified by the scheduler-age
rule, not by the stall alone.

| arm | guard | chunk/rank | froze at | outcome |
|---|---|---|---|---|
| 2 `c256-chunk8192` | off | 8192 | 8 % of warmup | hang |
| 3 `c256-chunk8192-gd` | **on** | 8192 | 73 % of warmup | hang |
| 4 `c256-chunk16384` | off | **16384** | 9 % of warmup | hang |

**Everything that varies across these three has been eliminated:** the delayer
mixed-slot guard (on and off both hang) and `CHUNK_PER_RANK` (8192 and 16384
both hang). What is left in common is `CONC=256`, and the one thing that is
structurally different about c256 in this launcher is that
`MAX_RUNNING_REQUESTS` clamps to 256 = **1x headroom**, where c64 and c128 get
2x. That is the leading hypothesis and it is **consistent with the launcher's own
warning** that AgentX concurrency counts session trees and that subagent fan-out
needs headroom rather than clipping.

It is a hypothesis, not a result: no arm has been run at c256 with the cap
raised, so 1x-headroom-causes-the-hang is **untested**. Confounded with
concurrency itself.

#### This blocks the delayer question harder than it first appears — read before proposing the c128 route

The earlier proposal in this file was: get tight slots at c128 by *lowering*
`MAX_RUNNING_REQUESTS` to 128 via a `${MAX_RUNNING_CAP:-256}` patch, since c256
was unusable. **Downgrade that plan.** If 1x headroom is what hangs the server,
then c128 at `MAX_RUNNING_CAP=128` is *also* 1x headroom and would be expected to
reproduce the hang, not dodge it. The two facts collide:

- the guard can only measure as non-null when slots are **tight** (at c64, 2x
  headroom, it measured +0.13 %, a null, because `#running-req` was median 2);
- tight slots are, on current evidence, exactly the condition under which this
  build hangs.

**So the delayer guard question is blocked on the hang, not merely postponed.**
Anyone picking this up should treat "make the server survive 1x headroom" as the
prerequisite work item, and should not spend arms hunting for a guard effect
until it is fixed. Running the c128 1x-headroom arm is still worthwhile, but
frame it as **a test of the hang hypothesis** (does 1x headroom hang at c128
too?), not as a delayer measurement -- it is a cheap, direct probe of the one
variable that survived elimination, and it is the fastest way to confirm or kill
the headroom story.

#### Driver bug found while writing the continuation: `kill_and_reclaim` misses aiperf

`kill_and_reclaim` greps `[s]glang` only. The aiperf processes carry `aiperf` in
their command line and match **nothing** in that pattern; they survive the
routine and are only cleaned up incidentally when `timeout` tears the launcher
down. It did not bite tonight (verified above), but a driver that leaves a
client alive against the next arm's server is one bad teardown away from a
silent cross-arm confound. The pattern wants to be `[s]glang|[a]iperf`.

#### The 150 min `PER_RUN_TIMEOUT` is too tight for anything above c64

The launcher sets `AGENTIC_WARMUP_GRACE_PERIOD=3600` for `CONC>=32`. A **healthy**
arm may therefore legitimately use 3600 s warmup + 3600 s benchmark + ~25 min
startup/finalise = **~145 min**, against a 150 min cap. c64 is safe (104 min
measured) but the c128 arms have only ~5 min of margin, and **a healthy c128 arm
killed at 150 min would log `FAILED rc=124` -- identical to a genuine hang.**
If a c128 arm reports rc=124, do not call it a hang without checking whether the
server's scheduler was still logging at the moment it died. The continuation
driver uses 200 min.

#### Intervention attempted 20:40 and BLOCKED — matrix left running unchanged

The plan was: stop the driver, skip arm 4 (`c256-chunk16384`, the same
configuration that has now failed twice), and run the c128 pair immediately,
recovering ~3.5 h. **The process-kill was denied by the permission classifier and
nothing was killed.** This was not worked around. State is unchanged and the
original driver is still in control.

`/workspace/claude-skills/agentx/overnight_matrix2.sh` **exists, is syntax-checked,
and was never launched.** It runs the c128 pair only, with the aiperf fix and the
200 min timeout. To use it, stop the driver and its run processes, then:

```bash
setsid nohup bash /workspace/claude-skills/agentx/overnight_matrix2.sh \
  < /dev/null > /workspace/results/overnight/driver2.log 2>&1 &
```

**Doing nothing is a sound fallback and the night still lands its remaining
answer.** Under the original driver: arm 3 times out ~21:49, arm 4 runs
21:49 -> ~00:19 and will hang, arm 5 `c128-chunk8192` ~00:19 -> ~02:05,
arm 6 `c128-chunk16384` -> ~03:50. **Both c128 arms should still finish before
morning**; the cost of not intervening is arm 4's ~2.5 h and the tight-timeout
risk above.

#### Consequence for the rest of the night — arms 3 and 4 are the same concurrency

Arm 3 `c256-chunk8192-gd` (the delayer question, the most valuable arm of the
six) and arm 4 `c256-chunk16384` are both c256 and will likely hang identically,
costing 150 min each and producing nothing. **They were not stopped pre-emptively
on this evidence**, for one reason: **arm 3 differs — it is the only arm with the
mixed-slot guard ON**, and the guard changes delayer admission, which is exactly
the subsystem implicated. It genuinely might not hang.

A stall detector is watching arm 3 and will fire on whichever comes first:
`Phase benchmark` reached (c256 viable, leave everything alone) or `returned=`
frozen 12 min (same hang, re-plan the night). Arm 3 started 19:19; arm 2 froze
~40 min after its own start, so the answer is due ~20:00.

**If arm 3 hangs too, the recommended re-plan** (needs the driver stopped first —
the lazy-read HAZARD applies to `overnight_matrix.sh` itself, not just the
launcher):

1. Skip both remaining c256 arms; run the c128 pair (arms 5, 6) next, so the
   night still returns the chunk-effect-at-c128 answer.
2. The delayer question then needs a **different** route to tight slots. Do not
   reach for higher concurrency — reach for a **lower `MAX_RUNNING_REQUESTS` at
   c128**, i.e. patch line 231 to `${MAX_RUNNING_CAP:-256}` and run c128 with
   `MAX_RUNNING_CAP=128` (1x headroom) against the existing c128 anchor. Same
   tight-slot condition, at a concurrency that demonstrably runs.
3. c256 itself is now a **bug report**, not a benchmark arm.

#### New, reusable: the noise floor is now known PER METRIC, not just on headline

From the replicate pair in "Closed 2026-08-28" (rows 1 and 2, same config, same
HEAD, same JIT cache, nothing changed):

| metric | replicate swing | so the smallest credible effect is |
|---|---|---|
| tok/s/GPU | 5.67 % | ~10 % (§25 bar) |
| TTFT mean | **26.3 %** | very large moves only |
| ITL p90 | **7.0 %** | ~10-15 % |
| intvty p90 | **6.6 %** | ~10-15 % |

This is a 1200 s pair and n=2, so treat it as an order of magnitude, not a
tolerance. It nonetheless **re-grades two claims already in this file**:

- b200align's **ITL p90 -32 % and intvty p90 +45 %** survive comfortably --
  4-7x their own noise, and reproduced independently at 1200 s. Still safe.
- b200align's **TTFT +68 %** ("7.21 s vs 3.35/4.28 s") is only ~2.6x the TTFT
  noise floor. Directionally believable and consistent with
  `--prefill-decode-interval 10`, but it is a much weaker number than its
  headline position implies. **TTFT is the noisiest metric on file** -- never
  build an argument on a TTFT delta under ~30 %.

### Reading the results — rules that must not be skipped

- **Never compare across concurrency on headline** (`conc-and-trace-mix.md`
  §19.6). c128/c256 numbers are meaningful **only** against the other chunk value
  at the *same* conc. The matrix is deliberately built in same-conc pairs.
- **The replicate spread at 3600 s is UNKNOWN.** At 1200 s it measured **5.67 %**
  on an unchanged config. 3600 s should be tighter — it averages over ~4x more
  traces — but nobody has replicated a 3600 s arm. Until someone does, do not
  call anything under ~5 % real; §25's >=10 % bar remains the safe threshold.
- Sanity-gate every arm before quoting it: `errors=0`, `records_error_dropped=0`,
  `duration_seconds` ~3630, and `input.mean` within a few % of its partner. An
  arm whose ISL moved a lot did a different workload.
- Judge health during a run from warmup progress in `benchmark.log` against a
  reference arm, **never** `/metrics` (trap 1) and never decode-batch counts
  (trap 2). Quote numbers from `benchmark.log` / the result JSON, never from a
  monitor notification (trap 14).

### Tree state the matrix depends on — verify before starting

| thing | state | how to check |
|---|---|---|
| topk_v2 (#36684) | **reverted** | `git -C /sgl-workspace/sglang status --porcelain -- python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh python/sglang/srt/server_args.py` must print **nothing**, and `server_args.py:5891` must read `set(False)`. Do **not** grep the whole status for "topk" — four untracked `.hip` build artefacts always match and look like a dirty revert. Diff saved at `/tmp/topkv2_36684.patch`; its built JIT cache parked at `~/.cache/sglang/jit.topkv2` |
| active JIT cache | pre-36684 | `~/.cache/sglang/jit` (restored from `jit.pre36684`) |
| delayer mixed-slot guard | **applied, defaults true** | `grep _mixed_slot_guard .../prefill_delayer.py` -> 3 hits. The driver sets the env var explicitly on **every** arm, so the default never leaks in |
| launcher chunk knob | **patched to honour `CHUNK_PER_RANK`** | lines 147/182 of `benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_b200align_mtp.sh`; backup `/tmp/b200align_launcher.pre_chunkenv.bak`. This file is **untracked** — do not lose the backup |

topk_v2 is deliberately **out** of this matrix: it measured -2.5 % / +3.3 %
depending on which same-config run you compare against (i.e. unresolvable at this
noise level), and toggling it unattended would need a source revert plus a JIT
cache swap between arms — a fragile thing to automate overnight. Test it later in
a **separate worktree on current main** (see Next actions).

### The one number that matters: b200align is +6.2 % over DP8+TBO at full length

**b200align c64, 3600 s, valid** (`/workspace/results/b200align-tp8-c64-3600s/`):

| | TP8 c64 | DP8+TBO c64 | **b200align** |
|---|---|---|---|
| aggregate tok/s/GPU | 17,080 | 17,443 | **18,518** |
| curve-reconstruction full | 17,219 | 17,594 | **18,670** |
| vs TP8 | — | +2.1 % | **+8.4 %** |
| vs DP8+TBO | — | — | **+6.2 %** |

Per-cutoff, same method for all three (cumulative tokens / cutoff / 8):

| cutoff | TP8 | DP8+TBO | b200align |
|---|---|---|---|
| 300 s | 13,613 | 10,988 | 14,002 |
| 600 s | 15,166 | 16,306 | 16,652 |
| 900 s | 17,320 | 17,740 | 18,963 |
| 1200 s | 17,325 | 17,322 | 18,792 |
| 1800 s | 18,266 | 18,171 | 19,232 |
| 2400 s | 18,649 | 18,521 | 19,337 |
| 3000 s | 17,856 | 18,180 | 19,132 |
| 3600 s | 17,219 | — | **18,670** |

b200align leads at **every** cutoff with no crossover — stronger than the
endpoint alone, given §25 records the TBO-vs-TP8 ordering flipping three times.
Still **below §25's >=10 % bar**, so: direction clear, ordering stable, magnitude
moderate, *not* "proven" by this file's own standard.

**Read this table with the 2026-08-28 replicate result in hand:** repeating the
b200align 1200 s arm with nothing changed moved it **-5.67 %**. That is close to
the entire +6.2 % claimed here. What still supports the claim is the *3600 s*
run's monotone per-cutoff lead, not any single 1200 s number — and no arm here
has been replicated, so the +6.2 % has no error bar. Treat every 1200 s row in
this file as one draw from a distribution roughly +-6 % wide. The 1200 s run
reading +9.1 % is that spread, not a short-window bias.

**Latency is the bigger story than throughput:**

| mode | conc | tok/s/chip | P90 intvty | ITL p90 | TTFT avg | cache hit | dur |
|---|---|---|---|---|---|---|---|
| TP8 | 64 | 17,080 | 13.8 | 72.7 ms | 3.35 s | 0.689 | 3629 s |
| DP8+TBO | 64 | 17,443 | 13.5 | 73.9 ms | 4.28 s | 0.955 | 3629 s |
| **b200align** | 64 | **18,518** | **20.0** | **50.0 ms** | 7.21 s | 0.955 | 3630 s |
| b200align | 64 | 18,485 | 20.7 | 48.2 ms | 5.75 s | 0.956 | 1227 s |

Fields: `request_metrics.latency.{intvty.p90, itl.p90, ttft.mean}`,
`server_metrics.cache.gpu_cache_hit_rate`. **P90 interactivity +45 %, ITL p90
-32 %** — far outside noise and reproduced independently by the 1200 s run.
**TTFT is the price: 7.21 s vs 3.35/4.28 s.** Consistent with
`--prefill-decode-interval 10` forcing decode between prefills: smoother
inter-token, slower first token.

**Attribution is exact.** A full `sglang_command.txt` diff against
`tbo-tp8-c64` shows **four** differing flags and nothing else — no other
additions, removals or value changes:
`--enable-dp-attention-local-control-broadcast`, `--prefill-decode-interval 10`,
`--stream-interval 20`, `--tokenizer-worker-num 8`. The +6.2 % / +45 % belongs to
those four **as a set**; no per-flag attribution exists yet.
`--incremental-streaming-output` was dropped during bisection and is **not** in
this result. Comparability: duration 3630 vs 3629 s, `input.mean` 116,923 vs
116,248/115,357, `output_actual.mean` 965.3 vs 956.0/919.4, `errors=0`, zero
hard failures.

**UNRESOLVED — TP8 cache hit reads 0.689 here but §22/§23 record TP8 at 91.8 %
and DP at 83.2 %, i.e. the opposite ordering.** Both DP arms read 0.955. §15
warns the cache metric is device-tier only. Do not cite either number until this
is reconciled.

### Closed 2026-08-28 — the 1200 s window cannot resolve <6 %; both patches null

Four c64 runs, all `agentx_b200align.sh` (DP-attn + TBO + delayer + the four
B200 flags). Rows 1 and 2 are the **same configuration measured twice**:

| run | tok/s/GPU | ITL p90 | intvty | TTFT | ISL mean | OSL mean | succ | err |
|---|---|---|---|---|---|---|---|---|
| `b200align-tp8-c64-1200s` 07:13 | **18,484.5** | 48.21 | 20.74 | 5.75 s | 108,503 | 766.7 | 1660 | 0 |
| `rebaseline-notopkv2-noguard-c64-1200s` 14:41 | **17,435.8** | 51.59 | 19.38 | 7.26 s | 107,603 | 738.1 | 1580 | 0 |
| `topkv2-nodelayerfix-c64-1200s` | 18,014.7 | 48.48 | 20.63 | 7.33 s | 109,245 | 773.1 | 1609 | 0 |
| `topkv2-delayerfix-c64-1200s` | 18,037.6 | 48.62 | 20.57 | 7.04 s | 108,522 | 764.8 | 1623 | 0 |

**THE FINDING: repeat-run spread on an unchanged config is -5.67 %.** Row 2 is a
true replicate of row 1 — same HEAD `a1f9508dd4`, the *same* restored JIT cache
(`jit.pre36684`), topk_v2 reverted via `git checkout`, guard off via env. Nothing
differed. The gap is in **work completed** (1580 vs 1660 successful requests,
-4.8 %) with ISL flat (-0.83 %), not in per-request speed.

Consequences, and they are large:

- **Any 1200 s single-run A/B below ~6 % is uninterpretable.** §25's >=10 % bar is
  the right instrument; most of today was spent reading 0.1-2.5 % deltas as if
  they meant something.
- **Do not cite "same-config spread is 1.7 %".** That number (18,792 from the
  3600 s curve vs 18,485 standalone) compared a curve-reconstruction value to an
  independent run — not a replicate. The measured replicate spread is 5.67 %.
- To measure anything smaller, either run 3600 s, or run each arm >=3 times and
  report a band. Budget accordingly before promising an answer.

**1. sglang#36684 topk_v2 — verdict: unresolved, not "no effect".** vs the 07:13
baseline it reads -2.54 %; vs the same-day replicate it reads **+3.32 %**. Both
are inside the 5.67 % replicate spread, so neither sign is established. What *is*
solid: **it builds and runs on gfx950.** The fear recorded here that it needs
`<cooperative_groups.h>` / `cg::this_cluster()` and would fail on ROCm was
**wrong** — `jit/gfx950/sgl_kernel_jit_dpsk_v4_topk_v2/.../*.so` compiled clean,
zero `cooperative_groups` hits.

*Also unresolved, and possibly the real question:* this was a **cherry-pick onto
a 2026-08-25 base**, not a mainline update (reflog shows only the original clone;
no fetch/pull ever). If 36684's benefit depends on other changes that landed in
main during those three days, cherry-picking it onto an older base would
understate it. Testing that needs a **separate worktree** updated to current main
(never this tree — it carries ~20 uncommitted local files), running topk_v2
on/off there.

**2. PrefillDelayer mixed-slot guard (`2d4cf9bd2`) — ported, green, no measurable
effect.** +0.13 % vs the topk_v2 arm; far inside the replicate spread. Unit test
passes (see traps for how to run it and how to prove your case actually ran).

*Why it plausibly cannot matter here:* `--max-running-requests 128` over DP8 =
**16 per rank**, and observed `#running-req` is median **2**, p90 **8**, max
**13** (n=7328). The workload never nears the slot ceiling, so "keep delaying
while slots are short" has nothing to protect. **Unproven:** whether the guard
ever fired — free slots ~8 means the condition trips whenever
`max_prefill_bs > 8`, so it is probably live but idle. Settle it with
`SGLANG_PREFILL_DELAYER_DEBUG_LOG=1`.

*This config is maximally exposed to the regression the fix targets*, which is
the one thing that makes the null informative: `token_usage_low_watermark=None`
and `queue_trigger_enabled=False`, so the watermark escape in the `mixed` branch
is inert and the generic `max_delay_passes=30` timeout is the **only** release
path — exactly what the fix intercepts.

*State of the tree right now:* topk_v2 is **reverted** (`git checkout`; the diff
is saved at `/tmp/topkv2_36684.patch`, active JIT cache is the restored
`jit.pre36684`, the topk_v2-built cache is parked at `~/.cache/sglang/jit.topkv2`).
The delayer guard is **applied and defaults to `true`** — every future run
carries it unless you set `SGLANG_PREFILL_DELAYER_MIXED_SLOT_GUARD=0`. Harmless
on this workload; a live variable on anything that saturates slots. Backups:
`/tmp/prefill_delayer.py.pre_delayerfix.bak`,
`/tmp/test_prefill_delayer.py.pre_delayerfix.bak`.

*Porting notes.* `git apply` fails on both files — the tree has diverged. Port by
hand; the commit is local at `/sgl-workspace/sglang-moega-moe`
(`origin/feat/flydsl-a2a`; `87524746c` on `origin/eval/megamoe-pr876-20260730` is
byte-identical). Three parts: env var (default true), `max_running_requests` into
the DP all-gather (buffer 5->6, read `tp0_info[:, 5]`), slot guard in the `mixed`
branch. The tree already had `max_running_requests` plumbed into the **"all"**
branch only — do not mistake that for the fix being present.

*Confirming each switch is really engaged*, both cheap and both worth doing:
`grep "mixed_slot_guard=" server.log` -> the `PrefillDelayer initialized with ...`
line (that field exists **only** because the port adds it), and for topk_v2 the
presence of a compiled `.so` under `~/.cache/sglang/jit/gfx950/`. Note
`SGLANG_OPT_USE_TOPK_V2` **cannot be turned off by env** — `server_args.py` does
`envs.SGLANG_OPT_USE_TOPK_V2.set(True)`, which overrides it; you must revert the
source.

### Traps learned 2026-08-28 — these cost most of a day

1. **`/metrics` under-reports when `--tokenizer-worker-num > 1`.**
   `sglang:generation_tokens_total` read ~1 token per returned request on runs
   actually generating 767-965 tok/req, and `sglang:num_running_reqs` read 0.0
   on all 8 DP ranks while requests were being served. The frontend is split
   across 8 tokenizer processes and the endpoint appears to report one.
   **This single artefact produced three wrong root causes in a row**
   (`--chat-template`, then the scratch-reclaim OOR, then "the four B200 flags
   break generation") and a healthy run was killed on the strength of it.
   Judge generation health by **warmup progress vs the reference arm at matched
   `elapsed`** — `tbo-tp8-c64/benchmark.log` carries the per-30 s series
   (13/21/31/38/40/44 at 30..180 s) — by `errors=` in the aiperf line, and
   afterwards by `request_metrics.tokens.output_actual.mean`.
2. **Early `Decode batch` counts prove nothing.** tbo logs **0** decode batches
   across its first 2147 prefill batches. A prefill-only early log is normal.
3. **The server survives the launcher.** Twice, a completed run left
   `sglang.launch_server` alive holding 92 % VRAM with no launcher process.
   `kill -TERM` on the process group did not work (a second launcher process
   even appeared); only per-PID `kill -9` matching `sglang.launch_server` **and**
   `sglang::` cleared it.
4. **KFD reclaim is bursty, not gradual.** After the kill, 5 of 8 GPUs sat at
   90 %+ for ~11 minutes with **no process holding memory**
   (`rocm-smi --showpids` showed only `gpuagent` at 0), then all 8 dropped to 0
   within 30 s. Wait for it; do not conclude a leak.
5. **`pgrep -f <pattern>` matches your own shell command.** It falsely reported
   vim open, aiperf running and stray servers, four separate times. Use
   `ps -eo comm` / `ps -eo args` with a bracketed first character instead.
6. **Monitors that dedupe with `comm` consume a token permanently.** A `FAILED`
   match on `AIPERF_FAILED_REQUEST_THRESHOLD=0.10` in the env dump meant a real
   later `FAILED` could never fire. Filter out `^[A-Z_]+=` / `AIPERF_` lines.
7. **Never let a curl's `--max-time` exceed the tool timeout**, and never probe a
   loaded server with a short timeout and read 0 bytes as "broken" — under
   AgentX load both router and backend return nothing within 60 s.

8. **`ps -eo args | grep <pattern>` matches your own `bash -c` wrapper.** This is
   trap 5 in a second form and it is worse: the wrapper's command line contains
   the pattern you just typed, so a kill loop built from it **kills your own
   shell** and the kills never run — silently, looking like a no-op. Always add
   `| grep -v "bash -c"` and exclude `$$`.
9. **A crash detector with no "ready" gate fires on every launch.** A monitor
   that declares death when `sglang::scheduler_DP0` is absent will report `CRASH`
   during the launcher's pre-server phase (`uv pip install`, `hf download`),
   which lasts minutes. Gate it on having seen "The server is fired up and ready
   to roll" first, and add a separate `[ABORT]` for "launcher gone before the
   server ever came up".
10. **Verify a monitor's regex against a real log before trusting its silence.**
   A pattern of `Warmup...` never matches the actual line, which is
   `Phase warmup progress | returned=13/707 | ... | errors=0 | elapsed=30.0s`
   (lowercase, fraction after a `|`). The monitor stays quiet, and quiet is
   indistinguishable from healthy. Test the grep against a finished arm's
   `benchmark.log` first; it should reproduce §the reference series 13/21/31.
11. **Compare warmup against the reference arm, not the fastest arm.** Judged
   against `topkv2-nodelayerfix` (an unusually fast run, 175 at 600 s), the
   delayer-fix run looked stalled at 66. Against `tbo-tp8-c64` (52/94/262/701 at
   300/600/900/1200 s) it was normal the whole way and finished 693/707. Pick the
   reference **before** the run, and expect single points to swing: the same run
   read -12 % at 150 s and +86 % at 600 s.
12. **`in_flight` collapsing to 1 mid-warmup is not a hang.** aiperf's warmup
   appears to barrier: 66 of 67 returned and one straggler (a ~109 k-token trace
   chunk-prefilled at 8192/chunk) holds the round. It clears. Check
   `#queue-req` on the server before blaming the server — `0` means the client
   is not sending, so nothing server-side is throttling.
13. **`test_prefill_delayer.py` is not one test.** Only
   `TestPrefillDelayerNegotiate` is the fast pure-`torch.distributed` path; the
   other four classes each `_launch_server`, which is what blows past a 600 s
   timeout. Run
   `pytest -q test/registered/scheduler/test_prefill_delayer.py::TestPrefillDelayerNegotiate::test_negotiate`
   (~21 s). And because `test_negotiate` **iterates** all cases inside one test,
   "1 passed" does not prove your new case ran — flip its expectation once and
   confirm the failure names your case, then flip it back.

14. **Trust the log file, not the monitor notification.** A monitor event
   reported `returned=143 | sent=245 | in_flight=102` at 600 s; `grep
   returned=143` over the whole result directory found **nothing**, and the log's
   only 600 s line read `returned=111 | sent=203 | in_flight=92`. There was no
   restart (`Phase warmup started` appears once). Cause never established. Quote
   `benchmark.log` for any number you are going to reason from, and finally the
   result JSON — the notification stream is a trigger, not a source.
15. **The mid-warmup `in_flight` collapse is intrinsic, not a symptom.** Seen at
   600 s in the delayer-fix arm (`in_flight=1`) and reproduced at 570 s in the
   re-baseline arm (`in_flight=2`) which had **neither** patch. aiperf's warmup
   barriers on a straggler (a ~109 k-token trace chunk-prefilled at 8192/chunk),
   then takes off. Do not spend an hour attributing it to whatever you just
   changed — check `#queue-req` on the server first: `0` means the client is not
   sending and nothing server-side is throttling.

16. **aiperf prints thousands separators, and a `[0-9]+` regex silently stops
   matching.** At c64 the warmup line reads `returned=13/707`; at c256 it reads
   `returned=223/2,845`. Every health check built on `returned=[0-9]+/[0-9]+`
   matched **nothing** on the c256 arm and reported quiet — trap 10 all over
   again, this time inside our own tooling. `warmup_check.sh` was fixed to use
   `[0-9,]` + `tr -d ,`. Any new parser must be tested against a **c256** log,
   not just a c64 one.

17. **The aiperf phase is called `profiling`, not `benchmark`.** A monitor
   grepping for `Phase benchmark` as a positive health signal **never fired once
   across five arms**, and its silence was indistinguishable from "no arm ever
   became healthy". The real phase names are `Phase warmup`, `Phase profiling`,
   and a single terminal `Phase complete`. This is trap 10 committed a second
   time in our own tooling (trap 16 was the first). **Before trusting any
   monitor's silence, grep its pattern against a finished log and confirm it
   produces hits.**

18. **Verifying that a JIT kernel is ACTIVE: read `/proc/<pid>/maps`, not the
   log.** SGLang never prints the kernel name, so grepping `server.log` for
   `topk_v2` returns **0 even when the kernel is loaded on all 8 ranks** -- an
   "activation check" built that way reports the arm is void and would have had
   a healthy 2-hour run killed. Trap 10 for the third time this session, and the
   first time it produced a **false negative** rather than false silence. The
   reliable checks, in order:

```bash
# 1. is the kernel built, in the cache that is actually mounted?
find ~/.cache/sglang/jit -iname "*topk*"          # pre36684 cache has none
# 2. DEFINITIVE: is it mapped into the running schedulers?
for p in $(ps -eo pid,args | grep "[s]glang::scheduler" | grep -v "bash -c" | awk '{print $1}'); do
  grep -q topk_v2 /proc/$p/maps && echo "$p loaded"; done   # expect 8
```

   Validate any activation check against a **known-positive** case before
   trusting it in either direction.

### RETRACTION OF A RETRACTION — the `--chat-template` story is unresolved

Earlier today this file (a) called for landing `--chat-template` as a
correctness fix, (b) landed it on all four MI355X launchers, then (c) retracted
it, blaming it for collapsing generation to ~1 token/request. **(c) is also
wrong.** The next run, with no template, showed the same "~1 tok/req" reading —
which turned out to be trap 1 above, a metrics artefact. There was very likely
no collapse at any point.

What is still solid: `--tool-call-parser deepseekv4` makes
`resolve_chat_encoding_spec()` return `"dsv4"`
(`entrypoints/openai/chat_encoding.py:112`), so `encoding_dsv4.py` — not a jinja
template — renders DSv4 prompts, and `--chat-template` overrides that native
path. The startup line `No chat template found, defaulting to 'string' content
format` is misleading log noise. So the flag is **unnecessary**; it was never
shown to be **harmful**. It is currently absent from all four launchers, which
matches every reference arm, and that is the right default. §24.1 in
`references/b200-alignment.md` still overstates the case and needs this
correction folded in.

**What genuinely went wrong in the 04:23 and 04:56 runs is still unexplained.**
Those two really were degraded — warmup 324/707 at 2430 s against tbo's 701/707
at 1200 s, and both ended in
`HSA_STATUS_ERROR_OUT_OF_RESOURCES` (`Available Free mem : 318 MB`) — that part
is not an artefact. Between then and the first healthy run, **two** things
changed: the template was removed **and** `--enable-prefill-delayer` +
`--enable-two-batch-overlap` were added back. Both changed at once, so neither
is established. `tbo_mtp.sh:170` calls guarded TBO+delayer "the validated
combination", which makes the delayer/TBO explanation the more likely one, but
it is a hypothesis. The 06:41 run (no template, no delayer/TBO) was tracking tbo
normally at elapsed=150 s when it was killed, which weakly argues **against** the
template being the cause.

`HSA_NO_SCRATCH_RECLAIM=0` was added, then removed again, and is **not** in the
working config; the container default (=1) is what all reference arms use.
`benchmarks/multi_node/amd_utils/env.sh:312` does pin it to 0 for
DeepSeek-V4-Pro with the comment "resolve the OOR issue", so if OOR ever recurs
on a healthy workload, that is the lever — and *that* result would be real
evidence rather than a workaround validated on a broken run.

### Next actions

1. **Establish the noise floor before any more A/Bs.** Everything below ~6 % at
   1200 s is currently unmeasurable. Either run each arm >=3x and report a band,
   or move to 3600 s. Cheapest useful step: two more b200align 1200 s replicates
   (~45 min each) to turn the single -5.67 % observation into an actual spread.
   **Do not run another single-shot 1200 s A/B until this exists.**
2. **Per-flag bisect of the four B200 flags** — still the highest-value open
   question (the set is worth +6.2 % / +45 %), but per (1) it needs 3600 s runs
   or replicates, so budget ~3 h per flag, not 90 min.
3. **topk_v2 on current main, in a separate worktree.** Today tested a
   cherry-pick onto a 2026-08-25 base; if the PR's benefit depends on the three
   days of main it was written against, that understates it. Never update this
   tree (~20 uncommitted local files).
4. ~~Reconcile the TP8 cache-hit contradiction (0.689 vs §22/§23's 91.8 %).~~
   **CLOSED 2026-08-29.** Not a contradiction: 0.689 is the **device tier** of a
   GPU-saturated run whose **overall** hit rate is 0.950. See the CORRECTION
   under "Matrix results as they land". Nothing to run.
5. Settle the 04:23/04:56 degradation: rerun the healthy config with delayer+TBO
   **removed**, template still absent.
6. Optional, cheap: `SGLANG_PREFILL_DELAYER_DEBUG_LOG=1` to settle whether the
   mixed-slot guard ever fires here.
7. §24.1 in `references/b200-alignment.md` still overstates the
   `--chat-template` case and needs the correction below folded in.

**Still parked from before:** `--tokenizer-worker-num 8` + `--stream-interval 20`
as an isolated pair (§24) — now partly answered, they are inside the +6.2 %
four; the 64-group cache-aware routing re-run (§23); stock-MoE `dp8 + ep8` at
c64 (§20.3c); `--enable-deepseek-v4-fp4-indexer` behind an accuracy gate;
`--load-balance-method total_tokens` (§18); the two sglang#35619 bugs (§17).

**Do not redo:** conc 48 full arm (§19); §20 H1/H2; the B200-vs-MI355X recipe
comparison (§20.3c); the full §21/§22 ladder; §23's routing run in its 8-key
form.

**Traps to re-read before touching anything** — §14 (stale-router trap), §15
(cache metric is device-tier only), §17 (#35619 blocks TBO), the four in §22.6
and §23, plus the seven above. While a run is in flight, test for the aggregate
by its **exact** `RESULT_FILENAME`, never `ls "$D"/*.json`.

### Reproducing an arm

```bash
# 1200 s b200align (current working config)
EP_SIZE=1 CONC=64 DURATION=1200 RESULT_DIR=/workspace/results/<name> \
  bash /workspace/claude-skills/agentx/agentx_b200align.sh
```
Launch it under `setsid nohup ... < /dev/null &` — a session kill took down the
03:35 run through its process group. Keep `--warmup-requests-per-lane 10`; cut
only `--benchmark-duration` (§25).

### Results index — every arm on file

Headline is tok/s/GPU for agentic trace replay, req/s for fixed-seq-len.
**Never compare agentic arms across concurrency on headline** — the trace mix
is an outcome of concurrency, not a controlled variable
(`references/conc-and-trace-mix.md` §19.6).

| arm | conc | headline | note | detail |
|---|---|---|---|---|
| TP8 agentic | 48 | 19,171.9 tok/s/GPU | +2.33 % vs published; longer traces | `conc-and-trace-mix.md` §19 |
| TP8 agentic | 64 | 17,079.5 | more reqs, more computed tokens than c48 | §16 |
| DP8+TBO agentic | 64 | 17,443.3 | +2.1 % over TP8, TTFT p50 3.10 s | §16 |
| MegaMoE+EPLB agentic | 64 | 12,630.8 | −26 % — MTPR padding vs variable ISL | `megamoe.md` §13 |
| ladder A/B/C/D | 64 | see table | DP +35 %, TBO −0.4 %/−10.8 % | `dp-tbo.md` §22 |
| ladder B + router | 64 | 2.20-3.12 req/s | routing fixes cache, imbalance kills it | `dp-tbo.md` §23 |
| b200align agentic | 64 | 18,484.5 | 1200 s, run 1 of 2 | "Closed 2026-08-28" |
| **matrix arm 1** `c64-chunk16384` | 64 | **18,724.0** | 3600 s, chunk 16384: **+1.11 % = null** vs 18,518.0 | "Matrix results as they land" |
| `c128-chunk8192` | 128 | 24,539.6 | **INVALID** -- aiperf coverage 94.1 % < 95 % | 2x rerun |
| `c128-chunk16384` | 128 | **25,968.4** | valid; TTFT 10.63 s vs 27.35 s at 8192 | 2x rerun |
| `c256-chunk8192-2x` | 256 | **27,171.7** | valid; **hang fixed by max-running 512**; TTFT 94 s, cache 0.661 | 2x rerun |
| `c256-chunk16384-2x` | 256 | **30,745.5** | valid; **+13.15 % vs 8192, clears the >=10 % bar**; TTFT p50 21.5 s | 2x rerun |
| `c128-chunk16384-topkv2` | 128 | 26,943.3 | valid; topk_v2 **+3.75 % = not resolvable** (all metrics under their noise floors) | topk_v2 test |
| **`c64-chunk16384-newmain`** | 64 | **20,131.6** | sglang main `cdbfe90b4a`: **+7.52 %** vs 18,724.0, all secondaries same-sign, n=1 | "CONTINUE HERE" |
| b200align **replicate** | 64 | **17,435.8** | same config, nothing changed: **-5.67 %** | same |
| + topk_v2 (#36684) | 64 | 18,014.7 | -2.5 % vs run 1, +3.3 % vs replicate — unresolved | same |
| + delayer slot guard | 64 | 18,037.6 | +0.13 % — inside noise; slots never tight | same |

### Where the detail lives

- `references/published-arms.md` — §10 reproducing the published points, §11
  smoke run, §12 verified reproduction.
- `references/megamoe.md` — §13 MegaMoE MTPR vs `--chunked-prefill-size`.
- `references/conc-and-trace-mix.md` — §16 three serving paths at conc 64,
  §19 the conc 48 result and why cross-concurrency headlines are invalid,
  §25 how short a window can be (the >=10 % rule).
- `references/dp-tbo.md` — §18 rank imbalance, §20 the open investigation,
  §21 the ladder design, §22 its 8-point result, §23 cache-aware routing.
- `references/b200-alignment.md` — §24 flags B200 sets that we do not.

**Scripts here:** `agentx_env.sh` (sourced by all), `agentx_run.sh` (published
arms), `agentx_tbo.sh`, `agentx_megamoe.sh`, `agentx_smoke.sh`,
`agentx_debug.sh` (fast loop, §14 — start here for any diagnosis),
`agentx_ladder.sh` (§22 fixed-seq-len ladder), `agentx_router.sh` (§23).
Long runs should point at a frozen copy (`.ladder_frozen.sh`,
`.router_frozen.sh`) kept **in this directory** — a copy in `/tmp` breaks
`SKILL_DIR` and cannot source `agentx_env.sh`.

**External state:** `/workspace/InferenceX` @ `8fcfc6283` (+launchers under
`benchmarks/single_node/agentic/`, NOT version-controlled),
`/workspace/agentx-runtime/venv`, `/shared_nfs/hf_cache`, `/workspace/results/*`
(all completed runs incl. `armB-tp8-c48` and `ladder/`, with Prometheus
exports).

## 1. Layout — what has to exist where

`benchmarks/benchmark_lib.sh` hardcodes a container layout:

```
INFMAX_CONTAINER_WORKSPACE   default /workspace        <- must be the REPO ROOT
  ├── utils/agentic-benchmark/   requirements.txt, scripts/, analysis/
  ├── utils/aiperf/              git submodule -> SemiAnalysisAI/aiperf
  └── utils/agentic/             aggregation + validation python packages
```

Two things follow, and both bite:

- `write_agentic_result_json` and the power/validation steps do
  `cd "$INFMAX_CONTAINER_WORKSPACE"` and then `python -m utils.agentic...`.
  If the repo is **not** cloned directly at `/workspace`, you must export
  `INFMAX_CONTAINER_WORKSPACE=<repo root>` or every post-run aggregation step
  fails with `No module named utils`.
- `utils/aiperf` is a **submodule and is empty after a plain `git clone`**.
  `install_agentic_deps` does `uv pip install -e "$AIPERF_DIR"`, which fails on
  an empty directory. Init it explicitly.

## 2. Build the environment

```bash
# 2.1 repo at the pinned commit + the aiperf submodule
git clone https://github.com/SemiAnalysisAI/InferenceX.git /workspace/InferenceX
cd /workspace/InferenceX
git checkout 8fcfc62830f76848b7431d051794349cf4680cf7
git submodule update --init --recursive utils/aiperf   # -> aiperf @ 754356e9

# 2.2 env file — ships with this skill, no need to write it.
#     Check the two paths inside still match this machine:
#       INFMAX_CONTAINER_WORKSPACE=/workspace/InferenceX   (repo root)
#       HF_HOME=/shared_nfs/hf_cache                       (NOT /workspace: 98% full)
cat /workspace/claude-skills/agentx/agentx_env.sh

# 2.3 isolated AIPerf venv (uv is auto-downloaded if absent)
cd /workspace/InferenceX
( source /workspace/claude-skills/agentx/agentx_env.sh
  source benchmarks/benchmark_lib.sh
  install_agentic_deps ) > /tmp/agentx_install.log 2>&1

# 2.4 pre-fetch the trace corpus (1.8 GB, public, no HF token needed)
source /workspace/claude-skills/agentx/agentx_env.sh
"$AIPERF_VENV/bin/hf" download --repo-type dataset semianalysisai/cc-traces-weka-062126
```

### Why `AIPERF_RUNTIME_DIR` is pinned

The default is `${TMPDIR:-/tmp}/inferencex-agentic-$$` — **`$$` is the shell
PID**, so every invocation gets a fresh directory and
`install_agentic_deps` (which starts with `rm -rf "$AIPERF_VENV"`) rebuilds the
whole venv from scratch. Pinning it to a stable path keeps the uv cache warm;
the rebuild then takes seconds instead of minutes.

### Why the venv is separate from the server's Python

`install_agentic_deps` deliberately refuses to share site-packages with
SGLang/vLLM: installing AIPerf into the server's interpreter can upgrade
FastAPI/Starlette/transformers underneath a running server. It also pins
**Python 3.11** via `uv venv --python 3.11`, because aiperf's
`requires-python = ">=3.11,<3.14"` while the sglang-rocm / vllm-rocm images
still ship 3.10 as `python3`. uv downloads a standalone 3.11 if the image has
none. Verified here: system `python3` is 3.10.12, venv is 3.11.16.

## 3. Trace source

`resolve_trace_source` picks the loader from `MODEL_PREFIX`, not from the
script name:

| `MODEL_PREFIX` | loader | HF dataset |
|---|---|---|
| `dsv4*`, `glm5.2*`, `minimaxm3*`, `kimik3*` | `semianalysis_cc_traces_weka_062126` | `semianalysisai/cc-traces-weka-062126` |
| anything else | `..._062126_256k` | `semianalysisai/cc-traces-weka-062126-256k` |

`MODEL_PREFIX` is **not** in the `check_env_vars` list, so leaving it unset does
not error — it silently falls through to the 256k-capped corpus. Set it.
Override with `WEKA_LOADER_OVERRIDE=<loader name>` (14 accepted names, see the
`case` in `resolve_trace_source`).

The corpus holds 393 unique traces; `build_replay_cmd` passes
`--num-dataset-entries 393` so all of them load.

## 4. Running

```bash
cd /workspace/InferenceX
source /workspace/claude-skills/agentx/agentx_env.sh

export MODEL="deepseek-ai/DeepSeek-V4-Pro"
export MODEL_PREFIX="dsv4"
export MODEL_PATH="/shared_nfs/models/DeepSeek-V4-Pro"   # 805 GiB, 64 shards
export TP=8 EP_SIZE=1 DP_ATTENTION="false"
export CONC=32
export IS_AGENTIC=1
export KV_OFFLOADING="none" TOTAL_CPU_DRAM_GB=0
export DURATION=3600
export PORT=8888
export RESULT_DIR="/workspace/results/dsv4-tp8-c32"
export RESULT_FILENAME="dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep1-dpafalse_disagg-false_spec-mtp_agentic_c32"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"
mkdir -p "$RESULT_DIR"
bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh
```

### Deltas from the "reference" command that circulates for this script

| Reference said | Reality |
|---|---|
| `cd /workspace` then `bash benchmarks/...` | only correct if the **repo itself** is at `/workspace`; otherwise `cd` into the repo and set `INFMAX_CONTAINER_WORKSPACE` |
| `MODEL_PATH=/models/DeepSeek-V4-Pro` | does not exist here; weights are `/shared_nfs/models/DeepSeek-V4-Pro` (symlink to `/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro`). If the path is missing **or empty** the script silently starts an 805 GiB `hf download` into it |
| `RESULT_DIR=/workspace/results` | fine, but `RESULT_FILENAME` is **also** required — `write_agentic_result_json` writes `$AGENTIC_OUTPUT_DIR/$RESULT_FILENAME.json` and produces `.json` (a dotfile) when unset. It is set by the CI runners, not by the launcher |
| — | `AGENTIC_OUTPUT_DIR` defaults to `$INFMAX_CONTAINER_WORKSPACE`, i.e. the aggregate lands in the repo root. Point it at `$RESULT_DIR` |
| `DURATION=3600` | the scenario enforces a **900 s minimum**; below that the launcher adds `--unsafe-override` and flags `submission_valid=false` |

### Knobs for a short smoke run

- `DURATION=300` → auto `--unsafe-override`.
- `AIPERF_WARMUP_REQUESTS_PER_LANE=1` → skip the 10-request-per-lane warmup ramp.
- `AIPERF_EXPERIMENTAL_FAST=1` → forces `duration=1200` **and** warmup 1/lane
  (it overrides `DURATION`, so don't combine it with a shorter `DURATION`).
- `EVAL_ONLY=true` → skips replay entirely, runs `run_eval` instead, and
  disables the simulated-acceptance-length pin.

## 5. What the launcher actually does

1. `check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION` — anything else (`MODEL_PREFIX`, `RESULT_FILENAME`, `PORT`) is unvalidated.
2. **GPU drain gate**: polls `rocm-smi --showmemuse` for up to 15 min and refuses to start until *every* GPU is ≤10 % VRAM. An 805 GiB checkpoint takes minutes to reclaim after a previous run, and booting into a half-drained node fails RCCL init with a bogus HIP "unhandled cuda error".
3. `resolve_trace_source` → `install_agentic_deps` → `hf download` the dataset.
4. Launch SGLang, `wait_for_server_ready` on `/health` (no timeout — it waits until the PID dies).
5. `build_replay_cmd` → `run_agentic_replay_and_write_outputs`:
   AIPerf replay → `process_agentic_result` → GPU power adapter →
   `analyze_benchmark_distributions.py` → `validate_agentic_result`.

Note the ordering: **results are written before validation**, and the function
returns the replay exit code *after* writing artifacts. A non-zero exit still
leaves usable artifacts in `$RESULT_DIR`.

## 6. Recipe specifics — dsv4 FP4 + SGLang MTP on MI355X

- `deepseek-ai/DeepSeek-V4-Pro` is FP4/FP8 **mixed** (FP4 MoE expert weights,
  FP8 elsewhere); InferenceX labels it `precision: fp4`. The top-level
  `quantization_config` in `config.json` reads `fp8` — that is expected, not a
  wrong checkpoint.
- Spec decode is **EAGLE with `--speculative-eagle-topk 1`**, not NEXTN: the
  V3/R1 NEXTN loader crashes on the V4 architecture. Depth 3.
- Throughput runs pin acceptance length: `SGLANG_SIMULATE_ACC_LEN=2.49` from
  `golden_al_distribution/dsv4_mtp.yaml` (thinking_on, depth 3). `EVAL_ONLY=true`
  turns this off so accuracy stays real.
- **No `--chat-template`** on purpose: `deepseek_v4_thinking.jinja` drops tool
  definitions and `role: tool` messages, which would truncate the tool-heavy
  AgentX prompts and distort ISL.
- `MEM_FRACTION_STATIC=0.89`, `CHUNKED_PREFILL_SIZE=8192`, `--page-size 256`,
  `--kv-cache-dtype fp8_e4m3`, `--watchdog-timeout 3600`.
- `MAX_RUNNING_REQUESTS = min(2*CONC, 256)`, `CUDA_GRAPH_MAX_BS = min(that, 128)`
  — AgentX `CONC` counts **session trees**, and subagent fan-out pushes
  instantaneous request concurrency above it.
- `CONC >= 32` bumps `AGENTIC_WARMUP_GRACE_PERIOD` to 3600.
- The DP-attention branch (`DP_ATTENTION=true`) starts `sglang_router` with
  consistent hashing on the AIPerf correlation id, on `PORT`, with the engine
  moved to `PORT+1`. It is dormant — no dp-attn arm exists for this key in
  `configs/amd-master.yaml`.

## 7. Gotchas hit while building this

- **Empty `utils/aiperf`** — plain clone leaves the submodule empty; the venv
  install then fails late, and the failure surfaces as `aiperf: No such file or
  directory` rather than as an install error.
- **Python 3.10 default** — see §2; without the 3.11 pin the venv silently ends
  up without `aiperf`/`hf` binaries.
- **`INFMAX_CONTAINER_WORKSPACE`** — see §1. Post-run aggregation is the only
  thing that breaks, i.e. *after* an hour of benchmarking.
- **`SGLANG_ENABLE_UNIFIED_RADIX_TREE` is deprecated** in sglang
  `0.5.18.dev20260825`; the launcher still exports it and the server prints a
  `UserWarning` at startup. Harmless.
- **Disk**: `/workspace` here is a 10 T NFS volume at 98 % (≈260 G free). The
  venv (~1 GB) and results fit; model weights and the HF cache must not go
  there. `HF_HOME` is pointed at `/shared_nfs/hf_cache` in the env file.
- **`rocm-smi` low-power warning** (`AMD GPU device(s) is/are in a low-power
  state`) on an idle node is normal and does not affect the ≤10 % drain gate.

## 8. Where the results land

```
$RESULT_DIR/
  sglang_command.txt         exact server argv
  server.log                 SGLANG_* env dump + server stdout/stderr
  benchmark_command.txt      exact aiperf argv
  benchmark.log              replay stdout
  gpu_metrics.csv            per-GPU power/util samples (ENABLE_AGENTX_POWER=1 default)
  aiperf_artifacts/          profile_export.json / .jsonl, server_metrics_export.json
$AGENTIC_OUTPUT_DIR/$RESULT_FILENAME.json     aggregate consumed by CI
```

Disable power collection with `ENABLE_AGENTX_POWER=0` if `amd-smi` sampling is
in the way.

## 9. Debugging a live run

Repo ships `.agents/skills/debug-agentx-runs/SKILL.md` for the cluster/Slurm
case. Single-node equivalent — one channel per log, filtered:

```bash
rg -n -i 'Phase |warmup|profiling|in_flight=|kv_usage=|prefix_cache_hit=|ERROR|Traceback|OOM|RCCL|timeout' \
   "$RESULT_DIR"/{server,benchmark}.log | tail -20 | cut -c1-200
```

Phase order to expect: GPU drain gate → deps install → dataset download →
weight load → `ready to roll` → AIPerf *Configure Profiling* (dataset
reconstruct + mmap, 4–14 min; timeout raised to 1800 s) → warmup → profiling.

## 14. Fast debug loop — `agentx_debug.sh`

A full launcher run is ~90 min at conc 64, and almost none of it is the thing
you are usually debugging: ~25 min weight load, ~40 min warmup (10 requests per
lane x CONC lanes), then 3600 s of profiling. For diagnosing a metric or a
crash you need none of that.

**First, though: check the artifacts you already have.** Every completed run
persists the full Prometheus scrape (`aiperf_artifacts/server_metrics_export.json`)
and the aggregate JSON. Most "why is this number strange" questions are
answerable at zero GPU cost — the §15 cache-tier finding below was.

```bash
agentx_debug.sh serve  /workspace/results/armB-tp8-c64      # once, ~25 min
agentx_debug.sh probe  /workspace/results/armB-tp8-c64 /tmp/p1 300 1   # ~5 min
agentx_debug.sh probe  /workspace/results/armB-tp8-c64 /tmp/p2 300 1   # ~5 min
agentx_debug.sh status
agentx_debug.sh stop
```

`serve` replays the argv from that run's `sglang_command.txt` plus the `SGLANG_*`
block persisted at the top of its `server.log`, so the server is identical to
the reference run without re-deriving anything. `probe` takes that run's
`benchmark_command.txt` and rewrites only `--benchmark-duration`,
`--warmup-requests-per-lane` and the artifact dir, so every other flag still
exercises the same code path.

Iteration drops from ~90 min to ~5 while the server stays resident.

**Probe numbers are for trends and bugs only.** `duration < 900` forces
`--unsafe-override` and stamps `submission_valid=false`; a 1-per-lane warmup
never reaches the steady state the published arms measure. Never quote a probe
against the leaderboard.

### `serve` replayed only SGLANG_* — the AITER_BF16_FP8_MOE_BOUND trap (FIXED 2026-08-29)

`cmd_serve` restored the `SGLANG_*` block from the reference `server.log` and
nothing else, but the launcher also exports **non-SGLANG_** vars that never land
in that dump (`PYTHONNOUSERSITE`, `AITER_BF16_FP8_MOE_BOUND=0`, and
`GPU_MAX_HW_QUEUES=5` in the DP branch). So "the server is identical to the
reference run" was false, and the one that matters is load-bearing:

```python
# aiter/fused_moe.py:722
bf16_fp8_bound = int(os.environ.get("AITER_BF16_FP8_MOE_BOUND", "256"))
...
elif activation == Swiglu or gate_mode == INTERLEAVE:
    if get_gfx() != "gfx950" or M < bf16_fp8_bound:
        q_dtype_a = dtypes.bf16     # no a4w4/a8w4 CK kernel exists for this
```

Unset, the default 256 sends every MoE call with `M < 256` down the
**bf16-activation** path. The generated dispatch for that module
(`module_moe_ck2stages_b16_fp4x2_..._per_1x32_mulWeightStage2`) requires
`dtype_checker<FP4X2>(x_dtype)`, so bf16 falls through to
`TORCH_CHECK(false, "Unsupported kernel config for moe heuristic dispatch")` and
**all 8 ranks die during decode cuda-graph capture at bs=6** (M = 6 draft-tokens
x 8 DP = 192 < 256). bs >= 8 captures fine, which is why it looks like a
small-batch kernel bug rather than a missing env var.

**This cost ~3 h on 2026-08-29 and was misattributed to shared-experts fusion**
through four wrong hypotheses (385/7 unsupported; `block_m`; missing tuned-CSV
coverage; fusion forcing bf16 activation). All four were wrong. What settled it
was running the **fusion-OFF baseline through the same tool**: it crashed
identically, which exonerated fusion in one run. The probe that produced the
evidence printed the dispatch key from `ck_moe_stage1`:
`tok=192 expert=384/385 topk=6/7 block_m=64 qtype=per_1x32 x_dtype=bfloat16` --
identical in both arms except expert/topk.

**Lesson for any future `serve`-based debugging: a crash under `agentx_debug.sh`
is not evidence about the change under test until the unchanged baseline has
been run through the same tool.** Fixed in `cmd_serve` (backup
`/tmp/agentx_debug.pre_envfix.bak`); it now exports the three vars and prints
`restored non-SGLANG launcher env (...)` so the replay is visible.

### The stale-router trap (worse than the stale server)

§11 notes the launcher leaves the SGLang server running. In **DP-attention
mode it also leaves `sglang::router`**, which binds `PORT` (8888) while the
engine sits on `PORT+1`. A router that outlives its backend still answers
`/health` — so the next run's `wait_for_server_ready` passes instantly, AIPerf
sends warmup traffic to a dead backend, and the run dies with
`Terminal warmup failure` while the real server is still loading weights.

VRAM and `pgrep sglang.launch_server` both miss it: the router holds no GPU
memory and does not match that pattern. Check the port:

```bash
ss -lntp | grep -E ':(8888|8889)\b'
```

`agentx_debug.sh status` / `stop` cover server, router, VRAM and ports together;
`serve` refuses to start unless all four are clear.

## 15. `gpu_cache_hit_rate` is device-tier only

With HiCache on, `server_metrics.cache.gpu_cache_hit_rate` counts **only the
device tier**. Hits served from the host DRAM tier land in
`cpu_cache_hit_rate`, and only `overall_cache_hit_rate` is comparable across
configurations.

Measured at conc 64 on identical traces (ISL mean within 1.3%):

| Run | device tokens | host tokens | gpu_hit | cpu_hit | overall_hit |
|---|---|---|---|---|---|
| TP8 + HiCache | 338,866,944 | 128,572,160 | 0.6888 | 0.2613 | **0.95008** |
| DP8 + TBO | 479,664,640 | — | 0.9549 | — | **0.95492** |

The TP8 arm looks 27 points worse on `gpu_cache_hit_rate` and is in fact within
0.5% on overall hit rate — 26% of its hits simply came from host DRAM. Reading
the device-only field as "the cache hit rate" makes a healthy HiCache
configuration look broken. Use `overall_cache_hit_rate`, and read
`cached_tokens_by_source` to see which tiers actually served.

## 17. sglang#35619 blocks TBO for every DP-attention config

With PR #35619 applied, any `--enable-dp-attention --enable-two-batch-overlap`
run dies at the first prefill:

```
Exception: 3 errors happen:
Field mega_moe_global_num_tokens_cpu has value, but is not yet supported
Field mega_moe_global_max_tokens has value, but is not yet supported
Field mega_moe_sync_tokens has value, but is not yet supported
```

`scheduler_components/dp_attn.py` `_update_gather_batch` sets
`batch.mega_moe_global_num_tokens` unconditionally on the DP gather path — it
checks neither `moe_a2a_backend` nor `SGLANG_AMD_USE_FLYDSL_MEGA_MOE` — and
`forward_batch_info.py` expands it into three `ForwardBatch` fields that the TBO
ubatch splitter's field whitelist rejects. **MegaMoE does not have to be enabled
for this to fire.**

Reverting the PR restores TBO, which is how the DP8+TBO row above was measured.
A fix needs either a MegaMoE gate on the assignment, or the three fields added
to the TBO whitelist with split semantics defined.

Reported upstream alongside the `SGLANG_AITER_MEGA_RANK_SYNC` IndexError
(§10 of `dsv4/megamoe/PR35619_UPSTREAM_REPRO.md`).
