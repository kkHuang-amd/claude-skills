# Session prompt — SGLang vs ATOM ITL p90 gap on DPA, then the OOR abort

Paste everything between the markers as the first message of a new session.

--------------------------------- BEGIN ---------------------------------

Read `/workspace/claude-skills/NEW_WORKSPACE_PROMPT.txt` and adopt its rules for
this whole session (token discipline, capped tool output, batched calls, docs as
durable state). Then read these two, in one batched message, and nothing else to
start with:

- `/workspace/claude-skills/agentx/FP4_INDEXER_REPORT.md` — the finished report
  from the previous session. Section 4 has every number this node has measured;
  section 5 is the OOR abort, which is task 2 below.
- `/workspace/claude-skills/agentx/AGENTX_20260901.md` — the node's live source of
  truth. Read the `CONTINUE HERE` block only.

Do NOT reconstruct prior chat history. Do not read
`/workspace/claude-skills/agentx/SKILL.md` — it describes a destroyed node.

## Task 1 (priority): why is SGLang's ITL p90 worse than ATOM's on DPA?

ATOM's recipe and table: https://github.com/ROCm/ATOM/pull/2068/changes
A saved copy of that page is at
`/root/.cursor/projects/sgl-workspace/uploads/changes-0.md` (69 KB, the diff is
rendered as markdown tables; the DPA server recipe is around line 430 and the
client section around line 466).

The question to answer: **is the ITL gap caused by a DPA+TBO scheduling
difference, or is SGLang's decode execution genuinely slower?**

### The gap

ATOM `per_chip` and `P90 intvty` use
`(ΣISL + ΣOSL) / duration / num_gpus` and `1 / p90(ITL)`.

| conc | ATOM tok/s/chip | SGLang | ATOM ITL p90 | SGLang | ATOM TTFT | SGLang |
|---|---|---|---|---|---|---|
| 64 | 21,888 | 20,730 | 34.9 ms | 46.8 ms | 10.5 s | 4.62 s |
| 128 | 30,709 | 27,895 | 61.5 ms | 78.4 ms | 10.9 s | 8.50 s |
| 256 | 44,722 | — | 97.6 ms | — | 13.4 s | — |

Cache hit is comparable throughout (ATOM 93.4–94.5 %, SGLang 94.2–95.5 %), so
both sides are getting their prefill from cache to the same degree.

### What is already verified — do not re-litigate these

1. **The metrics are directly comparable.** ATOM's `per_chip` formula was
   checked numerically against our raw values: `8033 × (99,954 + 871.6) /
   3629.4 / 8 = 27,895`, matching `arm_report.py`'s 27,894.8. `P90 intvty` also
   matches (`1 / 0.0784 = 12.76` vs 12.8). Our `tok/s/GPU` is the same quantity.
2. **The workload is identical**, flag by flag: same `--scenario
   inferencex-agentx-mvp`, same `--public-dataset
   semianalysis_cc_traces_weka_062126`, `--num-dataset-entries 393`,
   `--random-seed 42`, `--benchmark-duration 3600`, `--trajectory-start-min/max-ratio
   0.25/0.75`, `--warmup-requests-per-lane 10`, `--trace-idle-gap-cap-seconds 300`,
   `--slice-duration 1.0`, `--use-server-token-count`. Both sides run the
   SemiAnalysis aiperf fork 0.12.0 pinned at `754356e9`, which is also our
   `utils/aiperf` submodule pin.
3. **The server configs are closely matched.** Both: fp8 KV cache
   (`--kv-cache-dtype fp8_e4m3` / `--kv_cache_dtype fp8`), 16,384 prefill tokens
   per rank (`--chunked-prefill-size 131072` over dp8 / `--attn-prefill-chunk-size
   16384`), memory 0.90 (`--mem-fraction-static` / `--gpu-memory-utilization`),
   `max-running-requests 256` = ATOM's `--max-num-seqs $((CONC*2))`, MTP with 3
   speculative steps, measured accept length ≈ 2.4 vs ATOM's declared golden 2.49.
   **So fp8-KV decode bandwidth is NOT the explanation** — that was the most
   obvious candidate and it is ruled out.

### The two findings that reframe the question

**A. ATOM's DPA numbers were measured WITHOUT TBO.** The PR says so explicitly:

> The DP rows also enable two-batch overlap (`--enable-tbo`), which the reference
> does not run either — the measured table below predates it, so those numbers
> are the DP path WITHOUT TBO.

The recipe has `--enable-tbo`; the table predates it. So the comparison above is
SGLang **with** TBO against ATOM **without** it, and TBO restructures exactly the
prefill/decode overlap that sets ITL. This is the largest confounder and it is
cheap to remove.

**B. SGLang's prefill delayer was already ON.** Both c128 arms ran with
`--enable-prefill-delayer` (verified in `sglang_command.txt`), at defaults
(`prefill_delayer_max_delay_passes=30`, plus
`prefill_delayer_token_usage_low_watermark`). So "SGLang lacks ATOM's prefill
throttling" is **false** — `python/sglang/srt/managers/prefill_delayer.py` is
upstream now and enabled. What differs is the *tuning*: ATOM runs
`ATOM_ENABLE_PREFILL_DELAYER=1` with `ATOM_PREFILL_DECODE_INTERVAL=10`.

Note the previous port work in `/workspace/claude-skills/sglang-prefill-coalescer/`
targeted `/sgl-workspace/sglang-upstream`, which **does not exist on this node**.
Read that skill for the ATOM-side mechanics, not for paths.

### Experiments, in order

1. **SGLang DPA with TBO OFF at c128** (`ENABLE_TBO=0` in a copy of
   `dptbo_c128.sh`, mem-frac 0.90). One arm. This removes finding A, the only
   structural difference left, and it also answers a question the node has wanted
   since the MoRI arms: how much of DPA+TBO's throughput is TBO.
   - If ITL p90 drops toward ATOM's 61.5 ms → TBO is the cause, and the real
     question becomes whether TBO's throughput is worth its ITL cost.
   - If ITL p90 stays ~78 ms → the scheduler is not the explanation and the next
     step is **profiling a decode step**, not another arm.
2. **Prefill-delayer tuning**, only if (1) leaves a gap: sweep
   `--prefill-delayer-max-delay-passes` (default 30) and the token-usage low
   watermark toward ATOM's interval-10 behaviour. Pass them through the
   `EXTRA_SERVER_ARGS` hook already added to
   `benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_b200align_mtp.sh`.
3. **A c256 arm** to reach ATOM's best operating point, which SGLang has never
   measured.

### Two traps specific to this comparison

- **Our c64 row is mem-frac 0.85**, ATOM's is 0.9. Only the **c128** rows are
  memory-matched. There is no SGLang DPA+TBO c64 baseline at 0.90 (the 0.90 c64
  arm we do have has the FP4 indexer on), so any c64 statement needs a fresh
  baseline first.
- **The throughput gap at c64 (−5.3 %) is inside the 5.67 % replicate spread and
  means nothing yet.** Only the c128 throughput gap (−9.2 %) is near the bar. The
  **ITL gap (+27 % to +34 %) is the robust signal** — it is several times the
  noise floor. Frame the investigation around ITL, not tok/s.

## Task 2: the OOR abort

Section 5 of the report. Roughly **half of all hour-long arms die at ~45 min**
with `HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Available Free mem : 0 MB`. Ruled
out: KV pool, torch OOM, broken workload, corrupted tree, and
`HSA_NO_SCRATCH_RECLAIM` (a 2×2 showed a crossed pattern — each setting has one
survival and one abort, so the earlier "=0 is the fix" claim was wrong and is
retracted in the report).

The one real signal is the timing: aborts at **2680 s and 2731 s** of the
measurement window, at two different concurrencies, within 2 % of each other.

**The blocking gap is instrumentation.** `gpu_metrics.csv` is 41 MB of clocks,
power and activity with **no memory-usage column**, so no arm can say whether
free VRAM decays monotonically or dies to one spike. **Every arm from now on
should carry a VRAM sampler** — `rocm-smi --showmeminfo vram` every 30 s to a
file in `$RESULT_DIR`. It costs nothing, it makes task 1's arms do double duty,
and it is what decides between lowering mem-frac, lowering `CHUNK_PER_RANK`, and
reporting a genuine leak upstream.

## Ground rules for this node

- The FP4 integration (#37353 + aiter#5126) is applied as **uncommitted
  working-tree edits** in `/sgl-workspace/sglang` and `/sgl-workspace/aiter`,
  alongside other people's local changes. A `git checkout` / `stash` / branch
  switch in either repo destroys them — this already cost one arm.
- **Never touch those repos while an arm is running.** An arm reads the tree for
  its whole 1.5 h because the FP4 adapter's aiter imports are lazy, inside
  functions. Snapshot md5s of the key files into `$RESULT_DIR` at launch and
  verify them at the end, as `fp4-dptbo-c64-reclaim0` did.
- **The launcher never kills its own server.** Before every arm kill three
  process shapes — `python3 -m sglang.launch_server`,
  `sglang::tokenizer_worker:*`, `sglang::router`. A clean `rocm-smi` is not
  evidence the node is free: the workers hold ports without holding VRAM.
- **Never `pkill -f` with a pattern that matches your own command line.**
  `pkill -f aiperf` killed the cleanup shell here. Use a bracketed-pattern PID
  loop.
- A dead server keeps answering `GET /metrics` with HTTP 200 and aiperf then
  waits forever. Verify a running arm early, not at the end.
- Full log to a file, only markers to the chat. Show the launch command to the
  user for review before starting any arm.

## What a valid result looks like

- Gates first: `errors=0`, `records_error_dropped=0`, `duration_s` in
  3500–3750, aiperf coverage ok. Report with
  `python3 /workspace/claude-skills/agentx/arm_report.py <arm> [<baseline>]`.
- ISL matched within 3 % between a pair, cache hit flat.
- Replicate spread on this node is **5.67 %** — anything smaller is noise.
- Add finished arms to `ROWS` in `summary_table.py` and regenerate. Note that
  script prints a spurious `GPU-tier` column that duplicates `cache hit`, and
  does not print mem-frac, which is what decides comparability.
- Keep `AGENTX_20260901.md`'s `CONTINUE HERE` current as you go.

Start by reading the two files above, then state your plan for experiment 1.

---------------------------------- END -----------------------------------
