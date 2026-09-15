# AgentX ITL / throughput attribution method

Platform-neutral. Works on any node that runs an InferenceX agentic arm —
MI355X (`/mnt/home/wunhuang/...`) or B200 (`/workspace/...`), same NFS export,
different mount point.

Tools in this directory:

| tool | input | answers |
|---|---|---|
| `show_result.py` | one or more `agg_*.json` | the headline metrics, and A-vs-B as percentages |
| `decode_stats.py` | one or more `server.log` | where an ITL difference actually comes from |
| `trace_summary.py` | `*.trace.json.gz` | per-step-type role breakdown for one rank |
| `trace_ranks.py` | a trace **directory** | cross-rank compute vs barrier, and who the straggler is |
| `trace_common.py` | — | shared parsing + the role classifier (not run directly) |

```bash
python3 analysis/show_result.py  <baseline.json> <candidate.json>
python3 analysis/decode_stats.py <baseline server.log> <candidate server.log>
python3 analysis/trace_ranks.py  <trace dir>            # start here for traces
python3 analysis/trace_summary.py <trace dir>/*TP-0-*.gz
```

## Why these tools compare across platforms at all

Kernel names do not survive the crossing: `deep_gemm::smN_fp8_fp4_mega_moe_impl`
on CUDA and aiter's `asm_moe` / `ck_moe` on ROCm are the same *role*. So every
kernel is bucketed by role — `comm`, `moe`, `attn`, `gemm`, `quant`,
`norm_rope`, `sample`, `copy`, `other` — and the role patterns in
`trace_common.ROLES` cover both families. Anything unmatched lands in `other`
**and is printed**, so a missing pattern shows up instead of quietly distorting
the comparison. Add patterns there rather than special-casing a call site.

Compare `compute` (attn + gemm + quant + norm_rope + sample) per token, and read
`barrier` (moe + comm) as waiting whenever it anti-correlates with token count.
`trace_ranks.py` makes that call automatically and says so in its output.

## The one thing to internalise: ITL is not a kernel number

From the scheduler's own `Decode batch` lines,

```
ITL       = step_time / accept_len
step_time = running_req_per_rank * accept_len / gen_throughput_per_rank
```

So a platform or config can lose on ITL in four distinct ways:

1. **bigger batch per step** — more work per step; worse ITL is queueing, and the
   right fix is admission control, not kernels;
2. **lost cuda-graph replay** — DSPARK/MTP runs several small draft and verify
   forwards per step; outside graph replay the launch overhead alone is tens of
   ms;
3. **a fixed per-step surcharge** — most commonly prefill stealing the scheduler
   slot (see `prefill_decode_interval` below);
4. **genuinely slower kernels at matched batch**.

These need completely different fixes, and `decode_stats.py` separates 1-3 from
4 without a profiler. **Do not open a trace before running it.** A trace tells
you where time is spent inside a step; it cannot tell you that your step is
slower because the scheduler gave it twice the batch.

## Step 1 — compare the logs, not the summaries

`decode_stats.py` reports, over the steady-state middle half of the run:
`running-req/rank`, KV pool usage, `accept len`, `gen tput/rank`, implied
`step_ms`, implied ITL, and the **fraction of steps in cuda-graph replay**.

Match these before believing any ITL comparison. Concretely, if the two sides
differ in `running-req/rank`, you are not comparing kernels — you are comparing
schedulers.

## Step 2 — the `step_time(batch)` curve is what attributes

`decode_stats.py` also prints `step_ms` bucketed by `running-req/rank`. Two
curves, not two points:

- **constant offset** between the curves → a fixed per-step cost (interruption,
  launch overhead, sync). Look at scheduling and graph capture.
- **different slope** → per-request marginal work. Look at attention/KV.
- **both flat and low-sloped** → the step is dominated by a batch-independent
  cost (MoE weight traffic, fixed overheads), and it is *not* context-bound.

Measured example (B200, DSv4-Pro-0813, c128, dp8+ep8+megamoe): step time was
68 ms at batch=1 and only 85 ms at batch=16. A nearly flat curve like that
means KV reads are not the dominant cost at these batch sizes, however large the
context is — which redirects kernel work toward weight traffic and per-step
overhead, and means a "memory-bound" claim needs the achieved bandwidth number
on *weights*, not on KV.

## Known confound: `prefill_decode_interval` differs per launcher

`server_args.py`: "the number of decode rounds to run after a prefill batch
before scheduling the next prefill."

| launcher | value |
|---|---|
| `dsv4_fp4_mi355x_sglang_mtp.sh` | `${PREFILL_DECODE_INTERVAL:-10}` → **10** |
| `dsv4_fp4_b200_sglang_mtp.sh` | hardcoded **24** (20 at `CONC=160`) |

**Every cross-platform ITL number taken before 2026-09-15 carries this
difference.** MI355X interrupted its decode stream 2.4× more often than B200.

Measured A/B on B200 c128, single variable, everything else identical:

| metric | pdi=24 | pdi=10 | delta |
|---|---|---|---|
| ITL p90 | 20.7 ms | 32.1 ms | +55.2 % |
| TTFT p50 | 4.59 s | 2.66 s | −42.0 % |
| total tok/s/GPU | 46,170 | 40,400 | −12.5 % |

and from the curves, a near-constant **+26-28 ms per step** independent of batch
— i.e. the knob is a pure ITL ↔ TTFT trade, delivered as a fixed surcharge.

The amortised-prefill model fits: surcharge = `P/N` for a prefill batch costing
`P` admitted every `N` decode rounds, so `P × (1/10 − 1/24) ≈ 27 ms` gives
`P ≈ 460 ms`, the right order for one ~49k-token chunked prefill. It predicts
that raising the interval keeps improving ITL (pdi=48 → ~10 ms surcharge), at
further TTFT cost.

**Consequence:** the config-matched MI355X-vs-B200 ITL gap at c128 is
50.6 vs 32.1 ms = **1.58×**, not 50.6 vs 20.7 = 2.5×. About 38 % of the apparent
gap was a launcher config difference. Always match this knob before quoting a
gap.

## Step 3 — only then, a trace

Use SGLang's profile endpoint on the **server** port. With DP attention the
router sits on `$PORT` and the server on `$PORT+1`; `/start_profile` exists only
on the server.

Two `ProfileReq` fields make a host-side sleep window unnecessary:

- `num_steps` — the server stops profiling itself after N scheduler steps, which
  bounds trace size deterministically instead of hoping a wall-clock window is
  the right length;
- `profile_by_stage` — keeps prefill and decode apart, without which a
  100k-token agentic prefill swamps the decode steps you care about.

```bash
curl -fsS -X POST "http://127.0.0.1:$((PORT+1))/start_profile" \
  -H "Content-Type: application/json" \
  -d '{"activities":["CPU","GPU"],"num_steps":8,"profile_by_stage":true,"record_shapes":true,"with_stack":false}'
```

`SGLANG_TORCH_PROFILER_DIR` must be set **before the server starts**.
Leave `with_stack` off for the first pass: eight DP ranks with stacks produce
enormous traces and measurable overhead. Trigger on log state (wait for the
profiling phase's `done=` lines), never on a fixed sleep — warmup length varies
by node and by how warm the page cache is.

## Reading a trace without fooling yourself

**GPU step annotations nest — de-nest before counting anything.** One ~30 ms
`step[TARGET_VERIFY]` also emits hundreds of ~0.4 ms sub-slices under the same
name. Treating them all as steps turned 62 real steps into 2077 pseudo-steps of
mean 2.89 ms, and made kernel→step attribution arbitrary (a kernel lands in
whichever overlapping slice is tested first). `trace_common.load()` keeps only
outermost annotations. Cross-check the count against the CPU-side
`user_annotation` events, which are already per-step.

**Summed kernel time is not a busy fraction.** Kernels overlap across streams,
so `sum(kernel dur) / wall` can exceed 100 % — measured 25.7 ms of kernel inside
a 17 ms step (161 %). Report the sum as a sum. If you need real utilisation, it
has to come from achieved bytes or FLOPs against peak, not from a duration
ratio.

**The filename is not evidence.** `profile_by_stage` labelled eight files
`-DECODE` and every one of them contained a single `EXTEND` step. Always list the
`step[...]` annotations first; `trace_summary.py` prints them and attributes
kernels per step, because one extend (~450 ms of GPU time) buries ten decode
steps (~65 ms each) in any blind aggregate.

**In a DP-attention deployment, one rank's trace cannot attribute anything.**
Steps are group-synchronous, so every rank's step time equals the slowest rank's
work, and the wait is absorbed *inside* the fused megamoe all-to-all kernel.
Measured on B200 c128, one extend step per rank:

| rank own tokens | step ms | `mega_moe` ms | `mega_moe` per call |
|---|---|---|---|
| 541 | 442 | 391 | 3.20 ms |
| 1762 | 441 | 333 | 2.73 ms |
| 2584 | 443 | 292 | 2.39 ms |
| 5877 | 445 | 158 | 1.29 ms |
| 6144 | 446 | 163 | 1.33 ms |

`mega_moe` time runs *opposite* to the work the rank actually has, while
`sparse_attn` and `gemm_1d1d` track it. So a top-kernel list from a lightly
loaded rank says "MoE is 90 % of the step", and that 90 % is a barrier. Compare
ranks against each other, and take real compute from the heaviest rank.

Apply `perf-bottleneck-attribution`. The two failure modes that matter here:

- **Ranking kernels by time and naming the top one.** A stall on a matmul is
  usually that matmul waiting for operands. Falsify by making the suspected unit
  cheaper (lower precision, fewer FLOPs, smaller tile) and checking whether step
  time moves. If it does not, the bound was upstream.
- **Comparing datasheet peaks.** MI355X and B200 are both HBM3e-class, so a
  1.58× gap at comparable peak is an *achieved-utilisation* claim about software,
  not a hardware ceiling. The number that settles it is achieved bytes/s against
  peak, computed from `record_shapes` in the trace.

## Gotchas that cost time on the B200 side

- **The launcher orphans the server.** After results are written,
  `launch_server` and its schedulers keep running and keep ~178 GB/GPU
  allocated. `kill` on the parent is not enough; kill the per-GPU compute-app
  PIDs. Free the GPUs before the next arm or it OOMs.
- **A dead server keeps answering 200.** `/health` stays up when the schedulers
  die and the warmup barrier then waits forever with `errors=0`. Judge liveness
  by whether `returned=` / `done=` is *moving*, and prefer `/health_generate`.
  Crashes go to `server.log`, not launcher stdout.
- **Stall detectors break on the completion path, silently.** Both times ours
  broke, it heartbeat forever and the run looked unfinished. Test each candidate
  result directory separately (`compgen -G` per directory; `ls A B` returns
  non-zero when either is missing), and track both the warmup counter
  (`returned=`) and the profiling counter (`done=`/`ok=`) or every phase change
  reads as a stall.
- **`hw` / `image` / `recipe_fingerprint` are empty on local runs.** They come
  from `RUNNER_TYPE` / `IMAGE` / `RECIPE_FINGERPRINT`, which only the CI matrix
  sets. Set them yourself or the json cannot be told apart from any other run.
- **The existing `../watch_arm.sh` (MI355X) has a latent phase-detection bug**:
  it matches `Phase [a-z]+ (started|complete)`, but the runner actually prints
  `Phase warmup progress | ...`. Stall detection still works (it tracks the
  progress counters), but `PHASE:` events never fire. Not fixed here — it is the
  MI355X node's file.
