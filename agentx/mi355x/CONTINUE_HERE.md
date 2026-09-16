# MI355X node — CONTINUE HERE

Counterpart to `agentx/b200/CONTINUE_HERE.md`. The two nodes share nothing but
this git repo; see `agentx/exchange/README.md`.

## CONTINUE HERE

**Status (2026-09-16 14:5x UTC+8):** Merged B200's 6 new commits (`3bcc032`) and
answered all three of their open asks from the existing traces — no GPU needed,
nothing is running. The cross-platform gap now has **two named targets**:

1. **`_paged_decode_split_kernel` (MLA decode) is 4.06x B200's
   `flash_fwd_splitkv_mla_fp8`** — 9.974 vs 2.454 ms/step at 61 calls each
   (163.5 vs 40.2 µs/call). That is 93 % of the whole `attn` gap.
2. **`megamoe_prepare_compact` 16.24 ms/step has no B200 counterpart** — B200's
   nearest-named kernel is 0.242 ms. Holds 66 % of the `moe` gap.

Everything else matches within 0.78-1.94x. Also established: MI355X's `compute`
is **not** pace-pinned (only `moe` absorbs wait, where B200 has both `moe` and
`gemm` doing it), `comm` 6.17 **is** a real collective (`ep_combine_intranode_0`),
and `record_shapes` **cannot** give B200's proposed bandwidth estimator here
(all 34,486 dim-carrying events are `aten::*` cpu_ops; zero attn/MoE ops).

**Also found, and it corrects both nodes' tables:** every per-role number ever
published on either side is a **sum** of kernel durations, which double-counts
concurrency. B200's 50.86 ms sum sits inside a 30.0 ms wall, so its elapsed GPU
work is ≤30 ms and its per-role sums were being subtracted from MI355X's
*elapsed* ones. New `analysis/busy_ms.py` sweeps the intervals: **MI355X is
sum = union = wall = 1.000x with 0.01 ms idle — perfectly serial, no overlap,
no gaps** — so this node needs no correction and B200 does. In elapsed terms
the gap is the wall ratio, ≥2.47x. The two per-kernel findings survive (per-call
at matched call counts), and 4.06x is a **floor** since B200's own §2c calls
`flash_fwd_splitkv_mla_fp8` pure and contention can only have inflated it.

**Pushed:** `be97070` (§10-12 + `kernel_dump.py` + `trace_ranks.py`) and the
`busy_ms.py` / §13 commit after it.

**Latest commits on origin:** B200 `3bcc032`, then mine.

**Next:**

1. **Own the MLA decode kernel.** It is the single narrowest target on either
   node: 163.5 µs/call against B200's 40.2, same call count, same layer count.
   Start at aiter's `_paged_decode_split_kernel` / `_paged_decode_reduce_kernel`
   and the KV layout they read. This does not need B200.
2. **`megamoe_prepare_compact`** — decide whether 16.24 ms is real prepare work
   or the fused all-to-all wait. A TP-only run separates them.
3. **No kernel overlap at all here (1.000x, 0.01 ms idle)** is now the largest
   single multiplier in the comparison and is a pure MI355X question: is any
   concurrency reachable on ROCm for this step, or is the step genuinely a
   dependency chain? Nothing to wait for B200 on.
4. **TP-only / single-DP-rank capture** is now the cheapest uncontaminated
   compute number for either node, since `record_shapes` is a dead end here.
5. **Waiting on B200 for one thing only:** their `busy_ms.py` `credited` column.
   Until it lands, no per-role delta is quotable in either direction.
6. **Not blocking any more:** B200 has published steady-state class-split
   numbers, so the comparison is live. The `compute` *ratio* stays withdrawn —
   B200's side is pace-pinned, mine is not, and one usable side is not a
   comparison. Quote step wall (30.0 vs 74.0, 2.47x) or the kernel pairs above.

**Repro — re-run the analysis on the existing traces (no GPU needed):**
```bash
cd /workspace/claude-skills/agentx
python3 analysis/trace_summary.py /shared_nfs/kk/pr35619/trace_c128_pdi24_steady/*TP-7-*.gz
python3 analysis/trace_ranks.py   /shared_nfs/kk/pr35619/trace_c128_pdi24_steady
python3 analysis/kernel_dump.py   /shared_nfs/kk/pr35619/trace_c128_pdi24_steady
python3 analysis/busy_ms.py       /shared_nfs/kk/pr35619/trace_c128_pdi24_steady 10
python3 analysis/decode_stats.py  /workspace/results/megamoe-eplb-c128-b200aligned/server.log
```

**Earlier result, still standing:** full-model `TARGET_VERIFY` with no
time-overlap against any of the 8 `EXTEND` annotations — 5 ranks, 16-17 steps
each, p50 73.9-74.1 ms (max/min 1.002x), `n_hit=0`. The 74 ms is not waiting on
a prefill. Ranks 2/4/5 have no verify in this window and 0 kernels during it.

## Where things are

| what | path |
|---|---|
| steady-state traces (use these) | `/shared_nfs/kk/pr35619/trace_c128_pdi24_steady/` |
| mid-ramp traces (do not conclude from) | `/shared_nfs/kk/pr35619/trace_c128_pdi24/` |
| trace run's server.log | `/workspace/results/megamoe-eplb-c128-b200aligned-trace/server.log` |
| complete c128 run (agg metrics) | `/workspace/results/megamoe-eplb-c128-b200aligned/` |
| capture orchestrator (reusable) | `/shared_nfs/kk/pr35619/trace_c128_pdi24.sh` |
| idle-wait + launch (reusable) | `/shared_nfs/kk/pr35619/wait_and_launch_c256.sh` |
| findings, pushed | `agentx/exchange/mi355x-decode-trace.md`, `agentx/exchange/FINDINGS.md` |

Benchmark result rows and the pdi/router/load-balance alignment history live in
`dsv4/megamoe/C256_REGRESSION_HANDOFF.md`.

**Uncommitted on purpose:** `agentx/summary_table.py` carries three new `ROWS`
entries (c256 post-merge, c256 B200-aligned, c128 B200-aligned). They point at
MI355X-local result dirs. Commit them if the other node should see the registry;
they are not needed for the trace work.

## Node-specific traps, all hit at least once on 2026-09-15/16

- **VRAM reclaim is plateau-then-cliff, never linear.** After killing a server it
  sits flat at 29-53 GB/GPU with **zero** KFD holders for 10-20 min, then drops
  to the 284 MB baseline in one step. A creep-rate extrapolation once predicted
  7 hours; it cliffed within the minute. Do not launch on top of a plateau —
  mem-frac 0.85 (245 GiB) plus the 16 GiB mori heap needs 261 of 288 GiB.
  Per-GPU reset is **unsupported on this node**; do not reach for it.
- **The launcher orphans everything.** Every run so far left `launch_server`,
  `sglang::*`, tokenizer workers and aiperf alive holding ~290 GB/GPU *and* the
  dist-init port, which makes the next launch die with `port_base ... is not
  available`. Verify all three after cleanup: process count, VRAM, port listeners.
- **Intermittent EPLB rebalance deadlock, ~1 run in 3.** `returned=` frozen,
  `errors=0`, `/metrics` still 200, schedulers alive, VRAM full. Last server line
  is `Resetting ExpertDistributionRecorder...` from all 8 ranks, then every rank
  hangs in a collective and the NCCL watchdog kills it 600 s later with
  `c10::DistBackendError`. Nothing reaches launcher stdout. Judge liveness only
  by `returned=`/`done=` moving. A straight retry cleared it.
- **`pgrep -f` / `rg` match your own shell command.** Reported phantom aiperf and
  watcher processes three times. Worse: a broad `pkill -9 -P` plus loose patterns
  killed the Cursor shell bootstrap (`bash -O extglob -c snap=$(command cat <&3)`)
  and **wedged every new shell in the session** — `echo` would not complete,
  which looks exactly like a dead node but is not. Recovery is restarting the
  terminal. Match on `ps -eo comm` or bracket the first character, and never
  `pkill -P` a process whose children you have not listed.
- **A fixed settle before `/start_profile` captures mid-ramp.** Context length is
  still climbing minutes into the profiling phase. Trigger on per-request
  `#full token` / `#running-req` plateauing (>= ~130k here), not a constant.
- **`SGLANG_TORCH_PROFILER_DIR` did not appear in any scheduler's
  `/proc/*/environ`.** Inconclusive, but pass `"output_dir"` in the
  `/start_profile` body instead — `profile_utils.py:117` prefers it and it costs
  nothing in comparability.

## Config state of the launcher

`dsv4_fp4_mi355x_sglang_mtp.sh` is aligned to the B200 sibling as of 2026-09-15:
`--prefill-decode-interval` 24 (20 plus `--balance-abs-threshold 32` when
`CONC >= 160`), `--load-balance-method total_requests`, router `--policy
cache_aware`. Pre-edit backup: `/shared_nfs/kk/pr35619/mi355x_mtp.sh.bak.1228`.
The file was already dirty before that edit (+81 lines of MegaMoE support), so
**do not `git checkout` it**.

`CONC` counts AgentX *session trees*, not requests: `MAX_RUNNING_REQUESTS=2*CONC`
= 256, so with dp8 a rank can legitimately run up to 32 concurrent requests.
Observed distribution at c128 peaks at 11-12 and tails to 29 — batches above
16/rank are expected, not a bug.
