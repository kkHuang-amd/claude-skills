# MI355X node — CONTINUE HERE

Counterpart to `agentx/b200/CONTINUE_HERE.md`. The two nodes share nothing but
this git repo; see `agentx/exchange/README.md`.

## CONTINUE HERE

**Status (2026-09-16 11:4x UTC+8):** MI355X's side of the c128 cross-platform
ITL comparison is captured, analysed, corrected once, and pushed. Node is clean
(0 sglang processes, 0 ports bound; VRAM was still draining at handoff — see
"plateau-then-cliff" below). Nothing is running.

**Latest commits:** `2c7b0c9` (correction, current truth) supersedes `564a590`.

**Next, in priority order:**

1. **Fix `trace_ranks.py` the same way `trace_summary.py` was fixed.** It still
   groups `TARGET_VERIFY` without splitting the DSPARK draft step from the
   full-model step, so every cross-rank table it has ever printed mixes a 4 ms
   step with a 74 ms one. `trace_common.verify_classes(steps, owned)` already
   exists and returns `{step_index: "" | " draft" | " full"}`; wire it into the
   grouping key exactly as `trace_summary.py:52` does.
2. **Barrier-free comparison — doable here, no B200 needed.** Filter verify
   steps to those with no concurrent `EXTEND` anywhere in the 8-rank group and
   report that step wall. Today's 74.0 ms includes waiting for prefilling ranks
   (barrier is 41-60 % of it), so it is an upper bound on decode cost. This is
   the single most valuable local analysis left.
3. **Blocked on B200:** they must re-derive their numbers at steady state *and*
   with the class split before any ratio is quoted. Their 15.9 ms is retracted
   as mid-ramp and is very likely class-mixed too.

**Pass criteria for (2):** a step-wall number for full-model verify steps that
are provably not waiting on a prefill, on at least 3 ranks, with the rank spread
reported. If it lands near 74 ms the barrier is not prefill-driven; if it drops
a lot, the cross-platform gap shrinks accordingly.

**Repro — re-run the analysis on the existing traces (no GPU needed):**
```bash
cd /workspace/claude-skills/agentx
python3 analysis/trace_summary.py /shared_nfs/kk/pr35619/trace_c128_pdi24_steady/*TP-7-*.gz
python3 analysis/trace_ranks.py   /shared_nfs/kk/pr35619/trace_c128_pdi24_steady
python3 analysis/decode_stats.py  /workspace/results/megamoe-eplb-c128-b200aligned/server.log
```

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
