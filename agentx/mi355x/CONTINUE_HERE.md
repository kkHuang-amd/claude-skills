# MI355X node — CONTINUE HERE

Counterpart to `agentx/b200/CONTINUE_HERE.md`. The two nodes share nothing but
this git repo; see `agentx/exchange/README.md`.

## CONTINUE HERE (2026-09-17 07:0x UTC+8) — diagnosis closed, optimisation open

**The cross-platform investigation is finished.** Both nodes are serial, both
per-role tables are elapsed and subtract cleanly, and the 41.07 ms decode-step
gap is fully attributed and agreed (FINDINGS, B200 `e9dce10`). Nothing is
running on this node; no measurement is blocked on B200.

| component | ms | share | nature |
|---|---:|---:|---|
| MoE pipeline work — 3 stages vs 1 fused | **+12.4** | 30 % | kernel work |
| Cross-rank idle, parked in `prepare` | **+9.96** | 24 % | load balancing |
| MLA decode kernel | **+7.47** | 18 % | kernel work |
| `ep_combine` exposed vs fused a2a | **+6.17** | 15 % | structure |
| `copy` — fill kernels, no B200 counterpart | **+3.85** | 9 % | kernel work |
| quant + misc | +1.7 | 4 % | |
| `gemm` | −0.45 | −1 % | equal, settled |

### Directions, ordered by payoff ÷ effort

**A. Switch `--load-balance-method` to `total_tokens`** — one launcher flag,
worth up to ~10 ms (24 %). Both nodes' logs now justify it: `running-req` is
level (MI355X 1.38x, B200 1.29x) while `#full token` is not (2.27x / 2.45x),
because `cache_aware` concentrates long conversations. `total_tokens` exists
already (`data_parallel_controller.py:92,125-130`, fed by
`LoadSnapshot.num_total_tokens`). **A/B it** — it fights prefix reuse, so expect
a TTFT cost; same family as the pdi knob. EPLB is irrelevant here (this is
attention/KV, not expert routing).

**B. The MLA decode kernel** — 7.47 ms direct, *and* it is the multiplier on A,
so it is the only item that pays twice. 117.5-330.7 µs/call against B200's
18.7-47.5. **Do not start by rewriting it: first get achieved bandwidth**, which
nobody has on either node. `record_shapes` is a dead end here (all dim-carrying
events are `aten::*` cpu_ops), so derive the KV bytes per call from the model
config plus per-rank `#full token` and compare against MI355X's HBM roofline.
That says whether 4x is closeable or whether the kernel is already at the wall.
Two variants: `_paged_decode_split_kernel` (bs 9-10) and
`_paged_decode_fused_kernel` (bs 14-20); the 4.06x figure is the split one.

**C. `copy`, 3.85 ms of unfused fills** — the most ordinary win on the list and
entirely local. Seven kernels where B200 has 0.03 ms: `_fill_padded_rows` 0.757
(183 calls), `__amd_rocclr_fillBufferAligned` 0.533 (122 = 2/layer, a hipMemset
that smells like a buffer that could be persistent), two `direct_copy`
elementwise 1.428 total, bf16→fp32 copy 0.392, `index_elementwise` 0.363,
`_swa_scatter` 0.271, `_fill_compress_tail` 0.143.

**D. MoE pipeline structure, +12.4 ms** — the largest single work item, but it
is "fuse three stages into one", i.e. real aiter/FlyDSL work, not a knob. The
cheap adjacent piece is **pipelining `prepare(n+1)` against `stage1/2(n)`**,
which attacks the 6.28 ms protocol floor rather than the 12.4.

**E. `ep_combine` exposed, +6.17 ms** — B200 pays ~0 because its a2a is fused
inside `mega_moe_impl`. Same family as D: can the combine overlap stage2 or the
next layer? Structural, not tuning.

### Two measurements that gate the above

1. **TP-only / single-DP-rank run** (`npes = 1`, nobody to wait for). If
   `prepare`'s 102.9 µs/call floor collapses it is synchronisation and D's
   pipelining is the fix; if it holds, it is real plan emission and `pcu1`
   (one CTA) is worth raising after all. Also gives the first uncontaminated
   compute number on either node.
2. **Achieved bandwidth on the MLA decode kernel** — gates B (see above).

### Ruled out, with evidence — do not revisit

- **`prepare`'s CU count / `pcu1`+`qcu28` tuning.** A `wait_i32_until_equals`
  spin does not parallelise; the 226 idle CUs are idle *because the rank is
  waiting*. Withdrawn by both nodes.
- **Stream overlap / co-scheduling.** B200 measured true full serialisation at
  **3 % end to end** (8 % on the step wall, diluted by pdi=24). MI355X has no
  room in the wait anyway. Noise against a 2.20x kernel-work gap.
- **`gemm`.** Equal once B200's starvation artefact was removed (9.81 vs 9.36).

---

## ⚠ EARLIER: this session's shells were wedged — resolved by a terminal restart

**State at handoff (2026-09-16 21:1x UTC+8):** I ran `kill -9 <pid>` on the PID
the tool reported for a backgrounded `sed`, and that PID was the Cursor **shell
bootstrap**. Every new shell now hangs — even `echo alive` did not complete in
85 s. This is the exact trap already documented further down this file
("never `pkill -P` / kill a process whose children you have not listed"), and
hitting it again means the warning needs to be read as: *do not kill any PID the
tool hands you unless you have listed its children first.*

Nothing was broken on the node — no GPU work was running. A terminal restart
fixed it and the verification then ran normally. **Kept as a warning, because
this is the second time the same trap has been hit: do not `kill` a PID the tool
hands you for a backgrounded shell without listing its children first.**

---

**Newest, and it reframes the overlap question (2026-09-16 17:2x):** answered
B200's grid-vs-CU request. MI355X is gfx950 **SPX, 256 CUs**, and
`megamoe_prepare_compact` — the largest kernel in the step, 16.244 ms,
21.6 % of it — launches **grid 30**, so it cannot occupy more than **30 of
256 CUs**. Nothing co-resides with it: total co-resident time in the step is
**0.024 ms** against B200's 15.92 ms. **This is the opposite of B200's case**
(their MoE claims 146 of 148 SMs, so they have nothing to overlap into and
measured only 5-13 % upside for themselves). Their bound does not transfer.
Grid is *not* in the ROCm trace; it came from aiter encoding the launch config
in the kernel name (`pcu1` + `qcu28` + 1) plus the local generators.

**Next:**

1. **Co-schedule real work against `megamoe_prepare_compact` and watch the step
   wall.** This is the one experiment that settles the overlap upside, and it is
   local. Careful: grid 30 is an occupancy *ceiling*, not a utilisation
   measurement — the prepare stage is a producer/consumer ticket protocol and
   part of its 266 µs/call may be irreducible. If the wall does not move, this
   line closes. Cheaper probe first: are `pcu1` / `qcu28` simply mistuned for a
   256-CU part? That is a config in
   `aiter/ops/flydsl/kernels/mega_moe/mega_moe_prepare.py`, not a rewrite.
2. **Own the MLA decode kernel.** The narrowest target on either node:
   163.5 µs/call against B200's 40.2, same call count, same layer count, 13.3 %
   of the step. Start at aiter's `_paged_decode_split_kernel` /
   `_paged_decode_reduce_kernel` and the KV layout they read. No B200 needed.
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
