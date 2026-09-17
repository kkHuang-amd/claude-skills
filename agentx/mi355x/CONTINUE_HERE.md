# MI355X node — CONTINUE HERE

Counterpart to `agentx/b200/CONTINUE_HERE.md`. The two nodes share nothing but
this git repo; see `agentx/exchange/README.md`.

## CONTINUE HERE (2026-09-17 09:1x UTC+8) — the A/B is blocked: the reference arm is NOT reproducible with the current launcher

**Status:** three launches, three failures, all the same OOM at cuda-graph
capture (`236.93 GiB` PyTorch-allocated, <1 GiB free of 288, on a different GPU
each time). Root-caused. **Nothing is running; GPUs released.**

**Root cause: the launcher has UNCOMMITTED changes made after the reference arm
ran on 2026-09-15/16, and two of them move memory.** From
`git diff` of `InferenceX/benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh`
(97 insertions, uncommitted):

```
+    export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-40G}"
+        MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC_DP_MEGAMOE:-0.65}"
```

The mori symmetric heap is charged **outside** `mem-fraction-static`. Memory
accounting at the same point in both runs:

| | reference (2026-09-15) | now |
|---|---:|---:|
| PyTorch allocated | 236.93 GiB | 236.93 GiB |
| free at target-verify capture begin | **41.84** | **~1.0** |
| implied non-PyTorch | **~9.2 GiB** | **~50.1 GiB** |

The 40.9 GiB difference is the new 40G heap. Target-verify capture needs
20.99 GiB, which fits in the reference's 41.8 and not in our ~1.

**Two further traps found on the way, both already fixed in the arm script:**

1. **The reference arm ran from `/sgl-workspace/sglang-MegaMoE/python`** (36
   occurrences in its `server.log`), but a bare `import sglang` now resolves to
   **`/sgl-workspace/sglang`** — a different checkout with **27 dirty files**
   (including `dp_attn.py`, `forward_batch_info.py`, `eplb/*`) and a **live vim
   session**. Two launches went there. The arm script now pins
   `PYTHONPATH=/sgl-workspace/sglang-MegaMoE/python`, and the accidental patch
   to the other owner's tree has been reverted. *(The microbenchmark results are
   unaffected: the two trees' `paged_decode.py` are byte-identical apart from
   the instrumentation.)*
2. `mem-fraction-static` 0.65 vs the 0.85 that **all** published MegaMoE numbers
   used — fixed with `MEM_FRACTION_STATIC_DP_MEGAMOE=0.85`.

**`cmd_diff.py` cannot catch any of this.** It compares CLI flags, and all three
confounds were environment or code. Flags matched exactly (46, one expected
difference) in every failed launch. **A flag diff is necessary and not
sufficient; the tree and the memory-affecting env have to be checked too.**

**Next — a decision is needed, it is not mine to make:**
- `MORI_SHMEM_HEAP_SIZE=16G` is the value this file's trap notes record
  ("mem-frac 0.85 plus the 16 GiB mori heap needs 261 of 288 GiB") and would
  leave ~26 GiB for a 21 GiB capture. But the reference ran with ~9 GiB
  non-PyTorch, i.e. effectively **no** mori heap — the a2a backend here is
  `megamoe`, not mori, so the heap may simply be unused.
- Setting it to 16G reproduces the *documented* configuration, not the
  *reference* one. If the reference is the comparison target, the cleanest route
  is to **re-baseline**: run `total_requests` and `total_tokens` back to back on
  today's launcher, and compare those two to each other rather than to
  2026-09-15.

---

## Earlier (2026-09-17 08:3x UTC+8) — direction A arm staged

**Status:** the `total_tokens` A/B arm is written and was launched once, then
**killed 90 s in** because `cmd_diff` caught a confound (below). Waiting on the
VRAM cliff before relaunching — 50 GB/GPU plateau, KFD entries stale, processes
all dead. Do not launch on the plateau.

**Arm:** `agentx/agentx_c128_totaltokens.sh` (also at
`/shared_nfs/kk/pr35619/`). It answers two things at once:
- Does `total_tokens` help end to end? Report ITL, TTFT and cache hit together;
  expect a TTFT regression since it fights `cache_aware` prefix reuse.
- **It is a falsification test of the straggler finding.** The kernel's cost
  follows the batch's longest sequence, and `total_tokens` equalises the total,
  so the prediction is that **MLA µs/call barely moves**. If it moves a lot, the
  straggler result is wrong.
Verify the outcome with `analysis/kv_skew.py` on the new `server.log`, not by
the flag being present.

**Instrumentation, `agentx/mla_kvlen_stats.patch`** (applied in
`/sgl-workspace/sglang-MegaMoE`, env `SGLANG_MLA_KVLEN_STATS=1`): adds
`kvlen mean/max/min` and `kvlen straggler` to each decode log line. This is the
number that **sizes a straggler-aware rewrite** — `(max − mean)/max` — and
nothing already in the trace or the log carries it. `kv_indptr` is built *inside*
the cuda graph, so the reductions are device-only (capture-safe) and the
scheduler reads the buffer from outside at the existing log interval. Recording
in all 61 layers cost 14 % per call (345.6 → 394.4 µs); emitting in **one layer
per forward** puts it at noise (344.1 vs 346.2).

**⚠ New trap, cost an aborted launch: the launcher's MegaMoE+DP branch defaults
`mem-fraction-static` to 0.65** (`dsv4_fp4_mi355x_sglang_mtp.sh:216`,
`MEM_FRACTION_STATIC_DP_MEGAMOE`) while the reference arm ran **0.85**. Left
alone it changes the KV pool, and with it cache hit and batch composition.

**This is not specific to this arm. Every MegaMoE number we have published was
taken at 0.85**, so the 0.65 default is off-matrix and any MegaMoE arm launched
without `MEM_FRACTION_STATIC_DP_MEGAMOE=0.85` is not comparable to any of them.
Export it in every MegaMoE arm script, or change the launcher default. New
tool **`analysis/cmd_diff.py`** diffs a new arm's `sglang_command.txt` against
the reference and exits non-zero on any unexpected flag — **run it ~60 s after
every launch**. With the fix the two commands differ in exactly one flag.

`--load-balance-method` is now `${LOAD_BALANCE_METHOD:-total_requests}` at
launcher line 241, so the default is unchanged for every other arm.

**Relaunch, once VRAM has cliffed to the 284 MB baseline:**
```bash
nohup bash /shared_nfs/kk/pr35619/agentx_c128_totaltokens.sh \
      > /shared_nfs/kk/pr35619/tt_arm.log 2>&1 &
sleep 90 && python3 /workspace/claude-skills/agentx/analysis/cmd_diff.py \
  /workspace/results/megamoe-eplb-c128-b200aligned \
  /workspace/results/megamoe-eplb-c128-b200aligned-totaltokens \
  --expect load-balance-method
```

---

## Earlier (2026-09-17 08:0x UTC+8) — MLA is straggler-bound; A is the wrong knob for it

**Status:** microbenchmark done on an idle GPU 0, `analysis/mla_microbench.py`.
Full block in `exchange/FINDINGS.md` (last section). Four results:

1. **No absorbed stall.** Inverting the sweep gives implied kv_len 244-708 for
   the four in-situ points, all **below** the `index_topk = 1024` cap. The rank
   spread is a kv_len spread and the counterfactual is not circular.
2. **Cost follows the longest sequence, not the total.** At fixed mean kv_len,
   raising the max 500→1000 costs **+71 %**; halving the batch's *total* KV
   costs **−4 %**. One straggler CTA holds the grid. **So `total_tokens`
   balancing equalises something this kernel barely feels — expect direction A
   to do little for MLA.**
3. **Flat in bs within a wave, then a step.** kv_len=1024: bs 14-18 all
   469-480 µs, bs 19 **851 µs**. Marginal batch is free, then catastrophic.
4. **1.2-3.1 % of both rooflines**, corroborated in situ by `umc_activity`
   19.5 %. The 4.06x gap to B200 is a design gap, not a silicon gap.

**The fix, and the shape of it:** straggler-aware split-K. Uniform split-K wins
on ragged shapes (419 → 355 µs) and loses on uniform ones (247 → 317), and
`_kv_splits_heuristic` cannot tell them apart — by construction it reads only
`(T, H, block_h)` at capture time, never kv_len. It is correctly tuned for
uniform and wrong for ragged. Per-sequence split or a persistent-CTA work queue
closes the rest of the 419→247 gap.

**Next:** design that, in
`sglang-MegaMoE/python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_decode.py`.
The CUDA-graph constraint is the hard part: kv_len is not knowable at capture
time, so the split factor cannot depend on it — a persistent-CTA work queue
sidesteps that, since the grid is then capture-time constant and the *work
assignment* is what varies.

**Repro (GPU 0, ~7 s):**
```bash
HIP_VISIBLE_DEVICES=0 python3 /workspace/claude-skills/agentx/analysis/mla_microbench.py --quick
```

---

## Earlier (2026-09-17 07:4x UTC+8) — MLA is the whole imbalance; step 0 done

**Status:** offline critical-path counterfactual finished, no GPU used. The
component table below **double-counts**: the MLA row (+7.47) and the cross-rank
idle row (+9.96) are *the same slack*. Rank order by own work is exactly rank
order by MLA time; remove MLA and the cross-rank spread collapses from 13.50 ms
to 1.96 ms. Multiplier on an MLA speed-up is **1.5x** (the step wall responds to
the critical rank's 20.18 ms, a kernel table credits the 13.29 ms mean).
Full block and the joint pricing table: `exchange/FINDINGS.md`, last section.

Re-priced, from `analysis/mla_counterfactual.py`: MLA at B200 speed **−11.26 ms**,
perfect rank balancing alone (direction A, upper bound) **−7.35**, both
**−14.69**, MLA free **−19.54**. A is sub-additive with MLA and its value falls
as MLA improves, so price A at ~7 ms, not ~10.

The barrier model is self-validating: predicted slack sits below each rank's
observed `prepare` by a constant 5.08-5.25 ms on all five ranks, equal to the
independently measured 5.19 ms floor. So
`prepare = cross-rank slack + 5.19 ms irreducible protocol`.

**Next:** two GPU arms to validate, in this order, neither started — ask first.
1. **k=2 duplication arm.** Run the real MLA kernel twice per call, discard the
   extra. Outputs bit-identical, so accept len / OSL / ISL / KV are untouched:
   a true single-variable test with every metric quotable. Predicted step wall
   **94.16 ms** (+20.14 against a naive +13.29).
2. **k=0 fake arm.** Zeros, never uninitialised memory (NaN → the sampling
   `ASSERT_TRAP`). Predicted **54.48 ms** and per-rank `prepare` collapsing to
   the 5.19 ms floor. **Only trace-derived per-step numbers are quotable from
   this arm:** DSPARK is on (accept len 3.65, rate 0.44 of 7 draft tokens) and
   OSL is EOS-driven, so garbage logits move both.

Injection point for both: `sparse_attn_v4_paged_decode`,
`sglang-MegaMoE/python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_decode.py:895`
— single Triton file, env-gated, no aiter JIT rebuild.

**Repro (no GPU, ~7 s):**
```bash
python3 /workspace/claude-skills/agentx/analysis/mla_counterfactual.py \
        /shared_nfs/kk/pr35619/trace_c128_pdi24_steady
```

Still open: **C**, the 3.85 ms of unfused copies. **B is started and is blocked
on one measurement — see below.**

### B — the desk roofline is UNDER-DETERMINED. Do not publish a number from it.

The prompt's recipe (derive KV bytes from the DSv4 config plus per-rank
`#full token`) **does not work on this model**, for a reason worth recording:

- **DSv4 decode attention is sparse.** `index_topk = 1024`, `sliding_window =
  128`, `head_dim = 512`, and the decode path runs three ragged index streams
  (SWA / CSA / HCA, `unified_kv_kernels/runtime.py:403 build_decode_streams`)
  whose per-token length is a *prefix sum of real valid entries*, not the
  context length. So `#full token` is an upper bound on what the kernel reads,
  and on this workload it is ~100x too large.
- KV is **fp8_e4m3** with `page-size 256`, plus 1x64 block scales
  (`NUM_GROUPS = D/64 = 8` per token), so a KV token costs 512 + 32 = 544 B.
- Grid is `(N_tokens, ceil(H/BLOCK_H))`, one CTA per (token, head-tile), and
  each CTA re-reads that token's whole KV. With `H=128`, `BLOCK_H=16` that is
  **8 re-reads**, absorbed by L2 or not — which the desk calculation cannot
  decide either.

Bracketing it gives **2.3 % (topk-capped) to 32 % (whole working set) of the
8 TB/s peak**. That spread spans "nowhere near the wall" and "half way to it",
so it answers nothing.

**And the per-call times falsify every simple work model.** Capture-window KV
(server.log 02:55-03:01, matching the 03:00 trace):

| rank | trace bs | `#full token` | tok/req | kernel | us/call |
|---:|---:|---:|---:|---|---:|
| 0 | 17-18 | 892,928 | 52,525 | fused | 165-166 |
| 0 | 19-20 | 892,928 | 52,525 | fused | 223-232 |
| 1 | 14 | 1,978,240 | 152,172 | fused | **330.7** |
| 6 | 16 | 2,398,848 | 171,346 | fused | 266.1 |
| 3 | 9 | 1,834,368 | 141,105 | split | 117.5 |
| 7 | 10 | 2,395,776 | 171,127 | split | 163.5 |

Rank 6 has **more** context *and* **more** bs than rank 1 and is **20 %
faster**. So the cost is not `bs`, not `#full token`, and not tok/req. Within
rank 0 it *is* superlinear in bs (bs 17→20 costs +40 %), which smells like a
tiling or occupancy step rather than a data volume.

#### B, part 2: at a FIXED kv_len the kernel is nowhere near any wall — and that makes the rank spread unexplainable by work

Taking `kv_len = topk = 1024` (the sparse cap, so this is the honest upper
bound on a decode query's KV), per call, against MI355X peaks of 8 TB/s and
~2.5 PFLOP/s bf16, with `Nq = bs x 7` draft tokens:

| rank | bs | us/call | KV MB | achieved BW | of peak | x8 re-read | achieved | of peak |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 18 | 166.3 | 70.2 | 534 GB/s | 6.7 % | 43.6 % | 216 TF/s | **8.6 %** |
| 6 | 16 | 266.1 | 62.4 | 297 GB/s | 3.7 % | 24.2 % | 120 TF/s | 4.8 % |
| 1 | 14 | 330.7 | 54.6 | 209 GB/s | 2.6 % | 17.1 % | 85 TF/s | **3.4 %** |

**Two independent measurements confirm it is not bandwidth.** From the run's own
`gpu_metrics.csv` in the capture window (epoch 1789527600 ±60 s): `umc_activity`
**19.5 %** median across all 8 GPUs, and `gfx_0_clk` flat at **2372-2393 MHz**
(0.9 % spread) with `throttle_status = 0` everywhere. So the memory controller
is ~80 % idle during decode, and clock/power skew is **ruled out** as the
explanation for the rank spread.

**What is left is a 2.5x efficiency difference between two ranks running the
identical kernel on identical hardware at the same clock** (rank 0 at 216 TF/s,
rank 1 at 85). No work model produces that. Either `kv_len` is *not* constant
across ranks, or the kernel's measured duration is absorbing something that is
not its own work.

**⚠ A flaw in the evidence we have been leaning on.** `prepare_wait.py` reports
`mla_fused r=+0.961 -> real work`, but its `compute` proxy is
`attn+gemm+quant+norm_rope+sample` and **MLA is inside `attn`** — the kernel is
correlated against a bucket containing itself. B200's `flash_fwd_splitkv_mla
r=+0.917` row has the same defect. So **neither node has clean evidence that
the MLA per-call spread is work rather than absorbed stall**, and if it is
partly stall, the counterfactual above is circular: it would "remove the
imbalance" by removing the kernel that happens to be holding it. Fixing the
proxy to exclude the kernel under test is a small change to `prepare_wait.py`
and should be done before the next publication from either node.

**Next, and it is the cheapest decisive test:** a standalone microbenchmark of
`_paged_decode_fused_kernel` with swept `(N, kv_len)`. It gives achieved
bandwidth directly instead of by derivation, and by *inversion* recovers the
`kv_len` that reproduces 330.7 µs — which is the quantity the desk route could
not pin. GPUs are idle (297 MB/GPU baseline, no KFD PIDs). Minutes, not an arm.

---

## Earlier (2026-09-17 07:0x UTC+8) — diagnosis closed, optimisation open

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
