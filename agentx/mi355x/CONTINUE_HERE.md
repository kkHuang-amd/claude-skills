# MI355X node — CONTINUE HERE

Counterpart to `agentx/b200/CONTINUE_HERE.md`. The two nodes share nothing but
this git repo; see `agentx/exchange/README.md`.

## CONTINUE HERE (2026-09-17 21:3x UTC+8) — TWO ARMS ARE RUNNING UNATTENDED. Read the summary file first, do not relaunch

**A detached chain is driving the node with no operator attached.**

```bash
cat /shared_nfs/kk/chain_summary.md     # results land here as each arm finishes
tail -20 /shared_nfs/kk/chain.log       # chain progress / gates
ps -eo pid,args | rg 'run_chai[n]'      # is it still alive?
```

`run_chain.sh` (PID 1311610, started 13:37 UTC) runs, in order:

1. `agentx_c128_hcasplit_rep2.sh` → `megamoe-eplb-c128-hcasplit4-rep2` (total_requests)
2. `agentx_c256_hcasplit.sh` → `megamoe-eplb-c256-hcasplit4-totalreq` (total_requests)
3. `agentx_c128_hcasplit_tt.sh` → `megamoe-eplb-c128-hcasplit4-totaltokens` (total_tokens)

All three are MegaMoE+EPLB with HCA split-K=4, DURATION 3600. Expect ~5 h from
13:37 UTC. An earlier two-arm chain was aborted 7 min in to add arm 3; its
partial summary is `chain_summary_aborted.md` and can be deleted.

**What each one is for:**

- **The c128 arm is a REPLICATE** of `megamoe-eplb-c128-hcasplit4`, which
  already measured −7.50 ms. Same config, new result dir. Its value is the
  **run-to-run spread on this exact config** — we have only ever quoted a 5.67 %
  replicate spread for throughput, never one for matched-bs step time. Compare
  the two split4 runs against each other, not just against the reference.
- **The c256 arm is the CLEAN one.** The earlier c256 result (−5.37 ms) moved
  two variables against its reference (balancer *and* split-K). This one keeps
  `total_requests`, so split-K is the only difference and the number is
  attributable. Paired with the earlier run it also isolates the balancer *in
  the presence of split-K* — the interaction test.
- **Arm 3 fills the last cell of the c128 2x2.** Before today's chain that grid
  was: no-split/total_requests 121.76 (reference), no-split/total_tokens 122.32
  (+0.78, null), split4/total_requests 114.12 (−7.50), and
  split4/total_tokens never run. Arm 3 is that missing cell.

**When it finishes,** read `chain_summary.md` (it already contains
`decode_stats.py` output per arm), then do the matched-bs weighted comparison
against `megamoe-eplb-c128-b200aligned` / `megamoe-eplb-c256-b200aligned` the
same way as the table below.

### Three traps the chain script encodes — keep them if you rewrite it

1. **The agentx launcher never exits.** It leaves the server running after the
   benchmark ends, so completion is detected from the **log marker**
   (`Validated aiperf request error rate`), not from process exit.
2. **`pgrep '^sglang::'` misses the parent.** `python3 -m sglang.launch_server`
   gets reparented to init and respawns tokenizer workers, holding the port for
   hours. Kill it FIRST — and match it as `launch_serve[r]` so the pattern does
   not match your own command line, which self-killed a shell today.
3. **VRAM sits on a multi-GB plateau for 20-35 min** after the processes die.
   Gate on the 0.28 GB baseline confirmed twice a minute apart; launching on the
   plateau OOMs at cuda-graph capture.

### Still open when the chain finishes

- **GSM8K is NOT done.** `gsm8k_ab.sh` failed: hand-rolling the server from
  `sglang_command.txt` is not enough, because MegaMoE is turned on by launcher
  **env**, not by `--moe-a2a-backend megamoe` alone. The missing set includes
  `SGLANG_AMD_USE_FLYDSL_MEGA_MOE=1`, `SGLANG_AMD_FLYDSL_MEGA_QUANT=a8w4`,
  `SGLANG_USE_AITER=1`, `SGLANG_MOE_PADDING=1`; without them the MoE falls into
  the Triton path and dies on
  `fused_moe_triton_kernels.py:863 assert triton.cdiv(...) == B_scale.shape[-2]`.
  Copy the env block out of a launch log. Use `--max-new-tokens 8192` (DSv4
  reasoning CoT truncates at 2048 and scores 0) and do **not** set
  `SGLANG_SIMULATE_ACC_LEN`, which fakes MTP acceptance.
- **The kernel trace** still has no usable capture; see the trace section below.
- **PR [#39968](https://github.com/sgl-project/sglang/pull/39968)** is filled in
  but is a **draft**, and every `pr-gate` check fails on the step literally named
  `Block draft PR`. Marking it ready for review is what unblocks CI.

---

## EVERY ARM RUN ON 2026-09-17, ONE TABLE

All at mem-fraction 0.85, chunk/rank 8192, MegaMoE+EPLB EP8, DP8, MTP, tp8.
`Δstep` is the **n-weighted matched-bs** log-implied step delta against the
same-concurrency b200aligned reference — that is the claim. The aggregate
columns are context, and at these sizes the throughput differences sit inside
the 5.67 % replicate spread.

| mode | conc | step p50 | **Δstep** | tok/s/chip | P90 intvty | ITL p90 | TTFT avg | TTFT p50 | cache hit | GPU-tier | GPU pool | ISL mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref (b200aligned) | 128 | 121.76 | — | 36,995 | 29.5 | 33.9 ms | 11.07 s | 4.01 s | 96.00 % | 95.61 % | 91 % | 108,371 |
| total_tokens | 128 | 122.32 | **+0.78** | 39,029 | 29.2 | 34.3 ms | 9.61 s | 4.03 s | 96.10 % | 95.68 % | 100 % | 109,612 |
| fake-kernel **[INVALID]** | 128 | 103.59 | **−17.07** | 43,291 | 35.0 | 28.6 ms | 8.03 s | 3.44 s | 96.16 % | 95.73 % | 91 % | 110,409 |
| **HCA split-K=4** | 128 | 114.12 | **−7.50** | 40,346 | 31.4 | 31.8 ms | 9.31 s | 3.75 s | 96.12 % | 95.70 % | 84 % | 110,700 |
| ref (b200aligned) | 256 | 156.19 | — | 57,516 | 22.8 | 43.9 ms | 26.22 s | 15.69 s | 96.41 % | 95.52 % | 78 % | 118,262 |
| **split-K=4 + total_tokens** | 256 | 150.51 | **−5.37** | 58,508 | 23.2 | 43.1 ms | 26.30 s | 15.34 s | 96.39 % | 95.49 % | 69 % | 118,195 |

Reading notes, in order of how easy they are to get wrong:

- **The fake-kernel row is marked INVALID for every column except `Δstep`.**
  It clamps kv_len to 128, so the model emits garbage; OSL, queueing, TTFT,
  throughput and cache hit are all meaningless. It exists only as the
  **ceiling**: −17.07 ms is what deleting ~95 % of MLA buys, and it is the
  denominator the split-K result should be scored against.
- **split-K captures 44 % of that ceiling at c128** (−7.50 of −17.07) for ~20
  lines of code.
- **`total_tokens` is +0.78 ms, i.e. null**, confirming the earlier c128
  finding at full weighting. Its tok/s/chip looks +5.5 % better, which is
  exactly why throughput is not the metric here.
- **c256 gains less** (−5.37 ms on a 156 ms step, −3.4 %, versus −6.2 % at
  c128) and none of it reaches end-to-end throughput (+1.7 %, inside noise).
  **Now explained and measured — see the bs sweep below.**
- The c256 arm changed **two** variables against its reference (balancer and
  split-K), so it cannot attribute between them on its own. Given the c128
  `total_tokens` null, the −5.37 is presumed mostly split-K.
- Earlier prose quoted −7.67 ms for c128 split-K from bs 8-20; the −7.50 here
  is the same computation over every common bs cell and supersedes it.

### WHY c256 GAINS LESS — measured, and it is the grid filling up, not dispersion

`/shared_nfs/kk/hca_bs_sweep.log`, HCA shape (median 1,300, cap 5,000), % vs the
heuristic's splits=1:

| bs | T | base CTAs | dispersion | split 2 | split 4 | split 8 |
|---|---|---|---|---|---|---|
| 14 | 98 | 196 | 0.699 | −35.1 | −50.3 | **−54.2** |
| 18 | 126 | 252 | 0.707 | −25.6 | −41.8 | **−44.5** |
| 21 | 147 | 294 | 0.701 | −24.3 | −34.1 | **−36.8** |
| 24 | 168 | 336 | 0.706 | −13.8 | **−24.4** | −22.7 |
| 28 | 196 | 392 | 0.713 | −16.8 | **−24.3** | −21.9 |

**Dispersion is flat at ~0.70 in every row, so this is not a change in the
straggler — it is the base grid filling the device.** The target is
1.5 × 256 = 384 CTAs; the benefit decays as `T × 2` approaches it and flattens
once past it (bs≈28). Split-K helps by giving idle CUs something to do, and
past saturation there are none.

This predicts the c256 result quantitatively. c256 runs at bs p50 21 against
c128's 12-14, i.e. a per-call win of 34.1 % instead of ~50.3 %. Scaling the
c128 step delta by that ratio gives −7.50 × 34.1/50.3 = **−5.08 ms predicted
against −5.37 measured** — within 6 %, from a microbenchmark to an end-to-end
arm. (The step composition also differs between the two, so treat the agreement
as strong support rather than proof.)

**It also re-confirms splits=4 as the right static choice:** at bs≥24, 8 is
WORSE than 4 (−22.7 vs −24.4, and −21.9 vs −24.3). 4 degrades gracefully across
the whole batch range; 8 only wins at small batch.

---

## CONTINUE HERE (2026-09-17 16:3x UTC+8) — layer-aware split-K WORKS: −7.50 ms/step for a ~20-line change. Ship it, then trace it

**Node state:** arm complete, server and the orphaned launcher both killed,
zero `sglang::`, ports clear, VRAM draining from 32 GB/GPU.

### THE RESULT — predicted −8 ms, measured −7.67 ms

`megamoe-eplb-c128-hcasplit4` vs `megamoe-eplb-c128-b200aligned`, log-implied
step p50 at matched `bs`. The fake-kernel column is the ceiling (MLA ~deleted):

| bs | ref ms | split4 ms | Δ | ceiling Δ | captured |
|---|---|---|---|---|---|
| 10 | 116.45 | 111.55 | −4.90 | −13.72 | 36 % |
| 12 | 119.90 | 112.96 | −6.94 | −15.15 | 46 % |
| 14 | 123.83 | 117.27 | −6.56 | −18.19 | 36 % |
| 16 | 130.40 | 116.82 | −13.58 | −20.03 | 68 % |
| 18 | 134.65 | 121.83 | −12.82 | −22.87 | 56 % |

**n-weighted mean over bs 8-20: −7.67 ms/step** against a pre-registered −8.00,
with the falsification threshold at −2. The synthetic microbench transfers.

Aggregate: implied step p50 121.76 → 114.12 ms, ITL p50 32.28 → 30.24 ms,
`accept len` 3.78 → 3.77, cuda-graph replay 100 % both, 46/46 flags identical,
`0/10,488` errors. Output is CORRECT here (unlike the fake arm), and gen
tput/rank moved 389.36 → 393.73 tok/s — **inside the 5.67 % replicate spread,
so do not quote it**; the step-time result is the claim.

Per-cell Δ is noisy (bs=19 only −2.20 with n=41, bs=16 −13.58 with n=173),
which is why the weighted mean is the headline and single cells are not.

### THE TRACE ATTEMPT — partial, and at the WRONG operating point. Do not quote it

`/shared_nfs/kk/pr35619/trace_hcasplit4`, triggered 17:04 by
`trace_trigger.sh`. **Two independent problems, both fixable, neither fatal to
the arm result above.**

1. **Only 4 of 8 ranks flushed** (TP-1..4). At `num_steps=40` a rank died
   mid-capture and the rest cascaded through NCCL heartbeat / TCPStore reset —
   the exact risk `MI355X_CAPTURE_PROMPT.md` records for 40 steps, except here
   it landed before all ranks wrote.
2. **The window caught bs=5-7, while the reference trace is bs=9-20.** A
   120 s settle after warmup is not steady state: the profiling phase had only
   just begun and the batch was still ramping. Nothing at bs=5-7 can be
   compared against the reference at matched bs, and matched bs is the whole
   method.

What the partial capture still says, qualitatively:

- **`prepare` is STILL a wait, and if anything more so: r = −0.969** (reference
  −0.942). Its spread narrowed to 98.3-183.2 µs from the reference's
  102.9-325.0, which is the direction a smaller straggler predicts — but bs
  differs, so treat it as a hint, not a measurement.
- **`ep_combine` is also a wait** (r = −0.959), as before.
- **`mla_split` = 61 calls/step and `mla_fused` = 0.** Not a bug and not
  evidence the override leaked to CSA: at bs=5-7 the occupancy heuristic
  already picks splits=4 on its own (T=35, 70 base CTAs against a 384 target),
  so every stream takes the split path at that size. It does confirm the split
  path runs and is captured cleanly under cuda graph.

**Re-capture recipe (the two fixes):** settle to真 steady state — wait for
tok/req to reach ~150k rather than a fixed 120 s, i.e. 10-15 min into the
profiling phase — and drop to `num_steps=8-16` so all 8 ranks flush before any
instability. `trace_trigger.sh` takes `SETTLE` as an env var and `num_steps` as
`$3`.

### NEXT — in this order

1. **Re-capture the trace** with the two fixes above, then `prepare_wait.py`
   against the reference at MATCHED bs. The question is unchanged: how much of
   the −7.67 ms came from MLA per-call time versus the `prepare` wait
   collapsing, and whether `prepare` stays anti-correlated once the straggler
   is smaller. If it does, the imbalance has a second source and that is the
   next line of work.
2. **Then decide 4 vs 8 on real shapes.** The microbench said 4 is the robust
   pick and it was right about the magnitude, but the arm never tested 8 in
   situ. One arm with `SGLANG_MLA_HCA_KV_SPLITS=8` answers it, and the
   partial-buffer cost (205 MB vs 103 MB at bs=14) is the thing to watch.
3. **Upstream it.** The change is ~20 lines across three files and is
   env-gated; it is the first shippable win of this line.

### The change, for the record

`_kv_splits_for_stream(compress_ratio)` in `paged_decode.py` → threaded through
`runtime.decode(kv_splits=...)` → set at the one call site that knows the
stream (`deepseek_v4_backend_hip_radix.py`). HCA (ratio 128) gets splits=4;
SWA and CSA keep the occupancy heuristic. `SGLANG_MLA_HCA_KV_SPLITS=0` restores
the old behaviour with no code edit.

**Node state during the run (kept for reuse):** launched 15:11 UTC+8, PID
1203690, launch log `/shared_nfs/kk/hcasplit4_c128.log`, results
`/workspace/results/megamoe-eplb-c128-hcasplit4/`, `DURATION=3600`.

**The change:** `_kv_splits_for_stream(compress_ratio)` in `paged_decode.py`,
threaded through `runtime.decode(kv_splits=...)` from the one call site that
knows the stream (`deepseek_v4_backend_hip_radix.py`). HCA (ratio 128) gets
splits=4; SWA and CSA keep the occupancy heuristic. Env-tunable, 0 restores the
old behaviour, so the A/B needs no code edit.

Pre-flight passed (`analysis/layer_split_validate.py`, GPU0, ~8 s,
`/shared_nfs/kk/layer_split_validate.log`): stream gating `[None, None, 4]`,
the override reaching the kernel (heuristic would have picked 1), **numerics
against the fused path relL2 2.46e-03**, and CUDAGraph capture + replay on the
split path — which is a different code path (partial buffers + reduce kernel)
from the fused one production has been capturing.

**Prediction, written before the run:** HCA is ~80 % of MLA time, rank 1's MLA
is 20.17 ms/step, splits=4 takes ~50 % off the HCA part ⇒ **bs=14 should go
123.83 → ~116 ms (−8 ms)**, about 44 % of the fake-kernel arm's −18.19 ms
ceiling. **Under −2 ms means the synthetic microbench does not transfer to the
real per-layer shapes** — then investigate why, do not tune the split count.

Unlike the fake-kernel arm the output is CORRECT here, so throughput, TTFT and
cache hit are all readable; still lead with matched-bs step time for
comparability.

### Also fixed in this session (rides along with this arm)

`metrics_reporter.py`'s `kvlen straggler` now reports per-layer
`(max−mean)/max` averaged over layers, instead of dividing a layer-averaged
mean by the across-layer max. That was recorded debt #1. Inert while
`SGLANG_MLA_KVLEN_STATS=0`.

### Cleanup trap that cost a launch gate today — `pgrep '^sglang::'` IS NOT ENOUGH

After the fake-kernel arm, VRAM returned to the 0.28 GB baseline and
`pgrep '^sglang::'` returned zero, yet port 8889 was still held. The parent
**`python3 -m sglang.launch_server`** had been orphaned to init and was
respawning tokenizer workers for two hours; its command line does not start
with `sglang::`, so every check in the existing checklist missed it. Kill the
parent FIRST:

```bash
pgrep -af 'sglang.launch_server'     # list before killing
kill -9 <launch_server pid>; sleep 5
for p in $(pgrep '^sglang::'); do kill -9 "$p"; done
```

---

## Previous block (2026-09-17 14:3x UTC+8) — the fake-kernel arm PASSED. The MLA line is confirmed end to end; next is the tail microbench

**Node state:** arm complete, server killed, VRAM draining from a 32 GB/GPU
plateau at the time of writing. Wait for the 0.28 GB cliff before any GPU work.

### THE RESULT — "MLA faster ⇒ step wall drops" is now tested under intervention, and it holds

`megamoe-eplb-c128-fakekvlen` vs reference `megamoe-eplb-c128-b200aligned`,
log-implied step p50 at matched `bs` (`analysis/decode_stats.py`):

| bs | ref ms | fake ms | Δ |
|---|---|---|---|
| 8 | 111.11 | 97.58 | −13.53 |
| 10 | 116.45 | 102.73 | −13.72 |
| 12 | 119.90 | 104.75 | −15.15 |
| **14** | **123.83** | **105.64** | **−18.19 (−14.7 %)** |
| 16 | 130.40 | 110.37 | −20.03 |
| 18 | 134.65 | 111.78 | −22.87 |

**The prediction was written before the run and it landed: bs=14 was predicted
at ~104 ms from −19.5 ms of "MLA free", and measured 105.64.** The
falsification threshold was a drop of only 3-5 ms; the measured drop is 18.19.
So the floor does **not** grow and no new serialisation appears when MLA
shrinks — the component-level model of the step composes, and the trace
inversion's −19.54 ms for "MLA free" is trustworthy as an upper bound.

Δ grows monotonically with `bs` (−13.5 at 8 to −22.9 at 18), which is what MLA
cost scaling with batch predicts and is a second, independent consistency check
on the attribution.

**Not confounded:** `accept len` 3.78 vs 3.78, cuda-graph replay 100 % of steps
in both, KV pool identical at 12,077,312, 46/46 flags identical, aiperf
`0/11,284` errors. The `bs` mix did shift (running-req p50 11 vs 12) exactly as
expected from garbage output, which is why only matched-`bs` rows are quoted.
Aggregate ITL p90 34.54 vs 39.60 ms moves the same way but is part composition,
so do not quote it as the effect size.

**What this does and does not license.** It licenses spending on MLA: the
ceiling for a perfect straggler fix is 59 % of this 18 ms, i.e. ~11 ms/step at
bs=14. It does **not** say any realisable kernel reaches that — the clamp
removed 95 % of the work, which no straggler-aware design can.

### THE TAIL MICROBENCH IS DONE TOO, AND IT OVERTURNS "split-K is the wrong tool"

`analysis/mla_tail_bench.py`, GPU0, logs `/shared_nfs/kk/mla_tail_bench.log`
and `..._csa.log`. Ragged vectors from LogNormal(median 1,000, p99 3,096); the
`--cap` flag switches between the two layer families.

**The two families are different kernels in all but name.** At bs=14:

| | CSA layers (`--cap 1152`) | HCA layers (`--cap 5000`) |
|---|---|---|
| dispersion (max−mean)/max | 0.19 | 0.77 |
| ragged, heuristic splits | 504.1 µs (splits=1) | 1884.2 µs (splits=1) |
| flat at the mean | 441.5 µs | 539.3 µs |
| straggler-fix ceiling | **12.4 %** | **71.4 %** |
| split-K=8 vs heuristic | **+5.1 % (loses)** | **−59.4 % (wins)** |

**`_kv_splits_heuristic` picks splits=1 for both, which is right for CSA and
catastrophic for HCA.** Plain uniform split-K captures −59.4 % of the HCA call
— most of the 71.4 % dispersion ceiling — with no new kernel at all.

**This retires the "uniform split-K is the wrong tool" verdict, and the reason
is a one-line correction to the premise.** That verdict rested on "the
heuristic reads only capture-time scalars, so it cannot tell ragged from
uniform shapes". True of the batch — but the thing that decides which regime a
layer is in is **`compress_ratio`, a static per-layer constant from
`config.json`, which IS known at capture time.** CSA is clamped to
`index_topk`+128 = 1152 and is nearly uniform; HCA is unclamped and is where
all the dispersion lives. So the discrimination the heuristic was said to be
incapable of is available for free.

Also: at bs=10 the heuristic picks splits=2 and split-K=8 still wins −33.6 %;
at bs=18, −51.6 %. The mis-selection is not a bs=14 artefact.

**Read the ratios, not the absolute µs.** The synthetic applies one aggregate
distribution to *every* call, whereas real layers alternate between the two
regimes, so 1884 µs/call is far above the in-situ 330.7 µs/call at bs=14 (and
the probe's own stats are layer-averaged). The dispersion here, 0.77, is also
above the 0.61 measured in production.

### THE kv_len x splits SWEEP IS DONE — pick 4, not 8

`/shared_nfs/kk/hca_split_sweep.log`, bs=14, HCA shape (cap 5,000), median
swept over the run's range (HCA kv_len ~ context/128: a few hundred early,
~1,300 at the 165-170k tok/req steady state). % is vs the heuristic's splits=1:

| median | p50 | dispersion | split 2 | split 4 | split 8 | split 16 |
|---|---|---|---|---|---|---|
| 200 | 211 | 0.78 | −30.0 | **−41.8** | −35.8 | −21.8 |
| 500 | 529 | 0.78 | −35.7 | −51.4 | **−51.9** | −45.1 |
| 1300 | 1377 | 0.70 | −34.8 | −50.1 | **−54.0** | −49.7 |
| 3000 | 3179 | 0.36 | −13.5 | −23.8 | **−27.0** | −22.4 |

Three things the sweep settles:

1. **Split-K never loses anywhere on the HCA shape** — the worst cell is still
   −21.8 %. The concern that a static choice must survive the low-kv_len early
   run does not bite. The losing case is the CSA shape (+5.1 %), which
   `compress_ratio` gates out.
2. **16 is always worse than 8.** The sweep's earlier upper bound was not a
   ceiling artefact; the optimum is interior.
3. **Take 4.** It wins outright at the low end (−41.8 vs −35.8) and gives ~93 %
   of 8's benefit at the steady-state operating point (−50.1 vs −54.0), for
   **half the partial-buffer memory**: `acc_partial` is
   `T x splits x h_padded x D x 4 B` = 103 MB at splits=4 against 205 MB at
   splits=8 for bs=14, and splits=1 allocates none at all (fused path). That
   memory is the real reason the heuristic is conservative, and it is charged
   inside the graph pool at `mem_fraction_static` 0.85.

Even at dispersion 0.36 split-K wins 27 %, so the lever is not narrowly tuned
to the high-dispersion assumption.

### NEXT ACTION — make `_kv_splits_heuristic` layer-aware, then re-measure

Cheapest first, in this order:

1. **Confirm the premise in code:** find where `compress_ratio` is available at
   the `_sparse_attn_v4_paged_decode_triton` call site
   (`deepseek_v4_backend_hip_radix.py:333` is where CSA gets clamped to
   `index_topk`) and thread it, or the resulting kv_len cap, into
   `_kv_splits_heuristic` in `paged_decode.py`.
2. **Pick splits from the cap, not from `T`/`H` alone:** unclamped (HCA) ⇒ 4
   (see the sweep above); clamped (CSA) ⇒ leave at the current choice, which
   the table above shows is already right.
3. **Arm it exactly like the fake-kernel arm** — same launcher template, same
   gates, same matched-`bs` read. Budget the expectation against the fake-kernel
   arm's −18.19 ms at bs=14: that is MLA reduced ~95 %, so a split-K fix on half
   the layers should be scored as a fraction of it, not against zero.
4. Only if that disappoints: per-sequence split or a persistent-CTA work queue.

### OPEN QUESTION — does the MegaMoE `prepare` wait shrink too? The arm does NOT answer it

Asked of `megamoe_prepare_compact_m32_dcu32_pcu1_pc384_qcu28qcap256_fov_runtime_dyn_tss12488_v13`.
The fake-kernel arm produced **server-log step times only, no trace**, so this
is not measured and must not be asserted.

What IS established, from `analysis/prepare_wait.py` on the reference trace:
`prepare` is **r = −0.942 ANTI-correlated** with the rank's own compute, spread
102.9-325.0 µs across 61 calls/step — i.e. it is a **wait**, not work. Rank 1 at
bs=14 has the slowest MLA (330.7 µs) and the shortest prepare (102.9); rank 3 at
bs=9 has the fastest MLA (117.5) and the longest prepare (325.0). The straggler
does not wait; everyone else waits for it.

Two consequences, and the second is the trap:

1. **Expect the spread to collapse, not merely shrink.** The clamp removes MLA
   on every rank, so MLA stops contributing to the cross-rank difference that
   `prepare` absorbs. Whether `prepare` then goes to ~0 or simply re-forms
   around the next-largest rank-varying kernel is exactly what is unmeasured —
   and "a new straggler appears" is the same failure mode the arm was built to
   test at step level.
2. **Do not add it to the 18.19 ms.** `prepare` is a wait that already sits
   inside the step wall. Rank 1's MLA is 330.7 µs x 61 = **20.17 ms/step**, and
   the measured step drop is 18.19 ms — i.e. the drop is already ~90 % of
   "delete the straggler rank's entire MLA". Counting a prepare reduction on
   top would double-count the same time.

**Cheap way to settle it:** a short fake-kernel arm with tracing on (recipe in
`analysis/MI355X_CAPTURE_PROMPT.md`), then `prepare_wait.py` on the new trace
against the reference. It also re-tests the anti-correlation, which is the real
claim: if `prepare` stays large and stays anti-correlated after MLA is gone,
the imbalance has a second source that no MLA work can fix.

### Housekeeping before the next timing arm

`SGLANG_MLA_FAKE_KVLEN` defaults to 0, so leaving it unset disables the clamp —
but the code sits in the **dirty, uncommitted** working tree of
`/sgl-workspace/sglang-MegaMoE` next to `mla_kvlen_stats.patch`. Re-run
`cmd_diff.py` and check the env echo for any future arm regardless.

### How the arm was built and validated (for reuse)

The intervention is `SGLANG_MLA_FAKE_KVLEN=128`: `_fake_clamp_indptr` in
`paged_decode.py` rebuilds the indptr device-side as a compacted cumsum of
`clamp(len, 128)`, so every token reads at most 128 entries. It is the upper
bound of any MLA work, straggler-aware or otherwise. Offsets only ever shrink,
so `kv_indices` is never read out of bounds; the output is garbage.

Pre-flight passed before launch (`analysis/fake_kvlen_validate.py`, GPU0, ~7 s,
log `/shared_nfs/kk/fake_kvlen_validate.log`): indptr arithmetic exact on a
hand-computed case, eager cost at the production distribution
**1891.7 → 90.9 µs/call (0.05x)**, and a local `torch.cuda.CUDAGraph` capture +
replay around the real kernel succeeds with finite output. That third check is
the one that matters — launch #4 died with `hipErrorStreamCaptureUnsupported`.

**Verdict rule, fixed before the run.** Compare **log-implied step ms at matched
bs** and ITL p90 only — never throughput, TTFT, OSL or cache hit, all of which
the garbage output invalidates. Reference `megamoe-eplb-c128-b200aligned`:
**bs=14 p50 = 123.83 ms, n=247** (`analysis/decode_stats.py`). MLA free is
−19.5 ms on the pure-decode scale, so **bs=14 should land at ~104 ms**. A drop
of only 3-5 ms falsifies the whole MLA line, and then neither the microbenchmark
nor any kernel work should proceed.

**Read it with:**

```bash
cd /workspace/claude-skills/agentx
python3 analysis/decode_stats.py /workspace/results/megamoe-eplb-c128-fakekvlen/server.log
python3 arm_report.py /workspace/results/megamoe-eplb-c128-b200aligned \
                      /workspace/results/megamoe-eplb-c128-fakekvlen
```

### After it finishes, in this order

1. Read the verdict above and write it into `exchange/FINDINGS.md`.
2. Only if the step moved: the microbench at the real distribution (SECOND
   ACTION below), which is still unrun — `mla_microbench.py` sweeps kv_len
   ≤2048 while the real max is ~5,000. Needs 1 GPU, 10 min, so it must wait for
   this arm to release the node.
3. Revert the fake clamp before any timing arm that is not this one:
   `SGLANG_MLA_FAKE_KVLEN` defaults to 0, so unsetting it is enough — but the
   code lives in the dirty working tree of `/sgl-workspace/sglang-MegaMoE`
   alongside `mla_kvlen_stats.patch`, uncommitted.

---

## Previous block (2026-09-17 13:0x UTC+8) — fresh-session handoff. MLA straggler confirmed; next action is the fake-kernel arm

**Node state:** nothing running, GPUs released, VRAM draining from the probe arm
(28 GB/GPU plateau at handoff — wait for the 0.28 GB cliff before any arm).

### What is settled today, and must not be re-litigated

| line | verdict | evidence |
|---|---|---|
| **A `total_tokens`** | **closed — keep the flag, not an ITL lever** | skew 2.08x→1.69x but step time moved 0 to +2.4 % at matched bs. TTFT 11.07→9.61 s, cache unchanged, throughput +5.5 % is inside the 5.67 % replicate spread = null |
| **cross-rank balancing** | **falsified as a route** | the −7.35 ms "balanced ranks" counterfactual row is withdrawn |
| **MLA straggler** | **confirmed, within-batch** | 2,772 samples: at 150-200k tok/req the batch's p50 token reads ~1,000 entries, p99 reads ~3,100. Dispersion grows with KV (0.20→0.61) |
| **MLA roofline** | **1.2-3.1 % of both** | so the 4.06x gap to B200 is a design gap, not silicon |
| **uniform split-K** | **wrong tool** | wins ragged (419→355 µs), loses uniform (247→317); `_kv_splits_heuristic` reads only capture-time scalars so it cannot tell them apart |
| **C, copy kernels** | **attributed, low ceiling** | whole bucket 3.85 ms; largest single kernel 0.757 ms; the dead-write fix is ~0.25 ms (0.34 % of the step). Cleanup, not a main line |

### The structural fact that explains everything (found late, easy to miss)

**Two of the three index streams have different caps.** CSA (`compress_ratio 4`)
is clamped to `index_topk`=1024 (`deepseek_v4_backend_hip_radix.py:333`), so its
kv_len tops out at 1024+128. **HCA (`compress_ratio 128`) has no such clamp** —
it covers the whole context at 1/128 resolution, so its kv_len ≈ context/128 and
grows without bound (ISL p99 634,941 ÷ 128 = 4,960, matching the measured
per-layer maxima of 3,958-5,288). `config.json`'s `compress_ratios` alternates
128/4 per layer.

Consequences: `#full token` is **not** fully decoupled from per-step cost (it
drives the HCA layers); the within-batch straggler **is** the long-context
request on HCA layers; and a fix should target those layers, not all 61.

### NEXT ACTION — the fake-kernel arm. It is the decisive experiment

Everything above is component-level. The chain "MLA faster ⇒ step wall drops"
has never been tested under intervention, and the chain is what has been wrong
twice today. `floor` = 5.19 ms was measured with MLA unchanged; whether it grows
or another serialisation appears when MLA shrinks, no model can say.

**Design (single variable, capture-safe):** clamp per-token kv_len to a small
constant (128) inside `_sparse_attn_v4_paged_decode_triton`. Build the indptr
device-side (`cumsum` of a filled tensor) — **never write a host scalar into a
device tensor**, that is what aborted a launch today with
`hipErrorStreamCaptureUnsupported`. Validate under a local
`torch.cuda.CUDAGraph` capture+replay **before** launching, as the kv_len probe
now is.

**Quote only step-level numbers**: log-implied step time at matched `bs`, and
ITL p90. Not throughput, not TTFT — garbage output changes OSL and queueing.
`accept len` is **not** a confound: AgentX pins it
(`SGLANG_SIMULATE_ACC_LEN=3.77`, launcher :302-311).

**Prediction to falsify, written before the run:** MLA free takes the pure
decode step 74.02 → 54.48 ms, i.e. −19.5 ms. On the log-implied scale bs=14
should go **123.83 → ~104 ms**. A drop of only 3-5 ms kills the whole MLA line
and neither the microbenchmark nor the kernel work should proceed.

### SECOND ACTION — microbench at the real distribution (10 min, 1 GPU, no cliff needed)

Retires two debts and fills the wait for the VRAM cliff. `mla_microbench.py`
currently sweeps kv_len ≤2048 while the real max is ~5,000 — **the tail is
exactly where the win is and it has never been measured.** Three cases, ragged
vectors matching the measured distribution (p50 ~1,000, p99 ~3,100, max ~5,000):
ragged as-is; clamp-to-mean (identical total work, zero dispersion — the floor a
perfect straggler fix reaches); uniform split-K 1/2/4/8. Difference between the
first two is the **measured** achievable win per call.

### Two debts of mine, both recorded so they are not inherited silently

1. **The logged `kvlen straggler` field is the wrong statistic.** It mixes
   within-batch dispersion with across-layer dispersion (its `max` is across
   layers, its `mean` is layer-averaged). Use `(p99−mean)/p99`. Fixing it is one
   line in `metrics_reporter.py` (per-layer `(max−mean)/max`, then average) plus
   one arm.
2. **`mla_microbench.py`'s kv_len ceiling is 2048**, below the real maximum, so
   its absolute per-call figures understate the tail. The inversion's "implied
   kv_len 244-708" are 61-layer averages, not per-token lengths.

### Repro / tools (all no-GPU unless noted)

```bash
cd /workspace/claude-skills/agentx
python3 analysis/mla_counterfactual.py /shared_nfs/kk/pr35619/trace_c128_pdi24_steady
python3 analysis/copy_attrib.py        /shared_nfs/kk/pr35619/trace_c128_pdi24_steady
python3 analysis/kv_skew.py     /workspace/results/megamoe-eplb-c128-b200aligned-totaltokens/server.log
python3 analysis/decode_stats.py /workspace/results/megamoe-eplb-c128-b200aligned-totaltokens/server.log
python3 arm_report.py /workspace/results/megamoe-eplb-c128-b200aligned \
                      /workspace/results/megamoe-eplb-c128-b200aligned-totaltokens
HIP_VISIBLE_DEVICES=0 python3 analysis/mla_microbench.py --quick   # needs 1 GPU
```

### Launching any arm — the checklist that cost five failures today

1. VRAM at the **0.28 GB baseline**, confirmed twice a minute apart. Never on a
   plateau.
2. Zero `sglang::*` processes (`pgrep '^sglang::'`) — **list PIDs before
   killing**, and note a finished arm leaves its server running.
3. Ports 8888/8889 clear.
4. Use `agentx/agentx_c128_totaltokens.sh` as the template: it pins
   `PYTHONPATH=/workspace/InferenceX:/sgl-workspace/sglang-MegaMoE/python:/sgl-workspace/mori`,
   `MEM_FRACTION_STATIC_DP_MEGAMOE=0.85`, `MORI_SHMEM_HEAP_SIZE=16G`. **The
   launcher's own defaults (0.65, 40G, and a bare import resolving to
   `/sgl-workspace/sglang`) do not reproduce any published MegaMoE number.**
5. 90 s after launch run `analysis/cmd_diff.py` against the reference arm. It is
   necessary and **not sufficient** — it compares CLI flags only, and all five
   of today's failures had identical flags. Also check the tree in `server.log`
   and the environment echo in the launch log.

---

## Earlier today (2026-09-17 11:0x UTC+8) — `total_tokens` A/B: balancing is NOT an ITL lever

**Status:** arm complete and clean (3,628 s, `errors=0`, gates pass, 46 flags
with only `--load-balance-method` differing, KV pool identical at 12,077,312).
Nothing running. Full block in `exchange/FINDINGS.md`, last section.

**The intervention worked and the step did not care.** `#full token` skew
**2.08x → 1.69x** (verified with `kv_skew.py` on the new `server.log`, not from
the flag), and at matched `bs` the decode step moved **0 to +2.4 %** — nothing,
or marginally slower. The new arm even carries *more* KV at the same `bs`
(tok/req 140.8-232.6k vs 114.3-218.3k).

**So the balancing route is closed, by two independent methods agreeing.** The
microbenchmark said the kernel's cost follows the batch's **longest** sequence;
`total_tokens` equalises the **total**; the A/B confirms removing 19 % of the
skew buys zero step time. **The −7.35 ms "balanced ranks" row in this morning's
counterfactual is withdrawn.**

**End to end:** TTFT **11.07 → 9.61 s (−13.2 %)**, ITL p90 flat (+1.0 %),
throughput +5.5 % which is **inside the 5.67 % replicate spread, so a null**,
cache hit unchanged (0.956 → 0.957). **The predicted TTFT regression did not
happen** — `total_tokens` does not disturb the router's `cache_aware` prefix
reuse. Keep the flag (free, better TTFT, less skew) but **do not count it
against the MLA gap**.

**Next: MLA is the only remaining line, and it is unambiguous.** The kernel runs
at **1.2-3.1 % of both rooflines**, its cost is set by one straggler CTA, and
uniform split-K cannot fix that (it wins on ragged shapes, loses on uniform, and
the heuristic cannot tell them apart — it reads only capture-time scalars).
Two things gate the design, in this order:
1. **Size it.** `(max − mean)/max` of the per-token kv_len in production. Not in
   any artefact we have. The probe for it (`mla_kvlen_stats.patch`) is written
   but **not capture-safe** — see below. Cheapest correct route: a short
   `--disable-cuda-graph` run, where the distribution is identical and Python
   runs every step, so no device buffer is needed.
2. **Then design** per-sequence split or a persistent-CTA work queue in
   `paged_decode.py`, under the constraint that kv_len is unknowable at capture
   time.

**The kv_len probe is now capture-safe and validated locally.**
`agentx_c128_kvlenprobe.sh` + `mla_kvlen_stats.patch`. Two fixes to the bug that
killed launch #4: allocate the buffer only when
`torch.cuda.is_current_stream_capturing()` is false (sglang's warmup forwards
run eager, so it lands there), and never write a host scalar into the device
buffer. **Verified before use, not on the node**: a local `torch.cuda.CUDAGraph`
capture + replay around the real kernel succeeds and returns correct stats, and
the overhead is noise (345.6 vs 347.5 µs). It logs
`kvlen mean/p50/p99/max/min` and `kvlen straggler = (max−mean)/max` per decode
line.

**Why `(max−mean)/max` decides the next step.** The kernel's cost follows the
batch's longest sequence, so that ratio *is* the headroom a straggler-aware
kernel can win. It also discriminates two worlds: the 2.9x spread we know about
(per-rank implied kv_len 244-708) is **between** ranks, and nobody has measured
the spread **within** a batch. Within-batch dispersion ⇒ the straggler fix
works. Ranks internally uniform but at different levels ⇒ it does nothing, and
the only route left is making MLA faster outright.

Priced beforehand so the probe has something to falsify
(`mla_counterfactual.py`, same trace): equalising MLA across ranks to what the
**cheapest rank already achieves** takes the step wall **74.02 → 62.48 ms
(−11.54, −15.6 %)**; to the mean, −6.48; MLA free, −19.54. Dispersion removal
alone is 59 % of the total MLA opportunity and needs no work moved between
ranks — which is why it survives the `total_tokens` falsification.

**Read the probe against tok/req, never against the clock.** tok/req reaches
~140k by minute 15 and 147-172k by 25-40 (steady state 165-170k), so discard
early samples. `DURATION=1800`.

### C — the copy kernels are attributed. The biggest one is a ROCm-only fallback

`analysis/copy_attrib.py` on the existing trace, no GPU. 27.58 ms of `copy`-role
kernels across 8 ranks' EXTEND windows, 80 % resolved.

**`_fill_padded_rows_kernel` — 7.41 ms in EXTEND, grid 3254x256, and exactly
183 calls/step = 3 x 61 layers.** Three call sites, all MoE top-k padding
housekeeping: `topk.py:1578` (`_mask_topk_ids_padded_region`),
`topk.py:1591` (`_zero_topk_weights_padded_region`) and
`mega_moe_flydsl.py:263`. **B200 has no counterpart because the CUDA path never
reaches this kernel** — `topk.py:1573` is
`if _is_cuda and topk_ids.dtype == torch.int32 and fill_value == -1:
mask_topk_ids(...)`, and ROCm falls through to `_fill_padded_rows`. This is a
gated fast path, not a missing optimisation, which makes it the cheapest item on
the whole list to attack.

**`_swa_scatter_kernel` — 3.42 ms, grid 3254x512, 61 calls/step = one per
layer**, from `store_swa_into_unified` (`unified_kv_kernels/runtime.py:85`).
The SWA KV write.

Everything else resolves to ordinary aten ops (`aten::copy_`, `aten::cat`,
`aten::scatter_add_`, `aten::index`) and `Memcpy DtoD`.

**Two method notes worth keeping:**
- **`External id` does not attribute Triton kernels.** A kernel event carries
  `correlation`, not `External id`; the id lives on the `cuda_runtime` launch
  event, and only aten's `hipLaunchKernel` has one —
  `hipModuleLaunchKernel` (Triton) does not. The tool therefore walks
  `correlation -> launch event -> timestamp` and finds the innermost enclosing
  `cpu_op`/annotation. Triton launches sit inside no aten op, so the trace can
  only place them in the step; the call site came from `rg` on the kernel name
  plus the calls/step count as the cross-check (183 = 3x61 pins all three sites).
- **⚠ CORRECTION: grid IS in the ROCm trace.** Every kernel event carries
  `args.grid` and `args.block` (e.g. `_fill_padded_rows_kernel` grid
  `[3254,1,1]`, block `[256,1,1]`). This file previously said it was not, and
  that `megamoe_prepare_compact`'s grid 30 had to be inferred from aiter's
  kernel-name encoding. It can be read directly, and `copy_attrib.py` prints it.

---

## Earlier (2026-09-17 09:3x UTC+8) — reference environment recovered

**Status:** five launches, five failures, all root-caused, **all from launcher
and environment drift, none from the change under test**. Nothing running, GPUs
at the 0.28 GB baseline. The arm is now configured from the reference arm's own
launch log and is ready to go on approval.

**The reference environment, recovered verbatim** from
`/shared_nfs/kk/pr35619/b200aligned_c128.log` (the arm's own launch log — it
echoes the environment, which `sglang_command.txt` does not):

```
MEM_FRACTION_STATIC=0.85
MORI_SHMEM_HEAP_SIZE=17179869184        # 16 GiB, NOT the launcher's current 40G
PYTHONPATH=/workspace/InferenceX:/sgl-workspace/sglang-MegaMoE/python:/sgl-workspace/mori:
```

**Provenance of the 16 GiB, since no wrapper script sets it:** the launch log is
a `set -x` trace, and line 453 is `+ export MORI_SHMEM_HEAP_SIZE=17179869184`
immediately after `SGLANG_AMD_FLYDSL_MEGA_QUANT=a8w4` — the *launcher's own*
line, at the exact position where the working copy now reads `40G`. The
committed launcher has no such export at all, so that whole MegaMoE block has
always been uncommitted, and **the line was edited in place from 16 GiB to 40G
after the reference arm ran**. `git diff` shows the block as a pure addition and
therefore hides the edit; only the execution trace reveals it. Setting 16G is
restoring the reference configuration, not deviating from it.

**Why 40G cannot work here, arithmetically** — the heap is charged *outside*
`mem-fraction-static`:

```
236.93 (PyTorch static at 0.85) + 40 (heap) + 20.99 (target-verify capture)
  = 297.9 GiB  >  287.98 GiB      before the ~5 GiB driver context
```

Four launches OOMed on exactly this, on GPUs 1/2/7, each with <1 GiB free. The
reference log confirms it independently: at capture end it had `avail mem=20.85`
with 257.92 GiB PyTorch-allocated, leaving **9.23 GiB** non-PyTorch — a 40 GiB
heap cannot be resident in that.

**The five failures, so none is repeated:**

| # | cause | mine? |
|---|---|---|
| 1 | `mem-fraction-static` 0.65 (launcher's new MegaMoE default) vs 0.85 | no |
| 2 | wrong tree: bare `import sglang` → `/sgl-workspace/sglang`, 27 dirty files, live vim | no |
| 3 | same, plus 40G heap | no |
| 4 | **my kv_len probe is not capture-safe** — `hipErrorStreamCaptureUnsupported` | **yes** |
| 5 | 40G heap, probe off, correct tree — clean proof that 0.85+40G cannot fit | no |

Failure 4 is a real bug in `mla_kvlen_stats.patch`: writing the host-side
`d.numel()` into the device buffer is a pageable H2D copy, which aborts cuda
graph capture, and lazily allocating the buffer inside capture is a second
problem. **The probe is now default-OFF.** Fix both (drop the host scalar,
allocate from backend init) or sample the distribution from a
`--disable-cuda-graph` run instead — the distribution is workload-driven and
identical there, and Python runs every step so no device buffer is needed.

**Standing lesson, stronger than before:** `cmd_diff.py` reported "46 flags,
only the expected difference" on **every one of these failures**. A CLI-flag
diff is necessary and nowhere near sufficient. The launch log's environment
echo is the artefact that actually settles reproduction — **capture it for every
arm, and diff it too.**

**Ready to launch (needs approval — ~1 h GPU):**
```bash
nohup bash /shared_nfs/kk/pr35619/agentx_c128_totaltokens.sh \
      > /shared_nfs/kk/pr35619/tt_arm.log 2>&1 &
sleep 90 && python3 /workspace/claude-skills/agentx/analysis/cmd_diff.py \
  /workspace/results/megamoe-eplb-c128-b200aligned \
  /workspace/results/megamoe-eplb-c128-b200aligned-totaltokens \
  --expect load-balance-method
```
Only `--load-balance-method` differs from the reference. "16G + probe off" has
never actually been run: failure 4 was the probe, not memory.

---

## Earlier (2026-09-17 09:1x UTC+8) — root cause of the launch failures

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
