# DP load imbalance in AgentX c128 decode — report

*(mi355x, 2026-09-17. Consolidates work from 2026-09-16 across both nodes.
Cites B200's published numbers; B200's own blocks are in `FINDINGS.md`.)*

**Scope:** why one DP rank finishes its decode step long before another, what
that costs, and what to do about it. Configuration throughout: DeepSeek-V4-Pro
FP4, SGLang MegaMoE + DSPARK/MTP, dp8 + ep8, AgentX c128, `pdi=24`,
steady state (~167k KV tokens/request), MI355X 8x gfx950 SPX.

---

## 1. Summary

The DP attention ranks carry **2.27x different KV volumes** while the load
balancer keeps their **request counts level to 1.38x**. Decode attention cost
follows KV tokens, so the ranks do materially different amounts of work per
step; the group step is synchronous, so every rank pays the slowest one. On
MI355X the resulting idle is **9.96 ms/step — 24 % of the 41.07 ms
cross-platform decode-step gap**, and it is parked almost entirely inside a
single kernel, `megamoe_prepare_compact`.

The cause is not ROCm-specific and not a bug. It is
`--load-balance-method total_requests` levelling the wrong quantity, while the
router's `cache_aware` policy deliberately concentrates long conversations on
whichever rank already holds their prefix. **B200 has the same skew, slightly
worse (2.45x), and pays ~2.85 ms for it** because its attention kernel is
~4x cheaper per call.

---

## 2. How it was found

### 2.1 First claim — and its withdrawal

The per-rank `compute` spread was visible immediately in `trace_ranks.py`:
24.94 to 38.91 ms/step, **1.56x**. It was published as DP imbalance.

**B200 correctly rejected it.** Each rank sits at its own `bs`, so rank and
batch size move together and neither node's table separated them. A 1.56x
spread across ranks that are also at different batch sizes is not evidence of
imbalance — it is evidence of nothing. The claim was withdrawn
(`FINDINGS.md` retraction ledger, row 4).

This mattered: for a day the working assumption became "B200's ranks are level,
MI355X's spread is a `bs` artefact".

### 2.2 The re-discovery came from the opposite direction

Nobody went looking for imbalance again. It surfaced while investigating
`megamoe_prepare_compact`, which B200 had made target #1 on the grounds that it
runs 16.24 ms/step on a grid of only 30 of 256 CUs and therefore looked
under-parallelised.

Reading the generator
(`aiter/ops/flydsl/kernels/mega_moe/mega_moe_prepare.py`) showed it is not a
compute kernel at all but an inter-rank dispatch protocol:

| line | what |
|---|---|
| `:43,57,62` | parameterised by `npes` (rank count); `assert dispatch_blocks % npes == 0` |
| `:111` | `comm_ops.atomic_add_agent` — CTAs draw a ticket |
| **`:188`** | **`comm_ops.wait_i32_until_equals(epoch_gate, gate_epoch)` — an explicit spin-wait** |
| `:177-197` | `s_waitcnt(0)`, `store/load_i32_system`, `fence_system_acquire` — **system scope**, i.e. cross-device |
| `:201-235` | `emit_dispatch_plan` / `emit_dispatch_group` with `fz_npes`, `fz_rank` |

A kernel that spends time in a cross-rank gate is a place where imbalance would
be *absorbed*. That turned "is there imbalance?" from an unanswerable question
about `compute` into a testable one about a specific kernel.

---

## 3. Experiments

### E1 — Wait-vs-work by anti-correlation (`analysis/prepare_wait.py`)

**Hypothesis.** A barrier's duration anti-correlates with the rank's own work:
a busy rank arrives late and waits less. Real work tracks own work.

Per-call µs against each rank's own `compute`, steady capture, `TARGET_VERIFY
full`, all decoding ranks:

| rank | bs | compute (ms) | **prepare** | stage1 | stage2 | ep_combine | mla_split | mla_fused |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 9 | 24.94 | **325.0** | 221.5 | 88.4 | 106.6 | 117.5 | — |
| 7 | 10 | 28.36 | **266.3** | 226.2 | 89.2 | 101.1 | 163.5 | — |
| 0 | 17 | 28.76 | **251.2** | 238.6 | 88.9 | 82.2 | — | 165.3 |
| 0 | 18 | 28.78 | **252.1** | 240.4 | 89.7 | 85.4 | — | 166.3 |
| 0 | 19 | 34.45 | **217.5** | 245.1 | 90.2 | 84.6 | — | 223.1 |
| 0 | 20 | 35.06 | **216.3** | 249.5 | 91.4 | 84.1 | — | 231.7 |
| 6 | 16 | 35.25 | **156.1** | 237.5 | 92.1 | 87.0 | — | 266.1 |
| 1 | 14 | 38.91 | **102.9** | 238.9 | 92.5 | 83.8 | — | 330.7 |

| kernel | r | spread µs/call | verdict |
|---|---:|---|---|
| **`megamoe_prepare_compact`** | **−0.942** | 102.9-325.0 (3.16x) | **WAIT** |
| `megamoe_stage1_compact` | +0.672 | 221.5-249.5 | real work |
| `megamoe_stage2_compact` | +0.943 | 88.4-92.5 | real work |
| MLA decode (fused variant) | +0.961 | 165.3-330.7 | real work |
| `ep_combine_intranode` | −0.665 | 82.2-106.6 | **inconclusive** |

**Negative controls matter here.** Three kernels came out strongly *positive*,
so the method is discriminating rather than labelling everything a wait. One
(`ep_combine`) came out ambiguous and is reported as such rather than rounded
into the story.

### E2 — Within-rank control, which is what 2.1 was missing

`bs` never repeats across ranks on this node, in either capture, so the
cross-rank correlation above still carries the rank/`bs` confound that killed
the first claim. **Rank 0 alone spans bs 17→20**, which fixes the rank and
varies the batch:

| rank 0 | bs 17 | bs 18 | bs 19 | bs 20 | direction |
|---|---:|---:|---:|---:|---|
| compute | 28.76 | 28.78 | 34.45 | 35.06 | ↑ |
| **prepare** | **251.2** | **252.1** | **217.5** | **216.3** | **↓** |
| stage1 | 238.6 | 240.4 | 245.1 | 249.5 | ↑ |
| stage2 | 88.9 | 89.7 | 90.2 | 91.4 | ↑ |
| MLA decode | 165.3 | 166.3 | 223.1 | 231.7 | ↑ |

With rank held fixed, `prepare` is the **only** kernel that gets shorter as its
own rank gets busier. The confound is removed and the verdict survives.

### E3 — The invariant: `compute + prepare` is constant

Over every cell whose step wall is ~74.0 ms:

| rank | bs | compute | prepare (ms/step) | **sum** |
|---:|---:|---:|---:|---:|
| 3 | 9 | 24.94 | 19.83 | **44.77** |
| 7 | 10 | 28.36 | 16.24 | **44.60** |
| 0 | 17 | 28.76 | 15.32 | **44.08** |
| 0 | 18 | 28.78 | 15.38 | **44.16** |
| 6 | 16 | 35.25 | 9.52 | **44.77** |
| 1 | 14 | 38.91 | 6.28 | **45.19** |

`compute` spans **1.56x**; the sum holds within **2.5 %**. `prepare` is the
single synchronisation point of the step — `stage1`, `stage2` and `ep_combine`
are near-equal across ranks (221-250, 88-93, 82-107 µs) once it has levelled
everyone. This is a stronger statement than E1: it says the wait is not merely
present but is absorbing *precisely* the compute imbalance.

### E4 — Independent confirmation from the scheduler log (`analysis/kv_skew.py`)

E1-E3 all read the same trace. This one reads the scheduler's own
`Decode batch` lines instead — no trace, no GPU, no kernel inference.
Per-DP-rank medians over the steady-state half of the run:

| rank | running-req | #full token | tok/req |
|---:|---:|---:|---:|
| 0 | 9 | 1,036,928 | 110,789 |
| 1 | 8 | 1,536,256 | 190,572 |
| 2 | 11 | 2,153,984 | 187,977 |
| 3 | 9 | 1,478,400 | 168,725 |
| 4 | 10 | 1,558,400 | 159,602 |
| 5 | 11 | 1,628,160 | 156,935 |
| 6 | 11 | 2,356,864 | 213,393 |
| 7 | 11 | 2,318,080 | 207,663 |

| quantity | spread | max/min |
|---|---|---:|
| `running-req` — what `total_requests` balances | 8 - 11 | **1.38x** |
| `#full token` — what attention costs | 1.04M - 2.36M | **2.27x** |
| tok/req — request length per rank | 110,789 - 213,393 | **1.93x** |

**The balancer is working, on the wrong quantity.** Request count is level;
per-request length varies 1.93x; KV volume therefore ends up 2.27x skewed.
Rank 0 holds 9 short requests of ~111k; rank 6 holds 11 requests of ~213k.

Mechanism confirmed rather than assumed: the reference run's `router.log`
carries `cache_aware`, the policy that routes a conversation to whichever rank
already holds its prefix — so long conversations concentrate.

### E5 — Cross-platform replication

B200 ran both tools unchanged on its own serial arm:

| quantity | B200 | MI355X |
|---|---:|---:|
| `running-req` | 7-9, **1.29x** | 8-11, 1.38x |
| `#full token` | 752k-1,842k, **2.45x** | 1.04M-2.36M, 2.27x |
| tok/req | 125.6k-265.6k, **2.11x** | 110.8k-213.4k, 1.93x |
| the wait kernel | `mega_moe_impl` **r = −0.717** | `prepare` **r = −0.942** |
| per-rank MLA decode spread | 24.3-48.0 µs/call, 1.98x | 117.5-330.7 µs/call |

B200 also verified from its `sglang_command.txt` — the artefact, not script
intent — that it runs the same `--load-balance-method total_requests`.

**Two things follow.** The skew is not a ROCm phenomenon; it is the balance
method, identically on both platforms. And B200's earlier evidence that its
ranks were level (`compute` spread 0.0-0.3 ms at fixed `bs`) **was withdrawn**:
it came from the pace-pinned multi-stream capture, where `compute` read
31.16-31.60 regardless of `bs` — a saturated measurement, not a level load.
In the serial arm B200's per-rank `compute` is 16.75-19.79 ms.

### E6 — An artefact caught in passing

The first run of `prepare_wait.py` reported MLA decode as a WAIT (r = −0.625).
It was wrong: aiter selects **two** decode variants by batch size —
`_paged_decode_split_kernel` (bs 9-10) and `_paged_decode_fused_kernel`
(bs 14-20) — and the pattern matched only the first, reading 0.0 on three
ranks. Tracked separately it is the clearest *work* signal in the step
(r = +0.961). Recorded because the same trap applies to any kernel whose
implementation is selected by shape.

---

## 4. What it costs

The straggler waits for nobody, so **its** `prepare` time is the protocol's own
cost and everything above it is idle:

| | µs/call | ms/step | interpretation |
|---|---:|---:|---|
| straggler (rank 1) | 102.9 | **6.28** | protocol floor: 61 cross-rank gate crossings |
| rank 7 | 266.3 | 16.24 | floor + **9.96 idle** |
| lightest rank (rank 3) | 325.0 | 19.83 | floor + 13.55 idle |

So of the 41.07 ms cross-platform decode-step gap, **9.96 ms (24 %) is
cross-rank idle**, not slower kernels. The same accounting on B200 gives a
13.08 ms floor and only **2.85 ms** of idle.

**Why B200 pays ~3.5x less for a slightly worse skew.** Cost ≈ skew x per-unit
attention cost, and the second factor differs by ~4x: B200's MLA decode is
18.7-47.5 µs/call against MI355X's 117.5-330.7. Across each node's own rank
spread the attention kernel varies by **1.76 ms on B200** and **13.0 ms on
MI355X**.

**The consequence for prioritisation:** the imbalance penalty is *downstream* of
the attention kernel's per-call cost. Speed that kernel up by k and the
imbalance cost falls by roughly k as well — so the MLA decode kernel is the only
work item that pays twice, and the balance A/B should be re-measured after it,
not before.

---

## 5. What was ruled out

| ruled out | evidence |
|---|---|
| **Raising `prepare`'s CU count** (`pcu1`/`qcu28`, grid 30 of 256) — B200's original target #1 | A `wait_i32_until_equals` spin does not parallelise. The 226 idle CUs are idle *because the rank is waiting*. Both nodes withdrew this. |
| **Co-scheduling other work against `prepare`** | On the seven waiting ranks the capacity is already free and they still cannot finish before the straggler; the straggler has only 6.28 ms to hide. |
| **EPLB / expert rebalancing** | This is attention/KV imbalance. EPLB rebalances expert routing and cannot move KV. |
| **Stream overlap generally** | B200 measured true full serialisation at **3 % end to end**. MI355X has no overlap to lose (`sum/busy` 1.000x). |

---

## 6. Next actions

**N1 — A/B `--load-balance-method total_tokens`.** The metric already exists:
`LoadBalanceMethod.TOTAL_TOKENS` (`data_parallel_controller.py:92`) dispatches
to `min(total_tokens)` with `total_requests` as tie-breaker (`:125-130`), fed by
`LoadSnapshot.num_total_tokens` (`load_snapshot.py:201`). The launcher hardcodes
`total_requests` at
`InferenceX/benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh:241`.

- Baseline: `/workspace/results/megamoe-eplb-c128-b200aligned/`.
- **Pass criteria:** `#full token` skew measurably below 2.27x on the new
  `server.log` via `kv_skew.py`. **Verify the outcome, not the flag** — this
  project has already lost a day to a flag that silently did nothing.
- **Expect a TTFT regression.** This fights `cache_aware` prefix reuse, so it is
  an ITL↔TTFT trade of the same family as the `pdi` knob. Report ITL p90,
  TTFT p50 and cache-hit rate together; a throughput win bought with a large
  TTFT loss is a config choice, not an improvement.
- Upper bound on the prize: straggler `compute` 38.91 → ~31.2 mean is ~7.7 ms,
  so wall 74.0 → ~66 ms, order 10 %. Treat as an upper bound — it assumes
  perfect balance is reachable and that relocating KV is free.

**N2 — Re-measure after the MLA decode kernel changes.** Per §4 the two are
coupled; the value of N1 shrinks as that kernel gets faster.

**N3 — Check whether the straggler identity rotates.** The skew *magnitude* is
sustained (E4 medians over ~390 lines per rank across the steady half), but the
1.3 s trace window and the whole-run log do not name the same rank as heaviest.
If the identity rotates slowly, every rank pays the mean and the estimate in N1
holds; if it is pinned to one rank, that rank is also a thermal/power outlier
and the fix may be more urgent.

**N4 — If N1 underdelivers, look at admission rather than the metric.** A
dispatch-time heuristic cannot fix a skew that arrives with the workload; the
next lever would be capping per-rank KV occupancy directly.

---

## 7. Conclusion

The imbalance is **real, sustained, measured from two independent sources, and
present on both platforms**. It is a configuration consequence, not a defect:
`total_requests` levels request count while decode cost follows KV tokens, and
`cache_aware` routing actively creates the divergence between the two in
exchange for prefix reuse.

On MI355X it costs **9.96 ms/step, 24 % of the cross-platform decode gap**, and
it is invisible in any per-kernel profile that does not know to look for it —
it appears as a fast MoE "compute" kernel on the busiest rank and a slow one on
the idlest. The single most useful artefact from this investigation is the test
that exposes it: **correlate a kernel's per-call time against its own rank's
compute, with the rank held fixed**. That test cost nothing, needed no new
capture, and reversed two published conclusions — one on each node.

The remaining honest uncertainty is the size of the prize, not its existence.
~7.7 ms of wall is an upper bound that assumes perfect balance is achievable at
no routing cost, and `cache_aware` guarantees there *is* a cost. The A/B in N1
is what converts that estimate into a number.
