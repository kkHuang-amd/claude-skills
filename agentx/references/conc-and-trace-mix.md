# Concurrency, trace mix, and why headline tok/s cannot be compared across conc

_Split out of `SKILL.md` on 2026-08-28. Section numbers are
preserved so every `§N` cross-reference in the skill still resolves._

## 16. AgentX conc 64 — three serving paths compared (2026-08-27)

Same trace corpus, same MTP settings (EAGLE 3/1/4, `SGLANG_SIMULATE_ACC_LEN=2.49`),
same HiCache DRAM tier, 3600 s profiling each. ISL means agree within 1.3 %, so
the workload is matched.

| | throughput/chip | p90 intvty | TTFT p50 | reqs ok | overall cache hit | OSL mean |
|---|---|---|---|---|---|---|
| **TP8**, fusion on (published-arm config, conc extrapolated) | 17,079.5 | 13.76 | 1.15 s | 4,265 | 0.95008 | 919 |
| **DP8 + TBO**, fusion off | **17,443.3** | 13.53 | 3.10 s | 4,321 | 0.95492 | 956 |
| **DP8 + EP8, Aiter MegaMoE + EPLB** | 12,630.8 | 12.20 | 1.21 s | 3,115 | 0.95469 | 886 |

### What this says

**MegaMoE is 26–28 % behind both fused-MoE paths on AgentX**, matching the
conc 32 result (11,458 vs the 15,671 published TP8 arm). Not a memory artifact —
§11/§12 showed a 2.42x larger KV pool moved throughput by +0.9 %. The likely
cause is MTPR-aligned padding: MegaMoE dispatches in fixed 8192-token slots,
and the AgentX corpus has ISL p50 ≈ 72 K, p90 ≈ 216 K with high variance, so the
padding waste outweighs the fused-kernel saving. On fixed-seq-len 8 K/1 K the
same build reproduces the PR's numbers exactly (§10 of
`dsv4/megamoe/PR35619_UPSTREAM_REPRO.md`), so this is workload-specific, not an
integration defect.

**TP8 and DP8+TBO land within 2.1 %** — but with a large TTFT split (1.15 s vs
3.10 s p50). **Corrected 2026-08-27:** this paragraph used to attribute the
split to TBO's ubatch overlap. That is not the leading explanation — the TBO arm
also runs `--enable-prefill-delayer` and the MegaMoE arm does not, and delaying
prefill to align DP ranks is exactly what inflates TTFT. See §19.6. The
`dsv4/TBO_RESEARCH.md` §19 result (TBO *lowering* TTFT 10-13 % on a 70 K/300
sweep) is consistent with that reading: that sweep had no delayer.

### Confounds not yet eliminated

- **Shared-experts fusion differs**: on in the TP8 row, off in the other two.
  #32340 claims +8–11 % output throughput at low concurrency from fusion, so the
  TP8 number is probably flattered relative to DP8+TBO. Isolating it needs one
  more run: `tbo` config with fusion switched on.
- **Prefill delayer differs**: `--enable-prefill-delayer` is on in the DP8+TBO
  row and **off** in the MegaMoE row (TP8 has no DP so it is N/A). Confirmed
  from `sglang_command.txt` and the `PrefillDelayer initialized with
  max_delay_passes=30 ... queue_min_ratio=None` line in `tbo-tp8-c64/server.log`.
  This is the single largest uncontrolled variable behind the TTFT column and
  the reason the old ubatch-overlap explanation was withdrawn (§19.6).
  Upstream's DeepSeek-V4 cookbook prescribes it for any TP+DP high-concurrency
  recipe on B200/B300/GB200/GB300
  (`docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx:326`:
  `--dp 8 --enable-dp-attention --enable-prefill-delayer
  --prefill-delayer-max-delay-ms 5000`), so having it on in a DP arm is correct
  practice — it just is not controlled against MegaMoE. It is **never**
  auto-enabled: `enable_prefill_delayer` defaults `False` and `server_args.py`
  has no coupling to `dp_attention`.
- **EPLB only in the MegaMoE row.** It measured +8.4 % on fixed-seq-len, so
  removing it would widen MegaMoE's deficit, not narrow it.
- **Code base differs.** The MegaMoE row carries PR #35619; the other two do
  not, because the PR's three `mega_moe_*` ForwardBatch fields make the TBO
  ubatch splitter throw `Field ... is not yet supported` — see §17.
- **conc 64 is outside the published set** (MI355X TP8 stops at 48), so the TP8
  row is an extrapolation, not a leaderboard reproduction.

## 19. The conc 48 result — how to read it, and what it said

Framework written 2026-08-27 while the arm was still profiling; filled in the
same day when it landed. §19.3 has the numbers, §19.4 the verdict, §19.6 the
only cross-concurrency comparison that is valid.

### 19.1 Where the numbers live

All paths are inside `<RESULT_DIR>/<RESULT_FILENAME>.json`:

| What | Key path |
|---|---|
| headline throughput | `request_metrics.throughput.per_gpu.total_tput_tps` |
| — its input half | `request_metrics.throughput.per_gpu.input_tput_tps` |
| — its output half | `request_metrics.throughput.per_gpu.output_tput_tps` |
| interactivity p90 | `request_metrics.latency.intvty.p90` |
| TTFT p50 | `request_metrics.latency.ttft.p50` |
| mean ISL / OSL | `request_metrics.tokens.input.mean` / `tokens.output_actual.mean` |
| requests ok | `num_requests_successful` (and `request_accounting.records_*`) |
| cache (use this one) | `server_metrics.cache.overall_cache_hit_rate` |
| cache by tier | `server_metrics.cache.cached_tokens_by_source.{device,host}` |
| KV usage | `server_metrics.kv_cache.gpu_usage_pct` |
| effective duration | `request_metrics.throughput.duration_seconds` |

Two naming traps: `gpu_usage_pct` is a **fraction, not a percent** (the c64 run
reports `1`, meaning 100 % — the KV pool was full, not 1 % idle). And
`external_cache_hit_rate` duplicates `cpu_cache_hit_rate` under HiCache DRAM;
it is not a third tier.

### 19.2 The headline metric is a prefill-replay rate, not a compute rate

Verified exactly against the c64 run (residual 0):

```
per_gpu.total_tput_tps ≈ (num_requests_successful × tokens.input.mean)
                         / duration_seconds / num_gpus
4265 × 115,356.90 = 491,997,199 = server_metrics.tokens.prompt_total   (exact)
```

Two consequences, and they reframe the open question:

- **99.2 % of `total_tput_tps` is input tokens** (16,944.5 input vs 135.0 output
  per GPU at c64). The output side is almost invisible in the headline number.
- **Only 4.99 % of those prompt tokens were actually computed.** 491.997 M
  prompt tokens decompose into 338.867 M device-cache hits + 128.572 M
  host-cache hits + **24.558 M computed**. The metric therefore rewards *cache
  hit rate* and *large ISL* far more than it rewards GEMM throughput.
  (Corrected 2026-08-27: this bullet previously read "9.11 % / 44.8 M", which
  did not satisfy `computed = prompt_total - device - host` — the three terms
  summed to 512.3 M, above `prompt_total`. Always recompute it as that
  subtraction; there is no third cache tier, see the `external_cache_hit_rate`
  trap above.)

**The decomposition is an identity, not a causal test.** `total_tput_tps =
requests x mean_ISL / duration / num_gpus` holds by construction, so its
"residual 0" can never fail and can never establish *why* a number moved — it
only says *which term* moved. "The gap is entirely mean ISL" is an accounting
statement: ISL was the only term that moved in the gaining direction (+17.4 %),
while requests moved against it (-4.4 %). Do not upgrade that into a mechanism.

**Retracted 2026-08-27:** a causal chain was briefly asserted in discussion —
c64 saturates its KV pool -> spills 26 % of hits to the host tier -> that costs
wall clock -> lower headline. It is **not supported** and should not be revived.
Two findings kill it: (a) the concurrency-invariant metrics run the wrong way,
c64 completing *more* requests/s (1.1751 vs 1.1246) and computing *more*
tok/s/GPU (845.8 vs 627.7) than c48 — a cache-throttled server does less work,
not more; and (b) §20.3b measured the host-tier traffic at 0.32 GB/s aggregate,
i.e. free. The residency difference is real but its cost lands on the **TTFT
tail** (§19.6: TP8 c64 has the widest p90/p50 of any arm, 5.95), not on
throughput.

So a throughput gap between two AgentX runs decomposes into exactly three
factors — requests completed, mean ISL of the traces that got sampled, and
wall-clock duration — before any hardware explanation is needed. **Do that
decomposition first.** It is arithmetic on the JSON, zero GPU cost.

### 19.3 The numbers

Filled 2026-08-27 from
`/workspace/results/armB-tp8-c48/dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep1-dpafalse_disagg-false_spec-mtp_agentic_c48.json`.

| | our TP8 c64 (§16) | **our TP8 c48** | published TP8 c48 |
|---|---|---|---|
| `per_gpu.total_tput_tps` | 17,079.5 | **19,171.9** (+2.33 % vs pub) | **18,735.8** |
| — input / output halves | 16,944.4 / 135.0 | 19,036.8 / 135.1 | — |
| `intvty.p90` | 13.76 | **25.62** (+5.0 % vs pub) | **24.4** |
| `ttft.p50` | 1.15 s | **0.703 s** | — |
| `num_requests_successful` | 4,265 | **4,078** (-4.4 %) | — |
| `tokens.input.mean` | 115,356.9 | **135,423.4** (+17.4 %) | — |
| `tokens.output_actual.mean` | — | 961.3 | — |
| `overall_cache_hit_rate` | 0.95008 | **0.96706** | — |
| `kv_cache.gpu_usage_pct` | 1 (=100 %) | **0.8** (=80 %) | — |
| computed / prompt tokens | 4.99 % (24.558 M) | **3.29 %** (18.210 M) | — |
| `duration_seconds` | 3,629.5 | 3,626.2 | — |
| `avg_total_gpu_power_w` | — | 6,870.3 | — |

Note `kv_cache.gpu_usage_pct` fell to 0.8 at c48 — the KV pool was *not* full,
which is what you would expect from 48 lanes and is consistent with the retired
KV-saturation hypothesis.

### 19.4 Decision table for the open question

**RESOLVED 2026-08-27 — outcome: branch 1, with an amendment.** c48 measured
19,171.9, i.e. 2.33 % above the published 18,735.8 and comfortably inside §12's
+/-5 % reproduction band. Read the branches below as history; the live
conclusion is in §19.6 and the CONTINUE HERE block.

The amendment matters: branch 1's wording ("the throughput curve genuinely peaks
at or below conc 48, and c64 is past the peak") is **not** supported by the data
it was meant to interpret. §19.6 shows the c48-over-c64 gain is pure trace mix,
and that c64 completed more requests and computed more tokens per second. The
published arm's choice of 48 may still be deliberate, but this measurement does
not demonstrate a throughput peak, and no further AgentX run can — see §19.6.

The conc-48 client probe that lost 18 % is retired, as branch 1 anticipated.


The open question is why our c64 (17,079.5) sits below the published c48
(18,735.8). KV saturation and HiCache host-tier cost were both tested and both
failed (CONTINUE HERE). Read the new c48 number as follows:

- **c48 lands near 18,735.8 (within the ±5 % of §12's reproduction):** the
  throughput curve genuinely peaks at or below conc 48 on this recipe, and c64
  is past the peak. Open question resolved; the published arm's choice of 48 is
  deliberate, not arbitrary. Note this contradicts the earlier conc-48 *client
  probe* that lost 18 % — which is expected, because §14 says probes run with
  `duration<900` and 1-per-lane warmup never reach steady state. The full arm
  wins; retire the probe result.
- **c48 lands near our c64 (~17 k), i.e. flat and below published:** the deficit
  is a uniform offset, not concurrency-dependent — image, fleet, or recipe
  difference. Stop tuning concurrency; compare only in relative terms across our
  own arms, and treat §16 as internally valid but not leaderboard-comparable.
- **c48 lands *below* our c64:** throughput is monotone increasing in this
  range here, which directly contradicts the published curve. Suspect the arm
  definition rather than the machine — recheck §10 "which env goes with which
  arm", especially shared-experts fusion (backlog 2, the one uncontrolled
  variable) before concluding anything.
- **c48 clearly exceeds 18,735.8:** check `submission_valid` and the warmup
  count before celebrating; an under-warmed run inflates the headline by
  finishing more short requests early.

In every branch, run the §19.2 decomposition before assigning a cause. If
`num_requests_successful × tokens.input.mean` already explains the delta, the
explanation is workload sampling, not the server.

### 19.5 Validity gates — check before quoting anything

- `power_valid = 1` and `request_accounting.records_error_dropped = 0`.
  — c48: **1 / 0, pass.**
- `duration_seconds` ≈ 3600 (c64 measured 3629.5). — c48: **3626.2, pass.**
- `records_warmup_dropped` should be ~10×CONC (c64: 707 for 64 lanes).
  — c48: **531 for 48 lanes, pass.**
- `tokens.input.mean` within a few percent of 115,356.9, else the trace sample
  differs and the arms are not comparable — this is the §16 matching criterion.
  — c48: **135,423.4, +17.4 %, FAILS.** This gate is unpassable across
  concurrencies, not a defect of this run: see §19.6. It still holds as written
  for same-concurrency comparisons, which is what §16 uses it for.

There is no top-level `submission_valid` key in the aggregate JSON despite
§19.4's earlier wording — the validity signals are `power_valid`,
`request_accounting.*`, and `recipe_fingerprint`.

### 19.6 Concurrency-invariant comparison — use this across arms

The headline `total_tput_tps` is `requests x mean_ISL / duration / num_gpus`
(§19.2), and mean ISL is **not** a free parameter: it is decided by which traces
happen to finish inside the window, which is itself a function of concurrency.
Our c48 and c64 arms already share `--random-seed 42` and
`--num-dataset-entries 393`, and still differ by 17.4 % in mean ISL. **There is
no seed or flag that matches ISL across concurrencies.** Therefore the headline
metric is only comparable between runs at the *same* concurrency — which is
exactly the published-vs-ours comparison, and is why that one is valid.

For everything else, quote these instead. Both are read straight off the JSON:

| invariant | formula | c64 | c48 |
|---|---|---|---|
| requests/s | `num_requests_successful / duration_seconds` | **1.1751** | 1.1246 (-4.3 %) |
| computed tok/s/GPU | `(prompt_total - device - host) / duration / 8` | **845.8** | 627.7 (-25.8 %) |
| headline tok/s/GPU | (not comparable across conc) | 17,079.5 | 19,171.9 (+12.3 %) |

The three rows disagree in sign, which is the whole point: on the leaderboard
metric c48 looks 12 % better, while it finished fewer requests and did a quarter
less arithmetic. When comparing our own arms — TP8 vs DP8+TBO vs MegaMoE (§16),
load-balance methods (§18), fusion settings — hold concurrency fixed and report
requests/s and computed tok/s/GPU alongside the headline.

#### Say it as a pair, never as one number

"c64 recomputed 51 % more" is **wrong** and was corrected on 2026-08-27. There
are two different quantities and they differ by 17 points:

| | c64 | c48 | c64 vs c48 |
|---|---|---|---|
| recompute **ratio** (computed / prompt) | 4.99 % | 3.29 % | **+52 %** (ratio of ratios) |
| recompute **volume** (computed tokens) | 24.558 M | 18.210 M | **+35 %** |

GPU time is spent on the *volume*, so +35 % is the figure that explains wall
clock; the ratio is an efficiency statement only. Quoting either alone is
misleading because the denominators also moved. The honest one-liner, which
carries both numerator and denominator:

> **c64 replayed 11 % fewer prompt tokens (492.0 M vs 552.8 M) while computing
> 35 % more of them (24.558 M vs 18.210 M).**

#### Recompute across every arm on file

`computed = prompt_total - cached.device - cached.host` (§19.2). Cross-arm rows
are only comparable where ISL matches — the three conc-64 arms do (115.4-116.8 k,
within 1.3 %), so those three are a fair set; c48 and tp4-c10 are not.

| arm | conc | ISL | prompt | device / host | **computed** | comp tok/s/GPU | headline | **ttft p90** |
|---|---|---|---|---|---|---|---|---|
| TP8+HiCache | 64 | 115,357 | 492.0 M | 68.9 % / 26.1 % | **24.558 M (4.99 %)** | 845.8 | 17,079.5 | **6.826 s** |
| DP8+TBO | 64 | 116,248 | 502.3 M | 95.5 % / 0 % | **22.644 M (4.51 %)** | 779.9 | 17,443.3 | **7.889 s** |
| MegaMoE dp8 ep8 | 64 | 116,846 | 441.5 M | 95.5 % / 0 % | **20.001 M (4.53 %)** | 688.9 | 12,630.8 | **3.506 s** |
| TP8+HiCache | 48 | 135,423 | 552.8 M | 92.5 % / 4.3 % | **18.210 M (3.29 %)** | 627.7 | 19,171.9 | **2.283 s** |
| TP4 (§12 repro) | 10 | 134,980 | 166.9 M | 95.9 % / 0 % | **6.934 M (4.15 %)** | 239.8 | 11,628.6 | **1.440 s** |

#### The KV pool is not the same size across arms — check it before blaming concurrency

`kv_cache_pool_tokens` differs by 5.3x between the TP8 and DP-attention arms:

| arm | pool tokens | `kv_usage` | host tier used |
|---|---|---|---|
| TP8+HiCache c64 | 10,519,296 | **1.00 (full)** | 26.1 % of prompt |
| TP8+HiCache c48 | 10,598,400 | 0.80 | 4.3 % |
| DP8+TBO c64 | **56,090,624** | 0.50 | 0 % |
| MegaMoE c64 | **61,556,736** | 0.41 | 0 % |

This is the useful control for the c48-vs-c64 cache story. DP8+TBO ran the
*same* concurrency and an ISL within 0.8 % of TP8 c64, and with a 5.3x larger
pool it never touched the host tier at all (95.5 % device, `kv_usage` 0.50).
So the 26 % host-tier spill in TP8 c64 tracks **pool headroom**, not the trace
mix — which is the half of the §19.6 confound that can be settled from data
already on disk.

What is still *not* settled: within TP8, c64's live context (conc x ISL = 7.38 M)
is only 13 % above c48's (6.50 M), yet `kv_usage` goes 0.80 -> 1.00 and device
hits fall 92.5 % -> 68.9 %. A 13 % input producing a 24-point output is a
capacity cliff, but concurrency and trace mix moved together, so which one
pushed it over is unproven. The decisive run is one TP8 c48 arm with the pool
shrunk until its headroom matches c64's: if device hits collapse toward 69 %,
it is residency; if they hold, it is the trace mix.

#### TTFT: the shape of the distribution says more than p90

`latency.ttft` carries only `mean/p50/p75/p90/p95/std` — **no p99, no max**, so
the tail is only visible to p95.

| arm | p50 | p90 | p95 | **p90/p50** |
|---|---|---|---|---|
| TP8+HiCache c64 | 1.148 | 6.826 | 10.912 | **5.95** |
| DP8+TBO c64 | 3.096 | 7.889 | 13.133 | **2.55** |
| MegaMoE c64 | 1.205 | 3.506 | 5.502 | **2.91** |
| TP8+HiCache c48 | 0.703 | 2.283 | 3.135 | **3.25** |
| TP4 c10 | 0.510 | 1.440 | 2.322 | **2.83** |

Two readings fall out:

- **TP8 c64 has the widest spread (5.95)** while its p50 is the second-best.
  Most requests are fast and a subset is punished hard — the signature of the
  26 % host-tier spill in §19.6, since a host-tier hit pays a DRAM->GPU transfer
  the device-tier hits do not. c48 removes the spill and the ratio drops to 3.25.
- **DP8+TBO is shifted up uniformly, not fanned out (2.55, the narrowest).**
  Its p50 is 2.7x TP8's while its p90 is only 1.16x. That is a flat per-request
  cost applied to everybody, not a subset being delayed.

#### Is DP8+TBO's TTFT caused by the §18 rank imbalance? Probably not.

The imbalance is real (§18: 1.60x prefill work, 1.56x mean concurrent requests
under `round_robin`), but three things on file argue against it being the driver:

1. **MegaMoE is the counter-example.** It is *more* imbalanced than TBO on both
   §18 measures (3.66x prefill work, 1.96x concurrent requests) and its TTFT p50
   is **2.6x better** (1.205 s vs 3.096 s). If imbalance set TTFT, the ordering
   would reverse.
2. **Wrong distribution shape.** `round_robin` assigns per request, so a
   persistently loaded rank would delay a *subset* and widen p90/p50. TBO has
   the narrowest ratio of all five arms (2.55).
3. **Wrong magnitude.** A 1.56x load spread cannot produce a 2.7x median shift.

DP attention alone is not the cause either — MegaMoE also runs `dp_attention`
with dp8 and lands at 1.205 s, level with non-DPA TP8 (1.148 s).

**The variable that separates TBO from MegaMoE is the prefill delayer, not
two-batch overlap.** (Withdrawn 2026-08-27: this paragraph previously blamed
TBO's micro-batching. Checking the launch commands showed a more direct cause.)

```
tbo-tp8-c64      --enable-dp-attention --enable-prefill-delayer --enable-two-batch-overlap
megamoe-tp8-c64  --enable-dp-attention            <- no delayer
```

`tbo-tp8-c64/server.log` confirms it ran:
`PrefillDelayer initialized with max_delay_passes=30
token_usage_low_watermark=None queue_min_ratio=None max_delay_ms=5000`.

Holding prefill back so DP ranks enter the same forward pass is the delayer's
stated purpose, and inflating TTFT is its designed cost, not a side effect. It
also closes the loop on the imbalance question: rank imbalance does not cause
the TTFT directly — the *mitigation* for rank imbalance does. TBO buying
throughput with first-token latency is the expected trade, and TBO does post the
best req/s of the three (1.1906). Two-batch overlap may still contribute; it is
simply no longer the leading explanation, and it is not isolated.

**A live bug in the upstream recipe, found while checking this.** The cookbook's
`--prefill-delayer-max-delay-ms 5000` is **inert on its own.** In
`python/sglang/srt/managers/prefill_delayer.py`, `_max_delay_ms` is only read
inside `if self._queue_trigger_enabled and global_running_batch_max > 0:`
(~L256-273), and `_queue_trigger_enabled = self._queue_min_ratio is not None`.
Without `--prefill-delayer-queue-min-ratio` there is no wall-clock bound at all;
the only limit is the slot path's `max_delay_passes` (default 30). Our TBO run
set neither, so it delayed up to 30 forward passes with no ms cap. Read from
source, not verified by runtime experiment — worth reporting upstream after a
check.

**Caveats before treating this as settled.** TBO vs MegaMoE is not a clean
contrast: they differ in ep1-vs-ep8, MoE implementation, and shared-experts
fusion (§16, backlog 1). And MegaMoE issues fewer prefills/s (0.8583 vs 1.1906
req/s), so its prefill queue is genuinely shorter — some of its TTFT advantage
is lower offered load, not architecture. Note MegaMoE's *total* per-request
latency is the worst of the three by Little's law (64 / 0.8583 = 74.6 s vs
TBO's 53.8 s): it is fast to first token and slow thereafter, i.e. decode-bound.

**The decisive run is already backlog item 2:** rerun the TBO arm with
`--load-balance-method total_tokens`. If TTFT p50 stays near 3.1 s, imbalance is
exonerated and TBO's micro-batching owns the cost; if it falls toward 1.2 s,
this whole analysis is wrong and imbalance was the driver after all.



## 25. How short an AgentX window can be — measured, not guessed (2026-08-28)

Zero GPU cost: aiperf writes `aiperf_artifacts/profile_export_aiperf_timeslices.json`
(~2,261 one-second slices per arm, each with `request_count` and
`usage_total_tokens`). Summing `count x avg` over the first N seconds
reconstructs what the arm **would have reported** had it stopped at N.
Validated against the published aggregates at full duration: within 0.8 % on
all three arms. Reconstruct with `/tmp/curve.py` (recreate from this section).

Cumulative tok/s/GPU by cutoff:

| arm | 300 s | 600 s | 900 s | 1200 s | 1800 s | 2400 s | 3000 s | 3600 s |
|---|---|---|---|---|---|---|---|---|
| TP8 c48 | 11,208 | 16,797 | 17,955 | 18,090 | 18,662 | 18,899 | 19,134 | 19,312 |
| TP8 c64 | 13,613 | 15,151 | 17,320 | 17,325 | 18,258 | 18,649 | 17,856 | 17,219 |
| DP8+TBO c64 | 10,988 | 16,282 | 17,740 | 17,322 | 18,171 | 18,507 | 18,151 | 17,585 |

**The rule this gives us: a short window resolves >=10 % effects and nothing
smaller.**

- The **c48 > c64 ordering (~10 %) is stable from 600 s** and never flips
  again. A short arm can decide that class of question.
- The **TBO vs TP8 c64 ordering (2.1 %) is NOT stable**: TBO leads at 900 s,
  ties at 1200 s, **TP8 overtakes at 1800 s and 2400 s**, and it only settles
  back to TBO after ~3000 s. The curve's own wander exceeds the effect being
  measured. No short AgentX arm can settle a 2 % question.
- **Absolute values are ~6 % low at 900 s** (c48 reads 17,955 vs 19,172), and
  the c48-over-TBO gap reads +1.2 % instead of the true +9.9 %. Ordering can
  survive while magnitude is useless.

**Two further biases make a real short run worse than this table.** These
curves come from the *profiling* phase of arms that already completed a full
10-requests-per-lane warmup. A short run that also cuts warmup inherits:
(a) the cache is still filling *during* profiling — warmup-phase hit rate is
0.86 and only reaches 0.95/0.967 across the full window; and (b) **DP arms warm
slower** (warmup hit 0.8415 for DP8+TBO vs 0.8608 for TP8), because each rank
fills its own cache (§22.3 item 5). So cutting warmup biases *against* DP
specifically — the mechanism behind the retired conc-48 probe that "lost 18 %"
(§14).

**Therefore, for any short arm: keep `--warmup-requests-per-lane 10`, cut only
`--benchmark-duration`.** ~65 min/arm instead of ~125. And compare a short arm
only against the *reconstructed same-cutoff* value of the reference arms in the
table above — never against a published 3600 s number, which the short window
under-reads by ~6 % by construction.
