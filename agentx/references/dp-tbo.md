# DP attention vs TBO — the full investigation, ladder results, and cache-aware routing

_Split out of `SKILL.md` on 2026-08-28. Section numbers are
preserved so every `§N` cross-reference in the skill still resolves._

## 18. DP rank imbalance under `round_robin` is real (2026-08-27, zero GPU cost)

Backlog item 3, measured from the per-rank `Prefill batch` / `Decode batch`
lines already in each DP run's `server.log` — no re-run needed. Both DP runs
used the default router policy (`auto` resolves to `round_robin` on non-PD).

Per-rank totals over the whole log (prefill work = `#new-token` + `#cached-token`,
since a cached-token hit still costs the rank a scheduling slot and KV traffic):

| Run | prefill work max/min | mean `#running-req` max/min | busiest / idlest rank |
|---|---|---|---|
| `tbo-tp8-c64` (DP8+TBO, fusion off) | **1.60** | **1.56** | DP3 / DP7 |
| `megamoe-tp8-c64-eplb` (DP8+EP8) | **3.66** | **1.96** | DP2 / DP6 |

`tbo-tp8-c64` detail — decode-batch counts are near-uniform (max/min 1.05, i.e.
the ranks step in lockstep as expected) while the *work inside* each step is
not:

```
rank  decode_lines  mean_run_req  prefill_new_tok  cached_tok   mean_gen_tps
DP0            857          5.75           220564     8977628          172.9
DP3            851          6.85           308628    13806480          203.3
DP6            844          7.46           278556    12018480          221.3
DP7            864          4.77           222728     8595572          144.9
```

**Why this matters:** DP attention steps in lockstep — every rank waits at the
per-step sync, so the *busiest* rank sets the pace and the spread is pure idle
time on the rest. A 1.56x spread in concurrent requests is not a rounding
artifact of round-robin over 4k requests; it is round-robin distributing
*request count* evenly across a corpus whose ISL varies by orders of magnitude
(mean ISL ~90k, §11), so equal counts mean very unequal work.

The MegaMoE arm's 3.66x prefill-work spread is the worst of the three serving
paths and it is also the slowest (12,630.8 tok/s/GPU, §16) — consistent with
imbalance being a contributor, though EP8/MegaMoE remain confounded there.

**Conclusion:** `--load-balance-method total_tokens` is worth an arm. It is the
load-aware policy and this corpus is exactly the high-variance case it targets.
Command to reproduce the numbers above on any DP run:

```bash
awk '/DP[0-9]+ TP[0-9]+.*\] (Decode|Prefill) batch/ {
  match($0,/DP[0-9]+/); r=substr($0,RSTART+2,RLENGTH-2)+0
  if ($0 ~ /Prefill batch/) {
    if (match($0,/#new-token: [0-9]+/))    pt[r]+=substr($0,RSTART+13,RLENGTH-13)+0
    if (match($0,/#cached-token: [0-9]+/)) ct[r]+=substr($0,RSTART+16,RLENGTH-16)+0
  } else {
    if (match($0,/#running-req: [0-9]+/))  rr[r]+=substr($0,RSTART+14,RLENGTH-14)+0
    dn[r]++
  }
}
END { for(i=0;i<8;i++){ tw=pt[i]+ct[i]; mr=dn[i]?rr[i]/dn[i]:0
    printf "DP%d prefill_tok=%-12d mean_run_req=%.2f\n",i,tw,mr
    if(tw>mx)mx=tw; if(mn==0||tw<mn)mn=tw; if(mr>rmx)rmx=mr; if(rmn==0||mr<rmn)rmn=mr }
  printf "\nwork max/min=%.3f  run_req max/min=%.3f\n", mx/mn, rmx/rmn }' server.log
```

Note the `EP[0-9]+` suffix: EP runs log `[... DP7 TP7 EP7]`, so a regex anchored
on `TP[0-9]+\]` silently matches nothing and reports all zeros.

**Caveat:** this is whole-log, so it includes warmup. The lockstep decode-batch
counts and the size of the spread make a warmup artifact unlikely, but a
steady-state-only window would tighten it.

## 20. OPEN INVESTIGATION — why does DP8+TBO gain only 2.1 %?

> **Largely answered by §22**: DP alone gains +35 % at low cache hit, TBO alone
> never gains, and the two effects nearly cancel on AgentX's 95 %-hit workload.
> H1/H2 below stay as measured; read them for context, not as the live question.


Opened 2026-08-27, **handed to a fresh session.** Nothing is running; the
machine is idle. Read §15, §16, §18 and §19.6 before starting — this section
assumes them and does not repeat them.

### 20.1 The observation that started it

DP8+TBO has strictly better KV placement than TP8 and almost nothing to show
for it:

| | TP8+HiCache c64 | DP8+TBO c64 |
|---|---|---|
| pool tokens | 10,519,296 (shared) | 56,090,624 (8 x 7,011,328/rank) |
| `kv_usage` | 1.00 (full) | 0.50 |
| device hits | 68.9 % | **95.5 %** |
| host hits | 26.1 % = **128.572 M tok** | **0** |
| computed | 24.558 M (4.99 %) | 22.644 M (4.51 %) |
| req/s | 1.1751 | 1.1906 (+1.3 %) |
| headline tok/s/GPU | 17,079.5 | 17,443.3 (**+2.1 %**) |

ISL matches within 0.8 % (115,357 vs 116,248), so these two are comparable
(§19.5 gate passes).

### 20.2 Reframe first: the recompute is NOT the anomaly

The intuition "TBO has a 5.3x bigger pool and no host spill, so it should
recompute far less" does not survive the numbers. Overall cache hit is **95.0 %
(TP8) vs 95.5 % (TBO)** — 0.5 points apart, which §15 already recorded. The
extra pool converts *host hits into device hits*; it does not convert *misses
into hits*.

That means the residual ~4.5-5 % of prompt tokens are **compulsory misses** —
first occurrences of a prefix in the corpus — and no cache size removes them.
22.644 M is therefore close to the corpus floor, and only ~0.5 pt of hit rate
was ever available to win. Do not spend runs trying to lower it.

**The real question is: TBO avoids 128.572 M tokens of host->device KV transfer
and wins only 2.1 %. Where did that saving go?**

### 20.3 Hypotheses, cheapest first

**H1 — the saving was never worth much (zero GPU cost, do this first).**
Convert 128.572 M tokens to bytes and then to seconds. If host-tier traffic
only cost TP8 a few tens of seconds out of 3,600, there is no missing gain and
the investigation ends here. Note the model config has **no `kv_lora_rank`**
(DSV4-Pro exposes `o_lora_rank` 1024 / `q_lora_rank` 1536, `qk_rope_head_dim` 64,
61 layers), so do not assume the standard MLA formula — derive bytes/token
empirically from the pool: KV pool bytes / `max_total_num_tokens`
(`max_total_num_tokens=10519296` for TP8, `7011328`/rank for TBO; grep
`server.log` for the allocation line or the `mem usage=` lines). Then compare
against measured host->device bandwidth. **H1 is the null hypothesis and it is
the most likely answer — kill it before running anything.**

**H2 — DP rank imbalance burns the saving as lockstep idle (zero GPU cost).**
§18 measured 1.60x prefill-work and 1.56x mean-`#running-req` spread on this
exact run. DP attention syncs every step, so the busiest rank sets the pace. On
the 4 ranks §18 tabulated, mean/max `#running-req` = 6.21/7.46, i.e. **~17 %
idle on the average rank** — the right order of magnitude to erase a small
transfer saving. Extend the §18 script to all 8 ranks and compute the idle
fraction properly before believing that number.

**H3 — shared-experts fusion is off in the TBO arm (1 run).** §16 confound 1.
#32340 claims +8-11 %, so TP8 is flattered. This is §16 backlog 1 and it is the
only variable that could flip the ranking outright.

**H4 — the prefill delayer costs throughput as well as TTFT (1 run).** It ran
with `max_delay_passes=30` and **no wall-clock bound** (§19.6: `max_delay_ms` is
inert without `queue_min_ratio`). Rerunning TBO with the delayer *off* both
tests this and settles §19.6's TTFT attribution in the same run — good value.

**H5 — DP attention collective overhead offsets it.** No cheap test; leave last.

### 20.3b RESULTS — H1 killed, H2 does not show up at the GPU (2026-08-27, zero GPU cost)

**H1 is dead. There was never a saving to lose.** KV geometry is in
`server.log`: three paged pools at `pages=61636`, `page_size=256` tokens —
`deepseek_v4_c4` (30 layers x 65,536 B/page), `deepseek_v4_c128`
(31 layers x 2,048 B), `deepseek_v4_c4_indexer` (30 layers x 8,448 B). That is
**8,918 B/token**, cross-checked exactly: 61,636 x 256 = 15,778,816 host tokens
= 1.5000 x the 10,519,296 device pool, matching `hicache_ratio=1.5`.

So TP8's 128.572 M host-tier tokens = **1,146.6 GB** — which sounds large and
is not: over 3,629.5 s that is **0.32 GB/s aggregate, 40 MB/s per GPU**, about
0.6 % of a PCIe Gen5 x16 link. TBO eliminating the host tier saves
approximately nothing, and §20.1's premise ("where did the saving go?") is
answered: nowhere, because there was no saving. Do not run anything to chase it.

**H2 is real in the scheduler and invisible at the GPU.** Extending §18's script
to all 8 ranks on `tbo-tp8-c64`:

```
DP0 9,198,192   DP1 11,977,592  DP2 11,203,500  DP3 14,115,108
DP4 13,385,132  DP5 10,232,812  DP6 12,297,036  DP7 8,818,300     (prefill tok)
work max/min = 1.601   run_req max/min = 1.563
avg rank at 83.0 % of busiest  ->  17.0 % idle by run_req, 19.2 % by work
```

But `gpu_metrics.csv` (`gfx_activity`, second half of each run) shows **no
corresponding GPU idle**:

| | GPU0 | GPU1-7 | mean | spread |
|---|---|---|---|---|
| TP8 c64 | 48.5 % | 96.9-97.1 % | 91.0 % | 2.002 |
| DP8+TBO c64 | 47.9 % | 95.8-95.9 % | 89.9 % | 2.002 |

This is consistent, not contradictory: under DP lockstep a lightly-loaded rank
still enters every collective and launches the same kernels with fewer tokens,
so the waste is *inside* kernels as padding and wait, not as idle gaps between
them. **`gfx_activity` therefore cannot test H2** — it reports kernel residency,
not utilization. Any claim about imbalance cost needs kernel-level timing.

**Two anomalies surfaced that are more interesting than the 2.1 %:**

1. **GPU0 sits at ~48 % in BOTH arms** while GPU1-7 sit at ~96 %, with the ratio
   landing on exactly 2.002 in both. Identical across two different
   configurations smells like a metrics artifact (sampling rate, XCP mapping),
   but if it is real it is an eighth of the machine. Cross-check `socket_power`
   per GPU in the same CSV before believing either way.
2. **`umc_activity` is only ~17 % on every GPU in both arms** (GPU0 ~9 %). The
   memory controllers are nearly idle while gfx reports ~96 % busy.

**The reframed question for the rest of §20:** both arms burn ~90 % gfx activity
to produce roughly 850 computed prefill tok/s/GPU and ~135 output tok/s/GPU,
with memory controllers at 17 %, neither KV-transfer-bound nor recompute-heavy.
The 2.1 % gap is a symptom. The finding worth chasing is what consumes that
time in *both* arms — TBO cannot win a race whose cost is common to both.

### 20.3c The B200 published recipe reframes the question (2026-08-27, zero GPU cost)

Comparing the two published search spaces — `configs/amd-master.yaml` key
`dsv4-fp4-mi355x-sglang-agentic-mtp` vs `configs/nvidia-master.yaml` key
`dsv4-fp4-b200-sglang-agentic-hicache-mtp`:

| | MI355X (ours) | B200 |
|---|---|---|
| no-offload arm | tp4 ep1 dpa=false, conc 1-10 | tp8, conc 1-5 |
| HiCache TP arm | tp8 ep1 dpa=false, **conc 16,32,48** | tp8, **conc 8,10,16,32** |
| high-conc arm | **none** | **tp8 ep8 dpa=true + hicache + sglang-router 0.3.2, conc 32,64,96** (scaling range to 1024) |

**Pure TP8 on B200 stops at conc 32 — earlier than MI355X's 48.** B200 does not
show pure TP scaling further than we do; it shows that past ~32 the published
recipe *switches to DP attention + EP8*. Both platforms abandon pure TP in the
32-48 region, which is consistent with our c64 TP8 arm not gaining.

**So the question is not "why doesn't TBO beat TP8".** It is: *the configuration
B200 uses for its highest-concurrency points (dp-attn + ep8) is our worst arm
(MegaMoE, 12,630.8). Why?*

Two candidates fall out immediately, and one of them is a gap in coverage:

- **Our ep8 arm is not a plain ep8 arm.** `megamoe-tp8-c64-eplb` additionally
  carries AITER MegaMoE, EPLB, and PR #35619. §16 already attributes its
  26-28 % deficit to MTPR-aligned 8192-token padding against this
  high-variance-ISL corpus. B200's ep8 arm uses its own Blackwell MoE path, not
  AITER MegaMoE. **We have never run dp-attn + ep8 with the stock MoE**, which
  is the actual analogue of the B200 recipe.
- **Our TBO arm is ep1**, matching neither published search space.

**Ruled out: the router.** B200's config pins `sglang-router` 0.3.2 and the
installed router here is **also 0.3.2**. `megamoe-tp8-c64-eplb` sets
`--load-balance-method round_robin` explicitly; `tbo-tp8-c64` leaves it at the
`auto` default which resolves to the same policy (§18). Router version and
policy are not a differentiator.

### 20.4 Suggested order

1. H1 and H2 — both from files already on disk, no GPU. H1 may end it.
2. If a gap survives: one TBO run with fusion **on** (H3) — also §16 backlog 1.
3. One TBO run with `--enable-prefill-delayer` **removed** (H4) — also settles
   §19.6.
4. Then `--load-balance-method total_tokens` (§18's conclusion, backlog 2).

Runs 2-4 are ~75 min each (~25 min load + warmup, 3600 s profiling); see the
§11 warning that the launcher leaves the server holding all 8 GPUs, and the
CONTINUE HERE note on `rocm-smi --showpids` when `agentx_debug.sh stop` reports
`still draining`.

### 20.5 Report findings by

Filling a table in this section with the same columns as §20.1, and stating
which hypothesis survived. If H1 kills it, say so plainly and close §20 — a
2.1 % gain that is fully explained is a result, not a failure.

## 21. DONE (2026-08-28) — the DP+TBO fixed-seq-len ladder, as designed

> **Executed; results and verdict are in §22. Do not re-run this section.**
> Kept for its design rationale and for §21.2's pre-flight facts
> (`--random-range-ratio` semantics, the KV-capacity ceiling at conc 64), which
> are still the reference. §21.1's hypothesis and §21.3's DP reasoning were
> both refuted by the run — see §22.3.


Designed 2026-08-27 with the user, who supplied the decisive piece of context:
**the original DP+TBO PR was measured with the radix cache OFF, and the gain was
clearly visible there.** Everything below follows from that.

### 21.1 The leading hypothesis

TBO hides a DP collective behind the *other* micro-batch's compute. AgentX runs
with the radix cache on and a **95 % overall cache hit rate**, so only ~4.5 % of
prompt tokens are actually computed (§20.3b). **There is almost no prefill
compute left to overlap the collective with.** The PR's radix-off setting
recomputes every token, so it has plenty.

Prediction: TBO's gain is a function of **cache hit rate**, near zero at 95 %
and restored near 0 %. This subsumes the 2.1 % result without needing any
hardware explanation, and it is directly testable.

Note `bench_serving --dataset-name random` has a near-zero prefix hit rate, so
it **reproduces the PR's condition**. If TBO wins there, that CONFIRMS the
hypothesis rather than contradicting it. Do not read a win at 8K/1K random as
"TBO is fine".

### 21.2 Pre-flight, already settled — do not re-derive

- `--random-range-ratio` semantics (`benchmark/datasets/common.py:56`):
  `np.random.randint(max(int(full_len*range_ratio),1), full_len+1, num)`.
  So **1.0 = fixed length**, 0.25 = uniform over [0.25L, L] (4x spread),
  0.0 (default) = uniform over [1, L].
- Implementation moved: use `python -m sglang.benchmark.serving`
  (`sglang.bench_serving` is a deprecated shim).
- Cache hit rate is controllable: `--dataset-name generated-shared-prefix` with
  `--gsp-num-groups`, `--gsp-prompts-per-group`, `--gsp-num-turns`,
  `--gsp-question-len`, `--gsp-output-len`, `--gsp-range-ratio`.
  `--gsp-num-turns > 1` gives multi-turn, the closest synthetic analogue of the
  agentic corpus.
- **KV capacity ceiling at conc 64:** TP8's shared pool (10,519,296 tok) caps
  ISL at **164,364 tok/request**; TBO's per-rank pool (7,011,328 tok for 8
  req/rank) allows **876,416**. The agentic p90 ISL (216 K per §16, 272,800 in
  `armB-tp8-c64/workload_distribution_summary.txt`) therefore **does not fit
  TP8** at c64. Any long-ISL point above ~164 K measures *KV capacity*, not the
  DP mechanism — label it as such. 131,072 fits both (TP8 at 80 % of pool) and
  sits near the agentic mean ISL of 115 K, so it is the honest "long" point.

### 21.3 The config ladder — DP and TBO must be separated

**User-supplied prior (2026-08-27), not otherwise recorded here: with the radix
cache OFF, DP alone at conc 64 already beat TP — TBO was not required for the
win.** Every DP arm on file here (`tbo-tp8-c64`, `megamoe-tp8-c64-eplb`) carries
DP *and* something else, so **DP has never been measured alone in this
environment.** That prior forces a consequence worth stating plainly: if DP
alone beats TP and our DP+TBO arm only beats TP by 2.1 %, then **TBO may be net
negative here**, eating a gain DP would otherwise deliver. Nothing on file rules
that out.

Run a ladder where each rung adds exactly one thing:

| rung | config | isolates |
|---|---|---|
| A | TP8, no DP, no TBO, no delayer | baseline |
| B | + `--enable-dp-attention` (ep1) | **DP alone** — the user's prior |
| C | + `--enable-prefill-delayer` | the delayer's throughput cost |
| D | + `--enable-two-batch-overlap` | TBO alone (= our `tbo-tp8-c64` arm) |

Rung B is the diagnostic one, and it discriminates between hypotheses rather
than just adding a number:

- TBO's benefit is proportional to the prefill compute available to overlap a
  collective with, so §21.1 predicts it collapses at a 95 % cache hit rate.
- DP attention's benefit is proportional to attention work, and **decode
  attention reads the full KV every step regardless of cache hit rate**, so DP's
  benefit should *not* collapse the same way.

Therefore: if rung B shows no gain either, §21.1 does not explain the result and
the problem is elsewhere. If rung B gains and rung D gives it back, TBO is the
problem. Do not skip B.

### 21.3b The workload points

Configs reuse the exact agentic arm settings otherwise, so conclusions transfer
(MTP EAGLE 3/1/4 + `SGLANG_SIMULATE_ACC_LEN=2.49`, HiCache DRAM), all pinned at
**conc 64**.

| # | dataset | ISL/OSL | range_ratio | tests |
|---|---|---|---|---|
| 1 | random | 8K/1K | 1.0 | PR condition: low cache, uniform — **run first** |
| 2 | random | 8K/1K | 0.25 | + length variance |
| 3 | random | 131K/1K | 1.0 | + long ISL |
| 4 | random | 131K/1K | 0.25 | + both |
| 5 | gsp ~95 % hit | 8K/1K | 1.0 | **the cache axis** — same as 1 but cached |
| 6 | gsp ~95 % hit | 131K/1K | 0.25 | closest synthetic analogue of AgentX |

**Start with points 1 and 5 on all four rungs = 8 data points**, then stop and
report before running 2/3/4/6. Points 1 and 5 differ only in cache hit rate, so
that block alone tests §21.1 (across rung D) and the user's DP prior (across
rung B) at the same time.

Cost: one server launch per rung (~25 min) plus ~8 min per bench point, so the
8-point block is roughly 4 x (25 + 16) = **~2.7 h**. The equivalent in agentic
arms would be ~10 h — this is the whole reason for the fixed-seq-len detour.

Optional later: 272,800/1K to probe the capacity crossover where only DP fits
(§21.2).

### 21.4 If TBO shows no gain anywhere

Then it is not a workload-shape problem and the next move is the user's
fallback: **reproduce the original PR's numbers directly**, since a newer
SGLang may carry a regression. Known lead — there was a **prefill-delayer
regression** in some version, and the **mega-moe branch has a record of how it
was fixed**. Start from `../dsv4/megamoe/PR35619_UPSTREAM_REPRO.md`, which
already reproduced upstream numbers on fixed-seq-len in this environment.

### 21.5 Ground rules carried over

- Do not compare across concurrencies on headline throughput (§19.6). This is
  why the whole matrix is pinned at c64.
- Report the invariant metrics (req/s, computed tok/s/GPU) beside any headline.
- `gfx_activity` is kernel residency, not utilization (§20.3b) — it cannot
  settle a bound-ness claim.
- The launcher leaves the server holding all 8 GPUs; `agentx_debug.sh stop` can
  report `still draining` and exit 0 with schedulers alive. Find them with
  `rocm-smi --showpids` and `kill -9`, then confirm VRAM is 0 %.

## 22. RESULT — the §21 ladder, all 8 points (2026-08-28)

Ran §21.3's four-rung ladder on §21.3b's points 1 and 5 with
`agentx_ladder.sh` (new; see §22.6). All 8 points completed, 512/512 requests
successful each. **§21.1 is refuted, the user's DP prior is confirmed, and the
mechanism turned out to be neither of the two candidates.**

### 22.1 Configuration actually run

Common base for all four rungs (full argv in each
`/workspace/results/ladder/rung*/sglang_command.txt`): TP8, `dsv4` backend,
page 256, `swa-full-tokens-ratio 0.15`, `kv-cache-dtype fp8_e4m3`, MTP EAGLE
3/1/4 + `SGLANG_SIMULATE_ACC_LEN=2.49`, HiCache DRAM 1.5 write_through,
`max-running-requests 128`, `cuda-graph-max-bs 128`.

| rung | added |
|---|---|
| A | (nothing) |
| B | `--dp 8 --enable-dp-attention` + the recipe's 4-var DP env bundle |
| C | `+ --enable-prefill-delayer` |
| D | `+ --enable-two-batch-overlap` (= the on-file `tbo-tp8-c64` config) |

**Two knobs the recipe couples to DP were held constant instead**, so each step
is single-variable: shared-experts fusion **off in all four rungs** and
`--chunked-prefill-size 65536` in all four. Consequence: **rung A is NOT the
published TP8 arm** (that one is fusion-on, cps 8192). Ladder numbers are
internally comparable only — do not put them beside §12/§19 agentic numbers.
No `sglang_router` in any rung (bench_serving sends no routing key). The A->B
step is the DP *bundle*, not one flag: `SGLANG_SHARED_EXPERT_TP1` changes model
construction even without DP, so forcing it into rung A would confound more
than it controls.

Both points: conc 64, 512 prompts, seed 42, 64 warmup requests,
`--backend sglang` (native `/generate`), `POST /flush_cache` before each.
Point 1 = `random` 8192/1024 `--random-range-ratio 1.0` (measured ISL exactly
8192). Point 5 = `generated-shared-prefix`, 8 groups x 64 prompts,
`--gsp-system-prompt-len 7936` (31 x page 256) + `--gsp-question-len 256`,
`--gsp-output-len 1024`, `--gsp-range-ratio 1.0` (measured ISL 8375, ~2.3 %
above point 1 — gsp adds separator text; quote that, not "identical ISL").

### 22.2 The numbers

Point 1 — random 8K/1K, measured hit rate 8-11 % (the PR condition):

| rung | req/s | total tok/s | vs A | TTFT p50 | TPOT | hit % | computed prefill |
|---|---|---|---|---|---|---|---|
| A TP8 | 1.80 | 16,623 | — | 10.47 s | 24.52 ms | 10.77 | 4.21 M |
| B +DP | 2.43 | 22,356 | **+35.0 %** | 7.19 s | 19.12 ms | 8.24 | 4.33 M |
| C +delayer | 2.45 | 22,612 | +36.1 % | 5.93 s | 19.13 ms | 8.24 | 4.33 M |
| D +TBO | 2.44 | 22,456 | +35.6 % | 5.90 s | 19.27 ms | 8.24 | 4.33 M |

Point 5 — shared-prefix 8K/1K, measured hit rate 83-92 %:

| rung | req/s | total tok/s | vs A | TTFT p50 | TPOT | hit % | computed prefill |
|---|---|---|---|---|---|---|---|
| A TP8 | 3.81 | 35,781 | — | 1.54 s | 14.64 ms | **91.82** | **0.40 M** |
| B +DP | 3.70 | 34,780 | −2.9 % | 1.95 s | 14.36 ms | 83.24 | 0.82 M |
| C +delayer | 3.42 | 32,183 | −10.2 % | 1.20 s | 15.82 ms | 83.56 | 0.80 M |
| D +TBO | 3.05 | 28,691 | **−19.9 %** | 1.47 s | 17.48 ms | 83.45 | 0.81 M |

Per-step deltas (req/s), which is what the ladder was built to read:

| step | point 1 | point 5 |
|---|---|---|
| A->B  DP alone | **+35.0 %** | −2.9 % |
| B->C  delayer | +0.8 % | −7.6 % |
| C->D  TBO alone | **−0.4 %** | **−10.8 %** |

### 22.3 What this says

**1. The user's prior is confirmed, strongly.** DP attention alone, with no TBO
and no delayer, is **+35 % at a near-zero cache hit rate**. DP has never been
measured alone in this environment before; it is the single largest lever on
file for this workload shape.

**2. §21.1 is refuted.** The hypothesis was that TBO's gain is a function of
cache hit rate — near zero at 95 %, restored near 0 %. At an 8.2 % hit rate,
the condition the hypothesis says should restore it, **TBO gives −0.4 %**. It
does not gain anywhere on this ladder. There is no cache-hit-rate story for
TBO because there is no TBO gain to explain.

**3. TBO is net negative, and expensively so at high hit rate** (−10.8 % on
point 5). The §16 reading that "TP8 and DP8+TBO land within 2.1 %" now
decomposes: DP was delivering a large gain and TBO plus the delayer were giving
most of it back. That is why the agentic arm looked like a wash.

**4. The delayer costs throughput and buys TTFT, as designed.** +0.8 %/−7.6 %
on throughput, but TTFT p50 10.47->5.93 s on point 1 and 1.95->1.20 s on point
5. §19.6's TTFT attribution to the delayer stands; §16's backlog item 5 is
answered in the direction §19.6 predicted.

**5. The real mechanism is cache partitioning, not overlap headroom.** DP8
splits the radix cache per rank, so the same shared prefix must be materialised
on every rank that sees it. Measured on point 5: TP8 hits **91.8 %** and
computes **0.40 M** prefill tokens; every DP rung hits **83.2-83.6 %** and
computes **0.80-0.82 M** — **2x the prefill work for the same client
workload.** That, not a DP-attention slowdown, is why DP's +35 % becomes −2.9 %
when the workload has a shared prefix.

The reframing matters: by *computed* throughput (computed prefill + generated
over the same window, per GPU) DP is **faster** at point 5 too —
A 762, B 1,105 (+45 %), C 1,010, D 923 tok/s/GPU. DP is not slower; it is
doing more redundant work. Note this also breaks §21.3's stated reasoning
("DP's benefit should not collapse with hit rate, because decode attention
reads the full KV every step"). It does collapse — via prefill duplication,
which that argument did not consider.

**6. AgentX at 95 % hit is the regime where DP's advantage is smallest and
TBO's cost is largest.** Consistent with the 2.1 % of §16 without needing any
hardware explanation.

### 22.4 Where this leaves §21.4

§21.4's trigger has fired: TBO shows no gain anywhere, so this is not a
workload-shape problem. **Next move is §21.4's fallback — reproduce the
original PR's numbers directly**, starting from
`../dsv4/megamoe/PR35619_UPSTREAM_REPRO.md`, on the suspicion that a newer
SGLang carries a TBO regression. Point 1 of this ladder is already close to the
PR's condition and is cheap to re-run against an older build.

Second, **DP's cache partitioning is now the highest-value open question for
AgentX**: if the per-rank split can be avoided (cache-aware routing that pins a
prefix to a rank — which is exactly what the recipe's `sglang_router` routing
key does, and which this ladder deliberately omitted), DP's +35 % may survive
into the high-hit-rate regime. **That is a real candidate for beating the
published arm, and it was invisible until DP was measured alone.**

### 22.5 Caveats — read before quoting any of this

- **Rung A is not the leaderboard TP8 config** (§22.1). Internal comparisons
  only.
- **Short windows, single seed**: 135-284 s per point, 512 prompts, seed 42.
  Directionally strong (+35 % and −10.8 % are far outside run-to-run noise on
  these arms) but not submission-grade.
- **Point 5's hit rate is 83-92 %, not the 95 % targeted, and it differs
  between rungs.** So point 5 is *not* a controlled comparison at fixed hit
  rate — the hit rate is itself an outcome of the rung. That is the finding
  (§22.3 item 5), but it means "DP is −2.9 % at 95 % hit" would be a
  misstatement. The honest form is "DP is −2.9 % on a shared-prefix workload,
  because DP only achieves 83 % hit where TP achieves 92 %".
- Points 1 and 5 differ in hit rate *and* in available prefill compute — the
  same axis by construction. Fine for the §21.1 question, not usable for
  attributing one without the other.
- `computed tok/s/GPU` in §22.3 is window-based and **includes the 64 warmup
  requests**; client-side req/s and total tok/s are over the bench's own
  measured duration. Do not mix the two denominators.

### 22.6 Traps found while running this (all cost real GPU time)

- **`sglang:cache_hit_rate` reads 0.0 on this path** — all 8 ranks, every
  point, including the 92 %-hit one. The Prometheus gauge is not wired for the
  unified-radix/HiCache path. **Use the scheduler log instead**: every prefill
  batch logs `#new-token: N, #cached-token: M`. §22.2's hit column and computed
  prefill are summed from those over each point's time window
  (`/tmp/hit.py`). This is a stronger version of §15's warning — the metric is
  not merely device-tier, here it is absent.
- **`stop` must match `sglang::` too.** A DP8 server's scheduler children are
  `sglang::scheduler_*`; the old `sglang.launch_server|sglang::router` pattern
  reports `sglang_procs=0` while 91 % of VRAM is still pinned, and the next
  rung is then refused by the clean-check. An 8-rank server also needs **more
  than 200 s** to release VRAM after SIGTERM (135-205 s observed when it works).
  `agentx_ladder.sh stop` now matches `sglang::`, waits 600 s, and escalates to
  SIGKILL at 120 s.
- **`set -e` + `xargs -r kill -9`**: a nonzero kill aborts the whole script, so
  `stop` silently gave up at 120 s instead of 600 s. Needs `|| true`.
- **Never edit a shell script while it is running.** Patching
  `agentx_ladder.sh` in place mid-run made the running bash resume at a stale
  byte offset and die with `line 204: unexpected EOF` on a file with no line
  204 — *after* rung C's server had reached `READY after 1170s`, so a healthy
  20-minute weight load was thrown away. Write to a temp file and `mv`, and
  point long runs at a frozen copy (`.ladder_frozen.sh`). Keep that copy **in
  the skill dir**: a copy in `/tmp` breaks `SKILL_DIR` and cannot source
  `agentx_env.sh`.
- **`python -m sglang.benchmark.serving` fails from `/sgl-workspace`**: the
  repo directory `/sgl-workspace/sglang` shadows the installed package as a
  namespace package, so the module is "not found". Run it from any other cwd.
- **The output JSON records unused `random_*` defaults for gsp runs.** Point 5's
  JSON says `random_range_ratio: 0.0, random_input_len: 1024` — neither is in
  effect, and the `gsp_*` values that *are* in effect are not recorded at all.
  Read the invoked argv, not the JSON, when checking a gsp point's shape.

**Artifacts:** `/workspace/results/ladder/rung{A,B,C,D}/` — `sglang_command.txt`,
`server.log`, `point{1,5}.json`, `point{1,5}.metrics.{pre,post}.txt`. Bench
stdout in `/tmp/ladder_bench_<rung>_<point>.log`, driver logs
`/tmp/ladder_all.log` (A,B) and `/tmp/ladder_cd3.log` (C,D).

## 23. Cache-aware routing recovers DP's cache split — and loses more than it wins (2026-08-28)

§22.4's item 2, run with `agentx_router.sh`. Rung B config (DP8, no delayer,
no TBO) on backend 8889, `sglang_router` on 8888 with `--policy
consistent_hashing --dp-aware --request-id-headers x-smg-routing-key`.
bench_serving's `--gsp-send-routing-key` sends `X-SMG-Routing-Key` = one key
per gsp group (the recipe hashes `x-correlation-id` instead, which is what
aiperf sends — point the router at whichever header the client actually emits).

| point | req/s | vs rung B direct | hit % | computed prefill | computed tok/s/GPU |
|---|---|---|---|---|---|
| rung B direct (§22) | 3.70 | — | 83.24 | 0.82 M | 1,105 |
| router, no key | 3.12 | −15.7 % | 83.12 | 0.82 M | 980 |
| router + routing key | 2.20 | **−40.5 %** | **91.29** | **0.42 M** | 493 |
| TP8 reference (§22) | 3.81 | +3.0 % | 91.82 | 0.40 M | 762 |
| router, point 1 (random) | 2.03 | −16.5 % | 8.24 | 4.33 M | — |

**The cache half worked exactly as predicted.** Routing on a prefix-stable key
took DP's hit rate from 83.1 % to 91.3 % and halved computed prefill to 0.42 M
— landing on TP8's 91.8 % / 0.40 M. §22.3 item 5's cache-partitioning mechanism
is confirmed: the per-rank split *is* repairable by routing.

**The throughput half collapsed, from rank imbalance.** Prefill batches per DP
rank, same window:

```
no key : [32, 31, 33, 30, 29, 33, 31, 34]   max/min 1.1x
key    : [ 2, 38,  2, 24, 48,  2, 34,  2]   max/min 12.7x (computed tokens)
```

Four ranks went idle. The router itself also costs ~16 % on both datasets
(2.03 vs 2.43 on point 1, 3.12 vs 3.70 on point 5), independent of routing.

**Design flaw in this run — do not read it as a verdict on cache-aware
routing.** 8 gsp groups hashed onto 8 DP ranks is the pathological case for
consistent hashing: 8 keys into 8 buckets leaves ~2.6 buckets empty by
expectation, and we measured 4 effectively empty. Real AgentX has thousands of
distinct sessions, where the same hash spreads fine. **The decisive re-run is
the same experiment with many more keys than ranks** — e.g. `--gsp-num-groups
64 --gsp-prompts-per-group 8` (still 512 requests), which keeps cache affinity
while restoring balance. Until that runs, the honest claim is only: *routing on
a prefix-stable key fixes DP's cache split, and with keys ≈ ranks the resulting
imbalance costs more than the cache gain.*

Artifacts: `/workspace/results/ladder/rungB-router/` (`server.log`,
`router.log`, `windows.txt`, `{p1,nokey,key}.json`). Per-rank and per-window
hit rates are recomputed from `server.log`'s `DP<n> ... #new-token/#cached-token`
lines — regex must anchor as `(\d\d:\d\d:\d\d) DP(\d+)`, the rank tag sits
*inside* the leading bracket with the timestamp.

