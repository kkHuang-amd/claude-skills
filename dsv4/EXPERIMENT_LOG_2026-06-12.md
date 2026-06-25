# DeepSeek-V4-Pro serving perf — experiment log (2026-06-12)

Split from the master `EXPERIMENT_LOG.md` (chronological, by date). See that file for the index and `SKILL.md` for how-to.

---

## Exp 19 — Apple-to-apple PREFILL trace: methodology failure + corrections — 2026-06-12

### What was attempted
Per-kernel prefill comparison SGLang(QR=INT8/NONE) vs ATOM single-stream, ALL at
16384 tok/step/rank (aligned launch scripts), DP0/TP0 torch trace, 6s window.
Goal: see if SGLang prefill spends more in collectives than ATOM.

### Trace settings were NOT fully apple-to-apple (root issue)
Server launch WAS aligned (tp8+dp8, fp8 KV, page/block256, max-seqs512,
mem0.90, prefix-cache off, single-stream, 16384/rank). BUT the **client load
differed**:
- SGLang traces: ISL8192 OSL8 c8 np128
- ATOM trace (1st): same c8/np128 → UNUSABLE (ATOM drained before window, GPU 3%)
- ATOM trace (rerun, the one analyzed): **c32 np1024 OSL1** to sustain prefill.
⇒ SGLang(c8) vs ATOM(c32) — different load. Re-captured SGLang at MATCHED
**c32/np1024/OSL1** (prof_aa_sgl_int8_matched, verified #new-token=16384/step).

### FINDING 1 — "collective fraction" is LOAD-SENSITIVE (kills Exp 15/17 claim)
Same SGLang, same 16384/rank, only load differs:
| SGLang load | collective / union-busy |
|---|---|
| c8/np128  | 71% |
| c32/np1024| 33% |
The "SGLang prefill ~70% in collective" headline was a **low-concurrency (c8)
artifact**: tiny batch → compute kernels finish fast → fixed collective latency
dominates. At realistic c32 it drops to 33%. ⇒ collective-fraction is NOT a
valid cross-config metric. Consistent with Exp 18 retraction.

### FINDING 2 — trace `dur` is SYNC-INFLATED → absolute cross-engine compare INVALID
On a SINGLE stream, summed kernel-time exceeds wall-span (impossible w/o
inflation):
- SGL tid6: 12171ms over 9094ms span = 134% of span
- ATOM tid3: 35314ms over 9602ms span = 368% of span
Longest single "kernels" are 2.7–4.4 SECONDS (mhc_pre_big_fuse_rmsnorm 4362ms,
dynamic_per_group_scaled_quant 4361ms, gemm_a16w16 4362ms). No GPU kernel runs
4s — the recorded `dur` absorbs **host-side launch-barrier / stream-sync wait**.
⇒ "sum of dur", "us/launch", and "us/moe-sort" are ALL meaningless for absolute
cross-engine cost. ATOM's apparent 5–9× per-kernel cost is an artifact, not real.
This is the same symptom as the earlier %busy>100% / 7234us-per-launch anomaly.

### Implication for the gap investigation
Per-kernel torch-trace cost attribution CANNOT fairly compare SGLang vs ATOM
here (sync-wait inflation + load sensitivity + different stream layout). The gap
must be pursued via end-to-end throughput/TPOT under matched real load (Exp 18
2×2), NOT via trace kernel-time totals. Exp 18 conclusion stands: bottleneck is
**SGLang prefill↔decode scheduling/interleave efficiency**, not all-reduce
count/kernel. Trace artifacts here do not support an all-reduce-cost story.

### Artifacts
- /workspace/prof_aa_sgl_int8_matched/  (SGLang c32 matched trace, 8 ranks)
- /workspace/prof_aa_sgl_int8|none/, /workspace/prof_aa_atom/ (earlier)
- /workspace/union_busy.py, union_busy2.py (analysis; note caveats above)
- /workspace/run_sgl_trace_matched.sh (matched-load capture orchestrator)

## Exp 20 — prefill-delayer ON/OFF A/B: delayer is ESSENTIAL, not the gap — 2026-06-12

### Setup (clean: chunk NOT a variable)
SGLang dp8, 8k/1k c256, ratio0.8, np2048, warm512. BOTH runs
`--chunked-prefill-size 131072` (=16384/rank, matched to ATOM; confirmed
#new-token=16384 dominant). Only `--enable-prefill-delayer` differs.
Delayer config (verified ON-run log): max_delay_passes=30,
token_usage_low_watermark=None, queue_min_ratio=None, max_delay_ms=5000.

### Result (decisive)
| metric | delayer ON | delayer OFF | OFF vs ON |
|---|---|---|---|
| total tok/s | **23,908** | 14,103 | **−41.0%** |
| output tok/s | 2,656 | 1,567 | −41.0% |
| median TPOT ms | 92.2 | 167.7 | +81.8% |
| median TTFT ms | 2,106 | 1,428 | −32% (OFF lower) |
| bench duration s | 711 | 1,206 | +69.5% |

⇒ **prefill-delayer is NOT the gap cause; it is CRITICAL to SGLang throughput.**
Turning it off makes SGLang far worse (−41%). KI-1 "delayer over-throttles" is
REFUTED for this workload. (delayer ON ≈ Exp 16's 23,882 → consistent.)

### Mechanism (why OFF is 2× slower per prefill)
- retract events: **0 in BOTH** runs. swa-usage-1.00: **0 in BOTH**. new_token
  _ratio collapse: none. ⇒ at chunk16384/rank + swa-ratio0.15 + c256, **swa pool
  is sufficient** (confirms user hypothesis: no need to raise swa-full-tokens-ratio).
  The earlier user-pasted retract+swa1.00 log was a DIFFERENT config (likely
  2048/rank fragmentation or higher conc).
- Same total prefill work both (19,311,616 tokens). Same decode batch fullness
  (median #running-req 32, mean 30 both). Same step counts (~2.1k prefill,
  ~1.5k decode).
- KEY: **prefill input throughput median 3814 tok/s (ON) vs 1828 tok/s (OFF)** —
  OFF runs each prefill batch ~2× SLOWER despite identical work.
- ⇒ The delayer's real job in dp-attention is **cross-rank prefill timing
  coordination**: it gates prefill so the 8 DP ranks enter prefill together.
  Without it, ranks desync → per-step all-gather/MoE waits on the slowest rank →
  effective prefill throughput halves. This is the dp8 "wait on slowest rank"
  tax (cf. dp_attn.py prepare_mlp_sync_batch + idle-batch padding).

### Conclusion + next direction
The remaining SGLang↔ATOM gap is NOT delayer, NOT all-reduce count, NOT chunk
size, NOT retract/swa. It is **dp-attention's per-step cross-rank synchronization
efficiency**: SGLang's 8 DP ranks must agree (all-gather) every forward on
prefill-vs-decode and pad with idle batches; throughput is bounded by the slowest
rank each step. ATOM evidently coordinates prefill across ranks more cheaply (or
does not pay the same per-step global sync). Next: (a) compare how ATOM's
scheduler handles multi-rank/dp prefill coordination (user-suggested mechanism
comparison), (b) measure the all-gather/idle-batch overhead per step in SGLang
dp8 directly (scheduler timing, not torch trace).

### Artifacts (Exp 20)
- /workspace/run_delayer_ab.sh  (A/B orchestrator)
- /workspace/bench_delayer_ab/  (jsonl + bench logs)
- /workspace/server_logs/delayer_delayerON|OFF.serverlog
- run_sgl_dsv4_aligned.sh: added DELAYER=on|off knob (default on, preserves prior)

## Exp 21 — ATOM scheduler code study vs SGLang (mechanism comparison) — 2026-06-12

ATOM path: /opt/venv/lib/python3.10/site-packages/atom/model_engine/

### HEADLINE: ATOM scheduler is architecturally the SAME as SGLang
- `scheduler.py:schedule()`: tries **prefill first**, returns a **pure prefill
  batch** if any (lines 859-888), else falls to decode. Identical prefill-XOR-
  decode design to SGLang's `get_next_batch_to_run`. NOT mixed-batch.
- `prefill_delayer.py`: docstring line 4 literally says **"Direct port of
  SGLang's PrefillDelayer"**. Same all/none/mixed → delay logic, same defaults
  (max_delay_passes=30, max_delay_ms=5000, watermark=None), **enabled by default**
  (`ATOM_ENABLE_PREFILL_DELAYER=1`, utils/envs.py:200). So both engines ran WITH
  the same delayer in all our experiments. The +41% delayer benefit (Exp 20) is
  the SAME mechanism on both sides.
⇒ The gap is NOT high-level scheduling policy, NOT priority, NOT the delayer.

### Real differences found (candidates for the remaining gap)
1. **Decode KV pressure handling: preempt (ATOM) vs retract+new_token_ratio (SGL)**
   - ATOM `preempt()` (scheduler.py:967): on `can_append` failure, pops the
     TAIL of running, frees its KV, requeues to waiting head. Simple LIFO
     preempt, NO upfront memory reservation.
   - SGLang reserves decode memory UPFRONT via `new_token_ratio` (×0.3 under
     dp-attn → init 0.21) which SHRINKS prefill admission budget every step,
     then retracts on OOM. ATOM does NOT pre-reserve → more KV free for prefill
     admission, fewer artificial prefill stalls.
   - BUT Exp 20 showed 0 retract at our setting, so this is not active here.
2. **MoE collective: all_to_all/EP (ATOM) vs all-reduce (SGLang dp-attn)**
   - ATOM delayer docstring (lines 26-32): mixed forwards cost = "MoE all_to_all
     bottlenecked by prefill rank's tokens, 81.7% pad_waste, 67% wall in
     moe.gather". ATOM MoE uses **expert-parallel all_to_all dispatch**.
   - SGLang aligned config uses TP-MoE with **per-layer all-reduce** (Exp 19
     trace: AR cross_device + quickreduce + allgather dominate). Different MoE
     comm primitive. This is the likely structural differentiator and matches
     Exp 20's "SGLang prefill input tput = half of... its own potential" via
     per-step dp-sync + AR.
3. **schedule_conservativeness ×0.3 auto-applied under dp-attn** (SGL
   server_args.py:3313) → init new_token_ratio 0.7×0.3=0.21. ATOM has no
   equivalent auto-shrink.

### SGLang tunable knobs to try (no code change; all are ServerArgs/env)
A. `--moe-a2a-backend deepep --ep-size 8`  → switch SGLang MoE from TP-all-reduce
   to **expert-parallel all_to_all**, matching ATOM's MoE comm. MOST LIKELY to
   close the gap if diff #2 is the cause. (Earlier Exp 15 "next step" never run.)
B. `--prefill-delayer-token-usage-low-watermark X` (default None) → enable the
   safety valve so prefill is force-allowed when KV underutilized; may reduce
   over-delay tail at high conc.
C. `--prefill-delayer-queue-min-ratio R` → opt-in adaptive queue trigger; ATOM
   intentionally did NOT port this (delayer docstring line 38), so SGL has a
   knob ATOM lacks — could help fragment-prefill workloads.
D. `--schedule-conservativeness 1.5..3` → counteract the ×0.3 dp-attn shrink,
   raise init new_token_ratio so prefill admission budget is less throttled.
   (Only matters if retract/admission-throttle is active — not at Exp 20 setting,
   but may matter at higher conc / smaller chunk.)

### Recommended next experiment
**A/B SGLang `--moe-a2a-backend deepep --ep-size 8` vs aligned baseline**, 8k/1k
c256 (+ maybe c128), chunk16384/rank, ratio0.8. This directly tests whether
ATOM's lead is its EP/all_to_all MoE comm vs SGLang's TP all-reduce — the one
remaining structural difference consistent with all data so far.

## Exp 21b — CORRECTION: ATOM does NOT use EP; option A premise was WRONG — 2026-06-12

### User caught the error
ATOM aligned launch (run_atom_dsv4_aligned.sh) = `-tp 8 --enable-dp-attention`
ONLY. NO `--enable-expert-parallel`. Code confirms (moe.py:85-86,108):
`use_all2all_kernels = dp_size>1 AND use_ep AND has(mori)`, and
`use_ep = ... AND parallel_config.enable_expert_parallel`. With EP off →
**use_ep=False → NO mori all2all**. So ATOM ran TP-MoE, NOT EP/all_to_all.
⇒ Exp 21's "ATOM uses EP all_to_all" claim is RETRACTED. The delayer docstring's
"moe.gather / all_to_all" describes ATOM's EP-mode path, not the mode we ran.

### What BOTH actually do under dp-attention + no-EP (now confirmed)
Both flatten DP into a larger TP group for the MoE and bridge attn(dp)→MoE(tp):
- ATOM (moe.py:104-117): enable_dp_attention → `flatten_tp_across_dp` →
  tp_size = dp_size×tp_size = 64; MoE comm = `all_gather_with_padding` hidden
  states across DP ranks (line 205) → expert GEMM → `reduce_scatter_with_unpadding`
  (line 215). No per-rank all-reduce; it's all_gather + reduce_scatter.
- SGLang (dp_attention.py:243-244): attn_tp_size = tp_size//attn_dp_size (=1 at
  tp8dp8 for attn), MoE at full tp=8; transition uses all_gather before MoE +
  all-reduce/reduce-scatter after. Exp 19 trace showed AR + allgather dominate.
⇒ Same FAMILY (gather → expert compute → scatter/reduce). NOT a TP-allreduce vs
EP-all2all dichotomy. Option A (force SGLang EP) would make SGLang DIFFERENT from
ATOM, not closer — so A is NOT the right apple-to-apple test. SHELVED.

### The MoE intermediate-padding angle (user's real lead — needs measuring)
moe_inter_dim = 3072 (deepseek_v4.py:262). ATOM pads intermediate per partition
up to a 256 multiple (moe.py:780-793, pad_align=256, tracks `intermediate_pad`).
User's point: 3072 is divisible by 256 → if NOT TP-sharded (EP/full expert),
no pad waste; but if TP-sharded, 3072/tp may not align to 256 and pads.
Earlier work already found "SGLang MoE lacked intermediate_pad → extra compute"
and fixed it, yet gap persisted. So padding alone is not the whole gap, but the
EXACT per-partition intermediate size on each engine (and resulting pad) is still
unconfirmed and worth a direct check.

### Corrected next steps (do NOT run EP A/B)
1. Determine the EFFECTIVE MoE sharding on EACH engine in OUR config (tp8dp8,
   no EP): is the routed-expert intermediate (3072) TP-sharded (→ small per-part,
   heavy 256-pad) or kept whole per rank? Read both weight-creation paths and/or
   dump per-expert w13/w2 shapes at load time. This decides whether pad/GEMM
   shape is the differentiator.
2. Compare the dp-bridge comm volume: ATOM all_gather+reduce_scatter vs SGLang
   all_gather+all_reduce — per-step bytes and kernel, measured from scheduler
   timing / a SMALL controlled microbench (NOT the unreliable full trace).
3. Only after (1)(2) pinpoint a concrete divergence, pick the matching SGLang
   knob (e.g. moe_dense_tp_size, ep, or a pad fix) for an A/B.

## Exp 22 — ROOT CAUSE FOUND: variable-length vs MAX-pad DP-MoE gather — 2026-06-12

### Source: ATOM PR #930 (merged 5/27) + #157 (merged 1/21), both IN our build
Verified our atom build contains PR #930 code: moe.py has `all_gatherv`,
`reduce_scatterv`, `dp_uniform_decode`, `use_dp_gather_scatter`,
`dp_gather_hidden_and_router`. PR #157 added the 3-mode DP-MoE split.

ATOM has THREE DP-MoE modes (moe.py:3112-3115, PR #157+#930):
  1. Pure DP (`-dp N`, no dp-attn): NO MoE all_gather/reduce at all.
  2. DP-attn + EP (mori all2all): `--enable-expert-parallel`.
  3. DP-attn + TP all_gather/reduce (our mode): `-tp8 --enable-dp-attention`.
PR #157 perf note (1k/1k c128): **Pure DP 20400 >> EP-mori 14000 > all_gather/
reduce TP ~lower**. So even ATOM considers our mode (3) the SLOWEST of its three.

### THE KEY MECHANISM DIFFERENCE (answers the whole investigation)
In DP-attn + TP-MoE (mode 3, what BOTH engines run in our config), the MoE must
gather hidden states across DP ranks, run experts, scatter back. The cost hinges
on HOW the gather handles UNEVEN per-rank token counts (mixed prefill/decode):

- **ATOM (PR #930)**: `dp_eager_mode = not dp_uniform_decode`. When any rank is
  mid-prefill (ranks have DIFFERENT token counts), uses **variable-length
  `all_gatherv` / `reduce_scatterv`** with per-rank `sizes` (moe.py:261-282).
  Gathers exactly sum(real tokens). Also fuses hidden+router into ONE gather.
  → NO padding waste in mixed steps.
- **SGLang**: `DpPaddingMode` (dp_attention.py:53-86). `MAX_LEN` mode pads EVERY
  rank to `max(global_num_tokens)` then `all_gather_into_tensor`. In a mixed step
  (1 rank prefills 16384, 7 ranks decode ~32), ALL 8 ranks are padded to 16384 →
  gathers 8×16384 instead of 16384+7×32 → **~8× comm + 8× wasted expert compute
  on padding**. SGLang only escapes to `SUM_LEN` (all_reduce) when
  `is_extend_in_batch and dp_size>1` — i.e. the WHOLE batch is extend; the
  worst case (MIXED prefill+decode across ranks) still hits MAX_LEN pad.

⇒ THIS is why ATOM converts a big prefill chunk into throughput and SGLang does
not (Exp 18/20): ATOM's variable-length gather keeps mixed-step MoE cheap; SGLang
pads mixed steps to max-len, inflating both the all-gather volume AND the expert
GEMM (padding tokens are computed). Matches Exp 20 "SGLang prefill input tput =
½" and the delayer's value (delayer aligns ranks → fewer mixed/max-pad steps).
This is consistent with ALL prior data and needs NO unreliable trace.

### What SGLang can tune / do about it (concrete)
- SGLang ALREADY has variable-length `all_gatherv`/`reduce_scatterv` BUT only on
  the `should_use_flashinfer_cutlass_moe_fp4_allgather()` path
  (token_dispatcher/standard.py:152,229) — NOT active for our ROCm/aiter fp8
  MoE. The general dp-attn path uses DpPaddingMode MAX_LEN/SUM_LEN padding.
- Knob candidates to test (in priority order):
  A. Force **SUM_LEN** mode (all_reduce, no max-pad) for mixed steps — check if a
     server arg / env selects DpPaddingMode, or if `--moe-dense-tp-size` /
     attention-dp config changes the gather. SUM_LEN avoids the 8× max-pad blow-up.
  B. **Prefill-delayer tuning** to MINIMIZE mixed steps: lower max_delay_ms is
     wrong (more mixed); instead ensure delayer keeps ranks aligned. Already ON
     and helping (+41%, Exp 20). Try `--prefill-delayer-queue-min-ratio` (SGL-only
     knob ATOM didn't port) to further reduce fragmentation.
  C. File/seek a SGLang feature: variable-length dp gather for the aiter MoE path
     (port the flashinfer all_gatherv path to the general path) — the real fix,
     mirrors ATOM PR #930.

### Recommended next experiment
Test (A): find how SGLang selects DpPaddingMode and force SUM_LEN (or confirm it's
already SUM_LEN at our setting via logging `get_dp_padding_mode`), then A/B vs
MAX_LEN at 8k/1k c256. If SGLang is stuck in MAX_LEN for mixed steps, that's the
quantified gap; if already SUM_LEN, the residual is the all_reduce-vs-gatherv +
fused-gather efficiency, pointing to feature (C).

Refs: ATOM PR #930 (github.com/ROCm/ATOM/pull/930), PR #157
(github.com/ROCm/ATOM/pull/157).

## Exp 23 — DP-MoE gather comm microbench (isolates the primitive) — 2026-06-12

### Why: user noted SUM_LEN uses all_reduce (not gather) — primitive differs from
ATOM's all_gatherv even when both avoid padding. So a MAX_LEN-vs-SUM_LEN model
A/B can't prove "SGLang can match ATOM". Instead measured the 3 primitives
directly (no model, no unreliable trace). /workspace/moe_comm_microbench.py,
torchrun 8×MI355X, hid=7168 bf16.

### SGLang SUM_LEN is all_reduce-as-gather (confirmed, dp_attention.py:463-495)
`_dp_gather_via_all_reduce`: zero a sum_len buffer → memcpy local slice into own
offset → all_reduce(SUM). Result == all_gatherv result (sum_len, no pad) BUT each
rank ships the FULL sum_len buffer (mostly zeros) through ring all_reduce (~2×
traffic). MAX_LEN → `_dp_gather_via_all_gather` (all_gather_into_tensor, padded).

### Results — MIXED step (rank0 prefill 16384, rank1..7 decode 32; sum=16608)
| primitive | ms/iter | gathered rows | buffer |
|---|---|---|---|
| MAX_LEN allgather (SGL worst) | 3.974 | 131072 (7.9× pad) | 1879 MB |
| SUM_LEN allreduce (SGL better)| 1.240 | 16608 | 238 MB |
| ATOM allgatherv               | **0.877** | 16608 | 238 MB |

### Results — ALIGNED step (every rank prefill 16384; sum=131072)
| primitive | ms/iter |
|---|---|
| MAX_LEN allgather | 4.513 |
| SUM_LEN allreduce | 8.807 |
| ATOM allgatherv   | 5.248 |

### Findings (quantified gap decomposition)
1. **MIXED step is where the gap lives.** MAX_LEN (3.97ms) is **4.5× slower than
   ATOM allgatherv (0.88ms)** — the 7.9× padding blow-up. If SGLang is stuck in
   MAX_LEN on mixed steps, that alone is a huge per-step penalty.
2. **SUM_LEN closes MOST but not all of it**: 1.24ms vs ATOM 0.88ms → SUM_LEN is
   3.2× faster than MAX_LEN, but still **~1.4× slower than ATOM allgatherv** on the
   mixed step. THIS is the residual "all_reduce vs all_gatherv" primitive gap the
   user predicted — real but small (~0.36ms/layer) vs the padding penalty (~3ms).
3. **ALIGNED step: all_reduce is the WORST (8.8ms).** When every rank has equal
   big tokens, SUM_LEN all_reduce (ships 131072 rows ×~2 ring) is ~1.7× slower
   than both all_gather variants (~4.5-5.2ms). ⇒ SUM_LEN is only good for the
   UNEVEN/mixed case; for aligned big batches all_gather wins.
   This is exactly why SGLang picks SUM_LEN only when communication is cheaper
   (DpPaddingMode.get_dp_padding_mode: sum_len*2 vs max_len*dp).

### Conclusion / what it means for SGLang
- Most of the SGLang↔ATOM gap = **MIXED-step MAX_LEN padding (4.5×)**, NOT the
  all_reduce-vs-allgatherv primitive (only ~1.4× residual).
- So the high-value SGLang lever is: **avoid MAX_LEN on mixed steps**. Two routes:
  (i) the delayer (aligns ranks → fewer mixed steps; already +41% Exp 20), and
  (ii) make mixed steps use a variable-length gather instead of MAX_LEN pad.
- SGLang's existing SUM_LEN (all_reduce) already captures ~75% of the available
  win on mixed steps (3.97→1.24 vs 0.88 floor). The remaining ~1.4× needs the
  ATOM-style all_gatherv path (the flashinfer dispatcher has it; aiter MoE path
  does not — porting it = ATOM PR #930 equivalent, the real fix C).
- IMPORTANT: SUM_LEN must NOT be forced globally — it's 1.7× WORSE on aligned
  steps. SGLang's adaptive MAX_LEN/SUM_LEN choice is correct in principle; the
  problem is MAX_LEN's pad cost on mixed steps, which all_gatherv would fix.

### Next: verify which mode SGLang actually uses on our mixed steps
Add logging of `forward_batch.dp_padding_mode` (or `get_dp_padding_mode`) over a
real 8k/1k c256 run and histogram mixed vs aligned steps + chosen mode. If mixed
steps hit MAX_LEN → the 4.5× penalty is live and the delayer + a future
all_gatherv port are the fixes. Refs: dp_attention.py:53-95,463-531.
Artifact: /workspace/moe_comm_microbench.py

## Exp 24 — Padding-mode confirmation: SGLang ALWAYS uses SUM_LEN — 2026-06-12

### Method
Added env-gated log (SGLANG_DP_PADDING_MODE_DEBUG) at forward_batch_info.py:1103
printing chosen DpPaddingMode + token skew per step. Real 8k/1k c256 dp8 run,
chunk16384/rank, delayer ON. Deduped to DP0/TP0 (global decision). 54 steps.

### Result — SGLang NEVER uses MAX_LEN at this workload
| step type | MAX_LEN | SUM_LEN | total |
|---|---:|---:|---:|
| MIXED (prefill+decode skew) | 0 | 18 | 18 |
| aligned-ish extend | 0 | 36 | 36 |
| pure-decode | 0 | 0 | 0 |
| **overall** | **0** | **54** | 54 |

ALL 18 mixed steps (skew up to 951×, e.g. tmin=32 tmax=16301) chose **SUM_LEN**.
maxpad blow-up of mixed steps: p50=1.8×, max=7.6× — but since SUM_LEN is used,
that padding is NOT paid. Why: get_dp_padding_mode returns SUM_LEN whenever
`is_extend_in_batch and dp_size>1` (dp_attention.py:76-77) — and every prefill-
containing step IS extend_in_batch. So MAX_LEN is essentially never selected
for our mixed prefill steps.

### Conclusion — the 4.5× MAX_LEN penalty is NOT live; gap is the all_reduce residual
- Exp 22/23's "MAX_LEN 4.5× padding penalty" is REAL in isolation but **NOT
  occurring** in our runs — SGLang already avoids it via SUM_LEN.
- Therefore the SGLang↔ATOM MoE-comm gap that IS live = the **SUM_LEN all_reduce
  vs ATOM all_gatherv residual**, which Exp 23 measured at only **~1.4×** on the
  mixed step (1.24ms vs 0.88ms), i.e. ~0.36ms/layer. The user's original concern
  ("SUM_LEN still all_reduce, not gather") is exactly the live difference — and
  it is SMALL, not the dominant gap.
- ⇒ MoE dp-gather comm is NOT the main remaining throughput gap. The per-step
  all_reduce overhead is modest. The larger throughput gap (Exp 18: SGLang ~88%
  of ATOM at 8k/1k c256) must come from ELSEWHERE — candidates not yet isolated:
  expert GEMM efficiency / quant (Exp 19 showed ATOM's fp8 quant kernel heavy but
  trace unreliable), attention(MLA) kernel cost, or scheduler step-rate (CPU
  overhead / sync frequency), NOT the dp-gather padding.

### Net takeaways for the whole DP investigation
1. delayer: essential (+41%), keep ON. (Exp 20)
2. chunk size: must be 16384/rank but doesn't close gap. (Exp 16)
3. retract/swa: not active at this setting. (Exp 20)
4. DP-MoE gather: SGLang uses SUM_LEN (good), avoids MAX_LEN pad; residual vs
   ATOM all_gatherv is only ~1.4×/layer. (Exp 23/24)
5. ⇒ Remaining gap is NOT comm-padding. Next look at COMPUTE (expert GEMM/quant,
   MLA) per-step cost or scheduler step throughput, with a RELIABLE method
   (controlled microbench or careful per-kernel timing, not the inflated trace).

Artifacts: forward_batch_info.py +SGLANG_DP_PADDING_MODE_DEBUG log,
/workspace/run_padmode_check.sh, /workspace/server_logs/padmode_sgl.serverlog

## Exp 25 — Port ATOM all_gatherv+reduce_scatterv into SGLang (WIP, 2 bugs hit) — 2026-06-12

### Goal
Wire variable-length DP-MoE gather (ATOM PR #930 style) into SGLang's general
(aiter, non-flashinfer) dp-attention path, env-gated SGLANG_DP_USE_GATHERV.
Infra already existed: GroupCoordinator.all_gatherv / reduce_scatterv (pynccl,
parallel_state.py:813,1003); only the flashinfer fp4 path + EP combine used them.

### Implementation (env-gated, default OFF; only attn_tp_size==1, tp==dp)
- dp_attention.py: `is_dp_gatherv_active()`, `_dp_gather_via_all_gatherv()`,
  `_dp_gatherv_sizes()`; new branch in `_dp_gather` (gather) and reduce_scatterv
  in `dp_reduce_scatter_tensor`.
- communicator.py: gatherv branch in `_scatter_hidden_states` (combine,
  reduce_scatterv) + `should_use_reduce_scatter` returns True for gatherv.

### Bug 1 (FIXED): logits path size mismatch
`dp_gather_replicate` is reused by logits_processor with DIFFERENT per-rank sizes
(global_num_tokens_for_logprob, not global_num_tokens). Hardcoding
get_dp_global_num_tokens() → `assert input.shape[0]==sizes[rank]` crash at startup.
Fix: `_dp_gatherv_sizes(obj)` reads sizes from the passed ForwardBatch/
LogitsMetadata (global_num_tokens_for_logprob_cpu else global_num_tokens_cpu),
+ guard: only take gatherv path when local rows >= sizes[rank] and
sum(sizes) <= global buffer rows; else fall back to all_reduce. Startup passed.

### Bug 2 (OPEN): aiter MoE GPU fault under real load
After the logits fix, server reached READY but crashed under real c256 traffic:
"Fatal Python error: Aborted" → GPU fault inside aiter `flydsl_moe_stage1`
(aiter/ops/flydsl/moe_kernels.py:631) via watchdog. The variable-length gathered
tensor has sum_len rows (e.g. 16608 — odd, unaligned) which the aiter fused MoE
flydsl kernel cannot handle. MAX_LEN (graph-aligned max_len*dp) and SUM_LEN
(pre-sized buffer + all_reduce) both fed the kernel a "safe" shape; raw sum_len
all_gatherv does not. ATOM avoids this via its own pad path (moe.py pad_align=256,
pad_for_all_gather) — the gathered token dim must be padded/aligned before the
flydsl MoE GEMM.
⇒ A correct SGLang port must ALSO pad the gathered buffer to the aiter MoE's
required token-block alignment (then unpad on reduce_scatterv), not just swap the
collective. This is the non-trivial part ATOM's PR #930 actually solved.

### Status / result
- OFF (baseline, np1024 8k/1k c256): 24,794 tok/s, TPOT 85.5 — clean.
- ON: crashes in aiter MoE; throughput not measurable yet.
- Code left in place but DEFAULT OFF (SGLANG_DP_USE_GATHERV unset) → zero impact
  on normal runs. Needs the token-alignment fix before it can be benchmarked.

### Next step to finish the port
Add token-dim padding around the gatherv MoE path: pad each rank's local rows (or
the gathered sum_len) up to the aiter flydsl MoE block alignment before
quant_method.apply, and slice back before reduce_scatterv. Mirror ATOM
moe.py:780-793 (pad_align=256) + pad_for_all_gather. Then re-run the A/B.

Artifacts: dp_attention.py + communicator.py (gatherv path, env-gated OFF),
/workspace/run_gatherv_ab.sh, /workspace/run_gatherv_on_only.sh,
/workspace/server_logs/gatherv_*.serverlog, /workspace/bench_gatherv_ab/

## Exp 26 — gatherv Bug 2 FIXED (zero-pad buffer) + gsm8k correctness PASS — 2026-06-12

### Bug 2 root cause (M=32768 memory fault)
deepseek_v4.py:1536-1548: MoE runs on the ENTIRE global_dp_buffer
(M = global_tokens.shape[0]), not just the valid rows. MAX_LEN/SUM_LEN fill the
whole buffer; my all_gatherv filled only sum(real per-rank) → unfilled tail =
garbage → aiter flydsl MoE read OOB → "Memory access fault" (saw M=32768).

### Fix
_dp_gather_via_all_gatherv now zero-pads each rank's local tensor up to
sizes[rank] so sum(sizes) == buffer rows and every row is initialized; guard in
_dp_gather tightened to require sum(sizes) == global_tokens.shape[0] exactly
(else fall back to all_reduce). dp_attention.py only; env-gated, default OFF.

### Validation
- Diag (--disable-cuda-graph, c16 np64): **0 errors, 64/64 successful**, no fault.
- gsm8k 5-shot correctness A/B (chunk16384/rank, delayer ON):
  | | flexible-extract | strict-match | server errors |
  |---|---|---|---|
  | gatherv OFF | 0.9515 | 0.9522 | 0 |
  | gatherv ON  | 0.9507 | 0.9515 | 0 |
  Δ = 0.08% / 0.07% — within gsm8k noise (±0.59%). ⇒ **gatherv preserves
  accuracy**; the variable-length gather/scatter is functionally correct.

### Status
gatherv path now CORRECT (accuracy verified) and crash-free. Throughput A/B at
full c256 was interrupted to run the accuracy check first; rerun pending.
OFF baseline (earlier, np1024 8k/1k c256): 24,794 tok/s. Next: measure gatherv ON
throughput at same config and compare.

Artifacts: /workspace/run_gatherv_gsm8k_ab.sh, /workspace/gsm8k_gatherv/,
/workspace/run_gatherv_diag.sh (diag), dp_attention.py (zero-pad fix).

## Exp 27 — gatherv throughput A/B: NO improvement (≈0%) — 2026-06-12

### Result (8k/1k c256, np1024, ratio0.8, chunk16384/rank, delayer ON, cuda-graph ON)
| metric | OFF (SUM_LEN all_reduce) | ON (all_gatherv+reduce_scatterv) | ON vs OFF |
|---|---|---|---|
| total tok/s | 24,314 | 24,312 | **−0.0%** |
| output tok/s | 2,693 | 2,693 | −0.0% |
| median TTFT ms | 2,254 | 2,272 | +0.8% |
| median TPOT ms | 87.1 | 86.7 | −0.4% |
| p99 TPOT ms | 118.6 | 117.6 | −0.8% |
| bench dur s | 349.9 | 349.9 | +0.0% |
both 1024/1024 completed, 0 server errors.

### Conclusion — the comm-primitive swap does NOT move end-to-end throughput
- Despite the microbench (Exp 23) showing all_gatherv ~1.4× cheaper than SUM_LEN
  all_reduce on a mixed step, the FULL-MODEL throughput is identical (±0.4%).
- Why: the DP-MoE gather/scatter is only a SMALL fraction of total per-layer
  time. Saving ~0.36ms/layer on the collective is washed out by the dominant
  cost (expert GEMM, MLA attention, fp8 quant, and the rest of the comm that the
  zero-padding re-introduces — gatherv pads each rank to sizes[rank] which is the
  cuda-graph-aligned per-rank size, so the gathered M ≈ the same as SUM_LEN's
  buffer; the theoretical "no padding" win is largely given back by aligning to
  the buffer the MoE requires).
- ⇒ Porting ATOM's all_gatherv into SGLang is CORRECT (gsm8k 95.07% ≈ 95.15%,
  Exp 26) but does NOT close the SGLang↔ATOM throughput gap. The gap is NOT the
  DP-MoE collective primitive.

### Net (final) for the DP throughput investigation
The 8k/1k c256 SGLang≈88%-of-ATOM gap is NOT explained by any communication
factor we tested: not delayer (essential, Exp 20), not chunk size (Exp 16), not
retract/swa (Exp 20), not MAX_LEN padding (never used, Exp 24), not all_reduce-
vs-all_gatherv primitive (no end-to-end effect, Exp 27). Remaining suspect =
COMPUTE per-step (expert GEMM / fp8 quant / MLA) or scheduler step-rate, to be
measured with a controlled compute microbench (not the inflated full trace).

### Decision on the gatherv code
Keep env-gated (SGLANG_DP_USE_GATHERV), DEFAULT OFF — it is correct but gives no
throughput win, so it should not be enabled by default. Can be removed or left as
dormant infra. The DP-MoE comm line of investigation is CLOSED.

Artifacts: /workspace/run_gatherv_ab.sh, /workspace/bench_gatherv_ab/
(gathervOFF/ON jsonl), /workspace/run_gatherv_ab2.master.log.

## Exp 28 — gatherv TRACE verify: the gather was never the expensive collective — 2026-06-12

### Captured gatherv ON prefill trace (DP0/TP0, 16384/rank, --disable-cuda-graph)
Communication kernels in the ON trace:
| kernel | ms | count |
|---|---|---|
| quickreduce::allreduce_twoshot | 1648.9 | 244 |
| cross_device_reduce_2stage | 543.6 | 122 |
| nccl/rccl | 16.1 | 7 |
| **allgather_vec (my gatherv path)** | **0.3** | **3** |
reduce_scatter kernels: 0 named; allgather_vec: 3 (gatherv IS active).

### KEY FINDING — gatherv changes the trace, but the gather was tiny all along
- My gatherv path IS taking effect (allgather_vec present, dp_gather_partial in
  deepseek_v4.py:1540 confirmed reached for tp8dp8). But it accounts for only
  ~0.3ms — the DP attn→MoE hidden-state **gather was never the expensive comm**.
- The dominant collective (2.2s: quickreduce 1649ms + cross_device 544ms) is a
  SEPARATE per-layer all-reduce (MoE reduce_results / attention output all-reduce
  via moe_tensor_model_parallel_all_reduce, communicator.py:542), present in BOTH
  OFF and ON, untouched by my change.
- ⇒ This DEFINITIVELY explains Exp 27's 0% throughput delta: swapping the DP
  gather primitive (all_reduce→all_gatherv) optimizes a ~0.3ms collective; the
  real comm cost is the per-layer quickreduce all-reduce, which my change does
  not affect. The earlier Exp 23 microbench compared the WRONG collective for
  this code path (it modeled the full hidden-state gather as if it were the heavy
  one; in DSV4 the heavy one is the per-layer AR, not the dp-gather).

### Corrected understanding of SGLang DSV4 dp8 MoE comm
1. attn→MoE hidden gather: dp_gather_partial (all_reduce SUM_LEN, or my
   all_gatherv) — SMALL (~0.3ms over the window here).
2. per-layer MoE/attn output all-reduce: quickreduce + cross_device_2stage —
   LARGE (~2.2s), the real comm cost. This is what Exp 19 trace also flagged.
⇒ To attack SGLang comm, the target is the per-layer quickreduce all-reduce
(count × cost), NOT the dp-gather. Candidate levers: --enable-fused-moe-sum-all-
reduce, --enable-aiter-allreduce-fusion, or reducing the number of per-layer ARs.
But note Exp 18 already showed AR-count is not the throughput differentiator vs
ATOM, and ATOM pays a similar per-layer AR — so this likely won't close the gap
either; compute remains the prime suspect.

### Decision
gatherv code stays env-gated DEFAULT OFF (correct but irrelevant to throughput,
now trace-proven). DP-MoE gather investigation fully CLOSED with trace evidence.

Artifacts: /workspace/run_gatherv_trace.sh, /workspace/prof_gatherv_ON/ (8 gz),
/workspace/prof_gatherv_OFF/ (truncated — server killed early; ON is sufficient).
token-padding code: dp_attention.py `_dp_gather_via_all_gatherv` lines ~557-585.

## Exp 29 — ATOM vs SGLang per-layer MoE output reduce OP (code comparison) — 2026-06-12

### SGLang (our config): per-layer MoE/attn output = ALL_REDUCE
Trace (Exp 28): quickreduce::allreduce_twoshot 1649ms + cross_device_reduce_2stage
544ms dominate. SGLang reduces the MoE/attn output across TP via
`(moe_)tensor_model_parallel_all_reduce` (communicator.py:542, the custom
all-reduce: quickreduce / cross_device_2stage). The dp-attn hidden gather is a
separate small collective.

### ATOM (our config tp8dp8, no-EP): per-layer MoE output = REDUCE_SCATTERV
atom/model_ops/moe.py `forward_impl_graph`:
- `use_dp_gather_scatter` path (dp_size>1, no mori/EP, enable_dp_attention — i.e.
  OUR config): MoE output reduced via **`reduce_scatterv(final_hidden_states,
  sizes, dp_group)`** (line 3178, variable-length reduce-scatter) — NOT all_reduce.
- `self.reduce_results` all_reduce (line 3190) is **default False** for the
  routed FusedMoE (deepseek_v4.py:2179,2198 construct experts with
  reduce_results=False) → skipped.
- `combine_outputs` (deepseek_v4.py:2316-2328) does a `tensor_model_parallel_
  all_reduce` ONLY for the shared-expert add when tp_size>1; in the dp-attn
  flattened config the heavy routed-expert reduce is the reduce_scatterv above.

### The actual per-layer comm-OP difference
| | attn→MoE gather | routed-MoE output reduce |
|---|---|---|
| SGLang | all_reduce(SUM_LEN) / all_gatherv | **all_reduce** (quickreduce, 2-shot, full hidden replicated to every rank) |
| ATOM   | all_gatherv | **reduce_scatterv** (each rank keeps only its 1/dp slice) |

⇒ ATOM uses GATHER+SCATTER symmetric pair (all_gatherv in, reduce_scatterv out),
so each rank only ever holds/communicates its own token slice. SGLang uses
all_gather-or-allreduce in and **all_reduce out**, where all_reduce moves the FULL
hidden state (every rank ends with all tokens, ~2x ring traffic) then a separate
scatter. ATOM's reduce_scatterv is the natural inverse of all_gatherv and moves
~half the bytes of an all_reduce for the same result.

### Why my SGLang gatherv port didn't help (now fully explained)
I replaced the GATHER side (all_reduce→all_gatherv, ~0.3ms, irrelevant) but the
heavy per-layer collective is the OUTPUT **all_reduce** (2.2s), which SGLang's
MoE layer does via tensor_model_parallel_all_reduce — I did NOT replace that.
To match ATOM, SGLang's routed-MoE output would need to use **reduce_scatterv**
(reduce_results=False + a dp reduce_scatter combine), i.e. the symmetric inverse
of the gather. That is the real change; the gather swap alone is cosmetic.

### Candidate next step (if pursuing comm)
Make SGLang's DSV4 routed-MoE output use reduce_scatterv instead of all_reduce
in the dp-attn path (pair it with the all_gatherv gather already added). SGLang
HAS the EP combine reduce_scatterv (should_use_dp_reduce_scatterv) but it's gated
to ep_size==dp_size; the no-EP TP-MoE path still all_reduces. Reusing that
reduce_scatterv for the no-EP dp-attn path = the ATOM-equivalent fix. NOTE: Exp
18 suggests comm is not the throughput differentiator vs ATOM, so expected upside
is uncertain; measure before investing.

## Exp 30 — Symmetric reduce_scatterv combine: WRONG for SGLang (gsm8k 95%→43%) — 2026-06-12

### What was tried
Paired the all_gatherv gather with a reduce_scatterv combine in DSV4
deepseek_v4.py:1557 (and communicator _scatter_hidden_states), to mirror ATOM's
symmetric all_gatherv + reduce_scatterv (Exp 29).

### Result — accuracy COLLAPSED (no crash, wrong output)
gsm8k 5-shot: OFF 0.95 vs ON **0.4306** (both strict+flexible). 0 server errors.
⇒ reduce_scatterv is SEMANTICALLY WRONG for SGLang's non-EP TP-MoE path.

### Root cause — SGLang vs ATOM MoE token/expert layout differs
- ATOM (use_dp_gather_scatter, dp→tp64 flatten): experts ARE sharded across the
  flattened ranks. Each rank all_gatherv's all tokens, computes its PARTIAL
  expert contribution for all tokens, then **reduce_scatterv SUMS partials across
  ranks + scatters** each rank its slice. Sum is REQUIRED.
- SGLang (_use_tp_moe_gather, no EP): the gather replicates tokens to every rank
  and each rank computes the FULL routed-expert set for those tokens (experts NOT
  sharded the ATOM way). The correct combine is therefore a SLICE (dp_scatter) —
  the per-token result is already complete on the rank that owns it. Applying
  reduce_scatterv SUMS across ranks → each kept token is corrupted (summed with
  other ranks' full results) → garbage logits → 43% accuracy.
- reduce_scatterv is correct ONLY for SGLang's EP path
  (should_use_dp_reduce_scatterv: ep_size==dp_size, experts sharded).

### Conclusion (final on the comm line)
SGLang and ATOM's DP-MoE are NOT the same gather+scatter pair at the semantic
level: ATOM = all_gatherv + reduce_scatterv (sum, sharded experts); SGLang no-EP
= gather(all_reduce/all_gatherv) + dp_scatter(slice, replicated experts). You
CANNOT just swap SGLang's combine to reduce_scatterv — it's a different expert-
parallelization scheme. Matching ATOM would require switching SGLang to the EP /
expert-sharded MoE (--moe-a2a-backend / --ep-size), which is a much larger change
and a different config than the aligned baseline.
REVERTED the reduce_scatterv combine. Kept ONLY the all_gatherv GATHER path
(env-gated SGLANG_DP_USE_GATHERV, default OFF, paired with dp_scatter slice —
verified correct gsm8k 95.07% in Exp 26, and 0% throughput in Exp 27/28).

### Net
- The all_gatherv gather alone: correct, no throughput win (gather isn't the
  bottleneck — Exp 28).
- The reduce_scatterv combine: incorrect for SGLang's non-EP MoE — cannot adopt
  ATOM's symmetric pair without adopting ATOM's expert sharding (EP).
- DP-MoE comm line CLOSED. The SGLang↔ATOM gap is not reachable via comm-op swaps
  in the no-EP TP-MoE config. Remaining suspect = compute (expert GEMM/quant/MLA)
  or EP itself.

Files reverted to safe state: deepseek_v4.py (dp_scatter for non-EP, import
trimmed), communicator.py (no gatherv reduce_scatterv branch). dp_attention.py
all_gatherv gather retained (env-gated OFF).

## Exp 31 — CORRECTION: Exp 30 was a DOUBLE-REDUCE bug, not a semantic mismatch — 2026-06-12

### User pushback (correct): ATOM is also TP, how does it shard experts?
Re-read ATOM weight loading (moe.py:2776-2794, non-EP path):
- w13_weight: `loaded_weight[:, 2*tp_rank_start:2*tp_rank_end]` — slices the
  INTERMEDIATE dim. w2_weight: `[..., tp_rank_start//2:tp_rank_end//2]` — also
  intermediate. ⇒ ATOM TP-MoE shards each expert by INTERMEDIATE (every rank
  holds ALL experts, a 1/tp slice of each) → each rank produces a PARTIAL output
  → post-experts reduce is a SUM. reduce_scatterv (sum+scatter) is CORRECT.
- SGLang is the SAME: layer.py:215-216 `intermediate_size_per_partition =
  intermediate_size // moe_tp_size`, and moe_tp_size = tp//ep//moe_dp = 8//1//1
  = **8** (parallel_state.py:2030). So SGLang ALSO shards experts by intermediate
  across TP=8. My Exp 30 "experts not sharded" claim was WRONG.

### Real cause of Exp 30's 95%→43%: DOUBLE REDUCE
SGLang's gather→MoE→combine for TP-sharded experts is:
  gather (all_reduce/all_gatherv → all tokens on all ranks)
  → MoE computes partial (intermediate-slice) outputs
  → **MoE-INTERNAL post-experts all_reduce** sums TP partials (deepseek_v2.py:927,
     gated by should_skip_post_experts_all_reduce(use_reduce_scatter=...))
  → dp_scatter slices each rank its tokens.
Exp 30 added reduce_scatterv at the combine but LEFT the MoE-internal all_reduce
on (use_reduce_scatter stayed False) → summed TWICE → 43%.
should_skip_post_experts_all_reduce returns True iff use_reduce_scatter=True
(utils.py:455, doc: "reduce_scatter would double-reduce on top of an all-reduce").

### Fix (Exp 31)
deepseek_v4.py: add `_use_gatherv_pair = _use_tp_moe_gather and
is_dp_gatherv_active() and not max_len`. When set:
  - pass `use_reduce_scatter=True` to self.mlp → MoE SKIPS its internal
    all_reduce (no double reduce).
  - combine uses reduce_scatterv (the single sum+scatter), the symmetric inverse
    of the all_gatherv gather. = ATOM's exact scheme.

### Validation — accuracy RESTORED
gsm8k 5-shot: OFF 0.9477 / 0.9484 vs ON **0.9416 / 0.9424**. Δ ≈ −0.6% (within
±0.65% noise), 0 server errors. ⇒ symmetric all_gatherv + reduce_scatterv pair is
NOW CORRECT for SGLang's non-EP TP-MoE. (Exp 29's "different scheme" conclusion is
RETRACTED — same scheme; it was an implementation double-reduce.)

### Status
gather+scatter symmetric pair correct & env-gated (SGLANG_DP_USE_GATHERV).
Throughput A/B pending (Exp 32).

## Exp 32 — Symmetric pair throughput: +2.8% (real, small win) — 2026-06-12

### Result (8k/1k c256 np1024 ratio0.8 chunk16384/rank delayer ON cuda-graph ON)
| metric | OFF (all_reduce gather + dp_scatter) | ON (all_gatherv + reduce_scatterv) | ON vs OFF |
|---|---|---|---|
| total tok/s | 24,369 | **25,048** | **+2.8%** |
| output tok/s | 2,700 | 2,775 | +2.8% |
| median TTFT ms | 2,291 | 2,193 | −4.3% |
| median TPOT ms | 86.8 | 84.7 | −2.5% |
| bench dur s | 349.1 | 339.6 | −2.7% |
1024/1024 completed both, 0 errors. (Accuracy verified Exp 31: 94.2% ≈ 94.8%.)

### Conclusion — the SYMMETRIC pair (not the gather alone) gives the win
- Exp 27/28 (gather-only swap) = 0% because the heavy collective was the
  post-experts all_reduce, untouched.
- Exp 32 replaces BOTH: the gather (all_gatherv) AND the post-experts reduce
  (all_reduce → reduce_scatterv), skipping the MoE-internal all_reduce. Now the
  heavy per-layer all_reduce IS replaced by reduce_scatter → +2.8% total tput,
  −2.5% TPOT, −4.3% TTFT. Modest but real and consistent across metrics.
- This is exactly ATOM's scheme (all_gatherv + reduce_scatterv, skip internal AR).
  Confirms the per-layer all_reduce→reduce_scatter is a genuine (small) lever.

### Caveat / scale
+2.8% at c256 np1024. The gap vs ATOM (~12% at 8k/1k c256, Exp 18) is only
partially addressed — reduce_scatter moves ~half the bytes of all_reduce but the
collective is not the whole gap (compute remains). Worth: (a) confirm the win
holds across c64/c128/c512; (b) the change is a clean, correct, ATOM-aligned
improvement regardless — candidate to upstream behind a flag or enable for dp8.

### Decision
KEEP the symmetric-pair path. Still env-gated SGLANG_DP_USE_GATHERV (default OFF)
pending broader validation; it is correct (gsm8k) and a real +2.8%. Recommend
sweeping concurrency next, then consider making it default for tp==dp dp-attn.

Artifacts: deepseek_v4.py (_use_gatherv_pair + use_reduce_scatter skip +
reduce_scatterv combine), dp_attention.py (all_gatherv gather),
/workspace/run_gatherv_ab.sh, /workspace/bench_gatherv_ab/, run_gatherv_ab3.master.log.

## Exp 33 — Concurrency sweep: symmetric pair win GROWS with concurrency — 2026-06-12

### Result (8k/1k, ratio0.8, chunk16384/rank, delayer ON, cuda-graph ON; np=conc×8)
| conc | OFF tok/s | ON tok/s | Δtput | OFF TPOT | ON TPOT | ΔTPOT | OFF TTFT | ON TTFT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64  | 12,097 | 12,213 | +1.0% | 44.2 | 44.0 | −0.5% | 1815 | 1768 |
| 128 | 18,293 | 18,500 | +1.1% | 59.4 | 58.7 | −1.1% | 1924 | 1857 |
| 256 | 24,311 | 24,800 | +2.0% | 90.9 | 89.2 | −1.9% | 2062 | 2027 |
| 512 | 30,563 | 31,552 | **+3.2%** | 134.3 | 130.2 | −3.0% | 12999 | 12334 |
0 errors both configs, all completed. (Accuracy verified Exp 31.)

### Findings
- The symmetric all_gatherv + reduce_scatterv pair (replacing all_reduce gather +
  post-experts all_reduce) is a **consistent win at every concurrency**, and the
  win GROWS with concurrency: +1.0% (c64) → +1.1% (c128) → +2.0% (c256) → +3.2%
  (c512). TPOT improves in lockstep (−0.5%→−3.0%); TTFT also slightly lower.
- Makes sense: higher concurrency → larger batches → the per-layer collective
  moves more bytes, so halving its traffic (reduce_scatter vs all_reduce) helps
  more. This is exactly the regime where SGLang lags ATOM most (8k/1k high conc).
- Direction matches the gap shape (gap widens with conc, Exp original problem
  statement). The pair closes a real, growing slice of it — but it's a few %, not
  the full ~12%; compute remains the larger residual.

### Conclusion / recommendation
The symmetric gather+scatter (ATOM-aligned) MoE comm is correct (gsm8k) and a
real, concurrency-scaling throughput win (up to +3.2% at c512, the worst-gap
point). Recommend: enable for tp==dp dp-attention DSV4 (or upstream behind a
flag defaulting on for that config). Remaining SGLang↔ATOM gap is now
attributable mostly to COMPUTE (expert GEMM / fp8 quant / MLA), the last
un-quantified bucket.

Artifacts: /workspace/run_gatherv_sweep.sh, /workspace/bench_gatherv_sweep/
(OFF/ON jsonl c64-512), /workspace/run_gatherv_sweep.master.log.

