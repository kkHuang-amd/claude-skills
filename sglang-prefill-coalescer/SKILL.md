---
name: sglang-prefill-coalescer
description: >-
  Port ATOM's DP-attention prefill coalescer (ROCm/ATOM#1611) into SGLang's scheduler and A/B
  it on DeepSeek-V4-Pro (8xMI355X, TBO+DPA dp8). Use when working on SGLang's PrefillDelayer /
  get_new_batch_prefill, when reproducing ATOM's prefill-coalescing throughput gain, or when
  hitting the gotchas: the gain only shows on FRAGMENTED (variable-length) DP workloads
  (range-ratio<1), gsm8k needs max_tokens>=8192 for DSV4 reasoning, DSV4 MoE cuda-graph needs
  the run_sgl_dsv4_unified.sh env vars, and running ATOM in-place needs an aiter/flydsl/triton
  upgrade cascade.
---

# SGLang prefill coalescer (faithful ATOM port) — DeepSeek-V4-Pro, MI355X

Playbook + record for porting ATOM's cross-DP prefill **coalescer**
(ROCm/ATOM#1611, "Nagle's algorithm for prefill") into SGLang's DP-attention
scheduler, and confirming its throughput impact in the same environment
(8xMI355X gfx950, DeepSeek-V4-Pro, TP8 + DP-attention + TBO).

**TL;DR result:** the coalescer works and the SGLang port reproduces ATOM's
gain, **but only on fragmented / uneven-length DP workloads**. On uniform
same-length simultaneous arrival it is neutral (prefill already batches to a
full forward, nothing to coalesce). See `RESULTS.md`. Full chronological
problem log in `PROBLEMS.md`.

- SGLang change: branch `feat/prefill-coalescer-sum-fill-target`, single commit
  `aa9547b98` "feat: prefill coalescer for DP-attention (faithful ATOM port)"
  in `/sgl-workspace/sglang-upstream` (5 files).
- Reference: ATOM `atom/model_engine/prefill_delayer.py` (PR #1611, merge
  `4b2b574b`); installed ATOM wheel `0.1.4.dev335+gc5601741f`.

## 1. What it does

Under DP-attention each DP rank schedules independently; left alone, ranks fire
many prefill forwards that each carry only a few tokens (short fresh prompt or a
small chunked-prefill tail). Every prefill forward has ~fixed cost (kernel
launch, pad-to-shape, the lockstep MoE all-to-all), so a nearly-empty forward
wastes most of it. The coalescer **holds prefill admission until the accumulated
prefill is worth a forward, then releases**, keeping DP ranks phase-aligned so
they enter prefill together (MoE collective stays aligned). Decode keeps running
during the hold; TTFT is bounded so a held request never starves.

## 2. Design (tick-based FIRE/HOLD state machine)

`PrefillDelayer.should_allow_prefill(...)` is called **every scheduler tick on
every DP rank in lockstep**, does one cross-DP `all_gather` of a 7-field per-rank
vector (`prefillable, pending_tokens, running_decode, kv_high, kv_low,
has_partial, queue_hot`), reduces to global state, then FIRE/HOLD:

```
if n_prefillable == 0:                              FIRE   # vacuous
# must-fire bounds:
if G_running_dec == 0:                              FIRE   # no decode to hide the wait
if any_kv_high or any_kv_low:                       FIRE   # KV pressure / starvation
if any_queue_hot (oldest wait >= max_queue_ms):     FIRE   # end-to-end TTFT SLA
if hold_ticks >= ttft_max_ticks:                    FIRE   # single-hold TTFT bound
if any_partial and hold_ticks >= partial_max_ticks: FIRE
# alignment gate (anti-skew, the main throughput lever):
if n_prefillable < dp_size:                         HOLD
# fill target:
if SUM(pending) >= target_fill * n_prefillable * max_prefill_tokens:  FIRE
# stall give-up:
if SUM(pending) stopped growing for stall_ticks:    FIRE
HOLD
```

Key: **SUM (not MAX)** aggregation so quiet ranks keep accumulating during a
hold; **alignment gate** is what cuts the wasted DP idle forwards.

## 3. Files changed (SGLang)

- `python/sglang/srt/managers/prefill_delayer.py` — rewritten as the ATOM state
  machine (was the old slot/queue-ratio negotiation). Uses SGLang's proven
  `all_gather_into_tensor` over the tp cpu/nccl group; each DP rank represented
  by its attn-tp rank 0 (`buffer[:, 0, :]`), then SUM/OR reduce.
- `python/sglang/srt/managers/scheduler.py` — `get_new_batch_prefill` computes
  the 6 local inputs and calls `should_allow_prefill` ONCE at the top, every
  tick (before any early return → preserves lockstep). Veto → return None
  (fall through to decode). `_prefill_delayer_inputs()` helper added.
- `python/sglang/srt/managers/schedule_policy.py` — removed the per-request
  `PrefillDelayerSinglePassExecutor` checks in `PrefillAdder` (decision now made
  once per tick in the scheduler).
- `python/sglang/srt/server_args.py` — new flags (below).
- `test/registered/scheduler/test_prefill_delayer.py` — unit tests rewritten for
  the new state machine (world_size=4 gloo: alignment gate, fill target,
  nodecode/kv/queue-guard must-fire, stall give-up, vacuous, first-fire).

## 4. Server flags

```
--enable-prefill-delayer                       # on
--prefill-delayer-target-fill 0.7              # fill target fraction (opt-in coalescer)
--prefill-delayer-ttft-max-ticks 30
--prefill-delayer-partial-max-ticks 8
--prefill-delayer-stall-ticks 3
--prefill-delayer-kv-high-watermark 0.9
--prefill-delayer-token-usage-low-watermark <f>  # kv-low must-fire (optional)
--prefill-delayer-max-queue-ms <ms>              # end-to-end TTFT guard (optional)
```

## 5. How to launch + A/B (same env used to validate)

DSV4 needs the tuned env from `useful-scripts/benchmarking/dsv4/run_sgl_dsv4_unified.sh`
(SGLANG_USE_ROCM700A=0, SGLANG_USE_AITER=1, AITER_BF16_FP8_MOE_BOUND=0,
SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton, ...). Without them, MoE
cuda-graph capture fails ("Unsupported kernel config for moe heuristic
dispatch"). See PROBLEMS.md #3.

```bash
# coalescer ON (dp8 + TBO)
MODE=dp-tbo DELAYER=on \
SGL_EXTRA_ARGS="--prefill-delayer-target-fill 0.7 --enable-metrics" \
bash useful-scripts/benchmarking/dsv4/run_sgl_dsv4_unified.sh
# OFF: MODE=dp-tbo DELAYER=off SGL_EXTRA_ARGS="--enable-metrics" bash ...

# client — MUST use a fragmented workload to see the gain (range-ratio < 1):
python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 8000 \
  --model /shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.3 \
  --num-prompts 2048 --max-concurrency 1024 --request-rate inf
```

Confirm the state machine is firing via `/metrics`:
`sglang:prefill_delayer_outcomes_total{output_reason="fill|stall|ttft|nodecode|delay|vacuous",...}`.

## 6. Headline numbers (see RESULTS.md for full tables)

DeepSeek-V4-Pro, 8xMI355X, TP8+DPA+TBO, 1k/1k, c1024, request-rate inf:

| workload | OFF tok/s | ON tok/s | Δ |
| --- | --- | --- | --- |
| uniform (range-ratio 1.0)     | 29,524 | 29,256 | −0.9% (neutral) |
| fragmented (range-ratio 0.3)  | 15,747 | 20,571 | **+30.6%** |

ATOM's own coalescer on the identical fragmented point: 15,236 → 20,588
(**+35.1%**); uniform: −0.9%. gsm8k (full 1319, max_tokens 8192):
ON 0.939 vs OFF 0.931 (no regression).

## 7. Most important gotcha

**The coalescer only helps on fragmented / variable-length DP workloads.** If
you benchmark with `--random-range-ratio 1.0` (equal-length prompts arriving
together) you get ~0% (or slightly negative from the per-tick all_gather cost),
because prefill already fills a full forward and there is nothing to coalesce
and no cross-rank skew. This wasted a lot of time early on — always A/B with
`range-ratio` 0.3–0.8. See PROBLEMS.md.
