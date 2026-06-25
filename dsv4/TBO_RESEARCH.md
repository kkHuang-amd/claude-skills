# TBO research — ATOM `--enable-tbo` vs sglang `--enable-two-batch-overlap`

Researched 2026-06-23. Goal: understand the two TBO implementations, why sglang's
requires an EP a2a backend while ATOM's does not, what ATOM overlaps in the no-EP
DP-attention case, and the feasibility/effort of adding TBO to sglang's non-EP
TP-MoE DP-attention path (the DSV4 tp8dp8 config).

Code references are against:
- sglang: `/sgl-workspace/sglang-upstream/python/sglang/srt`
- ATOM: `/opt/venv/lib/python3.10/site-packages/atom`
- ATOM PR #515 "Support TBO in ATOM" (uploaded snapshot).

---

## 1. TL;DR

- **TBO = Two-Batch Overlap**: split a batch into 2 micro-batches (ubatches) and
  overlap one ubatch's MoE *communication* with the other's *compute*.
- **sglang TBO requires an EP a2a backend** (`deepep`/`mooncake`/`mori`/`nixl`);
  it hard-errors with `moe_a2a_backend == "none"`. The overlap primitive is the
  a2a token dispatcher's async split (`dispatch_a/b`, `combine_a/b`), which only
  EP dispatchers implement.
- **ATOM TBO does not require EP**: it is a generic thread + dual-stream
  micro-batch framework. With only `--enable-dp-attention` (no EP) it overlaps the
  **DP `all_gather` (pre-MoE gather) + `reduce_scatter` (post-MoE combine)**
  communication — the same TP-MoE collectives we optimize for DSV4.
- ATOM data: TBO helps **prefill only** (e.g. +13.6% total tok/s @512BS no-EP);
  TBO for decode (`--enable-tbo all`) **regresses** (-12~14%).
- Adding non-EP TBO to sglang is feasible but: cheap-ish for DeepseekV2 (op
  framework already exists), **expensive for DSV4** (custom monolithic forward not
  in the op framework → needs a risky refactor).

---

## 2. ATOM TBO (PR #515)

### Modes / flags
- `--enable-dp-attention --enable-expert-parallel --enable-tbo` → DP + EP(mori) +
  TBO; overlaps mori EP dispatch/combine.
- `--enable-dp-attention --enable-tbo` (NO EP) → DP + `all_gather`/`reduce_scatter`
  + TBO; overlaps the DP-attention TP-MoE gather/combine.
- `--enable-tbo` = prefill only (default); `--enable-tbo all` = prefill + decode
  (regresses, not recommended).

### Reported results (gpt-oss-120b, from the PR)
| config | BS | total tok/s | TBO impact |
|---|---:|---:|---:|
| DP + EP mori, prefill-only TBO | 256 | 28,411 | +3.99% |
| DP + EP mori, prefill-only TBO | 512 | 36,134 | +16.94% |
| DP + all_gather/reduce_scatter, prefill-only TBO | 256 | 30,759 | +2.08% |
| **DP + all_gather/reduce_scatter, prefill-only TBO** | **512** | **39,889** | **+13.58%** |
| TBO `all` (prefill+decode) | any | — | **regresses -12~14%** |

→ The no-EP path proves TBO overlaps the **DP gather/scatter comm**, and the win
is **prefill-only**.

### ATOM execution model — threads + shared dual stream (NOT 4 streams)
Files: `atom/utils/tbo/{ubatch_wrapper,ubatching,ubatch_splitting}.py`.

- 2 ubatches run in **2 Python threads** (+1 main): `ready_barrier = Barrier(3)`
  (`ubatch_wrapper.py:50`).
- **Only 2 CUDA streams total, shared by both threads**: one `comm_stream`
  (`torch.cuda.Stream()`, `ubatch_wrapper.py:56`) + one `compute_stream`
  (`= torch.cuda.current_stream()`, `ubatch_wrapper.py:74`). `make_tbo_contexts`
  passes the *same* pair to every ubatch context (`ubatching.py:456-467`).
- Threads **ping-pong cooperatively** via a ring of CPU events
  (`cpu_wait_event`/`cpu_signal_event`), and use per-ubatch GPU events
  (`gpu_comm_done` / `gpu_compute_done`) for cross-stream ordering.
- Overlap happens at the **GPU stream level**: while thread A enqueues ubatch-A's
  gather/scatter on `comm_stream`, it yields so thread B enqueues ubatch-B's
  attention/MoE GEMM on `compute_stream`; both streams run concurrently on the GPU.

So: **2 ubatches, 2 threads, 2 shared streams** (1 comm + 1 compute) — not 4.

**Model-agnostic**: `_ubatch_thread` runs the **unmodified** `self.model(ub_input_ids,
ub_positions)` inside the ubatch context (`ubatch_wrapper.py:150`). The model's
normal forward is reused as-is; the yield/stream-switch happens *inside* the
TBO-aware comm primitives (the gather/scatter or mori dispatch call the
`tbo_yield`/`switch_to_comm` helpers in `ubatching.py`). So enabling a new model
needs only its MoE comm path wired to the TBO hooks (+ per-ubatch attn metadata),
NOT a forward rewrite. (ATOM's PR only added small per-model edits: deepseek_v2
"disable dual-stream under TBO", moe.py MORI async wiring, attn backends'
per-ubatch metadata.)

---

## 3. sglang TBO

### Hard requirement: a2a backend != none
```python
# server_args.py:7809
if self.enable_two_batch_overlap and self.moe_a2a_backend == "none":
    raise ValueError("When enabling two batch overlap, moe_a2a_backend cannot be 'none'.")
```

### Mechanism — operations-based single-thread coroutine interleave (NOT threads)
Files: `srt/batch_overlap/{two_batch_overlap,operations,operations_strategy}.py`.

- Each decoder layer's forward is decomposed into a flat list of `Operation`s with
  `YieldOperation()` markers (`operations_strategy.py`).
- `_model_forward_tbo` → `execute_overlapped_operations(... delta_stages=[0, tbo_delta_stages])`
  runs 2 ubatches' op sequences **staggered** by `tbo_delta_stages`; at each
  `YieldOperation` it switches ubatch.
- The overlap primitive is the **a2a dispatcher's async split**: `op_dispatch_a`
  launches the async all-to-all on the dispatcher's comm stream, then yields;
  the other ubatch runs compute ops; later `op_dispatch_b` waits.
  (`deepseek_v2.py:1429-1465` op_dispatch_a/b, op_combine_a/b, op_experts.)

DeepseekV2 prefill op order (`operations_strategy.py:104-122`):
```
op_comm_prepare_attn, op_prepare, op_core, op_comm_prepare_mlp,
op_gate, op_select_experts, op_dispatch_a, YIELD, op_dispatch_b, op_experts,
op_combine_a, YIELD, op_shared_experts, op_combine_b, op_output, op_comm_postprocess_layer
```

### Why it needs EP
`MaybeTboDeepEPDispatcher` (`two_batch_overlap.py:1054`) only builds inner
dispatchers for `deepep`/`mooncake`/`mori`/`nixl`. With `a2a=none` (TP-MoE) there
is no async-split dispatcher, so `op_dispatch_a/b` / `op_combine_a/b` don't exist
→ nothing to overlap. The TP-MoE `all_gather`+`reduce_scatter` is done in the
layer/communicator as *synchronous* single calls, not as a/b async halves.

### Per-model, hardcoded decomposition
`OperationsStrategy.init_new_tbo` (`operations_strategy.py:33-67`) only supports
`DeepseekV2DecoderLayer`, `Qwen3MoeDecoderLayer`, `MiMoV2DecoderLayer`; else
`raise NotImplementedError`. **DSV4 `DeepseekV4DecoderLayer` is not included**, and
its forward is a custom monolithic method (MHC pre/post fusion, SE-local, the
gatherv branch) with **no op_ decomposition** at all.

---

## 4. Side-by-side

| | sglang `--enable-two-batch-overlap` | ATOM `--enable-tbo` |
|---|---|---|
| Execution model | operations list + coroutine interleave (1 thread) | 2 threads ping-pong |
| Streams | dispatcher's comm stream + default | 2 shared (1 comm + 1 compute) |
| Overlap primitive | a2a dispatcher `dispatch_a/b`, `combine_a/b` | generic ubatch on comm/compute stream |
| Needs EP/DeepEP? | **Yes** (a2a != none, hard error) | No |
| No-EP (pure dp-attn) supported? | No | **Yes** |
| No-EP overlap target | (unsupported) | **DP all_gather + reduce_scatter** |
| Model coverage | DeepseekV2 / Qwen3Moe / MiMoV2 (hardcoded) | generic wrapper |
| prefill vs decode | both (per strategy) | **prefill helps; decode regresses** |

---

## 5. What ATOM overlaps in the no-EP DP-attention case

The **DP-attention TP-MoE collectives**: before MoE each DP rank `all_gather`s the
global token set; after MoE it `reduce_scatter`s back. These are exactly the
`aiter allgather_vec` (gather) + `reduce_scatter`/`cross_device_reduce` (combine)
kernels we studied for DSV4. TBO splits the batch into 2 ubatches and hides
ubatch-A's gather/scatter comm behind ubatch-B's attention + expert-GEMM compute
(and vice versa). It is **not** EP dispatch/combine (no EP in this mode).

Relation to our work: our `reduce_scatter` PoC makes that comm **cheaper**; ATOM
TBO **hides** it behind compute. Complementary levers.

---

## 6. Feasibility & effort — adding non-EP TP-MoE TBO to sglang

### Work breakdown
| # | Item | Notes | Size |
|---|---|---|---|
| A | Async-split `all_gather` / `reduce_scatter` | new `gather_a/b`, `combine_a/b` on a dedicated comm stream + events. pynccl is stream-based (OK); verify aiter custom kernels (allgather_vec / reduce_scatter) run on arbitrary stream + event order | Medium + correctness risk |
| B | op decomposition + new OperationsStrategy for the no-a2a path | add op_gather_a/b, op_combine_a/b; a strategy variant without the dispatcher ops | DeepseekV2: Medium; **DSV4: Large** |
| C | Relax `moe_a2a_backend != none` guard + register layer in `init_new_tbo` | server_args.py:7809 + operations_strategy.py:33 | Small |
| D | ubatch split / attn metadata | `TboForwardBatchPreparer` / `TboAttnBackend` are generic (split by token/seq) → reusable | Small |
| E | decode CUDA-graph capture | `TboCudaGraphRunnerPlugin` exists, but multi-stream + comm-stream collectives must be capture-safe (finicky on ROCm/RCCL) | Medium |
| F | Correctness | reduce semantics, padding, SUM_LEN/MAX_LEN, no double-reduce; gsm8k | Medium |

### Effort by target
- **DeepseekV2 (R1/V3) only**: the op framework already exists (DeepseekV2DecoderLayer
  is already decomposed). Main new work = A + a new strategy + relax guard.
  **~1–2 weeks** to a working prototype.
- **DSV4 (our target)**: **much larger.** DSV4's custom monolithic forward must
  first be refactored into the op_ framework (preserving MHC fusion, SE-local, the
  gatherv path), then add async gather/scatter ops. **~3–5 weeks + heavy
  validation**, high refactor risk.

### Risks
- DSV4 custom-forward op-refactor is the dominant cost/risk.
- Multi-stream + decode CUDA graph on ROCm/RCCL is fragile.
- TBO needs batches large enough to split into 2 ubatches; small batches no benefit.
- ATOM data: TBO is **prefill-only positive**; decode regresses.

---

## 6.5 Per-model (sglang current) vs generic rewrite (ATOM-style) — trade-off

A natural follow-up: instead of paying the per-model op-decomposition cost, should
we refactor sglang's TBO into an ATOM-style **generic** framework (thread-based,
transparent forward, yields inside comm primitives)? Assessment: **larger and
riskier, and it does NOT buy more overlap** — it only moves the cost from
"per-model op decomposition" to "whole-runtime thread-safety".

### Why a generic rewrite is bigger
It is a paradigm change (cross-cutting), not an incremental feature:
| Item | Notes | Risk |
|---|---|---|
| Rewrite exec engine | `execute_overlapped_operations` (op-list coroutine) → thread-based `UBatchWrapper` | Medium |
| Move yield points | from the explicit op-list into the comm primitives (gather/scatter/dispatch call yield/stream-switch) | Medium |
| **Make whole forward thread-safe** | sglang relies on global/contextual state (`get_forward_context`, `zero_allocator`, dp buffers, attn-backend state, singletons); all must become thread-local/safe (ATOM made `forward_context` thread-local for this) | **High — dominant cost** |
| Threaded CUDA-graph capture | need a thread-based capture path (decode), fragile on ROCm | High |
| Re-validate existing TBO | DeepseekV2 / Qwen3 / MiMoV2 + all EP backends must be re-verified so shipped EP TBO doesn't regress | High |

By contrast, adding non-EP TBO in the **current** framework is local (op
decomposition + new comm ops + a strategy), without touching runtime
thread-safety.

### Key point: threads do NOT overlap more than op-lists
Both achieve the *same* effect — interleave 2 ubatches so comm hides behind
compute at the **GPU-stream** level:
- ATOM uses **thread ping-pong** (cooperative via CPU events; only one thread
  launches at a time — NOT true CPU parallelism) purely so it can reuse the
  **unmodified** forward.
- sglang reaches the same interleave **without threads** (op-list coroutine).

So sglang's op-list is a *deliberate* trade-off: deterministic kernel order
(clean CUDA-graph capture) + no thread-safety hazards, at the cost of per-model op
decomposition. Switching to threads *adds* thread-safety + capture complexity for
**no extra overlap**.

### Cost nature
| Route | Cost nature | Best when |
|---|---|---|
| Current per-model | pay op-decomposition once per model (DSV4 is expensive) | TBO only a few models |
| ATOM-style generic rewrite | one-time big investment (thread-safety + capture + regression); new models then ~free (just wire comm hooks) | want many models to get TBO cheaply, long-term |

- **For DSV4 only**: the per-model route is *smaller and lower-risk* (the rewrite's
  thread-safety cost outweighs DSV4's op-refactor cost).
- **For long-term N models**: a generic rewrite amortizes, but fights sglang's
  capture-friendly design and concentrates risk in thread-safety.

### Pragmatic middle option (best ROI to reduce per-model cost)
Stay in the op-list paradigm but add a **generic non-EP gather/scatter async op +
an auto-decomposition helper**. Keeps deterministic capture + no thread-safety,
while cutting per-model boilerplate (models don't each hand-write a full op
strategy). Much smaller than a thread-based rewrite, and partially achieves the
"generic" goal.

---

## 7. Recommendation (ROI order)

1. **Do not start with DSV4 TBO.** ATOM TBO only wins prefill; our prefill is
   already ~98% of ATOM (post CK-GEMM/RoPE); the c512 gap is TTFT/scheduling, not
   raw prefill compute; DSV4 forward refactor is the highest cost/risk.
2. **Quick upper-bound measurement**: enable EP (`--ep-size 8 --moe-a2a-backend
   deepep`) on DSV4 so sglang's existing TBO runs — measures how much TBO could
   buy, at the cost of breaking apple-to-apple with ATOM's no-EP config.
3. **If piloting non-EP TBO**, do it on **DeepseekV2 (R1)** first (op framework
   ready, ~1–2 wks). Only pay the DSV4 refactor cost if the R1 pilot shows a real
   win.
4. **Harvest the cheap, complementary lever now**: the `reduce_scatter` PoC
   (`SGLANG_DP_USE_REDUCE_SCATTER`, branch `feat/dsv4-aiter-reduce-scatter-decode`) already
   makes the comm cheaper (+3.07% @1k/1k c512, gsm8k 0.9507) at low risk.

---

## 8.5 Status: can sglang DeepSeek-V4 enable TBO today? (No) — checked 2026-06-24

**No.** `deepseek_v4.py` has zero TBO wiring (grep for `model_forward_maybe_tbo`,
`op_dispatch_a`, `op_gate`, `tbo`, `init_new_tbo` → 0 hits). Three independent gates
block it:

1. **DSV4 forward never enters the TBO engine.** `DeepseekV4Model.forward` is a
   plain layer loop and does **not** call `model_forward_maybe_tbo`. Setting
   `--enable-two-batch-overlap` does nothing for DSV4 (no-op / wasted flag).
2. **No op_ decomposition.** `DeepseekV4DecoderLayer` is a custom monolithic
   forward (MHC fusion + SE-local + gatherv); it has none of the `op_gate` /
   `op_dispatch_a/b` / `op_core` methods the op-list engine needs.
3. **`init_new_tbo` rejects it.** It keys on `layers[0].__class__.__name__` and only
   handles `DeepseekV2DecoderLayer` / `Qwen3MoeDecoderLayer` / `MiMoV2DecoderLayer`;
   `DeepseekV4DecoderLayer` falls to `else: raise NotImplementedError`
   (`operations_strategy.py:33-67`).
4. (Also) even with a supported layer, `server_args.py:7809` requires
   `moe_a2a_backend != "none"`, so the no-EP tp8dp8 config is blocked regardless.

**To enable DSV4 TBO** = the §6 work: refactor `DeepseekV4DecoderLayer` into the op_
framework + add a strategy branch in `init_new_tbo` + (for no-EP) add the non-EP
gather/scatter async ops + relax the a2a guard. ≈ **3–5 weeks, high risk**.

**DeepseekV2 (R1/V3) can** already TBO (layer is op-decomposed, `init_new_tbo`
supports it) — just needs an EP a2a backend (deepep/…).

### Per-model vs generic (recap of the 2026-06-24 Q&A)
- **sglang TBO is per-model**, with *two* layers of per-model work: (a) the layer
  must expose `op_` methods (split its forward into discrete ops), and (b)
  `init_new_tbo` needs a hardcoded `OperationsStrategy` listing those ops + yields.
- **ATOM is more generic**: `UBatchWrapper` runs the **unmodified** model forward in
  threads; the yield/stream-switch lives *inside* the TBO-aware comm primitives. A
  new model needs only its comm path wired to the TBO hooks (+ per-ubatch attn
  metadata), not a forward rewrite. (See §2 model-agnostic note, §6.5 trade-off.)

---

## 8. Key code pointers
- sglang guard: `srt/server_args.py:7809`
- sglang TBO entry: `srt/batch_overlap/two_batch_overlap.py` (`model_forward_maybe_tbo`,
  `_model_forward_tbo`, `MaybeTboDeepEPDispatcher:1054`, `TboForwardBatchPreparer:484`)
- sglang op strategy: `srt/batch_overlap/operations_strategy.py` (`init_new_tbo:33`)
- sglang DeepseekV2 ops: `srt/models/deepseek_v2.py` (`op_dispatch_a:1429` … `op_output:1465`,
  `op_comm_prepare_attn:2204`)
- ATOM TBO: `atom/utils/tbo/{ubatch_wrapper,ubatching,ubatch_splitting}.py`;
  ATOM PR #515.
