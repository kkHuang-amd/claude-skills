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
- **★ 2026-06-30 ROOT CAUSE FIXED (see §18, supersedes §17):** the DP+TBO high-conc
  HSA crash was caused by `op_combine`'s `record_stream(comm)` on the fresh MoE output
  deferring its free → 56GB caching-allocator reserved fragmentation → 288GB cap. Fix =
  ATOM-style **event + python-ref** (skip `record_stream`, drop ref after `wait_event`),
  commit `97ed68c27`, ~10 lines. Result: reserved 269.7→235.5GB, **mem0.9 high-conc no
  longer crashes (was HSA OOM at conc256)**; with the fix, TBO's win is realizable at mem0.9:
  **+8.7% tok/s / −14% TTFT / −6.4% TPOT vs non-TBO at conc256** (standard bench, 8k/1k),
  gsm8k 0.948 unchanged. On **long-context 70k/300 the win is even larger, +5.3%→+11.7% as
  conc grows** (prefill-dominated, §19). §17's "unfixable, just lower mem-fraction" was premature.

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

## 8.4 UPDATE 2026-06-25: DSV4 EP+TBO now IMPLEMENTED (prefill) — correct, stability bug open

§6.5/§7 estimated DSV4 TBO at 3–5wk, but that was the NON-EP path (needs new async
gather/scatter ops). The **EP/mori path is far smaller** and is now implemented &
numerically correct (gsm8k 0.9600/0.9567). Why it was tractable:
- DSV4 reuses `DeepseekV2MoE` → its `op_dispatch_a/b`/`op_combine_a/b`/`op_experts`/
  `op_gate`/`op_select_experts`/`op_shared_experts`/`op_output` (which decompose
  `forward_deepep`) are reused as-is. The mori dispatcher already implements the
  async dispatch/combine. So the expensive "MoE op decomposition" (§6 item B) was free.
- Disabling DSV4's cross-layer fused-mHC under TBO makes each layer self-contained →
  the only new model code is layer-level mHC wrap ops + a strategy + a small driver.
- TBO batch-prep is model-agnostic → no scheduler changes.
Full detail (files, bugs, op sequence, launch recipe): `EXPERIMENT_LOG_2026-06-25.md`
(TBO section). **OPEN**: intermittent `HSA_STATUS_ERROR_OUT_OF_RESOURCES` under
sustained TBO load (not VRAM, not aiter JIT; likely HSA queue/event exhaustion from
2 ubatch streams + mori's 2 inner dispatchers + decode cuda-graph). So the §8.5
"No" below is now historical — it's "Yes (prefill, EP), pending a stability fix".

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

---

## 9. UPDATE 2026-06-25/26 — DSV4 EP+TBO IMPLEMENTED; stability fixed; but perf REGRESSES (EP path)

Implemented prefill-only TBO for DSV4 on the **EP/mori** path and benchmarked it.
Full daily detail in `EXPERIMENT_LOG_2026-06-25.md` rounds 1–17.

### What was built (branch `feat/dsv4-ep-tbo-prefill`)
- `DeepseekV4DecoderLayer` op-decomposition (prefill-only): layer-level
  `op_mhc_prepare_attn` / `op_mhc_post_attn_pre_mlp` / `op_mhc_postprocess` (wrap
  hc_pre/hc_post; cross-layer fused-mHC disabled under TBO) + `MQALayer.op_attn`;
  the MoE ops (`op_gate`/`op_select_experts`/`op_dispatch_a/b`/`op_experts`/
  `op_combine_a/b`/`op_shared_experts`/`op_output`) are REUSED from `DeepseekV2MoE`
  (they decompose `forward_deepep`, the mori a2a path). Registered in `init_new_tbo`;
  driver `DeepseekV4Model._forward_layers_tbo` reuses the generic
  `execute_overlapped_operations` + filter/merge. `TboAttnBackend.__getattr__`→primary;
  `tbo_supports_cuda_graph=False` (children skip decode cuda-graph; DSV4 decode is
  non-TBO). Commit `61d6ca33`.
- **Correctness**: gsm8k 0.95–0.96 across configs (EP+TBO matches non-TBO mori).

### Stability: GPU_MAX_HW_QUEUES=5 is the fix (NOT bucketing)
- Symptom: `HSA_STATUS_ERROR_OUT_OF_RESOURCES` under sustained DSV4 EP+TBO+decode-
  cuda-graph (4–13 min). Op-stub bisection: NOT a specific subsystem (mHC-only,
  attn+MoE-without-mHC, full layer all crash; only pure passthrough stable) → ANY real
  per-shape kernel work on the TBO ubatches accumulates ROCm HSA HW-queue/signal
  resources.
- First fix tried (committed, `0b4fc8bc`): `SGLANG_TBO_PAD_BUCKET` pads tbo_padded_len
  to next pow2 → bounded ubatch shapes → stable at LOW conc. **But: a finer bucket
  (mult512) still crashed, and it did NOT scale (conc 256 still crashed at prefill
  ramp).** So bucketing is an indirect, low-conc-only workaround.
- **REAL fix (ATOM InferenceX PR #1717): `export GPU_MAX_HW_QUEUES=5`** — caps the
  ROCm HW-queue pool so streams multiplex instead of exhausting. DSV4 EP+TBO,
  **no bucketing**, `GPU_MAX_HW_QUEUES=5`, **conc 256**: full 8k/1k bench, NO crash.
  ATOM sets this (+ a bounded `--cudagraph-capture-sizes` list) exactly when it turns
  TBO on. ⇒ use `GPU_MAX_HW_QUEUES=5`; the bucket commit is superseded/optional.

### Perf: TBO REGRESSES DSV4 throughput on the EP path
8k/1k, mori EP, GPU_MAX_HW_QUEUES=5, no bucket:
| conc | EP (no TBO) tok/s | EP+TBO tok/s | Δ | TTFT EP→TBO |
|---:|---:|---:|---:|---|
| 64 | 4,504 | 4,371 | −3% | 19.5s→21.7s |
| 256 | **12,255** | 10,626 | **−13%** | 25.0s→39.8s |
ⓘ Also R1 EP 5,841 vs R1 EP+TBO 3,609 (−38%, but R1's init_new_tbo also enables
decode-TBO which regresses). DSV4 (prefill-only) regresses less but still negative.
⇒ **My prefill-TBO op-decomposition does NOT deliver the comm/compute overlap** —
prefill gets SLOWER (TTFT worse), so the op-list/2-mori-dispatcher overhead outweighs
any a2a overlap.

### ⚠️ KEY CAVEAT — we tested EP+TBO; ATOM's DSV4 win is the DP (ep=1) path
- **InferenceX PR #1717 (dsv4-fp4-mi355x-atom) runs `tp:8, ep:1, dp-attn:true,
  --enable-tbo`** — i.e. **DP-attention WITHOUT expert-parallel** (TP-MoE), TBO
  overlapping the DP **all_gather + reduce_scatter** (the no-EP path). It does NOT use
  EP/mori for the TBO rows.
- My sglang DSV4 TBO is the **EP/mori a2a** path (overlaps mori dispatch/combine) — a
  DIFFERENT overlap target. ATOM's +13–17% TBO numbers in §2 are: DP+EP-mori +16.94%
  AND DP-no-EP-all_gather +13.58% — but those are **gpt-oss-120b (PR #515)**, not DSV4.
- So it's an OPEN question whether DSV4's TBO win (if any) is on the EP path or the
  DP/no-EP path, and whether my EP+TBO −13% is (a) my impl not overlapping, or (b) the
  EP+mori path itself not benefiting DSV4 at these sizes.

### Cross-check on ATOM (DONE 2026-06-26) — the DSV4 TBO win is the DP (no-EP) path
Ran ATOM DSV4 at conc 256, 8k/1k (GPU_MAX_HW_QUEUES=5, max-num-seqs 256), via the
`bench_dsv4.py` wrapper:
| engine · path | no-TBO tok/s | +TBO tok/s | Δ | TTFT no→TBO |
|---|---:|---:|---:|---|
| **ATOM · DP (ep=1)** | 30,481 | **32,819** | **+7.7%** | 20.4s → **17.3s** (−15%) |
| sglang · EP (mori) | 12,255 | 10,626 | **−13%** | 25.0s → 39.8s (+59%) |
- **ATOM DSV4 TBO IS POSITIVE — on the DP/no-EP path (+7.7%, TTFT −15%)**, overlapping
  the DP all_gather + reduce_scatter (TP-MoE). This is exactly ATOM's InferenceX PR
  #1717 config (`tp:8, ep:1, dp-attn:true, --enable-tbo`).
- **ATOM EP/mori for DSV4 would NOT launch**: mori `Out of static heap memory`
  (requested 1.9GB vs 2GB default; `MORI_SHMEM_HEAP_SIZE` env didn't take across the DP
  engine subprocesses). Strong signal that ATOM does NOT run DSV4 on EP/mori in
  practice — it uses DP. So there is no ATOM EP+TBO DSV4 datapoint; the +16.94%
  EP-mori number in §2 is gpt-oss-120b, not DSV4.
- **Conclusion: the productive TBO target for DSV4 is the DP / no-EP all_gather+
  reduce_scatter overlap, NOT EP/mori.** My sglang impl targeted EP/mori (reusing
  DeepseekV2MoE's deepep ops) → wrong path → −13%. (Also: ATOM ≈ 2.5–3× sglang absolute
  here — partly EP/mori overhead in sglang vs DP in ATOM, partly engine.)

### TODO / next steps
1. **Re-target sglang DSV4 TBO to the non-EP (DP) path**: overlap the DP `all_gather`
   (pre-MoE gather) + `reduce_scatter` (post-MoE combine) with the other ubatch's
   attn+expert compute — i.e. async `gather_a/b` / `combine_a/b` ops on a comm stream
   (§6 item A; the larger effort I skipped). This is where ATOM's +7.7% comes from.
   The EP/mori op-decomposition (committed) is correct but not the throughput path for
   DSV4.

## 10. EVALUATION + PLAN — non-EP (DP) TBO for DSV4 (2026-06-26)

**Verdict: feasible + worth doing** — perf upside validated (ATOM DP+TBO +7.7%/TTFT
−15%), and the committed EP TBO already built most scaffolding to reuse.

**Target flow** (`deepseek_v4.py:1652-1713`, no-EP dp-attn TP-MoE):
`attn → dp_gather_partial (all_gatherv→global buf) → mlp (MoE on global) →
reduce_scatterv / dp_reduce_scatter_tensor (combine→local)`. TBO overlaps ubatch A's
gather/combine COMM with ubatch B's attn+MoE COMPUTE. Collectives are RCCL
`get_tp_group().all_gatherv` / `.reduce_scatterv` (`dp_attention.py:595,682`,
`parallel_state.py:909,1099`) — stream-based but currently SYNCHRONOUS on the compute
stream.

**Reusable from committed EP TBO (head start):** `op_mhc_prepare_attn`/
`op_mhc_post_attn_pre_mlp`/`op_mhc_postprocess`, `MQALayer.op_attn`, MoE compute ops
`op_gate`/`op_select_experts`/`op_experts`, the `_forward_layers_tbo` driver + filter/
merge. (MoE math identical; only the surrounding comm changes.)

**NEW work:**
1. Async gather/combine ops replacing mori `op_dispatch_a/b`/`op_combine_a/b`:
   `op_gather_a` (comm-stream wait_stream→all_gatherv→record_event), `op_gather_b`
   (compute waits gather event), `op_combine_a` (reduce_scatterv async), `op_combine_b`
   (wait + shared-expert-local add). Mirrors mori `_dispatch_core` `_comm_stream` +
   `_capture_event_if_async`.
2. Dedicated comm stream (reuse `CommStreamPool` / DSV4 `alt_streams`).
3. PER-UBATCH global DP buffers (`get_global_dp_buffer()` is shared → 2 ubatches clobber).
4. New no-a2a `OperationsStrategy` in `init_new_tbo`.
5. Relax guard `server_args.py:6900` (`moe_a2a_backend != "none"`) for dp-attn TP-MoE.

**Risks:** (a) BIGGEST — concurrent collectives on one RCCL communicator (A's gather +
B's combine on same `get_tp_group()` from 2 streams) may serialize/deadlock → may need
2 TP sub-communicators; validate FIRST. (b) aiter reduce_scatter on arbitrary stream +
async. (c) SUM_LEN/MAX_LEN + gatherv/reduce_scatter env interplay (no double-reduce,
correct sizes). (d) keep PREFILL-ONLY (decode non-TBO) + `GPU_MAX_HW_QUEUES=5`.

**Effort ~1–2 weeks** (vs §6's 3–5wk, since the DSV4 op framework now exists).

**Phased plan:**
1. (1d) **De-risk concurrent collectives**: microbench 2 staggered all_gatherv+
   reduce_scatterv on `get_tp_group()` from 2 streams → overlap vs serialize/deadlock.
   Gates the whole approach (1 communicator vs 2 sub-groups).
2. (2-3d) Async gather/combine primitives: stream+event split of all_gatherv/
   reduce_scatterv, per-ubatch global buffers, comm stream.
3. (1-2d) Ops + no-a2a strategy + guard relax + `_can_run_tbo` for no-EP.
4. (1d) Correctness gate: gsm8k ~0.95, no double-reduce, SUM_LEN+MAX_LEN.
5. (1d) Perf A/B: DSV4 DP vs DP+TBO @conc256 8k/1k → target ATOM's ~+7.7%.
2. **Trace** whether mori a2a dispatch/combine actually runs concurrently with the
   other ubatch's compute in our EP+TBO (profiler / nvtx) — confirm/refute "overlap
   not materializing".
3. If pursuing: try the non-EP (DP) TBO path for DSV4 (async all_gather/reduce_scatter
   ops) — that's what ATOM uses for DSV4 in production.
4. Decide whether to keep / drop the `SGLANG_TBO_PAD_BUCKET` commit (superseded by
   GPU_MAX_HW_QUEUES=5).
5. Recommended runtime today: `GPU_MAX_HW_QUEUES=5` for stability; do NOT enable TBO
   for DSV4 throughput yet (it regresses).

## 11. IMPLEMENTED — non-EP (DP) TBO for DSV4 + perf A/B (2026-06-29)

Built the §10 plan (steps 2-5; step 1 folded into the correctness gate). Single shared
comm stream (NOT 2 sub-communicators) — the feared concurrent-collective deadlock did
NOT occur (both ubatches share ONE comm stream → collectives serialize in-order, each
overlapping the other ubatch's compute).

**Code (all behind `SGLANG_ENABLE_DP_TBO=1`, opt-in):**
- `dp_attention.py`: `get_dp_tbo_comm_stream()` (lazy shared stream), `dp_gather_partial_async()`
  (wait_stream → all_gatherv on comm stream → record_event), `dp_reduce_scatterv_async()`.
- `deepseek_v4.py` `DeepseekV4DecoderLayer`: new ops `op_gather_a` (shared-expert-local +
  all_gatherv hidden + all_gatherv input_ids for hash routing, on comm stream, record event),
  `op_gather_b` (compute waits gather event), `op_moe` (`self.mlp` on GLOBAL buffer,
  `use_reduce_scatter=True`, `skip_shared_experts`), `op_combine_a` (reduce_scatterv async),
  `op_combine_b` (wait + add shared_local). `op_mhc_*`/`op_attn` reused.
- `deepseek_v4.py` `_forward_layers_tbo`: **per-ubatch DP count sync** — `tbo_padded_len` is
  computed per-rank locally (NOT synced), so all_gather both ubatches' padded lens once,
  populate each child's `global_num_tokens_cpu/gpu` + `global_dp_buffer_len` (else the
  gatherv buffer mis-sizes / falls to all_reduce on None → crash). KEY FIX.
- `operations_strategy.py`: `_compute_moe_deepseek_v4_prefill` branches on
  `get_moe_a2a_backend().is_none()` → gather/combine ops (else EP dispatch/combine ops).
- `environ.py` `SGLANG_ENABLE_DP_TBO` + `server_args.py` guard relax (a2a==none allowed
  when the env is set). Per-ubatch global buffers handled for FREE: `get_global_dp_buffer()`
  is a fresh `torch.empty` each call + executor sets `set_dp_buffer_len` per stage.

**Correctness: PASS** — gsm8k acc=0.970, invalid=0.0, no hang (DP tp8dp8, conc32).

**Perf A/B (DSV4 DP, conc256, 8k in / 1k out, mem 0.70):**
| config | chunk/rank | total tok/s | TTFT mean | TPOT |
|---|---|---|---|---|
| baseline | 1024 | **22500** | 23.1s | 77.1 |
| DP+TBO   | 1024 | 19136 (**−15%**) | 33.6s | 84.7 |
| baseline | 4096 | 16068 | 20.9s | 115.2 |
| DP+TBO   | 4096 | 16437 (**+2.3%**) | 21.1s | 112.4 |

**Findings:**
- `chunked_prefill_size // dp_size` (`server_args.py:5175`) → default 8192/8 = **1024
  tok/rank**, so TBO ubatches are ~512 tokens → GEMMs tiny, collective latency dominates →
  TBO **−15%**.
- At 4096 tok/rank (ubatch ~2048) TBO flips to a marginal **+2.3%** (overlap finally pays
  off, TPOT slightly better). BUT the large-chunk regime is itself ~28% slower than the
  optimal small-chunk baseline (long prefill steps stall decode at conc256/8k-1k, which is
  decode-bound in steady state).
- **Net: no win at this operating point** — mirrors the EP+TBO regression. ATOM's +7.7% is
  presumably a prefill-bound / different-concurrency regime, not decode-bound conc256/8k-1k.

## 12. ✅ DP+TBO IS A WIN once config + bench are aligned (2026-06-29)

Earlier §11's "−15% / no win" was a **config + benchmark artifact**, NOT the algorithm.
After aligning to our real DP-best config (`run_sgl_dsv4_aligned.sh` DP path) AND using the
SAME sweep client as ATOM (`sweep_dsv4_sglang_client.sh`, `--backend sglang-oai`,
`--warmup-requests`, np=conc×8), DP+TBO clearly helps.

**Aligned A/B (DSV4 DP, conc256, 8192:1024, 8192/rank, mem0.80, only diff = TBO flag):**
| metric | DP baseline | DP+TBO | Δ |
|---|---:|---:|---:|
| total tok/s | 28,140 | **30,888** | **+9.8%** |
| output tok/s | 3,127 | 3,432 | +9.8% |
| mean TTFT | 11,224 ms | 10,090 ms | **−10.1%** |
| mean TPOT | 70.2 ms | 64.2 ms | **−8.5%** |
→ **matches/exceeds ATOM's DP+TBO +7.7% / TTFT −15%.** Correctness: gsm8k quick 0.970;
lm_eval gsm8k flexible 0.960 / strict 0.945.

**What was wrong before (§11):** (1) mori-derived launch (`--moe-dense-tp-size 1`,
`--enable-dp-lm-head` — those are EP-only; wrong kernels; `AITER_BF16_FP8_MOE_BOUND`,
`SGLANG_HACK_FLASHMLA_BACKEND` mismatched); (2) hand-rolled `bench_serving --backend sglang`
instead of the `sglang-oai` sweep client w/ warmups (ATOM used the same client → comparable);
(3) tiny **1024/rank** chunk (ubatches ~512 → no overlap). Fixes: align env to
`run_sgl_dsv4_aligned.sh`, use the sweep client, **8192/rank** (`--chunked-prefill-size
65536`; note sglang divides by dp_size).

**Stability boundary (still open) — both chunk- AND concurrency-dependent:** DP+TBO is
stable at **8192/rank, conc256**; it hits `HSA_STATUS_ERROR_OUT_OF_RESOURCES` at
(a) **16384/rank** (any conc, ~18 forwards) and (b) **8192/rank conc512**. noTBO is fine in
both. → the 2-stream resource accumulation scales with BOTH prefill-chunk size and
concurrency. bucket extends survival; buffer/event/input_ids-once reuse didn't fix.

conc512 baseline (no crash): total **34,339 tok/s**, TTFT 16.0s, TPOT 115ms (np4096). TBO
@conc512 crashed → no number. So the TBO win is demonstrated at conc256 (+9.8%); higher
conc / larger chunk need the HSA crash fixed.

## 13. ATOM+TBO does NOT need GPU_MAX_HW_QUEUES=5 — sglang's DP+TBO does (2026-06-29)

Decisive cross-engine test at **conc256, 8192:1024, 16384/rank (ATOM per-rank
max-num-batched-tokens 16384 = sglang chunk 131072), cg1024 (dense capture list MATCHED
to sglang's 68-size set), maxrun1024, mem0.9**, ATOM's own bench client:

| engine · config | GPU_MAX_HW_QUEUES | total tok/s | crash? |
|---|---|---:|---|
| **ATOM+TBO** | **unset (default)** | **34,016** | **NO** (full 2048-prompt sweep clean) |
| ATOM+TBO | =5 (PR #1717) | 29,786 | no (but cap throttles −12%) |
| sglang baseline (no TBO, 8192/rank) | unset | 29,974 | no |
| sglang+TBO (8192/rank) | =5 | crashes >conc256 / >8192-rank | HSA |
| sglang+TBO (16384/rank) | =5 | — | HSA (~18 fwd) |

**Conclusions:**
- **The HSA `OUT_OF_RESOURCES` crash is NOT fundamental to TBO / multi-stream.** ATOM's
  generic thread + dual-stream TBO runs the heavy config (16384/rank, cg1024, maxrun1024,
  conc256) with NO cap and NO crash → so it's **sglang's DP+TBO implementation** that leaks
  / over-allocates HSA resources (comm-stream + per-ubatch gather), needing the cap as a
  band-aid — and even then sglang crashes at higher conc/chunk where ATOM is fine.
- **`GPU_MAX_HW_QUEUES=5` is a throttle, not free.** It cost ATOM ~12% (34,016→29,786). PR
  #1717 sets it presumably for the higher-conc points (512–2048) it sweeps, not needed at
  conc256.
- **ATOM+TBO (uncapped) 34,016 BEATS sglang's no-TBO baseline (29,974) by +13%** at this
  config — and sglang+TBO can't even run here. So the real gap to close is **sglang's TBO
  resource management** (match ATOM's lean HSA footprint so it can drop the cap and scale).

→ Next lever for sglang DP+TBO: find/fix the HSA resource over-allocation (likely the
dedicated comm stream + per-ubatch buffers creating too many HSA signals/queues vs ATOM's
single shared dual-stream) so it can run uncapped at the heavy config.

## 14. ROOT CAUSE of sglang DP+TBO HSA crash = VRAM headroom, NOT signals/JIT (2026-06-29)

Live monitor (`/workspace/hsa_monitor2.sh`: rocm-smi VRAM + per-sched-pid
eventfd/kfd fd counts + cumulative aiter LoadKernel) during a crashing run
(DP+TBO, conc256, 8192/rank, maxrun512/cg512, **mem0.80**, full 2048-prompt sweep):

```
13:50:40  VRAM 276.9GB  eventfd=113 kfd=8 LoadKernel=70   (steady)
13:50:50  VRAM 287.9GB  eventfd=113 kfd=8 LoadKernel=70   <- transient spike hits 288 cap
13:51:12  HSA_STATUS_ERROR_OUT_OF_RESOURCES "Available Free mem : 0 MB" -> crash
```

**eventfd / kfd / LoadKernel are FLAT the entire run.** So the crash is NOT HSA
signal/event leakage and NOT hsaco/JIT module accumulation (both earlier hypotheses
were WRONG). It is pure **VRAM exhaustion**: TBO's steady state already sits at
276.9/288 GB (only ~11 GB free), and a single forward's transient peak (the 2
in-flight ubatches' global gather buffers + fused_moe intermediates) spikes to
287.9 GB and hits the cap -> `free 0 MB` -> the HSA error is just the surface symptom.
(My stream/event/buffer are all pooled, so they are NOT the accumulation source.)

**FIX (confirmed): lower `--mem-fraction-static`** so the KV pool shrinks and leaves
headroom for the TBO transient. mem0.80 -> **mem0.72**:
```
mem0.80: steady 276.9 -> peak 287.9 (cap) -> CRASH
mem0.72: steady 263.4 -> peak 281.1 (7GB free) -> NO HSA error, full 2048-prompt
         sweep COMPLETES.  29,383 tok/s, TTFT 6.4s, TPOT 71.3ms, 2048/2048 ok.
```
→ **DP+TBO now runs conc256 stably at mem<=0.72.** (A late non-HSA teardown crash
after the sweep finished — NCCL "Broken pipe", exit -3 — is a separate benign-looking
race, not the OOM.)

**Why ATOM doesn't need this:** ATOM uses vLLM-style memory accounting (profile the
activation/transient peak, then size KV = budget - peak - safety), so the TBO
transient is reserved up front. sglang's `mem-fraction-static` statically splits
weights+KV and throws the transient at whatever is left -> spikes hit the cap.

**FAIR conc256 A/B at mem0.72 (both arms identical; cg512/maxrun512/8192-rank, ATOM
sweep client, 2048 prompts, only diff = TBO flag):**
| metric | baseline | DP+TBO | Δ |
|---|---:|---:|---:|
| total tok/s | 27,293 | **29,330** | **+7.5%** |
| output tok/s | 3,033 | 3,259 | +7.4% |
| mean TTFT | 6,539 ms | 6,000 ms | **−8.2%** |
| mean TPOT | 77.0 ms | 71.7 ms | **−6.9%** |
→ **DP+TBO = +7.5% throughput / −8% TTFT / −7% TPOT, matching ATOM's +7.7%.** Both
arms stable (2048/2048, no HSA). This is the trustworthy number (earlier §12 +9.8%
had a queue-cap-handicapped baseline). Note mem0.72 baseline (27.3k) is ~9% below the
mem0.90 baseline (30.0k) — lowering mem for TBO headroom costs the KV pool, so the
absolute ceiling is lower until the TBO transient peak is reduced (option 2/3 below).

**Baseline config-knob isolation (conc256, 8192/rank, no TBO) — CORRECTED:**
| mem | cg-max-bs | max-running | total tok/s | TTFT | note |
|---|---|---|---:|---:|---|
| 0.90 | 1024 | 1024 | 29,974 | 15.9s | run #1 — **OUTLIER, did not reproduce** |
| 0.90 | 1024 | 1024 | 28,513 | 15.9s | run #2 (rerun) |
| 0.90 | 512  | 512  | 28,540 | 16.0s | |
| 0.72 | 512  | 512  | 27,293 | 6.5s | |
**CORRECTION 2 (root cause = SGLANG_USE_ROCM700A, NOT variance):** the 29,974-vs-28,513
gap was NOT variance and NOT a script-param diff. It was a **`SGLANG_USE_ROCM700A`
env-pollution bug in the test harness**: the persistent shell session had
`SGLANG_USE_ROCM700A=1` exported; the `aligned` runs passed `SGLANG_USE_ROCM700A=0`
explicitly on the command line (→ fast), while `unified` runs (whose script uses
`${SGLANG_USE_ROCM700A:-0}`) inherited the session's `=1` (→ slow). Isolation:
| script | mem | cg/maxrun | ROCM700A | tok/s |
|---|---|---|---|---:|
| unified | 0.90 | 512/512 | **1** | 28,540 |
| unified | 0.90 | 512/512 | **0** | **29,996** |
| aligned | 0.90 | 1024/1024 | 0 | 30,051 / 29,974 |
→ **`ROCM700A=0` = +5.1%** (28,540→29,996), independent of cg/maxrun (512 with
ROCM700A=0 already hits 30k). Matches the known HANDOFF result (ROCM700A=0 wins via
aiter MAX_LEN decode kernels). **cg-max-bs/max-running 1024-vs-512 = no real
difference at conc256.** Real baseline mem0.90 = **~30,000** (with ROCM700A=0).

⚠️ **IMPLICATION for the §14 fair A/B:** both arms (baseline 27,293 + TBO 29,330) ran
via unified with the session-polluted **ROCM700A=1**, so the +7.5% relative gain holds
(identical condition) but both absolute numbers are ~5% low. Re-run the fair A/B with
**ROCM700A=0** (both arms, mem0.72) for correct absolutes.

**KEY (corrected) takeaway:** DP+TBO @mem0.72 = **29,330** already ≈/slightly EXCEEDS
the real baseline @mem0.90 = **28,513** (+2.9%), AND beats the same-mem baseline
(27,293) by +7.5%. So lowering mem for TBO headroom does NOT sacrifice the absolute
ceiling — TBO at mem0.72 is still the fastest DSV4 conc256 config measured. The mem
drop is effectively free here because mem0.90 was barely faster than mem0.72 for the
baseline.

## 15. MEASURED: ATOM TBO VRAM is FLAT (no transient spike) — confirms §14 (2026-06-30)

Ran the same VRAM monitor on **ATOM TBO** (conc256, 8192:1024, 16384/rank, mem0.9,
max-num-seqs 512, cudagraph dense→512, uncapped) — the apples-to-apples counterpart to
the crashing sglang TBO run:

| engine·TBO | idle | steady VRAM | transient spike | result |
|---|---:|---:|---|---|
| sglang (mem0.8) | 222 | 277 | **287.9 → hits 288 cap** | CRASH |
| **ATOM (mem0.9)** | 256.9 | **275.4** | **NONE (perfectly flat)** | OK, 2048/2048, 34,382 tok/s |

ATOM's VRAM sat at a **dead-flat 275.4GB** for the entire 2048-prompt sweep — eventfd/
kfd/LoadKernel all constant, zero transient excursion. Even with a LARGER KV pool
(mem0.9, idle 256.9 vs sglang's 222), ATOM's steady 275.4 is 12GB below sglang's spike
287.9, and crucially has **no per-forward spike at all**.

→ **Direct confirmation of the §14 root cause:** ATOM uses vLLM-style memory accounting
(profile the activation/transient peak, then size KV = budget − peak − safety), so the
TBO transient is pre-reserved and the running VRAM is flat. sglang's `mem-fraction-static`
statically splits weights+KV and throws the TBO transient at the leftover, so a single
forward's 2-ubatch gather/MoE transient spikes ~+11GB and tips over the 288 cap. This is
exactly why sglang DP+TBO must lower mem-fraction (option 3 is the proper fix).

## 16. ROOT CAUSE of the TBO VRAM spike = caching-allocator RESERVED fragmentation (2026-06-30)

Op-level probe (`SGLANG_TBO_MEM_PROBE`, prints stage when `reserved` hits a new high)
during DP+TBO conc256 8192/rank mem0.80:

```
alloc (live tensor):  FLAT ~206GB the whole run
reserved:             monotonically 250 -> 266 -> 274 -> hits 288 -> HSA crash
stage pushing it:     always the MoE stage (gather_b,moe,combine_a) + gather_a/attn
```

→ The spike is **NOT more live tensors (B's "2x ubatch" premise is WRONG — alloc is
flat 206GB)**. It is the **PyTorch/HIP caching allocator's `reserved` growing via
fragmentation**: each prefill batch has a different token count + TBO ubatch split
varies, so the big tensors (DP global gather buffer / fused_moe gate-up intermediates /
expert_out) take a NEW shape almost every forward → allocator opens a new segment, old
segments fragment and can't be reused → `reserved` climbs monotonically to the 288 cap.
(ROCm has no `expandable_segments`, so segments can't be coalesced.)

**Fixes tried:**
| attempt | reserved behavior | result |
|---|---|---|
| `PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512` | still climbs to 271 | CRASH (GC doesn't reclaim these segments) |
| `SGLANG_TBO_PAD_BUCKET=1` (per-rank pow2 bucket of ubatch len) | climbs slower, plateaus ~266 then creeps to 274 | CRASH (delayed not fixed) |

Why per-rank bucket only delays: it quantizes each rank's ubatch len, but **the GLOBAL
MoE shape = sum of per-rank lens across the DP group; ranks have different token counts
so the global sum still varies** → fused_moe still sees new shapes → reserved still
creeps up.

**Route-2 (uniform pow2 bucket) ATTEMPTED + FAILED (2026-06-30):** implemented
uniform shape bucket — all ranks + both ubatches pad to one shared `pow2(max ubatch
len)`, global = bucket×world (deepseek_v4.py: sync block + op_combine_a/b unpad).
Correctness OK (gsm8k 0.980). **But reserved still climbed to 273.68 and crashed**
(same as no-bucket 273.82). Why it didn't converge:
1. `bucket = pow2(max ubatch len)` still varies per forward (prefill batch sizes range
   small→8192/rank → bucket spans 256…8192, ~6-7 distinct global shapes); reserved
   grows to the largest bucket's footprint.
2. Even with global token count fixed, **fused_moe has OTHER routing-dependent shapes**
   (sorted-tokens-per-expert, group-gemm intermediates) + the **attn stage** has
   variable prefill seq-len intermediates — both push reserved (probe shows gather_a/
   attn stage also hits new maxes).
→ Fixing only the global token count is insufficient; there are too many variable-shape
sources for "fix the shape" to bound reserved on ROCm (no expandable_segments to
coalesce). Reverting route-2 (adds pow2 padding waste for no benefit).

**Proper fix = make the GLOBAL MoE shape land on a tiny fixed set:**
- **(A) Uniform shape bucket**: all ranks + both ubatches pad to ONE shared len
  (max-reduce across DP, round to pow2) → global = uniform_len×dp ∈ {a few values} →
  fused_moe shapes bounded → reserved converges. Needs changing the gatherv/
  reduce_scatterv `sizes` to be uniform + per-rank zero-pad to uniform_len, then
  un-pad after combine (correctness-sensitive).
- **(B) Lower mem-fraction (working workaround)**: mem<=0.72 leaves enough headroom that
  reserved's plateau stays under 288 (validated §14: 2048/2048, 29,330 tok/s).
- **(C) ATOM-style peak-aware KV sizing**: size KV below the reserved high-water (ATOM
  warms up at max prefill TBO shape so reserved tops out before KV sizing; §15).

## 17. CONCLUSION: reserved fragmentation is not fixable via allocator/shape tricks on
## ROCm; lower mem-fraction is the working solution (2026-06-30)

Exhaustive attempts to stop the prefill-TBO `reserved` growth (alloc flat ~206-232GB,
reserved climbs to ~274 → 288 cap → HSA crash), ALL FAILED:
| attempt | result |
|---|---|
| `PYTORCH_HIP_ALLOC_CONF` GC threshold + max_split_size_mb | reserved still → 271, crash |
| `SGLANG_TBO_PAD_BUCKET` (per-rank pow2) | delayed, → 274, crash |
| Route-2 uniform pow2 bucket (global shape fixed) | gsm8k 0.980 but reserved → 273.68, crash |
| `torch.cuda.empty_cache()` every TBO forward | reserved still → 274.28, crash |

**Why nothing works:** ROCm PyTorch has **no `expandable_segments`**, so fragmented
segments (live blocks — persistent TBO buffers + KV pool + in-flight fused_moe
intermediates — interleaved with freed blocks of varied shapes) **cannot be coalesced
or reclaimed**. `empty_cache` only frees segments with ZERO live blocks; the mixed
segments stay. Fixing the global MoE shape isn't enough either: fused_moe has
routing-dependent intermediate shapes and attn has variable prefill seq-len shapes, so
new segments keep appearing regardless.

**WORKING SOLUTION: lower `--mem-fraction-static` (≤0.72) for DP+TBO.** This shrinks the
KV pool so `weights + KV + reserved_ceiling(~274)` stays under 288. Validated §14:
conc256 2048/2048, **+7.5% throughput / −8% TTFT / −7% TPOT vs same-mem baseline**.
Cost: smaller KV pool than ATOM's peak-aware 75GB (sglang mem0.72 KV ~30GB), so the
absolute ceiling is lower than ATOM — but the TBO relative win holds.

Diagnostic code (mem probe, empty_cache, route-2 uniform bucket) was reverted; the
committed DP+TBO feature (52792aeec) is unchanged. The proper long-term fix would be
framework-level: ATOM-style peak-aware KV sizing AFTER a max-shape TBO warmup (needs
reordering sglang init: warmup→measure→size KV), which is out of scope here.

**Remaining options to raise mem-fraction back / scale to higher conc:**
1. (workaround, working) cap mem-fraction <=0.72 for DP+TBO.
2. (better) reduce the TBO transient peak: the 2 ubatches' global gather buffers +
   fused_moe intermediates coexist; chunk the global MoE, or stagger so only one
   ubatch holds its global buffer at a time.
3. (root) adopt vLLM-style peak-aware KV sizing for the TBO path.

**Earlier-section levers (historical):** (a) prefill-bound / large-batch throughput regime
(short output, or pure-prefill bench) where overlap dominates; (b) profile to confirm the
gather/combine actually overlap compute (nvtx) and aren't serialized; (c) tune
`tbo_delta_stages` (currently 0) / op staggering; (d) lower concurrency where prefill is a
larger fraction. Implementation is correct + opt-in; safe to leave disabled by default.

---

## 18. ✅ ROOT CAUSE FOUND & FIXED — `record_stream(comm)` deferred-free fragmentation (2026-06-30, SUPERSEDES §17)

**§17's conclusion ("reserved fragmentation is unfixable on ROCm, just lower mem-fraction")
was WRONG — it gave up before finding the real cause.** With sharper tools (in-process
torch memory probe + CUDA memory snapshot + `device_traces` + an ATOM A/B) we located the
true root cause and **fixed it with a ~10-line change**. Result: **mem0.9 + conc512 no
longer crashes, throughput +21.8%, gsm8k unchanged (0.948).**

### 18.1 Root cause
`DeepseekV4DecoderLayer.op_combine_a` launches the DP `reduce_scatterv` on the shared comm
stream and called `global_tokens.record_stream(comm)` on the **fresh MoE output**
(`global_expert_out`, ~512MB/layer). `record_stream` defers that tensor's free until the
**entire comm stream drains** (not a single event). Under TBO's large prefill chunks the
comm stream lags compute, so the deferred large block stays pinned and **forces every
subsequent allocation — including aiter `fused_moe`'s internal ~2.6GB workspace — to open a
NEW segment instead of reusing**. Result: 56GB of reserved-but-unused segments (60 idle
segments: 12×2.6GB + 30×512MB) → peak hits the ~288GB physical cap → `HSA_STATUS_ERROR_
OUT_OF_RESOURCES (free 0MB)` at high concurrency. **`record_stream` is a fragmentation
amplifier**, not just a 512MB leak.

This finally explains why §16/§17's allocator/shape tricks all failed: they targeted shape
variety / `empty_cache` / bucketing, but the live-set was pinned by `record_stream`, which
none of them touch.

### 18.2 The fix — ATOM-style event + python-ref (commit `97ed68c27`)
ATOM's entire codebase has **zero `record_stream`**; it manages cross-stream tensor lifetime
with `torch.Event` + `wait_event` + a python ref. We ported exactly that to the combine:
- `dp_attention.dp_reduce_scatterv_async(..., record_input=False)` → skip
  `record_stream(comm)` on the MoE output.
- `op_combine_a`: keep `state.combine_global_out = global_out` (python ref alive).
- `op_combine_b`: after `wait_event(combine_event)` (reduce_scatterv provably done), drop
  the ref → the block frees **promptly on the compute stream**, no deferral.

Race-free: the ref is dropped only after the compute stream has waited the combine event, so
the comm-stream `reduce_scatterv` has finished reading `global_out`. 2 files, +22/-2 lines.

### 18.3 Memory (the fix), measured (DP+TBO, DSV4-Pro, 8x MI355X, 8192/1024)
| metric | `record_stream` (pre-fix) | `event+ref` (fix `97ed68c27`) |
|---|---|---|
| reserved (mem0.80) | 269.7 GB | **235.5 GB** |
| fragmentation (reserved − max_alloc) | 56 GB | **22 GB** (< noTBO's 28.8 GB) |
| peak VRAM (mem0.80, conc256/512) | 287.9 (cap → crash) | **251.3** (37 GB headroom) |
| **mem0.9 conc256** | **CRASH (HSA, peak 287.2)** | **✅ no crash, peak 278** |
| gsm8k 5-shot exact_match | — | **0.948** (correctness preserved) |
| KV pool @ mem0.9 | 2.21M tok (~55GB) | same (big KV preserved) |

The pre-fix DP+TBO **cannot run at best-perf mem0.9** (crashes at conc256); the fix makes it
stable, so the TBO throughput win below is actually realizable at mem0.9.

### 18.3b TBO throughput win — non-TBO vs DP+TBO, STANDARD bench settings (2026-06-30)
**Bench config (this is the canonical one):** `num-prompts = conc×8`, `warmup = conc×2`,
`--random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 --request-rate inf`,
mem0.9. (An earlier run used `num-prompts = conc×2, warmup 0` — too few requests / no warmup,
so those numbers were unreliable and are superseded.)

| conc256, mem0.9 | non-TBO (`MODE=dp`) | **DP+TBO (`MODE=dp-tbo`, fix)** | TBO impact |
|---|---|---:|---:|
| Total throughput | 29,985 tok/s | **32,592 tok/s** | **+8.7%** |
| Mean TTFT | 16,705 ms | **14,363 ms** | **−14.0%** |
| Mean TPOT | 59.97 ms | **56.14 ms** | **−6.4%** |
| duration (2048 prompts) | 629 s | 579 s | −8% |

| conc512, mem0.9 | non-TBO (`MODE=dp`) | **DP+TBO (`MODE=dp-tbo`)** | TBO impact |
|---|---|---:|---:|
| Total throughput | 38,187 tok/s | **41,180 tok/s** | **+7.8%** |
| Mean TTFT | 39,219 ms | **34,777 ms** | **−11.3%** |
| Mean TPOT | 77.96 ms | **73.33 ms** | **−5.9%** |

Consistent with §12's ~+7.5%. **TBO win holds across concurrency: +8.7% @conc256, +7.8%
@conc512** (8k/1k decode-heavy); both stable at mem0.9 with the record_stream fix (no crash).
Long-context 70k/300 is larger still (§19: +5.3% → +11.7%).

**Net: TBO itself gives +8.7% tok/s / −14% TTFT at conc256; the record_stream→event+ref fix is
what lets you actually collect it at best-perf mem0.9 (pre-fix it crashes). conc256 speed is
unchanged by the fix vs the pre-fix TBO where the latter can run.** Earlier note (now corrected):
the "38,129 / +21.8%" figure came from the unreliable conc×2 bench and is superseded by
the standard-settings numbers in the table above.

### 18.4 HOW WE FOUND IT — methodology (a chain of falsified hypotheses)
The decisive lesson: **measure, don't guess.** Every "obvious" hypothesis was killed by data;
the real cause only fell out of allocator-level snapshots + an ATOM A/B.

| # | hypothesis | how tested | verdict |
|---|---|---|---|
| 1 | peak-aware KV sizing (shrink KV → headroom) | compared sglang vs ATOM KV-pool sizes from startup logs | ❌ sglang KV (23GB) already << ATOM (75GB); not the problem |
| 2 | TBO holds 2× activation buffers | op-level `max_memory_allocated` delta vs noTBO | ❌ TBO live delta (9.7GB) **< noTBO (17GB)** — no 2× |
| 3 | reserved grows unboundedly (leak) | rocm-smi trajectory conc256 vs conc512 | ❌ reserved **converges** (stable 287.9), not a runaway |
| 4 | shape variety → unbounded HSA/JIT kernels | `LoadKernel`/eventfd/kfd counts vs ATOM | ❌ all bounded & ≈ ATOM (LoadKernel 68–80) |
| 5 | warmup max-shape TBO prefill pre-reserves | manual max-shape warmup then sweep | ❌ peak identical with/without warmup |
| 6 | `expandable_segments` / `max_split_size_mb` | env on ROCm | ❌ expandable unsupported on ROCm; max_split crashed sooner |
| 7 | comm-stream lags many layers (bound depth) | per-layer `compute.wait_stream(comm)` gate | ❌ reserved unchanged (wait_stream ≠ allocator free) |
| 8 | **`record_stream` deferred-free** | **remove it → event+ref** | ✅ **reserved 269→235, conc512 stops crashing** |

**Tools that cracked it (in order):**
1. **In-process `RT_MEM` probe** (env-gated, in `Scheduler.run_batch`): logged
   `memory_allocated` / `memory_reserved` / `max_memory_allocated` per forward. Split the
   60GB runtime growth into **live activation ~8GB / reserved fragmentation ~56GB /
   non-torch (HSA/NCCL) ~18GB** → proved fragmentation, not live tensors, was the cost.
2. **noTBO vs TBO A/B** with the same probe: identical `max_alloc`, but reserved 242 (noTBO)
   vs 269.7 (TBO) → isolated **+27.5GB as TBO-specific** fragmentation.
3. **CUDA memory snapshot** (`torch.cuda.memory._record_memory_history` +
   `_dump_snapshot`, env-gated, dumped at fwd 60): showed **64 fully-idle segments = 60.9GB**,
   dominated by **12×2.6GB + 30×512MB** repeated same-size allocations → "alloc/free same
   size repeatedly, allocator opens new segments instead of reusing".
4. **`device_traces`** from the snapshot: the 2.6GB tensor was alloc'd **1201×** and 512MB
   **3234×** in 60 forwards → confirmed high-frequency churn (python frames stopped at the
   forward entry → tensors alloc'd inside aiter C++/torch).
5. **ATOM source A/B** (subagent trace of `atom/utils/tbo/`): ATOM uses the **same aiter
   fused_moe kernel** but has **zero `record_stream`** (event+ref) + CPU ping-pong bounding
   pipeline depth → flat allocator. This pinpointed the `record_stream` vs event+ref
   difference as the lever, given the MoE kernel is identical.

All diagnostic code (RT_MEM probe, MEM_SNAP dump, bound-depth gate) was reverted; only the
event+ref fix (`97ed68c27`) remains. Probe/snapshot scripts kept under `/workspace`
(`parse_snap*.py`, `partb_vram_monitor.sh`) for future reuse.

### 18.5 Follow-ups / TBO TODO
- [ ] **Check whether mori-ep + TBO needs the same record_stream→event+ref fix.**
  `MoriDispatcher` (`token_dispatcher/moriep.py`) `record_stream(comm_stream)`s its
  dispatch/combine **outputs** — `packed_recv_hidden` (line ~697-704) and
  `combined_hidden_states` (line ~797) — the SAME deferred-free pattern as the DP path we
  fixed. **Whether it fragments depends on one unknown:** are those `mori_op.dispatch/combine`
  return buffers (a) mori symmetric-memory preallocated + reused → record_stream is harmless,
  NO fix needed; or (b) fresh torch allocations per layer → fragments, DO apply event+ref.
  Can't tell from the sglang layer — must check the mori library's dispatch/combine output
  allocation, OR measure directly: run `MODE=mori-ep-tbo` at high conc with the same RT_MEM
  probe and see if `reserved` balloons toward the 288GB cap like the DP path did.
  **Priority: LOW** — §9 showed EP+TBO *regresses* DSV4 throughput (the DSV4 TBO win is the
  DP/ep=1 path, now fixed), so mori-ep-tbo is not the recommended path and its fix has little
  practical payoff. Fix only if we later care about the EP path.
- [x] `op_gather_a` symmetric event+ref cleanup — DONE (origin `efbe53a` uses
  `gather_keepalive`: gather input/output dropped after `op_gather_b` waits the gather event,
  no `record_stream` left on the DP gather path).
- [x] DeepseekV2 TBO ops checked (2026-07-01): **no DSV4-style fresh-output
  `record_stream(comm)` defer**. TBO combine (`op_combine_a/b`) delegates to the EP dispatcher,
  so any `record_stream` lives there (covered by the mori-ep item above). The only
  `record_stream` in `deepseek_v2.py` (line ~1224) is the shared-expert `alt_stream` dual-stream
  overlap — short-lived and immediately `record_event`-synced, not a comm-defer — so nothing to
  fix on the DeepseekV2 side.
- [ ] With 37GB headroom at mem0.80, mem-fraction can be pushed to ~0.90 for a 2.2M-token KV
  pool; conc512 peak 283 leaves ~5GB — fine, but conc>512 may want ~0.88.
- [x] conc512 non-TBO vs DP+TBO under STANDARD bench settings (2026-07-01): **+7.8% tok/s
  (38,187 → 41,180), TTFT −11.3%, TPOT −5.9%, both stable at mem0.9, no crash** (see §18.3b).

---

## 19. DP+TBO win is LARGER on long-context 70k/300 (2026-07-01)

non-TBO vs DP+TBO A/B on the 70k/300 long-context sweep (DeepSeek-V4-Pro, 8x MI355X,
`tp8dp8 CHUNK=16384 SWA=0.1 MEM=0.80 DELAYER=off`; TBO via `--enable-two-batch-overlap` +
`GPU_MAX_HW_QUEUES=5`, no env; client NP_MULT=4 WARM_MULT=1). Both sides stable (no HSA crash).

| conc | baseline tok/s | +TBO tok/s | throughput | TTFT |
|---:|---:|---:|---:|---:|
| 8  | 29,126 | 30,661 | **+5.3%**  | −10.1% |
| 16 | 36,698 | 40,376 | **+10.0%** | −11.7% |
| 32 | 42,035 | 46,935 | **+11.7%** | −13.0% |

**Key takeaways:**
- TBO's throughput win **grows with concurrency (+5.3% → +11.7%)** and is **larger than the
  8k/1k decode-heavy case** (§18.3b: +8.7% @conc256). Reason: 70k/300 is heavily
  prefill-dominated (70000 in / 300 out), and TBO overlaps the DP gather/combine comm with
  the other ubatch's attn+MoE compute — a bigger prefill fraction ⇒ more overlap.
- TTFT drops 10–13% (prefill latency benefits directly).
- The record_stream→event+ref fix (§18) keeps mem0.80 + long-context KV (3.88M tokens) stable
  under TBO; no crash at any conc.
- baseline matches TODO_70k300.md's best-of (29.2k/36.8k/42.1k), so the A/B baseline is the
  tuned config, not a strawman.
- **Practical guidance: DP+TBO is most valuable on prefill-dominated / long-context serving;
  enable it there for the biggest win.**
