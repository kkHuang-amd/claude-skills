# DeepSeek-V4-Pro serving perf — experiment log (2026-06-24)

Continuation of 2026-06-23 (aiter reduce_scatter decode combine). This day:
refactor the PoC into a shippable form (env rename + platform-conditional default
+ gatherv SUM_LEN gating + helper routing), validate no regression, file PR #29103,
and write the TBO research / trace-profiling topic docs.

Active clone: `/sgl-workspace/sglang-upstream/python` (HIP/MI355X). Branch
`feat/dsv4-aiter-reduce-scatter-decode`. See `TRACE_PROFILING.md`, `TBO_RESEARCH.md`.

---

## 1 — Refactor: gate gatherv on SUM_LEN + route decode combine via helper

The 06-23 PoC called `get_tp_group().reduce_scatter_tensor(...)` directly in the
MAX_LEN decode combine, on purpose — because `dp_reduce_scatter_tensor()` checks
`is_dp_gatherv_active()` (env+layout only, padding-mode-independent) and, with
`SGLANG_DP_USE_GATHERV=1`, would route to the variable-length RCCL `reduce_scatterv`
instead of the aiter equal-chunk path.

Fix (cleaner, more correct): add a SUM_LEN requirement to `is_dp_gatherv_active()`.
The gatherv pair (`all_gatherv` + `reduce_scatterv`) is **only valid under SUM_LEN**;
under MAX_LEN the buffer is equal-padded and the combine should be equal-chunk
reduce_scatter. Implemented via the global mirror `_DpGatheredBufferWrapper.
is_dp_max_padding()` (set by `set_dp_buffer_len` from `dp_padding_mode.is_max_len()`)
— **no signature change**, so callers without a `ForwardBatch` (i.e.
`dp_reduce_scatter_tensor`) become consistent.

```python
# dp_attention.py
def is_dp_gatherv_active() -> bool:
    return (
        _USE_DP_GATHERV
        and get_attention_tp_size() == 1
        and get_tensor_model_parallel_world_size() == get_attention_dp_size()
        and not _DpGatheredBufferWrapper.is_dp_max_padding()   # NEW: require SUM_LEN
    )
```

Now the MAX_LEN decode combine can call `dp_reduce_scatter_tensor(...)` (gatherv
inactive under MAX_LEN → falls to equal-chunk `reduce_scatter_tensor` → aiter).
`deepseek_v4.py` switched to the helper; locals renamed for clarity
(`_use_gatherv_pair` → `_use_reduce_scatterv`, `_use_aiter_rs` →
`_use_reduce_scatter`).

Correctness (this refactor): gsm8k flexible **0.9447** / strict **0.9454** (within
the ~±0.006 run-to-run band, correct).

Notes / corrections:
- Earlier worry that `dp_reduce_scatter_tensor` would assert-fail in MAX_LEN was
  WRONG: in MAX_LEN `forward_batch_info` rewrites `global_num_tokens` to
  `[max_num_tokens]*world` (equal), so `sum(sizes) == buffer rows` and reduce_scatterv
  would pass — it was just RCCL (slower), not a crash.
- Kept the per-batch `not is_max_len()` checks in `_use_gatherv_pair` / `_dp_gather`
  (redundant now, but use the authoritative per-ForwardBatch value + None-guard; the
  global mirror is the fallback for callers lacking a ForwardBatch).

## 2 — Env rename + platform-conditional default

`SGLANG_USE_AITER_RS` → **`SGLANG_DP_USE_REDUCE_SCATTER`** (platform-agnostic: the
reduce_scatter combine works everywhere — aiter custom kernel on ROCm, RCCL
elsewhere). Default is now **platform-conditional: ON for HIP, OFF otherwise**, via:
- a lazy `_default_hip()` (lru_cache; imports torch only on first use, so no torch
  import at environ load — environ stays stdlib-only),
- generic **callable-default** support in `EnvField.get()` (`_resolve_default()`;
  backward compatible — non-callable defaults unchanged).

```python
# environ.py
SGLANG_DP_USE_REDUCE_SCATTER = EnvBool(_default_hip)   # HIP -> True, else False
```
Verified: default True on this HIP box, `=0`/`=1` override works, non-HIP → False.

## 3 — No-regression validation (v3, same-session A/B)

c512, np4096/warm1024, ratio1.0, tp8dp8, ROCM700A=0, gatherv ON, SE-local; default
reduce_scatter ON. Dirs `dsv4_bench/v3_perf/{8k,1k}` vs `aiter_rs_poc*/baseline`.

| workload | baseline (RS off) | v3 (RS on) | Δ |
|---|---:|---:|---:|
| 8k/1k | 37,171 | **37,701** | +1.4% |
| 1k/1k | 18,026 | **18,666** | +3.5% |

gsm8k (default on): flexible **0.9545** / strict **0.9553**. No regression; matches
the 06-23 PoC direction (8k +0.72%, 1k +3.07% there; v3 ≥ both PoC and baseline).

## 4 — Analysis Q&A (clarifications)

- **`SGLANG_DP_USE_GATHERV=1` is NOT useless under `ROCM700A=0`.** Prefill is always
  SUM_LEN (extend; DSV4 prefill cuda graph disabled) → gatherv ACTIVE in prefill
  (the shipped PR #28216 win). Only DECODE under ROCM700A=0 is MAX_LEN → gatherv
  inactive. So: **gatherv handles prefill (SUM_LEN), reduce_scatter handles decode
  (MAX_LEN)** — complementary; neither phase pays full all_reduce traffic.
- **Padding-mode source table**:

  | ROCM700A | prefill | decode (cuda graph) |
  |---|---|---|
  | 0 | SUM_LEN (gatherv ✅) | MAX_LEN (reduce_scatter ✅, gatherv ❌) |
  | 1 | SUM_LEN (gatherv ✅) | SUM_LEN (gatherv ✅) |

## 5 — TBO research (see TBO_RESEARCH.md)

ATOM `--enable-tbo` vs sglang `--enable-two-batch-overlap`:
- sglang TBO **requires an EP a2a backend** (`server_args.py:7809` errors on
  `moe_a2a_backend == "none"`); the overlap primitive is the a2a dispatcher's async
  `dispatch_a/b` + `combine_a/b` (EP-only). It is operations-based (op-list coroutine
  interleave), per-model hardcoded (`init_new_tbo`: DeepseekV2 / Qwen3Moe / MiMoV2).
- ATOM TBO needs no EP: a generic thread + dual-stream `UBatchWrapper` runs the
  **unmodified** forward in 2 threads sharing 1 comm + 1 compute stream (NOT 4); in
  the no-EP dp-attention case it overlaps the **DP all_gather + reduce_scatter** comm.
  TBO helps **prefill only** (decode regresses).
- **DSV4 cannot enable TBO today**: its forward never calls `model_forward_maybe_tbo`,
  has no `op_` decomposition, and `init_new_tbo` rejects `DeepseekV4DecoderLayer`.
  Adding non-EP TBO to sglang: DeepseekV2 ~1–2wk, DSV4 ~3–5wk (custom-forward op
  refactor); an ATOM-style generic rewrite is larger (whole-runtime thread-safety)
  and buys no extra overlap. Full breakdown + recommendation in `TBO_RESEARCH.md`.

## 6 — Housekeeping

- Commit `3f622dbd6` (rename + SUM_LEN gating; 4 files) on the branch (not pushed
  locally; repo auto-sync pushed equivalents). `cohere2_moe.py` `@strict` no-op
  workaround kept uncommitted (reverted by sync; re-apply before launch).
- **PR #29103** (`sgl-project/sglang`) description + test commands filled (via REST
  PATCH; old `gh pr edit` hits the Projects-classic GraphQL bug). Net diff vs main =
  4 files (`environ.py`, `parallel_state.py`, `dp_attention.py`, `deepseek_v4.py`);
  the prefill opts (flat RoPE / SE-local) are already in main.
- Docs: split `EXPERIMENT_LOG.md` into dated files + slim `HANDOFF.md`
  (`HANDOFF_ARCHIVE_updates.md`); added topic docs `TBO_RESEARCH.md`,
  `TRACE_PROFILING.md`.

## Fastest known config (as of 2026-06-24)
`SGLANG_USE_ROCM700A=0 SGLANG_DP_USE_GATHERV=1 SGLANG_SHARED_EXPERT_TP1=1
SGLANG_DP_SHARED_EXPERT_LOCAL=1 SGLANG_USE_AITER=1` + tp8/dp8 dp-attention, dsv4
backend, fp8 kv, `--chunked-prefill-size 65536`, cuda-graph-max-bs 512,
max-running 512. CK-GEMM + flat RoPE default-on; reduce_scatter default-on (HIP).
→ 8k/1k 37,701 tok/s, 1k/1k 18,666 tok/s, gsm8k 0.9545/0.9553.
