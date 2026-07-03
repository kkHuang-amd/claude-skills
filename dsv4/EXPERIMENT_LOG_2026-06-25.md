# DeepSeek-V4-Pro serving perf — experiment log (2026-06-25)

New workload regime: **70k input / 300 output, low concurrency (2–32)** — a
LONG-CONTEXT, prefill-dominated profile (very different from the prior 1k/1k and
8k/1k high-conc work). Goal: find the best server settings for max total
throughput across conc {2,4,8,16,32}.

- **Date**: 2026-06-25
- **Model**: `/dockerx/data/deepseek-ai/DeepSeek-V4-Pro/` (FP8)
- **sglang clone (active editable)**: `/sgl-workspace/sglang-upstream/python`
  (HEAD `ffb1afd5e`, branch `feat/dsv4-aiter-reduce-scatter-decode`; has PR #28216
  gatherv + PR #29103 reduce_scatter + flat-RoPE + SE-local).
- **Client**: `python3 -m sglang.bench_serving` (sglang-oai), ratio 1.0,
  request-rate inf, ignore-eos, **num-prompts=conc×4, warmups=conc×1** (reduced
  from ×8/×2 because each 70k request is ~10× heavier).
- **Scripts**: new `run_sgl_dsv4_70k.sh` (MODE/CHUNK/SWA/MEM/DELAYER knobs,
  pins sglang-upstream via PYTHONPATH); `sweep_dsv4_sglang_client.sh` got
  `NP_MULT`/`WARM_MULT` env knobs.
- **Metric**: total token throughput (tok/s); since ISL≫OSL it ≈ prefill (input)
  throughput. Primary gate: **0 retracted**.
- Raw results under `/workspace/bench_results_dsv4_70k/`, logs under
  `/workspace/dsv4_70k_logs/`.

> **Headline result (best-of, 70k/300):**
> | conc | mode | total tok/s | TTFT(s) | TPOT(ms) | E2E(s) |
> |---:|:--|---:|---:|---:|---:|
> | 2  | tp8 | 13,937 | 4.94 | 17.2 | 10.1 |
> | 4  | tp8 | 17,655 | 8.13 | 26.1 | 15.9 |
> | 8  | dp8 | **29,194** | 12.96 | 20.9 | 19.2 |
> | 16 | dp8 | **36,779** | 17.39 | 43.9 | 30.5 |
> | 32 | dp8 | **42,140** | 27.29 | 86.6 | 53.3 |
>
> **Best config**: conc 2/4 → plain **TP8**; conc 8/16/32 → **TP8+DP-attention**
> with **chunk 16384/rank, swa 0.1, mem-fraction 0.80, prefill-delayer OFF**.

---

## Pre-flight: pin sglang-upstream (import was broken)

`import sglang` resolved to a **namespace package** at `/sgl-workspace/sglang`
(`__file__=None`, `srt` empty) because something puts `/sgl-workspace` on
`sys.path`, and the dir `/sgl-workspace/sglang` (no `__init__.py`) shadowed the
editable finder (which correctly maps `sglang`→`/sgl-workspace/sglang-upstream/
python/sglang`). Fix used: launch with
`PYTHONPATH=/sgl-workspace/sglang-upstream/python:/sgl-workspace/mori:/sgl-workspace/aiter`
(upstream `__init__.py` is a real package → wins over the namespace). Verified
`sglang/server_args/deepseek_v4` all resolve to upstream + `version g4f174ce74`.
cohere2_moe `@strict` no-op already present in upstream (hf_hub 1.19.0) → no crash.

Common launch args: `--tp 8 [--dp 8 --enable-dp-attention] --attention-backend
dsv4 --page-size 256 --kv-cache-dtype fp8_e4m3 --disable-radix-cache
--disable-shared-experts-fusion --context-length 73728 --cuda-graph-max-bs 64
--max-running-requests 64 --chunked-prefill-size <C> --mem-fraction-static <M>
--swa-full-tokens-ratio <S>`. Verified actual prefill chunk via server log
`Prefill batch #new-token:` (= chunked_prefill_size; `max_prefill_tokens=16384`
does NOT cap it — `max_prefill_buffer_tokens()` returns chunked_prefill_size).
Note dp-attention divides `--chunked-prefill-size` by dp_size (server_args.py:5139),
so per-rank chunk = global/8.

## Phase 1 — TP8 (no dp), chunk per-rank 32768

| conc | swa | total tok/s | TTFT(s) | TPOT(ms) | E2E(s) | retract |
|---:|---:|---:|---:|---:|---:|---:|
| 2  | 0.1  | 13,937 | 4.94  | 17.2  | 10.1  | 0 |
| 4  | 0.1  | 17,655 | 8.13  | 26.1  | 15.9  | 0 |
| 8  | 0.1  | 20,341 | 13.49 | 47.3  | 27.6  | 0 |
| 16 | 0.24 | 21,207 | 12.19 | 132.0 | 51.1  | 0 |
| 32 | 0.24 | 22,037 | 20.26 | 271.5 | 100.4 | 56 |

- **TP8 throughput SATURATES at conc≈8 (~20k)**: conc 8→32 only +8% but TPOT
  47→271ms and E2E 28→100s explode. A single request's attention is TP-sharded
  over all 8 GPUs; past conc≈8 the prefill compute is full and extra conc just
  queues. TP8 sweet spot = conc 8.

### SWA-pool binding (predicted by user) — DSV4 has a custom pool
DSV4 uses `DSV4PoolConfigurator` (NOT the generic HybridSWA): memory is split into
`full / swa / c4 / c128 (+ c4_state/c128_state)` sub-pools. **swa sub-pool =
full × swa_full_tokens_ratio**; `bytes_per_full_token` itself scales with
swa_ratio. The binding pool at high conc is **swa** (stores compressed/state KV
that grows with context, NOT window-bounded). At swa 0.1, conc=32 maxes the swa
pool (usage 1.00 while full only 0.43) → 64 retracts. Raising swa: 0.1→64,
0.22→8, 0.24→16 retracts (count is timing-noisy). Model fit (mem0.95):
`full(r)=avail/(X(1+32.6r))`, `swa(r)=full·r`; feasible window for conc=32
(full≥2.24M AND swa≥~540k) is only r∈[0.221,0.244], and the swa-pool asymptote
(~615–647k) is near the conc=32 demand. **Conclusion: conc=32 × 70k is at the
single-node memory ceiling in TP8 — a few retracts are essentially unavoidable;
swa≈0.24 + mem 0.97 is the best balance (full=2.39M, swa=574k).** swa_ratio only
repartitions memory (no compute effect), so low-conc points keep swa 0.1.

## Phase 2 — TP8+DP-attention, chunk per-rank 32768

First attempt OOM-crashed: dp-attention MoE runs on the **gathered global** batch
(per-rank chunk × dp = 32768×8 = 262144 tokens); `fused_moe` stage2 tried to
allocate 18.4 GB and there was only 13.7 GB free at mem-fraction 0.95 (KV pool ate
it). **Fix: drop mem-fraction to 0.80** — dp KV is hugely over-provisioned (per-DP-
rank full pool 1.81M ≫ conc32 need of 4 reqs×70k=280k/rank), so freeing memory for
MoE activation is free. swa 0.1 fine (per-rank swa 180k ≫ need).

| conc | total tok/s | TTFT(s) | TPOT(ms) | E2E(s) | retract |
|---:|---:|---:|---:|---:|---:|
| 8  | 24,873 | 16.31 | 20.8 | 22.5 | 0 |
| 16 | 33,467 | 20.33 | 42.7 | 33.5 | 0 |
| 32 | 39,026 | 25.37 | 104.3| 57.4 | 0 |

- **DP-attention SCALES** (24.9→33.5→39.0k) where TP8 plateaus, **0 retract** at
  all conc (per-rank memory load = conc/8 × 70k, tiny). vs TP8: c8 +22%, c16 +58%,
  c32 +77%. DP parallelizes across requests (each rank prefills its own), avoiding
  the shared-attention serialization that caps TP8. ⇒ **use DP-attention for
  conc≥8**; TP8 only for conc<8 (where DP would idle ranks, per design).

## Phase 3 — chunk-size sweep (dp8): NEUTRAL

| conc | chunk16384/rank | chunk32768/rank |
|---:|---:|---:|
| 8  | 25,091 | 24,873 |
| 16 | 33,453 | 33,467 |
| 32 | **39,741** | 39,026 |

chunk size is ~neutral (±2%) — it changes prefill granularity, not total FLOPs.
**chunk 16384/rank chosen**: marginally best AND halves the global MoE batch
(131072 vs 262144) → safer against the Phase-2 OOM. chunk 65536/rank skipped
(low value, higher OOM risk).

## Phase 4 — prefill-delayer ON vs OFF (dp8, chunk16384): OFF WINS

| conc | delayer ON | delayer OFF | Δ |
|---:|---:|---:|---:|
| 8  | 25,091 | **29,194** | +16.4% |
| 16 | 33,453 | **36,779** | +9.9% |
| 32 | 39,741 | **42,140** | +6.0% |

**prefill-delayer HURTS this regime** (TTFT/E2E also better OFF). The delayer
delays prefill admission to protect decode-batch occupancy; but 70k/300 low-conc
is prefill-dominated with almost no decode to protect, so delaying prefill is pure
loss. **This is the OPPOSITE of the 8k/1k high-conc finding (Exp 20: delayer
essential, +41%)** — the delayer's value is regime-dependent: keep ON for
decode-heavy/high-conc, turn OFF for long-context prefill-dominated low-conc.
(TP8 path in the wrapper never adds the delayer, so the conc 2/4 TP8 points are
already delayer-OFF.)

## Recommended settings (70k/300)

- **conc 2, 4** → `MODE=tp8` (no dp), chunk 32768/rank, swa 0.1, mem 0.92,
  delayer off (default in tp8 path).
- **conc 8, 16, 32** → `MODE=tp8dp8`, chunk **16384/rank** (`--chunked-prefill-size
  131072`), swa 0.1, **mem-fraction 0.80**, **prefill-delayer OFF**.
- env: `SGLANG_USE_ROCM700A=0 SGLANG_USE_AITER=1` (+dp: gatherv + reduce_scatter +
  SE-local + TP1 shared).
- Caveats: (1) dp MoE OOMs if mem-fraction too high (global MoE batch = chunk×dp);
  (2) TP8 conc=32 is swa-pool/memory-bound at 70k (few retracts unavoidable);
  (3) TP8 throughput saturates ~conc8.

## ATOM comparison (2026-06-26) — single-stream vs multi-stream

ATOM `run_atom_dsv4_aligned.sh` DP_MODE=tp8dp8 (dp-attention), `--max-model-len
73728`, max-num-batched-tokens 16384 (= per-rank, matches SGLang dp chunk16384),
gpu-mem 0.9, block 256, fp8 kv. Stream knob = `ATOM_DISABLE_SIDE_STREAMS`
(0=multi/side-streams ON, 1=single). SAME client as SGLang (`bench_dsv4.py` =
sglang.bench_serving + ATOM SSE shim; sglang-upstream on PYTHONPATH), same grid
(70k/300, NP_MULT=4 WARM_MULT=1). No OOM at gpu-mem 0.9 (ATOM dp KV per-rank is
small; global MoE batch = 16384×8 = 131072, fits). All ATOM runs are dp-attention
(incl. conc 2/4, where dp idles ranks — see note).

InferenceX-style (E2EL/TTFT/ITL in ms; interactivity = 1000/median ITL):

ATOM **single-stream** (side-streams OFF):
| ISL | OSL | par | conc | TTT tok/s | E2EL ms | TTFT ms | ITL ms | interact |
|---:|---:|:--|---:|---:|---:|---:|---:|---:|
| 70000 | 300 | TP8,DP8 | 2  | 8,755.4  | 16,035.0 | 8,088.4  | 23.699 | 42.20 |
| 70000 | 300 | TP8,DP8 | 4  | 16,013.3 | 17,547.9 | 10,268.5 | 24.010 | 41.65 |
| 70000 | 300 | TP8,DP8 | 8  | 27,481.4 | 20,469.1 | 13,186.0 | 24.249 | 41.24 |
| 70000 | 300 | TP8,DP8 | 16 | 35,135.8 | 32,001.8 | 17,716.9 | 26.882 | 37.20 |
| 70000 | 300 | TP8,DP8 | 32 | 41,166.3 | 54,639.3 | 27,900.0 | 30.576 | 32.71 |

ATOM **multi-stream** (side-streams ON, ATOM default):
| ISL | OSL | par | conc | TTT tok/s | E2EL ms | TTFT ms | ITL ms | interact |
|---:|---:|:--|---:|---:|---:|---:|---:|---:|
| 70000 | 300 | TP8,DP8 | 2  | 9,595.5  | 14,497.6 | 7,465.3  | 22.113 | 45.22 |
| 70000 | 300 | TP8,DP8 | 4  | 17,029.1 | 16,473.9 | 9,293.2  | 22.419 | 44.61 |
| 70000 | 300 | TP8,DP8 | 8  | 28,989.6 | 19,408.6 | 12,050.5 | 22.539 | 44.37 |
| 70000 | 300 | TP8,DP8 | 16 | 35,759.7 | 31,447.0 | 17,800.1 | 25.008 | 39.99 |
| 70000 | 300 | TP8,DP8 | 32 | 41,949.7 | 53,932.6 | 27,786.3 | 28.225 | 35.43 |

### Findings
- **multi-stream > single-stream**, margin shrinks with conc: c2 +9.6%, c4 +6.3%,
  c8 +5.5%, c16 +1.8%, c32 +1.9% (total tok/s); ITL/interactivity also better
  on multi. Consistent with Exp 40 (side-streams overlap Compressor+MoE → help
  mostly at low conc / decode). **Keep multi-stream (default) for this workload.**
- **ATOM (multi) vs SGLang best**: c2 145% (SGLang tp8 13,937 vs ATOM dp 9,596 —
  but mode-mismatched: ATOM is dp here, idling 6/8 ranks at c2); c4 104%; c8 101%;
  c16 103%; c32 100%. ⇒ **at conc≥8 the two engines are within ~1–3% in dp mode**;
  at conc 2/4 SGLang wins only because SGLang used TP8 (full-GPU per request) while
  ATOM ran dp (idle ranks). Apples-to-apple ATOM-tp8 at conc 2/4 not yet run.

| conc | ATOM single | ATOM multi | SGLang best (mode) | SGL/ATOMmulti |
|---:|---:|---:|---:|---:|
| 2  | 8,755  | 9,596  | 13,937 (tp8) | 145.2% |
| 4  | 16,013 | 17,029 | 17,655 (tp8) | 103.7% |
| 8  | 27,481 | 28,990 | 29,194 (dp8) | 100.7% |
| 16 | 35,136 | 35,760 | 36,779 (dp8) | 102.9% |
| 32 | 41,166 | 41,950 | 42,140 (dp8) | 100.5% |

Artifacts: `/workspace/bench_results_dsv4_70k/atom_{single,multi}stream/`.

### ATOM tp8 (no dp) at conc 2/4 (multi-stream) — apples-to-apple low-conc
`DP_MODE=tp8` (no `--enable-dp-attention`), multi-stream, max-num-batched-tokens
16384, max-model-len 73728. gpu-mem 0.9 fine (tp8 MoE is local, no gather OOM).

| ISL | OSL | par | conc | TTT tok/s | E2EL ms | TTFT ms | ITL ms | interact |
|---:|---:|:--|---:|---:|---:|---:|---:|---:|
| 70000 | 300 | TP8 (DP1) | 2 | 12,893.1 | 10,907.5 | 4,638.1 | 16.838 | 59.39 |
| 70000 | 300 | TP8 (DP1) | 4 | 16,605.6 | 16,927.2 | 7,467.2 | 17.284 | 57.86 |

- For ATOM too, **tp8 > dp at conc 2** (12,893 vs dp 9,596, +34%) and ≈ at conc 4
  (16,606 vs dp 17,029) → same tp8↔dp crossover (~conc 4) as SGLang. Confirms:
  use TP8 for conc<8, DP-attention for conc≥8 on BOTH engines.
- **Apples-to-apple best-of (multi-stream ATOM vs best SGLang), same mode per conc:**

| conc | mode | ATOM best | SGLang best | SGL/ATOM |
|---:|:--|---:|---:|---:|
| 2  | tp8 | 12,893 | 13,937 | 108.1% |
| 4  | tp8 | 16,606 | 17,655 | 106.3% |
| 8  | dp8 | 28,990 | 29,194 | 100.7% |
| 16 | dp8 | 35,760 | 36,779 | 102.9% |
| 32 | dp8 | 41,950 | 42,140 | 100.5% |

⇒ **SGLang ≥ ATOM across the whole curve** (+0.5–3% at conc≥8, +6–8% at conc 2/4).
Engines are essentially at parity in dp mode; SGLang's edge is largest at low conc.
Artifacts: `/workspace/bench_results_dsv4_70k/atom_tp8_multistream/`.

## Open / next
- conc>32 or longer context would push DP per-rank memory; would then need the
  same swa/mem tuning DP avoided here.
- EP/mori path not tried for this workload.

---

# 2026-06-25 (PM) — DSV4 EP+TBO implementation (prefill two-batch-overlap)

Goal (user-chosen scope): implement two-batch-overlap for DeepSeek-V4 on the
**EP / mori a2a** path, so `--enable-two-batch-overlap` overlaps one ubatch's MoE
a2a dispatch/combine with the other ubatch's attention + expert GEMM. Prefill-only
(ATOM data: decode TBO regresses). Active clone: `/sgl-workspace/sglang-upstream`,
branch `feat/dsv4-aiter-reduce-scatter-decode`.

## Pre-validation (both passed before coding)

1. **DSV4 mori EP runs & is correct.** aligned `run_sgl_dsv4_aligned.sh` +
   `EP_MODE=mori` boots/serves; gsm8k 0.9431/0.9439 (cross-check standalone
   `run_sgl_dsv4_mori-ep.sh` 0.9522/0.9530). Needs `PYTHONPATH` pin (namespace
   shadow at `/sgl-workspace/sglang`); does NOT need the standalone script's
   `MORI_*`/IB env (single-node intra-node mori uses IPC/shmem).
2. **dsv4 backend ubatch-split is feasible (code).** `DeepseekV4HipRadixBackend.
   init_forward_metadata` reads only GENERIC ForwardBatch fields (`req_pool_indices/
   seq_lens/seq_lens_cpu/out_cache_loc/extend_seq_lens[_cpu]`) — exactly what TBO
   `filter_batch` slices; does NOT read `out_cache_loc_dsv4` (NPU-only) or freqs_cis.
   `filter_batch` is strict (raises on any unhandled non-None field) → no silent
   corruption. TBO batch-prep is model-agnostic (`TboForwardBatchPreparer.prepare`
   runs unconditionally in `ForwardBatch.__init__`; `can_run_tbo` gated only by
   `is_tbo_enabled()` + deepep-mode — mori `normal` permits prefill TBO,
   `low_latency` would block it). ⇒ **no scheduler-side changes needed.**

## Design

DSV4's layer has CROSS-LAYER fused-mHC (returns `(hidden,residual,post,comb)`
consumed by next layer's `mhc_fused_post_pre`); that breaks clean per-layer ops.
**Fix: disable fused-mHC under TBO** → the non-fused branch makes every layer
self-contained → maps to ops. The MoE itself is `DeepseekV2MoE` (DSV4 reuses it),
so its `op_*` methods (which decompose `forward_deepep`) are reused as-is.

Op sequence (prefill strategy, 2 yields, same shape as V2 prefill):
```
op_mhc_prepare_attn, self_attn.op_attn, op_mhc_post_attn_pre_mlp,
mlp.op_gate, mlp.op_select_experts, mlp.op_dispatch_a, YIELD,
mlp.op_dispatch_b, mlp.op_experts, mlp.op_combine_a, YIELD,
mlp.op_shared_experts, mlp.op_combine_b, mlp.op_output, op_mhc_postprocess
```
Driver: `DeepseekV4Model._forward_layers_tbo` reuses the GENERIC
`execute_overlapped_operations` + `_model_forward_filter_inputs` (token-range slice
+ pad to tbo_padded_len over `forward_batch.tbo_children`) + `_model_forward_tbo_
merge_outputs` — avoids V2's `ScatterMode`/`layer_communicator` coupling (DSV4 has
neither; dp-attention input is already per-rank). `_can_run_tbo` gate: is_tbo_enabled
AND can_run_tbo AND tbo_children AND global_forward_mode.is_extend() AND not CP AND
pp_world==1.

## Files changed (all in /sgl-workspace/sglang-upstream/python/sglang/srt)

- `models/deepseek_v4.py`: `MQALayer.op_attn`; `DeepseekV4DecoderLayer.
  op_mhc_prepare_attn` (residual=hs; hc_pre(attn)+input_layernorm; stash
  residual/post/comb; set num_tokens for mori), `op_mhc_post_attn_pre_mlp`
  (hc_post(attn) → hc_pre(ffn)+post_attn_layernorm → set hidden_states_mlp_input),
  `op_mhc_postprocess` (hc_post(ffn); return next-layer dict incl. `residual=None`);
  `DeepseekV4Model._can_run_tbo` + `_forward_layers_tbo`; forward branches to TBO
  driver (forces non-fused) else the normal loop.
- `models/deepseek_v2.py`: `op_select_experts` passes `input_ids=
  state.forward_batch.input_ids` when `self.is_hash` (DSV4 IS a hash MoE; in EP
  dp-attn input_ids_global==input_ids, child fb input_ids is sliced+padded to match
  rows). No-op for V2/Qwen3/MiMo.
- `batch_overlap/operations_strategy.py`: `init_new_tbo` `DeepseekV4DecoderLayer`
  branch + `_compute_moe_deepseek_v4_prefill` (EXTEND only; decode → NotImplemented).
- `layers/attention/tbo_backend.py`: `TboAttnBackend.__getattr__` delegates unknown
  attrs to `self.primary` (so DSV4-specific backend methods called via
  get_attn_backend() — `get_unified_swa_loc`, `get_swa_out_cache_loc` — resolve in
  the non-overlap path; inside TBO get_attn_backend resolves to the child directly).

## Bugs hit & fixed (in order)

1. `'TboAttnBackend' object has no attribute 'get_unified_swa_loc'` → `__getattr__`.
2. Decode cuda-graph capture HIP OOM (TBO triples attn-backend cuda-graph state +
   doubles mori dispatchers; aligned mem 0.90 left 312MB free, MoE needed 576MB) →
   `--mem-fraction-static 0.72 --cuda-graph-max-bs 64 --max-running-requests 64`.
   (NOTE: this OOM is the NORMAL decode path, not the TBO driver.)
3. `HashTopK.forward() missing 'input_ids'` → op_select_experts hash handling.
4. `KeyError: 'residual'` in `_model_forward_tbo_merge_outputs` → op_mhc_postprocess
   returns `residual=None`.

## Results

- **TBO engages** on multi-seq prefill batches (server log: new-seq=2 ×56,
  new-seq=3 ×5 during a gsm8k run) via `_forward_layers_tbo`.
- **Correctness: gsm8k EP+TBO = flexible 0.9600 / strict 0.9567** (limit 300,
  conc 8). Correct band (no-TBO mori 0.9431/0.9522; broken-MoE fallback ~0.6).
  ⇒ the op decomposition is numerically correct.

## OPEN BUG — HSA OUT_OF_RESOURCES (stability)

Under SUSTAINED TBO load the scheduler aborts with
`HSA_STATUS_ERROR_OUT_OF_RESOURCES` ("…spawn threads or create internal OS-specific
events"; the printed "Available Free mem" is a garbage overflow). Non-deterministic:
crashed at ~13 min (conc16 full), ~4 min (conc16 + expandable_segments), but
limit-300/conc8 finished. Ruled OUT: (a) VRAM — `PYTORCH_HIP_ALLOC_CONF=
expandable_segments:True` did NOT help; (b) aiter JIT churn — only 76 LoadKernel
(10 unique), SAME as no-TBO (80). No-TBO mori ran a FULL gsm8k at conc 64 without
crashing → **TBO is the trigger**. Hypothesis: HSA queue/event/signal exhaustion
from TBO's 2 concurrent ubatch streams + mori's 2 inner dispatchers
(`MaybeTboDeepEPDispatcher` builds `num_inner_dispatchers=2` when tbo enabled) +
decode cuda-graph HSA reservations — peak concurrent HSA resource usage hits the
ROCm ceiling intermittently. NEXT: confirm via HSA resource accounting / whether
events or streams are created per-forward and not freed (check
`execute_overlapped_operations`, mori dispatch_a/b, and whether the 2 mori
dispatchers leak per-call queues on ROCm).

### Stability debug round 1 (2026-06-25 PM, cont.)
- **Located a per-call event source**: `MoriEPDispatcher` creates
  `torch.cuda.Event(blocking=False)` per dispatch/combine ONLY when
  `enable_dual_stream (=is_tbo_enabled())` AND `async_finish` (hardcoded True at
  `fused_moe_triton/layer.py:114`). deepep uses its own `Buffer.capture()` instead.
  This is why non-TBO mori (events off) never crashed. `CommStreamPool` reuses
  streams per group (no stream leak); `ulimit -n`=1048576 (NOT an fd-limit issue).
- **TEST: gated mori async_finish off** (wait_stream sync, no per-call events) and
  ran a FULL conc16 gsm8k. **Still crashed** (HSA_STATUS_ERROR_OUT_OF_RESOURCES),
  just LATER (~9 min / ~54%+ vs 4–13 min). ⇒ mori per-call events are a CONTRIBUTOR
  but NOT the root cause. Change reverted (layer.py back to `async_finish=True`).
- Crash context this time: a flurry of single-seq prefills (#new-seq:1) + decode
  (`cuda graph: True`) — NOT only TBO-split batches. Accumulation is broader than
  the TBO a/b dispatch events.
- **Still ruled out**: VRAM (expandable_segments no help), aiter JIT (76 loads),
  process fd limit (1M), mori async events alone.
- **Leading remaining hypotheses** (untested):
  1. decode CUDA-graph + TBO interaction (TboAttnBackend = primary + 2 children;
     init_cuda_graph_state on all 3). → TEST: `--disable-cuda-graph`.
  2. per-forward HSA queue/signal growth from 3 attn backends' init_forward_metadata
     (DSV4 compressor/indexer) ×(primary+2 children).
  3. general mori+TBO infra bug (not DSV4-specific). → TEST: R1 + mori + TBO.
- NEXT (ROI order): (a) `--disable-cuda-graph` rerun; (b) instrument hipEvent/queue
  counts per N iters; (c) R1+mori+TBO control. DSV4 op-decomposition is proven
  CORRECT (gsm8k 0.96) → this is an infra/runtime stability issue, likely not in
  the DSV4-specific ops.
- **STRONGEST hypothesis (explains why async_finish-off only delayed it):** mori's
  dual-stream path (`enable_dual_stream=is_tbo_enabled()`, independent of
  async_finish) calls `tensor.record_stream(comm_stream)` on many tensors every
  dispatch/combine (`_dispatch_core` lines ~648-651/689-696, and combine). PyTorch's
  caching allocator tracks each cross-stream `record_stream` with an internal
  CUDA/HIP event to defer the free; on ROCm these wrap limited HSA signals. Under
  TBO's sustained dual-stream churn the allocator's pending-event set grows →
  HSA OUT_OF_RESOURCES. Non-TBO mori has `_comm_stream=None` → no record_stream →
  no allocator events → stable (matches observations). async_finish-off removed the
  EXPLICIT mori events but NOT the record_stream allocator events → crashed later,
  not never. If confirmed, fixes are infra-level (reduce record_stream usage / event
  reuse / a ROCm allocator setting), not in DSV4 op code.

### Stability debug round 2 — R1 control (2026-06-25 PM): bug is DSV4-SPECIFIC

**Control: DeepSeek-R1-0528-MXFP4 (`DeepseekV3ForCausalLM` → `DeepseekV2DecoderLayer`,
TBO-supported upstream) + mori + TBO**, same box / same TBO infra / same mori
(`async_finish=True`), `aiter` attention backend, mem 0.72, cuda-graph-max-bs 64.
Ran the **FULL gsm8k (1319) at conc 16 → COMPLETED, NO crash**, 0.9484/0.9439.
Model `/dockerx/data/amd/DeepSeek-R1-0528-MXFP4/`, launch `/workspace/run_r1_mori_tbo.sh`.

**Conclusions (strong):**
- The HSA OUT_OF_RESOURCES is **DSV4-SPECIFIC**, NOT a general mori+TBO infra bug.
  Matrix: DSV4+mori+TBO = CRASH; DSV4+mori (no TBO, conc64 full gsm8k) = STABLE;
  **R1+mori+TBO = STABLE**. ⇒ it's the **DSV4 × TBO combination**.
- **mori per-call `torch.cuda.Event` is DEFINITIVELY exonerated**: R1+TBO uses the
  exact same mori dispatcher (async_finish=True, per-call events) and is stable.
  (Confirms round-1's "async_finish-off only delayed it" was a red herring.)
- The DSV4 multi-stream `record_event` paths (`deepseek_v4.py:585/588`,
  `indexer.py:395`) are OFF in our config (`SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=
  false`, `SGLANG_ROCM_USE_MULTI_STREAM=false` → normal prepare). No Python-level
  cuda.Event/Stream/record_stream in the ACTIVE DSV4 attention path; no per-forward
  list/dict growth in `DeepseekV4HipRadixBackend`.
- **Remaining root-cause locus**: DSV4-specific runtime under TBO. Amplification =
  TBO runs `init_forward_metadata` on TboAttnBackend's primary + 2 children (3× per
  forward) and the DSV4 attention/compressor/indexer per-ubatch (2×). Prime suspects:
  a DSV4-only aiter custom kernel (compressor `flydsl_hca`/compress_hip,
  `get_paged_mqa_logits_metadata`, fp8 paged decode) creating HSA queues/signals per
  invocation, hitting the ROCm ceiling at the higher TBO call rate.
- **NEXT (focused)**: (a) instrument hipEvent/HSA-queue count per N forwards on a
  DSV4+TBO run to see what grows; (b) bisect the DSV4 attention sub-path under TBO
  (e.g. temporarily route children's init_forward_metadata to reuse primary's; or
  disable the c4/c128 compressor) to localize; (c) try `--disable-cuda-graph` to rule
  cuda-graph in/out. The DSV4 op-decomposition itself is CORRECT (gsm8k 0.96) — the
  fix will be in the DSV4 attention-backend/TBO interaction, not the op code.

### Stability debug round 3 — instrumentation + cuda-graph bisect (2026-06-25 PM): ROOT-CAUSE LOCALIZED

**(a) Resource instrumentation** — added `DeepseekV4Model._resmon_maybe_log` (gated by
file `/workspace/RESMON_ON`; time-throttled ~8s; logs live `cuda.Event`/`cuda.Stream`
via gc + `torch.cuda.memory_allocated/reserved`). NOTE: env-var gating failed —
sglang spawns schedulers with a CURATED env (only registry `SGLANG_*` forwarded), so
ad-hoc env vars don't reach schedulers → used a file gate. Also note Python `forward`
is SKIPPED on cuda-graph-replayed decode, so the counter mostly samples eager prefill.
Findings (DSV4+TBO, conc16, up to the crash at fwd≈152):
- `/proc` fd=1072 FLAT, threads=1722 FLAT → not an fd/eventfd/thread leak.
- `live_cuda_Event=2` FLAT, `live_cuda_Stream=10` FLAT, `mem_alloc`~201GB FLAT,
  `mem_reserved` fluctuates 264–276GB (workload, non-monotonic) → **NO Python-level
  Event/Stream/memory leak, right up to the crash**. ⇒ the exhausted resource is
  purely C++/ROCr-internal HSA signals/queues (invisible to /proc and Python).

**(b) cuda-graph bisect — DECISIVE.** Re-ran DSV4+mori+TBO with `--disable-cuda-graph`
(decode now eager → 7–50× more Python forwards/min): **ran the FULL gsm8k (1319) with
NO crash, 0.9447/0.9454**, fwd reached >8000, RESMON flat throughout. vs WITH
cuda-graph it crashes by ~150 eager(prefill) forwards / 4–13 min.

**ROOT CAUSE (localized): a 3-way interaction — DSV4 × TBO × decode-cuda-graph.**
Matrix now:
| config | cuda-graph | result |
|---|---|---|
| DSV4 + mori + TBO | ON | CRASH (HSA OUT_OF_RESOURCES) |
| **DSV4 + mori + TBO** | **OFF** | **STABLE, full gsm8k 0.9447/0.9454** |
| DSV4 + mori (no TBO) | ON | STABLE (conc64 full gsm8k) |
| R1 + mori + TBO | ON | STABLE (full gsm8k) |

Mechanism: `--enable-two-batch-overlap` wraps the attn backend in `TboAttnBackend`
(primary + 2 children), and `init_cuda_graph_state` is called on all 3
(`tbo_backend.py:119-123`). Decode is NON-TBO (our `_can_run_tbo` gates to EXTEND) so
it runs graphed on the primary — but with the TBO wrapper + 3 DSV4 backends present,
the DSV4 decode cuda-graph capture/replay leaks HSA resources per step. R1's `aiter`
backend under the same wrapper does NOT leak; DSV4 WITHOUT the wrapper does NOT leak.
So it's the **DSV4 attention backend (compressor/indexer/unified-kv decode kernels) ×
cuda-graph × the TBO 3-backend wrapper**. This matches the user's V3-vs-V4 intuition:
the shared MoE/TBO ops are fine (R1 proves it); the DSV4-specific attention backend is
what needs handling — specifically its decode-cuda-graph path under the TBO wrapper.

**Workaround (works today):** DSV4 EP+TBO is stable & correct with
`--disable-cuda-graph` (cost: eager decode = slower decode; TBO's win is prefill, so
acceptable for prefill-heavy). **Proper fix (next):** the DSV4 decode-cuda-graph ×
TBO-wrapper HSA leak — candidates: (i) since DSV4 TBO is prefill-only & DSV4 prefill
cuda-graph is already disabled, the 2 TBO CHILD backends may not need decode
cuda-graph state at all → skip `init_cuda_graph_state`/capture on children (likely
removes the leak source); (ii) keep decode on the unwrapped primary backend's own
cuda-graph; (iii) find the per-replay HSA alloc in the DSV4 decode graph under the
wrapper. NEXT: try (i) — skip child cuda-graph capture for DSV4+TBO and re-test with
cuda-graph ON.

### Stability debug round 4 — fix attempt #1 (skip TBO children in cuda-graph): INSUFFICIENT

Implemented the suspected fix: a backend can declare `tbo_supports_cuda_graph=False`
(DSV4 HipRadix backend sets it); `TboAttnBackend` then skips the 2 children in ALL
cuda-graph paths (`init_cuda_graph_state` / `init_forward_metadata_{out,in}_graph` /
`on_after_cuda_graph_warmup` / `get_cuda_graph_seq_len_fill_value`), keeping them only
for eager prefill TBO (`init_forward_metadata`). Rationale: DSV4 decode is non-TBO
(model gate) so the children's per-replay compressor/indexer metadata rebuild
(`_dispatch_children_from_replay_view`) is pure waste + suspected leak.
Files: `tbo_backend.py` (+`_children_use_cuda_graph` gate), `deepseek_v4_backend_hip_radix.py`
(`tbo_supports_cuda_graph = False`).

**Result: STILL CRASHES** (HSA OUT_OF_RESOURCES) at ~11.5 min (vs ~4–13 min before —
no meaningful change). ⇒ the children were NOT the leak; it's the **PRIMARY decode
cuda-graph under TBO**.

**Key constraint found (`two_batch_overlap.py:336-337`):** *"when two_batch_overlap is
enabled, we only capture CUDA Graph for tbo=true"* (`assert tbo_split_seq_index is not
None`). So enabling TBO forces EVERY captured graph (incl. DSV4 decode) to be a
**tbo=true** graph. DSV4 decode *compute* runs non-TBO (my model `_can_run_tbo` gate →
normal loop), but the graph is captured/replayed in tbo=true mode (tbo_plugin sets
split indices, replay runs `replay_prepare` + the tbo replay path each step). That
tbo=true DSV4 decode graph is what leaks HSA on this ROCm/mori stack (R1's tbo=true
decode graph does not leak). The framework cannot capture a NON-TBO decode graph while
TBO is enabled (the assert forbids it) → choices are: tbo=true decode graph (leaks) OR
no decode graph.

**Remaining options:**
- (A) **Auto-disable decode cuda-graph for DSV4+TBO** (= the proven `--disable-cuda-graph`
  result, scoped; `--disable-decode-cuda-graph` since DSV4 prefill graph is already off).
  Guaranteed stable; cost ≈ 2× slower decode (eager). Acceptable since TBO targets
  prefill-heavy. Simple, ship-able.
- (B) Framework change: allow non-TBO decode graphs while TBO enabled (capture both
  variants / relax the line-337 assert per-model). Bigger, touches shared runtime.
- (C) GPU-level HSA tracing (rocprof / HSA intercept) to pin the exact per-replay HSA
  alloc in the tbo=true DSV4 decode graph and fix it. Deepest, keeps decode graph perf.
The children-skip change is correct (DSV4 children never run graphs) but not the fix;
kept for now (harmless, reduces child graph memory). DECISION PENDING (A vs C) — perf
(decode) vs effort tradeoff is the user's call.

### Stability debug round 5 — (C) GPU-level interposer + more ruled-out (2026-06-25 PM)

Wrote an `LD_PRELOAD` C interposer (`/workspace/hipcount.{c,so}`) counting create vs
destroy of hip resources (resolves real syms from libamdhip64 via RTLD_NOLOAD;
NULL-guarded). LD_PRELOAD DOES propagate to the spawn'd schedulers. Findings:
- **hipEvent: NOT leaking.** Fine-grained trend (log every 2000 creates):
  `ev_live` FLAT ~330 (created≈20000 / destroyed≈19670) across the whole run.
- **hipStream: NOT leaking.** `st_live` FLAT at 11.
- So the exhausted resource is NOT a hipEvent/hipStream handle leak.
- **`HSA_NO_SCRATCH_RECLAIM` is NOT it**: relaunched with `HSA_NO_SCRATCH_RECLAIM=0`
  (env had =1) → STILL crashed. Scratch ruled out.
- **hipGraph interposer BLOCKED by symbol versioning**: ROCm tags graph syms as
  `hipGraph*@@hip_4.x` and torch imports the versioned refs; a plain LD_PRELOAD def
  doesn't interpose them, and even a `.symver`/version-script default-version def did
  NOT bind (versioned-symbol interposition is the known-hard case). So could not
  measure hipGraphInstantiate/Launch/capture growth this way. (Tools: `nm -D
  libamdhip64.so | grep hipGraph` shows `@@hip_4.3/4.5`; `libtorch_hip.so` has
  `U hipGraphLaunch@hip_4.3`.)

**Cumulative ruled-OUT for the DSV4×TBO×decode-cuda-graph crash:** VRAM /
fragmentation, aiter JIT, process fd+threads, hipEvent, hipStream, mori per-call
events (R1 control), HSA scratch reclaim, TBO child-backend metadata
(`init_forward_metadata_out_graph` skip). Still-open suspects (need a working probe):
hipGraph exec re-instantiation per step, direct HSA `hsa_signal_create`/
`hsa_queue_create` (below hip; e.g. from mori RDMA), or a transient concurrency
spike of HSA command resources when the tbo=true decode graph replays alongside
eager prefill-TBO + eager mori a2a.

**Next-probe options (for a fresh session):** (i) make the versioned hipGraph
interposer actually bind (e.g. patch libamdhip64's GOT via a constructor, or use
`rocprofv3`/`roctracer` HIP-API trace to count hipGraphInstantiate at runtime);
(ii) HSA-layer interposer on `hsa_signal_create`/`hsa_amd_signal_create`/
`hsa_queue_create` in libhsa-runtime64 (small-struct-by-value ABI: hsa_signal_t/
hsa_agent_t are {uint64} → pass as uint64); (iii) `roctracer` HIP+HSA API count diff
between DSV4+TBO (crashes) and R1+TBO (stable) to see which API's live-count diverges.
Artifacts kept: `/workspace/hipcount.c` (+version script `/workspace/ver.map`),
`_resmon_maybe_log` (file-gated `/workspace/RESMON_ON`, off), children-skip change.

### Stability debug round 6 — attention-backend swap: NOT the attention kernel (2026-06-25 PM)

Tested user hypothesis: is it the DSV4 attention kernel? Swapped
`SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton` → `triton` (changes BOTH the MQA
decode path via `is_unified_kv_triton()`→False AND the SWA flash_mla kernel to the
NSA triton decode). Made the aligned script honor an external override
(`${SGLANG_HACK_FLASHMLA_BACKEND:-unified_kv_triton}`). **Result: STILL CRASHES**
(HSA OUT_OF_RESOURCES, ~7 min; remaining ranks then deadlock → GPU idle, client
timeouts). ⇒ **the leak is NOT in the flash_mla / MQA attention kernel.** Valid
backends seen in `hip_flash_mla.py`: unified_kv_triton / triton / tilelang / torch /
kernel / comparison.

**DSV4-specific decode-graph components R1(V3) lacks — narrowed:**
- attention (flash_mla/MQA): **RULED OUT** (both unified_kv_triton & triton crash).
- still-suspect: **mHC** (hc_pre/hc_post/hc_head + sinkhorn), **compressor** (c4/c128
  KV compression, compress_hip), **indexer** (DSA topk / paged_mqa_logits / aiter
  indexer). Next analogous cheap swaps: `SGLANG_OPT_USE_AITER_INDEXER`,
  `SGLANG_OPT_USE_FUSED_COMPRESS[_TRITON]`, `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH`.

### Stability debug round 7 — aiter indexer swap: NOT the indexer (2026-06-25 PM)

`SGLANG_OPT_USE_AITER_INDEXER=false` (→ torch `paged_mqa_logits` path instead of the
aiter custom indexer kernel; made the toggle override-friendly in the aligned script).
**Result: STILL CRASHES** (~10 min). ⇒ the aiter DSA indexer is NOT the (sole) leak.

**Summary after rounds 1–7 — every individually-swappable component RULED OUT:**
attention kernel (unified_kv_triton ↔ triton), aiter indexer (on↔off), mori async
events (on↔off), HSA scratch reclaim (on↔off), TBO children cuda-graph (skip),
hipEvent (flat), hipStream (flat), VRAM (expandable_segments no-op), aiter JIT, fd/
threads. NONE fixed the crash; ALL still hit HSA OUT_OF_RESOURCES.

**Interpretation:** the leak is NOT a single swappable kernel — it's intrinsic to the
**DSV4 decode-graph capture/replay machinery running under TBO mode** (the 3-way combo
DSV4 × TBO × decode-cuda-graph). The only proven-stable cut is removing the decode
cuda-graph (round 2). hipGraph and HSA-signal/queue interposers are both blocked by
ROCm symbol versioning (`@@hip_4.x`, `@@ROCR_1`) → LD_PRELOAD won't bind; need
roctracer/rocprofv3 (official HIP+HSA callback API, version-agnostic) to count which
HSA resource diverges between DSV4+TBO (crash) and R1+TBO (stable). That is the next
real probe; the blind-swap avenue is exhausted.

**Shippable status:** DSV4 EP+TBO is correct + stable with `--disable-cuda-graph`
(decode eager). The cuda-graph stability fix needs the roctracer deep-dive (or an
upstream/ROCm-runtime fix) — deferred pending decision.

### Stability debug round 8 — TBO-infra-present vs TBO-executed: CRASH NEEDS BOTH prefill-TBO-EXEC × decode-graph (2026-06-25 PM)

Key realization: DSV4 decode is NON-TBO (model `_can_run_tbo` gates EXTEND only), so
the decode graph runs the SAME kernels with/without TBO. So what differs between
DSV4+TBO+graph (crash) and DSV4-noTBO+graph (stable) is the TBO *infrastructure*, not
decode compute. Isolation test: keep `--enable-two-batch-overlap` (TboAttnBackend
wrapper + tbo-mode decode-graph machinery fully active) but force the model to NEVER
execute TBO (diagnostic gate `/workspace/TBO_NOEXEC` → `_can_run_tbo` returns False →
prefill also runs the normal loop). **Result: FULL gsm8k, NO crash, 0.9477/0.9484.**

**Truth table (all other knobs default, cuda-graph ON unless noted):**
| prefill-TBO EXECUTION | decode cuda-graph | result |
|---|---|---|
| OFF (infra still present) | ON | **STABLE** (round 8) |
| ON | OFF (round 2 --disable-cuda-graph) | **STABLE** |
| ON | ON | **CRASH** |

⇒ The crash requires BOTH (a) actually executing the prefill-TBO ops AND (b) decode
cuda-graph. TBO-infra presence alone is harmless; prefill-TBO-exec alone (eager
decode) is harmless. It's their INTERACTION. (So my op code is implicated — not just
the wrapper — but only in combination with decode-graph.)

**Leading mechanism (strong):** executing prefill TBO is the ONLY thing that uses the
**2nd mori inner dispatcher** — `MaybeTboDeepEPDispatcher` builds 2 `MoriEPDispatcher`s
when TBO enabled, and `_execute(name, tbo_subbatch_index)` routes subbatch 1 →
inner[1]. In NOEXEC, only inner[0] is ever used (subbatch_index None→0), so inner[1]
never lazily-initializes its mori shmem / HSA queues/signals. In EXEC, both
dispatchers go live → ~2× mori HSA control resources (queues/signals, RDMA-style),
which, combined with decode-graph's HSA reservations + DSV4's higher per-layer HSA
baseline (compressor/indexer/unified_kv), exceeds the ROCm per-process HSA
queue/signal ceiling. R1+TBO+graph survives because R1's lighter per-layer kernels
leave more HSA headroom. Fits all data incl. round-1 (mori async_finish off only
*delayed* it — removed mori's per-call events but not the 2nd dispatcher's base HSA
queues). Note hipEvent/hipStream were flat → these are HSA-level (below hip), not
hipEvent/hipStream.

**Next (confirm + fix candidates):**
- Confirm: force `MaybeTboDeepEPDispatcher._execute` to always use inner[0] (diag;
  may deadlock — risky) OR count mori HSA queues. Cleaner: HSA tool via `HSA_TOOLS_LIB`
  (official, version-agnostic; env IS inherited by schedulers like LD_PRELOAD was) to
  count `hsa_queue_create`/`hsa_signal_create` over time + diff EXEC vs NOEXEC.
- Fix candidates: (i) mori runtime env tuning the aligned script omits (the standalone
  `run_sgl_dsv4_mori-ep.sh` sets SGLANG_MORI_QP_PER_TRANSFER/NUM_WORKERS, MORI_IO_QP_MAX_*,
  MORI_SHMEM_MODE=ISOLATION, etc.) — may bound mori's HSA queue allocation; (ii) make
  the 2 TBO mori dispatchers share HSA queues / reduce per-dispatcher queue count;
  (iii) reduce decode-graph HSA footprint (smaller cuda-graph-max-bs) to leave headroom.

### Stability debug round 9 — standalone MORI_* env + TBO: env-var hypothesis DISPROVEN (2026-06-25 PM)

Tested: does the standalone `run_sgl_dsv4_mori-ep.sh` env (full MORI_* runtime tuning
the aligned script omits) + `--enable-two-batch-overlap` avoid the HSA crash? Made a
variant `/workspace/run_dsv4_moriep_tbo.sh` (all MORI_* env verbatim + TBO + crash-repro
args). **Could NOT even reach serving** — OOMs during decode-graph CAPTURE at every
mem-fraction tried (0.72/0.62/0.55/0.50), even with `PYTORCH_ALLOC_CONF=
expandable_segments:True` (note: the deprecated name is `PYTORCH_HIP_ALLOC_CONF` →
silently ignored; correct name is `PYTORCH_ALLOC_CONF`). Decisive datum: capturing the
bs=8 decode graph started at `avail_mem=138.61 GB` and still OOM'd → **a single decode
graph capture consumed ~137 GB**.

**Conclusion: env-var hypothesis DISPROVEN — the MORI_* vars are NOT a fix; they make
TBO+graph WORSE.** They pre-allocate large mori dispatch/combine buffers
(`SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384`, `MORI_MAX_DISPATCH_TOKENS_PREFILL
=8192`, …) which, under TBO's 2 dispatchers × per-captured-graph HIP private pools,
explode memory. (Those vars are for multi-node RDMA PD-disaggregation — the standalone
is a PD-disagg config — and only add overhead single-node.)

**This REINFORCES the round-8 mechanism:** the same "2nd mori dispatcher × decode-graph
per-graph duplication" that causes the slow HSA-queue/signal exhaustion in the aligned
(small-buffer) config shows up here, with big buffers, as an immediate CAPTURE OOM.
Both point to the same root cause: **TBO's 2nd mori dispatcher resources get
duplicated/retained across the decode-graph capture/replay path, even though DSV4
decode is non-TBO and never needs the 2nd dispatcher.**

**Most-promising fix direction (next):** since DSV4 decode is non-TBO (never uses
subbatch 1 → inner[1]), make the decode-graph path NOT instantiate/capture/retain the
2nd mori dispatcher's resources — e.g. share the 2 TBO dispatchers' queues, or skip the
2nd dispatcher entirely for the (non-TBO) decode graph. The DSV4 op-decomposition is
correct (gsm8k 0.96); the fix is in the TBO×mori×decode-graph resource handling, not
the ops.

### Stability debug round 10 — make DSV4 decode ALSO TBO (R1-consistency hypothesis): DISPROVEN, worse (2026-06-26)

Hypothesis (user): R1's `_compute_moe_deepseek_layer_operations_strategy_tbo` handles
BOTH prefill AND decode, so R1 decode is also TBO → prefill/decode TBO state is
*consistent*. DSV4 was prefill-only → *inconsistent* (prefill TBO, decode non-TBO but
graph captured tbo=true). Maybe the inconsistency causes the leak; make DSV4 decode TBO
too (like R1) → consistent → maybe stable. (The truth table fit: NOEXEC=all-non-TBO=stable,
R1=all-TBO=stable, DSV4=half-TBO=crash.)

Implemented: added `_compute_moe_deepseek_v4_decode` (mirrors R1 decode: delta_stages=2,
4 yields, op_attn unsplit), allowed DECODE/TARGET_VERIFY in `_can_run_tbo`, reverted
`tbo_supports_cuda_graph` (children now needed for the TBO decode graph).

Results:
- Decode-TBO graphs are MUCH bigger: **70.95 GiB in HIP-Graph private pools** (vs ~5–26
  GiB non-TBO decode) → capture-OOM at mem 0.72; needed mem 0.60 + cuda-graph-max-bs 32
  + `PYTORCH_ALLOC_CONF=expandable_segments:True` to reach serving (decode graph drives
  2 children backends + 2 mori dispatchers).
- **Still crashes — but now in ~15 s / 5 reqs (vs 7–13 min before).** Same
  HSA_STATUS_ERROR_OUT_OF_RESOURCES. ⇒ decode TBO makes it strictly WORSE: the decode
  graph now uses 2 children + 2 mori dispatchers per step → HSA resources exhaust almost
  immediately instead of accumulating over minutes.

**Conclusion: the "consistency" hypothesis is DISPROVEN.** R1 is stable NOT because of
prefill/decode TBO consistency, but because R1's per-layer kernels are lighter (lower
HSA baseline). DSV4's heavier per-layer kernels (compressor/indexer/unified_kv) ×
TBO-graph machinery (children + 2nd mori dispatcher) exceed the ROCm HSA ceiling; adding
MORE TBO-graph machinery (decode TBO) only accelerates the exhaustion. This is strong
additional confirmation of the round-8 mechanism (DSV4-specific HSA weight × TBO-graph
machinery). Reverted all decode-TBO edits → back to committed prefill-only
(`e299a3385`). Usable config remains prefill-TBO + `--disable-cuda-graph`.

### Stability debug round 11 — OP-STUB BISECTION → culprit is the mHC ops (2026-06-26)

Added file-gated diagnostic stub strategies (shape-correct passthroughs; wrong
numerics, crash-test only): `STUB_ATTN` (skip DSV4 compressor/indexer/MQA in op_attn),
`STUB_MOE` (skip all MoE/mori a2a), `STUB_ALL` (whole layer = passthrough). Bisection
at conc16/cuda-graph-ON (the crash config):

| prefill-TBO runs | result |
|---|---|
| mHC + attn + MoE (none) | CRASH |
| mHC + attn (STUB_MOE) | CRASH → NOT the mori a2a |
| mHC only (STUB_ATTN+MOE) | CRASH → NOT the DSV4 attention |
| **nothing / passthrough (STUB_ALL)** | **STABLE, full gsm8k (acc 0.834, degraded by passthrough but RAN)** |

⇒ **The HSA exhaustion is caused by the DSV4 mHC ops (`hc_pre`/`hc_post`)** executing
under prefill TBO. RULES OUT the mori a2a (2nd dispatcher — my earlier round-8 theory
was WRONG) and the DSV4 attention. (Decode mHC is identical across all stubs — only the
prefill-TBO mHC differs — so it's the prefill-TBO mHC execution.)

**Sub-mechanism candidate:** with the env's fast mHC paths OFF
(`SGLANG_OPT_USE_TILELANG_MHC_PRE/POST=false`, `SGLANG_OPT_DEEPGEMM_HC_PRENORM=false`,
aiter MHC unset), `hc_pre`/`hc_post` use nested torch impls decorated with
`@compile_in_capture_mode` (= `torch.compile` when capture-mode, plain torch else;
`runner_utils/capture_mode.py`). Under TBO the mHC ops run on 2 ubatches with
CONTINUOUSLY-VARYING padded shapes (tbo_padded_len differs per prefill batch) →
suspected repeated Triton/inductor (re)compilation accumulating ROCm HSA
module/queue resources. NEXT: switch mHC to the aiter kernel path
(`SGLANG_OPT_USE_AITER_MHC_PRE/POST=1`, no torch.compile) — confirms the torch-compile
mHC path AND is a candidate fix.

### Stability debug round 12 — CORRECTION: not mHC-specific; it's ANY real prefill-TBO kernel work × decode-graph (2026-06-26)

Round-11 concluded "mHC is the culprit" (STUB_BOTH=mHC-only crashed, STUB_ALL=passthrough
stable). Two follow-ups REFUTE the mHC-specific conclusion:
- **aiter-mHC OFF** (`SGLANG_OPT_USE_AITER_MHC_PRE/POST=0` → torch mHC path instead of
  the aiter mHC kernel): STILL CRASHES, even FASTER (~3%). ⇒ not the aiter mHC kernel
  specifically. (Aside: aiter MHC defaults True; the crashing config already used it.)
- **STUB_MHC** (real attn + real MoE/mori, mHC kernels stubbed to cheap norms): CRASHES
  (~12% / 2:16, decode graph active). ⇒ attn+MoE WITHOUT mHC also crashes.

**Full bisection truth table (prefill-TBO content; conc16, cuda-graph ON):**
| prefill-TBO runs | result |
|---|---|
| mHC + attn + MoE | CRASH |
| attn + mHC (STUB_MOE) | CRASH |
| mHC only (STUB_BOTH) | CRASH |
| **attn + MoE, no mHC (STUB_MHC)** | **CRASH** |
| **nothing / passthrough (STUB_ALL)** | **STABLE** |

**Corrected conclusion: it is NOT a specific subsystem.** mHC-alone crashes AND
attn+MoE-without-mHC crashes; only the pure passthrough (≈no real GPU kernels) is
stable. So the trigger is **executing ANY real DSV4 GPU kernels on the 2 TBO ubatches
(continuously-varying padded shapes) combined with decode cuda-graph.** Common factor
across mHC/attn/MoE that passthrough lacks: real shape-specialized kernel launches
(Triton/rocBLAS/aiter JIT + autotune) on the ever-changing tbo_padded_len ubatch shapes.

**Working model of the root cause:** decode cuda-graphs pin a fixed set of HSA
code-objects/queues; prefill TBO then launches kernels on continuously-varying ubatch
shapes → per-shape kernel JIT/autotune keeps allocating HSA module/queue resources from
the shrunken remaining pool → accumulates over minutes → HSA_STATUS_ERROR_OUT_OF_RESOURCES.
Fits ALL data: --disable-cuda-graph stable (full HSA pool for prefill JIT, no pinned
graphs); NOEXEC stable (no TBO ubatch shapes → no extra JIT); R1 stable (lighter kernels
/ fewer distinct shape-kernels); passthrough stable (≈no kernels); crash time variable
(accumulation). 

**Actionable fix direction:** bound the TBO ubatch shape variety — pad ubatches to a
small fixed bucket set (instead of every tbo_padded_len) so prefill TBO reuses a bounded
set of shape-specialized kernels → no unbounded HSA accumulation. (Alternative: cap
torch/Triton/aiter JIT cache, or pre-warm+pin the ubatch-shape kernels.) NEXT: test
ubatch shape bucketing.

All STUB_* diagnostics are file-gated working-tree changes (not committed); the committed
branch `e299a3385` is the clean prefill-only feature. Usable config: prefill-TBO +
`--disable-cuda-graph`.

### Round 13 — ATOM TBO shape handling (traced 2026-06-26): ATOM BUCKETS ubatch shapes

Traced ATOM's TBO (`/opt/venv/.../atom/utils/tbo/ubatch_wrapper.py`,
`ubatch_splitting.py`) for the user's question "does ATOM use finite/bounded shape
specialization?". **Yes:**
- **Decode (graphed):** TBO CUDA graphs keyed by `(graph_bs, max_q_len)`
  (`ubatch_wrapper.py:51,307`) — a BOUNDED set. Each ubatch padded to FIXED
  `padded_bs = full_graph_bs // N` (`:97-100,358-361`); `split_attn_metadata` pads all
  metadata up to `padded_bs` (`ubatch_splitting.py:199-350`). → bounded shape set → no
  unbounded kernel JIT/HSA accumulation.
- **Prefill:** eager only (`:336` "For prefill (eager only)"), `padded_bs=ub_num_reqs`
  (no bucket), BUT per-ubatch token count = cross-DP-rank MAX
  (`ctx.ub_max_tokens_across_dp`, `:343-349`) so all DP ranks share one shape/step.

Contrast sglang DSV4 (mine): prefill TBO pads `tbo_padded_len` only to attn_tp_size
(=1 under dp-attn) → exact, continuously-varying per-rank ubatch shapes; my round-10
decode-TBO also did NOT bucket → varying decode-graph shapes → crashed faster. ATOM
never feeds unbounded TBO shapes into a graph.

⇒ Confirms the round-12 root-cause model and the fix direction: **bound the TBO ubatch
shapes** (pad to a small fixed bucket set, à la ATOM's `graph_bs // N`), so prefill TBO
reuses a bounded set of shape-specialized kernels. NEXT: implement ubatch-shape
bucketing in the DSV4 TBO split/pad path and re-test with cuda-graph ON.

### Round 14 — FIX FOUND: TBO ubatch-shape BUCKETING stops the HSA crash (2026-06-26)

Implemented the round-12/13 fix: in `TboForwardBatchPreparer.filter_batch`
(`two_batch_overlap.py`), round `tbo_padded_len` UP to the next power-of-2 (≥256)
instead of the exact per-batch token count (attn_tp_size=1 → no padding). This bounds
the TBO ubatch shapes to {256,512,1024,2048,4096,8192,…} (like ATOM's graph_bs//N), so
prefill TBO reuses a BOUNDED set of shape-specialized kernels. `_pad_inputs_to_size`
already pads input_ids/positions/out_cache_loc to tbo_padded_len; num_token_non_padded
(active for EP) marks the real tokens so the padding is handled. File-gated
`/workspace/TBO_BUCKET` for the test.

**RESULT: DSV4 EP+TBO with cuda-graph ON ran the FULL gsm8k (1319) with NO crash,
accuracy 0.9507/0.9500** (correct band; matches non-TBO mori 0.9431/0.9522 and
prefill-TBO 0.96 — bucketing did NOT break attention/numerics). First stable
DSV4+TBO+decode-cuda-graph run. Confirms the root-cause model: continuously-varying
ubatch shapes → per-shape kernel JIT/autotune → unbounded ROCm HSA accumulation →
OUT_OF_RESOURCES; bounding the shapes fixes it.

**Status:** the bucketing logic lives in `filter_batch` gated by file
`/workspace/TBO_BUCKET` (diagnostic). TODO to ship: convert to a proper env/registered
flag (or always-on for the TBO path), scope it so it only affects models that need it
(or verify it's neutral/positive for R1), tune the bucket policy (pow2 is coarse →
some wasted padding compute; could use a finer/cAPPED bucket set), and benchmark the
overhead. The committed branch `e299a3385` is still the prefill-only feature without
this fix; STUB_* diagnostics + the bucket gate are uncommitted working-tree changes.

### Round 15 — FORMALIZED the fix + committed (2026-06-26)

- Removed ALL STUB_* / RESMON / TBO_NOEXEC diagnostic scaffolding (`git checkout`
  restored the clean committed feature for `operations_strategy.py` + `deepseek_v4.py`,
  since the stubs were pure additions).
- Converted the file-gated bucketing to a registered env: **`SGLANG_TBO_PAD_BUCKET`**
  (`EnvBool(False)` in `environ.py`); `two_batch_overlap.py:filter_batch` rounds
  `tbo_padded_len` up to next pow2 (≥256) when set. Default OFF → no behavior change for
  existing TBO models (R1 etc.); opt-in for DSV4.
- Branch `feat/dsv4-ep-tbo-prefill`, 2 commits now:
  - `61d6ca33` feat: DSV4 EP/mori prefill TBO op-decomposition (was e299a3385; repo
    auto-sync re-hashed it).
  - `0b4fc8bc` fix(tbo): SGLANG_TBO_PAD_BUCKET (environ.py + two_batch_overlap.py).
- `cohere2_moe.py` @strict workaround left uncommitted (launch-only, unrelated).

**Usage:** DSV4 EP+TBO with cuda-graph ON now stable by adding `SGLANG_TBO_PAD_BUCKET=1`
to the launch env (aligned `EP_MODE=mori` + `--enable-two-batch-overlap`, no need for
`--disable-cuda-graph`). Open follow-ups: benchmark bucketing overhead + EP+TBO vs EP
throughput; consider finer bucket policy; upstream review.

### Round 16 — TBO perf characterization + bucketing overhead + granularity (2026-06-26)

Benchmarks via `sglang.bench_serving` (sglang-oai), workload 8192:1024, mori EP, mem
0.72. (Server max-running/cuda-graph-bs capped by the TBO+graph capture HSA/mem limit:
DSV4 c64/cg64, R1 c32/cg32.)

**Throughput table (total tok/s, 8k/1k):**
| config | conc | total tok/s | input tok/s | TTFT ms | TPOT ms | vs EP |
|---|---:|---:|---:|---:|---:|---:|
| DSV4 EP (no TBO) | 64 | 4504.7 | 4004.2 | 19529 | 98.1 | — |
| DSV4 EP+TBO+bucket(pow2) | 64 | 4371.3 | 3885.6 | 21670 | 99.8 | **-3.0%** |
| R1 EP (no TBO) | 32 | 5841.3 | 5192.2 | 12911 | 31.5 | — |
| R1 EP+TBO (no bucket) | 32 | 3609.4 | 3208.3 | 20498 | 51.6 | **-38%** |
| R1 EP+TBO+bucket(pow2) | 32 | 3596.9 | 3197.2 | 20783 | 51.6 | (vs TBO-off: **-0.3%**) |

**Findings:**
1. **EP+TBO vs EP: TBO HURTS at the achievable (small-batch) configs.** DSV4
   prefill-only TBO −3%; R1 −38% (R1's `init_new_tbo` enables DECODE TBO too, which
   regresses — TPOT 31.5→51.6 and TTFT also worse). TBO's prefill-overlap win needs
   LARGE prefill batches (ATOM: +13% @BS512); at c32–64 with max-running 32–64 the
   ubatches are small and the TBO machinery (op-list interleave, 2 mori dispatchers,
   ubatch split/pad) overhead dominates. The large-batch regime that would favor TBO
   can't be reached here because TBO+decode-cuda-graph capture OOMs / hits the HSA
   ceiling above cg-bs ~64. ⇒ **TBO is not a throughput win in the currently-runnable
   regime.**
2. **Bucketing overhead: NEGLIGIBLE (~0.3%).** R1 EP+TBO bucket OFF 3609.4 vs pow2
   3596.9 = −0.3% (within noise). So the SGLANG_TBO_PAD_BUCKET fix is essentially free.
   (Caveat: at c32 prefill, TBO may engage on few batches — no [TBO_PAD] log fired —
   so this is an upper-ish bound, but padding cost is clearly tiny.)
3. **Finer bucket policy is NOT viable.** mult512 (multiples of 512) CRASHED on DSV4
   (~52% gsm8k, HSA OUT_OF_RESOURCES), while pow2 is stable. Finer buckets → more
   distinct shapes → HSA accumulation returns. **pow2 (≈6–7 buckets) coarseness is
   NECESSARY, not just sufficient** → keep pow2.
4. **R1 TBO+graph capture is itself near the HSA edge** (non-deterministic): one R1
   EP+TBO launch crashed during cuda-graph capture at cg-bs 64; cg-bs 32 was stable.
   Consistent with the HSA-resource-pressure root cause being broader than DSV4.

**Net recommendation:** keep the bucketing fix (free + necessary for stability), but
TBO itself is not currently worth enabling for throughput at the runnable batch sizes;
revisit if the capture memory/HSA ceiling is raised to allow large-batch TBO. All
experiment instrumentation (mode-file bucket sweep + padding counter in
`two_batch_overlap.py`, `/workspace/run_r1_mori_*.sh`) is uncommitted/temporary; the
committed branch has only the clean feature + SGLANG_TBO_PAD_BUCKET.

### Round 17 — GPU_MAX_HW_QUEUES is the REAL stability fix (from ATOM InferenceMax PR #1717); but DSV4 TBO still regresses throughput (2026-06-26)

User pointed to InferenceX PR #1717 (dsv4-fp4-mi355x-atom). Key diffs: ATOM enables
TBO with `--enable-dp-attention --enable-tbo` (= PREFILL-ONLY, same scope as our DSV4
TBO) only at HIGH conc (ISL8192/OSL1024 conc≥256; ISL1024 conc≥1024), and crucially
sets **`export GPU_MAX_HW_QUEUES=5`** + **`--cudagraph-capture-sizes
[1,2,4,8,16,32,48,64,128,256,512]`** (bounded list) whenever TBO is on. ATOM's prefill
TBO is NOT shape-bucketed (`padded_bs = ub_num_reqs`).

**`GPU_MAX_HW_QUEUES=5` is the real fix** (caps ROCm HW queues = exactly the resource
our HSA_STATUS_ERROR_OUT_OF_RESOURCES exhausts). Test: DSV4 EP+TBO, **NO bucketing**,
`GPU_MAX_HW_QUEUES=5`, **conc 256, cg-max-bs 256**: full bench, **NO crash**, 10,626
tok/s. (The bucketed run at conc256 had died at 66 prefill batches; here 2058 prefill +
168 decode ran clean.) ⇒ GPU_MAX_HW_QUEUES caps the HW-queue pool so streams multiplex
instead of exhausting; it both FIXES the crash AND enables high concurrency, with no
bucketing. **My SGLANG_TBO_PAD_BUCKET fix was an indirect low-conc workaround,
superseded by this.**

**BUT — proper EP vs EP+TBO A/B at high conc (256), GPU_MAX_HW_QUEUES=5, no bucket:**
| config | total tok/s | input tok/s | TTFT ms | TPOT ms |
|---|---:|---:|---:|---:|
| DSV4 EP (no TBO) | **12,255** | 10,894 | 25,011 | 119.9 |
| DSV4 EP+TBO | 10,626 | 9,446 | 39,816 | 133.7 |
| | **−13%** | −13% | **+59% (worse)** | +11% |

**TBO REGRESSES DSV4 throughput even at high conc (−13%, TTFT +59%).** So my prefill-TBO
op-decomposition does NOT deliver the comm/compute overlap benefit — prefill gets
SLOWER, not faster. (At conc64 it was −3%; at conc256 −13%.) The op-list interleave +
2 mori dispatchers + doubled per-ubatch mHC/attn processing outweigh whatever a2a
overlap is achieved. This contradicts ATOM's reported +13%; likely my op strategy
doesn't overlap effectively for DSV4 (heavy mHC/attn per ubatch), or needs a different
yield/stagger layout, or DSV4's eager prefill TBO doesn't pipeline like ATOM's
thread+dual-stream design.

**Net conclusions of the whole TBO effort:**
1. DSV4 EP+TBO op-decomposition: implemented + numerically correct (gsm8k 0.95–0.96).
2. Stability: **use `GPU_MAX_HW_QUEUES=5`** (ATOM-proven; caps HSA HW queues) — fixes
   the OUT_OF_RESOURCES crash at all concurrencies, no bucketing needed. (SGLANG_TBO_
   PAD_BUCKET works at low conc but doesn't scale and is superseded.)
3. Performance: **TBO is NOT a throughput win for DSV4 as implemented** (−3% to −13%,
   TTFT worse) — the overlap isn't materializing. Investigating the op strategy /
   overlap effectiveness is the next step if TBO is to pay off.

### Round 18 — ATOM EP-vs-DP cross-check: DSV4 TBO win is the DP path, not EP (2026-06-26)

User insight: we tested EP+TBO, but ATOM's DSV4 InferenceX recipe (PR #1717) is DP
(ep=1)+TBO. Ran ATOM DSV4 @conc256 8k/1k (GPU_MAX_HW_QUEUES=5, max-num-seqs 256):
- **ATOM DP (no TBO): 30,481 tok/s; ATOM DP+TBO: 32,819 → +7.7%, TTFT 20.4s→17.3s.**
  ⇒ DSV4 TBO IS POSITIVE on the DP/no-EP path (overlaps all_gather+reduce_scatter).
- ATOM EP/mori for DSV4 would NOT launch (mori `Out of static heap memory`,
  MORI_SHMEM_HEAP_SIZE env didn't take) → ATOM doesn't run DSV4 on EP/mori; uses DP.
- vs sglang EP(mori)+TBO −13%. ⇒ **EP/mori is the WRONG TBO target for DSV4; the DP/
  no-EP all_gather+reduce_scatter overlap is the productive one.** My sglang impl
  (reusing DeepseekV2MoE deepep ops) targeted EP/mori. (Also ATOM ≈2.5–3× sglang
  absolute at this point — EP/mori overhead + engine.)
- Next: re-target sglang DSV4 TBO to the non-EP DP path (async gather/scatter ops).
  Recorded in TBO_RESEARCH.md §9.

## Launch recipe (TBO)
```
cd useful-scripts/benchmarking/dsv4
PYTHONPATH=/sgl-workspace/sglang-upstream/python:/sgl-workspace/mori:/sgl-workspace/aiter \
EP_MODE=mori \
SGL_EXTRA_ARGS="--enable-two-batch-overlap --mem-fraction-static 0.72 --cuda-graph-max-bs 64 --max-running-requests 64" \
bash run_sgl_dsv4_aligned.sh
```
gsm8k: `lm_eval --model local-completions --model_args
model=$MODEL,base_url=http://localhost:8000/v1/completions,num_concurrent=8,...
--tasks gsm8k --num_fewshot 5 --limit 300`.

## Still TODO
- Fix the HSA OUT_OF_RESOURCES stability bug (blocks usable TBO).
- Full (unlimited) gsm8k once stable.
- EP vs EP+TBO throughput A/B (verify the prefill-overlap win).
- (later) decode TBO + cuda-graph capture (currently NotImplemented; ATOM says
  decode TBO regresses anyway).
