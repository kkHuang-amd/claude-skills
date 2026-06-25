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

## Open / next
- ATOM comparison at 70k/300 not yet run (this was SGLang-only by request).
- conc>32 or longer context would push DP per-rank memory; would then need the
  same swa/mem tuning DP avoided here.
- EP/mori path not tried for this workload.
