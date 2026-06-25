# DeepSeek-V4-Pro serving perf — experiment log (2026-06-23)

Daily log split out from `EXPERIMENT_LOG.md` (which grew to >3.4k lines). Same
project/hardware/clients as the main log; see `SKILL.md` for how-to and
`HANDOFF.md` for the running summary.

- **Date**: 2026-06-23
- **Model**: `/dockerx/data/deepseek-ai/DeepSeek-V4-Pro/` (FP8)
- **sglang clone (active editable)**: `/sgl-workspace/sglang-upstream/python`
  (HEAD `4f174ce74`); contains shipped gatherv + A-fix (PR #28216) + ported
  SE-local.
- **Aligned launch**: `useful-scripts/benchmarking/dsv4/run_sgl_dsv4_aligned.sh`
  (tp8 + dp8 dp-attention, dsv4 backend, fp8 kv, `--chunked-prefill-size 65536`
  via `SGL_EXTRA_ARGS`, cuda-graph-max-bs 512, max-running 512, prefill-delayer ON).
- **"Best" perf config (37.2k baseline)**: env
  `SGLANG_USE_ROCM700A=0 SGLANG_DP_USE_GATHERV=1 SGLANG_SHARED_EXPERT_TP1=1
  SGLANG_DP_SHARED_EXPERT_LOCAL=1`.
- **Client**: `python3 -m sglang.bench_serving` via `sweep_dsv4_sglang_client.sh`
  (`nprompts=conc*8`, `nwarm=conc*2`, ratio 1.0, request-rate inf, temp 0).
- **GPU cleanup recipe** (avoids `pkill -f` self-matching the shell wrapper):
  `for p in $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}'); do kill -9 $p; done`

Raw artifacts under `/sgl-workspace/dsv4_bench/`:
`trace_atom_8k1k_decode/`, `trace_c512_8k1k_upstream_v2/decode/` (baseline decode),
`trace_rocm1_decode/`, `trace_aiter_rs_decode/`, `rocm700a_ab/`,
`aiter_rs_poc/`, `aiter_rs_poc_1k/`, `comm_microbench.py`.

---

## A — ATOM decode-heavy (wave-tail) single trace

Goal: an ATOM decode-only trace to compare against the sglang decode trace.

Method: ATOM single-stream (`ATOM_DISABLE_SIDE_STREAMS=1`) + profiler dir; fire a
wave (512 reqs, 8k in / **OSL 3000** to lengthen the decode tail so the capture
window is forgiving — ATOM has no batch logs); wait ~95s for prefill to drain;
profile 12s during decode.

Result: `trace_atom_8k1k_decode/dp0_tp0/...061919...` — 247 decode steps
(`sample`), graph-internal per-layer kernels present (`qk_norm_rope` 122 ≈ 2-step
snapshot under cuda graph). Comparable to the sglang decode trace (266 samples).

---

## B — Root cause: sglang decode uses all_reduce, not reduce_scatter

Observation (decode trace): after `mfma_moe2`, sglang used aiter
`cross_device_reduce_2stage` (all_reduce) while ATOM used `reduce_scatter`.

Gating chain (DSV4 MoE combine, `deepseek_v4.py` `_use_tp_moe_gather` branch):
reduce_scatterv is taken only if `should_use_dp_reduce_scatterv() or _use_gatherv_pair`.
- `should_use_dp_reduce_scatterv()` needs `moe_ep_size == attn_dp_size` → **False**
  (no EP; tp8dp8 only).
- `_use_gatherv_pair` needs `SGLANG_DP_USE_GATHERV` on AND `dp_padding_mode` ==
  SUM_LEN (`not is_max_len()`).

Two facts combined to give all_reduce in the *baseline decode trace*
(`trace_c512_8k1k_upstream_v2/decode/`):
1. That trace was captured with `SGLANG_DP_USE_GATHERV` **off** (the aligned script
   doesn't export it) → gatherv path disabled entirely → 61×
   `cross_device_reduce_2stage`, 0 reduce_scatter, 62 `allgather_vec`, 266 samples.
2. Even with gatherv ON, **decode** under cuda graph takes
   `DpPaddingMode.get_default_mode_in_cuda_graph()`, which returns **SUM_LEN when
   `SGLANG_USE_ROCM700A=1`** and **MAX_LEN when =0**. Only SUM_LEN enables
   `_use_gatherv_pair`.

So: decode uses reduce_scatterv only when `gatherv ON` **and** `ROCM700A=1`.

---

## C — ROCM700A A/B (0 vs 1), 8k/1k c512

Fixed: gatherv + SE-local + TP1 + chunk 65536; only flip `SGLANG_USE_ROCM700A`.
Full client (np4096 / warm1024), single run each. Dirs `rocm700a_ab/{A_rocm0,B_rocm1}`.

| metric | A: ROCM700A=0 (decode all_reduce, MAX_LEN) | B: ROCM700A=1 (decode reduce_scatterv, SUM_LEN) | Δ |
|---|---:|---:|---:|
| Total tok/s | **37,180.8** | 36,366.6 | A **+2.24%** |
| Median TPOT (ms) | 97.36 | 99.76 | A better |
| Mean TTFT (ms) | 25,212 | 25,397 | ~= |

**ROCM700A=0 wins +2.2%.** Reason (see D/E): the MAX_LEN combine uses aiter's
custom `cross_device_reduce_2stage` (quickreduce) + symmetric memory, which beats
RCCL `reduce_scatterv` even though reduce_scatter moves ~half the bytes.

---

## D — ROCM700A=1 decode trace + naming clarification

`trace_rocm1_decode/` (gatherv ON, ROCM700A=1, SE-local): confirmed decode now
uses reduce_scatterv — `cross_device_reduce_2stage` 0 (was 61), replaced by
`ncclDevKernel_Generic` (RCCL, ≈124 = 2×61 under cuda-graph dedup). 255 samples.

**cuda-graph caveat**: decode runs inside a hipGraph; torch-profiler per-kernel
durations are garbage (mfma_moe shows 161ns, index kernels 5ns). Real time is
hidden inside `hipGraphLaunch`. So decode per-kernel timing must come from a
microbench, not the trace.

**Naming clarification** (a prior statement was imprecise):
- MAX_LEN gather → `all_gather_into_tensor` → aiter custom **`allgather_vec`**
  (`vec` = vectorized, equal-length).
- SUM_LEN/gatherv gather → `all_gatherv()` → **RCCL `all_gatherv`** (variable
  length), shows as `ncclDevKernel_Generic`.
- gatherv combine → RCCL `reduce_scatterv` → also `ncclDevKernel_Generic`.

| config | gather | combine |
|---|---|---|
| ROCM700A=0 (MAX_LEN) | aiter `allgather_vec` | aiter `cross_device_reduce_2stage` (all_reduce) |
| ROCM700A=1 (SUM_LEN) | RCCL `all_gatherv` | RCCL `reduce_scatterv` |

So "ncclDevKernel + ncclDevKernel" in ROCM700A=1 is correct — both gatherv
collectives are RCCL and share the generic kernel symbol. `allgather_vec` is the
MAX_LEN aiter all-gather, NOT the gatherv gather.

---

## E — Collective microbench (RCCL), decode sizes

`/sgl-workspace/dsv4_bench/comm_microbench.py` — torchrun 8-GPU, bf16, hidden 7168.

| M_global | RCCL all_reduce | RCCL reduce_scatter | AR/RS |
|---:|---:|---:|---:|
| 256 | 53.8us | 35.7us | 1.51× |
| 512 (decode 64/rank) | 74.7us | 54.6us | 1.37× |
| 1024 | 106.2us | 78.5us | 1.35× |
| 2048 | 175.6us | 112.7us | 1.56× |

reduce_scatter is ~1.35–1.56× cheaper than **RCCL** all_reduce (≈half traffic).
But the decode all_reduce in ROCM700A=0 is **aiter quickreduce**, not RCCL — which
is why ROCM700A=0 still wins net (C). The right comparison is aiter vs aiter →
motivates F.

aiter exposes equal-chunk `custom_all_gather`/`custom_reduce_scatter` (no
variable-length `v` version, size-limited via `should_custom_ar`). In **decode**
per-rank sizes are equal → aiter equal-chunk works. Prefill is variable + too
large → must stay RCCL.

---

## F — PoC: aiter reduce_scatter for the MAX_LEN decode combine

Idea: keep MAX_LEN (ROCM700A=0, aiter `allgather_vec` gather) but replace the
combine (aiter all_reduce + dp_scatter) with an **equal-chunk aiter
`reduce_scatter`** → both halves aiter AND ~half the combine traffic. Could beat
both ROCM700A=0 and =1.

### Code changes (env-gated, default OFF on 06-23)

> Env note: on 06-23 the flag was `SGLANG_USE_AITER_RS` (raw `get_bool_env_var`,
> default OFF). On 06-24 it was renamed to **`SGLANG_DP_USE_REDUCE_SCATTER`**
> (registered `EnvBool`, **default True**, platform-agnostic — aiter on ROCm, RCCL
> elsewhere). References below use the new name.

`python/sglang/srt/distributed/parallel_state.py`:
- import `get_bool_env_var`.
- `reduce_scatter_tensor()` now tries `_maybe_aiter_reduce_scatter()` first
  (else RCCL `reg_reduce_scatter_tensor`).
- new `_maybe_aiter_reduce_scatter()` + `_has_aiter_custom_reduce_scatter()`:
  mirrors `_all_gather_into_tensor`'s aiter path — HIP + env on + ca_comm has
  `reduce_scatter`/`should_custom_ar` + contiguous + dtype ok + `should_custom_ar(input)`
  + equal-chunk (`input.shape[0] == output.shape[0]*world`); handles cuda-graph
  capture (`reduce_scatter(registered=True)` when capturing,
  `output.zero_()` on warmup), else RCCL fallback (returns False).

`python/sglang/srt/models/deepseek_v4.py`:
- gate `_use_reduce_scatter` (06-23 name `_use_aiter_rs`):
  `envs.SGLANG_DP_USE_REDUCE_SCATTER.get() and _use_tp_moe_gather and not
  _use_reduce_scatterv and not should_use_dp_reduce_scatterv() and
  dp_padding_mode.is_max_len() and tp_size==attn_dp_size` (i.e. decode/MAX_LEN,
  gatherv inactive, no EP).
- mlp call: `use_reduce_scatter = _use_cp or _use_gatherv_pair or _use_aiter_rs`
  (skip MoE-internal all_reduce → avoid double-reduce).
- combine: new `elif _use_aiter_rs:` calls
  `get_tp_group().reduce_scatter_tensor(hidden_states, global_hidden_states)`
  **directly** (NOT `dp_reduce_scatter_tensor`, which would route to RCCL
  reduce_scatterv when `SGLANG_DP_USE_GATHERV` is set). `_shared_local` add-back
  unchanged (applies after the branch).

### Correctness
gsm8k 5-shot (PoC on, full best config): **flexible 0.9507 / strict 0.9515** —
matches baseline 0.95. No double-reduce.

### Trace verification (`trace_aiter_rs_decode/`)
decode combine kernel is now aiter **`_ZN5aiter24reduce_scatter_first_dim`** (61 =
per layer); gather stays aiter `allgather_vec` (62); `cross_device_reduce_2stage`
0; `ncclDevKernel` 2 (negligible). → both halves aiter custom, not RCCL fallback.
gen throughput in this run 1521 tok/s vs 1394 (ROCM700A=1 trace).

### Throughput A/B (same session, np4096 / warm1024, single run each)

8k/1k c512 (`aiter_rs_poc/` PoC, `aiter_rs_poc/baseline/`):
| metric | Baseline | PoC | Δ |
|---|---:|---:|---:|
| Total tok/s | 37,171.8 | 37,438.1 | **+0.72%** |
| Median TPOT (ms) | 96.93 | 90.72 | −6.4% |
| Mean TTFT (ms) | 25,365 | 32,474 | +28% ⚠ |

1k/1k c512 (`aiter_rs_poc_1k/poc`, `aiter_rs_poc_1k/baseline`):
| metric | Baseline | PoC | Δ |
|---|---:|---:|---:|
| Total tok/s | 18,026.9 | 18,579.8 | **+3.07%** |
| Median TPOT (ms) | 46.20 | 44.33 | −4.0% |
| Mean TTFT (ms) | 8,767 | 9,816 | +12% ⚠ |

Win grows on the more decode-bound workload (1k/1k +3.07% vs 8k/1k +0.72%), as
expected for a decode-combine optimization.

### TTFT analysis (per-batch log) — admission timing, NOT slower prefill
From server batch logs, `Prefill batch ... #new-token: 8192 ... input throughput`:
| | prefill@8192 median | mean | p90 |
|---|---:|---:|---:|
| Baseline | 6948 tok/s | 6667.8 | 7006.7 |
| PoC | 6976 tok/s | 6758.7 | 7029.9 |

→ **prefill compute identical** (PoC marginally faster). Decode per-step faster
(matched batch ~53–62: PoC ~41.8ms vs baseline ~43.4ms). PoC runs decode at
*smaller* batches (48–59 vs 60–64) and more total forwards (prefill 806/decode 290
vs 653/235) because decode steps are quicker → the prefill-delayer/admission
rebalances, shifting individual-request prefill admission later. Queue depth
unchanged (median 13, max ~120 both). **Conclusion: TTFT rise is a
scheduling/admission side-effect of faster decode, not a prefill regression.**

### Status / next
- PoC validated: correct, decode-faster, +3.07% at 1k/1k. env-gated, default OFF,
  low risk → PR candidate.
- Open: TTFT side-effect — proposed follow-up A/B with `--enable-prefill-delayer`
  off (or delayer tuning) to confirm and recover TTFT.
