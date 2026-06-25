# DeepSeek-V4-Pro serving perf — experiment log (2026-06-15)

Split from the master `EXPERIMENT_LOG.md` (chronological, by date). See that file for the index and `SKILL.md` for how-to.

---

## Exp 34 — A-fix: gatherv prefill path was silently falling back to all_reduce — 2026-06-15

Re-opened the DP-comm line on the latest upstream (branch
`feat/dp-moe-reduce-scatter`, editable install at `/sgl-workspace/sglang-upstream`)
to chase the residual c256 apple-to-apple gap. Methodology used: module-level
CUDA-event instrumentation (continuous record, single sync at flush — NOT
per-op sync, avoids both the per-op pipeline break and the Exp 19 trace `dur`
inflation) + isolated repeated microbench. ATOM module-event compare was
attempted but is unreliable (model module not imported in ATOM workers; custom
op / compile path) — Exp 19's "ATOM per-kernel inflated" still holds.

### B1 — per-module prefill breakdown (SGLang, graph-off eager, c256)
| module (per layer) | PREFILL (16384 tok/rank) | DECODE (32 tok/rank) |
|---|--:|--:|
| attn | ~14.3 ms | 0.77 ms |
| moe  | ~16.8 ms | 0.93 ms |
| forward total | ~31 ms | ~1.7 ms |
isolated fused_moe @131072 (= dp8 gathered buffer) = 12.2 ms (moe1 4.2 / moe2 4.6
/ top-k reduce 2.4 / quant 0.5), matching the server module's 16.8 ms minus
gate+shared+gather/scatter. KEY: the MoE runs on the *gathered global buffer*
(M = 8x16384 = 131072), i.e. every rank computes all ranks' tokens — but ATOM is
the SAME in tp8dp8 no-EP (engine_core_mgr resets to dp8/tp1, then
flatten_tp_across_dp -> tp=8; both shard expert intermediate 1/8). So the global
buffer is NOT the apple-to-apple gap.

### ROOT CAUSE FOUND (the actual bug) — c256 prefill never used gatherv
Coverage probe on `_dp_gather` over a real c256 run:
| step | gather via gatherv | fallback to all_reduce |
|---|--:|--:|
| PREFILL | **0** | **985 (100%)** |
| DECODE  | 15 | 0 |
dbg: `sizes=[3,3,3,3,3,3,3,3] sum=24  buffer_rows=129325`. `_dp_gatherv_sizes()`
returns `global_num_tokens_for_logprob_cpu` (the LOGPROB token counts, 3/rank),
not the MoE `global_num_tokens_cpu` (~16384/rank). Its sum (24) never equals the
ceil_align'd global buffer (129325), so the `sum(sizes)==buffer_rows` guard fails
and PREFILL always falls back to the heavier all_reduce. Only pure-decode (sizes
happen to match) ever took gatherv. ATOM, by contrast, unconditionally takes the
variable-length all_gatherv whenever any rank has prefill
(`dp_uniform_decode = not any_rank_has_prefill`, model_runner.py:1733).

### A-fix
`_dp_gather` now uses `get_dp_global_num_tokens()` (the buffer-aligned sizes
stored by set_dp_buffer_len, the SAME source the reduce_scatterv combine uses)
as the gatherv sizes; `_dp_gatherv_sizes()` is only the fallback (logits path).
One-line logic change, env-gated only by the existing SGLANG_DP_USE_GATHERV (no
new flag). Committed as `[DP] fix gatherv prefill path: use buffer-aligned sizes`.

### Verification
- Functional: c256 prefill gather now `sizes=[16193,16168,...] sum=129104 ==
  buffer`, ZERO mismatch -> all_gatherv taken on every prefill step.
- gsm8k 5-shot (OFF vs ON, chunk16384/rank, delayer on):
  OFF 0.9484 / ON 0.9431 (strict), Δ -0.53% within noise, 0 server errors.
- Throughput A/B at c256 (np1024, gatherv-ON, only the prefill-fix toggled):
  pre-fix 25,160 -> A-fix 25,885 tok/s (**+2.9%**), median TPOT 84.2 -> 82.0
  (-2.6%), TTFT 2161 -> 2108 (-2.4%).

### Concurrency sweep (gatherv OFF vs ON-with-A-fix, 8k/1k, np=conc*4)
| conc | OFF tok/s | ON tok/s | Δtput | OFF TPOT | ON TPOT |
|---:|---:|---:|---:|---:|---:|
| 64  | 13,157 | 13,387 | +1.7% | 39.9 | 39.3 |
| 128 | 18,579 | 19,421 | +4.5% | 56.4 | 54.2 |
| 256 | 24,937 | 25,754 | +3.3% | 84.8 | 82.3 |
| 512 | 30,255 | 31,637 | +4.6% | 127.5 | 122.0 |
Win grows with concurrency (heavier per-step collective -> larger all_gatherv vs
all_reduce saving). vs ATOM single-stream c256 (26,266): A-fix reaches ~98.5%
(gap ~4% -> ~1.5%).

### C2 — checked the gate/router-gather difference (rejected as a lever)
ATOM computes router locally (M=16384) then cat+gathers hidden+router in one
collective; SGLang gathers hidden only, then computes gate on the global buffer
(M=131072), i.e. an 8x-redundant gate GEMM. Isolated microbench made it look big
(gate GEMM 7168->384: 112us@16384 vs 623us@131072), BUT in-server module-event
measurement shows gate is only **0.068 ms/layer, 7.2% of the MoE module** at
prefill (the small GEMM overlaps with neighbors; isolated absolute time is not
representative). Expected refactor upside <~6%/layer of MoE and likely far less
end-to-end -> LOW ROI, high risk. Not pursued.

### Net
The A-fix is a clean, correct, ATOM-aligned bug fix that makes the existing
gatherv feature actually fire on prefill. c256 apple-to-apple now ~98.5% of ATOM.
Remaining ~1.5% is small; further compute-side levers (attn MLA breakdown, top-k
combine reduce ~19% of MoE) have diminishing ROI, and EP would beat ATOM but
breaks apple-to-apple.

Artifacts: dp_attention.py (`_dp_gather` fix); /workspace/b1_instrument.py,
/workspace/b1_iso_moe.py, /workspace/c1_coverage.py, /workspace/c2_gate_instrument.py
(env/sentinel-gated probes, removed from site-packages after use);
/workspace/run_afix_perf_ab.sh, /workspace/bench_afix/; /workspace/run_c8_sweep.sh,
/workspace/bench_c8_sweep/; /workspace/run_gatherv_gsm8k_ab.sh, /workspace/gsm8k_gatherv/.

### Exp 34 appendix — ATOM-matched sweep (np=conc*8, warm=conc*2) + 3-way table
The Exp 34 throughput A/B and the C8 sweep used np=conc*4. To compare
apple-to-apple with the ATOM Exp 1/13 numbers (which use np=conc*8,
warm=conc*2, ratio0.8), re-ran the SGLang A-fix sweep with the SAME client
settings (gatherv ON incl. A-fix, chunk16384/rank, delayer on, single-stream,
graph on, cons 1.0). Results: /workspace/bench_c9_aligned/, run_c9_aligned_sweep.sh.

SGLang A-fix detailed table (interact = 1000/median_TPOT; tok/s/gpu = total/8):
| workload | TP,DP | conc | total tok/s | tok/s/gpu | out tok/s | Med TTFT ms | Med TPOT ms | Med ITL ms | interact | Med E2E ms |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 8k/1k | 8,8 | 64  | 12,498 | 1,562 | 1,384 | 2,020  | 43.0  | 28.2 | 23.2 | 41,125  |
| 8k/1k | 8,8 | 128 | 18,809 | 2,351 | 2,084 | 1,883  | 57.8  | 32.2 | 17.3 | 54,929  |
| 8k/1k | 8,8 | 256 | 25,258 | 3,157 | 2,806 | 2,018  | 87.5  | 37.4 | 11.4 | 82,983  |
| 8k/1k | 8,8 | 512 | 32,322 | 4,040 | 3,591 | 12,062 | 127.2 | 44.4 |  7.9 | 128,644 |

3-way total tok/s comparison (all np=conc*8, ratio0.8, single-stream;
ATOM single/multi from Exp 13):
| conc | SGLang A-fix | ATOM single | ATOM multi | SGL/single | SGL/multi |
|---:|---:|---:|---:|---:|---:|
| 64  | 12,498 | 12,162 | 12,866 | 103% | 97% |
| 128 | 18,809 | 18,651 | 19,436 | 101% | 97% |
| 256 | 25,258 | 27,023 | 27,881 |  93% | 91% |
| 512 | 32,322 | 33,126 | 33,779 |  98% | 96% |

Median TPOT (ms):
| conc | SGLang A-fix | ATOM single | ATOM multi |
|---:|---:|---:|---:|
| 64  | 43.0  | 44.8  | 42.6  |
| 128 | 57.8  | 59.0  | 56.4  |
| 256 | 87.5  | 81.4  | 79.1  |
| 512 | 127.2 | 133.4 | 132.2 |

**Findings**
- vs ATOM single-stream (the true apple-to-apple): SGLang A-fix is 98-103% at
  c64/c128/c512 (matches or slightly beats), and 93% at c256 (the only clearly
  weak point: TPOT 87.5 vs 81.4).
- vs ATOM multi-stream (ATOM's best): 91-97%; the residual is largely ATOM's
  side-stream overlap (single->multi is +2-6%, Exp 13), not a dp-comm issue.
- The np=conc*8 c256 (25,258) is lower than the earlier np1024 A/B (25,885)
  purely due to the longer/heavier run (more prefill-ramp weight) — this is the
  fairer number vs ATOM. So c256 SGLang A-fix ~= 93% of ATOM single (honest),
  not the 98.5% the mixed-np estimate suggested.
- c256 remains the relative weak spot (prefill<->decode interference is worst
  there, consistent with Exp 16/B1). Further upside would need attn (MLA)
  breakdown (C3) or EP (C6, beats ATOM but breaks apple-to-apple).

