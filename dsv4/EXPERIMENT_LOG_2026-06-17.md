# DeepSeek-V4-Pro serving perf — experiment log (2026-06-17)

Split from the master `EXPERIMENT_LOG.md` (chronological, by date). See that file for the index and `SKILL.md` for how-to.

---

## Exp 39 — full re-benchmark on updated codebases (2026-06-17)

Fresh container (the prior `/sgl-workspace/sglang-upstream` clone is gone; ATOM
re-installed — `atom/__init__.py` timestamp 2026-06-17 04:05). Goal: re-measure
the tp8+dp8 numbers on the **updated** code bases over a wider grid, and re-check
SGLang↔ATOM with both engines driven by the SAME (ATOM-native) client so client
variance = 0.

### Common config (apple-to-apple)
- 8×MI355X, tp8 + dp-attention (tp8dp8), **multi-stream** (ATOM default; no
  `ATOM_DISABLE_SIDE_STREAMS`), FP8 KV, page/block 256, mem 0.90, max-running 512,
  cuda-graph-max-bs 512, **16384 prefill tokens / rank**, prefill-delayer ON.
- Client: ATOM `atom.benchmarks.benchmark_serving --backend vllm` for BOTH engines.
- Bench params: **ratio=1.0** (fixed lengths; note: differs from the 0.8 used in
  the 06-09 baseline header), request-rate inf, ignore-eos, num_prompts=conc*8,
  warmups=conc*2.
- Grid: ISL∈{1024,8192}, OSL=1024, conc∈{64,128,256,512}.
- Launch scripts: `useful-scripts/benchmarking/dsv4/run_atom_dsv4_aligned.sh`
  (DP_MODE=tp8dp8) and `run_sgl_dsv4_aligned.sh`.

### SGLang-specific config
- `SGLANG_DP_USE_GATHERV=1` (shipped gatherv+reduce_scatterv; PR #28216 is now in
  main — `SGLANG_DP_USE_GATHERV`, `reduce_scatterv`, and the A-fix
  `get_dp_global_num_tokens()` all present in the editable main clone
  `/sgl-workspace/sglang` @ 66ac385f52).
- **`SGLANG_USE_ROCM700A=0`** (per request).
- `SGL_EXTRA_ARGS="--chunked-prefill-size 131072"`: current main auto-divides
  chunked_prefill_size by dp_size when DP attention is on (server_args.py:3537
  `chunked_prefill_size //= dp_size`), so 131072 → **16384/rank** = matches ATOM.
  (The aligned script's default 16384 would give only 2048/rank — NOT aligned.)
- Needed a one-line repo fix to launch: `srt/configs/cohere2_moe.py` `@strict`
  crashes on import under huggingface_hub≥1.x (SKILL §2a). Made `strict` a no-op
  identity (no behavior change; runtime field validation only).

### Accuracy (ATOM, gsm8k 5-shot, lm_eval local-completions, num_concurrent=64)
| Filter | exact_match | stderr |
|---|---:|---:|
| flexible-extract | **0.9500** | ±0.006 |
| strict-match | **0.9492** | ±0.006 |
✓ Correct (~0.95) even WITHOUT `ATOM_USE_TRITON_MOE=1` on this updated build —
the SKILL §1 "silent wrong-MoE → ~0.6" caveat did NOT trigger here.

### ATOM throughput (updated build, multi-stream)
| ISL | OSL | conc | total tok/s | tok/s/gpu | Med TTFT (ms) | Med TPOT (ms) | Med E2E (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 64  | 4,211  | 526   | 1,307  | 29.2 | 31,132 |
| 1024 | 1024 | 128 | 7,330  | 916   | 1,725  | 33.1 | 35,783 |
| 1024 | 1024 | 256 | 12,418 | 1,552 | 3,564  | 37.1 | 41,480 |
| 1024 | 1024 | 512 | 19,828 | 2,478 | 5,489  | 45.4 | 51,641 |
| 8192 | 1024 | 64  | 14,530 | 1,816 | 5,929  | 33.3 | 40,111 |
| 8192 | 1024 | 128 | 21,609 | 2,701 | 9,545  | 43.0 | 53,988 |
| 8192 | 1024 | 256 | 30,880 | 3,860 | 18,854 | 55.4 | 75,319 |
| 8192 | 1024 | 512 | 39,291 | 4,911 | 38,630 | 80.0 | 118,296 |

vs the 06-09 baseline (SKILL.md, ATOM client, 8192:1024): c128 21,526→21,609
(+0.4%), c256 30,942→30,880 (−0.2%) — flat within noise. (Note 06-09 used
ratio0.8; here ratio1.0, but ATOM fixed-len makes this immaterial.)

### SGLang (gatherv ON, ROCM700A=0) vs ATOM — SAME client, both multi-stream
| ISL | conc | SGL tok/s | ATOM tok/s | SGL/ATOM | SGL TTFT | ATOM TTFT | SGL TPOT | ATOM TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 64  | 4,642  | 4,211  | **110.2%** | 1,458  | 1,307  | 26.1 | 29.2 |
| 1024 | 128 | 7,870  | 7,330  | **107.4%** | 2,185  | 1,725  | 30.3 | 33.1 |
| 1024 | 256 | 12,974 | 12,418 | **104.5%** | 4,016  | 3,564  | 35.3 | 37.1 |
| 1024 | 512 | 17,185 | 19,828 | 86.7%      | 7,120  | 5,489  | 44.0 | 45.4 |
| 8192 | 64  | 15,295 | 14,530 | **105.3%** | 6,525  | 5,929  | 31.2 | 33.3 |
| 8192 | 128 | 22,102 | 21,609 | **102.3%** | 12,055 | 9,545  | 40.2 | 43.0 |
| 8192 | 256 | 29,621 | 30,880 | 95.9%      | 23,124 | 18,854 | 55.2 | 55.4 |
| 8192 | 512 | 32,254 | 39,291 | 82.1%      | 55,368 | 38,630 | 72.7 | 80.0 |

### Findings
- **Low/mid concurrency (c64–c256): SGLang wins or ties.** 1024:1024 c64–c256 =
  104–110%; 8192:1024 c64/c128 = 102–105%. SGLang TPOT beats ATOM at every point.
- **High concurrency c512: SGLang regresses** (1024:1024 86.7%, 8192:1024 82.1%)
  with much higher TTFT (8192 c512: 55.4s vs 38.6s). Consistent with the standing
  conclusion: high-conc prefill↔decode interference is SGLang's weak point, and
  c512 (newly added here) is more extreme than the usual c256.
- 8192 c256 = 95.9%, slightly better than the historical ~93%.

### Artifacts
- ATOM: `/workspace/bench_results_dsv4_atom_0617/` (8 JSON + summary).
- SGLang: `/workspace/bench_results_dsv4_sgl_0617/` (8 JSON + summary).
- Logs: `/workspace/{atom,sgl}_server.log`, `/workspace/{atom,sgl}_sweep_0617.log`,
  `/workspace/gsm8k_eval.log`.
- Both servers cleaned up (kill process tree incl. DP `multiprocessing-fork`
  children; VRAM back to ~0.3 GB/GPU). NOTE: this container has no `lsof`; find the
  DP EngineCore children via `rocm-smi --showpids` / `ps` and kill the parent tree.

---

## Exp 40 — re-added ATOM_DISABLE_SIDE_STREAMS flag + ATOM single vs multi-stream (2026-06-17)

The updated ATOM build had **dropped** the `ATOM_DISABLE_SIDE_STREAMS` flag (a
knob we had added in a prior session for single-stream A/B). It was not in the new
centralized env registry (`atom/utils/envs.py`) and not read anywhere in the
package — so the earlier single-stream attempt would have been IDENTICAL to
multi-stream (caught before wasting a run). Re-added it as a single master switch.

### Side-stream architecture in the updated ATOM (so the re-add is correct)
Two independent side-stream mechanisms in `atom/models/deepseek_v4.py`, BOTH gated
on `alt_stream is not None`:
1. **Dual-stream MoE** (shared_experts // routed_experts on `alt_stream`):
   `self._use_dual_stream = shared_experts is not None and alt_stream is not None
   and envs.ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD > 0` (deepseek_v4.py:2156). Per-call
   token-count gated → prefill (large batch) skips it; mainly a DECODE optimization.
2. **Async Compressor/indexer overlap** (Main Compressor → `alt_stream`, Indexer
   Compressor → `indexer_stream`): `use_async_compress = self._use_async_compress
   and fc.in_hipgraph` (deepseek_v4.py:1635); `_use_async_compress = alt_stream is
   not None and compressor is not None` (line 1567).
The two `torch.cuda.Stream()` objects are allocated once at model __init__
(deepseek_v4.py:~2668), shared across all blocks. There is NO env toggle in the new
code (compressor overlap is gated only by in_hipgraph).

### The re-added flag (single master switch)
- `atom/utils/envs.py`: registered `ATOM_DISABLE_SIDE_STREAMS` (default "0").
- `atom/models/deepseek_v4.py` (~line 2668): `_enable_side_streams =
  torch.cuda.is_available() and not envs.ATOM_DISABLE_SIDE_STREAMS`; allocate
  `alt_stream`/`indexer_stream` only when true, else None. Leaving them None makes
  every downstream `is not None` guard run inline → disables BOTH mechanisms in one
  switch. Added an info log of the resolved state.
- Usage: `ATOM_DISABLE_SIDE_STREAMS=0` (default) = multi-stream; `=1` = single-stream.
- **Runtime-verified**: all 8 DP ranks log `DSV4 side-streams DISABLED
  (single-stream) (ATOM_DISABLE_SIDE_STREAMS=1): alt_stream=False
  indexer_stream=False`.
- CAVEAT: edited in the installed site-package (`/opt/venv/.../atom/`), NOT a git
  repo — lost on container rebuild. To persist, commit into the ATOM source repo.

### ATOM single-stream (SS) vs multi-stream (MS) — tp8dp8, ATOM client, ratio1.0
Same config/grid as Exp 39 (multi-stream = the Exp 39 ATOM numbers).
| workload | conc | MS tok/s | SS tok/s | SS/MS | MS TPOT | SS TPOT | MS TTFT | SS TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k/1k | 64  | 4,211  | 3,945  | 93.7% | 29.18 | 31.23 | 1,307  | 1,261  |
| 1k/1k | 128 | 7,330  | 6,947  | 94.8% | 33.14 | 34.51 | 1,725  | 2,445  |
| 1k/1k | 256 | 12,418 | 12,078 | 97.3% | 37.12 | 38.90 | 3,564  | 4,197  |
| 1k/1k | 512 | 19,828 | 19,348 | 97.6% | 45.41 | 46.52 | 5,489  | 5,932  |
| 8k/1k | 64  | 14,530 | 13,518 | 93.0% | 33.35 | 36.41 | 5,929  | 4,843  |
| 8k/1k | 128 | 21,609 | 20,982 | 97.1% | 43.02 | 44.69 | 9,545  | 10,044 |
| 8k/1k | 256 | 30,880 | 30,146 | 97.6% | 55.38 | 57.48 | 18,854 | 19,965 |
| 8k/1k | 512 | 39,291 | 38,867 | 98.9% | 79.96 | 80.79 | 38,630 | 38,666 |

### Findings
- **Multi-stream wins everywhere, small margin, shrinking with concurrency**:
  biggest at c64 (SS = 93–94% of MS), nearly even at c512 (97.6–98.9%).
- MS TPOT consistently lower → side-stream overlap mainly helps DECODE (consistent
  with the dual-stream MoE per-call gating that skips large prefill batches).
- ⇒ side-streams are a real but modest optimization (~+1–7% total tok/s), largest
  at low concurrency (decode-heavy).

### Artifacts
- SS: `/workspace/bench_results_dsv4_atom_ss_0617/`; MS: `.../bench_results_dsv4_atom_0617/`.
- Logs: `/workspace/atom_ss_{server,sweep_0617}.log`. Server cleaned (VRAM ~0.3 GB/GPU).

---

## Exp 41 — SGLang c512 stability (3 repeats) (2026-06-17)

Q: is SGLang's large c512 deficit vs ATOM (Exp 39: 1k/1k 86.7%, 8k/1k 82.1%) a
stable measurement or run-to-run noise? Re-ran SGLang c512 for both workloads
**3× each**, same config as Exp 39 (gatherv ON, ROCM700A=0, tp8dp8, ATOM client,
ratio1.0, np4096/warm1024, 16384 prefill tok/rank).

| workload | run1 | run2 | run3 | mean | std | range |
|---|---:|---:|---:|---:|---:|---:|
| 1k/1k c512 | 17,195 | 17,157 | 17,348 | **17,233** | 83 (0.48%) | 191 (1.11%) |
| 8k/1k c512 | 32,155 | 32,186 | 32,200 | **32,180** | 19 (0.06%) | 45 (0.14%) |

TPOT/TTFT also tight: 8k/1k TPOT 72.59–72.65, TTFT 55.6–55.7s; 1k/1k TPOT
43.80–43.88, TTFT 7.36–7.70s.

### Conclusion — the deficit is REAL and reproducible, not noise
- 3-run variance is tiny (std 0.06–0.48%). SGLang stably lags ATOM at c512.
- vs Exp 39 multi-stream ATOM: 1k/1k 17,233/19,828 = **86.9%** (matches the single
  run's 86.7%); 8k/1k 32,180/39,291 = **81.9%** (matches 82.1%).
- Consistent with the standing conclusion: high-conc (c512) is SGLang's weak point —
  prefill↔decode interference is worst there. The 8k/1k c512 TTFT ≈ 55.6s (vs ATOM
  ~38.6s) points at heavy prefill queueing as the driver.
- NOTE: this c512 is the worst point; c64–c256 SGLang ties/wins (Exp 39).

### Artifacts
- `/workspace/bench_results_dsv4_sgl_c512_run{1,2,3}/` (2 JSON each).
- Logs: `/workspace/sgl_c512_repeat.log`, `/workspace/sgl_server_c512.log`. Server
  cleaned (VRAM ~0.3 GB/GPU).

---

## Exp 42 — c512 gap root-cause + levers tried (swa-ratio, mixed-chunk) (2026-06-17)

Investigated WHY SGLang lags ATOM so much at c512 (Exp 39/41: 1k/1k 86.9%, 8k/1k
81.9%). All on 1k/1k c512, gatherv ON, ROCM700A=0, tp8dp8, ATOM client, ratio1.0,
np4096/warm1024.

### (a) swa-full-tokens-ratio sweep — NO throughput effect
Overrode the aligned-script default 0.15 via SGL_EXTRA_ARGS.
| metric | 0.15 (mean3) | 0.2 | 0.25 |
|---|---:|---:|---:|
| total tok/s | 17,233 | 17,173 (−0.35%) | 17,128 (−0.61%) |
| Med TTFT ms | 7,500 | 6,966 | 7,224 |
| Med TPOT ms | 43.84 | 43.68 | 43.98 |
Throughput flat (within the 1.1% run-to-run band). TTFT best at 0.2 (~−7%) but not
monotonic; 0.25 regresses. ⇒ swa ratio tunes KV-pool split, NOT prefill/decode
scheduling → no tput lever. Kept default 0.15.

### (b) Gap decomposition (client metrics) — the gap is ALL prefill/TTFT
| | SGL | ATOM |
|---|---:|---:|
| total tok/s | 17,195 | 19,828 (+15.3%) |
| wall duration s | 487.9 | 423.1 |
| Med TPOT ms (decode) | **43.85** | 45.41 (SGL FASTER) |
| Med TTFT ms | 7,355 | 5,489 |
| mean TTFT ms | 13,404 | 5,924 |
| p99 TTFT ms | 53,459 | 9,538 (SGL 5.6×) |
| std TTFT ms | 16,046 | 2,532 (SGL 6.3×) |
- Decode is NOT the problem (SGL TPOT lower). Gap = prefill/TTFT, and the signature
  is VARIANCE: SGL TTFT std 6.3× and p99 5.6× ATOM; SGL mean≫median (right-skew
  tail), ATOM mean≈median (tight). Wall-duration ratio (1.153) == tput gap.

### (c) Scheduler-log root cause (BOTH engines, same case — two-sided)
Per-rank (~64 reqs/rank at c512):
| | SGLang | ATOM |
|---|---|---|
| decode batch occupancy | #running-req median **53/64**, range 2–64 (drains) | output median **64/64** (stable full) |
| prefill granularity | almost always FULL 16384 tok (16 reqs) | MIXED: full 16384 (most common) + many small 1024/2048/3072/4096 (1–4 reqs) |
| prefill-delayer | n/a | delay_rate **2.69%** (well-tuned, mostly allows) |
ROOT CAUSE: **SGLang cannot keep the decode batch full at c512** (median 53/64 ≈
83% occ, dips to single digits) while ATOM holds 64/64. The ~17% decode
under-occupancy ≈ the 15% tput gap. Mechanism: SGLang injects prefill as RIGID full
16384-token chunks and (default) non-mixed steps → each prefill chunk stalls ALL
decode for a step → decode occupancy collapses + TTFT bursts. ATOM uses ADAPTIVE
prefill granularity (small 1–4-req batches when needed) + a well-tuned prefill-delayer
to slip prefill in smoothly, keeping decode full and TTFT low/uniform. ATOM trades
slightly slower decode (TPOT 45.4 vs 43.9) for stable-full occupancy → +15% total.
Why only c512 breaks: more in-flight decode at high conc ⇒ each rigid prefill chunk
disrupts more; c64–c256 has little in-flight decode so SGLang ties/wins (Exp 39).
Logs: `/workspace/{sgl,atom}_server_sched.log` (+ `*_sched_sweep.log`); parsed DP0/
all-rank "Prefill batch"/"Scheduled prefill batch"/decode lines.

### (d) Lever tried: --enable-mixed-chunk — REJECTED (made it WORSE)
gsm8k OK with mixed-chunk (flexible 0.9386 / strict 0.9393, ~0.94 — not a
correctness issue). But:
| metric | SGL base (mean3) | SGL +mixed-chunk | MC vs base |
|---|---:|---:|---:|
| total tok/s | 17,233 | 16,321 | **−5.30%** |
| Med TTFT ms | 7,500 | 8,336 | +11.2% |
| mean TTFT ms | 13,260 | 15,921 | +20.1% |
| std TTFT ms | 15,870 | 18,055 | +13.8% |
| Med TPOT ms | 43.84 | 44.61 | +1.8% |
MC drops 86.9%→**82.3%** of ATOM. Mixing prefill tokens into the decode step
enlarges per-step batch (incl. large prefill chunk) → higher TPOT + WORSE TTFT
variance; with gatherv/MoE padding it nets negative. ⇒ baseline (non-mixed) is the
better SGLang c512 config. ATOM's edge is adaptive-granularity + full-occupancy
scheduling, NOT prefill+decode co-stepping.

### Not yet tried (candidate levers, more on-target than mixed-chunk)
- Raise `schedule_conservativeness` (DP attn auto ×0.3 → 0.3): admit prefill more
  conservatively to protect decode occupancy (directly targets the 53/64 drop).
- Smaller `chunked-prefill-size` (e.g. 8192/rank): shrink per-step prefill shock to
  mimic ATOM's small-batch injection (watch prefill efficiency).

### Artifacts
- swa: `/workspace/bench_results_dsv4_sgl_swa0{2,25}/`; mixed-chunk:
  `/workspace/bench_results_dsv4_sgl_mc/`; scheduler runs:
  `/workspace/bench_results_dsv4_{sgl,atom}_sched/`. gsm8k(MC): `/workspace/gsm8k_mc.log`.
  All servers cleaned (VRAM ~0.3 GB/GPU).

---

## Exp 43 — client validation: SGLang client vs ATOM client, same server (2026-06-17)

Q: does the c512 number change if driven by SGLang's OWN bench client instead of
ATOM's? Same SGLang server (gatherv ON, ROCM700A=0, tp8dp8, chunk16384/rank,
baseline), same `/v1/completions` endpoint, same params (np4096/warm1024/ratio1.0,
1k/1k c512). Only the client differs:
- ATOM client: `atom.benchmarks.benchmark_serving --backend vllm`.
- SGLang client: `python3 -m sglang.bench_serving --backend sglang-oai` (no shim
  needed vs an SGLang server).

| metric | ATOM client | SGLang client | diff |
|---|---:|---:|---:|
| total tok/s | 17,195 | **15,751** | **−8.4%** |
| wall duration s | 487.9 | 532.6 | +9.2% |
| Med TTFT ms | 7,355 | 8,261 | +12.3% |
| Mean TTFT ms | 13,404 | 15,612 | +16.5% |
| p99 TTFT ms | 53,459 | 58,155 | +8.8% |
| std TTFT ms | 16,046 | 18,370 | +14.5% |
| Med TPOT ms | 43.85 | 46.36 | +5.7% |

### Conclusion — the data is NOT the same; client effect is ~8% at c512
- SGLang's own client reports SYSTEMATICALLY LOWER throughput (and higher
  TTFT/TPOT/duration) than the ATOM client on the IDENTICAL server. The client
  itself accounts for ~8% at c512 — LARGER than the ~3% SKILL §4 saw at c128/c256,
  i.e. **client effect grows with concurrency** (client-side dispatch/burstiness
  becomes part of the bottleneck in high-conc closed loop).
- Implications:
  1. Validates the methodology choice: all SGLang-vs-ATOM engine numbers in this log
     use the ATOM client for BOTH engines, so the client effect cancels → the
     reported gaps (e.g. c512 ~87%) are pure engine differences.
  2. If someone instead used "SGLang client for SGLang, ATOM client for ATOM",
     SGLang would be under-reported ~8% → c512 would look ~79% instead of ~87%.
     ⇒ cross-engine comparison MUST use one client.
- Artifacts: `/workspace/bench_results_dsv4_sglclient/`, log
  `/workspace/sglclient_sweep.log`. Server cleaned (VRAM ~0.3 GB/GPU).

---

