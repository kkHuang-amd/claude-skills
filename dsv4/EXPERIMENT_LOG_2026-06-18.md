# DeepSeek-V4-Pro serving perf — experiment log (2026-06-18)

Split from the master `EXPERIMENT_LOG.md` (chronological, by date). See that file for the index and `SKILL.md` for how-to.

---

## Exp 44 — c512 levers: conservativeness (neutral) + chunk-size (helps) + CORRECTED mechanism (2026-06-18)

Tried the two Exp-42 candidate levers on 1k/1k c512 (gatherv ON, ROCM700A=0,
tp8dp8, ATOM client, ratio1.0, np4096/warm1024). Baseline = chunk 16384/rank,
conservativeness eff 0.3.

### Lever 1: schedule_conservativeness eff 0.3 → ~1.0 (`--schedule-conservativeness 3.3`, ×0.3 DP) — NEUTRAL
total 17,233 → 17,009 (−1.3%, in noise), TTFT not improved. Conservativeness
controls admit-caution to avoid RETRACTS; this case never retracts (KV fits), so it
does nothing to the real driver. Rejected.

### Lever 2: chunked-prefill 16384/rank → 8192/rank (`--chunked-prefill-size 65536`) — HELPS +5% (single run)
| metric | base 16384 | 8192 | ATOM |
|---|---:|---:|---:|
| total tok/s | 17,195 | **18,064 (+5.1%)** | 19,828 |
| duration s | 487.9 | 464.4 | 423.1 |
| Med TPOT | 43.85 | **45.62 (WORSE)** | 45.41 |
| Mean TPOT | 43.39 | 45.35 (worse) | 45.84 |
| Med ITL | 40.14 | 41.43 (worse) | 40.67 |
| Mean TTFT | 13,404 | **9,025 (−33%)** | 5,924 |
| std TTFT | 16,046 | 11,543 (−28%) | 2,532 |
| Med TTFT | 7,355 | 6,482 | 5,489 |
| p99 TTFT | 53,459 | 53,985 (~same) | 9,538 |
| Mean E2E | 57,797 | 55,423 (−4.1%) | 52,821 |
8192 closes ~1/3 of the gap: 86.9% → 91.1% of ATOM.

### CORRECTED mechanism (Exp 42's "smaller chunk → smoother decode occupancy" was WRONG)
The smaller-chunk win is a TRADE, and decode actually gets slightly WORSE:
- **Decode is hurt, not helped**: TPOT 43.85→45.62 (+4%), ITL up. More prefill steps
  DO interrupt decode more often (the intuitive objection is correct).
- **The win is on the PREFILL/queue side**: mean TTFT −33%, std −28%. Throughput is
  closed-loop ∝ 1/mean_E2E. Decompose mean E2E change: TTFT −4.4s, decode +1.8s
  (=ΔTPOT 1.77ms × 1024), net −2.6s ≈ measured mean-E2E −2.4s. So +5% tput = (TTFT
  queue win) − (TPOT decode cost); at 1k/c512 the queue win dominates.
- **Why smaller chunk shortens TTFT**: TTFT≈queue-wait (1k prefill compute is tiny).
  What matters is how OFTEN a prefill step fires, not how many reqs it carries.
  Big chunk → scheduler/prefill-delayer batches into infrequent big prefill waves →
  a new req can wait a whole decode interval (high mean/var TTFT). Small chunk →
  cheaper, more frequent prefill steps → reqs admitted in smaller steadier waves →
  lower mean/var TTFT. Cost = more prefill steps eat decode time → TPOT up.
- NOT clean: p99 TTFT unchanged (worst tail same); TPOT is a real tradeoff
  (interactivity-sensitive workloads lose). Single run — repeat ×3 to confirm.

### Reconciles with the old "2048/rank worse than 16384" finding — non-monotonic, two opposing effects
- Effect A (prefill efficiency): smaller chunk → more steps → per-step overhead +
  low-M GEMM MFU → HURTS. Dominates at 8k input (chunk also splits a single 8192-tok
  prefill → efficiency loss amplified) → "bigger is better down to the 16384/rank
  floor" (Exp 16/18).
- Effect B (queue fairness): smaller chunk → more frequent prefill → lower TTFT
  mean/variance → HELPS. Dominates at 1k input (1024 < chunk so a request is never
  intra-chunked; chunk only sets reqs-per-prefill-step granularity).
- ⇒ optimal chunk is workload×concurrency dependent: 8k/c256 wants big (A); 1k/c512
  wants smaller (B). Both prior+current findings are correct in their regime.

### CORRECTION to Exp 42 root-cause framing
Exp 42 attributed the c512 vs-ATOM gap to "SGLang decode occupancy 53/64". But the
client metrics show **SGLang TPOT is BETTER than ATOM at c512 (43.85 < 45.41)** — if
decode were truly under-occupied/inefficient, TPOT would be WORSE. So the vs-ATOM
c512 gap is PURELY TTFT (prefill admission queueing/fairness; SGLang TTFT mean 13.4s
& std 16k vs ATOM 5.9s & std 2.5k), NOT decode occupancy. The "decode 53/64" log
reading was likely a sampling artifact / not the throughput driver.

### Untried / next
- Repeat 8192 ×3 (stability); re-capture scheduler logs at 8192 vs 16384 to confirm
  the "more frequent prefill steps + shorter queue" causal chain (prefill-step
  frequency + queue depth). Try 4096/rank (push effect B; watch effect A). Validate
  best chunk on 8k/c512 (gap is larger there, 82%) and on c128/c256 (don't regress).

### Artifacts
- conservativeness: `/workspace/bench_results_dsv4_sgl_cons/`; chunk8192:
  `/workspace/bench_results_dsv4_sgl_cps/`; logs `/workspace/sgl_{cons,cps}_sweep.log`,
  `/workspace/sgl_server_{cons,cps}.log`. Servers cleaned (VRAM ~0.3 GB/GPU).

---

## Exp 45 — chunk-size sweep: 8192 stability + 4096 + 8k validation (2026-06-18)

Verified the Exp 44 chunk lever properly: 8192 stability ×3, 4096/rank (push effect
B), and the 8k workload (where chunk < 8192 SPLITS a request → effect A). 1k & 8k,
c512, gatherv ON, ROCM700A=0, ATOM client, ratio1.0, np4096/warm1024.

### 1k/1k c512 (1024<chunk for all → no intra-req split; chunk just sets reqs/step)
| chunk/rank | reqs/step | total tok/s | std | vs ATOM | Med TPOT | Mean TTFT |
|---|---:|---:|---:|---:|---:|---:|
| 16384 (base ×3) | 16 | 17,233 | 83 | 86.9% | 43.84 | 13,260 |
| **8192 (×3)** | 8 | **18,106** | **11** | **91.3%** | 45.46 | 8,874 |
| 4096 (×1) | 4 | 18,099 | — | 91.3% | 46.88 | 7,123 |
| ATOM | — | 19,828 | — | 100% | 45.41 | 5,924 |
- 8192 STABLE (×3 = 18,095/18,121/18,104, std 11 = 0.06%) → the +5% is REAL.
- **8192→4096 plateaus** (18,106≈18,099): TTFT keeps dropping (8,874→7,123) but TPOT
  keeps rising (45.46→46.88) — they cancel. Sweet spot = 8192/rank.

### 8k/1k c512 (chunk<8192 SPLITS the 8192-tok request → triggers effect A)
| chunk/rank | behavior | total tok/s | vs ATOM | Med TPOT | Mean TTFT |
|---|---|---:|---:|---:|---:|
| 16384 (base) | 2 reqs/step | 32,254 | 82.1% | 72.70 | 65,771 |
| **8192** | 1 req/step, no split | **33,475** | **85.2%** | 82.61 | 53,521 |
| 4096 | SPLITS req into 2 | 33,268 | 84.7% | 84.89 | 48,299 |
| ATOM | — | 39,291 | 100% | 79.96 | 38,745 |
- 16384→8192 helps (+3.8%, 1 req/step, no split). **8192→4096 REGRESSES** (85.2→84.7%)
  — crossing into intra-request splitting makes effect A (prefill efficiency) bite,
  exactly as the two-effects model predicts.

### Confirmations
1. The chunk win is REAL & reproducible (8192 ×3 std 0.06%).
2. TPOT rises MONOTONICALLY as chunk shrinks (1k 43.8→45.5→46.9; 8k 72.7→82.6→84.9)
   → smaller chunk DOES interrupt decode more (the intuitive objection is correct);
   throughput is the net of (TTFT gain − TPOT cost), peaking at 8192.
3. The "don't split a request" boundary is real: at 8k, 8192 (=1 req, no split) is
   best; 4096 (splits) regresses. At 1k, 4096 is fine (still >1024, no split) but no
   extra gain. ⇒ **8192/rank is the universal c512 sweet spot** (1k 86.9→91.3%, 8k
   82.1→85.2%).
4. chunk tuning recovers ~1/3 of the gap; ATOM still leads (TTFT 5.9s/38.7s vs
   8.9s/53.5s) → ATOM's scheduling (prefill fairness) is still better; the residual
   is NOT a chunk-size issue.

### Artifacts
- 8192: `/workspace/bench_results_dsv4_A_8192_1k_run{1,2,3}/`, `.../_A_8192_8k/`.
- 4096: `/workspace/bench_results_dsv4_B_4096_1k/`, `.../_B_4096_8k/`.
- Logs: `/workspace/server{A,B}_bench.log`, `/workspace/sgl_server{A,B}.log`.
  Servers cleaned (VRAM ~0.3 GB/GPU).

---

## Exp 46 — old vs new ATOM at c512: is ATOM's speed a recent change? (2026-06-18)

Q: is ATOM fast vs SGLang because of a recent ATOM scheduler update or faster
prefill kernels? Compared two ATOM commits at c512 (tp8dp8, multi-stream, ATOM
client, ratio1.0, np4096/warm1024).
- **OLD = `914d50323` (2026-06-08)** = `/sgl-workspace/ATOM-previous`.
- **NEW = `bcd38f67` (2026-06-17)** = `/sgl-workspace/ATOM` = the version ALL prior
  ATOM data in this log (Exp 39–45) was measured on.
- (Version note: the installed-pkg metadata `0.1.4.dev80+g<hash>` identifies which
  commit is live; `pip install ./<dir>/` swaps it. OLD has no `ATOM_DISABLE_SIDE_STREAMS`
  flag — that edit was on NEW's site-package, overwritten by installing OLD.)

| | OLD 914d50323 | NEW bcd38f67 | NEW vs OLD |
|---|---:|---:|---:|
| 1k/1k c512 total tok/s | 19,995 | 19,828 | −0.8% |
| 1k Med TTFT / TPOT | 5,766 / 45.33 | 5,489 / 45.41 | −4.8% / +0.2% |
| 8k/1k c512 total tok/s | 39,721 | 39,291 | −1.1% |
| 8k Med TTFT / TPOT | 38,573 / 78.40 | 38,630 / 79.96 | +0.1% / +2.0% |

### Conclusion — the two ATOM versions are IDENTICAL in perf (all metrics ±2%, noise)
- ATOM did NOT change (perf-wise) between 06-08 and 06-17: neither scheduler nor
  prefill-kernel speedup in that window. ATOM was ALREADY this fast at 914d50323.
- ⇒ ATOM's advantage over SGLang is NOT a recent patch; it's inherent to its design
  (adaptive prefill injection + prefill-delayer fairness, per Exp 42/44/45).
  Diffing 914d50323↔bcd38f67 will NOT locate "why ATOM is fast" (no perf delta).
- To find WHEN ATOM became fast, would need a much older ATOM (pre-improvement);
  both of these are already post-improvement.

### Install state after this exp
Installed = OLD 914d50323 (no side-stream flag). Restore to NEW with
`pip install /sgl-workspace/ATOM/` and re-apply the flag if persistent single-stream
A/B is needed (perf is equivalent either way).

### Artifacts
- OLD: `/workspace/bench_results_dsv4_atomOLD/` (2 JSON), log
  `/workspace/atomOLD_sweep.log`, `/workspace/atom_old_server.log`. NEW = the
  existing `/workspace/bench_results_dsv4_atom_0617/`. Server cleaned (VRAM ~0.3 GB/GPU).

---

## Exp 47 — pure-prefill (OSL=1) compute: is SGLang's per-step prefill slower? (2026-06-18)

Q: besides the scheduler, is SGLang's per-STEP prefill execution itself slower? To
isolate prefill COMPUTE from the prefill↔decode scheduler interference, ran a
PURE-PREFILL workload (OSL=1, almost no decode) on both engines, same client, same
chunk 16384/rank, conc512, np4096/warm1024, rate inf. Prefill (input) throughput at
saturation = per-step prefill compute rate (per-step time = 131072 tok ÷ system tps).

| ISL | SGL input tok/s | ATOM input tok/s | ATOM/SGL | per-step SGL | per-step ATOM |
|---|---:|---:|---:|---:|---:|
| 1024 | 48,291 | 57,821 | **+19.7%** | 2,714 ms | 2,267 ms |
| 8192 | 47,013 | 55,999 | **+19.1%** | 2,788 ms | 2,341 ms |

### Conclusion — NOT purely scheduler: SGLang prefill compute is ~20% SLOWER
- With decode interference removed (OSL=1), ATOM still prefills ~19–20% faster →
  **SGLang's per-step prefill execution is genuinely ~20% slower** (real
  compute/kernel gap, NOT scheduling).
- So the c512 gap has TWO components: (1) **prefill compute ~20% slower** (this exp)
  + (2) scheduler/queueing (Exp 42/44/45, partly recoverable via chunk tuning).
  Reconciles the mixed-c512 picture: SGLang decode (TPOT) is fine/better, but TTFT
  is high because prefill is BOTH slower to compute AND queued.
- Direction matches Exp 36 (prefill compute gap lives in the engine-specific MLA
  path; MoE GEMM + comm are shared/equal). Magnitude here (~20% at c512) > the ~8%
  measured per-token at c256 — bigger under c512 saturation / system-throughput view.
- Caveat: input_tps under saturation also reflects prefill BATCHING efficiency, not
  only raw kernel; but both are engine-side (not the prefill↔decode interference).
  Next to pin raw kernel: isolated MLA-prefill kernel microbench at matched shapes.

### Artifacts
- `/workspace/bench_pp_sgl/`, `/workspace/bench_pp_atom/` (isl{1024,8192}_osl1_c512),
  logs `/workspace/pp_{sgl,atom}_sweep.log`, `/workspace/{sgl,atom}_pp_server.log`.
  Servers cleaned (VRAM ~0.3 GB/GPU).

---

## Exp 48 — prefill TRACE: split the ~20% into raw-kernel vs overhead (2026-06-18)

Followed up Exp 47 ("SGLang prefill ~20% slower") with torch-profiler traces to split
it into raw-kernel-compute vs host/launch overhead. BOTH single-stream (SGLang aligned
is already single-stream; ATOM run with ATOM_DISABLE_SIDE_STREAMS=1 so kernels
serialize → clean GPU-busy), pure-prefill load (ISL=8192 OSL=1, conc512), torch
profiler GPU activity, rank0 trace.

### Method note (heeds Exp 19): per-kernel `dur` is UNRELIABLE (esp. ATOM — durs
inflated / overlapping streams gave impossible >20,000 ms/step for some kernels). The
ONLY trusted metric = GPU-active UNION (wall time with ≥1 kernel running). Normalized
per `pa_prefill` count (= attn-layers × steps; SAME aiter kernel + same model both
engines → L-independent unit). SGLang pa=549 (~10 steps), ATOM pa=183 (~3.3 steps).

### Result — GPU-active per attn-layer-step (overlap-robust, the reliable number)
| | GPU-active/step | wall/step (Exp47) | per-step bubble |
|---|---:|---:|---:|
| SGLang | 2.46 s | 2.79 s | **12%** |
| ATOM | 2.28 s | 2.34 s | **3%** |
- GPU-active per unit work: SGLang **+8%** (44.81 vs 41.54 ms/attn-layer-step).
- Per-step wall ratio 2.79/2.34 = 1.19 (= the Exp 47 prefill tput gap) DECOMPOSES as:
  **raw GPU kernel ×1.08 (8%) × host-overhead/bubble ×1.10 (10%) ≈ 1.19 (19%).**

### Conclusion — answer to "are the raw kernels fine?"
NOT identical, but the raw-kernel gap is modest: **~8% is real GPU compute (SGLang
kernels take 8% more GPU-active time), and ~10% is host/launch overhead+bubbles**
(SGLang has 12% per-step idle between kernels vs ATOM's 3%). So the ~20% prefill gap
is roughly HALF kernel, HALF glue/overhead — it's NOT purely scheduler, NOT purely
kernel.
- Per-kernel attribution of the 8% is BLOCKED by ATOM's unreliable per-kernel dur
  (Exp 19). The shared aiter `pa_prefill` (MLA core attn) is the same kernel on both
  (~179 ms/step on SGLang) and should be equal; the 8% likely sits in the MLA
  projection GEMMs / glue (per Exp 36 direction), but needs an ISOLATED kernel
  microbench at matched shapes to pin — trace dur can't do it.
- SGLang's 12% per-step bubble (launch gaps) is a concrete, addressable target
  (kernel launch batching / fewer host syncs / CUDA graph for prefill).

### Artifacts
- SGLang trace `/workspace/sgl_prof/1781761067*TP-0-DP-0.trace.json.gz` (clean 10-step,
  89% busy); ATOM `/workspace/atom_prof/dp0_tp0/*.pt.trace.json.gz`. Analyzer:
  `useful-scripts/benchmarking/dsv4/analyze_trace.py` (+ global-union script inline).
  Both servers cleaned via `rocm-smi --showpids` → kill (VRAM ~0.3 GB/GPU).

---

## Exp 50 — shared-expert-local PoC: +6–7% prefill (2026-06-18)

Per-layer trace (Exp 48 follow-up) showed the MoE "shared-expert + gate" block ~2×
bigger on SGLang. Investigated whether it's a kernel diff or a logic diff.

### Shapes & redundancy check (microbench, avoids unreliable trace dur)
- shared-expert (n,k) are config constants → SAME on both engines. The differing
  factor is M (tokens).
- SGLang (code-confirmed): `disable_shared_experts_fusion` → separate
  `self.shared_experts`; `_shared_expert_use_tp1=False` ⇒ shared expert is
  **TP-sharded**. deepseek_v4.py decoder gathers local→global THEN `self.mlp(global)`
  → shared expert runs on the GLOBAL buffer (M≈131072).
- ATOM: shared expert on LOCAL tokens (M≈16384), before the gather.
- **CORRECTION to the initial "8× redundant" guess:** a TP-sharded shared expert is
  NOT redundant — per-rank FLOPs are identical (TP8-global: M=131072×FFN/8 = 16384×FFN
  ≡ TP1-local: M=16384×FFN). Only the GATE is truly redundant (replicated; Exp 38).
- Same-FLOPs shape microbench (`gemm_a8w8_blockscale_bpreshuffle_ck`):
  | GEMM (same FLOPs) | TP8-global | TP1-local | TP1 faster |
  |---|---:|---:|---:|
  | gate_up (K7168) | M131072,N768: 1.38ms | M16384,N6144: 1.21ms | 13% |
  | down | M131072,N7168,K384: 1.10ms | M16384,N7168,K3072: 0.63ms | 1.74× |
  (the global down GEMM's K=384 is too small → low arithmetic intensity.)
  ALSO ck_xdl (SGLang) is FASTER than ck_tile (ATOM) at these shapes (~1.6×) → ATOM's
  shorter trace time is purely M/shape, NOT a faster kernel; do NOT swap to ck_tile.

### PoC implementation (env-gated SGLANG_DP_SHARED_EXPERT_LOCAL=1, needs SHARED_EXPERT_TP1=1)
Compute the (replicated, TP1) shared expert on LOCAL hidden in the decoder layer
BEFORE the dp gather; skip it inside `self.mlp` (forward_normal); add it back to this
rank's reduce-scattered LOCAL slice. Prefill-only (gated on is_extend). Files:
`models/deepseek_v2.py` (skip_shared_experts param in forward/forward_normal),
`models/deepseek_v4.py` (compute local + skip + add after reduce_scatterv).

### Result — pure-prefill (OSL=1), gsm8k correct (flex 0.9386 / strict 0.9393)
| ISL | SGL base | SGL +SE-local | ATOM | gain | base→new vs ATOM |
|---|---:|---:|---:|---:|---:|
| 1024 | 48,291 | 51,171 | 57,821 | **+6.0%** | 84% → 88% |
| 8192 | 47,013 | 50,324 | 55,999 | **+7.0%** | 84% → 90% |
- **+6–7% prefill — BIGGER than the ~1.4% GEMM-shape microbench predicted.** The extra
  gain is because the local path also runs 8× FEWER ROWS through the activation
  fp8-quant + elementwise (global processed M=131072 rows; local M=16384), on top of
  the better GEMM shape. So it's NOT redundant FLOPs but it IS redundant
  per-row quant/overhead on the global buffer. The user's "test it empirically" call
  was right; the microbench under-counted.
- Caveats: (1) requires TP1 shared expert → ~8× shared-expert weight memory/rank
  (~+0.5 GB). (2) pure-prefill only here; **c512 end-to-end (decode-bound) gain
  unverified** — Exp 38 gate-local was neutral at c512, so confirm with a full A/B.
  (3) prefill-path only (decode keeps global shared).

### Next
- FULL c512 1k/1k & 8k/1k end-to-end A/B with SE-local (does the prefill win move
  total tput, or is it decode-bound/neutral like gate-local?).
- Stack with Exp 49 (CK GEMM + batched rope) — are the gains additive?

### Artifacts
- `/workspace/bench_pp_se/` (isl{1024,8192}_osl1), gsm8k `/workspace/gsm8k_se.log`,
  server `/workspace/sgl_se_server.log`. Microbench `/workspace/gemm_micro.py`.
  Edits in `/sgl-workspace/sglang` (deepseek_v2.py, deepseek_v4.py), env-gated default
  OFF. Server cleaned (VRAM ~0.3 GB/GPU).

---

## Exp 51 — ALL 3 prefill levers stacked: 84% → 97–98% of ATOM (2026-06-18)

Stacked the three prefill fixes (all env-gated, default OFF):
`SGLANG_FORCE_CK_W8A8=1 SGLANG_ROPE_BATCHED=1 SGLANG_DP_SHARED_EXPERT_LOCAL=1
SGLANG_SHARED_EXPERT_TP1=1` (+ gatherv ON, ROCM700A=0). gsm8k correct: flex 0.9469 /
strict 0.9477.

### Pure-prefill (OSL=1, ATOM client) — gains are ~ADDITIVE
| ISL | base | +CK+ROPE | +SE-local | ALL3 | ATOM | ALL3 vs base | ALL3 vs ATOM |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 48,291 | 52,506 | 51,171 | **55,944** | 57,821 | +15.8% | **97%** |
| 8192 | 47,013 | 51,205 | 50,324 | **54,645** | 55,999 | +16.2% | **98%** |
⇒ the three independent levers stack to ~+16% and **close the prefill gap to 97–98%**
of ATOM (was 84%).

### Per-layer trace (ISL8192, pa_prefill-windowed) — gap nearly gone
GPU-active/attn-layer-step: base 44.81 → **ALL3 38.53** → ATOM 41.54 ms (SGLang now
BELOW ATOM in raw GPU compute). Per-layer WINDOW (ratio-4): base ~7000us SGLang-slower
→ **ALL3 39.5ms vs ATOM 38.8ms (~600us)**. Op-by-op now aligned: o-proj GEMM both
ck_tile (1904 vs 1802 ≈), shared-expert both local-ish (744+silu+444 vs 752+act+442 ≈),
MLA/MoE/comm/kv-q-proj all shared+equal. Remaining small diffs:
- **RoPE**: SGL `rope_batched` 332us vs ATOM fused `inverse_rope_gptj` 155us (~180us).
- **compressor glue**: SGL `fused_norm_rope`+`fill`×3+`rocprim`×2+elementwise vs ATOM's
  3 tight fused kernels (`hca_compress_forward`+`hca_norm_rope_scatter`+`compressor_update`).
  This is the residual per-step bubble (Exp 48); now the dominant remaining item.

### Reusable per-layer diff METHOD (persisted)
`useful-scripts/benchmarking/dsv4/layer_diff.py` (PERSISTENT, not /workspace):
- `overview <trace>`: list pa_prefill windows (=layers) with window dur + GPU-active union.
- `seq <trace> <ratio:4|128>`: one layer's ordered kernel sequence (grouped).
- `cmp <sglA> <atomB> <ratio>`: side-by-side (the main debug view).
Boundary = `pa_prefill` kernel (alternates dur by compress_ratio 128/4 per config.json,
so match SAME ratio across engines). Trust WINDOW span + GPU-active UNION; per-kernel
`dur` is unreliable (Exp 19). Capture both single-stream + pure-prefill (OSL=1).

### Per-layer op table — SGLang ALL3 vs ATOM (ratio-4 layer, pa→pa window, us)
| stage | op | SGL ALL3 | ATOM | status |
|---|---|---:|---:|---|
| MLA attn | pa_prefill | 4610 | 4680 | = shared |
| out RoPE | rope | batched 332 | inverse_gptj 155 | ⚠ SGL ~2× (~180us) |
| o-proj | cijk + quant | 1437 + 143 | 1526 + 140 | = |
| o-proj | main GEMM | **ck_tile 1904** | **ck_tile 1802** | ✅ aligned (was Triton) |
| mhc | post/pre | 368/483 | 386/522 | = |
| shared-exp | up_gate ck_tile | 744 | 752 | ✅ aligned (local) |
| shared-exp | silu/act | 65 | 48 | = |
| shared-exp | down ck_xdl | 444 | 442 | ✅ aligned (local) |
| gate | router cijk | 572 | 99 | ⚠ SGL gate still GLOBAL (redundant, Exp 38) |
| comm | gather nccl | 4498 | 4694 | = |
| routed MoE | sort+moe1+moe2+reduce | ~12030 | ~12030 | = shared |
| comm | reduce-scatter nccl | 4615 | 4861 | = |
| next pre-attn | kv_a/q_a ck_xdl | 2739 | 2730 | = shared |
| next pre-attn | qk_norm_rope_fused | 729 | 967 | SGL slightly better |
| compressor | glue | fused_norm_rope+fill×3+**rocprim×2**+elementwise (~80us, many launches) | **hca_compress_forward+hca_norm_rope_scatter+compressor_update** (3 fused, ~28us) | ⚠ bubble — top remaining item |
| **TOTAL** | window | **39,472** | **38,846** | gap **~600us** (base was ~7000) |
| | GPU-active/layer | **38.53 ms** | 41.54 ms | SGL now below ATOM |

### Status / next
Prefill essentially matched (97–98%). Remaining prefill items are small: (1)
compressor glue/bubble (SGL fill/rocprim/fused_norm_rope vs ATOM 3 hca_* fused) —
the top remaining; (2) rope (batched 332 vs ATOM fused-inverse 155); (3) gate still
global (Exp 38 gate-local was c512-neutral). Caveat: SE-local needs TP1 shared
(~+0.5GB/rank). Artifacts: `/workspace/bench_pp_all/`, `/workspace/gsm8k_all.log`,
trace `/workspace/sgl_prof_all/*TP-0-DP-0*`. Cleaned (VRAM ~0.3 GB).

---

## Exp 52 — ALL3 c512 END-TO-END A/B (OSL=1024, with decode) (2026-06-18)

Does the matched prefill (Exp 51) move c512 TOTAL throughput, or is it diluted by the
decode-bound regime (gate-local Exp 38 was neutral)? Ran ALL3 (FORCE_CK_W8A8 +
ROPE_BATCHED + SHARED_EXPERT_LOCAL + SHARED_EXPERT_TP1, gatherv ON, ROCM700A=0) at
c512, full OSL=1024, ATOM client, np4096/warm1024.

| workload | SGL base | SGL ALL3 | ATOM | ALL3 vs base | base→ALL3 vs ATOM |
|---|---:|---:|---:|---:|---:|
| 1k/1k | 17,233 | 17,877 | 19,828 | **+3.7%** | 86.9% → 90% |
| 8k/1k | 32,254 | 34,613 | 39,291 | **+7.3%** | 82.1% → 88% |
ALL3 detail: 1k MedTTFT 6855 TPOT 45.36 E2E 53084; 8k MedTTFT 47984 TPOT 69.56 E2E 103016.

### Conclusion — the prefill win DOES move c512 total tput (unlike gate-local)
- c512 end-to-end gain is REAL (+3.7% 1k, +7.3% 8k), bigger at 8k (more prefill-heavy).
  Closes c512 gap 86.9%→90% (1k), 82.1%→88% (8k) of ATOM.
- Diluted vs pure-prefill (+16%) because c512 is decode-bound — the matched prefill
  only helps the prefill share of the step. But it is NOT neutral (gate-local was),
  because these levers cut a much larger prefill chunk (GEMM + shared-expert rows +
  rope) than the gate alone.
- Remaining c512 gap (~10–12%) is now decode/scheduling + the residual prefill bubble
  (compressor glue) — not the kernels we fixed.

### Net summary of the 2026-06-18 prefill work (Exp 49–52)
3 env-gated levers (default OFF), gsm8k correct (0.9477):
- pure-prefill: 84% → 97–98% of ATOM (+16%); per-layer gap ~7000us → ~600us.
- c512 end-to-end: +3.7% (1k) / +7.3% (8k); 87%/82% → 90%/88% of ATOM.
Levers: `SGLANG_FORCE_CK_W8A8` (MLA proj Triton→CK), `SGLANG_ROPE_BATCHED` (batched
compressor rope), `SGLANG_DP_SHARED_EXPERT_LOCAL` (+`SGLANG_SHARED_EXPERT_TP1`,
shared expert on local hidden). Edits: fp8_utils.py, deepseek_v4_rope.py,
deepseek_v2.py, deepseek_v4.py (all in /sgl-workspace/sglang, default OFF).

### Artifacts
- `/workspace/bench_c512_all3/`, log `/workspace/c512_all3_sweep.log`,
  server `/workspace/sgl_b_server.log`. Baselines: Exp 39/41 + `/workspace/bench_results_dsv4_atom_0617/`.
  Server cleaned (VRAM ~0.3 GB/GPU).

### Follow-up — add chunk 8192/rank on top of ALL3 (Exp 45 lever stacks)
Stacked `--chunked-prefill-size 65536` (=8192/rank, the Exp 45 c512 sweet spot) ON TOP
of ALL3. c512 end-to-end:
| wl | base | ALL3 (16k/r) | **ALL3 + chunk 8k/r** | ATOM | best/ATOM |
|---|---:|---:|---:|---:|---:|
| 1k/1k | 17,233 | 17,877 | **17,921** | 19,828 | **90%** |
| 8k/1k | 32,254 | 34,613 | **36,031** | 39,291 | **92%** |
- 8k/1k: chunk8k adds **+4.1%** on top of ALL3 → vs base **+11.7%**, **82%→92% of ATOM**.
- 1k/1k: +0.2% (saturated; chunk effect already small at 1k c512), 90% of ATOM.
⇒ **best c512 config = ALL3 + chunk 8192/rank**: 1k 90%, 8k 92% of ATOM (8k from 82%).
The chunk lever (effect B, TTFT/queue fairness) is independent of and stacks with the
kernel/locality levers, especially at 8k. Artifacts:
`/workspace/bench_c512_all3_chunk8k/`, `/workspace/c512_all3_chunk8k_sweep.log`.

---

## Exp 53 — A2: output inverse-RoPE full-fuse (contiguous kernel) (2026-06-18)

The remaining rope item (Exp 51): the hot 337us/layer rope is the ATTENTION-OUTPUT
inverse rope `fused_rope_inplace(o[..., -rd:], k=None, ..., inverse=...)` at
deepseek_v4.py:1012. On HIP it fell back to `apply_rotary_emb_triton` (my Exp49
batched, STRIDED 2i/2i+1 interleaved loads = 337us); on CUDA it uses a single fused
kernel. ATOM uses `inverse_rope_gptj` (CONTIGUOUS load + reshape/flip) = 155us.

FIX: new `apply_rotary_emb_contig_kernel` (deepseek_v4_rope.py), mirrors ATOM —
loads the rope slice as a CONTIGUOUS [BLOCK_M, RD] tile (coalesced) and does the
GPT-J pair rotation via tl.reshape + tl.flip; derives cos/sin from the interleaved
freqs_real (cos=fr[2*(d//2)], sin=fr[2*(d//2)+1]). Supports forward+inverse. Wired
in apply_rotary_emb_triton for the 3D case under SGLANG_ROPE_BATCHED (the 2D
compressor rope keeps the prior batched kernel).

Result (trace, ALL3 + contig rope): `apply_rotary_emb_contig_kernel` = **142.7 us/call**
(was strided batched 337; **≤ ATOM's inverse_rope_gptj 155us**). gsm8k correct: flex
0.9439 / strict 0.9447. Per-layer saving ~194us × 61 ≈ 12ms/step (~0.5% prefill) — the
rope item is now fully closed (SGL ≤ ATOM). Remaining per-layer residual is the
compressor glue/bubble (A1) and gate-global (Exp 38).

Artifacts: trace `/workspace/sgl_prof_a2/*TP-0-DP-0*`, gsm8k `/workspace/gsm8k_a2.log`.
Edit in `/sgl-workspace/sglang/python/sglang/srt/layers/deepseek_v4_rope.py`
(under SGLANG_ROPE_BATCHED). Server cleaned (VRAM ~0.3 GB/GPU).

---

## Exp 54 — make CK-GEMM + batched/contig-RoPE DEFAULT-ON for DSV4 (no env) (2026-06-18)

Converted two levers from env-flag-gated to module toggles (default OFF) that the
DeepseekV4 model flips ON in `__init__` — so DSV4 gets them WITHOUT any env var:
- `fp8_utils.py`: `_FORCE_CK_W8A8=False` + `set_force_ck_w8a8()`; `use_aiter_triton_gemm_w8a8_tuned_gfx950`
  checks `_FORCE_CK_W8A8 or env`. (CK GEMM for MLA proj.)
- `deepseek_v4_rope.py`: `_USE_BATCHED_ROPE=False` + `set_batched_rope()`;
  `apply_rotary_emb_triton` checks `_USE_BATCHED_ROPE or env`. (batched/contig rope.)
- `deepseek_v4.py` `DeepseekV4ForCausalLM.__init__`: imports + `set_force_ck_w8a8(True)`,
  `set_batched_rope(True)`. The env vars `SGLANG_FORCE_CK_W8A8` / `SGLANG_ROPE_BATCHED`
  still work as overrides.
(NOT changed: shared-expert-local stays env-gated — needs TP1 shared, ~+0.5GB/rank;
chunk-prefill is a launch arg.)

Verification — launched DSV4 with NO opt env flags, trace confirms defaults active:
contig rope present=True, strided batched rope=False, Triton a8w8 GEMM=False, ck_tile
QuantGemm=True. gsm8k correct: flex 0.9507 / strict 0.9515. ⇒ DSV4 now uses CK GEMM +
contig rope by default (no env needed). Artifacts: `/workspace/sgl_prof_def/*`,
`/workspace/gsm8k_def.log`. Server cleaned (VRAM ~0.3 GB/GPU).

---

## Exp 49 — FIX the prefill kernel gap: Triton→CK GEMM + batched RoPE (2026-06-18)

From the trace (Exp 48) two prefill kernels differ between engines by IMPLEMENTATION
(same logical op). Identified the call paths and made SGLang match ATOM:

### Kernel 1 — w8a8-block FP8 GEMM (the big one: MLA q/kv/o projections, ~28% of step)
- SGLang `apply_w8a8_block_fp8_linear` (fp8_utils.py) picks the **Triton**
  `gemm_a8w8_blockscale` for the MLA projection shapes because the hardcoded
  `use_aiter_triton_gemm_w8a8_tuned_gfx950(n,k)` list contains them (k=7168 shapes:
  2112/512/4096/4608×7168 etc.). HIP is 7.2.0 so the CK bpreshuffle path is available.
- ATOM `linear.py` per_1x128 path uses the **CK** `gemm_a8w8_blockscale_bpreshuffle`
  (preshuffle) by default (`ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=1`, Triton off) — with
  an explicit comment "Triton FP8 Blockscale GEMM is mostly slower than AITER [CK] GEMM".
- FIX: env-gate `SGLANG_FORCE_CK_W8A8=1` → `use_aiter_triton_gemm_w8a8_tuned_gfx950`
  returns False → SGLang uses CK bpreshuffle (= the trace's `ck_tile…QuantGemmKernel`),
  matching ATOM. (fp8_utils.py:72.)

### Kernel 2 — RoPE (compressor fallback rope; small, ~2% of step)
- SGLang `apply_rotary_emb_triton` (deepseek_v4_rope.py, called in compress_hip.py):
  grid (batch, heads, dim_blocks) = ONE program per token (fine-grained launches).
- ATOM `_inverse_rope_gptj_kernel`: batches BLOCK_S=32 tokens/program.
- FIX: env-gate `SGLANG_ROPE_BATCHED=1` → added `apply_rotary_emb_triton_kernel_batched`
  (BLOCK_M=32 tokens/program, same math), mirroring ATOM. (deepseek_v4_rope.py.)

### Result — pure-prefill (OSL=1, ATOM client) with BOTH flags on
gsm8k stays correct: flexible 0.9462 / strict **0.9469** (≈ baseline 0.94 — both
changes numerically safe).
| ISL | SGL base | SGL +CK+ROPE | ATOM | gain | base→new vs ATOM |
|---|---:|---:|---:|---:|---:|
| 1024 | 48,291 | 52,506 | 57,821 | **+8.7%** | 84% → **91%** |
| 8192 | 47,013 | 51,205 | 55,999 | **+8.9%** | 84% → **91%** |
- **+~8.8% prefill throughput → recovers essentially the entire ~8% raw-kernel gap**
  from Exp 48 (84%→91% of ATOM). The remaining ~9% to ATOM = the host/overhead/bubble
  share (Exp 48), which kernel swaps don't touch.
- Dominant contributor is almost certainly the GEMM (28% of step vs RoPE's ~2%); not
  yet isolated GEMM-only vs RoPE-only (tested combined per request). Can split if needed.

### Status / how to use
- Both changes are env-gated and DEFAULT OFF (`SGLANG_FORCE_CK_W8A8`,
  `SGLANG_ROPE_BATCHED`). Edits in the editable repo `/sgl-workspace/sglang`
  (fp8_utils.py, deepseek_v4_rope.py) — persist in this clone, lost on container rebuild.
- Next: (a) isolate GEMM-only vs RoPE-only; (b) run a FULL c512 1k/1k & 8k/1k A/B
  (not just pure-prefill) to confirm the end-to-end throughput gain; (c) attack the
  remaining ~10% per-step bubble (Exp 48: launch batching / prefill CUDA graph).

### Artifacts
- `/workspace/bench_pp_kern/` (isl{1024,8192}_osl1), gsm8k `/workspace/gsm8k_kern.log`,
  server `/workspace/sgl_kern_server.log`. Baselines: `/workspace/bench_pp_{sgl,atom}/`.
  Server cleaned (VRAM ~0.3 GB/GPU).

### Follow-up — ISL=8192 pure-prefill TRACE of the fixed build (confirms kernel swap)
Re-captured the prefill trace WITH both flags on (ISL8192 OSL1, rank0, 10 steps),
same GPU-active-union-per-attn-layer-step method as Exp 48.
| | GPU-active/step | busy% | vs ATOM |
|---|---:|---:|---:|
| SGL base (Triton GEMM) | 44.81 ms | 89% | +7.9% |
| SGL +CK+ROPE | **42.58 ms** | 92% | **+2.5%** |
| ATOM | 41.54 ms | 82% | — |
- Kernel-level confirmation: Triton `_gemm_a8w8_blockscale` is GONE; CK
  `QuantGemmKernel` (1048 ms ×549, same as ATOM) is now used; old per-token rope GONE,
  batched rope present.
- GPU-active raw-kernel gap to ATOM shrank from +7.9% → **+2.5%** (kernel swap recovered
  most of the raw compute). busy% 89→92% ⇒ CK GEMM also cut launch/bubble slightly,
  which is why pure-prefill THROUGHPUT gain (+8.8%) > GPU-active reduction (−5%).
- Remaining ~2.5% GPU-active to ATOM = minor residual (some projection GEMM / other
  kernel). Trace: `/workspace/sgl_prof_kern/*TP-0-DP-0.trace.json.gz` (92% busy);
  baseline `/workspace/sgl_prof/1781761067*`, ATOM `/workspace/atom_prof/dp0_tp0/*`.

---

