# DeepSeek-V4-Pro serving perf — experiment log (2026-06-16)

Split from the master `EXPERIMENT_LOG.md` (chronological, by date). See that file for the index and `SKILL.md` for how-to.

---

## Exp 35 — c256 gap localized to prefill per-token COMPUTE (+8%), symmetric measurement — 2026-06-16

Re-opened the c256 residual (~93% of ATOM single-stream) on latest upstream
(branch `feat/dp-moe-reduce-scatter`, editable `/sgl-workspace/sglang-upstream`,
SGLANG_DP_USE_GATHERV=1, chunk131072=16384/rank, delayer on, graph on, cons 1.0).
Artifacts in `/sgl-workspace/c256_analysis/`.

### Methodology guardrail (two corrections made mid-investigation)
This experiment twice produced a WRONG conclusion from single-sided / stale data;
both were caught and retracted. Recorded here so the trap isn't repeated:
1. **"SGLang fragments prefill, ATOM packs 16384"** — WRONG. It cited the Exp 15
   trace (OSL128/ratio1.0/chunk16384), not this config, and compared SGLang
   PER-RANK numbers against ATOM numbers misread as GLOBAL.
2. After collecting ATOM's real per-step data it first looked like ATOM ran one
   global scheduler — also WRONG.
The rule confirmed: **never conclude an SGLang-vs-ATOM gap without measuring BOTH
engines with the SAME method, same config, same basis (per-rank).**

### Verified the gatherv+reduce_scatterv pair is actually live (probe)
Env-gated counter probe on `_dp_gather` / `dp_reduce_scatter_tensor` /
deepseek_v4 combine over a real c256 run: prefill `gatherv_prefill` increments,
`allreduce_fallback_prefill=0`, `allgather_maxlen=0`, `moe_combine_reduce_scatterv`
increments. So A-fix is firing on every prefill step (no fallback). Probe removed
after verification (git clean).

### Fast repro config (3.8 min/run instead of ~13)
c256, ISL8192, OSL1024, ratio0.8, **np=512 warm=256** (vs full np=2048/warm=512).
Holds the prefill<->decode regime (conc/ISL/OSL fixed); only np shrinks. Verified
on BOTH engines it preserves the gap:
| config | SGLang tok/s | ATOM tok/s | SGL/ATOM |
|---|---:|---:|---:|
| fast np512 | 24,761 / 24,984±94 (3x) | 26,742 / 26,705 | 92.6–93.4% |
| full np2048 (handoff) | 25,258 | 27,023 | 93.5% |
OSL must stay 1024 — OSL256 spikes total to 34k (over-weights prefill, leaves the
valley regime; matches Exp 16 warning).

### Phase 1 — prefill vs decode split (per-request bench --output-details)
Identical workload both engines (same input/output token totals):
| metric | SGLang | ATOM | SGL/ATOM |
|---|---:|---:|---:|
| total tok/s | 24,761 | 26,742 | 92.6% |
| decode-only tok/s | 3,870 | 4,026 | 96% (near parity) |
| median TTFT (prefill) | 4,213 | 3,308 | 127% (27% slower) |
| median TPOT (decode step) | 66.4 | 64.0 | 104% |
⇒ The c256 gap is in PREFILL (TTFT), decode is parity. (Confirms Exp 14 with a
clean per-request apple-to-apple split, not the old trace.)

### Prefill chunk composition is NEARLY IDENTICAL (apple-to-apple, per-rank)
Confirmed from source that ATOM `enable_dp_attention=True` with launch
`data_parallel_size=1` is RESET by engine_core_mgr.py:32-40 to dp_size = tp×dp = 8,
tp=1 → **8 independent per-rank EngineCores**, each with prefill budget
`max_num_batched_tokens=16384` (NOT divided by dp; model_runner.py:1114-1115).
SGLang dp8 = chunk131072÷8 = 16384/rank. So **both = 16384 tok/rank/step → the
configs ARE apple-to-apple** (answers the recurring "is this fair?" question).
ATOM `Scheduled prefill batch:` log is PER-RANK (8 EngineCores share one stdout).
Per-rank prefill composition over the fast run:
| | per-rank steps | mean tok/step | full-chunk(>=16k) |
|---|---:|---:|---:|
| ATOM | ~59 | 12,103 | 46% |
| SGLang | ~64 | 11,428 | 45% |
⇒ chunk granularity is NOT the gap (both ~45% full, tok/step within 6%).

### Phase 3 — symmetric per-rank prefill forward-wall timer (THE KEY RESULT)
Added an identical env-gated CUDA-event timer (continuous record, single sync at
flush; rank0 only) around the model forward on BOTH engines:
- SGLang: `model_runner.forward` around `_forward_raw` (SGLANG_FWD_TIMER=1).
- ATOM: `model_runner.run_model` whole body (ATOM_FWD_TIMER=1).
Same fast config, aggregated over the run:
| | prefill steps | prefill tok | **us/tok** | tok/ms | decode ms/step |
|---|---:|---:|---:|---:|---:|
| ATOM | 59 | 729,218 | **168.7** | 5.93 | 41.4 |
| SGLang | 64 | 719,838 | **182.3** | 5.48 | 40.2 |
| SGL/ATOM | | | **+8.1% slower** | | 0.97 (parity) |

**ROOT CAUSE (quantified): SGLang's prefill forward costs ~8% more per token
than ATOM's, at the SAME per-rank token count. Decode forward is parity.**
This is the long-suspected "compute residual" (expert GEMM / fp8 quant / MLA),
now measured directly, not inferred. The 8%/token amplifies to TTFT +27%
(a request spans many prefill forwards + queues behind decode) → total tok/s ~7%.

Caveat on scope match: SGLang timer brackets `_forward_raw` (model only); ATOM
`run_model` includes `compute_logits` (small for prefill: last-token only). Both
prefill paths are eager. The ~8% is robust to this minor scope diff but a
per-module breakdown (Exp 36 plan) should re-confirm.

### Levers already rejected (do NOT re-run)
- `--schedule-conservativeness 2.0` (Exp 5 / handoff C5): no stacking benefit
  after A-fix (-1%). The 8% is compute, not admission throttle.
- prefill-delayer tuning (Exp 5): default is the sweet spot; stronger over-throttles.

### Diagnostic code removed
SGLang `model_runner.py` + `dp_attention.py` + `deepseek_v4.py` and ATOM
`model_engine/model_runner.py` all reverted (git diff clean on the SGLang repo;
ATOM site-packages restored). Scripts kept in `/sgl-workspace/c256_analysis/`:
`bench_one.sh`, `analyze_phase1.py`, `analyze_phase2.py`, `ATOM_baseline.txt`.

### Next: Exp 36 — per-module / per-kernel breakdown of the prefill 8%
KEY HYPOTHESIS TENSION (user): both engines call the SAME aiter kernels for MoE
GEMM / fp8 quant / MLA, so a per-kernel gap "shouldn't" exist. So the 8% must
come from HOW each engine invokes the shared kernels, not the kernels themselves.
Candidate sources to isolate (all testable SGLang-internal + controlled microbench):
1. **GEMM M / gather buffer size**: SGLang MoE runs on the gathered global buffer
   (M = 8×16384 = 131072), ATOM too (flatten tp64) — but verify the EXACT per-rank
   expert-GEMM M and whether SGLang pads more (DpPaddingMode rounding) → bigger M →
   more padding-token compute at the SAME tok/ms denominator (inflates us/REAL-tok).
2. **fp8 quant path**: SGLANG_OPT_FP8_WO_A_GEMM / per-token-group quant shape vs
   ATOM AITER_BF16_FP8_MOE_BOUND=0 path — confirm both hit the same aiter quant
   kernel and same group size; a different quant granularity changes cost.
3. **MLA prefill attention**: indexer / unified_kv_triton (SGLANG_HACK_FLASHMLA_
   BACKEND) vs ATOM's MLA — these may NOT be the same kernel (SGLang uses triton
   indexer + flashmla; ATOM may use a different prefill MLA). Likely the biggest
   suspect since MoE GEMM is genuinely shared aiter.
4. **gate/router redundancy** (C2, Exp 34): SGLang computes gate on global M=131072
   (8× redundant) vs ATOM local — measured ~7% of MoE earlier; re-confirm under
   the per-module timer at prefill.
5. **host-side launch / layout**: extra contiguous()/copy, more kernel launches per
   layer on SGLang's path (non-kernel wall inside the forward).
Method: add a per-MODULE CUDA-event timer (attn / gate / gather / moe1 / moe2 /
quant / topk-combine / scatter) inside one SGLang prefill forward (continuous
record, single sync), get the ms breakdown of the ~182us/tok; then isolated
aiter microbench at the EXACT prefill shapes (M per rank, fp8) to see if the
shared kernel is equal in isolation (→ gap is invocation/shape) or different
(→ gap is a non-shared kernel, e.g. MLA). Start with #3 (MLA) and #1 (M/padding)
as highest-probability; #2/#4/#5 if residual remains.

## Exp 36 — prefill +8% broken down: it is NOT MoE/comm (shared), points to attn(MLA) — 2026-06-16

Executed the Exp 36 plan. Two-stage: (1) SGLang per-module CUDA-event timer to
split the prefill forward; (2) neutral isolated microbench of the shared aiter
kernels + RCCL collectives. ATOM per-module probe attempted and CONFIRMED
infeasible (below). Artifacts `/sgl-workspace/c256_analysis/`.

### SGLang prefill forward per-module breakdown (CUDA-event, rank0, prefill-only)
Timer wrapped attn / hc_norm / gather / moe(mlp) / scatter in the DSV4 decoder
layer (deepseek_v4.py forward). Aggregated over a fast-config c256 run; the 5
blocks sum to 128,982 ms ≈ the Exp 35 forward wall 131,249 ms (≈98% coverage):
| block | share of prefill forward | nature |
|---|---:|---|
| moe (expert GEMM) | 35.6% | shared aiter fused_moe |
| attn (MLA) | 34.2% | SGLang-specific (unified_kv_triton + aiter indexer + compressor) |
| gather (all_gatherv) | 18.2% | DP-MoE comm (RCCL) |
| scatter (reduce_scatterv) | 10.1% | DP-MoE comm (RCCL) |
| hc_norm | 1.9% | fused HC norm |
CORRECTION to the Exp 35 first cut ("other 40%"): the "other" is actually
gather+scatter = **28% DP-MoE communication**, not gate/quant/host glue.

### Stage 2a — isolated aiter MoE GEMM (neutral, engine-independent)
`/workspace/b1_iso_moe.py`, aiter `fused_moe` at the gathered prefill shape:
| M (tokens) | fused_moe | note |
|---|---:|---|
| 16,384 (1 rank real) | 1.72 ms | |
| 131,072 (gathered 8×) | 12.02 ms | both engines run on this M |
8× tokens → 7× time (sub-linear) ⇒ compute-bound, efficient at large M, NO
redundancy penalty from the global buffer. Per-kernel: moe2 37% / moe1 34% /
top-k reduce_kernel 19% / quant 4% / sort 5%. Both engines call THIS SAME aiter
kernel at the SAME M ⇒ MoE GEMM is equal, NOT the gap. (Confirms the user's
"shared aiter kernel shouldn't differ" intuition for MoE.)

### Stage 2b — neutral collective microbench (RCCL floor)
`/workspace/moe_comm_microbench.py`, 8×MI355X, hid7168 bf16, aligned all-prefill
(every rank 16384 → 131072 global):
- all_gatherv = 5.25 ms/call (moves 1879 MB). This is the RCCL hardware floor.
Reconcile vs the in-server module timer: SGLang gather = 6.10 ms/layer-call,
scatter = 3.41 ms/layer-call. gather 6.1 vs floor 5.25 ⇒ SGLang's all_gatherv is
within ~16% of the raw RCCL floor (per-layer reuse overhead). ATOM also uses
all_gatherv+reduce_scatterv (PR #930) over the same bytes on the same RCCL ⇒
**comm is at the hardware floor on both, NOT the gap.**

### ATOM per-module probe — CONFIRMED INFEASIBLE (validates Exp 34 note)
Tried the symmetric per-module timer inside ATOM's DSV4 decoder layer
(model_runner run_model + models/deepseek_v4 layer forward). ATOM crashed at
init: `torch._dynamo.exc.BackendCompilerFailed: cannot extract sympy expressions
from <torch.cuda.Event>`. ATOM's decoder layer is **torch.compile-wrapped
(VllmBackend)**, so inserting CUDA-Event objects breaks Dynamo graph tracing.
⇒ In-model module probes are impossible on ATOM. The neutral microbench (shared
kernel + RCCL floor) is the ONLY way to get an ATOM-comparable baseline for the
shared blocks; the non-shared block (attn) can only be bounded, not directly
ATOM-probed. All ATOM edits reverted (site-packages restored).

### CONCLUSION (by elimination)
The 8% prefill per-token gap (Exp 35: SGLang 182.3 vs ATOM 168.7 us/tok) is:
- NOT MoE GEMM (35.6% of fwd, shared aiter, isolated-equal).
- NOT comm gather+scatter (28% of fwd, at RCCL floor, both engines same primitive).
- ⇒ **most likely in attn / MLA (34.2% of fwd)** — the only large block that is
  NOT a shared kernel: SGLang uses unified_kv_triton + aiter indexer + compressor;
  ATOM has its own prefill MLA. This is exactly consistent with the user's framing:
  shared aiter kernels (MoE) match; the gap lives in the engine-specific MLA path.
(Caveat: this is by elimination, not a direct ATOM attn measurement, which is
impossible via in-model probe. A direct attn comparison would need an isolated
MLA-prefill microbench replicating each engine's attention kernel sequence — the
recommended Exp 37.)

### Next: Exp 37 — isolate the MLA prefill attention cost
Build an isolated microbench of SGLang's MLA prefill kernel sequence (q/kv proj,
aiter indexer, flashmla/unified_kv core attn) at the c256 prefill shape (16384
tok/rank, the DSV4 head/dim config), time it standalone; compare against ATOM's
MLA prefill kernels (the aiter MLA ops ATOM calls, run standalone — these MAY be
shared aiter ops even if the surrounding glue differs). If the isolated MLA
kernels are equal, the residual is SGLang attn glue/launch overhead (host-side);
if different, it's a genuine kernel-path difference. This is the last bucket;
expected end-to-end upside is ~the 8% prefill share of the c256 gap (a few % of
total tok/s), diminishing ROI now that MoE/comm are ruled out.

### Diagnostic code removed
SGLang deepseek_v4.py module timer + ATOM model_runner/deepseek_v4 timers all
reverted (SGLang git diff clean; ATOM site-packages restored). Microbench
scripts reused: /workspace/b1_iso_moe.py, /workspace/moe_comm_microbench.py.

## Exp 37 — split the attn block; out_proj wo_a einsum is NOT slow (correction) — 2026-06-16

Followed up Exp 36 (gap is in attn/MLA, 34% of prefill fwd). Added an env-gated
CUDA-event timer splitting the MLA attn forward (deepseek_v4.py MQALayer.forward)
into qkv_prep / core_attn / out_proj; ran the fast c256 config. Then a neutral
microbench of the wo_a output projection. Diagnostic code removed after (git clean).

### attn sub-block split (CUDA-event, prefill, rank0) — see caveat below
| sub-block | share of attn | spans |
|---|---:|---|
| out_proj | 49.3% | rope_inplace + o.view + wo_a einsum + wo_b |
| qkv_prep | 30.5% | q/kv proj + aiter indexer + compressor |
| core_attn | 20.2% | unified_kv_triton / flashmla core attention |

### Hypothesis (out_proj einsum slow) — TESTED and REJECTED
Our config has `SGLANG_OPT_FP8_WO_A_GEMM=false`, so wo_a runs as
`torch.einsum("tgd,grd->tgr")` in bf16 (not the fp8/deep_gemm path). Hypothesis:
this einsum is the slow bucket. Two findings refuted it:
1. **Can't even enable the fp8 path here**: `SGLANG_OPT_FP8_WO_A_GEMM=true`
   crashes at init — `ModuleNotFoundError: No module named 'deep_gemm'`. This
   ROCm build has NO deep_gemm, so the aligned config MUST use the einsum path.
   (Informative: SGLang on this build is forced onto the bf16 einsum wo_a; ATOM
   presumably has an equivalent optimized o-proj. But we cannot A/B it here.)
2. **The einsum is already fast**: isolated microbench at the real shape
   (T=16384, G=16, R=1024, D=3584) — `torch.einsum` = 1.36 ms = **1411 TFLOP/s**
   (torch.bmm same, 1.44 ms). 0.083 us/tok × 61 layers ≈ 5 us/tok of the ~182 —
   small, near-peak. The wo_a einsum is NOT the gap. (`wo_a_microbench.py`)

### CAVEAT — attn sub-block split is unreliable (single-stream async tail)
Per-block CUDA events on one stream attribute the async tail of the previous
block's still-draining kernels to the next block's measured window. core_attn's
attention kernels likely drain into the out_proj window, inflating out_proj's
49%. So the in-attn sub-attribution (out_proj 49 / qkv 31 / core 20) is NOT a
trustworthy decomposition; only the attn-BLOCK-as-a-whole (34% of fwd, Exp 36)
and the isolated microbenches (MoE equal, comm at floor, wo_a fast) are reliable.

### Net (Exp 37)
Ruled OUT the wo_a einsum as the gap. The reliable picture stands from Exp 35/36:
- gap is prefill, +8.1% us/tok; decode parity.
- MoE GEMM shared/equal; gather+scatter at RCCL floor → not the gap.
- ⇒ the 8% is in the MLA attn block, but it could NOT be cleanly sub-attributed
  (async-tail) and the one concrete sub-suspect (wo_a) is fast. The remaining
  candidates inside attn are core_attn (unified_kv_triton vs ATOM's MLA — a
  genuine non-shared kernel) and qkv_prep (aiter indexer/compressor). A reliable
  next step needs ISOLATED timing of each attn sub-kernel (separate launches with
  their own sync, NOT chained single-stream events) — deferred (Exp 38, low ROI:
  the whole attn gap is ~8% of prefill ≈ a few % of total tok/s).

### Conclusion of the c256 investigation (Exp 34–37)
The c256 gap (~93% of ATOM single-stream) decomposes as: comm fixed/at-floor
(gatherv+reduce_scatterv, shipped), MoE compute equal (shared aiter), decode
parity; the irreducible residual is ~8% per-token PREFILL compute living in the
engine-specific MLA attention path (not the shared MoE/quant kernels, consistent
with the "same aiter kernel shouldn't differ" expectation). Diminishing ROI to
chase further; the shipped gatherv+A-fix is the main, defensible c256 win.

## Exp 38 — re-evaluate the 8x-redundant gate/router GEMM (C2 corrected) — 2026-06-16

User asked to actually quantify the "SGLang does ~8x the gate GEMM" lever that
C2 (Exp 34) had dismissed at "7.2%, low ROI". Re-measured it NEUTRALLY (isolated
microbench of the exact aiter gate kernel), because the C2 number came from a
single in-server module-event that is subject to the same async-tail
mis-attribution found in Exp 37.

### The redundancy (confirmed real, SGLang-specific)
DSV4 dp-attn + TP-MoE (no EP): SGLang gathers hidden → global buffer (M=131072)
then computes the gate on the WHOLE buffer (`MoEGate.forward`, deepseek_v2.py:985
→ `linear_bf16_fp32` → aiter `tgemm.mm`, hidden7168→384 experts, fp32 logits).
So every rank computes the router logits for ALL 8 ranks' tokens and uses only
its own 1/8. ATOM computes the router LOCALLY (M=16384) then gathers hidden+router
together (cat). This IS a genuine SGLang-only redundancy (unlike the expert GEMM,
which is global-buffer on both engines — Exp 36).

### Isolated gate-GEMM microbench (aiter tgemm.mm, hidden7168→384, fp32)
| | M | ms/layer | GFLOP/s |
|---|---:|---:|---:|
| ATOM local | 16,384 | 0.127 | 709 |
| SGLang global | 131,072 | 0.644 | 1120 |
- Redundancy = **5.1x** (not 8x): the small M=16384 GEMM is launch/mem-bound
  (709 GFLOP/s) while M=131072 is more efficient (1120 GFLOP/s), so 8x tokens
  cost only 5.1x time.
- SGLang-only WASTE = 0.517 ms/layer × ~58 MoE layers = **~30 ms/prefill step**.
- vs prefill forward ~2050 ms/step (Exp 35) ⇒ **~1.5% of the prefill forward**
  ⇒ closing it would gain ~1.5% of prefill ≈ **~+1% total tok/s at c256**.
Script: `/sgl-workspace/c256_analysis/gate_microbench.py`.

### CORRECTION to C2 (Exp 34)
C2 reported gate = 0.068 ms/layer = 7.2% of MoE and called it low-ROI. The
isolated kernel is **0.644 ms/layer — ~10x higher** than the in-server event
measured. The C2 in-server number under-counted because the small gate GEMM's
real cost was hidden in neighbor-kernel overlap (single-stream event tail, same
artifact as Exp 37). So C2's "7.2% / negligible" was an under-estimate; the true
redundant cost is ~1.5% of the prefill forward (still modest in absolute terms,
but ~2x what C2 implied, and it is a clean, shared-kernel, low-risk target).

### Is it worth fixing? (assessment)
- Upside: ~+1% total tok/s at c256 (the prefill-weighted point); less at low conc.
- Approach: mirror ATOM — compute the gate on LOCAL hidden (M=16384) BEFORE the
  gather, then carry router_logits through the gather (cat hidden+logits, or a
  second small gather) so the per-rank slice has its routing. This touches the
  DSV4 gather payload + MoE entry, adjacent to the shipped gatherv/reduce_scatterv
  path — needs care to keep the reduce_scatterv combine symmetric and accuracy
  intact.
- Risk: medium (changes the gather contract); ROI ~+1%. Reasonable as a
  follow-up optimization PR, separate from the gatherv+A-fix that already landed.
  NOT a large win, but it is REAL, quantified, and the cleanest remaining lever
  (shared kernel, no MLA/non-shared-kernel surgery).

Net: the "8x gate GEMM" is a real ~1.5%-of-prefill redundancy (C2 under-stated
it). It's the most actionable remaining c256 lever after gatherv+A-fix, but a
modest ~+1% — consistent with the overall finding that the big buckets (MoE,
comm) are already equal/shipped.

### Implementation (gate-local) — built, debugged to CORRECT, but net-zero tput
Implemented env-gated `SGLANG_DP_GATE_LOCAL` (default OFF): `MoEGate.forward`
returns a stashed precomputed-logits tensor (skip gate GEMM);
`DeepseekV4DecoderLayer._dp_gate_local_gather` computes the gate on LOCAL hidden,
gathers hidden + router-logits, fills the global buffer, stashes the global logits.

Two bugs found & fixed (a COLLECTIVE PROBE — ENTER/EXIT per rank+layer+op,
sentinel-gated, flush-each — pinpointed both):
1. **HANG (collective desync).** First version gated gate-local on per-rank
   `forward_batch.forward_mode.is_extend()`. The probe showed: in a MIXED step
   only the prefilling rank (`pf=True`) took gate-local → `all_gatherv(width
   7552)` while the other 7 ranks (`pf=False`) took the normal
   `dp_gather_partial` → `all_gatherv(width 7168)`. Same collective, DIFFERENT
   width across ranks → RCCL hang. FIX: gate-local must be an ALL-RANKS decision
   — key it ONLY on `_use_gatherv_pair` (derived from the synced dp_padding_mode,
   identical on all ranks), NOT per-rank is_extend. After: no hang, "17+25=42" OK.
2. **ACCURACY collapse (bf16 router logits).** gsm8k A/B: OFF 0.9469 vs ON
   **0.5876**. Cause: I cast the fp32 router logits to bf16 to cat+gather them
   with the bf16 hidden — the bf16 round-trip changes the ungrouped top-6 expert
   selection → wrong routing. FIX: gather the logits in **fp32** via a SEPARATE
   small `all_gatherv` (384 fp32 = 1.5KB/tok). After: gsm8k ON **0.9424** ≈ OFF
   0.9469 (Δ−0.45%, within ±0.64% noise). Routing correct.

### Result: gate-local is CORRECT but gives ~0 total throughput (TTFT −11%)
c256 fast-config A/B (gatherv ON both, only SGLANG_DP_GATE_LOCAL toggled):
| | total tok/s | median TTFT | median TPOT |
|---|---:|---:|---:|
| OFF | 24,736 | 4,206 | 76.9 |
| ON  | 24,725 | 3,759 | 76.9 |
| Δ   | **−0.04% (noise)** | **−10.6%** | ~0 |
So the gate redundancy is real and **TTFT improves ~11%** (the gate GEMM saving
shows up in the prefill phase), but **total throughput does NOT move**:
- the second fp32-logits `all_gatherv` (needed for correctness) costs roughly
  what the gate GEMM saved — net ~0 (exactly the "option (a) nets ~0" predicted
  earlier; the fused single bf16 gather that WOULD net-win is incorrect).
- c256 total tput is decode-dominated, so an 11% prefill-TTFT win barely moves it.

### Why a SECOND gather was needed — ATOM uses ONE; the diff is router dtype
Checked ATOM's `dp_gather_hidden_and_router` (model_ops/moe.py:240). ATOM does
the single FUSED gather: cast router_logits to hidden dtype, `cat`, ONE
`all_gatherv`, split. It has NO second gather. The reason it works in bf16:
**ATOM's gate is `ReplicatedLinear` → bf16 logits** (gate output dtype = input
bf16), so the cast in the fused gather is a NO-OP. SGLang's DSV4 gate is
`linear_bf16_fp32` → **fp32 logits** (deliberately, for router numerical
stability), so casting to bf16 for a fused gather LOSES precision.

### Option A tested: make SGLang's gather bf16 too (single fused, like ATOM)
Re-implemented gate-local with a single fused bf16 gather (cast logits→bf16,
exactly ATOM's scheme). gsm8k = **0.5754** — collapses AGAIN (matches the v1 bf16
result 0.5876; fp32 gave 0.9424). So **SGLang's DSV4 router CANNOT tolerate bf16
logits, even though ATOM's can.** Confirmed twice (independent runs). The DSV4
router (sqrtsoftplus + fp32 e_score_correction_bias + ungrouped top-6) is
numerically sensitive enough in SGLang's topk path that bf16 logits flip the
top-6 expert selection → wrong routing. ATOM tolerates bf16 because its
gate/topk/bias path is structured differently (its gate is natively bf16 and it
ships fine at 0.95).

### Conclusion (gate-local, FINAL)
gate-local is CORRECT only with fp32 logits, which forces a SECOND all_gatherv,
which offsets the saved gate GEMM → **net ~0 total tput at c256** (TTFT −11% but
decode-bound total unmoved). The single-fused-bf16 gather that WOULD net-win
(ATOM's scheme) BREAKS SGLang accuracy. Pushed further to find out WHY.

### Single gather IS possible (bitcast); the REAL blocker is per-M GEMM routing
Got a SINGLE gather with fp32 precision via BITCAST: reinterpret the fp32 logits
as 2x bf16 columns (byte-identical), cat with bf16 hidden, ONE bf16 all_gatherv,
split + view back to fp32. Verified byte-exact through RCCL all_gather, even AND
uneven `sizes` (maxerr=0, standalone). So the dtype/2-gather issue is solvable —
**a single gather works.** But it STILL gave gsm8k 0.58. Clean isolation found
the true root cause (NOT precision, NOT gather, NOT memory lifetime — .clone()
didn't help):
- **NOSKIP test** (fused gather but DON'T skip the gate; recompute gate on the
  gathered global buffer) → gsm8k **0.9477**. ⇒ gather + hidden fill are PERFECT;
  the bug is purely in skipping the gate via the stash.
- **GLTOPK probe** (stash logits = gate on LOCAL hidden M≈12–16384/rank, vs gate
  recomputed on GLOBAL M=131072): **8.3% of tokens get a DIFFERENT top-6 expert
  set**, logits_maxdiff≈0.03.
- ROOT CAUSE: aiter's router GEMM (`aiter_dsv3_router_gemm`/`tgemm.mm`) is
  **autotuned per-M** — different kernel/tile for small per-rank M vs global
  M=131072 → different bf16 rounding (~0.03) → flips ~8% of the borderline
  ungrouped-top-6 expert picks → degraded routing → 0.58.

⇒ gate-local is **fundamentally unsafe in SGLang**: computing the gate at local-M
diverges numerically from the model's reference global-M gate (the gate GEMM is
NOT M-invariant on aiter). ATOM avoids this because its whole pipeline gates
locally end-to-end (no global-M reference to diverge from); SGLang's reference IS
the global-M gate.

### CORRECTION — gate-local DOES work; 0.58 was a buffer bug, NOT per-M/local-M
The "per-M GEMM / local-M reference" theory above was WRONG. Verified by
microbench: the SAME tokens gated at M=16384 vs M=131072 give **top-6 flip = 0.0%**
(mean logit diff 3e-05) — `tgemm.mm` is effectively M-invariant for routing. And
ATOM's gate is literally the SAME aiter `tgemm.mm` (linear.py:603), computed at
local-M; ATOM's whole flow is gate@local-M → gather(hidden+logits) → topk@global
— IDENTICAL to gate-local's flow. So local-M routing is NOT the problem.

REAL ROOT CAUSE of the 0.58 (found by NOSKIP isolation + buffer inspection):
`get_global_dp_buffer()` returns a **fresh `torch.empty` every call** (not a
persistent buffer). My gate-local called it TWICE — the helper filled buffer #1
with the gathered hidden, then the call site did `hidden_states =
get_global_dp_buffer()` again → buffer #2 (uninitialized GARBAGE). The MoE then
ran experts on garbage hidden while routing with correct logits → 0.58.
FIX: the helper returns the buffer it filled; the caller uses THAT instance
(never re-calls get_global_dp_buffer). After the fix: **gsm8k 0.9492 ≈ OFF 0.9469**.
(This also explains why NOSKIP=0.9477: NOSKIP recomputes the gate on buffer #2,
but in that variant buffer #2 WAS the one dp_gather filled — different code path
that happened to use one buffer. The bug was specific to the skip-path's double
get_global_dp_buffer.)

### gate-local FINAL result — CORRECT, single gather, but throughput-neutral
With the buffer fix + single fused bitcast gather (fp32 logits losslessly carried
as 2x bf16) + local-M gate skipping the redundant global gate GEMM:
- **Correctness: gsm8k 0.9492** (OFF 0.9469) — fully correct, ATOM-equivalent flow.
- **c256 throughput (fast config np512)**: ON 24,524 vs OFF 24,687–24,736 = −0.7%.
- **c256 throughput (FULL config np2048/warm512, ROCM700A=0, gatherv ON)** —
  the authoritative measurement:
  | | total tok/s | out tok/s | TTFT | TPOT | dur |
  |---|---:|---:|---:|---:|---:|
  | OFF | 25,354 | 2,817 | 2,008 | 87.2 | 670.8s |
  | ON  | 25,306 | 2,811 | 1,983 | 87.2 | 672.1s |
  | Δ   | **−0.19% (neutral)** | −0.2% | **−1.3%** | ~0 | +0.2% |
  Full config confirms & tightens the fast-config result: total tput is NEUTRAL
  (−0.19%, in noise), TTFT slightly better (−1.3%), TPOT parity (decode-bound).
So the ATOM local-M logic IS correctly portable to SGLang (single gather, gsm8k
passes), but at c256 it does NOT improve total throughput: the saved gate GEMM
(~1.5% of prefill) is offset by the wider fused gather (7936 vs 7168) AND c256 is
decode-bound so prefill savings barely move total. TTFT improves slightly.
⇒ gate-local is a correct, ATOM-aligned refactor but NOT a c256 throughput win.
Could help prefill-heavy / TTFT-sensitive workloads. REVERTED (git checkout; PR
pristine). c256 throughput-lever search closed: nothing beyond shipped
gatherv+A-fix moves total tput.
Scripts kept: `/sgl-workspace/c256_analysis/gate_microbench.py`,
`launch_sgl_wofp8.sh` (GATE_LOCAL/WO_FP8 knobs), `gsm8k_gatelocal/`.

---

