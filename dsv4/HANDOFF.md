# DeepSeek-V4 DP performance — Handoff summary (updated 2026-06-16)

Read this first to pick up the investigation in a new session. Full detail is in
`EXPERIMENT_LOG.md` (Exp 1–38 + appendix); methods/scripts in `SKILL.md`.

> **Latest status (2026-06-16) — c256 investigation CLOSED. No throughput lever
> remains beyond shipped gatherv+A-fix.** Breakdown (Exp 35–38): comm at RCCL
> floor (shipped), MoE GEMM equal (shared aiter), decode parity; residual ~8%
> prefill compute is in the engine-specific MLA path (couldn't sub-attribute
> reliably; wo_a einsum ruled out as fast). The "8x gate GEMM" redundancy (Exp 38)
> is REAL but is throughput-neutral at c256 (Exp 38, fully resolved). ATOM's
> local-M gate logic IS correctly portable to SGLang: gate on LOCAL hidden +
> ONE fused all_gatherv (fp32 logits losslessly bitcast as 2x bf16) + skip the
> redundant global gate GEMM → **gsm8k 0.9492 ≈ OFF 0.9469 (correct)**. ATOM's
> gate is the SAME aiter `tgemm.mm`; it is M-invariant (microbench: gate@M16384
> vs M131072 = 0% top-6 flip). The earlier "local-M routing diverges / per-M
> GEMM" and "router can't tolerate bf16" theories were BOTH WRONG. The real 0.58
> bug was a **buffer-instance error**: `get_global_dp_buffer()` returns a fresh
> torch.empty each call; calling it twice made the MoE run on uninitialized
> garbage hidden. Fixed by reusing the filled buffer. **But c256 throughput: ON
> 25,306 vs OFF 25,354 = −0.19% (NEUTRAL, FULL config np2048 ROCM700A=0 gatherv
> ON), TTFT −1.3% (2008→1983), TPOT parity** (fast-config np512 gave −0.7%, same
> direction) — the saved gate GEMM is offset by the wider fused gather and c256
> is decode-bound. So gate-local is a correct ATOM-aligned refactor but NOT a
> c256 tput win (may help prefill-heavy/TTFT workloads). All experimental code
> reverted (PR pristine).
> Shipped gatherv+A-fix (PR #28216) remains the defensible c256 win; remaining
> gap is MLA compute. c256 throughput-lever search CLOSED.

> **Earlier (Exp 36) — c256 prefill +8% narrowed to attn/MLA:**
> the c256 gap is prefill per-token compute (SGLang 182.3 vs ATOM 168.7 us/tok,
> +8.1%; decode parity). Exp 36 broke the SGLang prefill forward into moe 36% /
> attn 34% / gather 18% / scatter 10% / hc_norm 2%, then ruled out the shared
> pieces with neutral microbenches: **MoE GEMM is the SAME aiter `fused_moe` at
> the SAME M=131072 (12.0ms, isolated-equal) → not the gap; gather+scatter sit at
> the RCCL hardware floor (gather 6.1ms vs floor 5.25ms, both engines use
> all_gatherv+reduce_scatterv) → not the gap.** By elimination the 8% lives in
> **attn / MLA (34% of fwd, the only large NON-shared block)** — SGLang uses
> unified_kv_triton + aiter indexer + compressor; ATOM has its own prefill MLA.
> This matches the user's intuition exactly (shared aiter kernels match; the gap
> is in the engine-specific MLA path). NOTE: ATOM in-model per-module probes are
> IMPOSSIBLE (torch.compile breaks on inserted cuda.Event — confirmed). Next:
> Exp 37 isolate the MLA-prefill kernel sequence. See bottom section.
>
> **Methodology rule (learned the hard way in Exp 35):** never claim an
> SGLang-vs-ATOM gap from single-sided or stale data. Measure BOTH engines, same
> method, same config, PER-RANK basis. (Two wrong "fragmentation" conclusions
> were made and retracted before the symmetric timer settled it.)

> **Prior status (2026-06-15 cont.):** found & fixed a real bug — the gatherv
> path was silently falling back to all_reduce on EVERY prefill step (A-fix).
> After the fix, gatherv ON reaches **~98% of ATOM single-stream** at most
> concurrencies (c256 ~93%). A CI test was added and the PR Speed table updated.
> See the "Update 2026-06-15 (cont.) — A-fix + CI test" section at the bottom.

## The problem
SGLang DeepSeek-V4-Pro on 8×MI355X, tp8+dp8 (dp-attention), 8k/1k high
concurrency: total token throughput lagged ATOM, gap widening with concurrency
(~88% of ATOM at 8k/1k c256). Goal: find and close the gap.

## What was ruled OUT (with evidence)
- **prefill-delayer**: ESSENTIAL (+41%), not the cause. Keep ON. (Exp 20)
- **chunked-prefill size**: must be 16384/rank (`--chunked-prefill-size 131072`
  ÷ dp8), but enlarging it doesn't close the gap. (Exp 16/18)
- **retract / SWA pool**: not active at this setting (0 retracts). (Exp 20)
- **MAX_LEN padding blow-up**: SGLang always picks SUM_LEN for mixed steps, so
  the 4.5× pad penalty never occurs. (Exp 24)
- **trace-based per-kernel compute compare**: UNRELIABLE — trace `dur` is
  inflated by host sync waits (kernel-time > wall-time). Don't trust it. (Exp 19)

## What was FOUND + SHIPPED
Root mechanism (Exp 22/28/29): the heavy per-layer DP-MoE collective in SGLang
is the post-experts **all_reduce** (quickreduce + cross_device, ~2.2s in a 6s
trace window), which moves the FULL hidden state to every rank (~2× ring
traffic). ATOM uses a symmetric **all_gatherv (gather) + reduce_scatterv
(combine)** pair where each rank only holds its own token slice.

Both engines TP-shard experts by intermediate (SGLang moe_tp_size = tp//ep//moe_dp
= 8), so the post-experts reduce is a SUM → reduce_scatterv (sum+scatter) is the
correct symmetric inverse of all_gatherv. (Exp 29/31)

**Implemented** the ATOM-style symmetric pair in SGLang, env-gated
`SGLANG_DP_USE_GATHERV` (default OFF), for attn_tp_size==1, tp_size==dp_size:
- gather: variable-length `all_gatherv` (zero-pad each rank to its buffer slot).
- combine: `reduce_scatterv`, AND pass `use_reduce_scatter=True` to the MoE so it
  SKIPS its internal post-experts all_reduce (else double-reduce → gsm8k 43%,
  Exp 30/31).

Files: `python/sglang/srt/layers/dp_attention.py`,
`python/sglang/srt/models/deepseek_v4.py`.

## Results
- Correctness: gsm8k 5-shot ON ≈ OFF (~0.94, within noise), 0 errors. (Exp 26/31, re-verified after review fixes)
- Throughput sweep (8k/1k, ratio0.8, chunk16384/rank): +1.0% (c64), +1.1%
  (c128), +2.0% (c256), +3.2% (c512) total tok/s; win GROWS with concurrency,
  matching TPOT/TTFT drops. (Exp 33)
- SGLang best (gatherv ON) vs ATOM single-stream: c64 100.4%, c128 99.2%,
  c256 91.8%, c512 95.2% of ATOM.

## PR status
- Branch `feat/dp-moe-reduce-scatter` on HaiShaw/sglang (clone at
  `/sgl-workspace/sglang-upstream`). Runtime/dev clone at `/sgl-workspace/sglang`.
- PR: **sgl-project/sglang#28216** (Draft). Description filled (Motivation /
  Modifications / Accuracy + Speed with commands / Checklist).
- 2 commits: `fccc7d1` (feature) + `a939d839b` (gemini review robustness fixes:
  torch.cat for all_gatherv list, dp_padding_mode None guards, ValueError catch,
  sizes-None fallback). Both pushed.

## Open / next ideas
- The remaining SGLang↔ATOM gap (a few %, worst at c256) is NOT comm — likely
  COMPUTE (expert GEMM / fp8 quant / MLA). Not yet quantified with a reliable
  method (use a controlled compute microbench, NOT the inflated trace).
- PR follow-ups: unit tests, docs, consider defaulting the pair ON for tp==dp
  dp-attn; broader model/shape validation.

## Key scripts (in /workspace)
- `run_gatherv_ab.sh` / `run_gatherv_sweep.sh`: throughput A/B + concurrency sweep.
- `run_gatherv_gsm8k_ab.sh`: gsm8k correctness A/B.
- `moe_comm_microbench.py`: isolated collective-primitive microbench.
- aligned launch: `/dockerx/home/wunhuang/tmp/useful-scripts/benchmarking/dsv4/run_sgl_dsv4_aligned.sh`

---

## Update 2026-06-15 (cont.) — A-fix + CI test (Exp 34)

Continued on latest upstream (`/sgl-workspace/sglang-upstream`, editable install,
branch `feat/dp-moe-reduce-scatter`). Method: module-level CUDA-event probes
(continuous record, single sync at flush — avoids both per-op pipeline break AND
the Exp 19 trace-dur inflation) + isolated repeated microbench.

### Root cause found (a real bug): gatherv never fired on prefill
A coverage probe on `_dp_gather` over a real c256 run: **PREFILL gather = 100%
fallback to all_reduce (985/985)**, only pure-decode took gatherv. Cause:
`_dp_gatherv_sizes()` returned `global_num_tokens_for_logprob_cpu` (the logprob
token counts, e.g. [3]×8, sum=24) instead of the MoE `global_num_tokens_cpu`
(~16384/rank). Its sum never equals the ceil_align'd global buffer (~129k), so
the `sum(sizes)==buffer_rows` guard failed → all_reduce fallback on every
prefill step. ATOM unconditionally takes variable-length all_gatherv whenever any
rank has prefill.

### A-fix (shipped)
`_dp_gather` now uses `get_dp_global_num_tokens()` (the buffer-aligned sizes
stored by `set_dp_buffer_len`, the SAME source the reduce_scatterv combine uses)
as the gatherv sizes; `_dp_gatherv_sizes()` is only the fallback (logits path).
One-line logic change in `dp_attention.py`, no new flag (the temporary A/B flag
`SGLANG_GATHERV_PREFILL_FIX` was removed before merge).
Verified: c256 prefill gather now `sum==buffer` (zero mismatch) → all_gatherv.

### Verification
- gsm8k 5-shot OFF 0.9484 vs A-fix ON 0.9431 (strict), Δ −0.5% within noise, 0 err.
- c256 A/B (np1024, only the prefill-fix toggled): pre-fix 25,160 → A-fix 25,885
  (+2.9%), median TPOT 84.2 → 82.0.

### ATOM-matched sweep (np=conc*8, warm=conc*2, ratio0.8, single-stream)
gatherv OFF vs ON-with-A-fix:
| conc | OFF tok/s | ON tok/s | Δ tput | OFF TPOT | ON TPOT |
|---:|---:|---:|---:|---:|---:|
| 64  | 11,798 | 12,498 | +5.9%  | 46.0  | 43.0  |
| 128 | 17,750 | 18,809 | +6.0%  | 61.4  | 57.8  |
| 256 | 23,821 | 25,258 | +6.0%  | 92.7  | 87.5  |
| 512 | 28,165 | 32,322 | +14.8% | 141.9 | 127.2 |

vs ATOM single-stream (Exp 13): c64 103%, c128 101%, c256 93%, c512 98%.
(The earlier +1~3.2% sweep was the pre-A-fix gatherv, where prefill still fell
back to all_reduce — so it understated the real win. This supersedes it.)

### Other levers checked (rejected, low ROI / not a gap)
- **gate/router on global buffer** (C2): SGLang computes the gate GEMM on the
  gathered global buffer (M=131072, 8× redundant) vs ATOM local (M=16384).
  Isolated GEMM looked big (112us@16k vs 623us@131k) BUT in-server module-event
  shows gate is only 0.068ms/layer = 7.2% of MoE (overlaps with neighbors).
  Refactor ROI low. Not pursued.
- **moe2 top-k combine `at::native::reduce_kernel`** (C4): from
  `aiter/ops/flydsl/moe_kernels.py:1191` `torch.sum`, only on the moe2 *reduce*
  mode (large-tile/prefill `t64x256_reduce`); decode/small-tile uses *atomic*
  (fused). mode is picked by the aiter tuner → it's an aiter/kernel issue, and
  SGLang+ATOM share aiter so it's not an apple-to-apple gap.
- **--schedule-conservativeness 2.0** (C5): +1.4% on the OLD baseline but
  NO stacking benefit after A-fix (25,694 → 25,445, −1%). A-fix already covers
  what cons2 was compensating for. Keep cons at default 1.0.

### CI test (shipped)
`test/registered/dp_attn/test_dp_attention.py::TestDPAttentionGatherv`:
tp2+dp2 dp-attention server launched with `env={"SGLANG_DP_USE_GATHERV":"1"}`
(the layout where gatherv activates) + GSM8KMixin (thres 0.6). Directly closes
the coverage gap amd-bot flagged (the feature was gated behind the env var and
NO PR-CI test exercised it). Verified the path end-to-end on tp8+dp8 with
SGLANG_DP_USE_GATHERV=1: gsm8k stays correct on both DeepseekV4 (0.940) and the
DeepseekV3 family (R1-0528-MXFP4, 0.945 — same DeepseekV3ForCausalLM family as
the CI dsv3-test model). Could not pull `lmsys/sglang-ci-dsv3-test` locally
(token lacks lmsys-org read perm); CI runner has the right perms.

### PR status (updated)
- PR **sgl-project/sglang#28216**, branch `feat/dp-moe-reduce-scatter`.
- Commits now include: feature + gemini-review fixes + **`ec317733b` (A-fix:
  buffer-aligned gatherv sizes)** + **`8798aa97b` ([DP][test] gatherv CI
  coverage)**.
- PR body "Speed Tests and Profiling" table UPDATED to the A-fix numbers above
  (was the old +1~3.2%). Updated via `gh api -X PATCH` because `gh pr edit`
  fails on this repo with a Projects-classic GraphQL deprecation error.

### Remaining gap
c256 is the only clearly weak point (~93% of ATOM single; TPOT 87.5 vs 81.4) —
prefill↔decode interference is worst there. Further upside: attn (MLA) breakdown
(C3, not done) or EP (`--ep-size 8 --moe-a2a-backend deepep`, beats ATOM but
breaks apple-to-apple). Both diminishing ROI now that gatherv+A-fix lands.

### New scripts (in /workspace)
- `run_afix_perf_ab.sh`: A-fix on/off c256 A/B (uses removed SGLANG_GATHERV_PREFILL_FIX).
- `run_c8_sweep.sh` (np*4) / `run_c9_aligned_sweep.sh` (np*8, ATOM-matched): OFF vs ON sweeps.
- `run_c5_cons2.sh`: cons2 stacking A/B.
- `b1_instrument.py` / `b1_iso_moe.py` / `c1_coverage.py` / `c2_gate_instrument.py`:
  env/sentinel-gated diagnostic probes (removed from site-packages after use).

---

## Update 2026-06-16 — c256 prefill compute localized (Exp 35) + breakdown plan

Full detail in EXPERIMENT_LOG Exp 35. Artifacts in `/sgl-workspace/c256_analysis/`.

### What was found
The c256 residual gap is **prefill per-token COMPUTE**, measured directly with a
symmetric per-rank CUDA-event forward timer on BOTH engines (same config):
| | prefill us/tok | decode ms/step |
|---|---:|---:|
| ATOM single | 168.7 | 41.4 |
| SGLang (gatherv+A-fix) | 182.3 | 40.2 |
| SGL/ATOM | **+8.1% (prefill)** | 0.97 (decode parity) |

Supporting facts (all apple-to-apple, per-rank, same fast config):
- decode is parity; gap is entirely prefill (TTFT 4213 vs 3308 ms = +27%).
- prefill chunk composition near-identical (ATOM 12,103 vs SGLang 11,428 tok/step,
  both ~45% full 16384/rank) → **chunk fragmentation is NOT the cause** (the
  earlier "fragmentation" framing was a stale-data error, retracted).
- gatherv+reduce_scatterv verified firing on every prefill step (probe, no fallback).

### Fast repro (use this for all c256 work — 3.8 min vs ~13)
c256 ISL8192 OSL1024 ratio0.8 **np512 warm256**, single-stream. SGLang
≈24,900 tok/s, ATOM ≈26,700 (SGL/ATOM ≈93%), matches full np2048. Keep OSL=1024.
- SGLang launch: `SGLANG_DP_USE_GATHERV=1 SGL_EXTRA_ARGS="--chunked-prefill-size 131072" bash run_sgl_dsv4_aligned.sh`
- ATOM launch: `ATOM_DISABLE_SIDE_STREAMS=1 bash run_atom_dsv4_aligned.sh` (single-stream)
- SGLang client: `python3 -m sglang.bench_serving --backend sglang ...`
- ATOM client: `python3 bench_dsv4.py --backend vllm ...` (SSE choices shim)
- ATOM `enable_dp_attention` resets to dp8/tp1 (engine_core_mgr): 8 per-rank
  EngineCores, each prefill budget 16384 (NOT /dp). Its `Scheduled prefill batch:`
  log is PER-RANK. So both engines = 16384 tok/rank → genuinely apple-to-apple.

### NEXT: Exp 36 — break down the prefill +8% (per-module + per-kernel)
**Core tension (must resolve):** both engines call the SAME aiter kernels for MoE
GEMM / fp8 quant, so a per-kernel gap "should not" exist → the 8% must be in HOW
each engine invokes the shared kernels, OR in a kernel that is NOT shared.
Suspects, in priority order (all SGLang-internal + controlled microbench, no
unreliable full trace):
1. **MLA prefill attention (highest suspect)** — likely NOT the same kernel:
   SGLang uses `SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton` + aiter indexer;
   ATOM has its own prefill MLA. MoE GEMM is genuinely shared aiter, so a
   non-shared MLA path is the most plausible source of a real 8%.
2. **Expert-GEMM M / padding** — SGLang MoE runs on the gathered global buffer
   (M=131072). Verify the EXACT per-rank expert-GEMM M and whether SGLang pads
   more (DpPaddingMode rounding) → extra padding-token compute counted against
   the same real-token denominator (inflates us/real-tok even with identical kernel).
3. **fp8 quant granularity** — confirm both hit the same aiter per-token-group
   quant kernel with the same group size (SGLANG_OPT_FP8_WO_A_GEMM=false vs ATOM
   AITER_BF16_FP8_MOE_BOUND=0).
4. **gate/router redundancy** (C2) — SGLang gate GEMM on global M=131072 (8×
   redundant); measured ~7% of MoE in Exp 34, re-confirm under the per-module timer.
5. **host-side launch/layout** — extra contiguous()/copies or more launches/layer.

Method: per-MODULE CUDA-event timer (attn/gate/gather/moe1/moe2/quant/topk/scatter)
inside ONE SGLang prefill forward → ms breakdown of the 182 us/tok; then isolated
aiter microbench at the EXACT prefill shapes (per-rank M, fp8) — if the shared
kernel is equal in isolation, the gap is invocation/shape (suspects 2-5); if a
module is structurally different (suspect 1, MLA), that's the gap. Start with #1
and #2. Note ATOM per-module probe is unreliable (Exp 34), so the ATOM side of the
breakdown leans on the already-captured forward-wall total + the shared-kernel
microbench, not an ATOM per-module timer.

### c256_analysis/ scripts (this session)
- `bench_one.sh` — single fast-config bench point with --output-details.
- `analyze_phase1.py` — per-request prefill/decode split from bench jsonl.
- `analyze_phase2.py` — SGLang scheduler-log prefill/decode/running-batch parser.
- `ATOM_baseline.txt` — captured ATOM fast-config prefill stats.
- Diagnostic timers/probes were added to model_runner.py (both engines) +
  dp_attention.py + deepseek_v4.py and **removed after use** (git clean).

---

## Update 2026-06-16 (cont.) — Exp 36: prefill +8% narrowed to attn/MLA

Full detail in EXPERIMENT_LOG Exp 36.

### SGLang prefill forward breakdown (per-module CUDA-event, prefill-only, rank0)
| block | % of prefill fwd | shared with ATOM? |
|---|---:|---|
| moe (expert GEMM) | 35.6% | YES (aiter fused_moe) → equal |
| attn (MLA) | 34.2% | NO (engine-specific) → SUSPECT |
| gather (all_gatherv) | 18.2% | comm @ RCCL floor → equal |
| scatter (reduce_scatterv) | 10.1% | comm @ RCCL floor → equal |
| hc_norm | 1.9% | tiny |

### Neutral microbenches (engine-independent) ruled out the shared blocks
- aiter `fused_moe` isolated: M=131072 → 12.0 ms; M=16384 → 1.72 ms (8×tok→7×t,
  no large-M redundancy penalty). Both engines call this same kernel at M=131072
  → MoE GEMM equal. (`/workspace/b1_iso_moe.py`)
- RCCL all_gatherv floor (aligned 131072) = 5.25 ms; SGLang in-server gather =
  6.1 ms/layer → within ~16% of floor; both engines use the same primitive over
  the same bytes → comm equal. (`/workspace/moe_comm_microbench.py`)

### ATOM per-module probe = IMPOSSIBLE (don't retry)
ATOM decoder layer is torch.compile-wrapped (VllmBackend). Inserting cuda.Event
inside it crashes Dynamo: `cannot extract sympy expressions from <cuda.Event>`.
So ATOM's attn/moe internal split CANNOT be measured by in-model probe. Use
neutral microbench for shared blocks; the non-shared attn can only be bounded.

### Conclusion + Next (Exp 37)
8% prefill gap is, by elimination, in **attn/MLA** (the only large non-shared
block). Next: isolate SGLang's MLA-prefill kernel sequence (q/kv proj + aiter
indexer + flashmla/unified_kv core attn) at the c256 prefill shape and compare
to ATOM's MLA aiter ops run standalone. If the isolated MLA kernels are equal →
residual is SGLang attn host/glue overhead; if different → genuine kernel-path
diff. This is the last bucket; ROI is the ~8% prefill share of the c256 gap
(a few % of total tok/s), diminishing now that MoE+comm are ruled out.

## Update 2026-06-16 (cont.) — Exp 37: attn sub-split + wo_a einsum ruled out

Tried to localize within the attn block. Two outcomes:
- **wo_a out-projection is NOT the gap.** Our build has
  `SGLANG_OPT_FP8_WO_A_GEMM=false` and **no `deep_gemm`** (fp8 path crashes at
  init: ModuleNotFoundError), so wo_a runs as `torch.einsum` bf16 — but that
  einsum is already 1411 TFLOP/s (≈peak, isolated microbench), only ~5 us/tok of
  182. Rejected as the gap. (`/sgl-workspace/c256_analysis/wo_a_microbench.py`)
- **attn sub-block split is unreliable**: single-stream CUDA-event windows let
  one block's async kernel tail bleed into the next, so the in-attn split
  (out_proj 49 / qkv 31 / core 20) cannot be trusted. Only the attn-BLOCK total
  (34% of fwd) and the isolated microbenches are reliable.

**Bottom line for the c256 gap (final):** comm shipped/at-floor, MoE equal
(shared aiter), decode parity; residual = ~8% per-token PREFILL compute in the
engine-specific MLA path — could not be cleanly sub-attributed, and the one
concrete suspect (wo_a) is fast. Remaining in-attn suspects = core attention
(unified_kv_triton vs ATOM MLA, genuinely non-shared) and indexer/compressor.
Reliable next step = ISOLATED per-attn-sub-kernel timing with separate sync (not
chained single-stream events). The shipped gatherv+A-fix remains the main
defensible c256 win; further attn chasing is diminishing returns.

## Update 2026-06-16 (cont.) — Exp 38: the "8x gate GEMM" quantified (C2 corrected)

User asked to actually evaluate the redundant gate/router GEMM (C2 had dismissed
it). DSV4 dp-attn + TP-MoE: SGLang computes the gate on the GATHERED global
buffer (M=131072) — every rank computes router logits for all ranks' tokens,
uses 1/8. ATOM computes the router LOCALLY (M=16384). (NOTE: the *expert* GEMM is
global-buffer on BOTH engines — only the gate is SGLang-redundant.)

Isolated aiter gate-GEMM microbench (hidden7168→384, fp32, `gate_microbench.py`):
| | M | ms/layer |
|---|---:|---:|
| ATOM local | 16,384 | 0.127 |
| SGLang global | 131,072 | 0.644 |
- 5.1x (not 8x — small M is launch/mem-bound). Waste = 0.517 ms/layer × ~58
  layers = ~30 ms/prefill step = **~1.5% of the prefill forward ≈ ~+1% total
  tok/s at c256** if fixed.
- **CORRECTS C2**: C2's in-server "gate 0.068 ms/layer, 7.2%, negligible" UNDER-
  counted ~10x (single-stream async-tail artifact, same as Exp 37). True gate =
  0.644 ms/layer.

**Assessment:** real, clean, shared-kernel, low-risk lever — the most actionable
remaining c256 item, but only ~+1%. Fix = mirror ATOM (gate on LOCAL hidden
before the gather, carry router_logits through the gather). Touches the DSV4
gather payload adjacent to the shipped gatherv path → medium risk, modest ROI;
candidate for a separate follow-up PR. Consistent with the overall finding that
the large buckets (MoE GEMM, comm) are already equal/shipped.

**Implementation (built, debugged to CORRECT, but net-zero tput — Exp 38):**
env-gated `SGLANG_DP_GATE_LOCAL`. A collective probe (ENTER/EXIT per rank+layer)
found & fixed 2 bugs:
1. HANG: keyed gate-local on per-rank `is_extend` → in a mixed step only the
   prefilling rank took the wide (7552) all_gatherv while others took the normal
   7168 gather → RCCL width-mismatch hang. FIX: key ONLY on `_use_gatherv_pair`
   (synced across ranks), all-ranks-consistent.
2. ACCURACY: cast fp32 router logits→bf16 for the fused gather → top-6 routing
   changed → gsm8k 0.95→0.59. FIX: gather logits in fp32 via a separate small
   all_gatherv → gsm8k 0.9424 ≈ OFF 0.9469 (noise). Routing correct.
**c256 A/B result:** OFF 24,736 vs ON 24,725 tok/s = **−0.04% (no tput gain)**,
but **TTFT −10.6%** (4206→3759). The correctness-required fp32-logits gather
offsets the saved gate GEMM (net ~0), and c256 total tput is decode-bound so the
11% prefill-TTFT win doesn't move total. ⇒ gate-local is VIABLE + CORRECT but
NOT a c256 throughput win; only helps TTFT/prefill-heavy workloads. REVERTED
(git checkout, PR pristine). The "8x gate GEMM" is real but not a throughput
lever — closing the loop on this line: **no remaining c256 throughput lever
beyond the shipped gatherv+A-fix.** Scripts kept in c256_analysis/
(gate_microbench.py, launch_sgl_wofp8.sh, gsm8k_gatelocal/).
