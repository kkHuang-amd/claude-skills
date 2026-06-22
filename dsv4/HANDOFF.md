# DeepSeek-V4 DP performance — Handoff summary (updated 2026-06-17)

Read this first to pick up the investigation in a new session. Full detail is in
`EXPERIMENT_LOG.md` (Exp 1–56 + appendix); methods/scripts in `SKILL.md`.

> **Also 2026-06-21 (Exp 56) — C5: SE-local extended to DECODE → c512 1k/1k regression
> FIXED.** The Exp 55 −3.4% at c512 1k/1k was the TP1 replicated shared expert running
> at full dim on the gathered global decode batch (~dp_size x), because SE-local was
> prefill-only. Fix = broaden the gate (drop `is_extend()`, `_use_gatherv_pair` →
> `_use_tp_moe_gather`) so SE-local also runs in decode (dp_scatter), computing the
> shared expert on local tokens (`M_local*dim == M_global*dim/tp`). Per-token MLP →
> mathematically identical (gsm8k flex 0.9507 / strict 0.9515). c512 (vs gatherv-only
> base): **1k/1k 17,738→17,954 (+1.2%, was −3.4%); 8k/1k 33,828→35,403 (+4.7%)**; decode
> TPOT 48.08→46.54 / 79.10→73.69. Now ~91% of ATOM (was ~87%). TP1 weight-memory cost
> (~+0.5 GB/rank) unchanged — C5 only removes the decode compute penalty. Amended into
> SE-local commit `30fa179536` (dev clone). NOTE A1 (compressor glue) was investigated
> but parked: aiter `flydsl_hca_compress_attn` is usable + output layout compatible under
> unified_kv (plain bf16), but the STATE buffer format differs (SGLang flat ring
> `[size,2*head_dim]` vs aiter `[slots,STATE,head_dim]` f32) → large/risky bridge for
> ~0.5% prefill; deprioritized.

> **Also 2026-06-21 (Exp 55) — SE-local ported to sglang-upstream; "upstream slower"
> + "BLOCK_M" scares both DEBUNKED.** Ported shared-expert-local onto `sglang-upstream`
> (edits: `deepseek_v2.py` `skip_shared_experts`, `deepseek_v4.py` `_SHARED_EXPERT_LOCAL`,
> `cohere2_moe.py` @strict patch). gsm8k 0.9500. Upstream A/B (plan B: baseline=gatherv
> only → +TP1+se-local): **pure-prefill +6.7% (1k) / +6.1% (8k)**; c512 1k/1k **−3.4%**,
> 8k/1k +0.7% (TP1's replicated shared-expert adds ~8× decode FLOPs → eats the win in the
> decode-bound c512 1k case). **BUG FOUND**: a stale `__editable__` finder made every
> "dev clone" run silently use upstream (two finders coexisted; upstream won). Fixed by
> `rm`-ing the upstream `.pth`/`_finder.py`/`dist-info`. ALWAYS verify `import sglang;
> .__file__` + server-log `Editable project location` after `pip install -e`. After fix:
> dev clone **BLOCK_M=32 (35,101) ≈ BLOCK_M=8 (35,086)** → rope BLOCK_M has **zero c512
> e2e effect** (it's a sub-ms prefill op; c512 is decode-bound). upstream ≈ dev clone
> (both 34–35k); the 36,031 (6/18) vs ~35,100 (today) is ~2.5% cross-day variance, NOT a
> regression. Net: SE-local helps prefill (+6–7%); at c512 the bundled TP1 cost dilutes/
> reverses it; BLOCK_M=8 default kept (free prefill win).

> **Also 2026-06-18 (Exp 54) — CK-GEMM + batched/contig-RoPE are now DEFAULT-ON for
> DSV4 (no env needed).** Converted from env flags to module toggles (default OFF)
> flipped True in `DeepseekV4ForCausalLM.__init__`: `set_force_ck_w8a8(True)` +
> `set_batched_rope(True)`. Env `SGLANG_FORCE_CK_W8A8`/`SGLANG_ROPE_BATCHED` still
> override. Verified (no env): trace shows ck_tile GEMM + contig rope active (Triton
> GEMM / strided rope gone), gsm8k 0.9515. shared-expert-local stays env-gated
> (`SGLANG_DP_SHARED_EXPERT_LOCAL`, needs TP1 ~+0.5GB); chunk is a launch arg.

> **Also 2026-06-18 (Exp 53) — A2: output inverse-RoPE full-fuse.** The hot 337us/layer
> rope = attention-output inverse rope (deepseek_v4.py:1012, `fused_rope_inplace`); on
> HIP it fell back to strided `apply_rotary_emb_triton`. New
> `apply_rotary_emb_contig_kernel` (CONTIGUOUS load + reshape/flip, mirrors ATOM
> inverse_rope_gptj) → **142.7 us/call (was 337, now ≤ ATOM's 155)**. gsm8k 0.9447.
> Wired under SGLANG_ROPE_BATCHED for the 3D rope. Rope item now fully closed.

> **Also 2026-06-18 (Exp 52) — ALL3 c512 END-TO-END (OSL=1024): real gain, decode-
> diluted.** The 3 prefill levers (Exp 49–51) at full c512: 1k/1k 17,233→17,877
> (+3.7%, 86.9%→90% ATOM), 8k/1k 32,254→34,613 (+7.3%, 82.1%→88% ATOM). gsm8k 0.9477.
> The matched prefill DOES move c512 total tput (unlike gate-local's neutral, because
> these levers cut a far bigger prefill chunk), but diluted vs the +16% pure-prefill
> since c512 is decode-bound; bigger at 8k (prefill-heavier). Net 2026-06-18 prefill
> work: pure-prefill 84%→97–98%, c512 end-to-end 87/82%→90/88% of ATOM. Remaining c512
> gap is decode/scheduling + compressor-glue bubble, not the fixed kernels.

> **Also 2026-06-18 (Exp 51) — ALL 3 prefill levers stacked → 97–98% of ATOM.**
> `SGLANG_FORCE_CK_W8A8=1 SGLANG_ROPE_BATCHED=1 SGLANG_DP_SHARED_EXPERT_LOCAL=1
> SGLANG_SHARED_EXPERT_TP1=1`. Pure-prefill: 1k 48,291→55,944 (97% ATOM), 8k
> 47,013→54,645 (98% ATOM), **+16% vs base** (gains ~additive). gsm8k 0.9477. Per-layer
> trace: window gap ~7000us→**~600us**; GPU-active/layer base 44.81→ALL3 38.53 ms (now
> BELOW ATOM 41.54). Remaining tiny diffs: rope (SGL batched 332us vs ATOM fused
> inverse 155us) + compressor glue/bubble (SGL fill/rocprim vs ATOM hca_* fused).
> Reusable per-layer diff tool persisted: `useful-scripts/benchmarking/dsv4/layer_diff.py`
> (`cmp <sgl> <atom> <ratio>`). STILL TODO: c512 END-TO-END A/B (prefill matched, but
> c512 is decode-bound — does it move total tput?).

> **Also 2026-06-18 (Exp 50) — shared-expert-local PoC: +6–7% prefill, gsm8k OK.**
> SGLang computes the shared expert on the GATHERED global buffer (M≈131072); ATOM on
> LOCAL (M≈16384). NOTE: shared expert is TP-sharded → per-rank FLOPs are IDENTICAL
> (NOT 8× redundant like the gate); the win is (a) better GEMM shape + (b) 8× fewer
> rows through fp8-quant/elementwise. PoC (`SGLANG_DP_SHARED_EXPERT_LOCAL=1` +
> `SGLANG_SHARED_EXPERT_TP1=1`): compute replicated shared expert on local hidden
> before gather, skip in mlp, add to local slice after reduce_scatterv. Result
> pure-prefill **+6.0% (1k) / +7.0% (8k), 84%→88–90% of ATOM**, gsm8k 0.9393. Caveats:
> needs TP1 shared (~+0.5GB/rank); c512 END-TO-END (decode-bound) unverified (Exp 38
> gate-local was neutral). Edits: deepseek_v2.py + deepseek_v4.py (env-gated, default OFF).

> **Also 2026-06-18 (Exp 49) — FIXED most of the prefill kernel gap (+8.8%).** The
> Exp-48 ~8% raw-kernel gap is from two impl choices: (1) SGLang uses the **Triton**
> w8a8-block FP8 GEMM for MLA q/kv/o projections (hardcoded
> `use_aiter_triton_gemm_w8a8_tuned_gfx950` list) while ATOM uses the **CK
> bpreshuffle** GEMM (and calls Triton "mostly slower"); (2) SGLang RoPE is one
> program/token vs ATOM batching 32 tokens/program. FIX (both env-gated, default OFF):
> `SGLANG_FORCE_CK_W8A8=1` (→ CK GEMM) + `SGLANG_ROPE_BATCHED=1`. Pure-prefill
> **+8.7% (1k) / +8.9% (8k), 84%→91% of ATOM**, gsm8k still 0.9469. Recovers ~all the
> raw-kernel share; remaining ~9% to ATOM is the host/bubble overhead (Exp 48).
> Edits in `/sgl-workspace/sglang` fp8_utils.py + deepseek_v4_rope.py.

> **Also 2026-06-18 (Exp 48) — prefill TRACE splits the ~20% into kernel vs
> overhead.** Profiler traces (both single-stream, pure-prefill ISL8192 OSL1).
> Reliable metric = GPU-active UNION per attn-layer-step (per-kernel `dur` is
> unreliable, Exp 19 — ATOM durs inflated). Result: per-step wall 2.79s (SGL) vs
> 2.34s (ATOM) = 1.19× DECOMPOSES into **raw GPU kernel ×1.08 (8%) × host
> overhead/bubble ×1.10 (10%)**. SGLang per-step bubble 12% vs ATOM 3%. So the ~20%
> prefill gap ≈ HALF raw kernel + HALF launch/glue overhead — NOT purely kernel, NOT
> purely scheduler. Shared `pa_prefill` (MLA core attn) is equal; the 8% likely in
> MLA proj GEMMs/glue but needs isolated kernel microbench to pin (trace dur can't).
> CLEANUP: `rocm-smi --showpids` → kill the listed VRAM-holding PIDs (robust; pkill
> misses reparented DP children).

> **Also 2026-06-18 (Exp 47) — c512 gap is NOT purely scheduler: SGLang prefill
> compute is ~20% SLOWER.** Pure-prefill test (OSL=1, decode removed, same client,
> chunk 16384/rank): ATOM prefill tok/s beats SGLang by +19.7% (1k) / +19.1% (8k).
> So the c512 gap = (1) prefill COMPUTE ~20% slower (real kernel/engine gap, likely
> MLA path per Exp 36) + (2) scheduler/queueing (Exp 42/44/45). SGLang decode (TPOT)
> is still fine/better; high TTFT comes from slower prefill compute AND queueing.
> Next: isolated MLA-prefill kernel microbench to pin the raw-kernel share.

> **Also 2026-06-18 (Exp 46) — ATOM's speed is NOT a recent change.** Compared two
> ATOM commits at c512: OLD `914d50323` (6/8, =/sgl-workspace/ATOM-previous) vs NEW
> `bcd38f67` (6/17, =/sgl-workspace/ATOM, the version Exp 39–45 used). **Identical
> perf (all metrics ±2%, noise)**: 1k 19,995 vs 19,828, 8k 39,721 vs 39,291. ⇒ no
> scheduler/kernel change in that window; ATOM was already this fast at 6/8. ATOM's
> edge is inherent design (adaptive prefill + prefill-delayer fairness), not a recent
> patch → diffing these two commits won't locate it. NOTE: install is now OLD
> 914d50323 (no side-stream flag); restore with `pip install /sgl-workspace/ATOM/`.

> **Also 2026-06-18 (Exp 45) — chunk-size sweep validated.** 8192/rank stability
> confirmed (1k ×3, std 0.06%, +5% REAL). **8192/rank is the universal c512 sweet
> spot**: 1k 86.9→91.3%, 8k 82.1→85.2% of ATOM. 1k 8192→4096 plateaus; 8k 8192→4096
> REGRESSES (4096<8192 SPLITS the request → effect A bites) — confirms the
> two-effects model + the "don't split a request" boundary. TPOT rises monotonically
> as chunk shrinks (decode interrupted more — intuitive objection confirmed); tput =
> (TTFT gain − TPOT cost), peaks at 8192. chunk tuning recovers ~1/3 of the gap;
> ATOM still leads on TTFT (scheduling/prefill-fairness), residual is NOT chunk-size.

> **Also 2026-06-18 (Exp 43–44).** (43) **Client validation**: SGLang's own bench
> client reports ~8% LOWER tput than the ATOM client on the SAME server at c512
> (15,751 vs 17,195) — client effect grows with conc (was ~3% at c128/256). All our
> SGLang-vs-ATOM numbers use ATOM client for BOTH, so gaps are pure engine. (44)
> **c512 levers**: `schedule_conservativeness`↑ = neutral; **chunked-prefill
> 16384→8192/rank = +5%** (86.9%→91.1% of ATOM), single run. CORRECTED mechanism:
> smaller chunk does NOT smooth decode (TPOT actually +4% WORSE); the win is on the
> PREFILL/queue side (mean TTFT −33%) and tput∝1/mean_E2E. Two opposing chunk
> effects (prefill-efficiency vs queue-fairness) → optimal chunk is
> workload×conc-dependent (8k/c256 wants big, 1k/c512 wants smaller; reconciles the
> old "2048 bad" finding). CORRECTION to Exp 42: vs-ATOM c512 gap is TTFT/queueing,
> NOT decode occupancy (SGLang TPOT is actually < ATOM). See bottom Exp 43/44.

> **Also 2026-06-17 (Exp 42) — c512 gap ROOT-CAUSED + 2 levers tried.** The big
> c512 deficit is **100% prefill/TTFT** (decode TPOT is actually FASTER on SGLang);
> signature is TTFT variance (SGL std 6.3×, p99 5.6× ATOM). Two-sided scheduler-log
> analysis: **SGLang can't keep the decode batch full at c512** (median 53/64,
> drains to single digits) because it injects prefill as RIGID full-16384 chunks
> that stall all decode; **ATOM holds 64/64** via ADAPTIVE prefill granularity
> (small 1–4-req batches) + a well-tuned prefill-delayer (delay 2.69%). ~17% decode
> under-occupancy ≈ the 15% tput gap. Levers: `swa-full-tokens-ratio` 0.15/0.2/0.25
> = NO tput effect (KV-pool knob, not scheduling); **`--enable-mixed-chunk` REJECTED
> — made it WORSE** (86.9%→82.3% of ATOM, gsm8k fine 0.94). Untried/on-target:
> raise `schedule_conservativeness`, smaller chunked-prefill-size. See bottom
> "Update 2026-06-17 (cont.) — c512 root cause".

> **Also 2026-06-17 (Exp 41) — SGLang c512 deficit confirmed STABLE (3 repeats).**
> Re-ran SGLang c512 ×3 per workload (same Exp 39 config). Variance tiny (std
> 0.06–0.48%): 1k/1k mean 17,233 (86.9% of ATOM), 8k/1k mean 32,180 (81.9%). The
> big c512 gap is REAL/reproducible, not noise — driven by prefill queueing (8k/1k
> c512 TTFT ~55.6s vs ATOM ~38.6s). c64–c256 still ties/wins (Exp 39).

> **Also 2026-06-17 (Exp 40) — re-added `ATOM_DISABLE_SIDE_STREAMS` flag (updated
> ATOM had dropped it) + ATOM single vs multi-stream sweep.** Flag is a single
> master switch (envs.py + deepseek_v4.py: allocate alt_stream/indexer_stream only
> when `not ATOM_DISABLE_SIDE_STREAMS`; None ⇒ both dual-stream MoE AND async
> Compressor overlap fall back to inline). `=0` default = multi-stream, `=1` =
> single-stream; runtime-verified via per-rank log. **Result: multi-stream wins
> everywhere by a small, concurrency-shrinking margin (SS = 93–94% of MS at c64,
> 97.6–98.9% at c512); side-streams mainly help DECODE/low-conc.** CAVEAT: edited
> in the installed site-package, not a git repo — lost on container rebuild. See
> bottom "Update 2026-06-17 (cont.) — side-stream flag".

> **Latest (2026-06-17) — full re-benchmark on UPDATED code bases (Exp 39).**
> Fresh container (old `sglang-upstream` clone gone; ATOM reinstalled). Re-ran
> tp8+dp8 over a wider grid (ISL∈{1024,8192}, conc∈{64,128,256,512}) with BOTH
> engines driven by the SAME ATOM-native client (client variance = 0), multi-stream,
> ratio1.0, 16384 prefill tok/rank. SGLang: gatherv ON, **ROCM700A=0**,
> `--chunked-prefill-size 131072` (auto ÷dp8 = 16384/rank). PR #28216 (gatherv+A-fix)
> is now in main. **Result: SGLang ties/wins at c64–c256 (102–110% of ATOM, TPOT
> better everywhere) but regresses at c512 (82–87%, TTFT much worse) — high-conc
> prefill↔decode interference remains SGLang's weak point.** ATOM gsm8k 0.9500
> (correct even without ATOM_USE_TRITON_MOE=1). Numbers + table in the bottom
> section "Update 2026-06-17" and EXPERIMENT_LOG Exp 39. One repo fix needed to
> launch sglang: `cohere2_moe.py` `@strict`→no-op (hf_hub≥1.x import crash, SKILL §2a).

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

---

## Update 2026-06-17 — full re-benchmark on updated code bases (Exp 39)

Fresh container. ATOM reinstalled (`atom/__init__.py` ts 04:05). The old
`/sgl-workspace/sglang-upstream` is gone; the editable `/sgl-workspace/sglang`
(main @ 66ac385f52) already contains PR #28216's gatherv + A-fix
(`SGLANG_DP_USE_GATHERV`, `reduce_scatterv`, `get_dp_global_num_tokens()`).

### Config (apple-to-apple, SAME client both engines)
tp8+dp8, **multi-stream**, FP8 KV, page/block256, mem0.90, max-running512,
**16384 prefill tok/rank**, prefill-delayer ON, ratio1.0, num_prompts=conc*8,
warmups=conc*2. Client = ATOM `benchmark_serving --backend vllm` for BOTH
(client variance = 0). Grid ISL∈{1024,8192} × conc∈{64,128,256,512}, OSL=1024.
- ATOM launch: `DP_MODE=tp8dp8 bash run_atom_dsv4_aligned.sh`
- SGLang launch: `SGLANG_USE_ROCM700A=0 SGLANG_DP_USE_GATHERV=1 DP_MODE=tp8dp8 SGL_EXTRA_ARGS="--chunked-prefill-size 131072" bash run_sgl_dsv4_aligned.sh`
  - main auto-does `chunked_prefill_size //= dp_size` (server_args.py:3537) under
    DP attn, so 131072 → 16384/rank = ATOM-aligned (default 16384 → 2048/rank, NOT
    aligned). Needed `cohere2_moe.py` `@strict`→no-op to launch (SKILL §2a).
- Bench client: `RATIO=1.0 WORKLOADS="1024:1024 8192:1024" CONCS="64 128 256 512" bash sweep_dsv4_atom_client.sh`

### ATOM gsm8k (updated build): flexible 0.9500 / strict 0.9492 ✓
Correct even WITHOUT `ATOM_USE_TRITON_MOE=1` (the SKILL §1 wrong-MoE caveat did
not trigger on this build).

### SGLang (gatherv ON, ROCM700A=0) vs ATOM — total tok/s
| ISL | conc | SGL | ATOM | SGL/ATOM | SGL TTFT | ATOM TTFT | SGL TPOT | ATOM TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 64  | 4,642  | 4,211  | **110.2%** | 1,458  | 1,307  | 26.1 | 29.2 |
| 1024 | 128 | 7,870  | 7,330  | **107.4%** | 2,185  | 1,725  | 30.3 | 33.1 |
| 1024 | 256 | 12,974 | 12,418 | **104.5%** | 4,016  | 3,564  | 35.3 | 37.1 |
| 1024 | 512 | 17,185 | 19,828 | 86.7%      | 7,120  | 5,489  | 44.0 | 45.4 |
| 8192 | 64  | 15,295 | 14,530 | **105.3%** | 6,525  | 5,929  | 31.2 | 33.3 |
| 8192 | 128 | 22,102 | 21,609 | **102.3%** | 12,055 | 9,545  | 40.2 | 43.0 |
| 8192 | 256 | 29,621 | 30,880 | 95.9%      | 23,124 | 18,854 | 55.2 | 55.4 |
| 8192 | 512 | 32,254 | 39,291 | 82.1%      | 55,368 | 38,630 | 72.7 | 80.0 |

### Takeaways
- **c64–c256: SGLang ties/wins** (102–110%); TPOT better at every point.
- **c512: SGLang regresses** (82–87%, TTFT much worse, e.g. 8192 c512 55.4s vs
  38.6s) — high-conc prefill↔decode interference, the known weak spot, more
  extreme at c512 than the usual c256.
- 8192 c256 = 95.9% (slightly better than historical ~93%).
- ATOM updated build vs 06-09: 8192 c128/c256 flat within noise (+0.4% / −0.2%).

### Artifacts
- `/workspace/bench_results_dsv4_atom_0617/`, `/workspace/bench_results_dsv4_sgl_0617/`
  (8 JSON + summary each); logs `/workspace/{atom,sgl}_{server,sweep_0617}.log`,
  `/workspace/gsm8k_eval.log`. Both servers cleaned (VRAM ~0.3 GB/GPU).
  Cleanup note (ROBUST METHOD): no `lsof` here, and `pkill -f` often MISSES the
  reparented DP `multiprocessing-fork` EngineCore children (they keep ~290 GB/GPU
  resident). Best practice: `rocm-smi --showpids` lists the exact PIDs holding VRAM
  (4th col = VRAM bytes) — `kill -9` those PIDs directly, then confirm with
  `rocm-smi --showmeminfo vram` (~0.3 GB/GPU = clean). Don't rely on pattern-killing.

---

## Update 2026-06-17 (cont.) — side-stream flag re-add + ATOM SS vs MS (Exp 40)

Updated ATOM had DROPPED our `ATOM_DISABLE_SIDE_STREAMS` flag (not in the new
centralized `atom/utils/envs.py`, not read anywhere). Re-added as a single master
switch. Full detail in EXPERIMENT_LOG Exp 40.

### Side-stream architecture (updated ATOM, deepseek_v4.py)
Two independent mechanisms, BOTH gated on `alt_stream is not None`:
- **Dual-stream MoE** (shared // routed experts on alt_stream): also needs
  `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD>0` (default 1024); per-call token-gated so
  prefill skips it → mainly DECODE.
- **Async Compressor/indexer overlap** (alt_stream + indexer_stream): also gated by
  `fc.in_hipgraph` (CUDAGraph only). No env toggle existed in the new code.
Streams allocated once at model __init__ (~deepseek_v4.py:2668).

### The flag (re-added)
- `atom/utils/envs.py`: `ATOM_DISABLE_SIDE_STREAMS` (default "0").
- `atom/models/deepseek_v4.py` (~2668): allocate alt_stream/indexer_stream only when
  `torch.cuda.is_available() and not envs.ATOM_DISABLE_SIDE_STREAMS`; else None →
  both mechanisms inline. + info log of resolved state.
- `=0`(default)=multi-stream, `=1`=single-stream. Runtime-verified: 8 ranks log
  `DSV4 side-streams DISABLED (single-stream) ... alt_stream=False`.
- CAVEAT: edited in installed site-package (`/opt/venv/.../atom/`), NOT git — lost
  on container rebuild; commit to ATOM source to persist.

### ATOM single-stream (SS) vs multi-stream (MS) — tp8dp8, ATOM client, ratio1.0
| workload | conc | MS tok/s | SS tok/s | SS/MS | MS TPOT | SS TPOT |
|---|---:|---:|---:|---:|---:|---:|
| 1k/1k | 64  | 4,211  | 3,945  | 93.7% | 29.18 | 31.23 |
| 1k/1k | 128 | 7,330  | 6,947  | 94.8% | 33.14 | 34.51 |
| 1k/1k | 256 | 12,418 | 12,078 | 97.3% | 37.12 | 38.90 |
| 1k/1k | 512 | 19,828 | 19,348 | 97.6% | 45.41 | 46.52 |
| 8k/1k | 64  | 14,530 | 13,518 | 93.0% | 33.35 | 36.41 |
| 8k/1k | 128 | 21,609 | 20,982 | 97.1% | 43.02 | 44.69 |
| 8k/1k | 256 | 30,880 | 30,146 | 97.6% | 55.38 | 57.48 |
| 8k/1k | 512 | 39,291 | 38,867 | 98.9% | 79.96 | 80.79 |

Multi-stream wins everywhere, margin shrinks with concurrency (biggest at c64,
near-even at c512). Side-streams mainly help DECODE/low-conc (~+1–7%).
Artifacts: `/workspace/bench_results_dsv4_atom_ss_0617/` (SS) vs
`.../bench_results_dsv4_atom_0617/` (MS); logs `/workspace/atom_ss_*.log`.

---

## Update 2026-06-17 (cont.) — SGLang c512 stability, 3 repeats (Exp 41)

Checked whether SGLang's large c512 deficit vs ATOM (Exp 39) is stable or noise.
Re-ran SGLang c512 ×3 per workload (gatherv ON, ROCM700A=0, tp8dp8, ATOM client,
ratio1.0, np4096/warm1024).

| workload | run1 | run2 | run3 | mean | std | vs ATOM (Exp 39) |
|---|---:|---:|---:|---:|---:|---:|
| 1k/1k c512 | 17,195 | 17,157 | 17,348 | **17,233** | 83 (0.48%) | 86.9% |
| 8k/1k c512 | 32,155 | 32,186 | 32,200 | **32,180** | 19 (0.06%) | 81.9% |

**Deficit is REAL and reproducible** (std 0.06–0.48%, not noise). Matches the
single-run Exp 39 ratios (86.7% / 82.1%). High-conc prefill↔decode interference is
the driver — 8k/1k c512 TTFT ≈ 55.6s vs ATOM ~38.6s (heavy prefill queueing).
c512 is the worst point; c64–c256 SGLang ties/wins. Artifacts:
`/workspace/bench_results_dsv4_sgl_c512_run{1,2,3}/`, log
`/workspace/sgl_c512_repeat.log`.

---

## Update 2026-06-17 (cont.) — c512 gap ROOT CAUSE + levers (Exp 42)

Full detail in EXPERIMENT_LOG Exp 42. All 1k/1k c512.

### The gap is 100% prefill/TTFT (decode is fine)
| | SGL | ATOM |
|---|---:|---:|
| total tok/s | 17,195 | 19,828 (+15.3%) |
| Med TPOT (decode) | **43.85** | 45.41 (SGL faster) |
| Med / mean / p99 / std TTFT | 7,355 / 13,404 / 53,459 / 16,046 | 5,489 / 5,924 / 9,538 / 2,532 |
Decode is parity-or-better; the gap is prefill, and the signature is TTFT VARIANCE
(SGL std 6.3×, p99 5.6× ATOM). Wall-duration ratio == tput gap.

### Root cause (two-sided scheduler logs, ~64 reqs/rank)
- **SGLang**: decode #running-req median **53/64** (drains to single digits);
  prefill = rigid FULL 16384-tok chunks; non-mixed steps ⇒ each prefill stalls ALL
  decode ⇒ decode occupancy collapses + TTFT bursts.
- **ATOM**: decode output **64/64** stable-full; ADAPTIVE prefill granularity (full
  16384 + many small 1–4-req batches); prefill-delayer delay_rate 2.69%.
- ~17% decode under-occupancy ≈ 15% tput gap. ATOM trades slightly slower decode
  for stable-full occupancy + smooth prefill. Only c512 breaks because high conc =
  more in-flight decode to disrupt; c64–c256 ties/wins.

### Levers tried
- **swa-full-tokens-ratio 0.15/0.2/0.25**: NO tput effect (17,233/17,173/17,128,
  within 1.1% band). KV-pool knob, not scheduling. TTFT best at 0.2 (~−7%) but not
  monotonic. Kept 0.15.
- **`--enable-mixed-chunk`: REJECTED — WORSE.** gsm8k fine (0.94). total
  17,233→16,321 (−5.3%), 86.9%→**82.3%** of ATOM; TTFT mean +20%, std +14%. Mixing
  prefill into decode steps enlarges per-step batch → higher TPOT + worse TTFT
  variance (gatherv/MoE padding). Baseline (non-mixed) stays the better c512 config.

### Untried (more on-target than mixed-chunk)
- Raise `schedule_conservativeness` (DP attn auto ×0.3 → 0.3) to protect decode
  occupancy. - Smaller `chunked-prefill-size` (e.g. 8192/rank) to mimic ATOM's
  small-batch prefill injection.

Artifacts: `/workspace/bench_results_dsv4_sgl_{swa02,swa025,mc,sched}/`,
`/workspace/bench_results_dsv4_atom_sched/`, scheduler logs
`/workspace/{sgl,atom}_server_sched.log`, `/workspace/gsm8k_mc.log`.

---

## Update 2026-06-18 — client validation (Exp 43)

Same SGLang server (baseline, gatherv ON, ROCM700A=0, chunk16384/rank), same
`/v1/completions` endpoint + params (1k/1k c512, np4096/warm1024/ratio1.0); only the
client differs.
| metric | ATOM client | SGLang client (sglang-oai) | diff |
|---|---:|---:|---:|
| total tok/s | 17,195 | 15,751 | **−8.4%** |
| Med / Mean TTFT ms | 7,355 / 13,404 | 8,261 / 15,612 | +12% / +17% |
| Med TPOT ms | 43.85 | 46.36 | +5.7% |
SGLang's own client reports SYSTEMATICALLY LOWER tput on the identical server;
client effect ~8% at c512 (> the ~3% SKILL §4 saw at c128/256) → grows with conc.
⇒ all our SGLang-vs-ATOM numbers use the ATOM client for BOTH engines, so the
client cancels and the reported gaps are pure-engine. (If one used sgl-client for
SGLang + atom-client for ATOM, SGLang would be under-reported ~8% → c512 ~79% not
~87%.) Artifacts: `/workspace/bench_results_dsv4_sglclient/`.

---

## Update 2026-06-18 — c512 levers + CORRECTED chunk mechanism (Exp 44)

1k/1k c512, baseline = chunk 16384/rank, conservativeness eff 0.3.

### Levers
- **schedule_conservativeness eff 0.3→1.0**: NEUTRAL (17,233→17,009, −1.3% noise).
  It guards against retracts; this case never retracts → no effect. Rejected.
- **chunked-prefill 16384→8192/rank**: **+5% (17,195→18,064, 86.9%→91.1% of ATOM)**,
  single run.

### CORRECTED mechanism (Exp 42's "smaller chunk → smoother decode" was WRONG)
| | base 16384 | 8192 |
|---|---:|---:|
| total tok/s | 17,195 | 18,064 (+5.1%) |
| Med TPOT | 43.85 | 45.62 (**WORSE +4%**) |
| Mean TTFT | 13,404 | 9,025 (**−33%**) |
| std TTFT | 16,046 | 11,543 (−28%) |
| p99 TTFT | 53,459 | 53,985 (~same) |
| Mean E2E | 57,797 | 55,423 (−4.1%) |
- Smaller chunk HURTS decode (TPOT/ITL up — more prefill steps interrupt decode
  more; the intuitive objection is right). The WIN is on the PREFILL/queue side
  (mean TTFT −33%). tput is closed-loop ∝ 1/mean_E2E; ΔE2E = TTFT −4.4s + decode
  +1.8s = −2.6s ≈ measured −2.4s. So +5% = (queue win) − (decode cost).
- Why smaller chunk shortens TTFT: TTFT≈queue-wait; what matters is how OFTEN a
  prefill step fires. Big chunk → infrequent big prefill waves (delayer batches) →
  reqs wait a whole decode interval. Small chunk → cheaper/more-frequent prefill →
  steadier admission → lower mean/var TTFT.
- NOT clean: p99 TTFT unchanged; TPOT is a real tradeoff; single run (repeat ×3).

### Non-monotonic chunk size (reconciles old "2048/rank worse than 16384")
Two opposing effects: (A) prefill efficiency — smaller=more steps+low-M GEMM → hurts
(dominates at 8k input where chunk also splits a single prefill); (B) queue fairness
— smaller=more frequent prefill → lower TTFT var → helps (dominates at 1k input,
1024<chunk so no intra-req split, chunk only sets reqs/step). ⇒ optimal chunk is
workload×conc dependent: 8k/c256 wants big, 1k/c512 wants smaller.

### CORRECTION to Exp 42 root cause
vs-ATOM c512 gap is **TTFT/prefill-admission-queueing, NOT decode occupancy**:
SGLang TPOT (43.85) is actually BETTER than ATOM (45.41), so decode isn't the
problem; the "decode 53/64" Exp-42 reading was a sampling artifact, not the driver.
ATOM wins by low/uniform TTFT (mean 5.9s std 2.5k vs SGL 13.4s std 16k).

### Next
Repeat 8192 ×3; re-capture scheduler logs (prefill-step frequency + queue depth) at
8192 vs 16384 to confirm causality; try 4096/rank; validate best chunk on 8k/c512
(gap larger) and c128/c256 (no regression). Artifacts:
`/workspace/bench_results_dsv4_sgl_{cons,cps}/`, logs `/workspace/sgl_{cons,cps}_*.log`.

---

## Update 2026-06-18 — chunk-size sweep validated (Exp 45)

Verified the chunk lever: 8192 stability ×3, 4096/rank, and 8k workload.

### 1k/1k c512 (1024<chunk → no intra-req split; chunk = reqs/step)
| chunk/rank | total tok/s | std | vs ATOM | Med TPOT | Mean TTFT |
|---|---:|---:|---:|---:|---:|
| 16384 (base ×3) | 17,233 | 83 | 86.9% | 43.84 | 13,260 |
| **8192 (×3)** | **18,106** | **11** | **91.3%** | 45.46 | 8,874 |
| 4096 | 18,099 | — | 91.3% | 46.88 | 7,123 |
| ATOM | 19,828 | — | 100% | 45.41 | 5,924 |

### 8k/1k c512 (chunk<8192 SPLITS the 8192-tok request)
| chunk/rank | behavior | total tok/s | vs ATOM | Med TPOT | Mean TTFT |
|---|---|---:|---:|---:|---:|
| 16384 (base) | 2 reqs/step | 32,254 | 82.1% | 72.70 | 65,771 |
| **8192** | 1 req/step, no split | **33,475** | **85.2%** | 82.61 | 53,521 |
| 4096 | SPLITS req | 33,268 | 84.7% | 84.89 | 48,299 |
| ATOM | — | 39,291 | 100% | 79.96 | 38,745 |

### Confirmed
- 8192 win is REAL/reproducible (×3 std 0.06%). **8192/rank = universal c512 sweet
  spot** (1k 86.9→91.3%, 8k 82.1→85.2%).
- 1k 8192→4096 PLATEAUS; 8k 8192→4096 REGRESSES (splitting a request triggers effect
  A) → two-effects model + "don't split a request" boundary both confirmed.
- TPOT rises monotonically as chunk shrinks (decode interrupted more — the intuitive
  objection holds); tput = TTFT-gain − TPOT-cost, peaks at 8192.
- chunk tuning recovers ~1/3 of the gap; ATOM still leads on TTFT (prefill-fairness
  scheduling) → residual is NOT chunk-size. Artifacts:
  `/workspace/bench_results_dsv4_{A_8192_1k_run{1,2,3},A_8192_8k,B_4096_1k,B_4096_8k}/`,
  logs `/workspace/server{A,B}_bench.log`.

---

## Update 2026-06-18 — old vs new ATOM: speed is NOT a recent change (Exp 46)

Q: is ATOM's c512 edge a recent scheduler update or faster prefill kernels? Compared
OLD `914d50323` (6/8, =ATOM-previous) vs NEW `bcd38f67` (6/17, =/sgl-workspace/ATOM,
the version Exp 39–45 used). c512, tp8dp8, multi-stream, ATOM client.
| | OLD 914d50323 | NEW bcd38f67 | NEW vs OLD |
|---|---:|---:|---:|
| 1k/1k total tok/s | 19,995 | 19,828 | −0.8% |
| 8k/1k total tok/s | 39,721 | 39,291 | −1.1% |
| TTFT / TPOT (both wl) | — | — | all ±2% (noise) |
**Identical perf** → ATOM did NOT change (scheduler or kernel) between 6/8 and 6/17;
it was already this fast at 914d50323. ATOM's edge is inherent design (adaptive
prefill injection + prefill-delayer fairness), NOT a recent patch. Diffing these two
commits won't locate the cause (no perf delta); would need a much older ATOM.
NOTE: install is now OLD 914d50323 (side-stream flag gone — was on NEW's site-pkg).
Restore: `pip install /sgl-workspace/ATOM/` + re-apply flag if needed (perf same
either way). Artifacts: `/workspace/bench_results_dsv4_atomOLD/`,
`/workspace/atomOLD_sweep.log`; NEW = `/workspace/bench_results_dsv4_atom_0617/`.
(Restored to NEW bcd38f67 + flag re-applied after this exp.)

---

## Update 2026-06-18 — pure-prefill compute: SGLang ~20% slower (Exp 47)

Q: is SGLang's per-STEP prefill slower (compute), or is the c512 gap purely
scheduler? Isolated prefill COMPUTE via a PURE-PREFILL run (OSL=1 → ~no decode → no
prefill↔decode interference), same client, chunk 16384/rank, conc512.
| ISL | SGL input tok/s | ATOM input tok/s | ATOM/SGL |
|---|---:|---:|---:|
| 1024 | 48,291 | 57,821 | **+19.7%** |
| 8192 | 47,013 | 55,999 | **+19.1%** |
**NOT purely scheduler.** With decode removed, ATOM still prefills ~20% faster ⇒
SGLang's per-step prefill compute is genuinely ~20% slower (real kernel/engine gap,
direction matches Exp 36 = engine-specific MLA path; MoE+comm are shared/equal).
So c512 gap = (1) prefill compute ~20% slower + (2) scheduler/queueing (Exp
42/44/45). SGLang decode (TPOT) is fine/better; high TTFT = slower prefill compute +
queueing. Caveat: saturated input_tps also includes prefill batching efficiency, not
only raw kernel → next: isolated MLA-prefill kernel microbench at matched shapes.
Artifacts: `/workspace/bench_pp_{sgl,atom}/`, logs `/workspace/pp_{sgl,atom}_sweep.log`.

---

## Update 2026-06-18 — prefill TRACE: kernel vs overhead split (Exp 48)

Split the Exp 47 ~20% prefill gap via torch-profiler traces (both single-stream,
pure-prefill ISL8192 OSL1, rank0). Reliable metric = GPU-active UNION per
attn-layer-step (per-kernel `dur` UNRELIABLE per Exp 19 — ATOM durs inflated; only
the union-of-kernel-intervals is trustworthy). Normalized by `pa_prefill` count
(same aiter kernel + same model → L-independent).
| | GPU-active/step | wall/step | per-step bubble |
|---|---:|---:|---:|
| SGLang | 2.46 s | 2.79 s | 12% |
| ATOM | 2.28 s | 2.34 s | 3% |
**Decomposition:** wall ratio 2.79/2.34 = 1.19 (=Exp 47 tput gap) = raw GPU kernel
×1.08 (8%) × host overhead/bubble ×1.10 (10%). So ~20% prefill gap ≈ HALF raw kernel
(SGLang kernels +8% GPU-active) + HALF launch/glue overhead (SGLang 12% per-step
bubble vs ATOM 3%). NOT purely kernel, NOT purely scheduler.
- Shared `pa_prefill` (MLA core attn, ~179 ms/step SGL) is the same kernel both → equal.
  The 8% likely in MLA projection GEMMs / glue (Exp 36 direction) but per-kernel
  attribution is blocked by ATOM dur inflation → needs isolated kernel microbench.
- SGLang's 12% per-step bubble (launch gaps / host syncs) is a concrete addressable
  target (launch batching / prefill CUDA graph). Artifacts:
  `/workspace/sgl_prof/1781761067*DP-0*`, `/workspace/atom_prof/dp0_tp0/*`.

---

## Update 2026-06-18 — FIXED the prefill kernel gap (+8.8%) (Exp 49)

Located the two differing kernels (Exp 48) and made SGLang match ATOM:
- **w8a8-block FP8 GEMM (MLA q/kv/o proj, ~28%/step)**: SGLang used the **Triton**
  `gemm_a8w8_blockscale` for these shapes (hardcoded `use_aiter_triton_gemm_w8a8_tuned_gfx950`
  list, fp8_utils.py); ATOM uses **CK bpreshuffle** (`gemm_a8w8_blockscale_bpreshuffle`,
  linear.py, comment: "Triton FP8 blockscale mostly slower than CK"). FIX:
  `SGLANG_FORCE_CK_W8A8=1` → return False → SGLang uses CK.
- **RoPE (compressor fallback, ~2%/step)**: SGLang `apply_rotary_emb_triton` = 1
  program/token; ATOM `_inverse_rope_gptj` batches 32 tok/program. FIX:
  `SGLANG_ROPE_BATCHED=1` → added `apply_rotary_emb_triton_kernel_batched` (BLOCK_M=32).

### Result (pure-prefill OSL=1, both flags on)
| ISL | SGL base | +CK+ROPE | ATOM | gain | vs ATOM |
|---|---:|---:|---:|---:|---:|
| 1024 | 48,291 | 52,506 | 57,821 | +8.7% | 84%→91% |
| 8192 | 47,013 | 51,205 | 55,999 | +8.9% | 84%→91% |
gsm8k 0.9469 (correct). Recovers ~all the Exp-48 raw-kernel gap; remaining ~9% to
ATOM = host/bubble overhead. GEMM is the dominant contributor (28% vs 2%); combined
test (not yet split). Both flags DEFAULT OFF; edits in `/sgl-workspace/sglang`
fp8_utils.py + deepseek_v4_rope.py (lost on container rebuild).
### Trace confirmation (ISL8192 pure-prefill, fixed build)
GPU-active per attn-layer-step (union method): SGL base 44.81 ms → **+CK+ROPE 42.58
ms** → ATOM 41.54 ms. Raw-kernel gap to ATOM **+7.9% → +2.5%**. Kernel-level: Triton
`_gemm_a8w8_blockscale` GONE, CK `QuantGemmKernel` now used (= ATOM), batched rope
present. busy% 89→92% (CK also cut some bubble → throughput +8.8% > GPU-active −5%).
Trace `/workspace/sgl_prof_kern/*TP-0-DP-0*`.

### Next
(a) isolate GEMM-only vs RoPE-only; (b) FULL c512 1k/1k & 8k/1k end-to-end A/B with
the flags (confirm real-workload gain, not just pure-prefill); (c) the ~10% per-step
bubble. Artifacts: `/workspace/bench_pp_kern/`, `/workspace/gsm8k_kern.log`.

---

## Update 2026-06-18 — shared-expert-local PoC (Exp 50)

Per-layer trace showed MoE shared-expert+gate block ~2× SGLang. Root cause: SGLang
runs shared expert on the GATHERED global buffer (M≈131072), ATOM on LOCAL (M≈16384).
IMPORTANT: shared expert is TP-sharded → per-rank FLOPs IDENTICAL (NOT 8× redundant,
unlike the replicated gate, Exp 38). Same-FLOPs microbench: TP1-local shapes faster
(down GEMM TP8 K=384 is inefficient, 1.74×); and ck_xdl (SGLang) is actually faster
than ck_tile (ATOM) — so it is NOT a kernel issue, do NOT swap kernels.

PoC `SGLANG_DP_SHARED_EXPERT_LOCAL=1` (+`SGLANG_SHARED_EXPERT_TP1=1`): compute the
replicated shared expert on local hidden before the gather, skip it in self.mlp, add
to this rank's reduce-scattered local slice. Prefill-only.
| ISL | SGL base | +SE-local | ATOM | gain | vs ATOM |
|---|---:|---:|---:|---:|---:|
| 1024 | 48,291 | 51,171 | 57,821 | +6.0% | 84%→88% |
| 8192 | 47,013 | 50,324 | 55,999 | +7.0% | 84%→90% |
gsm8k 0.9393 (correct). Gain > the ~1.4% GEMM-shape microbench because the local path
also runs 8× FEWER ROWS through fp8-quant/elementwise on the global buffer.
Caveats: needs TP1 shared (~+0.5GB/rank); **c512 end-to-end (decode-bound) UNVERIFIED**
(gate-local was neutral). Edits in deepseek_v2.py (skip_shared_experts param) +
deepseek_v4.py (compute local + skip + add post-reduce-scatter), env-gated default OFF.
### Next
FULL c512 end-to-end A/B with SE-local; stack with Exp 49 (CK GEMM+rope) — additive?
Artifacts `/workspace/bench_pp_se/`, `/workspace/gsm8k_se.log`.

---

## Update 2026-06-18 — ALL 3 prefill levers stacked (Exp 51)

`SGLANG_FORCE_CK_W8A8=1 SGLANG_ROPE_BATCHED=1 SGLANG_DP_SHARED_EXPERT_LOCAL=1
SGLANG_SHARED_EXPERT_TP1=1` (+ gatherv ON, ROCM700A=0). gsm8k 0.9477 (correct).
Pure-prefill (OSL=1), gains ~additive:
| ISL | base | ALL3 | ATOM | ALL3 vs base | vs ATOM |
|---|---:|---:|---:|---:|---:|
| 1024 | 48,291 | 55,944 | 57,821 | +15.8% | 97% |
| 8192 | 47,013 | 54,645 | 55,999 | +16.2% | 98% |
Per-layer (pa_prefill-windowed, ratio-4): base ~7000us SGLang-slower → **ALL3 ~600us**
(39.5 vs 38.8ms). GPU-active/layer base 44.81→ALL3 38.53 ms (now < ATOM 41.54).
Op-by-op aligned: o-proj GEMM both ck_tile, shared-expert both local, MLA/MoE/comm/kv-q
all shared+equal. Residual small diffs: rope (SGL batched 332 vs ATOM fused-inverse
155us) + compressor glue/bubble (SGL fill/rocprim/fused_norm_rope vs ATOM 3 hca_* fused
kernels). **Prefill is essentially matched (97–98%).**

### Per-layer diff METHOD (persisted, reuse for future debug)
`useful-scripts/benchmarking/dsv4/layer_diff.py`: `overview <trace>` / `seq <trace>
<ratio>` / `cmp <sgl> <atom> <ratio>`. Boundary = `pa_prefill` kernel (alternates by
compress_ratio 128/4; match same ratio). Trust window span + GPU-active union, NOT
per-kernel dur (Exp 19). Capture single-stream + pure-prefill (OSL=1).

### Per-layer op table — SGLang ALL3 vs ATOM (ratio-4 layer, us)
| stage | SGL ALL3 | ATOM | status |
|---|---:|---:|---|
| pa_prefill (MLA attn) | 4610 | 4680 | = shared |
| out RoPE | batched 332 | inverse_gptj 155 | ⚠ ~180us |
| o-proj main GEMM | ck_tile 1904 | ck_tile 1802 | ✅ aligned |
| shared-exp up_gate/silu/down | 744/65/444 | 752/48/442 | ✅ aligned (local) |
| gate router cijk | 572 | 99 | ⚠ SGL still global (Exp 38) |
| comm gather / reduce-scatter | 4498 / 4615 | 4694 / 4861 | = |
| routed moe1/2/reduce | shared | shared | = |
| kv_a/q_a proj ck_xdl | 2739 | 2730 | = shared |
| compressor glue | fused_norm_rope+fill×3+rocprim×2 | hca_* 3 fused | ⚠ bubble (top remaining) |
| **window total** | **39,472us** | **38,846us** | gap ~600us (base ~7000) |
GPU-active/layer: ALL3 38.53 ms < ATOM 41.54. Aligned: o-proj GEMM, shared-expert
(local), MLA/MoE/comm/kv-q. Remaining: compressor glue/bubble, rope-fuse, gate-global.

---

## Update 2026-06-18 — ALL3 c512 END-TO-END A/B (Exp 52)

Does the matched prefill move c512 TOTAL tput? ALL3 (FORCE_CK_W8A8 + ROPE_BATCHED +
SHARED_EXPERT_LOCAL + SHARED_EXPERT_TP1) at c512, full OSL=1024, ATOM client.
| workload | SGL base | SGL ALL3 | ATOM | ALL3 vs base | vs ATOM |
|---|---:|---:|---:|---:|---:|
| 1k/1k | 17,233 | 17,877 | 19,828 | +3.7% | 86.9%→90% |
| 8k/1k | 32,254 | 34,613 | 39,291 | +7.3% | 82.1%→88% |
gsm8k 0.9477. REAL c512 gain (NOT neutral like gate-local — these levers cut a much
bigger prefill chunk), but diluted vs +16% pure-prefill because c512 is decode-bound;
bigger at 8k. Remaining c512 gap (~10–12%) = decode/scheduling + compressor-glue
bubble, not the fixed kernels.

### + chunk 8192/rank stacked on ALL3 (Exp 45 lever) — BEST c512 config
`--chunked-prefill-size 65536` (=8192/rank) ON TOP of ALL3:
| wl | base | ALL3 (16k/r) | ALL3 + chunk 8k/r | ATOM | best/ATOM |
|---|---:|---:|---:|---:|---:|
| 1k/1k | 17,233 | 17,877 | 17,921 | 19,828 | 90% |
| 8k/1k | 32,254 | 34,613 | **36,031** | 39,291 | **92%** |
8k: chunk8k adds +4.1% on ALL3 (vs base +11.7%, 82%→92% of ATOM). 1k saturated (+0.2%).
⇒ BEST c512 = ALL3 + chunk 8192/rank. Chunk lever (queue-fairness, Exp 45) is
independent of and stacks with kernel/locality levers, esp. at 8k.

### NET 2026-06-18 prefill optimization summary (Exp 49–52)
3 env-gated levers (default OFF), gsm8k 0.9477:
- `SGLANG_FORCE_CK_W8A8` (MLA proj Triton→CK), `SGLANG_ROPE_BATCHED` (batched compressor
  rope), `SGLANG_DP_SHARED_EXPERT_LOCAL`+`SGLANG_SHARED_EXPERT_TP1` (shared expert local).
- pure-prefill 84% → **97–98%** of ATOM (+16%); per-layer gap ~7000us → ~600us.
- c512 end-to-end **+3.7% (1k) / +7.3% (8k)**; 87/82% → 90/88% of ATOM.
- Edits: fp8_utils.py, deepseek_v4_rope.py, deepseek_v2.py, deepseek_v4.py (default OFF;
  lost on container rebuild). SE-local needs TP1 shared (~+0.5GB/rank).
- Next ideas: compressor-glue fuse (ATOM hca_norm_rope_scatter) + rope full-fuse to
  shave the residual prefill bubble; then the c512 gap is decode/scheduling-bound.
  Artifacts: `/workspace/bench_c512_all3/`, `/workspace/c512_all3_sweep.log`.
