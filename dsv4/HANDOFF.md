# DeepSeek-V4 DP performance — Handoff summary (updated 2026-06-25)

Read this first to pick up the investigation in a new session. Full detail is in
`EXPERIMENT_LOG.md` (index → dated `EXPERIMENT_LOG_<date>.md` files) + topic docs
`TBO_RESEARCH.md`, `TRACE_PROFILING.md`; methods/scripts in `SKILL.md`.

> **Latest 2026-06-25 (PM) — DSV4 EP+TBO IMPLEMENTED & numerically CORRECT
> (prefill two-batch-overlap on the mori EP path); stability bug open.** Wired
> `DeepseekV4DecoderLayer` into sglang's TBO op-engine so `--enable-two-batch-
> overlap` now overlaps one ubatch's mori a2a dispatch/combine with the other's
> attn+expert GEMM (prefill only). **gsm8k EP+TBO = 0.9600/0.9567** (limit 300;
> correct band, vs no-TBO mori 0.9431/0.9522). Approach: TBO disables DSV4's
> cross-layer fused-mHC so each layer is self-contained → maps to ops; the MoE
> ops (`op_gate`/`op_dispatch_a/b`/`op_combine_a/b`/`op_experts`/`op_shared_experts`/
> `op_output`) are REUSED from `self.mlp` (DeepseekV2MoE, decompose `forward_deepep`);
> DSV4 adds layer-level `op_mhc_prepare_attn`/`op_mhc_post_attn_pre_mlp`/
> `op_mhc_postprocess` (wrap hc_pre/hc_post) + `MQALayer.op_attn`. **No scheduler
> changes** — TBO batch-prep is model-agnostic (mori `normal` deepep-mode permits
> prefill TBO). Files changed (all `/sgl-workspace/sglang-upstream`): `models/
> deepseek_v4.py` (ops + `_can_run_tbo`/`_forward_layers_tbo` driver, reuses generic
> `execute_overlapped_operations`+filter/merge), `models/deepseek_v2.py`
> (`op_select_experts` passes `input_ids` for hash MoE — DSV4 is hash; no-op else),
> `batch_overlap/operations_strategy.py` (DSV4 branch + prefill strategy),
> `layers/attention/tbo_backend.py` (`__getattr__`→primary so DSV4 backend methods
> like `get_unified_swa_loc` resolve through the wrapper). Bugs fixed en route:
> TboAttnBackend missing `get_unified_swa_loc`; decode cuda-graph OOM (→ mem-frac
> 0.72, cuda-graph-max-bs 64); HashTopK missing input_ids; merge missing `residual`
> key. **OPEN: intermittent `HSA_STATUS_ERROR_OUT_OF_RESOURCES`** ("spawn threads/
> create OS events") under sustained TBO load — **DSV4-SPECIFIC** (control:
> R1-0528-MXFP4 + mori + TBO ran a FULL gsm8k with NO crash, 0.9484/0.9439; DSV4+
> mori WITHOUT TBO is stable → it's the DSV4×TBO combo). Ruled out: VRAM
> (expandable_segments no help), aiter JIT (76 loads =no-TBO), fd limit (1M), and
> **mori per-call events (R1+TBO uses them and is stable)**. DSV4 multi-stream
> record_event paths are OFF in our config. Locus = DSV4 attn/compressor/indexer
> under TBO (init_forward_metadata 3× on TboAttnBackend primary+2 children, attn 2×
> per ubatch); prime suspect = a DSV4-only aiter kernel creating HSA queues/signals
> per call. **ROOT-CAUSED 2026-06-25 PM (round 3): it's a 3-way interaction
> DSV4 × TBO × decode-cuda-graph.** Instrumentation (`_resmon_maybe_log`, file gate
> `/workspace/RESMON_ON`) showed NO Python Event/Stream/mem leak (flat to the crash)
> + flat /proc fd/threads → HSA resource is C++/ROCr-internal. Decisive bisect:
> **`--disable-cuda-graph` → DSV4+TBO runs FULL gsm8k with NO crash (0.9447/0.9454)**
> (cuda-graph ON crashes by ~150 prefill forwards). Matrix: DSV4+TBO+cg=CRASH,
> DSV4+TBO+no-cg=STABLE, DSV4-no-TBO+cg=STABLE, R1+TBO+cg=STABLE. Mechanism: TBO
> wraps attn in TboAttnBackend (primary+2 children, init_cuda_graph_state on all 3);
> the DSV4 decode cuda-graph + this wrapper leaks HSA (R1's aiter backend doesn't).
> **Workaround today: DSV4 EP+TBO is stable+correct with `--disable-cuda-graph`.**
> Proper fix (next): skip child-backend decode cuda-graph for DSV4+TBO (DSV4 TBO is
> prefill-only; children likely don't need decode graphs) → re-test with cg ON.
> **UPDATE (rounds 4–8): child-skip INSUFFICIENT; blind kernel swaps all still crash
> (attention unified_kv_triton↔triton, aiter indexer on↔off, mori async on↔off, HSA
> scratch on↔off). hipEvent/hipStream interposer = FLAT (not leaking); hipGraph/HSA
> interposers blocked by ROCm symbol versioning. DECISIVE round-8 isolation: crash
> needs BOTH (a) executing prefill-TBO ops AND (b) decode cuda-graph — TBO infra
> present-but-not-executed (gate `/workspace/TBO_NOEXEC`) ran full gsm8k clean
> (0.9477), and prefill-TBO-exec + `--disable-cuda-graph` is clean (round 2); only
> exec×graph crashes. Leading mechanism: executing prefill-TBO is the ONLY thing that
> uses the 2nd mori inner dispatcher (`MaybeTboDeepEPDispatcher` builds 2
> `MoriEPDispatcher`s; subbatch1→inner[1]); inner[1]'s mori HSA queues/signals + decode-
> graph HSA reservations + DSV4's heavier per-layer HSA baseline exceed the ROCm HSA
> ceiling (R1 lighter → survives). Next: confirm via `HSA_TOOLS_LIB` HSA-queue/signal
> count (version-agnostic, env inherited by schedulers) diffing exec-vs-noexec; fix
> candidates = mori runtime env tuning the aligned script omits (see
> run_sgl_dsv4_mori-ep.sh MORI_*), share/bound the 2 dispatchers' HSA queues, or
> smaller cuda-graph-max-bs. Workaround today: DSV4 EP+TBO stable+correct w/
> `--disable-cuda-graph`.**
> **FIXED 2026-06-26 (rounds 11-14): root cause is NOT a specific subsystem — op-stub
> bisection showed mHC-only crashes AND attn+MoE-without-mHC crashes, only pure
> passthrough (no real kernels) is stable. ⇒ ANY real prefill-TBO kernel on the
> continuously-varying ubatch shapes (tbo_padded_len pads only to attn_tp_size=1 → no
> bucketing) → per-shape kernel JIT/autotune accumulates ROCm HSA resources →
> OUT_OF_RESOURCES (decode-graph reduces headroom). ATOM avoids this: it pads decode
> TBO ubatches to fixed graph_bs//N buckets + keys graphs by (graph_bs,max_q_len).
> FIX: round tbo_padded_len up to next pow2 (≥256) in filter_batch → bounded ubatch
> shapes → DSV4 EP+TBO + cuda-graph ON ran FULL gsm8k NO crash, 0.9507/0.9500. (Gated
> by file /workspace/TBO_BUCKET; TODO: make env/registered flag, scope/tune, bench
> overhead, then commit on top of e299a3385.)**
> **SUPERSEDED 2026-06-26 (round 17, from ATOM InferenceX PR #1717): the REAL fix is
> `GPU_MAX_HW_QUEUES=5`** (caps ROCm HW queues = the resource HSA OUT_OF_RESOURCES
> exhausts). DSV4 EP+TBO + GPU_MAX_HW_QUEUES=5, **no bucketing, conc 256** ran a full
> 8k/1k bench NO crash → fixes the crash AND scales to high conc (bucketing was a
> low-conc-only workaround; finer bucket mult512 even crashed). **But proper A/B @conc
> 256: EP 12,255 vs EP+TBO 10,626 tok/s (TBO −13%, TTFT 25→40s) → my prefill-TBO
> op-decomposition is correct+stable but does NOT deliver the overlap benefit (regresses
> −3%@c64, −13%@c256). TBO not a perf win for DSV4 as implemented.** Recommended:
> `GPU_MAX_HW_QUEUES=5` for stability; don't enable TBO for throughput yet (investigate
> why overlap doesn't materialize). bucket commit 0b4fc8bc is now optional/superseded.**
> Launch: aligned `EP_MODE=mori` +
> `SGL_EXTRA_ARGS="--enable-two-batch-overlap --mem-fraction-static 0.72
> --cuda-graph-max-bs 64 --max-running-requests 64"` + PYTHONPATH pin. Details +
> next steps: `EXPERIMENT_LOG_2026-06-25.md` (TBO section).

> **Latest 2026-06-25 — NEW regime: 70k input / 300 output, low concurrency (2–32),
> SGLang-only config sweep.** Long-context prefill-dominated profile. Best-of (total
> tok/s): c2 13,937 / c4 17,655 (**TP8**); c8 29,194 / c16 36,779 / c32 42,140
> (**DP-attention**). Settings: conc<8 → plain TP8; **conc≥8 → TP8+DP-attention,
> chunk 16384/rank, swa 0.1, mem-fraction 0.80, prefill-delayer OFF**. Key findings:
> (1) **TP8 throughput saturates at conc≈8** (~22k; extra conc only adds latency) —
> a single request's attention is TP-sharded over all 8 GPUs. (2) **DP-attention
> scales** (each rank prefills its own req; +22/58/77% over TP8 at c8/16/32) with
> **0 retract** (per-rank KV load = conc/8×70k, tiny). (3) **DP MoE OOMs unless
> mem-fraction ≤ 0.80** — dp MoE runs on the GATHERED global batch (per-rank chunk ×
> dp), 18GB activation; dp KV is over-provisioned so dropping mem is free. (4) chunk
> size is throughput-NEUTRAL (±2%); 16384/rank chosen (halves global MoE batch →
> safer). (5) **DSV4 uses a custom `DSV4PoolConfigurator`** (full/swa/c4/c128 +state
> sub-pools); **swa sub-pool = full×swa_full_tokens_ratio** and binds TP8 at conc=32
> (swa usage→1.0, retracts) — conc=32×70k is at the single-node memory ceiling in
> TP8 (best balance swa≈0.24/mem0.97, a few retracts unavoidable). (6) **prefill-
> delayer OFF wins +6–16%** here — OPPOSITE of the 8k/1k high-conc finding (delayer
> protects decode occupancy, useless when prefill-dominated). **Pre-flight gotcha:**
> `import sglang` was shadowed by a namespace pkg at `/sgl-workspace/sglang`; pin
> upstream with `PYTHONPATH=/sgl-workspace/sglang-upstream/python:/sgl-workspace/
> mori:/sgl-workspace/aiter`. Scripts: `run_sgl_dsv4_70k.sh` (MODE/CHUNK/SWA/MEM/
> DELAYER). Details: `EXPERIMENT_LOG_2026-06-25.md`.

> **Latest 2026-06-24 — aiter reduce_scatter decode combine SHIPPED (PR #29103).**
> Decode (MAX_LEN, no EP, tp==dp) MoE combine now uses an equal-chunk reduce_scatter
> (aiter custom `reduce_scatter_first_dim` on ROCm, RCCL elsewhere) instead of
> all_reduce (`cross_device_reduce_2stage`) + dp_scatter → ~½ the combine traffic.
> Both gather (`allgather_vec`) and combine are now aiter custom kernels. Env
> `SGLANG_DP_USE_REDUCE_SCATTER` (platform-conditional default: ON for HIP, OFF
> else; renamed from SGLANG_USE_AITER_RS). Made shippable: `is_dp_gatherv_active()`
> now also requires SUM_LEN (gatherv pair only valid under SUM_LEN), so the MAX_LEN
> decode combine routes through `dp_reduce_scatter_tensor` → equal-chunk path.
> **c512 (no-regression A/B): 8k/1k 37,171→37,701 (+1.4%), 1k/1k 18,026→18,666
> (+3.5%); gsm8k 0.9545/0.9553.** Complements gatherv (which is SUM_LEN → prefill
> only): gatherv handles prefill, reduce_scatter handles decode; neither pays full
> all_reduce traffic. Files: environ/parallel_state/dp_attention/deepseek_v4.
> Details: `EXPERIMENT_LOG_2026-06-24.md`.

> **Also 2026-06-23 — root-caused why sglang decode used all_reduce not
> reduce_scatter + built the PoC.** Decode gatherv/reduce_scatterv only fires under
> SUM_LEN; decode under cuda graph takes `get_default_mode_in_cuda_graph()` =
> SUM_LEN iff `SGLANG_USE_ROCM700A=1`, else MAX_LEN. ROCM700A=0 vs =1 A/B (8k/1k
> c512): **=0 wins +2.24%** — because MAX_LEN decode uses aiter custom kernels
> (allgather_vec + quickreduce) which beat RCCL all_gatherv+reduce_scatterv even
> though RS moves less data (RCCL microbench: AR 74.7us vs RS 54.6us @M512, but
> aiter beats RCCL). ⚠️ **cuda-graph trace caveat**: decode per-kernel durations are
> garbage (graph-replayed) — use the collective microbench, not the decode trace,
> for timing. Captured ATOM + sglang decode-heavy (wave-tail) traces. Details:
> `EXPERIMENT_LOG_2026-06-23.md`, `TRACE_PROFILING.md`.

> **Also 2026-06-24 — TBO research (TBO_RESEARCH.md).** sglang
> `--enable-two-batch-overlap` requires an EP a2a backend (op-list coroutine over
> the dispatcher's async dispatch_a/b; per-model hardcoded; DSV4 NOT supported, no
> op_ decomposition). ATOM `--enable-tbo` is generic (threads + shared dual-stream,
> unmodified forward) and overlaps DP all_gather+reduce_scatter even without EP
> (prefill only; decode regresses). DSV4 can't TBO today; adding non-EP TBO ≈ 1–2wk
> (DeepseekV2) / 3–5wk (DSV4 forward op-refactor); ATOM-style generic rewrite is
> larger and buys no extra overlap.

> **Also 2026-06-22 (Exp 58) — gatherv MoE-gather: removed 2 redundant 940MB DtoD
> copies, gate region 1011→332us.** Trace showed nccl→topk_softplus = Memcpy DtoD 358
> + Memcpy DtoD 356 + gate GEMM 297 = 1011us (ATOM ~354us). (DtoD memcpys are
> cat="gpu_memcpy", NOT "kernel" — filter both.) Cause: `_dp_gather_via_all_gatherv`
> did `all_gatherv` (NCCL allocs own out) then `torch.cat([out])` + `global_tokens.copy_`
> = two full-buffer (sum(sizes)*hidden = 939MB @ c512/ISL8192) copies. Fix: added
> `output=` to `GroupCoordinator.all_gatherv` and pass the pre-alloc dp buffer so NCCL
> gathers straight in; dropped cat+copy_. gate region **1011→332us (~3x, below ATOM)**,
> gsm8k 0.9538/0.9545. Committed `e6cec15e18`. `output=` defaults None (other callers
> unchanged).

> **Also 2026-06-22 (Exp 57) — flat-row RoPE kernel: attention-output rope 168→59us,
> now BELOW ATOM.** A c512/ISL8192 trace showed our `apply_rotary_emb_contig_kernel`
> at p50 168us vs ATOM `inverse_rope_gptj` 83us (~2x). Root cause = access pattern, NOT
> BLOCK_M: trace grid `[1024,128]` revealed the real shape is **[8192,128,64]** (DP-attn
> = all 128 heads/rank; my PR #28783 BLOCK_M tuning used the wrong 16-head shape). The
> contig kernel reads BLOCK_M tokens for one head strided by n_heads*head_dim → scattered,
> ~2.2 TB/s cold (a pure streaming rw hits 5.37 TB/s cold). New
> `apply_rotary_emb_flat_kernel` iterates `row = token*n_heads + head`, BLOCK_ROWS
> consecutive rows/program (rows only head_dim apart) → **~4.5 TB/s cold (~2x)**, bit-exact.
> Validated: gsm8k 0.9560/0.9568; trace rope p50 **168→59us (~2.85x, below ATOM 83us)**.
> Removed the unused contig kernel. Committed `d6e817b0f4` (dev clone). Method note: rope
> must be benchmarked at the REAL [tokens, 128, 64] shape and COLD (rotate distinct
> tensors) to match production — warm/16-head microbench is misleading.

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
- aligned launch: `/workspace/useful-scripts/benchmarking/dsv4/run_sgl_dsv4_aligned.sh`

---


---

## Detailed update archive

The long `## Update <date>` detail sections were moved to `HANDOFF_ARCHIVE_updates.md` (and are also covered per-day in `EXPERIMENT_LOG_<date>.md`).
