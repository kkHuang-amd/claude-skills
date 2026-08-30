# STATUS — gfx1250 DeepSeek-R1-0528-MXFP4 accuracy gap investigation

Single entry point / handover snapshot. Last updated **2026-07-09**.

> # ✅ SOLVED (E37): FIX A alone recovers R1 to >= gfx950 at FULL SPEED.
> Production recipe (cuda-graph, REAL a8w4 MoE, bf16 KV) + FIX A = GSM8K **0.925 (40Q) / 0.980
> (200Q)** @ 80-135 tok/s. The entire gap was the softmax-weight **bf16 downcast in the attention
> P·V dot**. FIX = keep `p` fp32: `tl.dot(p, v.to(tl.float32), out_dtype=tl.float32)` in
> `triton_ops/decode_attention.py::_fwd_grouped_kernel_stage1` + `extend_attention.py::_fwd_kernel`
> (prefix + extend-local). QK needs no change (Triton bf16 dot already fp32-accumulates). **MoE
> kernel change / FIX B NOT needed** (real MoE + FIX A = 0.98). (CORRECTION: `alt_stream` is None on
> gfx1250, so my Triton qk-rmsnorm runs in BOTH eager and cuda-graph — it is fine, not the gap.
> The E31/E34/E35 eager "both kernels" matrix vs FIX-A is a loose end likely due to my imperfect
> E34 torch-naive attention hook; the VERIFIED result stands: real MoE + FIX A + cuda-graph =
> 0.925/0.98.) See E36/E37.
>
> **UPDATE (E40): aiter bisect fix REMOVED** (reverted to `ceil(log2)`): its `bit_length()` (9
> iters, 256 experts) read layout_buffer[256] OOB and crashed DSv4 at high concurrency; the
> off-by-one it fixed is accuracy-neutral. Re-tested without it: 200Q = 0.950 (unchanged).
> **Shipping code changes = gfx1250 weight shuffle + AITER_GROUPED_FORCE_SPLIT_K1 + FIX A (fp32
> P·V attention). NOT the bisect fix.**

Read this first, then EXPERIMENT_LOG.md (E17-E43) for detail, CHANGES.md for the exact
code edits, HANDOVER_crossnode_dump.md for the next experiment.

> **★ UPDATE 2026-07-10 (E43, node H21-18 gfx1250): fp8 KV cache is FIXED & usable — SKILL §4.7
> "fp8 KV broken, use bf16" is SUPERSEDED.** New image `henryx/xsgl:v0.5.14-...-20260709-trial-3`
> (sglang 3923a34d, aiter 9af05b91) reproduces bf16 baseline (1319Q 0.951). Root cause of the fp8-KV
> decode crash, PINNED: **gfx1250 triton `tl.dot(fp8,fp8)` returns garbage (~1e34) for contraction
> dim K>=128** (K=64 fine; bf16 fine at all K) — a gfx1250-specific fp8-MFMA codegen bug (gfx950's
> fp8 dot at K=512 is correct -> gfx950 fp8 KV = 0.941, same code). MLA nope QK dot is K=512 and the
> triton kernels downcast q to fp8 to match the fp8 KV cache -> garbage -> softmax(inf) -> NaN ->
> degenerate decode. Two spots: `decode_attention.py::_fwd_grouped_kernel_stage1` (decode) and
> `extend_attention.py` prefix loops (fires on radix-cache prefix reuse). **FIX (both files, no-op
> for bf16 KV): keep q bf16, upcast fp8 K to bf16 in the dot.** Validated E2E fp8 KV: 40Q 0.950 /
> 1319Q **0.949** (== bf16 0.951), full recipe (radix + cuda-graph, NO workaround flags), 182 tok/s,
> halves KV memory. RED HERRINGS ruled out (all retracted): overflow->NaN (real |q|max ~330 < 448),
> multi-split (fails at splits=1 too), fp8 cast semantics (identical gfx950==gfx1250), fp8 gemm at
> small K (correct 1e-7). Upstream triton/ROCm repro: `artifacts/triton_fp8_dot_largek_gfx1250_repro.py`
> + `artifacts/BUGREPORT_triton_fp8_dot_largek_gfx1250.md`.
> **gfx1250 DEV RULE: never `tl.dot(a_fp8,b_fp8)` with K>=128 (upcast to bf16 / tile K<=64 / scaled-fp8
> gemm); bf16 tl.dot unaffected.** See EXPERIMENT_LOG E43 + results_gfx1250-H21-18.md.
>
> **FIX A (fp32 P·V) is NOT a codegen bug — it is genuine bf16 precision (checked E43).** Triton
> `tl.dot` P·V (p=softmax rows, v=gaussian) on gfx1250: bf16xbf16 = ~2.4e-3 (normal bf16 output
> rounding), STABLE across K=16..512, no garbage; fp32xfp32 = ~1e-7. So the original code's downcast
> of the softmax weights `p` to bf16 loses ~0.2-0.4%/weight, which ACCUMULATES over long CoT
> (0.85->0.925); keeping p fp32 fixes it. Categorically different from the fp8 K>=128 defect (which
> returns ~1e34 garbage): FIX A = a dtype/accumulation-precision choice, the fp8 bug = a broken
> instruction. gfx1250 bf16 tl.dot is numerically well-behaved.
> **⚠️ CONTRADICTION — RESOLVED via candidate (b): FIX A is NOT the lever on this image.** The
> "genuine bf16 precision" story could not explain why gfx950 didn't need FIX A. Direct single-var
> A/B on gfx1250 (this image, sglang 3923a34d): toggle ONLY the P·V p-dtype (fp32 vs bf16) in
> decode+extend, everything else identical, bf16 KV, production recipe:
>   FIX-A ON (p fp32): 200Q 0.960, 1319Q 0.951.   FIX-A OFF (p bf16): 200Q 0.965, **1319Q 0.948**.
> => reverting FIX A does NOT drop accuracy (0.948 vs 0.951 = noise), vs the historical claim of
> 0.85 (off) -> 0.925 (on). So on THIS stack **p-fp32 is accuracy-neutral — FIX A is not the lever**;
> the historical 0.85->0.925 attribution (E36/E37/E39) was a CONFOUND (that saga's masking/emul
> setup) or fixed by the newer sglang/aiter. This dissolves the gfx950 contradiction: nobody needs
> FIX A on this image, so there is no gfx950-vs-gfx1250 puzzle. (FIX A left in place — it is
> accuracy-neutral and fp32 P·V is safe; it could be dropped for a small perf gain. The fp8-KV
> upcast fixes are the ones that matter.) Candidate (a) — whether gfx950 even downcasts p — is now
> moot for accuracy but a gfx950 agent prompt was written to confirm the code path anyway.

## ⚠️ MULTI-NODE CONVENTION (several machines share this dir)
Multiple nodes run these experiments and write to these SAME md files — concurrent edits can
clobber. Rules:
- **Tag every result/entry with its node + arch**, e.g. `[node: H21-18 gfx1250]` or
  `[node: <hostname-short> gfx950]`. Known nodes so far:
  - `ctheliosp-...-H21-18` = **gfx1250** (this session's box; the current gfx1250 stack).
  - `ctheliosp-...-H21-17` = **gfx1250** (the original 7aa6082 docker; now gone).
  - the **gfx950** cross-val box (TP2) = `/sgl-workspace/sglang_gfx-1250` tree.
- **APPEND, never rewrite** shared files; keep your node's edits in clearly-tagged blocks.
- **Write raw runs to your node's file first, then promote a short summary into
  EXPERIMENT_LOG/STATUS** (this avoids the concurrent-write conflicts we already hit). Per-node
  files: `results_gfx1250-j19-10.md` (this box), `results_gfx1250-H21-18.md` (2nd gfx1250),
  `results_gfx950.md` (gfx950 ref). EXPERIMENT_LOG/STATUS = curated cross-node summary only.
- Numbers without a node tag are ambiguous — always state which arch/node produced them.
- Additional known node: `ctheliosr-rck-g02-j19-10` = **gfx1250** (4x; primary box for the
  token-sweep E31 through E39; sglang 000a61a2, aiter 8815f4b5).

> **★ UPDATE 2026-07-09 (E39 — node: gfx1250 SECOND box): BREAKTHROUGH — the gap = TWO real kernels
> that MUTUALLY MASK.** Idealizing (bf16/torch) the MoE AND the attention independently and together:
> real+real ~0.80; **idealize MoE only ~0.825; idealize attention only ~0.825; idealize BOTH = 0.925
> (== gfx950).** => the gap is the FlyDSL a8w4 grouped MoE (large-token) AND the Triton MLA softmax
> attention (decode/extend) EACH injecting error, masking each other. This resolves all prior
> "innocent" verdicts as the masking confound: E31 "MoE fix neutral" (masked by attention), E22/E34
> "attention clean" (per-op 0.16% ACCUMULATES over long CoT; only shows once MoE is also idealized).
> qk-rmsnorm is fine (combo kept Triton qk-norm and still hit 0.925) — the attention issue is the
> **softmax decode/extend kernel**. **ACTIONABLE: fix BOTH gfx1250 real kernels, validate
> END-TO-END (small-token op-tests are blind): (1) Triton MLA softmax attention -> match torch SDPA
> over paged KV; (2) FlyDSL a8w4 grouped MoE large-token -> the bisect targets this but is OOB-crashy
> (E38), reimplement OOB-safe or find the true large-token accumulation issue. Fix one -> ~0.825;
> both -> ~0.925.**
>
> **UPDATE 2026-07-09 (E37/E38 — node: gfx1250/ctheliosr-rck-g02-j19-10): the DSv4 crash is caused
> by our own bisect "fix", now REVERTED.** DSv4 + cuda-graph + GSM8K `--parallel 32` crashed
> reproducibly (`hipErrorIllegalAddress`) — but ONLY with the E31 bisect edit
> (`gemm_mxscale_gfx1250.py:3010` `bit_length()`); reverting to the original `ceil(log2)` makes DSv4
> run clean at **0.925** (parallel 32, cuda-graph). The `bit_length` extra iteration is an OOB read
> of the layout buffer. Since that "fix" was **end-to-end neutral** for R1 (E31) and crashes DSv4,
> it is **NET-NEGATIVE and left reverted**. => STATUS "3 code changes" #1 is now **do-NOT-apply**;
> only #2 (weight shuffle) and #3 (AITER_GROUPED_FORCE_SPLIT_K1) are needed.
>
> **UPDATE 2026-07-09 (E36 — node: gfx1250/ctheliosr-rck-g02-j19-10): DSv4 = 0.925 on gfx1250.**
> Directly tested the user's "if it were hardware, DSv4 would also be bad" argument. DeepSeek-V4-Flash
> (fp8; experts fp4 -> SAME a8w4 MoE kernel as R1) GSM8K 40Q = **0.925** on gfx1250 (eager+small-batch;
> batched cuda-graph decode throws hipErrorIllegalAddress — DSv4 dsv4-attn/fp8-KV, unrelated). =>
> gfx1250 **hardware is fine** and the **a8w4 MoE kernel is fine** (DSv4 uses it, 0.925). So the R1
> gap (0.80-0.83) is in R1's **specific non-MoE path** (triton MLA + mxfp4->bf16 linears + bf16 KV),
> which DSv4 (dsv4-attn + fp8) does not share — most plausibly a cross-platform (wave32/64)
> accumulation diff in R1's triton MLA, or an mxfp4 quant-matching effect. Localizer: step-2 emul-dump,
> focus the attention sublayer. (Traps: v4 script lacks coredump guard -> a 95 GB gpucore; needs
> AITER_GROUPED_FORCE_SPLIT_K1=1. See E36.)
>
> **UPDATE 2026-07-09 (E33 — node: gfx950 control): gap is DECISIVELY non-MoE.** The gfx950
> chunked-emul control passed: **gfx950 chunked a8w4 emul = 0.950 (40Q)** == E30 batched 0.925 ==
> native ~0.93 -> the chunked emul is faithful (confound ruled out). So with the **identical
> chunked a8w4 bf16-MoE emul, gfx950=0.950 vs gfx1250=0.800** — MoE math byte-identical, so the
> **0.15 gap is 100% gfx1250 non-MoE platform** (attention/norm/rope/linear/sampling). MoE fully
> excluded. NEXT (localizer): cross-node per-layer dump with `SGLANG_MOE_EMUL=a8w4` on BOTH nodes
> (MoE identical -> every layer comparable) -> earliest above-floor rel_l2 = the offending non-MoE op.
>
> **UPDATE 2026-07-09 (E32 — node: gfx1250 SECOND box, GPU3): the gap is NON-MoE.** Ran the
> bf16 MoE emul ON gfx1250: even the **fully ideal MoE (a16w4, zero activation quant) = 0.825**
> (40Q) ≈ real 0.81, vs gfx950 0.925. Making the MoE ideal does NOT close the gap => the gap is
> **gfx1250's non-MoE platform execution** (attention/norm/rope/linear/sampling), NOT the MoE
> (real kernel [E31], scheme, or activation quant — all excluded). CAVEAT: gfx1250 emul is the
> **chunked** variant; gfx950's 0.925 was **batched** — must run the SAME chunked emul on gfx950
> (expect 0.925) to rule out a chunked-emul bug before fully trusting this. Next: (1) gfx950
> chunked-emul control; (2) whole-forward cross-node **logits** diff (not per-op) to localize the
> systematic non-MoE bias — note E22's attention check was "vs torch", never "vs gfx950".
>
> **UPDATE 2026-07-09 (E31 — node: gfx1250/ctheliosr-rck-g02-j19-10): the "large-token a8w4
> kernel numerics" hypothesis is OVERTURNED (end-to-end).** Token-swept op-test localized the
> growth to the **contiguous-M path bisect off-by-one** (`gemm_mxscale_gfx1250.py:3010`,
> `ceil(log2(256))` -> `bit_length()`; 256 experts = power of two). Fix -> op-test flat **3.4e-6
> at all token counts** (was 10% @ 512). **But GSM8K 100Q = 0.830, UNCHANGED** from the
> 0.80/0.83 baseline -> the large-token kernel error is **end-to-end NEUTRAL** (reconfirms E18,
> overturns E30's actionable claim). New clean contradiction: real fused MoE == a8w4 ref @ 3e-6
> (op-test) yet scores 0.83, while E30's ideal a8w4 emul = 0.925 (40Q). Next: on gfx1250, same
> 100Q, run the E30 bf16 MoE emul (`SGLANG_MOE_EMUL=a8w4`) vs the real kernel — emul>>real =>
> non-gemm real-kernel issue; emul~=real => the gap is scheme/measurement (E30's 0.925 was
> gfx950/40Q-optimistic). [SUPERSEDED by E38: the bisect fix was later REVERTED — it is end-to-end
> neutral AND causes an OOB crash on DSv4; do not apply.]

> **UPDATE 2026-07-09 (E26): the gfx950 absorb ablation is DONE — absorb is NOT the gap.**
> Forcing gfx950's MLA absorb BMM to bf16 (mimic gfx1250) left GSM8K at **0.942 vs 0.944
> (1319Q), 0.950 vs 0.950 (200Q)** = noise. gfx1250 weight-side absorb emulation (`w_fp4`)
> is thus expected neutral too — skip it.
>
> **UPDATE 2026-07-09 (E27/E28): gfx950 cross-node dump produced + PARTIAL diff done.**
> `hs_dump_gfx950.json` (a4w4, bf16 absorb) vs `hs_dump_gfx1250.json` (a8w4, STALE):
> **dense layers 0-2 match to the bf16 floor (rel_l2 ~3e-3), divergence starts EXACTLY at
> layer 3 (first MoE): input still matched (0.010) but post_layer jumps to 0.21**, then
> propagates. => the ONLY cross-node divergence is the MoE; non-MoE is consistent (no bug).
> Chart: `crossnode_dump_compare.png`. **Remaining blocker: re-dump gfx1250 with the new
> rank-gated hook** to also diff `post_attn` at MoE layers 3-60 (strict attention check).
>
> **UPDATE 2026-07-09 (E36): FIX A DONE — attention side recovered (0.825 -> 0.950).** Root
> cause was the softmax weights `p` being downcast to bf16 before the P·V `tl.dot`. Fix: keep p
> fp32 — `tl.dot(p, v.to(tl.float32), out_dtype=tl.float32)` — in `decode_attention.py::_fwd_
> grouped_kernel_stage1` and `extend_attention.py::_fwd_kernel` (prefix + extend-local). Validated
> WITH MoE-emul (attention the only real kernel): GSM8K 40Q = 0.950 (== gfx950). (QK already
> fp32-accumulates by Triton default — not the loss.) **NEXT: FIX B (flydsl a8w4 MoE kernel)** to
> recover the MoE side without the emul; then both fixes at full speed. See E36 + HANDOVER_kernel_fixes.md.
>
> **UPDATE 2026-07-09 (E35, PIVOTAL — SOLVED the localization): BOTH the gfx1250 MoE kernel AND
> the triton attention kernel are the gap; each masks the other.** MoE-emul(ideal) + attention-
> torch(naive) TOGETHER = **0.925** (== gfx950) @40Q. Matrix: both-real ~0.81; ideal-MoE+real-attn
> 0.825 (E31); real-MoE+torch-attn 0.825 (E34); **ideal-MoE+torch-attn 0.925**. So E31/E34's
> single-op "exonerations" were the masking confound (the OTHER real kernel still injects error).
> **Fix BOTH:** (1) triton MLA softmax attention (decode/extend) numerics; (2) flydsl a8w4 grouped
> MoE numerics (E16 large-token). Either alone -> ~0.825; both -> ~0.925. Validate END-TO-END
> (small-token op-tests blind). Next: 200Q/1319Q combo to confirm. See E35.
>
> **UPDATE 2026-07-09 (E33/E34, SUPERSEDED BY E35): attention "exonerated" — WRONG (masking).**
> E33: R1's RoPE on gfx1250 is pytorch `forward_native` (exact 0.0 vs ref) — arch-independent.
> E34: replacing ALL attention (decode+extend) with pure-torch naive => GSM8K still **0.825**
> (== baseline). So NEITHER MoE (E31) NOR attention (E34) NOR rope (E33) NOR qk-norm is the gap;
> each idealized alone leaves gfx1250 R1 at ~0.82 vs gfx950 0.93. Remaining un-idealized: linear
> bf16-dequant / shared-expert (n_shared=1) / lm_head / embedding / inter-block norms / sampling,
> OR a DISTRIBUTED bf16-accumulation gap. NEXT: (1) run MoE-emul AND attention-torch TOGETHER
> (still 0.82 => neither; 0.92 => the pair); (2) check the shared-expert path. See E33/E34.
>
> **UPDATE 2026-07-09 (E32, REFOCUS): the gap is R1-PATH-SPECIFIC (triton MLA), NOT general
> gfx1250.** [E33/E34 above now narrow this further: NOT attention either.] DSv4 (DeepSeek-V4-Flash) runs on the SAME gfx1250 box at GSM8K ~0.94 — it does
> NOT drop. DSv4 uses `--attention-backend dsv4` (sparse indexer) + fp8 KV; R1 uses
> `--attention-backend triton` (classic MLA) + bf16 KV (fp8 KV broke R1 decode, E13). So the
> gfx1250 hardware/shared infra is fine; the R1 gap is in R1's **triton MLA attention path**
> (the one DSv4 never touches). Combined with E31 (MoE bypassed, R1 still 0.82), the suspect is
> now the **R1 triton MLA path on gfx1250**, NOT MoE, NOT general gfx1250. E22 only checked the
> MLA softmax *kernel* vs torch (0.16%), not the FULL MLA sublayer end-to-end. NEXT: single-node
> R1 env A/B (`SGLANG_USE_ROCM700A=0`, qk-rmsnorm-torch) + whole-MLA-sublayer numerical check +
> try an alt MLA backend. The gfx950 apples-to-apples emul is now OPTIONAL. See E32.
>
> **UPDATE 2026-07-09 (E31, FLIPS E30): the gap is NOT the MoE at all — it is a gfx1250
> NON-MoE effect (case B).** Ran the SAME bf16 MoE emul DIRECTLY ON gfx1250 (bypass the real
> flydsl kernel): a8w4-emul = **0.800**, a16w4-emul (most ideal, zero MoE quant) = **0.825**
> (40Q) — i.e. ~= gfx1250's real 0.81, NOT gfx950's 0.925. So making the MoE ideal does NOT
> recover gfx1250 => the real a8w4 kernel is NOT the gap (E30's inference was from gfx950-only
> and is wrong for gfx1250). A new/fixed MoE kernel will NOT help. Caveats: 40Q noise; the
> gfx1250 emul is the CHUNKED variant (TP1 memory) — run the same chunked emul on gfx950 for a
> strict apples-to-apples (expect ~0.925). Next: bisect the NON-MoE gfx1250 path with a
> WHOLE-FORWARD gfx1250-vs-gfx950 logit compare (E22 attention checks were vs torch, not vs
> gfx950, so a systematic small gfx1250 bias could pass per-op yet accumulate). See E31.
>
> **UPDATE 2026-07-09 (E30, SUPERSEDED BY E31): the a8w4 SCHEME is accuracy-neutral — [E30
> inferred] the gap is the gfx1250 REAL KERNEL.** bf16 MoE emulation on gfx950 (`scripts/moe_emul_sitecustomize.py`,
> `SGLANG_MOE_EMUL`): **a4w4-emul = a8w4-emul = 0.925 (40Q)**, matching native a4w4 (~0.93).
> So an *idealized* a8w4 (fp8 act x fp4 weight) does NOT lose accuracy vs a4w4 — quant-matching
> is FALSE. Yet the gfx1250 real flydsl a8w4 kernel scores ~0.85 for the same scheme. =>
> **the shortfall is the gfx1250 grouped a8w4 kernel's real-execution numerics** (token/max_m-
> dependent error, cf. E16: 512-tok logits_diff ~10%), NOT the scheme/attention/absorb/serving.
> This FLIPS the earlier "MoE exonerated / quant-matching" leaning: **fixing the gfx1250 a8w4
> kernel's large-token numerics is now the actionable target** and should recover toward ~0.92.

## The problem
Serve `DeepSeek-R1-0528-MXFP4` on **gfx1250** (SGLang+aiter, forced **a8w4** MoE).
GSM8K (5-shot completion greedy):
- **gfx1250 (a8w4) = ~0.81-0.85** (1319Q = 0.811)
- **gfx950 (a4w4)  = 0.93** — **stably reproducible** (user-confirmed), same HF
  checkpoint, same sglang model/attn/MoE code (only fp8.py/fp8_kernel.py differ),
  same eval. So the ~8-12% gap is **REAL and comparable**, not a measurement artifact.

## Working recipe (env + args), on a healthy gfx1250
`AITER_FORCE_A8W4=1`, `SGLANG_MOE_SHUFFLE_GFX1250=1`, `AITER_GROUPED_FORCE_SPLIT_K1=1`,
`--kv-cache-dtype auto` (bf16; fp8 KV breaks decode), `--attention-backend triton`,
cuda-graph ON. Model at `/shared_nfs/huggingface_models/amd/DeepSeek-R1-0528-MXFP4`. Launch:
`/sgl-workspace/sglang/run_ds-r1.sh`. Serves, decode coherent, GSM8K ~0.85.

## 3 code changes THIS environment needed (re-apply after any machine switch)
See CHANGES.md "2026-07-08 session edits" for exact diffs. Summary:
1. ~~**aiter** `ops/flydsl/kernels/gemm_mxscale_gfx1250.py` ~L3010: bisect off-by-one
   `math.ceil(math.log2(batch_count))` -> `int(batch_count).bit_length()`.~~ **DO NOT APPLY
   (E38).** It fixes only the contiguous-M op-test (3.2e-3->3.4e-6) which is **end-to-end
   NEUTRAL**, and the extra `bit_length` iteration is an **OOB read of the layout buffer that
   crashes DSv4** (cuda-graph + parallel 32, `hipErrorIllegalAddress`; R1 tolerated it latently).
   Keep the ORIGINAL `ceil(log2)`. Only #2 and #3 below are needed.
2. **sglang** `quark/schemes/quark_w4a4_mxfp4_moe.py`: enable weight `shuffle_weight
   (w,(16,16))` on gfx1250 when a8w4 (mirror DSv4 fp8.py). MANDATORY on THIS aiter build —
   without it GSM8K = 0.000 garbage. NUANCE (E25c, verified 2026-07-09): this is because
   THIS docker's aiter grouped a8w4 kernel requires (16,16)-shuffled weight — NOT because
   raw weight is inherently broken (commit 7aa6082 got 0.82 on gfx1250 with raw weight on
   the OLD docker's aiter). Reverting the bisect fix does not restore raw-weight, so the
   break is the aiter kernel version, not sglang/bisect/split_k1.
3. **aiter** `ops/flydsl/grouped_moe_gfx1250.py`: add `AITER_GROUPED_FORCE_SPLIT_K1`
   env (force split_k1=split_k2=1; CSV picks 2 for token=1 decode -> illegal-address)
   + `AITER_GROUPED_FORCE_TILE_M` investigation knob (default off).

## What is RULED OUT (do not re-chase) — EXPERIMENT_LOG E18-E23
- **MoE a8w4 SCHEME / small-token numerics**: op-test == quant-ref @ 3e-6; a8w4 act quant
  (3.6%) is 4x more accurate than a4w4 (15%); and E30 bf16 emulation shows an *idealized*
  a8w4 scores the SAME as a4w4 end-to-end (0.925). So the scheme is NOT the gap.
  **CAVEAT (E30, revised):** this does NOT clear the gfx1250 *real* grouped a8w4 kernel — its
  logits_diff GROWS with token/max_m (E16: 512 tok ~10%), which the small-token op-test misses.
  That large-token kernel error IS now the leading suspect. (`scripts/moe_quant_probe.py`,
  `scripts/moe_emul_sitecustomize.py`)
- **Attention** (triton MLA decode + MHA prefill): both match torch fp32 to ~0.16-0.18%
  across all 61 layers, no outlier. Clean. (`scripts/attn_probe_sitecustomize.py`)
- **bf16 GEMMs** (attn proj / lm_head / dense-MLP): log `torch solution:0` = torch itself.
- cuda-graph vs eager (0.85 vs 0.84), `--disable-radix-cache` (0.83),
  `--triton-attention-reduce-in-fp32` (no-op), qk-rmsnorm torch-vs-triton — all neutral.
- `SGLANG_ROCM_FUSED_DECODE_MLA=1` **crashes** (keep =0).
- Failure modes (`%?%?` loops) are generic high-confidence greedy loops, not numeric collapse.

=> On gfx1250 **every component is numerically correct in isolation**. No single-kernel bug.

## Key reframing (E25): gfx1250 is MORE precise everywhere, yet scores lower
Audit of `_use_aiter_gfx95`-gated paths: for R1 (bf16 attention) most don't fire; the one
non-MoE path that does is the **MLA absorb BMM** — **fp4 on gfx950, bf16 on gfx1250**
(gfx950 runs `quark_post_load_weights` to mxfp4 w_kc/w_vc; gfx1250 skips it, stays bf16).
`quark_post_load_weights` is NOT used on gfx1250 (gated by `_use_aiter_gfx95`, False on
gfx1250). Direction matters: bf16 absorb (gfx1250) > fp4 absorb (gfx950); a8w4 MoE
(gfx1250) > a4w4 (gfx950). So gfx1250 is equal-or-MORE precise everywhere yet lower =>
leading thesis is a **quantization-MATCHING effect** (the PTQ model performs best under
its own a4w4+fp4 error pattern; gfx1250's more-precise bf16/a8w4 is a distribution mismatch).

## DONE (E26, 2026-07-09): gfx950 absorb-BMM ablation — RULED OUT
Tested the quant-matching thesis with NO cross-node confound: on gfx950, forced the MLA
absorb BMM to bf16 (like gfx1250) and re-measured GSM8K. **Result: no change** (fp4 0.944
vs bf16 0.942 @ 1319Q; 0.950 vs 0.950 @ 200Q). So **absorb precision is NOT the gap.**
- Tool used: `scripts/gfx950_disable_absorb_fp4_sitecustomize.py`
  (env `SGLANG_DISABLE_QUARK_ABSORB_FP4=1`), runtime monkeypatch, no source edit.
  Ran on the modified tree `/sgl-workspace/sglang_gfx-1250` (TP2, GPU2,3). Detail: E26.
- Consequence: the gfx1250 **weight-side** absorb emulation
  (`SGLANG_ABSORB_QUANT_EMUL=w_fp4`, `HANDOVER_gfx1250_absorb_quant.md`) is now expected
  neutral too — SKIP it. Only the activation-side a8w4/a4w4 emul would be worth doing, and
  only if the cross-node dump points back at the absorb.
- NOTE (user 2026-07-09): gfx1250 A0 cannot do a4w4 gemm (no fp4-act scaled-WMMA) — the
  "force fp4 absorb on gfx1250" test remains infeasible regardless.

## NEXT (now primary): cross-node per-layer dump diff
With MoE (E20), attention (E22), bf16 gemm (E22), and absorb (E26) all ruled out, the
only remaining localizer is diffing the two nodes' per-layer residual stream on the same
fixed prompt. See "Fallback path" below (now the primary next step).

## Fallback path: cross-node per-layer diff
The gap must be a cross-layer/hardware interaction only visible by comparing the two
nodes' **per-layer residual stream** on the SAME fixed prompt.
- **HANDOVER_crossnode_dump.md** — give to the gfx950 node's agent. Contains the
  validated dump hook + launch flags (eager, `--disable-radix-cache`,
  `--skip-server-warmup`) + the byte-identical fixed prompt.
- **scripts/hsdump_sitecustomize.py** — canonical dump hook (post-attention split:
  records per layer `input`, `post_attn` (pre-MoE), `post_layer` (post-MoE)). **Updated
  2026-07-09: rank-gated write** (only rank 0 writes `$HS_DUMP_OUT`, other ranks write
  `.rank{N}`) — the original corrupted the JSON when >1 TP rank wrote the same file.
- **hs_dump_gfx950.json** — gfx950 (TP2) dump **DONE 2026-07-09** (E26 setup): new hook
  (has `post_attn`), run under the **ablation config** (`SGLANG_DISABLE_QUARK_ABSORB_FP4=1`
  = bf16 absorb, so the non-MoE numerics are as close to gfx1250 as possible), eager +
  no-radix + skip-warmup, fixed "Natalia" prompt (greedy, 1 tok -> " First"). 61 layers,
  post_attn all captured; norms grow input 2.3->78, post_layer 2.4->358; rank0==rank1
  (post_layer norm diff 0.0, residual replicated).
- **hs_dump_gfx1250.json** — gfx1250 (TP1) dump, **STALE**: made with the OLD hook (no
  `post_attn` field). **BLOCKER for the diff: must re-dump gfx1250 with the new hook**
  (a gfx1250 box) before comparing; the gfx950 side is ready.
- Diff logic: clean signals are the **dense layers 0-2 (no MoE)** and **`post_attn`** —
  they should match cross-node to the ~1e-2 TP/bf16 floor; a jump = non-MoE bug.
  `post_layer` at MoE layers 3-60 diverges by construction (a4w4 vs a8w4) = control.

### Confounds to remember
1. **MoE differs by construction** (a4w4 vs a8w4) → residual diverges from layer 3 and
   **propagates** downstream. Not a bug. (Why the hook splits `post_attn`.)
2. **expert-routing flips**: top-k argmax on gate logits can select different experts
   from tiny FP diffs → large localized residual change (not a bug).
3. **TP2(gfx950) vs TP1(gfx1250)**: residual replicated so comparable, but adds a ~1e-3
   reduction-order floor compounding to ~1e-2. gfx1250 currently **cannot run TP2**
   (user, 2026-07-09), so this floor is unavoidable — look for divergence clearly above it.
   (A TP2 dump on GPU 2,3 was launched 2026-07-08 but the machine shut down first.)

## FIX PLAN (E35 -> action): HANDOVER_kernel_fixes.md
Both gfx1250 kernels need fixing. **FIX A (attention)**: force fp32 `tl.dot` in
`decode_attention.py::_fwd_grouped_kernel_stage1` (QK + PV) and `extend_attention.py` prefix
loop; cheap A/B first via `--triton-attention-num-kv-splits 1`; `triton_attention_reduce_in_fp32`
is a DEAD flag (wire it or hardcode fp32). **FIX B (MoE)**: token>16 auto-switches to contiguous-M
(DeepGEMM) with a compile-`max_m` vs runtime-`contiguous_m` mismatch — isolate first with
`AITER_GROUPED_CONTIGUOUS_TOKEN_THRESHOLD=99999` (scheduler bug vs numerics), then fix compile-M
alignment or carry fp32 through stages (`out_dtype="f32"` is supported but wrapper hardcodes bf16).
Validate each fix END-TO-END with the OTHER component idealized (attn-fix + MoE-emul -> ~0.925;
MoE-fix + torch-attn -> ~0.925). See HANDOVER_kernel_fixes.md.

## Artifacts in this skill dir
- `SKILL.md`, `EXPERIMENT_LOG.md` (E0-E24), `CHANGES.md`, `gfx1250.md` — the playbook.
- `HANDOVER_crossnode_dump.md` — cross-node dump instructions + hook.
- `hs_dump_gfx1250.json` — gfx1250 TP1 per-layer dump (fixed prompt) — STALE (no post_attn).
- `hs_dump_gfx950.json` — gfx950 TP2 per-layer dump, new hook + ablation config (2026-07-09).
- `crossnode_dump_compare.png` — E28 chart: gfx950-vs-gfx1250 per-layer rel_l2 + norms.
- `scripts/plot_crossnode_dump.py` — regenerates the E28 chart from the two dumps.
- `scripts/matmul_prec.py` — bf16-vs-fp32-vs-fp64 matmul precision microbench (E41: gfx1250 bf16
  ~0.16-0.28% off, fp32 ~1e-6).
- `HANDOVER_gfx950_bf16_microbench.md` — **DONE (E42, gfx950 smci355-ccs-aus-m12-33):** gfx950 bf16
  matmul = **IDENTICAL** to gfx1250 (bf16 ~1.65e-3, fp32 ~1e-7). => gfx950 bf16 is NOT more accurate;
  **E41's "gfx950 bf16 accurate / gfx1250 uniquely lossy" is REFUTED.** The torch/hipblas 1.65e-3 is
  bf16 **output-rounding** (arch-independent), not accumulation. FIX A still works (E37), but its
  benefit is a **kernel-level dtype choice** (gfx1250 triton MLA downcasts P·V to bf16 where gfx950
  keeps fp32), NOT the matmul unit's raw precision. Confirm by microbenching the ACTUAL attention
  kernel per arch, not the torch proxy.
- `scripts/moe_quant_probe.py` — MoE a4w4-vs-a8w4 quant-error probe (E20).
- `scripts/moe_emul_sitecustomize.py` — bf16 MoE emulation (SGLANG_MOE_EMUL=a4w4|a8w4|a16w4).
  **CHUNKED version (2026-07-09, node H21-18)** — memory-safe on TP1; use on BOTH nodes.
- `HANDOVER_gfx950_moe_emul_apples.md` — **NEXT**: run this chunked emul on gfx950 to confirm
  it still gives ~0.925 (rules out a chunked-emul bug), making E31's "gap is non-MoE" airtight.
- `HANDOVER_gfx1250_moe_emul.md` — **NEXT STEP**: run the E30 emul ON gfx1250 to decide
  flydsl-kernel-bug (case A, emul~0.925>>0.85) vs deeper gfx1250 issue (case B). No new kernel.
- `HANDOVER_moe_emul_method.md` — self-contained METHOD/implementation guide for the bf16 MoE
  emulation (so an agent on gfx1250 can re-implement it against whatever code/commit is there).
- `scripts/attn_probe_sitecustomize.py` — attention decode/prefill torch-fp32 hook (E22).
- `scripts/hsdump_sitecustomize.py` — cross-node per-layer dump hook (post-attn split).
- `scripts/gfx950_disable_absorb_fp4_sitecustomize.py` — gfx950 absorb→bf16 ablation (E25).
- `HANDOVER_gfx950_ablation.md` — gfx950 single-node A/B instructions (preferred next test).
- `scripts/gfx1250_absorb_quant_emul_sitecustomize.py` — gfx1250 bf16-EMULATE quantized
  absorb (w_fp4 ready; a4w4/a8w4 activation via doc). No fp4 kernel (safe on A0).
- `HANDOVER_gfx1250_absorb_quant.md` — gfx1250 emulation instructions + activation skeleton.
- `scripts/gfx1250_disable_moe_scale_shuffle_sitecustomize.py` — disable gfx1250 B-scale
  n32k4 shuffle (E25b: with `SGLANG_MOE_SHUFFLE_GFX1250=0` = both-raw consistency test).

## Machine-switch checklist
1. Confirm GPUs: `ls /dev/kfd /dev/dri && python3 -c "import torch;print(torch.cuda.device_count())"`.
2. Re-apply code changes #2 (weight shuffle) + #3 (AITER_GROUPED_FORCE_SPLIT_K1) if missing;
   **do NOT apply #1 (bisect/bit_length) — it crashes DSv4, see E38**; `rm -rf /root/.flydsl/cache`.
3. Point `run_ds-r1.sh --model-path` at the checkpoint; verify it's W4A4 (E2) / matches HF fingerprint (E23).
4. Launch working recipe, sanity curl (coherent, not token-0), GSM8K ~0.85.
5. Next real work = the cross-node dump diff (HANDOVER_crossnode_dump.md).
