# Per-node results — gfx1250 @ H21-18 (SECOND gfx1250 node)

Raw run log for the H21-18 gfx1250 box only. Write new runs here first, then promote a short
summary into the shared `EXPERIMENT_LOG.md` / `STATUS.md`.

This node's shared-log entries so far: **E32-H21** (bf16 MoE emul on gfx1250 — gap not the MoE;
same run summarized as E32), and the **E39 BREAKTHROUGH** combo (idealize MoE AND attention:
only both -> 0.925; each alone -> 0.825). See `EXPERIMENT_LOG.md`.

---

## (append new H21-18 runs below)

## 2026-07-10 — [node: H21-18 gfx1250] fp8 KV cache FIXED (overturns SKILL §4.7 / E13)

New docker image `henryx/xsgl:v0.5.14-gfx1250-rocm-nightlies-20260709-trial-3`
(sglang `3923a34d`, aiter `9af05b91`). Model `/shared_nfs/huggingface_models/amd/DeepSeek-R1-0528-MXFP4`
(W4A4 confirmed). Production fixes (FIX A fp32 P·V, gfx1250 weight shuffle,
AITER_GROUPED_FORCE_SPLIT_K1) are already baked into this image.

Baseline re-validation (bf16 KV, `--kv-cache-dtype auto`, cuda-graph, radix on):
GSM8K 40Q 0.950 / 200Q 0.960 / **1319Q 0.951** (invalid 0.000, ~160-180 tok/s). Reproduces prior.

### fp8 KV was broken, now FIXED — root cause = query downcast to fp8 (NOT the fp8 gemm)
- `--kv-cache-dtype fp8_e4m3` (unchanged recipe) originally: GSM8K = **0.000 / invalid 1.000**,
  decode degenerates to empty/whitespace. Confirms SKILL §4.7 / E13 on this image.
- ROOT CAUSE (two spots, same bug): the triton MLA attention kernels **downcast the query
  to fp8** when the KV cache is fp8 — `q.to(K_Buffer.dtype)` — then do an fp8×fp8 dot. MLA's
  q (post-absorb) needs bf16 precision; forcing it to fp8 e4m3 (3-bit mantissa) destroys the
  qk logits -> softmax picks garbage -> degenerate decode. Two code paths:
  1. `decode_attention.py::_fwd_grouped_kernel_stage1` `q_k = q.to(K_Buffer.dtype.element_ty)`
     (decode reading accumulated fp8 KV).
  2. `extend_attention.py` **prefix loops** (2 kernels × {nope, rope} = 4 dots):
     `tl.dot(q.to(k.dtype), k)` + `tl.dot(qpe.to(kpe.dtype), kpe)`. This path only fires when
     **radix cache reuses a shared prefix** (prefill reads cached fp8 KV via the extend kernel) —
     which is why SKILL §4.7's "prefill never reads the fp8 cache" was a blind spot: WITH radix
     cache, prefill DOES read it.
- FIX (both files, no-op for bf16 KV): keep q in its native dtype and **upcast the fp8 K to q's
  dtype** in the dot instead — `tl.dot(q, k.to(q.dtype))` / `tl.dot(qpe, kpe.to(qpe.dtype))`.

### fp8 gemm on gfx1250 is NOT broken (settles the "is gfx1250 fp8 gemm abnormal?" question)
Single-tile triton microbench on gfx1250: fp8×fp8 tl.dot vs exact-fp8-value fp64 ref =
**1.1e-7** (bf16×bf16 = 2.3e-3). The fp8 tensor-core dot is correct. The bug is purely the
**q→fp8 downcast of an operand** feeding the dot, not the matmul. So "fp8 gemm fine" and
"must upcast K to bf16" are NOT contradictory: we upcast so the QUERY is never degraded to fp8;
KV stays stored in fp8 (memory saving preserved), only the dot runs in bf16.
Decode-kernel offline harness (real sglang kernel, synthetic MLA inputs) is numerically correct
for fp8 at seq up to 2048, bs up to 32, splits up to 16 — the kernel math was never the issue.

### WHY gfx950 (same code) survives but gfx1250 collapses — TRIGGER confirmed, exact mechanism NOT pinned
Facts (verified):
- Kernels are **identical code** on both archs: `git show 000a61a2:...extend_attention.py` has the
  SAME `q.to(k.dtype)` at lines 424/444/999/1019 as current 3923a34d. So this is NOT a version diff
  and NOT an arch-gated path (my earlier "gfx950 uses a different path" claim was WRONG, retracted).
- The `q.to(fp8)` downcast IS the trigger: removing it (upcast K instead) fixes gfx1250 0.000->0.949.
- gfx1250-specific: gfx950 runs the SAME downcast code and scores 0.941 (results_gfx950 smci355).

Two natural mechanisms were TESTED and BOTH REFUTED at the real operating point (do not repeat):
1. "overflow -> NaN": gfx1250 `bf16->fp8_e4m3fn` cast DOES map >448 to NaN (measured: 449->448 ok,
   500/1000->NaN), BUT the REAL query never reaches 448 — measured on live R1+GSM8K (probe on
   extend/decode wrappers, radix prefix reuse firing): **|q|max ~276 (extend) / ~330 (decode),
   over448 count = 0**. So overflow->NaN is NOT the trigger for normal prompts. (Earlier "overflow
   is the mechanism" claim RETRACTED.)
2. "fp8 tensor-core dot wrong at large magnitude": single-tile gfx1250 microbench with q-magnitude
   matched to real (|a|max=332): fp8xfp8 dot vs exact-fp8 ref = **3e-7 (correct)**; only |a|>448
   (qscale 300, |a|max=992) NaNs. So the fp8 matmul is correct at the real operating point.

=> CONFIRMED: fp8 gemm on gfx1250 is fine (1e-7 small, 3e-7 at mag~330); the trigger is the
`q.to(fp8)` downcast; it is gfx1250-specific. NOT pinned: the exact low-level reason gfx1250
collapses (0.000) while gfx950 tolerates the identical downcast (0.941) when q<448 and the isolated
fp8 dot is correct.

CROSS-ARCH CONTROL COMPLETE (gfx950 node smci355, results_gfx950-smci355 tests 1/2/3, 2026-07-10):
- Test 1 (bf16->fp8 cast overflow): gfx950 == gfx1250 EXACTLY (449->448 round, 500/1000/5000->NaN,
  threshold ~464). fp8 overflow handling is NOT the arch difference.
- Test 2 (triton fp8xfp8 dot): gfx950 correct to |a|max=320 (rel_l2 1.3e-5; gfx1250 3e-7 — gfx950
  slightly LESS precise, opposite of "gfx1250 broken"); qscale 300 NaN on BOTH from the cast, not
  the dot. fp8 dot semantics are the SAME on both archs.
- Test 3 (real kernel, unmodified q->fp8, fp8 KV + radix + cuda-graph): gfx950 = 1319Q 0.941 (good).
### ROOT CAUSE PINNED (2026-07-10): gfx1250 triton tl.dot(fp8,fp8) is broken for contraction K>=128
Decisive experiments on gfx1250:
- A/B in the REAL decode kernel (constexpr toggle `DOWNCAST_Q`, host-side env, threaded into
  `_fwd_grouped_kernel_stage1`), realistic q (|q|max<=330, NO cast overflow), vs torch ref on same
  fp8 values: UPCAST-K (fix) = rel_l2 ~1e-3 correct at splits 1/2/4; **DOWNCAST-Q (fp8xfp8 dot) =
  NaN at EVERY config, including qmax=30 and splits=1** (so NOT overflow, NOT multi-split).
- Single-tile microbench, fp8xfp8 tl.dot, small in-range values, K-sweep on gfx1250:
    K=64  -> rel_l2 0.0    (correct)
    K=128 -> rel_l2 3.2e34 (GARBAGE)
    K=256 -> rel_l2 4.9e36 (GARBAGE)
    K=512 -> rel_l2 3.1e36 (GARBAGE)
  bf16xbf16 at K=512 -> 0.0 (correct). AMD kargs (matrix_instr_nonkdim=16/kpack=2/num_stages=1)
  make no difference — garbage with or without them.
=> On gfx1250, **triton `tl.dot(fp8, fp8)` returns garbage (~1e34+) once the contraction dim
K >= 128** (K=64 is fine; bf16 is fine at all K). The MLA nope QK dot has K=512, so downcasting q
to fp8 and doing fp8xfp8 there yields garbage -> softmax(inf) -> NaN -> degenerate decode. This is
the single root cause of BOTH bug#1 (decode) and bug#2 (extend prefix); the "overflow->NaN" and
"multi-split" leads were both red herrings.
Why gfx950 is unaffected: same kernel, but gfx950's fp8 tl.dot at K=512 is correct (its real MLA
kernel with the identical q->fp8 downcast scores 0.941). So this is a **gfx1250-specific triton/ROCm
fp8-MFMA codegen bug for large-K fp8 matmul**, not the fp8 ISA cast (identical on both) and not a
model/recipe issue.
FIX = never feed an fp8xfp8 dot with K>=128 on gfx1250: keep q bf16 and upcast the fp8 K to bf16 in
decode_attention.py + extend_attention.py (no-op for bf16 KV). Validated E2E: fp8 KV 1319Q 0.949.

### gfx1250 DEV RULE (for future kernels)
On gfx1250, do NOT use `tl.dot(a_fp8, b_fp8)` when the contraction dimension K >= 128 — it silently
returns ~1e34 garbage (no error, no NaN at the dot; NaN only appears downstream). Mitigations:
upcast one/both operands to bf16 before the dot (K-dim bf16 accumulation is correct), or tile K into
<=64 chunks, or use a validated scaled-fp8 GEMM (e.g. the aiter/flydsl a8w4 path). bf16 tl.dot is
unaffected. Suspected triton/ROCm fp8-MFMA codegen defect for large K on gfx1250. Minimal upstream
repro + bug report: `artifacts/triton_fp8_dot_largek_gfx1250_repro.py` +
`artifacts/BUGREPORT_triton_fp8_dot_largek_gfx1250.md` (verified triton 3.7.1: K=64 ok, K>=128 garbage).

### Is FIX A (fp32 P·V, E36/E37) also a gfx1250 codegen bug? NO — genuine bf16 precision.
Checked with a triton `tl.dot` P·V microbench (p = softmax rows, v = gaussian), vs fp64, on gfx1250:
| K | bf16xbf16 rel_l2 | fp32xfp32 rel_l2 |
|---|---|---|
| 16 (real BLOCK_N) | 2.6e-3 | 6.3e-8 |
| 64 | 2.3e-3 | 1.4e-7 |
| 128 | 2.4e-3 | 2.2e-7 |
| 256 | 2.5e-3 | 2.9e-7 |
| 512 | 2.4e-3 | 3.9e-7 |
bf16 P·V dot on gfx1250 is **normal bf16 output-rounding (~2.4e-3), STABLE across K, no garbage,
no blow-up** — categorically unlike the fp8 K>=128 defect (~1e34). So FIX A is NOT codegen: the
original code downcast the softmax weights `p` to bf16 (~0.2-0.4%/weight), and that rounding
ACCUMULATES over long CoT generation (0.85 -> 0.925); keeping p in fp32 (FIX A) removes it. It is a
dtype/accumulation-PRECISION choice, whereas the fp8 K>=128 bug is a broken instruction. gfx1250's
bf16 `tl.dot` itself is numerically well-behaved. (Consistent with E42: bf16 matmul rounding ~1.6e-3
is arch-independent; FIX A's benefit is the fp32-vs-bf16 dtype choice, not the matmul unit.)

**⚠️ CAVEAT (RESOLVED via candidate (b)): FIX A is accuracy-neutral on this image — it is NOT the
lever, so there is no gfx950 contradiction.** The bf16-precision story could not explain why gfx950
didn't need FIX A (same kernel, fp32 accumulate, arch-independent bf16 rounding). So I ran the direct
single-variable A/B on gfx1250 (this image): toggle ONLY the P·V p-dtype (fp32 vs bf16) in
decode_attention.py L531 + extend_attention.py L487/L592, nothing else, bf16 KV, production recipe:
| P·V p dtype | 200Q | 1319Q | invalid |
|---|---|---|---|
| fp32 (FIX A ON) | 0.960 | 0.951 | 0.000 |
| **bf16 (FIX A OFF)** | **0.965** | **0.948** | 0.000 |
Reverting FIX A does NOT drop accuracy (0.948 vs 0.951 = noise) — utterly unlike the historical
0.85 (off) -> 0.925 (on) claim. => on THIS stack **p-fp32 is accuracy-neutral; FIX A is not the
lever.** The historical 0.85->0.925 attribution (E36/E37/E39) was a CONFOUND (that saga's
masking/emul setup) or was fixed by the newer sglang(3923a34d)/aiter(9af05b91). This dissolves the
gfx950 contradiction — nobody needs FIX A on this image, so there is no arch puzzle. FIX A left in
place (accuracy-neutral, fp32 P·V safe; droppable for a small perf gain). Candidate (a) — does
gfx950 even downcast p — is now moot for accuracy but a gfx950-agent prompt was written to confirm
the code path. Candidate (c) (systematic bias) is not needed given the A/B is neutral.

A/B carrier note: this A/B was run on **bf16 KV** on purpose. On bf16 KV, `v` is bf16 so FIX-A-off
(`p.to(v.dtype)`) = `p.to(bf16)`, cleanly isolating the single variable "p fp32 vs bf16". It must
NOT be run on fp8 KV: there `v = tl.trans(k)` is fp8, so `p.to(v.dtype)` would downcast p to fp8 and
— because the P·V contraction dim (seq) can be >=128 — trip the fp8 `tl.dot` K>=128 garbage bug,
confounding the test. Corollary: **on fp8 KV the P·V dot MUST stay fp32** (keep p fp32, `v.to(fp32)`)
— not for accuracy but to avoid the fp8 codegen bug; downcasting p to fp8 there would break decode.

### End-to-end validation (fp8 KV, FULL production recipe — radix ON, cuda-graph ON, no flags)
| KV dtype | 40Q | 1319Q | invalid | tok/s |
|---|---|---|---|---|
| fp8_e4m3 (before fix) | 0.000 | — | 1.000 | — |
| **fp8_e4m3 (after fix)** | **0.950** | **0.949** | 0.000 | **182** |
| bf16 (`auto`, ref) | 0.950 | 0.951 | 0.000 | 178 |

=> fp8 KV is now accuracy-neutral vs bf16 AND full speed, no `--disable-radix-cache` / no
`num_kv_splits=1` workaround needed. Halves KV-cache memory on gfx1250.

RED HERRING logged for honesty: initially mis-attributed the residual break to the
multi-split (num_kv_splits>1) path, because the validation script that "worked" (0.947) also
carried `--disable-radix-cache`. Clean single-variable test (radix OFF + splits DEFAULT = 0.900;
splits=1 + radix ON = 0.000) showed **radix cache** was the real trigger -> extend prefix fp8
read. The num_kv_splits=1 backend gate was reverted.

Files changed (both no-op for bf16):
- `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`
- `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
