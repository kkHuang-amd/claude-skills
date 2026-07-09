# Kernel fix plan — recover gfx1250 R1 GSM8K 0.82 -> ~0.92 (BOTH kernels)

E35 proved BOTH real gfx1250 kernels must be fixed (each masks the other):
| MoE | attention | GSM8K 40Q |
|---|---|---|
| real a8w4 | real triton | ~0.81 |
| ideal emul | real triton | 0.825 |
| real a8w4 | torch naive | 0.825 |
| **ideal emul** | **torch naive** | **0.925** |
So: fix attention -> ~0.825 (MoE still real); fix MoE -> ~0.825 (attn still real); fix BOTH -> ~0.925.
Validate each fix END-TO-END (small-token op-tests are blind, E18/E30/E16).

## Validation protocol (isolate each fix's end-to-end effect)
- **Attention fix alone**: run with the MoE bf16-emul ON (`SGLANG_MOE_EMUL=a16w4`, ideal MoE) so
  attention is the ONLY real kernel. Target ~0.925. (If the fixed attention == torch, this == E35.)
- **MoE fix alone**: run with torch-naive attention ON (`SGLANG_ATTN_TORCH_DECODE=1 _EXTEND=1`) so
  MoE is the only real kernel. Target ~0.925.
- **Both fixes, no emul/torch**: the real deliverable — target ~0.925 at full speed.
References: torch-naive attention = `scripts`-style hook (E34); bf16 MoE = `scripts/moe_emul_sitecustomize.py` (E31).

---

## FIX A — Triton MLA attention: force fp32 (decode + extend)
Root causes (explore, `python/sglang/srt/layers/attention/triton_ops/`):
- `decode_attention.py::_fwd_grouped_kernel_stage1`: QK dot in bf16 — `q_k = q.to(K_Buffer.dtype)`
  then `qk = tl.dot(q_k, k)` (no out_dtype), `qk += tl.dot(qpe, kpe.to(qpe.dtype))` (~L447-490);
  PV dot `acc += tl.dot(p.to(v.dtype), v)` (~L523).
- 16-way split-KV (`triton_attention_num_kv_splits=16` forced on HIP, `server_args.py:4539`) chains
  16 LSE merges of bf16-derived partials.
- `triton_attention_reduce_in_fp32` is DEAD (defined `server_args.py:1466`, referenced nowhere).
- `extend_attention.py::_fwd_kernel`: prefix loop bf16 QK (`tl.dot(q.to(k.dtype),k)` ~L424-482),
  extend-local fp32 QK (`out_dtype=tl.float32` ~L546-587) — asymmetric; PV bf16 in both.

Steps (low-risk first):
1. **Cheap A/B (no code):** `--triton-attention-num-kv-splits 1` (and try 4). Fewer merges -> should
   move decode toward torch. Measure with MoE-emul ON. Quick signal that split-count matters.
2. **Force fp32 dots in decode stage1** (`_fwd_grouped_kernel_stage1`):
   - `qk = tl.dot(q_k, k, out_dtype=tl.float32)`; `qk += tl.dot(qpe, kpe, out_dtype=tl.float32)`
     (drop the `.to(...)` casts; keep q/k loads as-is, out_dtype forces fp32 accumulate).
   - `acc += tl.dot(p, v, out_dtype=tl.float32)` (keep `p` fp32; V may stay bf16 input, acc fp32).
3. **Extend parity** (`_fwd_kernel` prefix loop): `tl.dot(q, k, out_dtype=tl.float32)` and fp32 PV,
   matching the extend-local tile. (Also `_fwd_kernel_unified` if deterministic mode is used.)
4. **(Optional, cleaner) wire `triton_attention_reduce_in_fp32`**: plumb from TritonAttnBackend into
   the kernels as a `constexpr` that toggles the out_dtype casts (so it is a real knob, default on gfx1250).
5. Validate: with MoE-emul ON, expect ~0.925; compare kernel-vs-torch rel_l2 (E22 hook) -> should
   drop from 0.16% toward ~0. Watch perf (fp32 dots are slower; measure tok/s).
Risk: low. fp32 `tl.dot` is well-supported; only perf cost. gfx1250 uses generic `_is_hip` blocks
(no arch-special code), so this is portable.

---

## FIX B — flydsl a8w4 grouped MoE: token-dependent error (E16)
Findings (explore, `aiter/aiter/ops/flydsl/`):
- WMMA accumulator IS fp32; per-1x32 e8m0 scales applied inside the 16x16x128 `wmma_scale_*`
  (hardware, correct). So it is NOT a raw-accumulate precision bug.
- Strongest token correlate: **auto switch to contiguous-M (DeepGEMM) at `token_num > 16`**
  (`grouped_moe_gfx1250.py:518-529`, default threshold 16). At 8 tok = per-expert masked_m; at 512
  = contiguous flat + `m_tile_map` psum + expert bisect.
- **Compile `max_m` vs runtime `contiguous_m` mismatch**: e.g. 512 tok -> kernel compiled M=4096 but
  runtime `contiguous_m` ~12032 (`grouped_moe_gfx1250.py:551-558, 655-658`; descriptors use runtime
  `m_idx`, `gemm_mxscale_gfx1250.py:713-715`). Prime suspect for large-M scale/tile indexing drift.
- 3 bf16 truncation points (stage1 out, stage2-in re-quant to MXFP8, stage2 out) + SwiGLU via
  exp2/rcp approx (token-independent). `out_dtype="f32"` is supported (`gemm_mxscale_gfx1250.py:116`)
  but MoE wrapper hardcodes `"bf16"`.

Steps (ISOLATION FIRST — decide scheduling-bug vs numerics):
1. **Threshold isolation:** run large-token (or GSM8K) with
   `AITER_GROUPED_CONTIGUOUS_TOKEN_THRESHOLD=99999` (force per-expert masked_m even at large token),
   validate with torch-naive attention ON.
   - If GSM8K -> ~0.925: the **contiguous-M scheduler/indexing is the bug** (go to 2).
   - If still ~0.82: it is **numerics** (go to 3). (Note: per-expert mode at 512 tok may be slow/mem-heavy.)
2. **If scheduler:** fix the compile-`M` vs runtime-`contiguous_m` inconsistency — compile the kernel
   with `max_m = contiguous_m` (or a consistent padded UB used for routing + scales + JIT key), and
   re-check the psum/bisect (`gemm_mxscale_gfx1250.py:3008-3073`) + route-indexed quant/scatter +
   `make_desc_as` scale-row bounds for large `contiguous_m`. Use `AITER_GROUPED_DEBUG=1` /
   dump hooks (`AITER_GROUPED_DUMP_A2`, per-stage) to localize where error appears (after stage1 gemm
   / after re-quant / after stage2).
3. **If numerics:** carry fp32 through the pipeline — request `out_dtype="f32"` for stage1+stage2,
   keep fp32 through SwiGLU before the MXFP8 re-quant, bf16-cast only at the final gather-reduce
   (`moe_gather_reduce.py`, already fp32-sum). Reduces the 3 trunc points to 1.
4. Do NOT enable CSV `split_k=2` (stage2 reduction + raw atomic path are broken; production forces
   split_k=1 — keep `AITER_GROUPED_FORCE_SPLIT_K1=1`).
5. Validate: with torch-naive attention ON, expect ~0.925; and the E16 token-swept logits_diff vs the
   bf16 emul should stop growing with token count.
Risk: medium-high. The contiguous-M scheduler is intricate (the bisect off-by-one, E17/E25c, lived
here). Prefer the isolation test (step 1) before editing the scheduler.

---

## Suggested order
1. FIX A (attention fp32) — low risk, quick; validate with MoE-emul -> expect ~0.925.
2. FIX B step 1 (threshold isolation) — cheap, decides the MoE root cause.
3. FIX B step 2 or 3 accordingly.
4. Final: both fixes, no emul/torch, full-speed GSM8K 200Q/1319Q -> target ~0.92-0.93.

## Interim (if a correct-but-slower serve is needed now)
torch-naive attention + bf16 MoE emul already gives 0.925 (E35) but ~2.4 tok/s — not for production,
only as the accuracy ceiling / reference while the kernels are fixed.
