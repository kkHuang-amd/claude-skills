---
name: flydsl-kernel-authoring
description: Repeatable playbook for authoring or modifying a FlyDSL GEMM/MoE kernel on gfx950 (aiter/ops/flydsl/kernels/*) — especially deep fusions that touch the MFMA accumulate / c_shuffle epilogue (masked stage GEMMs, fused a2 quant, fused combine). Use this BEFORE editing any flydsl kernel so debugging is observable + fast + validated against a true reference, instead of blind IR iteration. Encodes the concrete traps from the 2026-07 masked-MoE work (self-referential oracle -> gsm8k=0 agreed with a wrong ref; e_vec=2 grouped_masked_m combine-fusion acc-layout dead-end).
---

# FlyDSL kernel authoring / modification playbook (gfx950)

The masked-MoE effort (see `MORI_EP_DECODE_ROOTCAUSE.md`, `MASKED_MOE_GFX950_CHANGES.md`)
proved two things about touching FlyDSL kernels:

1. **The Python orchestration layer is tractable** (routing, scatter, dequant/
   requant glue, gather_reduce combine) — we wrote `grouped_moe_gfx950.py` from
   scratch and it hit gsm8k 0.92–0.93 = baseline.
2. **Deep fusion into the MFMA GEMM epilogue is where we got stuck** — the
   combine-fusion e_vec=2 `grouped_masked_m` acc-layout bug was "beyond reliable
   blind IR iteration" because we had no acc-register observability and (earlier)
   no *true* reference.

This playbook is the setup that makes #2 tractable. Do these steps **in order**;
do not start editing the kernel until the harness + reference (steps 1–3) exist.

Kernels in scope live in `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/`:
`mixed_moe_gemm_2stage.py` (the 2-stage MFMA GEMM + epilogue — the hard part),
`moe_route_maps.py`, and the callers `grouped_moe_gfx1250.py` (shipping ref) /
`grouped_moe_gfx950.py` (our masked port; in `masked_moe_rollback/`).

---

## Rule 0 — decide which layer you are in (and whether to stop)

| Layer | Example | Who should do it |
|---|---|---|
| Integration / orchestration | sglang hooks, weight requant, RSF, env gates, routing glue, gather_reduce wiring | **Me (agent).** Tractable; use the precision-parity checklist (Rule 5). |
| MFMA kernel body / epilogue | `c_shuffle_epilog`, `write_row_to_lds`/`store_pair`, acc-fragment→store mapping, `e_vec`, tile geometry | **Only with steps 1–4 below.** If a fix needs changing MFMA register indexing and the micro-harness (step 3) still can't localize it → **STOP and hand to the FlyDSL/aiter kernel owner** (this is exactly the C6 dead-end). |

Blind IR iteration on the MFMA epilogue without steps 1–4 is the failure mode.
Don't repeat it.

---

## Step 1 — establish kernel-level observability FIRST

You cannot fix an acc-layout bug you cannot see. Before editing, stand up ONE of:

- **Device-side value dump**: a debug flag in the kernel that writes selected
  `(tile, row, col)` intermediates (and, for the epilogue, the acc fragment as it
  is read by `store_pair`) to a scratch HBM buffer the host test reads back.
- **`printf`-style trace** if the FlyDSL/HIP toolchain build in this container
  supports it (verify once: does a `printf` in a compiled flydsl kernel actually
  emit? record the answer here).

Concretely, the symptom that stumped us was `out[r] = per-row-varying scale *
s2[r]` (token0 ~0.03, per-row ratio mean 0.11, range 0.00–0.33). With an acc-
fragment dump keyed by MFMA lane/row this is a one-shot diagnosis; without it, it
is unbounded guessing. **If you cannot get either dump working, that itself is the
signal to escalate to the kernel owner** — do not proceed to blind edits.

TODO (fill in once verified in this env): exact flag name + how the dump buffer is
plumbed + whether HIP `printf` works from a flydsl kernel.

---

## Step 2 — build a TRUE reference (never a self-referential oracle)

This is the trap that made the earlier masked garbage read as "correct" (the
in-branch REALDIFF ref was *also* a4w4-separated, so a wrong kernel agreed with a
wrong ref at 0.02). Hard rules:

- The reference MUST come from an **independent correct source**, i.e. the
  **shipping precision path** (default `fused_moe`: `a8w4` = fp8 activation +
  `gate_mode=interleave` (GUGU), per-1x32 scales) — NOT the dtype/gate_mode your
  new kernel happens to use.
- A unit test whose torch reference dequantizes **the same quantized tensors the
  kernel consumes** validates *kernel mechanics only*, NOT precision-path parity.
  That is fine for step 3 (mechanics) but is NOT sufficient to claim correctness —
  see `op_tests/test_flydsl_masked_moe_stage1_gfx950.py` (`_ref_stage1_grouped`
  dequants the kernel's own inputs; good for mechanics, blind to path mismatch).
- Always also diff against the **known-good sibling path** as a second oracle:
  the non-masked/sorted accumulate store (e_vec=8) that already works, and the
  shipping `grouped_moe_gfx1250.py` design. If your masked path disagrees with
  BOTH the shipping precision path and the sorted-layout sibling, the bug is
  yours, not the reference's.

---

## Step 3 — iterate on an isolated micro-harness, not e2e

- Use / extend `op_tests/test_flydsl_masked_moe_stage1_gfx950.py` as the template:
  tiny shapes, seconds per run, `_logits_diff` with an explicit tol
  (`LOGITS_DIFF_TOL = 0.01`; landed masked unit was 4e-6–5.9e-4).
- One kernel concern per harness (stage1 recv, stage1 GEMM, a2 fused-quant,
  stage2 GEMM, combine) so a failure localizes to one epilogue path.
- Keep the repro for any live bug as a standalone script (we used
  `/tmp/dbg_combine.py`: identity map grouped-row r→token r, weight=1, so
  `out[r]` should == grouped `s2[r]` — any deviation is pure acc→store mapping).
- The edit→compile→run→inspect loop stays at the micro-harness level. Only after
  the micro unit passes do you promote to the e2e gate.

The loop:

```bash
cd /sgl-workspace/aiter
# edit aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py
python op_tests/test_flydsl_masked_moe_stage1_gfx950.py --stage <recv|gemm1|a2|gemm2|combine>
# read the printed logits_diff; if regressed, dump acc fragment (step 1) and diff
# vs the sorted-layout sibling before touching anything else.
```

---

## Step 4 — known acc-layout facts (fill this in as you learn; start from these)

The concrete invariants that bit us, kept explicit so the next pass starts warm:

- The **non-masked (sorted) accumulate store works with e_vec=8**; the
  **masked (`grouped_masked_m`) accumulate store with e_vec=2 mis-reads the MFMA
  acc registers** — the acc-fragment→store mapping differs per MFMA row under
  `grouped_masked_m`, so a store path written for the sorted arrangement is wrong.
  The scale-per-row varies → it is the fragment/store *mapping*, not placement,
  not weight, not column coverage (those were all ruled out).
- **Fused-quant scale write must use the masked GLOBAL row** `expert_idx*max_m +
  row`, not the local tile `row` — else every expert collides on `[0,max_m)`
  (this one WE fixed; unit 0.85 → 5.9e-4). Guard with `if const_expr(
  grouped_masked_m)` so the default/sorted path is untouched.
- bf16 `llvm.AtomicRMWOp fadd` is unsupported → use `raw_ptr_buffer_atomic_fadd`
  (buffer-atomic) + an OOB-sentinel byte-offset for invalid rows.
- Every kernel edit must stay **cuda-graph-safe**: no host↔device memcpy, no
  `.item()`, no data-dependent shapes inside the capture region (see the 5
  graph-safety fixes in `MORI_EP_DECODE_ROOTCAUSE.md`).

> Add new invariants here whenever a debug session establishes one. This section
> is the point of the whole playbook — turn hard-won MFMA facts into a checklist.

---

## Step 5 — precision-path parity checklist (the integration-layer traps)

Even a correct kernel loses accuracy if the surrounding conversion is off. These
are the same *class* of bug the FlyDSL MegaMoE PR (sglang #31322) documents, so
check them whenever wiring a flydsl MoE:

- [ ] **Match the shipping precision path**: a8w4 (fp8 act) + `gate_mode=interleave`.
      Wrong dtype (a4w4) ≈ 15% coarse-quant diff; wrong gate_mode (separated) =
      garbage (gate/up halves misread).
- [ ] **Weight re-quant, not byte-reuse**: do NOT feed a checkpoint's fp4 bytes
      straight into `shuffle_weight`. Different MXFP4 encode conventions silently
      corrupt magnitudes (MegaMoE: gsm8k 0.96 → ~0.5). Dequant with OCP decode,
      re-encode with the kernel's OWN quantizer, then shuffle.
- [ ] **Scaling factors applied exactly once**: e.g. `routed_scaling_factor` —
      aiter's `aiter_biased_grouped_topk` folds it into topk weights, so a post-MoE
      multiply double-applies (MegaMoE: gsm8k 0.41 → 0.97). Mirror the reference
      guard `not (should_fuse or _use_aiter)`.
- [ ] **Gate every new branch on both** the backend selector AND the platform/
      feature env, so it can't fire on the wrong platform (the #31322 review flagged
      4 `is_megamoe()` sites missing the `SGLANG_AMD_USE_FLYDSL_MEGA_MOE` gate).

---

## Step 6 — promotion gates (unit → e2e)

1. Micro-harness `logits_diff` < 1e-3 (target ≤ the landed 5.9e-4) on all stages.
2. Standalone repro (step 3) for the specific fused path == its analytic identity.
3. e2e gsm8k on real DSV4 mori-EP decode, **eager AND cuda-graph**, ≥ baseline −
   noise (baseline ~0.93 full-set; use limit=200 for iteration, full n=1319 to
   sign off; z-test vs baseline, not eyeballing).
4. cuda-graph capture = 0 faults.

Do not claim correctness on the unit test alone — it can pass a self-oracle
(Rule/step 2).

---

## Escalation criteria (when to hand to the kernel owner)

Stop and escalate — with the step-1 acc dump + step-3 standalone repro attached —
when ALL of:

- the bug is localized to MFMA register indexing in the epilogue
  (`c_shuffle_epilog` / `write_row_to_lds` / `store_pair`), AND
- the micro-harness + acc dump cannot pin the exact fragment→store stride, AND
- the fix would be blind IR iteration.

This is the documented C6 dead-end. Handing over a *localized* repro (like
`/tmp/dbg_combine.py`) is far more valuable than more blind iteration — and is the
correct division of labor: agent owns integration + parity + harness + validation;
kernel owner owns the co-designed MFMA fusion.
