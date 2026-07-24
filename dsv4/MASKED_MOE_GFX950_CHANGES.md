# Masked (deep-gemm-style) MoE on gfx950 — code changes & rollback record

Date: 2026-07-16. This records the DSV4 mori-EP **masked MoE** feature built in
`aiter` + `sglang`, and how to roll it back / restore it. Result achieved:
gsm8k 0.92, cuda-graph-safe, decode 2157 tok/s = 1.12x default (see
MORI_EP_DECODE_ROOTCAUSE.md for the perf story). This doc = the *code* record.

Backup (full restore material): `masked_moe_rollback/`
- `grouped_moe_gfx950.py`            (new file, 934 lines — the core primitive)
- `test_flydsl_masked_moe_stage1_gfx950.py` (new unit test)
- `masked_aiter_tracked.patch`       (git diff of the 4 tracked aiter files)
- `sglang_moe_runner_aiter_FULL.patch` (git diff of sglang moe_runner/aiter.py;
                                         contains BOTH the masked branch AND the
                                         prefix-slice — prefix-slice is KEPT)

## What the feature is
Token-major mori recv `[M,K]` -> route/scatter into expert-major `[E,max_m,K]` ->
masked stage1/stage2 GEMMs (only `masked_m[e]` live rows) -> weighted gather-reduce
combine -> `[M,model_dim]`. Mirrors the shipping gfx1250 grouped MoE design.

## Files & changes

### NEW: `aiter/ops/flydsl/grouped_moe_gfx950.py` (the core, untracked)
Key symbols:
- `_mxfp4_dequant_bf16_graphsafe` (L35): graph-safe mxfp4->bf16 (cached LUT + bit-op
  e8m0, avoids fp4_utils' per-call `torch.tensor(list,device=)` + bool-mask assign).
- `_quant_per1x32_fp8` (L66): self-contained graph-safe per-1x32 fp8 quant
  (bit-ops + torch.where; no fp4_utils bool-mask-assign).
- `flydsl_masked_moe_gfx950_recv` (L200): the sglang bridge primitive. graph-safe
  routing (SGLANG_MORI_MASKED_ROUTE=ondevice: build_moe_route_maps_guarded; else
  torch one_hot). a1 dtype fp4(a4w4, direct)/fp8(a8w4). a2 fused into stage1 (opt).
  combine: gather_reduce (default) OR fused (SGLANG_MORI_MASKED_FUSE_COMBINE, DEAD-END).
- `flydsl_masked_moe_stage1` (L671): masked stage1; out_dtype fp4/fp8 -> fused a2
  quant in epilogue.
- `flydsl_masked_moe_stage2` (L825): masked stage2; combine= flag (dead-end path).
- `flydsl_masked_moe_gfx950` (L104) + `_maybe_grouped_gfx950_masked_moe` (L550):
  full-route entry (used by the aiter.fused_moe hook).

### MODIFIED (tracked) `aiter/fused_moe.py` (+39)
Added the opt-in hook `_maybe_grouped_gfx950_masked_moe(...)` call (gated by
`AITER_GFX950_MASKED_MOE`; returns None/no-op by default).

### MODIFIED (tracked) `aiter/ops/flydsl/kernels/moe_route_maps.py` (+85)
Added `build_moe_route_maps_guarded_module()` (L99): bounds-guarded atomic route
kernel (skips e>=experts / slot>=max_m; the unguarded one OOB-faults under
cuda-graph).

### MODIFIED (tracked) `aiter/ops/flydsl/grouped_moe_gfx1250.py` (+11)
Added `_get_compiled_route_maps_guarded()` (L1162) wrapper for the above.

### MODIFIED (tracked) `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py` (+327/-69)
All guarded by `grouped_masked_m`:
- stage1: `out_dtype="fp4"/"fp8"` -> emit fused-quantized a2 + tiled e8m0 scale in
  the epilogue (masked-aware).
- **scale-write bug fix**: fused-quant scale used the LOCAL tile row; changed to the
  masked GLOBAL row `expert_idx*max_m + row` (else experts collide). (Correct.)
- combine-fusion (masked_combine): relaxed the `accumulate=False` assert, added
  row2tok_rsrc, token-major precompute_row, buffer-atomic. **DEAD-END** (e_vec=2
  masked-accumulate acc-layout bug -> per-tile-row scalar; opt-in OFF).

### MODIFIED (tracked) `sglang .../moe_runner/aiter.py`
- `_maybe_run_mori_masked` (+ its call in `AiterRunnerCore.run`): the masked
  integration (gate SGLANG_MORI_MASKED_MOE, default off; gate_mode=interleave;
  device-tensor base/total_recv; graph-safe). ALL MINE (not in HEAD).
- **prefix-slice** block (SGLANG_MORI_DECODE_PREFIX_SLICE): SEPARATE feature, KEPT.

### Config: `run_sgl_dsv4_masked.sh` (non-git useful-scripts)
SGLANG_MASKED_CUDA_GRAPH toggle, SGLANG_MORI_MASKED_A2_DTYPE=fp8 default, etc.
(env-only; inert once masked is rolled back.)

## Rollback performed (2026-07-16)
- aiter: `rm grouped_moe_gfx950.py` + test; `git checkout --` the 4 tracked files
  (fused_moe.py, grouped_moe_gfx1250.py, mixed_moe_gemm_2stage.py, moe_route_maps.py).
- sglang: surgically removed `_maybe_run_mori_masked` (def + call) from
  moe_runner/aiter.py; **KEPT the prefix-slice block**.
- KEPT: prefix-slice (separate active feature); MORI_EP_DECODE_ROOTCAUSE.md;
  masked_moe_rollback/ backup.

## How to RESTORE the masked feature
1. `cp masked_moe_rollback/grouped_moe_gfx950.py /sgl-workspace/aiter/aiter/ops/flydsl/`
2. `cp masked_moe_rollback/test_flydsl_masked_moe_stage1_gfx950.py /sgl-workspace/aiter/op_tests/`
3. `cd /sgl-workspace/aiter && git apply masked_moe_rollback/masked_aiter_tracked.patch`
4. Re-add `_maybe_run_mori_masked` to sglang moe_runner/aiter.py (from
   `sglang_moe_runner_aiter_FULL.patch`, the masked-branch hunks only; prefix-slice
   is already present).
5. Enable: SGLANG_MORI_MASKED_MOE=1 (+ A1_DTYPE=fp4, A2_DTYPE=fp4,
   ROUTE=ondevice, SGLANG_MASKED_CUDA_GRAPH=1).
