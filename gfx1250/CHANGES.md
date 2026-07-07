# Code changes & how to re-apply (after docker image switch)

All edits were in `/sgl-workspace/{sglang,aiter}` on the old (now dead) docker.
Copies live in `artifacts/`. Base commits the diffs were taken against:
- sglang `8d30387cd671a3bc8eae178988f7c119544d08b7`
- aiter  `8815f4b56dbaf416a3370659b777839c70ff3bf9`

If the new image is at a different commit, apply by hand (the changes are small
and self-contained) rather than force-applying the patch.

## 1. SGLang — linear a4w4 -> bf16 dequant  (MODIFIED, tracked)

File: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`
- Full patched copy: `artifacts/quark_w4a4_mxfp4.py.patched`
- Diff only:        `artifacts/quark_w4a4_mxfp4.linear-bf16-dequant.diff`

Apply:
```bash
cd /sgl-workspace/sglang
git apply /path/to/gfx1250/artifacts/quark_w4a4_mxfp4.linear-bf16-dequant.diff
# or just copy artifacts/quark_w4a4_mxfp4.py.patched over the file
```
Summary of the edit:
- import `get_bool_env_var`; add
  `_dequant_linear_to_bf16 = _is_hip and get_bool_env_var("AITER_FORCE_A8W4","false")`
- add `_MXFP4_VALUES` LUT + `_dequant_mxfp4_to_bf16(weight,scale)` helper
- `process_weights_after_loading`: if flag -> replace `layer.weight` with bf16
  dense weight, set `layer.weight_scale=None`, `layer.dequantized_bf16=True`
- `apply_weights`: if `getattr(layer,"dequantized_bf16",False)` ->
  `return torch.nn.functional.linear(x if not tuple else x[0], layer.weight, bias)`

## 2. aiter — dense a8w4 gemm (DROPPED 2026-07-07, do NOT re-apply)

- `aiter/aiter/ops/flydsl/gemm_a8w4_gfx1250.py`  <- `artifacts/gemm_a8w4_gfx1250.py`
- `aiter/op_tests/test_gemm_a8w4_gfx1250.py`     <- `artifacts/test_gemm_a8w4_gfx1250.py`

STATUS: **removed / superseded.** The dense flydsl a8w4 kernel launch faults
(§6 of SKILL.md) and was never on any code path (only its own test imported it).
The linear layers use the bf16-dequant path instead (#1 / §4.2 = a16w16 F.linear),
so these two files were removed from the aiter tree and commit. The artifacts are
kept only as a historical starting point if a real dense a8w4 kernel is ever built;
do not copy them back for the working recipe.

## 3. run_ds-r1.sh  (MODIFIED, tracked)

File: `/sgl-workspace/sglang/run_ds-r1.sh`  <- `artifacts/run_ds-r1.sh`
Diff: `artifacts/run_ds-r1.sh.diff`
Key added knobs:
- `ulimit -c 0` (first line) — stop 260 GB GPU coredumps filling disk
- env: `HSA_ENABLE_COREDUMP=0`, `HSA_COREDUMP_PATTERN=/dev/null`, `AMD_COREDUMP=0`
- env: `AITER_FORCE_A8W4=1` — MoE a8w4 + triggers linear bf16-dequant (#1)
- `HIP_VISIBLE_DEVICES` -> a healthy GPU
- model-path `/dockerx/models/amd/DeepSeek-R1-0528-MXFP4`

## 4. No code change needed for MoE dtype switch

MoE a8w4 is purely `AITER_FORCE_A8W4=1` (handled inside `aiter/fused_moe.py`
gfx1250 branch, ~L435). No source edit for the dtype switch itself. (But the
grouped a8w4 kernel then needs #6/#7 below to actually run.)

## 5. SGLang — Triton qk-rmsnorm replacing unbuildable aiter kernel (NEW + MODIFIED)

See SKILL.md §4.4. The aiter `fused_qk_rmsnorm` JIT build fails on this docker
(ck_tile mismatch in `rope_common.h` / `ck_tile/vec_convert.h`).
- NEW file: `python/sglang/srt/models/deepseek_common/attention_forward_methods/triton_qk_rmsnorm.py`
  (Triton `fused_qk_rmsnorm_triton` + torch `fused_qk_rmsnorm_torch` reference).
- MODIFIED: `.../attention_forward_methods/forward_mla.py` — in the `if _use_aiter:`
  block, under `get_bool_env_var("AITER_FORCE_A8W4")` bind `fused_qk_rmsnorm_bf16`
  to the Triton (or, if `SGLANG_QK_RMSNORM_TORCH=1`, the torch) impl instead of
  importing from aiter; also `import get_bool_env_var` from `sglang.srt.utils`.
Decode-only path (triton backend: prefill=MHA, decode=MLA-absorb).

## 6. aiter — FlyDSL stage1 raw gemm missing swiglu arg (MODIFIED)

See SKILL.md §4.5. File `aiter/aiter/ops/flydsl/kernels/moe_grouped_gemm_mxscale_gfx1250.py`.
In the stage1 `launch` (gemm1), the 6 raw `_run_compiled(_get_raw_base()/_get_raw_base_bias(), ...)`
calls omitted the trailing `swiglu_limit` runtime arg that
`launch_mxscale_gemm_masked[_bias]` expects (stage2 already passes it). All 6 end with:
```
                        2 * cfg.inter_dim,
                        stream,
                    )
```
Insert `                        _swiglu_lim_rt,` before the closing `)` in each
(6 occurrences). Fixes `IndexError: tuple index out of range` at `<flydsl-dispatch>`.

## 7. aiter — AITER_GROUPED_FORCE_SPLIT_K1 env to force fused MoE path (MODIFIED)

See SKILL.md §4.6. File `aiter/aiter/ops/flydsl/grouped_moe_gfx1250.py`, right after
the `if cfg_row is not None:` CSV-override block (after `_grouped_dbg("using grouped
CSV config...")`), add:
```python
    if os.environ.get("AITER_GROUPED_FORCE_SPLIT_K1", "0") in _TRUTHY_ENV:
        split_k1 = 1
        split_k2 = 1
        _grouped_dbg("AITER_GROUPED_FORCE_SPLIT_K1: forcing split_k1=split_k2=1")
```
Then set `AITER_GROUPED_FORCE_SPLIT_K1=1`. Routes stage1/stage2 to the fused
(split_k=1) path, avoiding the raw+finalize+split-k-reduce GPU illegal-address.

## 8. Run knobs (MODIFIED, tracked) — updated working recipe

`/sgl-workspace/sglang/run_ds-r1.sh` (or its diag variant):
- `AITER_FORCE_A8W4=1`, `AITER_GROUPED_FORCE_SPLIT_K1=1`
- `--kv-cache-dtype auto`  (bf16; **NOT** fp8_e4m3 — fp8 KV breaks decode, §4.7)
- `--attention-backend triton`
- CUDA graph ON for real serving (drop `--disable-cuda-graph`); only add
  `--disable-cuda-graph` + `AMD_SERIALIZE_KERNEL=3` when diagnosing a new async fault.
- `ulimit -c 0` etc. (coredump trap, §7)

---

## Quick bring-up checklist on the new docker

```bash
# 0. GPUs present?
ls /dev/kfd /dev/dri && python3 -c "import torch;print(torch.cuda.device_count())"
# 1. apply changes #1 (linear bf16-dequant), #5 (triton qk-rmsnorm), #6 (swiglu arg), #7 (force split_k1)
# 2. sanity: model is still W4A4 (see EXPERIMENT_LOG E2)
# 3. WORKING launch (cuda graph on, bf16 KV, a8w4 MoE, force split_k=1):
cd /sgl-workspace/sglang
AITER_FORCE_A8W4=1 AITER_GROUPED_FORCE_SPLIT_K1=1 bash run_ds-r1.sh   # with --kv-cache-dtype auto
# 4. verify:  curl :8000/generate should decode coherent text (not token 0 repeats)
# 5. GSM8K:   python3 -m sglang.test.few_shot_gsm8k --num-questions 40 --parallel 20 --port 8000
#             expect Accuracy ~0.85, Invalid 0.000
#
# Diagnosing a NEW async GPU fault instead? use the eager+serialized variant:
sed 's/  --page-size 64/  --page-size 64 --disable-cuda-graph/' run_ds-r1.sh > /tmp/run_diag.sh
AMD_SERIALIZE_KERNEL=3 bash /tmp/run_diag.sh 2>&1 | tee /tmp/run_diag.log
# if disk-clean issues recur: rm -f /sgl-workspace/*/gpucore.*.gpu
```
