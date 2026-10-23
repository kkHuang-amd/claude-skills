Owner node: ctheliosmc-rck-g02-h13-17.rck.dcgpu

# Session prompt: continue MI455 (gfx1250) proxy-MoE optimization

Paste into a fresh session on h13-17 (inside the same container), after the
`NEW_WORKSPACE_PROMPT.txt` bootstrap:

```text
Read /workspace/claude-skills/proxy-moe/MI455_RUN_h13-17.md: the CONTINUE HERE block first,
then only the sections you need. That doc is the source of truth; do not reconstruct chat history.
Follow gfx1250/mi400-optimization-skills (README, shared/measurement-contract.md,
correctness-checklist.md, skill-quality-gates.md); profile before optimizing.
Kernel sourcing: check ROCm/FlyDSL and ROCm/aiter main for an existing gfx1250 kernel first;
write our own (FlyDSL) only if none exists.
Push targets: sglang to HaiShaw/sglang, aiter to kkHuang-amd/aiter, branch
feat/proxy-model-gfx1250-1023. Commit without AI trailers, fetch before push, never force-push.
Next task: <pick from the CONTINUE HERE "Next" list>
```

## Node-local state on h13-17 (re-check before trusting: `hostname`, `ps`, `git status`)

- **RCCL:** the container's SDK `librccl.so.1` is REPLACED by RCCL 2.32.3 (rocm-systems `4ca8623`). Restore with `bash /workspace/proxy_work/rccl_swap.sh restore`.
- **sglang** `/sgl-workspace/sglang`: branch `feat/proxy-model-gfx1250-1023` @ `ac677aad64`, clean.
  - The old-base work is in `git stash` "oldbase-1023: ...".
  - The image's build-only edits (`pyproject.toml`, `kernels/aot`) stay in that stash.
- **aiter** `/sgl-workspace/aiter`: branch `feat/proxy-model-gfx1250-1023` @ `dae49be2b`.
  - Uncommitted on purpose: `csrc/cpp_itfs/torch_utils.py`, the image's torch 2.12 stream fix. Do not push it.
  - C++ JIT rebuilt from this source: `module_aiter_core`, `module_custom_all_reduce`, `module_custom_all_reduce_gfx1250`. `module_gemm_common` is STILL the image build.
  - The image JIT `.so` backup is in `/workspace/proxy_work/oldbase_backup_1023/aiter_jit/`.
- **Model:** `/workspace/proxy_work/proxy_12l` (12L dummy, seed 0). Tokenizer: `/workspace/proxy_work/tokenizer`.
- **Results:** `/shared_nfs/kk/results/proxy_12l/<run-tag>/`.

## One-point repro of the current best (1K1K, c=2)

```bash
export NCCL_CUMEM_ENABLE=1 RCCL_DDA_ENABLE=0 SGLANG_OPT_LAGUNA_ROUTER_FLYDSL=1 AITER_CUSTOM_AR_MAX_SIZE=1048576 SMOKE=1
bash /workspace/claude-skills/proxy-moe/scripts/mi455_point.sh 1k1k 2 <run-tag>
# expect: smoke passed=True, Median ITL ~6.7 ms, ~291 output tok/s (c=64: ~31.6 ms, ~1886 tok/s)
```
