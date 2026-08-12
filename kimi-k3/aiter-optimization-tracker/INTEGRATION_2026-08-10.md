# Kimi-K3 SGLang + AITER PR integration — 2026-08-10

## CONTINUE HERE

- SGLang worktree: `/sgl-workspace/sglang`
- SGLang branch: `integration/kimi-k3-pr-stack`
- AITER worktree: `/sgl-workspace/aiter`
- AITER branch: `integration/kimi-k3-pr-stack`
- SGLang #33838/#33916 integration is committed as `0dae20b83`; #34198 is
  the parent commit `8950e2e0d`.
- AITER #4495/#4617 remains staged and intentionally not committed.
- Accuracy and fixed 8192/1024 validation are complete; see
  [`VALIDATION_2026-08-10.md`](VALIDATION_2026-08-10.md).
- Exact next step: capture a decode trace to verify copy/cast removal and
  fused-KDA selection, then review and commit each repository separately.

## Integrated SGLang PRs

1. [sglang#34198](https://github.com/sgl-project/sglang/pull/34198)
   - Already present as branch base commit `8950e2e0d`.
   - Fuses the ROCm KDA decode boundary and defers `f_b` projection to AITER.
2. [sglang#33838](https://github.com/sgl-project/sglang/pull/33838)
   - A4W4/A8W4 SiTU weight-layout precedence.
   - Correction-bias dtype cache.
   - Explicit published SiTU MoE output-buffer contract.
   - Three focused unit-test files.
3. [sglang#33916](https://github.com/sgl-project/sglang/pull/33916)
   - Passes SGLang's published MoE output buffer to AITER.
   - Removes the K3 AITER routed-input contiguous copy.
   - Stores router correction bias in the dtype consumed by the selected route.

### SGLang overlap resolution

PR #33916 was based on an older `kimi_k3.py`. Current main replaced
`_moe_front_needs_contiguous` with `_moe_front_needs_dense_bf16`; the PR's
AITER stride support was semantically ported into the newer property instead of
adding an unused legacy field.

The zero-copy changes in #33838 and #33916 are complementary:

- #33838 makes the upper SiTU path return its published destination.
- #33916 threads that destination through the generic AITER runner.
- A pointer comparison remains as the fail-closed fallback for older or
  incompatible AITER builds.

## Integrated AITER dependencies

Base: `origin/main@7c5e20170`.

Already present on main:

- [aiter#4463](https://github.com/ROCm/aiter/pull/4463): A4W4 SiTUv2 mode and
  tuner fixes.
- [aiter#4534](https://github.com/ROCm/aiter/pull/4534): optimized K3
  Opus/FlyDSL A8W4 SiTUv2 dispatch required by SGLang #33838.

Applied as staged changes:

1. [aiter#4495](https://github.com/ROCm/aiter/pull/4495)
   - gfx950 FlyDSL KDA decode.
   - Fused KDA decode + `128×128 f_b` projection boundary.
   - Public wrapper, two kernel implementations, and focused tests.
2. [aiter#4617](https://github.com/ROCm/aiter/pull/4617)
   - Optional caller-provided `output` for `fused_moe`.
   - Strict shape/dtype/device/contiguity matching with private-allocation
     fallback.
   - Covers both accumulate and route-reduce two-stage paths.
   - Aligns the fake op signature for `torch.compile`.

### AITER conflict resolution

`aiter/ops/flydsl/__init__.py` changed on main after #4495. The integration
keeps the newer `flydsl_mla_reduce_v1` export and adds all three K3 KDA exports.

The pre-existing untracked directory `aiter/jit/flydsl_cache/` was preserved and
is not part of the stack.

## Verification completed

### SGLang

```text
11 passed, 3 warnings
```

Focused tests:

- `test_topk_correction_bias_cache.py`
- `test_mxfp4_situ_output.py`
- `test_mxfp4_situ_weight_layout.py`

Additional checks:

- SGLang pre-commit hooks: passed.
- Python compile checks: passed.
- No conflict markers or whitespace errors.

### AITER

```text
12 passed, 2 warnings
```

Focused suite:

- `op_tests/flydsl_tests/test_kimi_k3_kda_decode.py`

Additional checks:

- Caller output-buffer predicate/signature smoke: passed.
- Python compile checks: passed.
- No conflict markers or whitespace errors.
- IDE lint diagnostics: none.
- Repository has no pre-commit config and the environment has no standalone
  Ruff executable; the upstream #4495 Ruff result was not rerun locally.

## End-to-end validation

Completed on 8×MI355X:

- GSM8K 200, 5-shot: `0.990`.
- Fixed 8192/1024 C2/4/8/16/32: all 496 measured requests succeeded.
- Full-stack total throughput: `954.11 / 1703.90 / 2766.01 / 4151.73 /
  5630.64 tok/s`.
- Direct comparison with #33838: `+4.63%` to `+9.43%` throughput.

Full commands, metrics, comparisons, caveats, and artifact paths:
[`VALIDATION_2026-08-10.md`](VALIDATION_2026-08-10.md).

The validated environment was:

Use the environment expected by the PR stack:

```bash
SGLANG_USE_AITER=1
SGLANG_AITER_K3_OPT=1
AITER_FLYDSL_FORCE=1
AITER_SITUV2_A8W4=1
AITER_SITUV2_A4W4=0
SGLANG_K3_FLYDSL_AR_NORM=1
```

Minimum acceptance:

1. TP8 server reaches `ready to roll`.
2. KDA fused boundary is selected on gfx950 and fallback remains available.
3. GSM8K accuracy matches the known A8W4 range (~0.98 on 200 samples).
4. No per-layer routed-output D2D copy, routed-input contiguous copy, or router
   bias cast appears in the decode trace.
5. Fixed 8192/1024 concurrency sweep is compared against the #33838 baseline.

## Commit state

- SGLang: `0dae20b83 perf(kimi-k3): integrate AITER MoE optimizations`
- AITER: staged, not committed
