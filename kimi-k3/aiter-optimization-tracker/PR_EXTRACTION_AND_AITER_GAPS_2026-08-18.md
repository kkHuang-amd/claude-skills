# Kimi-K3 PR extraction and AITER dependency gaps — 2026-08-18

## Draft PRs created

```text
Independent integrations: https://github.com/sgl-project/sglang/pull/35287
KDA decode update:         https://github.com/sgl-project/sglang/pull/34198
```

```text
perf/k3-gfx950-independent-fusions-clean
  base: origin/main 0077f84d3
  worktree: /sgl-workspace/sglang-pr-independent-clean
  ahead: 3 local commits

perf/k3-gfx950-independent-fusions (audit branch retained locally)
  final tree is byte-identical to clean branch
  contains MLA Q/cache add/fix/removal history

perf/k3_fused_kda_decode
  base: origin/perf/k3_fused_kda_decode 8950e2e0d
  worktree: /sgl-workspace/sglang-pr34198
  ahead: 3 commits over the original PR head
```

Both branches are pushed to `HaiShaw/sglang`; both upstream PRs remain Draft.

## Independent SGLang feature branch

Final diff contains 30 paths and excludes the feature bodies of:

```text
#33599 attn-residual
#32796 K3 DCP
#34580 Triton MLA decode tune
#33838 K3 MoE layout/copy
#34198 KDA decode boundary
#33916 MoE copy/cast removal
#34490 Radix-4 router
#34837 12-head prefill concat/cast
```

Included functionality:

```text
12-head AITER MLA decode-side padding
SGLang-vendored gfx950 FlyDSL source selector and selected kernels
MLA output gate
KDA group64 + KDA B2 specialization
unified M2/M4 cooperative MoE preroute
optional FP8 latent tail
M16384 GEMM profile
```

Local commits:

```text
d34cbbcab [AMD] Vendor Kimi-K3 gfx950 FlyDSL integrations
36e8e310a [AMD] Unify Kimi-K3 M2/M4 MoE preroute
8372cadb1 [AMD] Add Kimi-K3 M16384 AITER profile
```

The branch also contains three local commits that added, adapted, then removed
MLA Q/cache fusion after exact pinned-AITER validation failed:

```text
bb6a6960a [AMD] Fuse Kimi-K3 MLA Q and cache preparation
96ca166b8 [AMD] Support pinned AITER MLA cache API
3db67d4dc [AMD] Remove incompatible Kimi-K3 MLA Q/cache fusion
```

The preferred clean branch was rebuilt from main with only the three effective
commits. Its tree matches the validated audit branch exactly.

## Why MLA Q/cache was excluded

Main pins AITER:

```text
d9e5ef7ce08ee7045d583aed768cff41aa9210fe
```

That revision exposes the 14-argument
`fused_qk_rope_concat_and_cache_mla` schema. Removing the newer
`compute_all_q_rope` argument made token=1 tests pass, but token=64 produced
incorrect output and the combined suite could abort.

The required behavior arrived later in AITER commit `770790cd` via AITER
#4342. Therefore merging only SGLang is insufficient until the Docker AITER
pin includes that capability. The feature was removed from the independent
branch and is recorded as a pin dependency, not an unproposed AITER change.

## Pinned-AITER validation

Against AITER `d9e5ef7`:

```text
selected vendored FlyDSL tests: 35 passed
forbidden #4617/#4647 API/static references: none
upstream aiter.ops.flydsl.kimi_k3_* modules: absent
SGLANG_K3_FLYDSL_SOURCE=sglang: local modules selected
```

Full model startup with this checkout was blocked by a container/toolchain
mismatch in stock AITER MoE:

```text
mixed_moe_gemm_2stage.py: x_elem = T.f8
AttributeError: Types has no attribute f8
```

This occurs in the pinned stock AITER MoE path with both
`AITER_FLYDSL_FORCE=0/1`, before the new SGLang features. The currently
installed FlyDSL is newer than the old pin's expected toolchain.

Using the current runnable AITER checkout, while leaving #4617/#4647-specific
flags disabled, the independent branch reached ready state and passed:

```text
C2 smoke: 8/8, 3270.86 tok/s, TPOT 16.65 ms (8192/256)
C4 smoke: 8/8, 5222.32 tok/s, TPOT 19.46 ms (8192/256)
GSM8K 50: 1.000, invalid 0.000
GSM8K 1319: 0.953, invalid 0.001
```

## PR #34198 local update

Existing remote branch:

```text
origin/perf/k3_fused_kda_decode @ 8950e2e0d
```

Prepared local commits:

```text
b1c2bf2fe [AMD] Vendor Kimi-K3 KDA FlyDSL decode
1ab645049 [AMD] Optimize Kimi-K3 C2 KDA decode
d79001988 [AMD] Enable Kimi-K3 C2 KDA winner by default
```

The update vendors only the KDA wrapper/kernels, changes the #34198 adapter to
prefer the SGLang module, and adds the guarded C2 winner. It does not contain
MoE, MLA, group64, latent-tail or independent-branch code.

Pinned-AITER validation:

```text
KDA focused tests: 15 passed
C2 graph benchmark: p50 12.7723 us, mean 12.7692 us
branch status: pushed to Draft PR #34198
full GSM8K 1319: 0.950, invalid 0.001
```

## AITER dependency inventory

### Hard unmerged AITER dependencies with existing PRs

```text
AITER #4617 caller-provided fused_moe output
  used by SGLang #33838/#33916 zero-copy path

AITER #4647 reusable FlyDSL stage1 scratch
  used by AITER_FLYDSL_STAGE1_SCRATCH_REUSE
```

Both already have AITER PRs. Neither is present in the independent branch.

### Stock/pin AITER APIs used by the independent branch

```text
FlyDSL compiler/JIT bootstrap
tensor_shim, buffer_ops, vector
standard AITER MLA decode and MoE operators
GEMM tuning config loader
```

These require no new AITER PR, but must use the FlyDSL version matched to the
Docker pin.

### AITER-related features that lacked SGLang PR coverage

The new independent branch now stages the SGLang PR for:

```text
vendored FlyDSL source selector
MLA output gate
KDA group64/B2
unified M2/M4 cooperative preroute
M16384 profile
12-head MLA decode-side padding
optional latent tail
```

KDA C2 is staged directly on SGLang #34198.

### Remaining gap

MLA Q/cache BF16-Q + FP8-KV requires an AITER pin containing the post-d9e5
operator semantics (AITER #4342 / `770790cd`) before it can be proposed as a
SGLang-only change against current main.

There are currently no active hard AITER code changes without an existing
AITER PR. The unmerged hard dependencies are #4617 and #4647, and both are
already proposed.
