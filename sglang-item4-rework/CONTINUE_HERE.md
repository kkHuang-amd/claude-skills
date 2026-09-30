# PR #34624 rework — stop touching common JIT code

## CONTINUE HERE

**Status:** Done and verified locally. Changes are staged (not committed, not pushed).
**Next:** Decide commit message / whether to push to `karverma-amd/sglang`.
**Repo:** `/workspace/sglang-item4` (fork `karverma-amd/sglang`, branch `item4-fused-compress-norm-rope`, upstream merge-base `1df78c2cf`)
**Repro:**
```bash
cd /workspace/sglang-item4 && PYTHONPATH=$PWD/python python3 -m pytest -q \
  test/registered/kernels/ops/attention/test_fused_compress4_norm_rope.py \
  test/registered/kernels/ops/attention/test_c4_v2.py
```
**Pass criteria:** 24 passed (fused) + 29 passed (c4_v2). Both green on gfx950 as of this run.

## The problem

The branch refactored `python/sglang/kernels/jit/csrc/deepseek_v4/c4_v2.cuh`
(-171 lines): the body of `c4_forward` was replaced by a call into a new shared
header, and `c4_write_decode` was moved out of the file entirely. `c4_v2.cuh` is
the default compress path on **every** platform, so a gfx95-only feature was
changing code that CUDA builds depend on. That was the reviewer's objection.

The refactor came from an earlier review round (DarkSharpness asked the fused
kernel to reuse code from `c4_v2.cuh` rather than duplicate it), so the two
requests pull in opposite directions. This rework picks "do not touch the common
path", and pays for it with an explicit, documented, test-guarded copy.

## What changed

1. `c4_v2.cuh` restored to upstream **byte-for-byte** (`git diff 1df78c2cf -- <file>` is empty).
2. `include/sgl_kernel/deepseek_v4/c4_compress_core.cuh` renamed to
   `c4_compress_core_hip.cuh`, and its contents moved into `namespace
   sglang::fused_c4`. It is a Trait-parameterized copy of the `c4_v2.cuh`
   compress, included only by `fused_compress4_norm_rope_hip.cuh`.
   The nested namespace means `c4_write_decode` no longer collides with the
   identically-named helper `c4_v2.cuh` defines. (Not strictly required today —
   each JIT module is its own translation unit, see `cuda_files=[...]` in
   `python/sglang/kernels/ops/attention/dsv4/compress.py` — but it makes
   co-inclusion safe.)
3. `fused_compress4_norm_rope_hip.cuh`: include path updated, call sites
   qualified with `fused_c4::`, and the header comment rewritten to say the copy
   is deliberate and that `test_fused_compress4_norm_rope.py` is what keeps the
   two in step.

Net effect on the diff against upstream: `c4_v2.cuh` disappears from it. Every
remaining file is either new or additive. Total deletions across the whole PR
are now 6 lines, all outside `kernels/jit/`.

## Equivalence argument

The copy in `c4_compress_core_hip.cuh` merges `c4_v2.cuh`'s same-dtype and
mixed-dtype branches into one branch-per-element form. Pointer arithmetic and
the softmax operation order are unchanged, so it is numerically identical; the
guard is `test_fused_compress4_norm_rope.py::test_fused_matches_chain_decode`,
which compares the fused kernel against `compress_forward` +
`compress_norm_rope_store`.

**Maintenance hazard to state in the PR:** a future change to the compress math
in `c4_v2.cuh` must be mirrored into `c4_compress_core_hip.cuh`. The test catches
it, but nothing at compile time does.

## Not changed (reviewed, judged safe)

- `srt/layers/layernorm.py` — the only remaining edit to a shared runtime file.
  Gated by `_NATIVE_BPRESHUFFLE_SCALE` (env `SGLANG_OPT_NATIVE_BPRESHUFFLE_SCALE`,
  default 0) **and** `_use_aiter_bpreshuffle_gfx95`, so behaviour off gfx95 and
  by default is byte-identical to upstream.
- `srt/environ.py`, `srt/layers/quantization/fp8_utils.py`,
  `srt/layers/attention/dsv4/compressor_v2.py`,
  `kernels/ops/attention/dsv4/{__init__,compress}.py` — additive, HIP-gated,
  default off.
- `test/kernels/deepseek_v4/common.py` — `make_state_pool` gained a `dtype`
  kwarg defaulting to the previous `torch.float32`.
