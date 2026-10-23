# Handoff: gfx1250 cooperative cluster TopK v2 route

The route is now unblocked on AMD gfx1250 without
`__builtin_amdgcn_map_shared_rank`. Keep correctness and per-shape speed as the
acceptance criteria; do not treat a correctness-only result as success.

## Resolved status (2026-10-06)

- Device/toolchain: gfx1250, HIP 7.15.26333.
- `__builtin_amdgcn_map_shared_rank` is absent, and HIP
  `this_cluster().sync()` hangs.
- Split `barrier_arrive()` / `barrier_wait()` works with 1,024-thread
  workgroups.
- `__builtin_amdgcn_cluster_load_b32` works across all 1,024 lanes. The
  production route stages each rank's histogram in global memory and
  multicasts bins into registers. It does not use remote LDS, the rejected TDM
  descriptor path, or per-candidate global atomics.
- The coarse histogram is 12-bit (4,096 bins). The earlier 10-bit version
  overflowed the 2,048-entry tie buffer on duplicate-heavy inputs.
- `SGLANG_GFX1250_TOPK_CLUSTER=1` enables the route; it remains default-off.
- Final correctness: 306/306 TopK v2 tests pass with the flag on and 306/306
  pass with it unset.

Final fresh-process timings (20 warmups, 100 iterations, k=512):

| batch | seq | flag unset (us) | cluster (us) | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 131072 | 20.83 | 17.35 | 1.201x |
| 4 | 131072 | 24.52 | 18.19 | 1.348x |
| 1 | 262144 | 25.80 | 22.91 | 1.126x |
| 31 | 131072 | 43.86 | 25.28 | 1.735x |
| 64 | 131072 | 48.79 | 39.26 | 1.243x |
| 64 | 262144 | 69.62 | 41.86 | 1.663x |

## What is already in your tree

[PR #41687](https://github.com/sgl-project/sglang/pull/41687) is already merged. It only changes `python/sglang/kernels/aot/include/hip/dsa_topk_coop.cuh`: `kWaveSize` is 32 on GFX10/11/12 (gfx1250) and 64 otherwise, plus a `check_wave_size()` trap. That is the AOT DSA indexer top-k (wave shuffles inside one CTA). It is not TopK v2, not `__cluster_dims__`, and not `map_shared_rank`. Do not edit that file. Do not re-test it.

## Source of truth

The resolved implementation is in the `/sgl-workspace/sglang` working tree.
`gfx1250_cooperative_cluster_route.diff` is the earlier split-barrier bootstrap
and does not contain the final register-multicast implementation; do not apply
it over this tree.

The diff touches only:

- `python/sglang/kernels/jit/utils/arch.py`
- `python/sglang/kernels/jit/include/sgl_kernel/utils.cuh`
- `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh`
- `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh`

What the working-tree implementation does:

- Defines `SGL_ROCM_ARCH_GFX1250` from `gcnArchName` for the JIT compile only.
- Compiles `TopKCluster` on gfx1250. CUDA SM90/SM100 behavior is unchanged. gfx942/gfx950 stay `SUPPORT_CLUSTER false`.
- Replaces CUDA `barrier.cluster.*` PTX with HIP's split
  `cooperative_groups::cluster_group::barrier_arrive()` /
  `barrier_wait()` on ROCm. The combined `this_cluster().sync()` path hung on
  this gfx1250 toolchain, while the split path passed a two-workgroup,
  1,024-thread-per-workgroup probe. HIP implements split arrive with a local
  workgroup barrier and one cluster-barrier signal per workgroup.
- Launches cluster kernels with `hipLaunchKernelEx` and `hipLaunchAttributeClusterDimension` set to `{1, 8, 1}` (y-oriented). `{8, 1, 1}` was rejected by HIP in the earlier bring-up. Non-cluster launches stay on `hipLaunchKernelGGL`.
- Uses `cluster_load_b32` register multicast instead of `map_shared_rank`.
- Cluster dispatch is off unless `SGLANG_GFX1250_TOPK_CLUSTER=1`. With the variable unset, the existing ROCm split path (two launches, global histogram scratch, no DSMEM) and the register/streaming ladder are unchanged, including the plan kernel.

Cluster floor when the flag is on: seq > 32768 for batch <= 15, else seq >
65536, and batch <= 512. All gfx1250 batches use the fused direct-row cluster
kernel. Tuned cluster sizes are 8/15 for batch <= 60 at short/long lengths and
4/2 for larger batches.

## Do not revive the rejected replacements

Measured on gfx1250 on 2026-08-31, then removed:

- `__builtin_amdgcn_map_shared_rank` was missing. HIP's `map_shared_rank` returned nullptr and the kernel faulted.
- A global-workspace substitute passed 278 TopK v2 tests and was 90×–1,000× slower than v2 streaming. Rejected.
- A TDM multicast cluster (`topk_v2_gfx1250_cluster_tdm_poc_0831.diff`) was slower than streaming at C4 length 262144: streaming 56.0 µs, 2 workgroups 87.3 µs, 4 workgroups 92.1 µs. Batch 2 and 4 were worse. No production threshold inside a 1,048,576-token context. Do not start from that diff.

The missing DSMEM builtin is expected. Do not revive `map_shared_rank`, TDM
descriptors, or the old all-global reduction. The accepted workaround is
global histogram staging plus `cluster_load_b32` register multicast.

## Step 1 — probe, before the full kernel

Confirm the device is gfx1250 and whether the compiler actually has the DSMEM builtin:

```bash
python3 - <<'PY'
import torch
props = torch.cuda.get_device_properties(0)
print("hip", torch.version.hip)
print("device", props.name)
print("gcnArchName", getattr(props, "gcnArchName", None))
PY

rg -n "map_shared_rank|__builtin_amdgcn_map_shared_rank" \
  "$(hipconfig --rocmpath)/include/hip" "$(hipconfig --rocmpath)/include/hip/amd_detail" \
  || true
```

Use `probe_hip_split_cluster_barrier.hip` and `probe_hip_cluster_load.hip` to
check the two primitives used by the implementation. A `map_shared_rank` probe
returning null is the known toolchain behavior, not a blocker for this route.

Also confirm a non-cluster TopK launch still works with the flag unset, so `hipLaunchKernelEx` did not change the default path.

## Step 2 — correctness

JIT will rebuild `topk_v2` on first call. Use one GPU. If a launch faults or hangs, kill the process immediately and do not retry in a loop.

Find the current test module. On this tree it is expected at `test/registered/kernels/ops/attention/test_topk_v2.py` (older trees used `test/registered/jit/deepseek_v4/test_topk_v2.py`). Run the raw-index / output-index cases that cross the cluster floor, with the flag on and off:

- Fused small-batch: batch 1 and 4, seq 65536, 65537, 131072, 262144, topk 512 and 2048.
- Persistent pool: batch 31 and 64, same lengths and topk.
- Just below the floor (seq 32768 and 65536 at batch 16) must stay on streaming/split, not cluster.
- Flag unset at batch 64, seq 262144 must match the pre-diff ROCm split result.

Compare index sets against the unfused reference used by the existing test. Allow the test's existing tie tolerance. A GPU fault, a trap, or an out-of-range index is a fail.

## Step 3 — speed, only after correctness passes

Compare three routes at the same shapes, 20 warmups, 100 timed iterations, GPU events, rank-averaged if you use more than one device (this test is single GPU):

1. `SGLANG_GFX1250_TOPK_CLUSTER` unset: current ROCm split or streaming path. This is the baseline.
2. `SGLANG_GFX1250_TOPK_CLUSTER=1`: cooperative cluster.
3. TopK v1 (`topk_transform_512` or the current v1 entry), only as a reference. v2 streaming already beat v1 by about 1.5×–5× on older measurements; that is not the question.

Required rows, k=512:

| batch | seq | which cluster |
| ---: | ---: | --- |
| 1 | 131072 | fused small-batch |
| 4 | 131072 | fused small-batch |
| 1 | 262144 | fused small-batch |
| 31 | 131072 | persistent |
| 64 | 131072 | persistent |
| 64 | 262144 | persistent |

A row is a win only when cluster is correct and `time_unset / time_cluster > 1.0`. Keep every row. An average that hides a regression is not a pass.

Do not launch GSM8K or a server until every row above is correct and faster. If you do run a server afterward, use a short smoke only, and say that it is not signoff.

## Report

Return:

- `gcnArchName`, HIP version, and whether `__builtin_amdgcn_map_shared_rank` exists.
- Probe result (null vs a real DSMEM pointer).
- Per-shape correctness and the six timing rows, flag off vs flag on.
- Whether you left the flag default off (you must; do not change the default).
- If it lost or faulted, say so and do not keep experimental substitutes in the tree.

The previous 2026-08-31 conclusion was overturned by register multicast:
remote LDS is still unavailable, but the cooperative route now passes the full
suite and wins every required timing row.
