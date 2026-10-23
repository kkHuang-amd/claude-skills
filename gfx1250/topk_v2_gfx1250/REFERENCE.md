# gfx1250 Cooperative Kernel Reference

## Validated environment

- Architecture: gfx1250
- HIP: 7.15.26333
- Workgroup size: 1,024 threads
- Cluster launch orientation: `{1, N, 1}`; `{N, 1, 1}` was rejected during
  bring-up.

Re-run the probes after any compiler, runtime, firmware, or architecture
change. Do not infer remote-LDS support from successful cluster launch or
multicast.

## Primitive semantics

### Split cluster barrier

Use:

```cpp
static_cast<void>(cooperative_groups::this_cluster().barrier_arrive());
using Token = cooperative_groups::cluster_group::arrival_token;
cooperative_groups::this_cluster().barrier_wait(Token{});
```

The HIP arrive helper performs a local workgroup barrier before signaling the
cluster barrier. All participating workgroups must execute barriers in the same
order. Use release/acquire intent around global data publication.

### Register multicast

`__builtin_amdgcn_cluster_load_b32(ptr, 0, mask)` loads from global memory and
returns the multicast value in the corresponding lane of each selected
workgroup. The mask is `(1u << cluster_size) - 1u` for the contiguous clusters
used by TopK.

The validated TopK maps one histogram bin group to each of 1,024 lanes:

```cpp
constexpr uint32_t items = 4096 / 1024;
for (uint32_t i = 0; i < items; ++i) {
  const uint32_t bin = threadIdx.x * items + i;
  // Sum the same bin from every rank.
}
```

This primitive cannot address a peer's existing LDS or perform remote LDS
atomics.

## Why the final TopK design works

The CUDA cluster implementation aggregates into rank 0 through remote shared
memory. gfx1250 cannot implement that operation, so the successful design
changes the dataflow:

- Global memory is only a bounded publication surface: histograms, two counts
  per rank, and at most 2,048 ties per row.
- Histogram reads are multicast into registers.
- Every rank computes the same threshold, eliminating threshold broadcast.
- Rank-prefix placement replaces global output-position atomics.
- Only rank 0 resolves the final bounded tie set.

This differs from rejected approaches:

- The old global-workspace shim mechanically replaced remote LDS and generated
  excessive device-scope traffic.
- The TDM prototype staged and rematerialized matrices in LDS and lost to the
  existing split path.
- The successful route uses direct global-to-register multicast and avoids a
  per-candidate atomic stream.

## Correctness traps

### Histogram precision

Ten histogram bits can put more than 2,048 non-identical candidates in the
threshold bin. Truncating that bin before exact tie selection can discard a
better candidate. Twelve bits passed the duplicate-heavy suite and requires
four bins per thread.

Whenever histogram bits change, update:

- LDS histogram size.
- Global workspace allocation.
- Per-thread staging/reduction loop count.
- Threshold boundary extraction.

### Synchronization

- Never return from a branch if peers will reach another workgroup or cluster
  barrier.
- `__syncthreads()` is still required for local LDS visibility; a cluster
  barrier does not replace careful local ordering.
- Rank 0 may perform page transformation while peers wait at a final cluster
  barrier.
- Tie handling must finish before rank 0 reuses or transforms output storage.

### Workspace indexing

Index workspace by logical row slot, rank, then bin/count. Allocate for the
largest compiled cluster width and the 4,096-bin histogram. A 1,024-bin
allocation silently overlaps later regions.

### Launch and fallback

- Define `SGL_ROCM_ARCH_GFX1250` only for matching JIT architecture.
- Pass `hipLaunchAttributeClusterDimension` and use `hipLaunchKernelEx` only
  for cluster launches.
- Preserve ordinary `hipLaunchKernelGGL` behavior elsewhere.
- Allocate cluster workspace lazily only when the opt-in flag is enabled.
- If allocation or eligibility fails, retain the existing streaming/split
  route.

## Performance method

Use 20 warmups and 100 GPU-event iterations. Run each shape in a fresh Python
process and validate output sets before timing.

Required reference shapes for k=512:

- `(1, 131072)`
- `(4, 131072)`
- `(1, 262144)`
- `(31, 131072)`
- `(64, 131072)`
- `(64, 262144)`

The accepted 2026-10-06 run produced per-row speedups of 1.201x, 1.348x,
1.126x, 1.735x, 1.243x, and 1.663x respectively. These are reference results,
not permanent thresholds; remeasure after toolchain or kernel changes.

## Tensor parallelism

Do not equate cluster width with TP degree.

Cluster width partitions one TopK row across workgroups on one GPU. Tensor
parallelism partitions model execution across GPUs. The current DSV4 TopK call
receives the same row geometry on each TP rank, so TP1/TP2/TP4 share one kernel
tuning.

If serving changes introduce TP-dependent TopK geometry:

1. Capture per-rank row count, compressed sequence lengths, topk, and output
   mode.
2. Add those local shapes to the benchmark matrix.
3. Tune by shape bucket, not by the `tp_size` label.
4. Add TP-specific dispatch only when measured optima differ enough to justify
   extra compiled variants.

## Relevant source

- `sglang/python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh`
- `sglang/python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh`
- `sglang/python/sglang/kernels/jit/include/sgl_kernel/utils.cuh`
- `sglang/python/sglang/kernels/jit/utils/arch.py`
- `PROMPT_gfx1250_cooperative_cluster.md`
- `benchmark_cluster_load_topk.py`
- `probe_hip_split_cluster_barrier.hip`
- `probe_hip_cluster_load.hip`
