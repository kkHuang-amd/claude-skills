# Cluster Launch Beginner Guide for gfx1250 / MI455

This note is written for learning and future kernel-porting work in SGLang/AITER.
It uses the current SGLang TopK v2 pattern and AITER gfx1250 cluster GEMM kernels
as concrete references.

## 0. One-Sentence Summary

Cluster launch lets multiple CTAs / workgroups launch as one co-resident group so
they can cooperate on one larger logical task.

Do not think of it as a magic launch flag. Think of it as an algorithm design:

```text
Before:
  one workgroup handles one independent tile/item

After:
  several workgroups form one cluster
  the cluster handles one larger tile/item together
  workgroups inside the cluster may share data, synchronize, or reduce partial results
```

On NVIDIA, people often say CTA cluster. On AMD/HIP/gfx1250, it is natural to
think in terms of workgroup clusters.

## 1. Normal Kernel Mental Model

A normal GPU kernel usually works like this:

```text
grid
  workgroup 0 -> one tile / row / chunk
  workgroup 1 -> another tile / row / chunk
  workgroup 2 -> another tile / row / chunk
```

Inside one workgroup:

- Threads can cooperate.
- They can use shared memory / LDS.
- They can synchronize with workgroup-level barriers.

Across workgroups:

- They are mostly independent.
- They cannot cheaply synchronize.
- They cannot normally use each other's LDS.

This is excellent when every workgroup has enough independent work and there is
little cross-workgroup data reuse.

## 2. Cluster Launch Mental Model

Cluster launch groups multiple workgroups together.

Example:

```text
cluster_dims = (4, 4, 1)

one cluster = 16 workgroups

cluster
  local workgroup (0,0)
  local workgroup (0,1)
  ...
  local workgroup (3,3)
```

The device code can ask:

```text
Which cluster am I in?
Which local workgroup am I inside this cluster?
```

Then the kernel can map one cluster to a larger logical item.

For GEMM:

```text
one workgroup  -> one C tile
one cluster    -> one C super-tile made of many C tiles
```

For TopK:

```text
one workgroup  -> scans one chunk of candidates
one cluster    -> scans one full long row and combines partial results
```

## 3. What Cluster Launch Can Optimize

Cluster launch is useful when it enables one of these wins.

### 3.1 Reduce Repeated Global Loads

GEMM is the classic example.

Suppose a `4x4` cluster computes a C super-tile:

```text
          N direction
        WG00 WG01 WG02 WG03
M dir   WG10 WG11 WG12 WG13
        WG20 WG21 WG22 WG23
        WG30 WG31 WG32 WG33
```

Workgroups in the same row may reuse the same A tile.
Workgroups in the same column may reuse the same B tile.

Without cluster behavior:

```text
each WG loads its own A/B from global memory
```

With cluster multicast / cluster-aware loads:

```text
one producer load can fan out to peer WGs inside the cluster
```

This is the idea behind AITER gfx1250 cluster GEMM.

### 3.2 Split One Huge Logical Item Across Multiple Workgroups

SGLang TopK v2 is the best example.

For sparse attention top-k, one row may have a very long sequence length:

```text
seq_len = 65536, 131072, 262144, ...
```

A single workgroup scanning all candidates can become slow or shared-memory
limited.

Cluster version:

```text
one row
  rank 0 scans chunk 0
  rank 1 scans chunk 1
  rank 2 scans chunk 2
  ...
  rank 7 scans chunk 7

then the cluster combines histograms / thresholds / selected indices
```

This helps when the per-row work is large and batch is small enough that latency
matters.

### 3.3 Build Better Persistent Schedulers

Some AITER kernels combine:

```text
cluster launch + persistent workgroup scheduling
```

Instead of launching exactly one workgroup per logical tile, the kernel launches a
fixed pool of workgroups/clusters and lets the shader walk through the tile grid.

This can improve occupancy and scheduling regularity for specific shapes, but it
also adds constraints. Shapes often need to tile cleanly.

## 4. When Should You Consider Cluster Launch?

Use this checklist before touching code.

Cluster launch is promising when all or most are true:

- The kernel is latency-bound or memory-traffic-bound.
- A single logical item is too large for one workgroup to process efficiently.
- Neighboring workgroups reload the same data.
- The work can be split into independent chunks and then merged.
- Shapes are regular enough to satisfy cluster-dim divisibility.
- There is a clear fallback path for unsupported shapes or architectures.
- A microbenchmark can prove a crossover point where cluster is faster.

Examples that often fit:

- Long-sequence TopK.
- Large GEMM tiles with A/B reuse across neighboring output tiles.
- Histogram/reduction/prefix-sum style kernels with huge per-item work.
- Attention-like kernels where several output tiles reuse the same K/V data.

Examples that usually do not fit:

- Simple elementwise kernels like `C[i] = A[i] + B[i]`.
- Kernels with no cross-workgroup reuse.
- Kernels already fully saturated by a very large grid.
- Irregular small shapes with many tails.
- A kernel where cluster sync/launch constraints cost more than the saved work.

## 5. Red Flags

Do not use cluster launch just because the hardware supports it.

Red flags:

- You cannot explain what data is shared or what partial results are reduced.
- You only add `__cluster_dims__` but the algorithm is unchanged.
- The shape checks are missing.
- There is no non-cluster fallback.
- Correctness depends on all workgroups being present, but the grid may contain
  partial clusters.
- You did not benchmark against the old path.
- The batch is large enough that normal workgroup-level parallelism already fills
  the GPU.

## 6. SGLang TopK v2 Pattern

Reference skill:

```text
/sgl-workspace/skills_0712/cuda-cluster-launch-kernels/SKILL.md
```

Reference files in SGLang:

```text
python/sglang/jit_kernel/csrc/deepseek_v4/topk_v2.cuh
python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/topk_impl.cuh
python/sglang/jit_kernel/include/sgl_kernel/utils.cuh
python/sglang/jit_kernel/dsv4/topk.py
test/registered/jit/deepseek_v4/test_topk_v2.py
```

Mental model:

```text
Host plan:
  inspect seq_lens
  choose which rows are long enough for cluster path
  write metadata for those rows

Device cluster kernel:
  one cluster handles one long row
  each rank handles a chunk
  ranks build local histograms
  cluster reduces histograms
  primary rank finds threshold
  ranks collect candidates
  primary rank finalizes top-k output

Fallback:
  shorter rows use register or streaming single-workgroup paths
```

Why it can win:

- One row has too much work for one workgroup.
- Work splits naturally by candidate range.
- Partial histograms can be reduced.
- Batch can be small, so latency matters.

Important rule:

```text
cluster path must be selected by measured crossover, not by a magic constant
```

## 7. AITER Cluster GEMM Pattern

Reference files:

```text
/sgl-workspace/aiter_0716/csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_ws_gfx1250.cuh
/sgl-workspace/aiter_0716/csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_cluster_tdm_splitk_ws_gfx1250.cuh
/sgl-workspace/aiter_0716/csrc/py_itfs_cu/asm_f4gemm.cu
/sgl-workspace/aiter_0716/csrc/py_itfs_cu/asm_mxfp8fp4gemm.cu
/sgl-workspace/aiter_0716/csrc/include/aiter_hip_common.h
/sgl-workspace/aiter_0716/hsa/gfx1250/f4gemm/f4gemm.csv
/sgl-workspace/aiter_0716/hsa/gfx1250/mxfp8fp4gemm/mxfp8fp4gemm.csv
```

Plain GEMM pattern:

```text
grid = (M tiles, N tiles, split_k)

each workgroup:
  owns one C tile
  loads its own A tile
  loads its own B tile
  computes and stores one output tile
```

Cluster GEMM pattern:

```text
cluster_dims = (4, 4, 1)

one cluster:
  owns a C super-tile
  contains 16 workgroups
  each local workgroup computes one C tile inside the super-tile

data reuse:
  workgroups with same M tile can share A
  workgroups with same N tile can share B
```

Device mapping:

```cpp
const int cluster_x = __builtin_amdgcn_cluster_id_x();
const int cluster_y = __builtin_amdgcn_cluster_id_y();
const int local_x   = __builtin_amdgcn_cluster_workgroup_id_x();
const int local_y   = __builtin_amdgcn_cluster_workgroup_id_y();

const int tile_row =
    (cluster_x * kClusterWgM + local_x) * kBlockM;
const int tile_col =
    (cluster_y * kClusterWgN + local_y) * kBlockN;
```

Host-side shape requirement:

```text
tiles_x % cluster_x == 0
tiles_y % cluster_y == 0
```

If the kernel has `tile_m=256`, `tile_n=256`, `cluster_x=4`,
`cluster_y=4`, then:

```text
N must be multiple of 256 * 4 = 1024
M must be multiple of 256 * 4 = 1024
```

Some variants use `cluster_y=1`, so M may have weaker constraints depending on
the shader's masking/persistent scheduler.

## 8. Host-Side Launch Checklist

When adding a cluster path, host code usually needs:

1. Architecture guard

```text
only enable on gfx1250 / MI455 or known-supported arch
```

2. Kernel variant metadata

```text
tile_m, tile_n, cluster_x, cluster_y, persistent flag, kernel name
```

3. Shape checks

```text
gridDim.x must be divisible by clusterDim.x
gridDim.y must be divisible by clusterDim.y
gridDim.z must be divisible by clusterDim.z if cluster z is used
```

4. Launch attribute

For HIP/AITER:

```cpp
hipLaunchAttribute attrs[1];
attrs[0].id = hipLaunchAttributeClusterDimension;
attrs[0].val.clusterDim.x = cluster_x;
attrs[0].val.clusterDim.y = cluster_y;
attrs[0].val.clusterDim.z = cluster_z;

HIP_LAUNCH_CONFIG launch_config{};
launch_config.gridDimX = gdx;
launch_config.gridDimY = gdy;
launch_config.gridDimZ = gdz;
launch_config.blockDimX = bdx;
launch_config.attrs = attrs;
launch_config.numAttrs = 1;

hipDrvLaunchKernelEx(&launch_config, kernel_func, nullptr, kernel_args);
```

5. Fallback

```text
if arch unsupported or shape unsupported:
  use non-cluster kernel
```

Do not silently drop cluster attributes for a cluster-aware kernel. That can
produce wrong results or deadlock.

## 9. Device-Side Checklist

Before writing device code, answer these:

```text
What is one logical task?
  one row?
  one C super-tile?
  one long sequence?

What is one cluster responsible for?

What is each local workgroup responsible for?

What data is shared?
  A tile?
  B tile?
  K/V tile?
  histogram bins?

What synchronization is required?

What happens on partial/tail tiles?

Which workgroup writes final output?

Can multiple workgroups write the same output?
  if yes, do you need atomics or a reduction stage?
```

Device code often needs:

- `__cluster_dims__(x, y, z)` or equivalent compile-time cluster declaration.
- Cluster ID and local workgroup ID.
- Cluster-aware data movement or DSMEM/multicast behavior.
- Cluster synchronization where required.
- Careful out-of-bound guards for rounded-up grids.

## 10. Common Pitfalls

### 10.1 Grid Not Divisible by Cluster Dim

If cluster dim is `4x4`, the launch grid must be a multiple of `4` in both
directions unless the runtime/kernel explicitly supports tails.

Bad:

```text
grid = (10, 10)
cluster = (4, 4)
```

Good:

```text
grid = (12, 12)
cluster = (4, 4)
```

But if you round up the grid, device code must guard out-of-bound tiles.

### 10.2 Using Cluster Multicast Without Real Cluster Launch

In AITER plain GEMM, the mask must be zero. A non-zero TDM workgroup mask can make
the hardware wait for cluster peers that do not exist, causing deadlock.

### 10.3 Thinking Cluster Always Helps

Cluster launch can reduce memory traffic or split large work, but it also reduces
scheduling freedom. On large batches, a normal grid may already be better.

### 10.4 Missing Benchmark Crossover

TopK v2 should not route every sequence to cluster. It needs thresholds like:

```text
small seq -> register path
medium seq -> streaming path
long seq -> cluster path
```

The exact threshold must come from measurement.

### 10.5 Too Much Synchronization

Cluster barriers are powerful but expensive and easy to misuse. Only add them
when the algorithm truly requires cluster-wide alignment.

## 11. Porting Template for a New Kernel

Use this template before implementing.

```text
Kernel name:

Baseline behavior:
  - one workgroup currently handles:
  - bottleneck:
  - baseline benchmark:

Why cluster may help:
  - shared data:
  - large per-item work:
  - reduction needed:

Proposed cluster geometry:
  - cluster_x:
  - cluster_y:
  - cluster_z:
  - workgroups per cluster:

Mapping:
  - cluster_id maps to:
  - local workgroup id maps to:
  - output ownership:

Shape constraints:
  - M:
  - N:
  - K:
  - batch:

Fallback:
  - unsupported arch:
  - unsupported shape:
  - small problem size:

Correctness tests:
  - smallest valid shape:
  - shape at cluster boundary:
  - shape just below threshold:
  - shape just above threshold:
  - tail / padding case:

Performance tests:
  - baseline command:
  - cluster command:
  - metric:
  - crossover:
```

## 12. Beginner Exercises

### Exercise 1: Read a 2-WG Cluster

File:

```text
/sgl-workspace/aiter_0716/op_tests/opus/device/test_tdm_gfx1250.cu
```

Questions:

- What does `__cluster_dims__(2, 1, 1)` mean?
- What values can `__builtin_amdgcn_cluster_workgroup_id_x()` return?
- How does local cluster ID change the C output offset?
- Which operand is multicast?

Goal:

```text
Be able to draw the 2 workgroups and explain what each one computes.
```

### Exercise 2: Compare Plain GEMM and Cluster GEMM

Files:

```text
/sgl-workspace/aiter_0716/csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_cluster_tdm_splitk_ws_gfx1250.cuh
/sgl-workspace/aiter_0716/csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_ws_gfx1250.cuh
```

Questions:

- How does plain GEMM compute `tile_row` and `tile_col`?
- How does cluster GEMM compute `tile_row` and `tile_col`?
- Why does plain GEMM use `mask_a = 0` and `mask_b = 0`?
- Why does cluster GEMM build non-zero multicast masks?

Goal:

```text
Understand the difference between independent workgroups and clustered workgroups.
```

### Exercise 3: Inspect Host Shape Guards

Files:

```text
/sgl-workspace/aiter_0716/csrc/py_itfs_cu/asm_f4gemm.cu
/sgl-workspace/aiter_0716/csrc/py_itfs_cu/asm_mxfp8fp4gemm.cu
```

Questions:

- Where does `cluster_x` come from?
- Where does `cluster_y` come from?
- Why does `N` need to be a multiple of `tile_n * cluster_x`?
- When does `M` need to be a multiple of `tile_m * cluster_y`?
- How is persistent launch grid derived?

Goal:

```text
Be able to explain why unsupported shapes are rejected before launch.
```

### Exercise 4: Classify a Kernel

Pick any kernel you may want to port.

Answer:

```text
Is it GEMM-like reuse?
Is it TopK-like reduction?
Is it attention-like K/V reuse?
Is it simple elementwise?
```

If the answer is simple elementwise, cluster launch is probably not useful.

### Exercise 5: Build a Crossover Table

For a candidate kernel, benchmark:

```text
baseline vs cluster
small shape vs medium shape vs large shape
batch 1 vs batch 8 vs batch 32 vs batch 128
```

Goal:

```text
Find when cluster becomes faster and when it becomes slower.
```

## 13. Quick Decision Tree

```text
Do neighboring workgroups reuse the same data?
  yes -> cluster may help through multicast / shared loads
  no  -> continue

Is one logical item too large for one workgroup?
  yes -> cluster may help through chunking + reduction
  no  -> continue

Is the normal grid too small to fill the GPU?
  yes -> cluster/persistent design may help
  no  -> continue

Are shapes regular and divisible by cluster dims?
  yes -> prototype and benchmark
  no  -> keep fallback or avoid cluster
```

## 14. Final Mental Model

Cluster launch is worth learning because gfx1250 / MI455 gives you a new way to
organize cooperation between workgroups.

But the feature only matters when the algorithm uses it:

```text
GEMM cluster:
  share A/B loads across neighboring tiles

TopK cluster:
  split one huge row across many workgroups and reduce results

Persistent cluster:
  launch a controlled pool of clusters and let them walk the work grid
```

When reviewing a future kernel, always ask:

```text
What becomes cheaper if several workgroups are guaranteed to be a cluster?
```

If you have a clear answer, cluster launch may be a good optimization path.
If not, keep the kernel simple and benchmark other bottlenecks first.
