---
name: cuda-cluster-launch-kernels
description: >-
  Design and port multi-CTA cluster-launch GPU kernels using the DeepSeek-V4 TopK v2
  pattern (CUDA cluster dims, DSMEM via cooperative_groups, host plan kernel, PDL).
  Use when adding or enabling cluster-cooperative JIT kernels, or when next-gen AMD
  GPUs may expose an equivalent to CUDA cluster launch.
---

# CUDA Cluster Launch Kernels (TopK v2 Pattern)

Use this skill when a kernel is **latency-bound at small batch** because one CTA cannot
hold enough shared memory / parallelism for long per-row work (e.g. 256K candidates in
sparse attention top-k). Cluster launch lets **multiple CTAs cooperate on one logical
item** via distributed shared memory (DSMEM).

**Canonical reference in SGLang:** `python/sglang/jit_kernel/csrc/deepseek_v4/topk_v2.cuh`
and `include/sgl_kernel/deepseek_v4/topk_impl.cuh` (`TopKCluster`).

## When Cluster Launch Wins

Use cluster when ALL are true:

1. **Per-item work is large** (seq_len or candidate count >> single-CTA SMEM budget).
2. **Batch is small** (latency-bound; one cluster per batch row is affordable).
3. **Work decomposes into independent chunks** that need a **global reduction**
   (histogram, prefix sum, top-k threshold) across chunks.
4. **Measured crossover**: cluster beats single-CTA streaming at your target shapes
   (TopK v2: ~36–40K seq for batch ≤ 15 on B200; floor 64K for larger batch).

Do **not** use cluster for large batch alone — persistent-pool waves add overhead;
TopK v2 caps cluster batch at 512 and often routes batch > 128 to streaming.

## Architecture (TopK v2 Mental Model)

```text
Host plan (once per forward)
  topk_plan → metadata[0] = {cluster_threshold, num_cluster_items}
            → metadata[1..N] = {batch_id, seq_len} for long rows

Host transform (per layer / per call)
  if max_seq > cluster_floor && batch <= cluster_max:
    small batch:  grid = (batch, cluster_size)  → topk_small_batch_kernel
    large batch:  persistent cluster pool       → topk_persistent_cluster_kernel
                  + main epilogue per row       → topk_main_kernel<level=3>
  else:
    ladder: Register2 → Register4 → Streaming (single CTA per row)
```

### Dispatch ladder (single-CTA paths)

| Level | Condition (max seq in batch) | Implementation |
|-------|------------------------------|----------------|
| trivial | seq ≤ topk | identity indices |
| 0 | ≤ 8192 | `TopKRegister<2>` — scores in registers |
| 1 | ≤ 16384 | `TopKRegister<4>` |
| 2 | ≤ cluster_floor | `TopKStreaming` — two passes over HBM |
| 3 | > cluster_floor | cluster does selection; main kernel epilogue only |

## Device-Side Checklist

### 1. Declare cluster kernel shape

```cpp
#define TOPK_KERNEL __global__ __launch_bounds__(kBlockSize, kOccupancy)
#define CLUSTER_TOPK_KERNEL TOPK_KERNEL __cluster_dims__(1, kClusterSize, 1)
```

- `__cluster_dims__(x, y, z)`: cluster is `x*y*z` CTAs (TopK uses `(1, 8, 1)`).
- Fixed `kClusterSize` at compile time (DSMEM addressing assumes constant cluster width).

### 2. Identify rank within cluster

```cpp
const auto cluster = cooperative_groups::this_cluster();
const auto this_rank = blockIdx.y;   // grid.y = cluster_size for fused launch
const bool is_primary = (this_rank == 0);
```

- **Input partition**: each rank owns `chunk_size` elements of the row.
- **Primary rank (0)**: owns global counters (`count_gt`, `count_eq`) and final output buffer.

### 3. DSMEM patterns (from `TopKCluster::forward`)

| Phase | Pattern |
|-------|---------|
| Local histogram | `atomicAdd` into rank-local SMEM |
| Cross-rank reduce | `cluster.sync()` → `map_shared_rank(&smem->histogram[which], peer_rank)` → warp reduce → write back |
| Global threshold | `find_threshold` on rank 0 SMEM after histogram is global |
| Collect | Each rank scans its chunk; atomics into primary's counters |
| Staging | Non-primary ranks write to `tmp_out`; prefix-sum slot offsets via atomics on primary |
| Tie-break | Primary runs exact radix on threshold bin |

**Rule:** always `cluster.sync()` before/after DSMEM reads across ranks.

### 4. Fused small-batch vs persistent pool

**Fused** (`batch ≤ kNumPersistentClusters`, TopK: ≤ 30):

- Grid `(batch_size, kClusterSize)` — one cluster per batch row, all in one launch.
- Elect `worker_rank = blockIdx.x % kClusterSize` to run epilogue (page-table transform) to avoid imbalance.

**Persistent pool** (`batch > 30` but still cluster-eligible):

- Cluster kernel grid `(num_persistent_clusters, kClusterSize)` loops over `metadata[1..N]`.
- Main kernel grid `(batch_size)` handles rows ≤ threshold locally; long rows only epilogue.

### 5. PDL (Programmatic Dependent Launch)

Pair cluster + epilogue kernels on the same stream:

```cpp
device::PDLWaitPrimary<kUsePDL>();      // consumer waits for producer
device::PDLTriggerSecondary<kUsePDL>(); // producer signals dependent launch
```

- Cluster `forward()` uses `PDLWaitPrimary` once per item; **no** trailing barrier inside `TopKCluster`.
- Main / small-batch kernels trigger secondary after output is complete.
- On ROCm / pre-Hopper: PDL is a no-op (`utils.cuh`); keep calls for portable source.

## Host-Side Checklist

### 1. Launch via `LaunchKernel` + `cudaLaunchKernelEx`

```cpp
LaunchKernel({batch_size, kClusterSize}, kBlockSize, device)
    .config({.use_pdl = true, .cluster_dim = dim3{1, kClusterSize}})
    .launch(topk_small_batch_kernel<true>, params);
```

`enable_cluster` sets `cudaLaunchAttributeClusterDimension` (`utils.cuh`).

### 2. Plan kernel (host preprocessing)

Run **once per forward** when seq_lens distribution is known:

- Choose `cluster_threshold` from `(threshold, max_items)` candidate table tuned to
  **measured** cluster-vs-streaming crossover (not a single magic constant).
- Compact `metadata[1..N]` with `{batch_id, seq_len}` for rows above threshold.
- Store `num_cluster_items` for persistent pool iteration.

Reuse plan across layers (TopK: `DSAMetadata.topk_v2_plan`, refreshed in-place under CUDA graph).

### 3. Host dispatch guards

```cpp
const bool use_cluster =
    (max_seq_len > params.cluster_floor) && (batch_size <= kClusterMaxBatch);
```

Also set `cluster_floor` per launch (TopK lowers floor to 32K when `batch ≤ 15`).

## Platform Portability (CUDA and gfx1250)

ROCm 7.15 on gfx1250 exposes cluster-launch surface area, but it is **not fully
CUDA-equivalent** for the TopK v2 fused small-batch pattern.

| Piece | CUDA | gfx1250 ROCm 7.15 | Notes |
|-------|------|-------------------|-------|
| `__cluster_dims__` | yes | accepted for `(1, 8, 1)` | x-oriented `{8,1,1}` launch was rejected by HIP in local tests. |
| `cooperative_groups::this_cluster()` | yes | available via `hip/hip_cooperative_groups.h` | Include HIP header under `USE_ROCM`. |
| `LaunchKernel::enable_cluster` | `cudaLaunchAttributeClusterDimension` | `hipLaunchAttributeClusterDimension` | Gate behind `SGL_ROCM_ARCH_GFX1250`; keep old HIP launch path elsewhere. |
| `map_shared_rank` | DSMEM | header exposes `__builtin_amdgcn_map_shared_rank` | Persistent pool path works; fused small-batch path faults locally. |
| PDL | Hopper+ asm | no-op | Keep calls for portable source; do not rely on PDL ordering on ROCm. |

**gfx1250 TopK v2 rule of thumb:** enable v2, but route `batch <= kNumPersistentClusters`
through the safe streaming/register v2 path until fused small-batch cluster is fixed.
Use persistent cluster only for larger cluster-routed batches.

### Enabling a cluster path on AMD

1. Add a runtime/JIT arch macro (current pattern: `SGL_ROCM_ARCH_GFX1250` from `gcnArchName`).
2. Implement `enable_cluster()` for that arch only; do not change generic CUDA behavior.
3. Include `hip/hip_cooperative_groups.h` under `USE_ROCM`.
4. Keep CUDA dispatch in the `#else` branch byte-for-byte where possible.
5. Re-run the boundary grid from `test/registered/jit/deepseek_v4/test_topk_v2.py`, especially:
   `batch 30/31`, `seq 65536/65537/131072/262144`.
6. Micro-benchmark v1 vs v2 for target shapes before enabling in model args.
7. E2E compare server behavior with `SGLANG_OPT_USE_TOPK_V2=false` and `true`.

## Testing Matrix (copy when porting)

From `test_topk_v2.py` — every dispatch template + boundary:

- **Templates**: trivial, Register2, Register4, Streaming, Cluster
- **Seq boundaries**: 8192/8193, 16384/16385, 65535/65536/65537, 131072, 262144
- **Batch boundaries**: 30/31 (fused vs pool), 128/129 (cluster vs streaming-only)
- **topk**: 512, 1024, 2048
- **Tolerance**: ≤ 5 index mismatches per row (fp16 coarse-bin tie swaps)

## Common Pitfalls

1. **Negative `seq_lens`** — kernel reads lengths as `uint32_t`; clamp padded rows to 0.
2. **Missing plan refresh** — plan must match current `seq_lens`; do not recompute per layer silently.
3. **Cluster without DSMEM reduce** — local histograms are wrong; threshold bin drifts.
4. **Epilogue before cluster finishes** — use PDL or explicit sync; main level-3 skips work for long rows.
5. **Large batch + cluster** — pool waves hurt; prefer streaming when batch > tuned cap.
6. **ROCm warp masks** — if porting non-cluster radix code, use `device::warp::kFullMask` (64-bit on HIP).
7. **gfx1250 fused small-batch cluster** — `(batch <= 30, long seq)` currently faults in TopK v2; do not enable it without a minimal DSMEM repro and boundary tests.

## Related Files

| File | Role |
|------|------|
| `jit_kernel/csrc/deepseek_v4/topk_v2.cuh` | Kernels, plan, host dispatcher |
| `jit_kernel/include/sgl_kernel/deepseek_v4/topk_impl.cuh` | `TopKCluster`, Register, Streaming |
| `jit_kernel/include/sgl_kernel/utils.cuh` | `LaunchKernel`, PDL, `enable_cluster` |
| `jit_kernel/dsv4/topk.py` | Python API: `plan_topk_v2`, `topk_transform_512_v2` |
| `srt/layers/attention/dsv4/indexer.py` | Production caller |
| `test/registered/jit/deepseek_v4/test_topk_v2.py` | Correctness grid |

See [reference.md](reference.md) for API snippets and TopK v2 tuning constants.
See [examples.md](examples.md) for gfx1250 commands and debug patterns.
