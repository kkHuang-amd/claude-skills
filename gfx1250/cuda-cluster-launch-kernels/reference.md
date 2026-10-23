# Cluster Launch Reference (TopK v2)

## Problem TopK v2 Solves

DeepSeek-V4 sparse attention indexer: with 4:1 compression, a 1M-token context yields
~256K logits per decode step. Single-CTA top-k at batch=1 can exceed **100 µs**;
cluster + radix path targets **~15 µs** on NVIDIA B200 (see
[DeepSeek-V4 Lightning TopK blog](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/#lightning-topk)).

## CUDA APIs Used

### Kernel annotation

```cpp
#define CLUSTER_TOPK_KERNEL \
    __global__ __launch_bounds__(kBlockSize, kOccupancy) \
    __cluster_dims__(1, kClusterSize, 1)
```

### Cooperative groups (device)

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

const auto cluster = cg::this_cluster();
cluster.sync();
auto* peer_hist = cluster.map_shared_rank(&smem->histogram[bin], peer_rank);
```

### Launch (host)

```cpp
// cudaLaunchKernelEx path (SGLang wrapper)
LaunchKernel({grid_x, kClusterSize}, kBlockSize, device)
    .config({
        .use_pdl = true,
        .cluster_dim = dim3{1, kClusterSize, 1},  // must match __cluster_dims__
    })
    .launch(kernel, params);
```

Underlying attribute:

```cpp
attr.id = cudaLaunchAttributeClusterDimension;
attr.val.clusterDim = {x, y, z};
```

### PDL (Hopper+, pairs producer/consumer on same stream)

```cpp
// Device — gated on SGL_ARCH_HOPPER_OR_GREATER in utils.cuh
asm volatile("griddepcontrol.wait;" ::: "memory");           // PDLWaitPrimary
asm volatile("griddepcontrol.launch_dependents;" :::);       // PDLTriggerSecondary
```

## TopK v2 Constants (B200-tuned)

```cpp
constexpr uint32_t kClusterSize = 8;
constexpr uint32_t kClusterFloor = 65536;        // default seq floor for cluster path
constexpr uint32_t kClusterFloorSmall = 32768;   // when batch <= 15
constexpr uint32_t kSmallBatchLowFloor = 15;
constexpr uint32_t kClusterMaxBatch = 512;
constexpr uint32_t kNumPersistentClusters = 15 * kOccupancy;  // occ=2 → 30
```

### Plan candidate table (`topk_plan`)

Threshold `T` lowered while `#{seq > T} <= cap`:

| T (threshold) | cap (max items to cluster pool) |
|---------------|----------------------------------|
| 65536 | 30 |
| 98304 | 48 |
| 131072 | 60 |
| 196608 | 80 |
| 262144 | 112 |
| 393216 | 128 |

`cluster_threshold = max(chosen_T, kClusterFloor)`.

## TopKCluster Algorithm Sketch

For one batch row, `kClusterSize` CTAs split `seq_len` into aligned chunks.

```
Phase 1   Each rank: vectorized load → local coarse histogram (fp16 bins)
Phase 1.5 cluster.sync → DSMEM all-reduce histogram across ranks
Phase 2   find_threshold(topk, global_seq_len) on merged histogram
Phase 3   Each rank: classify elements vs fp32 boundaries → atomics to primary
Phase 3.5 Non-primary: stage tmp_out; prefix offsets via atomics on primary SMEM
Phase 4   Primary: exact radix tie-break on threshold-bin candidates
```

Output: raw indices in `problem.out`; separate `problem_transform` pass applies page table.

## Host Dispatch Pseudocode

```cpp
if (use_cluster) {
  if (batch_size <= kNumPersistentClusters) {
    // Fused: one cluster per row
    launch_grid(batch_size, kClusterSize, topk_small_batch_kernel);
  } else {
    // Pool + epilogue
    launch_grid(min(batch, kNumPersistentClusters), kClusterSize,
                topk_persistent_cluster_kernel);
    launch_grid(batch_size, 1, topk_main_kernel<level=3>);
  }
} else if (max_seq <= 8192) {
  launch_grid(batch, 1, topk_main_kernel<level=0>);
} else if (max_seq <= 16384) {
  launch_grid(batch, 1, topk_main_kernel<level=1>);
} else {
  launch_grid(batch, 1, topk_main_kernel<level=2>);
}
```

## Metadata Layout

Tensor shape `(batch + 1, 2)` int32:

```
Row 0: [cluster_threshold, num_cluster_items]
Row 1..N: [batch_id, seq_len]  // only rows with seq_len > cluster_threshold
```

Python: `plan_topk_v2(seq_lens)` in `jit_kernel/dsv4/topk.py`.

## ROCm / gfx1250 Guard Template

```cpp
#if defined(USE_ROCM) && defined(SGL_ROCM_ARCH_GFX1250)
#include <hip/hip_cooperative_groups.h>
// Use HIP cluster launch attributes only on the proven arch.
// Keep fused small-batch cluster disabled until DSMEM faults are resolved.
const bool use_cluster =
    (max_seq_len > params.cluster_floor) &&
    (batch_size > kNumPersistentClusters) &&
    (batch_size <= kClusterMaxBatch);
#else
#include <cooperative_groups.h>
const bool use_cluster =
    (max_seq_len > params.cluster_floor) && (batch_size <= kClusterMaxBatch);
#endif
```

Current gfx1250 bring-up notes:

1. Define `SGL_ROCM_ARCH_GFX1250` from `torch.cuda.get_device_properties(...).gcnArchName`.
2. Implement `enable_cluster` in `utils.cuh` with `hipLaunchAttributeClusterDimension` only under that macro.
3. Include `hip/hip_cooperative_groups.h` under `USE_ROCM`.
4. Keep CUDA dispatch in the `#else` branch to avoid changing CUDA behavior.
5. Persistent cluster TopK v2 works on local gfx1250; fused small-batch cluster faults and should route to streaming.

## Validation Commands

```bash
# Focused unit coverage used during gfx1250 enablement
python3 -m pytest -q test/registered/jit/deepseek_v4/test_topk_v2.py::test_topk_v2_raw_indices -s

# Cluster boundary probes
python3 -m pytest -q test/registered/jit/deepseek_v4/test_topk_v2.py -k 'test_topk_v2_output_indices and 31-131072' -s
python3 -m pytest -q test/registered/jit/deepseek_v4/test_topk_v2.py -k 'test_topk_v2_output_indices and 40-262144' -s

# Micro-benchmark
python test/registered/jit/benchmark/bench_topk.py

# E2E accuracy gate
python benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --port <port>
# Target: accuracy > 0.94 for two runs (project TODO); CI uses > 0.92
```

## Design Principles (generalize beyond TopK)

1. **Plan once, execute many** — expensive host/device scheduling off hot path.
2. **Ladder of single-CTA paths** — cluster is last resort for long + small batch.
3. **Persistent pool** — amortize launch when many long rows exceed one wave of clusters.
4. **Separate selection vs transform** — cluster writes raw indices; epilogue does gathers.
5. **Tune crossover tables** — cluster advantage is shape-dependent; encode as data, not `#ifdef`.
6. **Keep fallback always correct** — cluster is optimization; streaming path is the safety net.
