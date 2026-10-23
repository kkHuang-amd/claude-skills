#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

namespace cg = cooperative_groups;

constexpr unsigned int kClusterSize = 2;
constexpr unsigned int kBlockSize = 1024;

__global__ __cluster_dims__(1, kClusterSize, 1) void probe_kernel(int* scratch, int* output) {
  const auto cluster = cg::this_cluster();
  const unsigned int rank = cluster.block_rank();
  const unsigned int tx = threadIdx.x;
  scratch[rank * kBlockSize + tx] = static_cast<int>(rank * 10000 + tx);

  static_cast<void>(cluster.barrier_arrive());
  using Token = cg::cluster_group::arrival_token;
  cluster.barrier_wait(Token{});

  int sum = 0;
#pragma unroll
  for (unsigned int peer = 0; peer < kClusterSize; ++peer) {
    sum += __builtin_amdgcn_cluster_load_b32(
        scratch + peer * kBlockSize + tx, 0, (1u << kClusterSize) - 1u);
  }
  output[rank * kBlockSize + tx] = sum;
}

int main() {
  constexpr size_t kItems = kClusterSize * kBlockSize;
  int* scratch = nullptr;
  int* output = nullptr;
  hipError_t status = hipMalloc(&scratch, kItems * sizeof(int));
  if (status == hipSuccess) status = hipMalloc(&output, kItems * sizeof(int));

  hipLaunchAttribute attr{};
  attr.id = hipLaunchAttributeClusterDimension;
  attr.val.clusterDim = {1, kClusterSize, 1};
  hipLaunchConfig_t config{};
  config.gridDim = {1, kClusterSize, 1};
  config.blockDim = {kBlockSize, 1, 1};
  config.attrs = &attr;
  config.numAttrs = 1;

  if (status == hipSuccess) status = hipLaunchKernelEx(&config, probe_kernel, scratch, output);
  if (status == hipSuccess) status = hipDeviceSynchronize();

  std::vector<int> host(kItems);
  if (status == hipSuccess) {
    status = hipMemcpy(host.data(), output, kItems * sizeof(int), hipMemcpyDeviceToHost);
  }

  bool valid = status == hipSuccess;
  for (unsigned int rank = 0; valid && rank < kClusterSize; ++rank) {
    for (unsigned int tx = 0; tx < kBlockSize; ++tx) {
      const int expected = 10000 + static_cast<int>(2 * tx);
      if (host[rank * kBlockSize + tx] != expected) {
        std::fprintf(
            stderr,
            "mismatch rank=%u tx=%u got=%d expected=%d\n",
            rank,
            tx,
            host[rank * kBlockSize + tx],
            expected);
        valid = false;
        break;
      }
    }
  }

  const hipError_t output_free = hipFree(output);
  const hipError_t scratch_free = hipFree(scratch);
  if (status == hipSuccess) status = output_free;
  if (status == hipSuccess) status = scratch_free;
  if (status != hipSuccess) {
    std::fprintf(stderr, "probe failed: %s\n", hipGetErrorString(status));
    return 1;
  }
  std::printf("cluster_load_b32 %s (%zu values)\n", valid ? "passed" : "failed", kItems);
  return valid ? 0 : 2;
}
