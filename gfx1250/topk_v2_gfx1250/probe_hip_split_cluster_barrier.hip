#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <cstdio>

namespace cg = cooperative_groups;

__global__ __cluster_dims__(1, 2, 1) void probe_kernel(int* result) {
  const auto cluster = cg::this_cluster();
  const unsigned int rank = cluster.block_rank();

  if (threadIdx.x == 0) {
    result[rank] = 10 + rank;
  }

  static_cast<void>(cluster.barrier_arrive());
  using Token = cg::cluster_group::arrival_token;
  cluster.barrier_wait(Token{});

  if (threadIdx.x == 0) {
    result[2 + rank] = 20 + rank;
  }
}

int main() {
  int* device_result = nullptr;
  int host_result[4] = {};
  hipError_t status = hipMalloc(&device_result, sizeof(host_result));
  if (status == hipSuccess) {
    status = hipMemset(device_result, 0, sizeof(host_result));
  }

  hipLaunchAttribute attr{};
  attr.id = hipLaunchAttributeClusterDimension;
  attr.val.clusterDim = {1, 2, 1};

  hipLaunchConfig_t config{};
  config.gridDim = {1, 2, 1};
  config.blockDim = {1024, 1, 1};
  config.attrs = &attr;
  config.numAttrs = 1;

  if (status == hipSuccess) {
    status = hipLaunchKernelEx(&config, probe_kernel, device_result);
  }
  if (status == hipSuccess) {
    status = hipDeviceSynchronize();
  }
  if (status == hipSuccess) {
    status = hipMemcpy(host_result, device_result, sizeof(host_result), hipMemcpyDeviceToHost);
  }
  const hipError_t free_status = hipFree(device_result);
  if (status == hipSuccess) {
    status = free_status;
  }

  if (status != hipSuccess) {
    std::fprintf(stderr, "probe failed: %s\n", hipGetErrorString(status));
    return 1;
  }

  std::printf(
      "result=%d,%d,%d,%d\n", host_result[0], host_result[1], host_result[2], host_result[3]);
  return host_result[0] == 10 && host_result[1] == 11 && host_result[2] == 20 &&
          host_result[3] == 21
      ? 0
      : 2;
}
