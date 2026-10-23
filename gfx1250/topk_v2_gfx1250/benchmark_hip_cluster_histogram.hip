#include <hip/hip_cooperative_groups.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;

constexpr unsigned int kBlockSize = 1024;
constexpr unsigned int kHistSize = 1024;

__device__ __forceinline__ unsigned int coarse_bin(float x) {
  const uint32_t b = static_cast<uint32_t>(__half_as_ushort(__float2half_rn(x))) << 16;
  const uint32_t s = static_cast<uint32_t>(static_cast<int32_t>(b) >> 31);
  return (b ^ (s | 0x80000000u)) >> 22;
}

__global__ __launch_bounds__(kBlockSize, 2) void streaming_histogram(
    const float* scores, uint32_t* output, unsigned int seq_len) {
  __shared__ uint32_t hist[kHistSize];
  const unsigned int tx = threadIdx.x;
  const unsigned int row = blockIdx.x;
  hist[tx] = 0;
  __syncthreads();

  const float* input = scores + static_cast<size_t>(row) * seq_len;
  for (unsigned int i = tx; i < seq_len; i += kBlockSize) {
    atomicAdd(hist + coarse_bin(input[i]), 1u);
  }
  __syncthreads();
  output[static_cast<size_t>(row) * kHistSize + tx] = hist[tx];
}

template <unsigned int ClusterSize>
__global__ __launch_bounds__(kBlockSize, 1) __cluster_dims__(1, ClusterSize, 1)
    void cluster_histogram(
        const float* scores, uint32_t* scratch, uint32_t* output, unsigned int seq_len) {
  __shared__ uint32_t hist[kHistSize];
  const auto cluster = cg::this_cluster();
  const unsigned int tx = threadIdx.x;
  const unsigned int row = blockIdx.x;
  const unsigned int rank = cluster.block_rank();
  hist[tx] = 0;
  __syncthreads();

  const float* input = scores + static_cast<size_t>(row) * seq_len;
  for (unsigned int i = rank * kBlockSize + tx; i < seq_len; i += ClusterSize * kBlockSize) {
    atomicAdd(hist + coarse_bin(input[i]), 1u);
  }
  __syncthreads();

  uint32_t* row_scratch =
      scratch + (static_cast<size_t>(row) * ClusterSize + rank) * kHistSize;
  row_scratch[tx] = hist[tx];

  static_cast<void>(cluster.barrier_arrive());
  using Token = cg::cluster_group::arrival_token;
  cluster.barrier_wait(Token{});

  uint32_t sum = 0;
#pragma unroll
  for (unsigned int peer = 0; peer < ClusterSize; ++peer) {
    sum += __builtin_amdgcn_cluster_load_b32(
        reinterpret_cast<int*>(scratch) +
            (static_cast<size_t>(row) * ClusterSize + peer) * kHistSize + tx,
        0,
        (1u << ClusterSize) - 1u);
  }
  output[(static_cast<size_t>(row) * ClusterSize + rank) * kHistSize + tx] = sum;
}

template <unsigned int ClusterSize>
hipError_t launch_cluster(
    const float* scores,
    uint32_t* scratch,
    uint32_t* output,
    unsigned int batch,
    unsigned int seq_len) {
  hipLaunchAttribute attr{};
  attr.id = hipLaunchAttributeClusterDimension;
  attr.val.clusterDim = {1, ClusterSize, 1};
  hipLaunchConfig_t config{};
  config.gridDim = {batch, ClusterSize, 1};
  config.blockDim = {kBlockSize, 1, 1};
  config.attrs = &attr;
  config.numAttrs = 1;
  return hipLaunchKernelEx(
      &config, cluster_histogram<ClusterSize>, scores, scratch, output, seq_len);
}

template <typename Launch>
float benchmark(Launch&& launch) {
  constexpr int kWarmup = 20;
  constexpr int kIterations = 100;
  for (int i = 0; i < kWarmup; ++i) {
    if (const hipError_t status = launch(); status != hipSuccess) {
      std::fprintf(stderr, "warmup launch failed: %s\n", hipGetErrorString(status));
      std::exit(2);
    }
  }
  if (const hipError_t status = hipDeviceSynchronize(); status != hipSuccess) {
    std::fprintf(stderr, "warmup sync failed: %s\n", hipGetErrorString(status));
    std::exit(2);
  }

  hipEvent_t begin = nullptr;
  hipEvent_t end = nullptr;
  hipEventCreate(&begin);
  hipEventCreate(&end);
  hipEventRecord(begin);
  for (int i = 0; i < kIterations; ++i) {
    if (const hipError_t status = launch(); status != hipSuccess) {
      std::fprintf(stderr, "timed launch failed: %s\n", hipGetErrorString(status));
      std::exit(2);
    }
  }
  hipEventRecord(end);
  hipEventSynchronize(end);
  float elapsed_ms = 0;
  hipEventElapsedTime(&elapsed_ms, begin, end);
  hipEventDestroy(end);
  hipEventDestroy(begin);
  return elapsed_ms * 1000.0f / kIterations;
}

int main(int argc, char** argv) {
  const unsigned int batch = argc > 1 ? static_cast<unsigned int>(std::strtoul(argv[1], nullptr, 10)) : 1;
  const unsigned int seq_len =
      argc > 2 ? static_cast<unsigned int>(std::strtoul(argv[2], nullptr, 10)) : 262144;
  const size_t score_items = static_cast<size_t>(batch) * seq_len;

  std::vector<float> host_scores(score_items);
  uint32_t state = 1;
  for (float& score : host_scores) {
    state = state * 1664525u + 1013904223u;
    score = static_cast<float>(static_cast<int32_t>(state)) * (4.0f / 2147483648.0f);
  }

  float* scores = nullptr;
  uint32_t* scratch = nullptr;
  uint32_t* output = nullptr;
  constexpr unsigned int kMaxCluster = 8;
  hipError_t status = hipMalloc(&scores, score_items * sizeof(float));
  if (status == hipSuccess) {
    status = hipMalloc(&scratch, static_cast<size_t>(batch) * kMaxCluster * kHistSize * sizeof(uint32_t));
  }
  if (status == hipSuccess) {
    status = hipMalloc(&output, static_cast<size_t>(batch) * kMaxCluster * kHistSize * sizeof(uint32_t));
  }
  if (status == hipSuccess) {
    status = hipMemcpy(scores, host_scores.data(), score_items * sizeof(float), hipMemcpyHostToDevice);
  }
  if (status != hipSuccess) {
    std::fprintf(stderr, "setup failed: %s\n", hipGetErrorString(status));
    return 1;
  }

  const float stream_us = benchmark([&] {
    streaming_histogram<<<dim3(batch), dim3(kBlockSize)>>>(scores, output, seq_len);
    return hipGetLastError();
  });
  const float cluster2_us =
      benchmark([&] { return launch_cluster<2>(scores, scratch, output, batch, seq_len); });
  const float cluster4_us =
      benchmark([&] { return launch_cluster<4>(scores, scratch, output, batch, seq_len); });
  const float cluster8_us =
      benchmark([&] { return launch_cluster<8>(scores, scratch, output, batch, seq_len); });

  std::printf(
      "batch=%u seq=%u stream=%.2f us cluster2=%.2f us (%.3fx) "
      "cluster4=%.2f us (%.3fx) cluster8=%.2f us (%.3fx)\n",
      batch,
      seq_len,
      stream_us,
      cluster2_us,
      stream_us / cluster2_us,
      cluster4_us,
      stream_us / cluster4_us,
      cluster8_us,
      stream_us / cluster8_us);

  hipFree(output);
  hipFree(scratch);
  hipFree(scores);
  return 0;
}
