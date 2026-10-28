#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
typedef unsigned u4 __attribute__((ext_vector_type(4)));
typedef int i8 __attribute__((ext_vector_type(8)));
typedef int i4 __attribute__((ext_vector_type(4)));
// 2D TDM load: tile of rows x cols 4-byte elements from a row-major tensor with row pitch `pitch` elements.
// VAR 0: pitch in tensor_dim0_stride ; VAR 1: pitch in tensor_dim1_stride
template <int VAR>
__global__ void k(const int* g, int* out, int rows, int cols, int pitch, int trows) {
  __shared__ int s[8 * 256];
  for (int i = threadIdx.x; i < 8 * 256; i += blockDim.x) s[i] = -1;
  __syncthreads();
  if (threadIdx.x < 32) {
    const unsigned long long ga = (unsigned long long)g;
    const unsigned la = (unsigned)(unsigned long long)(__attribute__((address_space(3))) int*)s;
    u4 g0; g0.x = 1; g0.y = la; g0.z = (unsigned)ga; g0.w = ((unsigned)(ga >> 32) & 0x1ffffff) | (2u << 30);
    const unsigned dim0 = pitch, dim1 = trows, tile0 = cols, tile1 = rows;
    const unsigned long long s0 = VAR == 0 ? pitch : 0, s1 = VAR == 1 ? pitch : 0;
    i8 g1;
    g1[0] = (int)(2u << 16);                                     // wg_mask 0, data_size 4 B
    g1[1] = (int)((dim0 & 0xffff) << 16);
    g1[2] = (int)((dim0 >> 16) | ((dim1 & 0xffff) << 16));
    g1[3] = (int)((dim1 >> 16) | (tile0 << 16));
    g1[4] = (int)(tile1);
    g1[5] = (int)(unsigned)s0;
    g1[6] = (int)((unsigned)(s0 >> 32) & 0xffff) | (int)(((unsigned)s1 & 0xffff) << 16);
    g1[7] = (int)(unsigned)(s1 >> 16);
    i4 z4 = {}; i8 z8 = {};
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, z4, z4, z8, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 8 * 256; i += blockDim.x) out[i] = s[i];
}
int main() {
  const int PITCH = 1024, TR = 8, ROWS = 8, COLS = 256;
  std::vector<int> h(PITCH * TR);
  for (int r = 0; r < TR; ++r) for (int c = 0; c < PITCH; ++c) h[r * PITCH + c] = r * 100000 + c;
  int *g, *o; hipMalloc(&g, h.size() * 4); hipMalloc(&o, 8 * 256 * 4); hipMemcpy(g, h.data(), h.size() * 4, hipMemcpyHostToDevice);
  for (int var = 0; var < 2; ++var) {
    if (var == 0) k<0><<<1, 128>>>(g, o, ROWS, COLS, PITCH, TR); else k<1><<<1, 128>>>(g, o, ROWS, COLS, PITCH, TR);
    hipError_t e = hipDeviceSynchronize();
    std::vector<int> r(8 * 256); hipMemcpy(r.data(), o, r.size() * 4, hipMemcpyDeviceToHost);
    int bad = 0; for (int i = 0; i < 8 * 256; ++i) bad += r[i] != (i / 256) * 100000 + i % 256;
    printf("VAR %d (%s): err=%s bad=%d  s[0]=%d s[255]=%d s[256]=%d s[2047]=%d\n", var, var ? "pitch in dim1_stride" : "pitch in dim0_stride",
           hipGetErrorString(e), bad, r[0], r[255], r[256], r[2047]);
  }
}
