#pragma once
#include <cuda_runtime.h>
#include "gpt2/executor.h"

namespace gpt2 {
constexpr int threads = 256;
constexpr int tile_m = 64;
constexpr int tile_n = 128;
constexpr int shared_bytes = 32 * 1024;

__device__ __forceinline__ float warp_sum(float x) {
    for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d);
    return x;
}
__device__ __forceinline__ float warp_max(float x) {
    for (int d = 16; d; d >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, d));
    return x;
}
template<bool Max = false>
__device__ __forceinline__ float reduce(float x, float *s) {
    int t = threadIdx.x;
    x = Max ? warp_max(x) : warp_sum(x);
    if ((t & 31) == 0) s[t >> 5] = x;
    __syncthreads();
    x = t < 8 ? s[t] : (Max ? -INFINITY : 0.f);
    x = Max ? warp_max(x) : warp_sum(x);
    if (t == 0) s[8] = x;
    __syncthreads();
    x = s[8];
    __syncthreads();
    return x;
}
} // namespace gpt2
