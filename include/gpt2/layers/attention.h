#ifndef GPT2_LAYERS_ATTENTION_H
#define GPT2_LAYERS_ATTENTION_H

#include <cuda_runtime.h>

/* Attention layer - optimized CUDA implementation
 * Uses 2D grid, shared memory, parallel reduction, and atomic-free backward pass
 */

// Block configuration for attention kernels
#define ATTN_BLOCK_SIZE 256
#define ATTN_WARP_SIZE 32

// Forward pass: 2D grid (B*NH, T), 1D block handles one query position
// Each block computes attention for one (batch, head, query_time)
#define ATTN_FWD_GRID(B, T, NH) dim3((B) * (NH), (T))
#define ATTN_FWD_BLOCK() ATTN_BLOCK_SIZE
#define ATTN_FWD_SMEM(T, hs) (sizeof(float) * ((hs) + (T) * 3)) // query + scores/max/sum tiles

// Backward pass: 2D grid (B*NH, T), blockIdx_y = query position t
// Still uses atomics for g_key/g_value but with shared memory optimization
#define ATTN_BWD_GRID(B, T, NH) dim3((B) * (NH), (T))
#define ATTN_BWD_BLOCK() ATTN_BLOCK_SIZE
#define ATTN_BWD_SMEM(T, hs) (sizeof(float) * ((T) * 2 + 8)) // s_att[T] + s_g_att[T] + warp reduction

// Optimized kernels with 2D grid
__global__ void attention_forward(float *out, float *preatt, float *att,
                                  const float *inp, int B, int T, int NH, int C);
__global__ void attention_backward(float *g_inp, float *g_preatt, float *g_att,
                                   const float *g_out, const float *inp, const float *att,
                                   int B, int T, int C, int NH);

// Device functions for megakernel compatibility
__device__ void attention_forward_device(float *out, float *preatt, float *att,
                                         const float *inp,
                                         int B, int T, int NH, int C,
                                         int blockIdx_x, int blockIdx_y);

__device__ void attention_backward_device(float *g_inp, float *g_preatt, float *g_att,
                                          const float *g_out, const float *inp, const float *att,
                                          int B, int T, int C, int NH,
                                          int blockIdx_x, int blockIdx_y);

#endif /* GPT2_LAYERS_ATTENTION_H */
