#ifndef GPT2_LAYERS_LAYERNORM_H
#define GPT2_LAYERS_LAYERNORM_H

#include <cuda_runtime.h>

/* Layer normalization - optimized CUDA implementation
 * Uses shared memory for parallel reduction across n_embd dimension
 * Each block handles one (batch, seq) position
 */

// Launch configuration: B*S blocks, 256 threads per block
// Shared memory: 8 floats for warp reduction
#define LN_BLOCK_SIZE 256
#define LN_SMEM_SIZE (sizeof(float) * 8)

__global__ void layernorm_forward(float *out, float *input, float *weight, float *bias, float *ln_mean, float *ln_rstd, int seq_len, int n_embd);
__global__ void layernorm_backward(float *g_input, float *g_weight, float *g_bias, const float *g_out, const float *input, const float *weight, const float *ln_mean, const float *ln_rstd, int batch_size, int seq_len, int n_embd);

__device__ void layernorm_forward_device(float *out, float *input, float *weight, float *bias, float *ln_mean, float *ln_rstd,
                                         int seq_len, int n_embd,
                                         int blockIdx_x);

__device__ void layernorm_backward_device(float *g_input, float *g_weight, float *g_bias, const float *g_out, const float *input, const float *weight, const float *ln_mean, const float *ln_rstd,
                                          int batch_size, int seq_len, int n_embd,
                                          int blockIdx_x);

#endif /* GPT2_LAYERS_LAYERNORM_H */
