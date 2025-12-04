/* Residual layer implementation */

#include "gpt2/layers/residual.h"
#include <cuda_runtime.h>

// Device function for residual forward pass - Megakernel compatible
// out: [batch_size, seq_len, n_embd]
// input: [batch_size, seq_len, n_embd]
// residual: [batch_size, seq_len, n_embd]
__device__ __noinline__ void residual_forward_device(float *out, const float *input, const float *residual, int batch_size, int seq_len, int n_embd,
                                        int blockIdx_x) {
    int idx = blockIdx_x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * seq_len * n_embd;

    if (idx >= total_threads) return;

    out[idx] = input[idx] + residual[idx];
}

// out: [batch_size, seq_len, n_embd]
// input: [batch_size, seq_len, n_embd]
// residual: [batch_size, seq_len, n_embd]
// call with <<<CEIL_DIV(batch_size * seq_len * n_embd, 256), 256>>>
__global__ void residual_forward(float *out, const float *input, const float *residual, int batch_size, int seq_len, int n_embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * seq_len * n_embd;

    if (idx >= total_threads) return;

    out[idx] = input[idx] + residual[idx];
}

// Device function for residual backward pass - Megakernel compatible
__device__ __noinline__ void residual_backward_device(float *g_inp1, float *g_inp2, const float *g_out, int N,
                                         int blockIdx_x) {
    int idx = blockIdx_x * blockDim.x + threadIdx.x;
    
    if (idx >= N) return;
    
    g_inp1[idx] += g_out[idx];
    g_inp2[idx] += g_out[idx];
}

__global__ void residual_backward(float *g_inp1, float *g_inp2, const float *g_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= N) return;
    
    g_inp1[idx] += g_out[idx];
    g_inp2[idx] += g_out[idx];
}