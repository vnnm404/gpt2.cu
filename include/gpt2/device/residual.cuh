#pragma once

#include <cuda_runtime.h>

// out: [batch_size, seq_len, n_embd]
// input: [batch_size, seq_len, n_embd]
// residual: [batch_size, seq_len, n_embd]
__device__ void residual_forward_device(float *out, const float *input,
                                        const float *residual, int batch_size,
                                        int seq_len, int n_embd,
                                        int blockIdx_x) {
  int idx = blockIdx_x * blockDim.x + threadIdx.x;
  int total_threads = batch_size * seq_len * n_embd;

  if (idx >= total_threads)
    return;

  out[idx] = input[idx] + residual[idx];
}

__device__ void residual_backward_device(float *g_inp1, float *g_inp2,
                                         const float *g_out, int N,
                                         int blockIdx_x) {
  int idx = blockIdx_x * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  g_inp1[idx] += g_out[idx];
  g_inp2[idx] += g_out[idx];
}