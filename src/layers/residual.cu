/* Residual layer implementation */

#include "gpt2/layers/residual.h"
#include <cuda_runtime.h>

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