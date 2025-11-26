/* GELU layer implementation */

#include "gpt2/layers/gelu.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_forward(float *out, const float *input, int batch_size, int seq_len, int n_embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * seq_len * n_embd;

    if (idx >= total_threads) return;

    float x = input[idx];
    float cube = 0.044715f * x * x * x;
    out[idx] = 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + cube)));
}



__device__ void gelu_forward_mk(float *out, const float *input, int batch_size, int seq_len, int n_embd, int bidx) {
    int idx = bidx * blockDim.x + threadIdx.x;
    int total_threads = batch_size * seq_len * n_embd;

    if (idx >= total_threads) return;

    float x = input[idx];
    float cube = 0.044715f * x * x * x;
    out[idx] = 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + cube)));
}
