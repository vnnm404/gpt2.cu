#pragma once

#include <cuda_runtime.h>

__global__ void gelu_forward(float *out, const float *input, int batch_size, int seq_len, int n_embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * seq_len * n_embd;

    if (idx >= total_threads) return;

    float x = input[idx];
    float cube = 0.044715f * x * x * x;
    out[idx] = 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + cube)));
}

__global__ void gelu_backward(float *g_inp, const float *inp, const float *g_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= N) return;
    
    float s = sqrtf(2.0f / M_PI);
    float x = inp[idx];
    float cube = 0.044715f * x * x * x;
    float tanh_arg = s * (x + cube);
    float tanh_out = tanhf(tanh_arg);
    float coshf_out = coshf(tanh_arg);
    float sech_out = 1.0f / (coshf_out * coshf_out);
    float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * s * (1.0f + 3.0f * 0.044715f * x * x);
    g_inp[idx] += local_grad * g_out[idx];
}