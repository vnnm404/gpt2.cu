#ifndef GPT2_LAYERS_GELU_H
#define GPT2_LAYERS_GELU_H

#include <cuda_runtime.h>

/* GELU layer structures and functions will be defined here */

__global__ void gelu_forward(float *out, const float *input, int batch_size, int seq_len, int n_embd);
__global__ void gelu_backward(float *g_inp, const float *inp, const float *g_out, int N);

__device__ void gelu_forward_device(float *out, const float *input, int batch_size, int seq_len, int n_embd,
                                    int blockIdx_x);
__device__ void gelu_backward_device(float *g_inp, const float *inp, const float *g_out, int N,
                                     int blockIdx_x);

#endif /* GPT2_LAYERS_GELU_H */
