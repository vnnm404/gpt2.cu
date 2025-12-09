#ifndef GPT2_LAYERS_ATTENTION_H
#define GPT2_LAYERS_ATTENTION_H

#include <cuda_runtime.h>

/* Attention layer structures and functions will be defined here */

__global__ void attention_forward(float *out, float *preatt, float *att, const float *input, int batch_size, int seq_len, int n_head, int n_embd);
__global__ void attention_backward(float *g_inp, float *g_preatt, float *g_att,
                                   const float *g_out, const float *inp, const float *att,
                                   int B, int T, int C, int NH);

__device__ void attention_forward_device(float* out, float* preatt, float* att,
                                         const float* inp,
                                         int B, int T, int NH, int C,
                                         int blockIdx_x);

__device__ void attention_backward_device(float* g_inp, float* g_preatt, float* g_att,
                                   const float* g_out, const float* inp, const float* att,
                                   int B, int T, int C, int NH,
                                   int blockIdx_x);

#endif /* GPT2_LAYERS_ATTENTION_H */
