#ifndef GPT2_LAYERS_EMBEDDING_H
#define GPT2_LAYERS_EMBEDDING_H

/* Embedding layer structures and functions will be defined here */

#include <cuda_runtime.h>

__global__ void embedding_forward(float *out, const int *input_tokens, const float *wte, const float *wpe, int seq_len, int n_embd, int vocab_size, int n_positions);
__global__ void embedding_backward(float *g_wte, float *g_wpe, const float *g_out, const int *inp, int B, int T, int C);

__device__ void embedding_forward_device(float *out, const int *input_tokens, const float *wte, const float *wpe, 
                                         int seq_len, int n_embd, int vocab_size, int n_positions,
                                         int blockIdx_x);
__device__ void embedding_backward_device(float *g_wte, float *g_wpe, const float *g_out, const int *inp, 
                                          int B, int T, int C,
                                          int blockIdx_x);

#endif /* GPT2_LAYERS_EMBEDDING_H */
