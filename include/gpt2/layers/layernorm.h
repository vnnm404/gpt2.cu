#ifndef GPT2_LAYERS_LAYERNORM_H
#define GPT2_LAYERS_LAYERNORM_H

#include <cuda_runtime.h>

/* Layer normalization structures and functions will be defined here */

__global__ void layernorm_forward(float *out, float *input, float *weight, float *bias, int seq_len, int n_embd);

__device__ void layernorm_forward_mk(float *out, float *input, float *weight, float *bias, int seq_len, int n_embd, int bidx);

#endif /* GPT2_LAYERS_LAYERNORM_H */
